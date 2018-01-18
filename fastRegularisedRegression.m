function [ res ] = fastRegularisedRegression(X, Y, varargin)
%fastRegularisedRegression perform feature selection for regression which
%is regularized by Tikhonov regularization (ridge regression) with
%automated selection of the optimal value of regularization parameter
%Alpha.
%
%Inputs:
%   X is numeric matrix with n rows and p columns. Each row represents one
%       observation, and each column represents one predictor (variable). 
%   Y is numeric vector of length n, where n is the number of rows of X.
%       Y(i) is the response to row i of X.
%
%   IMPORTANT NOTES: data matrix X and response vector Y are standardized
%       anyway but values of coefficients and intercept are calculated for
%       the original values of X and Y
%
%   Name, Value is one or more pairs of name and value. There are several
%       possible names with corresponding values: 
%       'Weights' is vector of observation weights. It must be a vector of
%           non-negative values, of the same length as columns of X. At
%           least two values must be positive. Default (1/N)*ones(N,1).
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. Default: {}
%       'CV', 'AlphaCV', 'FSCV'. If 'CV' is presented, then it indicates
%           the method used to compute the final quality statistics (MSE,
%           MAD or MAR). If 'AlphaCV' is presented, then it indicates the
%           method used to compute accuracy (MSE, MAD or MAR) for searching
%           of Alpha. If 'FSCV' is presented, then it indicates  the method
%           used to compute accuracy (MSE, MAD or MAR) for feature
%           selection. When parameter value is a positive integer K,
%           fastRegularisedRegression uses K-fold cross-validation. Set
%           parameter to a cross-validation partition, created using
%           CVPARTITION, to use other forms of cross-validation. You cannot
%           use a 'Leaveout' partition with fastRegularisedRegression. When
%           parameter value is 'resubstitution', fastRegularisedRegression
%           uses X and Y both to fit the model and to estimate the
%           accuracy, without cross-validation.  
%           The default is 
%              'resubstitution' for 'CV'
%               10 for 'AlphaCV'
%               'resubstitution' for 'FSCV'
%           'AlphaCV' parameter is ignored if value of parameter 'Alpha' is
%           not 'CV'. 
%       'MCReps', 'AlphaMCReps', 'FSMCReps' is a positive integer
%           indicating the number of Monte-Carlo repetitions for
%           cross-validation. If 
%           'CV' ('AlphaCV', 'FSCV') is a cvpartition of type 'holdout',
%           then 'MCReps' ('AlphaMCReps', 'FSMCReps') must be greater than
%           one. Otherwise it is ignored. 
%           The default value is 1. 
%       'Options', 'AlphaOptions', 'FSOptions' is a structure that contains
%           options specifying whether to conduct cross-validation
%           evaluations in parallel, and options specifying how to use
%           random numbers when computing cross validation partitions. This
%           argument can be created by a call to STATSET. CROSSVAL uses the
%           following fields:  
%               'UseParallel'
%               'UseSubstreams'
%               'Streams'
%           For information on these fields see PARALLELSTATS.
%           NOTE: If supplied, 'Streams' must be of length one.
%       'Regularize' is one of the following values:
%           'Full' is used for full regularization by addition of identity
%               matrix multiplied by 'Alpha'.
%           'Partial' is used for partial regularization by substitution of
%               'Alpha' for all eigenvalues which are less than 'Alpha'.
%           Default value is 'Full'.
%       'Alpha' is one of the following values:
%           'Coeff' means that alpha is minimal which provide that all
%               coefficients are less than or equal to 1. 
%           'Iterative' means usage of iterative method with recalculation
%               alpha.
%           'CV' means usage of cross-validation to select alpha with
%               smallest value of specified criterion. In this case
%               behaviour of function can be customized by parameters
%               'AlphaCV', 'AlphaMCReps', and 'AlphaOptions'. 
%           nonnegative real number is fixed value of alpha. 
%           negative real number -N is used for regularisation on base of
%               condition number. If 'Regularize' has value 'Full' or
%               omitted then
%                   alpha = (maxE - N*minE)/(N-1);
%               where maxE and minE are maximal and minimal singular
%               values. If 'Regularize' has value 'Partial' then all
%               eigenvalues which are less than alpha are substituted by
%               alpha with alpha = maxE/N;
%           Default is 'CV'
%       'Criterion' is criterion which is used to search optimal 'Alpha'
%           and for statistics calculation. Value of this parameter can
%           be:
%           'MSE' for mean squared error. 
%           'MAD' for maximal absolute deviation.
%           'MAR' for maximal absolute residual.
%           handle of function with two arguments:
%               res = fun(residuals, weights),
%               where residual is n-by-k (k>=1) matrix of residuals'
%               absolute values and weights is n-by-1 vector of weights.
%               All weights are anyway normalised.
%               res must be 1-by-k vector. 
%               For example, for MSE function is
%               function res = MSE(residuals, weights)
%                   res = (weights'*(residuals.^2))/(1-sum(weights.^2));
%               end
%           For CV statistics is calculated for each fold (test set) and
%           then is averaged among folds (test sets).
%           Default value is 'MSE'.
%       'FS' is feature selection method. It must be one of string:
%           'Forward' means forward feature selection method.
%           'Backward' means backward feature selection method (Feature
%               elimination).
%           'Sensitivity' means backward feature selection on base of
%               feature sensitivity.
%
%Output variable is structure with following fields:
%   PredictorNames is a cell array of names for the predictor
%           variables, in the order in which they appear in X. 
%   MultiValue - array which contains the values of multicollinearity
%       detection criteria: 1. VIF, 2. Corr, 3. Condition number
%   MultiDecision is Boolean array of decision of existence of
%       multicollinearity (true means that multicollinearity is found).
%   Alphas is vector of values of regularization parameter for all sets of
%       used input features.
%   Criterion is name of used criterion for alpha selection and statistics
%       calculation.
%   Statistics is the vector of statistics specified by Criterion for all
%       sets of used input features. This data are calculated by usage of
%       all data for fitting and testing
%   StatisticsCV is the vector of statistics specified by Criterion for all
%       sets of used input features. This value is calculated for
%       cross-validation only ('CV' is not 'resubstitution').
%   SE is the vector standard errors of mean of Statistics for all sets
%       of used input features. This value is calculated for
%       cross-validation only ('CV' is not 'resubstitution').
%   Intercept is vector of intercepts for all sets of used input features.
%   Coefficients is matrix of regression coefficients with one column for
%       each set of used input features. 
%   
    %Sanity check for X and Y
    %X must be real valued matrix without Infs and NaNs
    if ~isreal(X) || ~all(isfinite(X(:))) || isscalar(X) || length(size(X))~=2
        error(['Incorrect value for argument "X".',...
            'It must be real valued matrix without Infs and NaNs']);
    end
    
    %Define dimensions
    [n, m] = size(X);
    
    %Y must be real valued vector without Infs and NaNs and with number of
    %elements equals to n
    if ~isreal(Y) || ~all(isfinite(Y)) || numel(Y)~=n
        error(['Incorrect value for argument "X". It must be real valued ',...
            'vector without Infs and NaNs and with number of elements ',...
            'equals to number of rows in X']);
    end
    %Transform to column vector.
    Y=Y(:);

    %Get optional parameters
    weights = [];
    cv = 'resubstitution';
    cvReps = 1;
    options = [];
    predNames = {};
    alphaMeth = 'CV';
    alphaCV = 10;
    alphaMCReps = 1;
    alphaOptions = [];
    fscv = 'resubstitution';
    fscvReps = 1;
    fsoptions = [];
    criterion = 'MSE';
    regularize = [];
    fs = 'Sens';

    %Search Name-Value pairs
    for i=1:2:length(varargin)
        strTmp = varargin{i};
        if strcmpi(strTmp,'Weights')
            weights = varargin{i+1};
        elseif strcmpi(strTmp,'CV')
            cv = varargin{i+1};
        elseif strcmpi(strTmp,'MCReps')
            cvReps = varargin{i+1};
        elseif strcmpi(strTmp,'Options')
            options = varargin{i+1};
        elseif strcmpi(strTmp,'FSCV')
            fscv = varargin{i+1};
        elseif strcmpi(strTmp,'FSMCReps')
            fscvReps = varargin{i+1};
        elseif strcmpi(strTmp,'FSOptions')
            fsoptions = varargin{i+1};
        elseif strcmpi(strTmp,'PredictorNames')
            predNames = varargin{i+1};
        elseif strcmpi(strTmp,'Alpha')
            alphaMeth = varargin{i+1};
        elseif strcmpi(strTmp,'AlphaCV')
            alphaCV = varargin{i+1};
        elseif strcmpi(strTmp,'AlphaMCReps')
            alphaMCReps = varargin{i+1};
        elseif strcmpi(strTmp,'AlphaOptions')
            alphaOptions = varargin{i+1};
        elseif strcmpi(strTmp,'Criterion')
            criterion = varargin{i+1};
        elseif strcmpi(strTmp,'Regularize')
            regularize = varargin{i+1};
        elseif strcmpi(strTmp,'FS')
            fs = varargin{i+1};
        else
            error('Incorrect name or type of "Name, Value" pair');
        end
    end
    
    %Check the weights
    if isempty(weights)
        weights = ones(1,n);
    else
        %Weights must be a vector of nonnegative finite reals with at least
        %two values greater than zero and with number of elements equal to
        %number of rows in X. 
        if ~isreal(weights) || ~isfinite(weights) || sum(weights<0)>0 ||...
                sum(weights>0)<2 || numel(weights)~=n
            error(['Incorrect value for argument "Weights". It must be ',...
                'a vector of nonnegative finite reals with at least two ',...
                'values greater than zero and with number of elements ',...
                'equal to number of rows in X.']);
        end
        weights = weights(:);
    end
    %Normalise weights
    weights = weights(:)/sum(weights);
    
    %Form output structure
    res = struct();

    %Criterion
    res.Criterion = criterion;
    if isa(criterion,'function_handle')
        res.Criterion = ['User defined: ',fun2str(criterion)];
    elseif strcmpi(criterion,'MSE')
        criterion = @(r,w)(w'*(r.^2))./(1-sum(w.^2));
    elseif strcmpi(criterion,'MAD')
        criterion = @(r,w)(w'*r)./sum(w);
    elseif strcmpi(criterion,'MAR')
        criterion = @(r,w)max(r);
    else
        error(['Incorrect value of parameter "Criterion". It must be one ',...
            'of the strings: "MSE", "MAD" or "MAR" or a function handle']);
    end
    
    %PredictorNames is a cell array of strings with m elements
    if ~isempty(predNames) 
        if ~iscellstr(predNames) || length(predNames) ~= m
            error(['Incorrect value for argument PredictorNames. ',...
                'It must be a cell array of strings with m elements.']);
        else
            predNames = predNames(:)';
        end
    end

    %All cross-validations
    cv = testCV(cv, n, '', cvReps);
    alphaCV = testCV(alphaCV, n, 'Alpha', alphaMCReps);
    fscv = testCV(fscv, n, 'FS', fscvReps);

    %Method of alpha search
    if strcmpi(alphaMeth,'Coeff')
        alphaMeth = 'Coeff';
    elseif strcmpi(alphaMeth,'Iterative')
        alphaMeth = 'Iterative';
    elseif strcmpi(alphaMeth,'CV')
        alphaMeth = 'CV';
    elseif ~isnumeric(alphaMeth) || ~isfinite(alphaMeth)...
            || ~isreal(alphaMeth) || ~isscalar(alphaMeth)
        error(['Incorrect value of "Alpha" parameter. Value must be'...
            ' real number or one of the strings "Coeff",'...
            ' "Iterative" or "CV".']);
    end    
    
    %Sanity check of FS. We can accept part of words too.
    if strncmpi(fs,'Forward',length(fs))
        fs = 'Forward';
    elseif strncmpi(fs,'Backward',length(fs))
        fs = 'Backward';
    elseif strncmpi(fs,'Sensitivity',length(fs))
        fs = 'Sensitivity';
    else
        error(['Incorrect value of "FS" parameter. It must be one of ',...
            'the strings: "Forward", "Backward", "Sensitivity", ',...
            'and "ttest".']);
    end    
    
    res.PredNames = predNames;

    %Standardize data and calculate the criteria of multicollinearity
    [XS, YS, stat] = standardization (X, Y, weights);
    [res.MultiValue, res.MultiDecision] = criteria(XS, weights);
    res.Statistics = zeros(1,m+1);
    if ~strcmp(cv.Type,'resubstitution')
        res.SE = zeros(1,m);
        res.StatisticsCV = zeros(1,m);
    end
    
    if isempty(regularize) || strcmpi(regularize,'Full')
        full = @fullCorrect;
    elseif strcmpi(regularize,'Partial')
        full = @partialCorrect;
        if strcmpi(fs,'Sensitivity')
            error(['Value "Sensitivity" for parameter "FS" cannot be ',...
                'used with value "Partial" of parameter "Regularize".']);
        end
    else
        error(['Incorrect value of "Regularize" parameter. ',...
            'It must be "Full" or "Partial" only']);
    end
    
    %Form structure for feature selection methods
    custom = struct();
    custom.Func = full;
    custom.Criterion = criterion;
    custom.bestAlpha = @(mask) bestAlpha(mask, alphaMeth, XS, YS, weights,...
        stat, full, alphaCV, alphaMCReps, alphaOptions, criterion, m);
    custom.Estimate = @(mask, alpha) searchFun(alpha, [weights, X], Y,...
        fscv, fscvReps, fsoptions, full, criterion, mask);
    custom.calcCoef = @(mask, alpha) fitModel(XS, YS, weights, full,...
        alpha, mask);
    custom.sensitivity = @(mask, alpha, coeff) sensitivity(XS, YS,...
        weights, alpha, mask, coeff);

    %Feature selection
    switch fs
        case 'Forward'
            [res.Alphas, res.Coefficients] = FFS(m, custom);
        case 'Backward'
            [res.Alphas, res.Coefficients] = BFS(m, custom);
        case 'Sensitivity'
            [res.Alphas, res.Coefficients] = Sens(m, custom);
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Final data transformation
    %1. Calculate number of zero coefficients
    res.DF = sum(res.Coefficients~=0);
    %2. Calculate final value of criterion
    res.Statistics = criterion(bsxfun(@minus,YS,(XS*res.Coefficients))...
                *stat.SDY,weights);
    if ~strcmp(cv.Type,'resubstitution')
        %There is cross-validation
        %Loop for each set of features
        for k=1:m+1
            %Get mask
            mask = res.Coefficients(:,k)==0;
            alpha = res.Alphas(k);
            %Form function to use in crossval
            cvfun = @(Xtrain,Ytrain,Xtest,Ytest) modelFitAndPredict( ...
                Xtrain,Ytrain,Xtest,Ytest, ...
                full, alpha, criterion, mask);
            %Perform cross-validation
            cvStat = crossval(cvfun,[weights, X],Y, ...
                'Partition',cv,'Mcreps',cvReps,'Options',options);
            %Calculate and save statistics of CV
            res.StatisticsCV(k) = mean(cvStat);
            res.SE(k) = std(cvStat) / sqrt(size(cvStat,1));
        end
    end
    %Renormalize regression coefficients and intercept
    %3. Denormalisiation of regression coefficietns
    res.Coefficients = bsxfun(@rdivide, res.Coefficients*stat.SDY, stat.SDX');
    %4. Recalculate denormalized intercepts
    res.Intercept = stat.meanY - stat.meanX*res.Coefficients;
end

function cv = testCV(cv, n, name, cvReps)
%testCV tests consistency of cross-validation parameters.
    if isnumeric(cv) && isscalar(cv) && (cv==round(cv)) && (0<cv)
        %cv is a kfold value. Create a cvpartition.
        if (cv > n)
            error(['Incorrect number in a "', name,'CV" parameters: ',...
                'number of folds cannot be greater than number of points']);
        end
        cv = cvpartition(size(X,1),'Kfold',cv);
    elseif isa(cv,'cvpartition')
        if strcmpi(cv.Type,'leaveout')
            error(['Type "leaveout" of cvpartition in parameter "', name,...
                '" is forbidden.']);
        elseif strcmpi(cv.Type,'holdout') && mcreps<=1
            error([name,'MCReps must be greater than 1 for "holdout" ',...
                'type of cvpartition']);
        end
    elseif strncmpi(cv,'resubstitution',length(cv))
        %We assume that cv can contain the part of this word only
        cv = cvpartition(n, 'resubstitution');
    else
        error(['Incorrect type of value in "', name, 'CV" parameter.']);
    end
    if strcmpi(cv.Type,'resubstitution') && cvReps ~= 1
        error('MCReps must be 1 for "resubstitution" value of "CV" parameter');
    end
    if (cv.N ~= n) || (min(cv.TrainSize) < 2)
        %Number of elements in cv must be the same as in data matrix
        %Number of cases in each training set must be at least 2.
        error([name, 'CV. One of test sets is too small or number of ',...
            'cases in cvpartition is not the same as in data matrix']);
    end
end

function [XTrain, YTrain, stat, XTest, YTest] =...
    standardization (XTrain, YTrain, weightsTrain, XTest, YTest)
%This function is used for weighted standardization of matrix XTrain and
%vector YTrain and, possible, matrix XTest and YTest. All statistics are
%calculated from matrix XTrain and vector YTrain only.
%
%Inputs:
%   XTrain is n-by-m matrix of data in training set
%   YTrain is n-by-1 vector of response in training set
%   weightsTrain is n-by-1 vector of normalized (with unit sum) weights of
%       cases in training set 
%   XTest is k-by-m matrix of data in test set
%   YTest is k-by-1 vector of response in test set
%Outputs:
%   XTrain is n-by-m standardised matrix of data in training set
%   YTrain is n-by-1 standardised vector of response in training set
%   XTest is n-by-m standardised matrix of data in test set
%   YTest is n-by-1 standardised vector of response in test set
%   stat is structure with four field:
%       meanX is 1-by-m vector of X columns means
%       SDX is 1-by-m vector of X columns standard deviations 
%       meanY is Y means (scalar)
%       SDY is Y standard deviations (scalar)
%       DF is degree of freedom for second moment (covariance)

    stat = struct();
    %Calculate degree of freedom
    stat.DF = 1-sum(weightsTrain.^2);
    %Centring of training set
    stat.meanX = weightsTrain'*XTrain;
    XTrain = bsxfun(@minus,XTrain,stat.meanX);
    stat.meanY = weightsTrain'*YTrain;
    YTrain = YTrain - stat.meanY;
    %Standardize the training set
    stat.SDX = sqrt(weightsTrain'*(XTrain.^2)/stat.DF);
    stat.SDX(stat.SDX==0) = 1;
    stat.SDY = sqrt(weightsTrain'*(YTrain.^2)/stat.DF);
    XTrain = bsxfun(@rdivide, XTrain, stat.SDX);
    YTrain = YTrain/stat.SDY;
    %Work with test set
    if nargin>3
        %Centring of test set
        XTest = bsxfun(@minus,XTest,stat.meanX);
        YTest = YTest - stat.meanY;
        %Standardize the training set
        XTest = bsxfun(@rdivide, XTest, stat.SDX);
        YTest = YTest/stat.SDY;
    end
end

function [stat, decis] = criteria(X, weights)
%criteria calculate three criteria of multicollinearity: 
%   1. VIF is variance inflation factor. Multicollinearity is observed if
%       VIF>10 
%   2. Value of correlation coefficient. Multicollinearity is observed if
%       maximum of absolute value of off diagonal elements of correlation
%       matrix is greater than 0.4.
%   3. Condition number is fraction of maximal eigenvalue to the minimal
%       eigenvalue of correlation matrix. Multicollinearity is observed if
%       condition number is greater than 100.
%
%Inputs:
%   X is n-by-m data matrix which is standardizes with weights.
%   weights is n-by-1 vector of normalized (with unit sum) cases weights.
%Outputs:
%   stat is vector of values of three listed statistics
%   decis is vector of three decisions

    %Preallocate results
    stat = zeros(1,3);
    decis = true(1,3);
    %For all criteria we need to have covariance matrix. Calculate it.
    %Firstly calculate degree of freedom for the second moment
    df = 1-sum(weights.^2);
    C = (X'*bsxfun(@times,X,weights))/df;
    
    %The first criterion
    p = size(X,2);
    V = zeros(p,1);
    for i = 1:p
        pred = setdiff(1:p,i);
        %form SLAE
        A = C(pred,pred);
        b = C(pred,i);
        %Calculate coefficients
        x = A\b;
        %Calculate residuals
        r = (weights'*((X(:,i)-X(:,pred)*x).^2))/df;
        V(i) = 1/r;
    end
    stat(1) = max(V);
    decis(1) = stat(1) > 5;
    %The second criterion
    CC = triu(abs(C),1);
    stat(2) = max(CC(:));
    decis(2) = stat(2)>0.4;
    %The tird criterion
    s = svd(C);
    if any(s == 0)   % Handle singular matrix
        stat(3) = Inf(1);
    else
        stat(3) = max(s)./min(s);
        if isempty(stat(3))
            stat(3) = 0;
        end
    end
    decis(3) = stat(3)>100;
end

function b = fitModel(X, Y, weights, regFunc, regValue, mask)
%fitModel fits model with n-by-m data matrix X and n-by-1 response vector
%Y. Function regFunc is used to regularize covariance matrix.
%Inputs:
%   X is n-by-m data matrix which must be standardized.
%   Y is n-by-1 response vector which must be standardized.
%   weights is n-by-1 cases weights vector.
%   regFunc is function to regularize covariance matrix.
%   regValue is addition argument of regularization function.
%   mask is musk to exclude features
%
%Output:
%   b is m-by-1 vector of regression coefficients.
%
    %Form covariance matrix
    df = 1-sum(weights.^2);
    C = (X'*bsxfun(@times,X,weights))/df;
    %Calculate right hand part of SLAE
    R = X'*(weights.*Y);
    b = simpleSolve(C, R, mask, regFunc, regValue);
end

function D = partialCorrect(C, threshold, ~)
%condCorrect regularizes covariance matrix C by correction all eigenvalues
%which are less than threshold.
%If two input arguments are specified then
%Inputs:
%   C is m-by-m covariance matrix to regularize.
%   Threshold is threshold of regularization
%Output
%   C is regularized covariance matrix.
%If three input arguments are specified then 
%Inputs:
%   C is vector if singular values
%   threshold is specified value for condition number
%Output
%   D is value of alpha
    if nargin<3
        %Decompose C
        [U,S,V] = svd(C);
        %Get eigen values and check criterion
        s = diag(S);
        s(s<threshold) = threshold;
        D = U*diag(s)*V';
    else
        D = -C(1)/threshold;
    end
end

function D = fullCorrect(C, threshold, ~)
%condCorrect regularizes covariance matrix C by adding unit matrix
%multiplied by threshold. 
%If two input arguments are specified then
%Inputs
%   C is m-by-m covariance matrix to regularize 
%   Threshold is threshold of regularization
%Output
%   C is regularized covariance matrix.
%
%If three input arguments are specified then 
%Inputs:
%   C is vector if singular values
%   threshold is specified value for condition number
%Output
%   D is value of alpha
    if nargin<3
        D = C + threshold*eye(size(C,1));
    else
        D = (C(1)+C(end)*threshold)/(-threshold-1);
    end
end

function res = modelFitAndPredict(Xtrain, Ytrain, Xtest, Ytest,...
    regFunc, alpha, criterion, mask)
%modelFitAndPredict fit model by Xtrain and Ytrain and predict value of
%Ytest by Xtest and fitted model.
%Inputs:
%   Xtrain is n-by-(m+1) data matrix of training set with weights in the
%       first column. 
%   Ytrain is the n-by-1 response vector of training set.
%   Xtest is n-by-(m+1) data matrix of test set with weights in the
%       first column. 
%   Ytest is the n-by-1 response vector of test set.
%   regFunc is function to regularize covariance matrix.
%   alpha is the second argument of regFunc.
%   criterion is criterion to search Alpha
%   mask is mask of features to exclude.

    %Extract weights from training set
    trainWeights = Xtrain(:,1);
    Xtrain = Xtrain(:,2:end);
    trainWeights = trainWeights/sum(trainWeights);
    %Extract weights from test set
    testWeights = Xtest(:,1);
    testWeights = testWeights/sum(testWeights);
    Xtest = Xtest(:,2:end);
    %Centralize and standardize Xtrain, Ytrain, Xtest and Ytest
    [Xtrain, Ytrain, stat, Xtest, Ytest] =...
        standardization(Xtrain, Ytrain, trainWeights, Xtest, Ytest);
    %Fit model by Xtrain, Ytrain and trainWeights
    b = fitModel(Xtrain, Ytrain, trainWeights, regFunc, alpha, mask);
    %Calculate fitted value and renormalize it 
    yFitted = Xtest * b;
    %Calculate residuals
    resid = abs(yFitted - Ytest)*stat.SDY; 
    %Calculate statistics 
    res = criterion(resid,testWeights); 
end

function res = searchFun(c, X, Y, cv, cvReps, options, func, criterion, mask)
%searchFun calculate mean of specified quality criterion for cross
%validation and specified regularization c
%Inputs:
%   c is parameter of global regularization
%   X is data matrix X with weights as the first column
%   Y is the response vector
%   cv is object for cross-validation
%   func is regularization function
%   criterion is criterion to quality estimation
%   mask is mask of used features (true means that this feature is excluded)
        %reasonable values of c are non-negative
        if c<0
            res = Inf;
            return;
        end
        
        %Function for cross-validation
        cvfun = @(Xtrain,Ytrain,Xtest,Ytest) modelFitAndPredict( ...
            Xtrain,Ytrain,Xtest,Ytest, ...
            func, c, criterion, mask);

        cvMSE = crossval(cvfun, X,Y, ...
            'Partition',cv,'Mcreps',cvReps,'Options',options);
        %Calculate and save statistics of CV
        res = mean(cvMSE);
end

function b = simpleSolve(C, R, mask, full, param)
%simpleSolve regularizes matrix C with function full and parameter param.
%Remove all attributes which are marked in mask and solve SLAE.
    C = full(C, param);
    C(mask,:) = 0;
    C(:,mask) = 0;
    R(mask) = 0;
    %Solve new SLAE
    b = (C+diag(mask))\R;
end

function alpha = bestAlpha(mask, alphaMeth, XS, YS, weights, stat, full,...
    alphaCV, alphaMCReps, alphaOptions, criterion, m)
%bestAlpha search the best Alpha for specified parameters.
%Inputs:
%   mask is the mask of used features. True means excluded feature.
%   alphaMeth is method of Alpha search (see description of 'Alpha' in main
%       function);
%   XS is standardized data matrix
%   YS is standardized response vector
%   weights is array of weights of cases.
%   stat is structure of standardization data
%   full is function to regularize covariance matrix
%   alphaCV, alphaMCReps, alphaOptions are parameters for cross-validation
%       search of alpha (see description in main function).
%   criterion is criterion to search Alpha
%   m is number of features.
%
    %Form covariance matrix
    C = (XS'*bsxfun(@times,XS,weights))/stat.DF;
    %Calculate right hand side vector
    R = XS'*(weights.*YS);

    %Alpha selection
    if ischar(alphaMeth)
        switch alphaMeth
            case 'Coeff'
                %We want to have coefficients which are at most 1.
                %Find coefficients of non-regularized SLAE
                x = simpleSolve(C, R, mask, full, 0);
                x = max(abs(x));
                if x<=1 
                    alpha = 0;
                else
                    %We need to search coefficients
                    %Search the regularization parameter which provide x<=1
                    c = 0.5;
                    while x>1
                        c = c*2;
                        x = max(abs(simpleSolve(C, R, mask, full, c)));
                    end
                    if c<2
                        d = 0;
                    else
                        d = c/2;
                    end
                    alpha = fzero(@(x)max(abs(...
                        simpleSolve(C, R, mask, full, x)))-1,[d,c]);
                end
            case 'Iterative'
                xOld = zeros(m,1);
                x = simpleSolve(C, R, mask, full, 0);
                while sum((x-xOld).^2)>1e-6
                    c1 = (weights'*(((YS-(XS*x))*stat.SDY).^2))*m/...
                        ((x'*x)*(n-m-1));
                    if c1>1 
                        c = c/2;
                        break;
                    else
                        c = c1;
                    end
                    xOld = x;
                    x = simpleSolve(C, R, mask, full, c);
                end
                alpha = c;
            case 'CV'
                %Prepare function for fminsearch
                fmsFun = @(x)searchFun(x, [weights, XS], YS,...
                    alphaCV, alphaMCReps, alphaOptions,...
                    full, criterion, mask);
                %Start search
                alpha = fminbnd(fmsFun,0,1);
        end
    else
        %for nonnegative value then it is alpha
        if alphaMeth>=0
            alpha = alphaMeth;
        else
            %It is condition number definition of correction
            %Calculate value of threshold
            %Decompose covariance matrix
            s = svd(C);
            alpha = full(s, alphaMeth, 0);
        end
    end
end

function res = sensitivity(XS, YS, W, alpha, mask, coeff)
%sensitivity calculates sensitivity of solution with respect to removing
%each coefficient (put it zero).
%Inputs:
%   XS is n-by-m standardized data matrix
%   YS is n-by-1 standardized response vector
%   W is n-by-1 normalized vector of weights
%   alpha is parameter of full regularization 
%   mask is the mask of used features. True means excluded feature.
%   coeff is 1-by-m vector of regression coefficients 
%Output:
%   res is 1-by-m vector of importance coefficients.

    %Calculate residuals
    r = YS-XS*coeff;
    tmp = bsxfun(@plus,bsxfun(@times,r,XS),alpha*coeff');
    res = W'*abs(tmp);
    res = res .* abs(coeff)';
    res(mask) = Inf;
end

function [ alphas, coef ] = FFS( m, custom )
%FFS provide forward feature selection for system with m attributes.
%Inputs
%   m is number of features.
%   custom is structure with required data for search:
%       Func if function for regularization
%       Criterion is criterion for accuracy estimation
%       bestAlpha is function to calculate the best alpha
%       Estimate is function to estimate accuracy of solution for specified
%           value of regularization parameter and set of used attributes.
%       calcCoef is function to calculate regression coefficients for
%           specified value of regularization parameter and set of used
%           attributes.
%       sensitivity is function to calculate sensitivity for all features
%           which are specified by mask, known regularization parameter and
%           calculated regression coefficients.
%
    %Form mask
    mask = true(1,m);
    %Preallocate alpha and coef
    alphas = zeros(1,m+1);
    coef = zeros(m,m+1);
    for k = 2:m+1
        %Loop of features enumeration
        list = find(mask);
        R = length(list);
        best = inf;
        bestK = -1;
        for r = 1:R
            mask(list(r)) = false;
            res = custom.Estimate(mask, alphas(k-1));
            mask(list(r)) = true;
            if res<best
                best = res;
                bestK = r;
            end
        end
        %Now add the element with number bestK
        mask(list(bestK)) = false;
        %Select the best alpha for new feature set
        alphas(k) = custom.bestAlpha(mask);
        %Calculate coefficients of current feature set
        coef(:,k) = custom.calcCoef(mask, alphas(k));
    end
end

function [ alphas, coef ] = BFS( m, custom )
%BFS provide backward feature selection (elimination) for system with m
%attributes.
%Inputs
%   m is number of features.
%   custom is structure with required data for search:
%       Func if function for regularization
%       Criterion is criterion for accuracy estimation
%       bestAlpha is function to calculate the best alpha
%       Estimate is function to estimate accuracy of solution for specified
%           value of regularization parameter and set of used attributes.
%       calcCoef is function to calculate regression coefficients for
%           specified value of regularization parameter and set of used
%           attributes.
%       sensitivity is function to calculate sensitivity for all features
%           which are specified by mask, known regularization parameter and
%           calculated regression coefficients.
%
    %Form mask
    mask = false(1,m);
    %Preallocate alpha and coef
    alphas = zeros(1,m+1);
    coef = zeros(m,m+1);
    for k = m+1:-1:2
        %Select the best alpha for current feature set
        alphas(k) = custom.bestAlpha(mask);
        %Calculate coefficients of current feature set
        coef(:,k) = custom.calcCoef(mask, alphas(k));
        %Loop of features enumeration
        list = find(~mask);
        R = length(list);
        best = inf;
        bestK = -1;
        for r = 1:R
            mask(list(r)) = true;
            res = custom.Estimate(mask, alphas(k));
            mask(list(r)) = false;
            if res<best
                best = res;
                bestK = r;
            end
        end
        %Now exclude the element with number bestK
        mask(list(bestK)) = true;
    end
end

function [ alphas, coef ] = Sens( m, custom )
%Sens perform feature elimination algorithm with elimination of feature
%with least sensitivity.
%Inputs
%   m is number of features.
%   custom is structure with required data for search:
%       Func if function for regularization
%       Criterion is criterion for accuracy estimation
%       bestAlpha is function to calculate the best alpha
%       Estimate is function to estimate accuracy of solution for specified
%           value of regularization parameter and set of used attributes.
%       calcCoef is function to calculate regression coefficients for
%           specified value of regularization parameter and set of used
%           attributes.
%       sensitivity is function to calculate sensitivity for all features
%           which are specified by mask, known regularization parameter and
%           calculated regression coefficients.
%
    %Form mask
    mask = false(1,m);
    %Preallocate alpha and coef
    alphas = zeros(1,m+1);
    coef = zeros(m,m+1);
    for k = m+1:-1:2
        %Select the best alpha for current feature set
        alphas(k) = custom.bestAlpha(mask);
        %Calculate coefficients of current feature set
        coef(:,k) = custom.calcCoef(mask, alphas(k));
        %Calculate sensitivities
        mu = custom.sensitivity(mask, alphas(k), coef(:,k));
        %Select the number of minimal value
        [~, bestK] = min(mu);
        %Now exclude the element with number bestK
        mask(bestK) = true;
    end
end