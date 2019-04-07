function [B, FitInfo] = PQSQRegularRegr(X, Y, varargin)
%PQSQRegularRegr calculates PQSQ regularization of linear regression.
%Syntax:
%   B = PQSQRegularRegr(X, Y)
%   B = PQSQRegularRegr(X, Y, Name, Value)
%   [B, FitInfo] = PQSQRegularRegr(X, Y)
%   [B, FitInfo] = PQSQRegularRegr(X, Y, Name, Value)
%
%Examples:
%       %(1) Run the lasso and PQSQ lasso simulation on data obtained from
%       the 1985 %Auto Imports Database  of the UCI repository.  
%       %http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
%
%       load imports-85;
%
%       %Extract Price as the response variable and extract non-categorical
%       %variables related to auto construction and performance
%       %
%       X = X(~any(isnan(X(:,1:16)),2),:);
%       Y = X(:,16);
%       Y = log(Y);
%       X = X(:,3:15);
%       predictorNames = {'wheel-base' 'length' 'width' 'height' ...
%           'curb-weight' 'engine-size' 'bore' 'stroke' 'compression-ratio' ...
%           'horsepower' 'peak-rpm' 'city-mpg' 'highway-mpg'};
% 
%       %Compute the default sequence of lasso fits. Fix time.
%       tic;
%       [B,S] = lasso(X,Y,'CV',10,'PredictorNames',predictorNames);
%       disp(['Time for lasso ',num2str(toc),' seconds']);
% 
%       %Compute PQSQ simulation of lasso without trimming
%       tic;
%       [BP,SP] = PQSQRegularRegr(X,Y,'CV',10,...
%           'PredictorNames',predictorNames);
%       disp(['Time for PQSQ simulation of lasso without trimming ',...
%            num2str(toc),' seconds']);
% 
%       %Compute PQSQ simulation of lasso with trimming
%       tic;
%       [BT,ST] = PQSQRegularRegr(X,Y,'CV',10,...
%           'TrimmingPoint',-1,'PredictorNames',predictorNames);
%       disp(['Time for PQSQ simulation of lasso with trimming ',...
%            num2str(toc),' seconds']);
% 
% 
%       %Display a trace plot of the fits.
%       %Lasso
%       lassoPlot(B,S);
%       %PQSQ without trimming
%       PQSQRegularRegrPlot(BP,SP);
%       %PQSQ with trimming
%       PQSQRegularRegrPlot(BT,ST);
% 
%       %Display the sequence of cross-validated predictive MSEs.
%       %Lasso
%       lassoPlot(B,S,'PlotType','CV');
%       %PQSQ without trimming
%       PQSQRegularRegrPlot(BP,SP,'PlotType','CV');
%       %PQSQ with trimming
%       PQSQRegularRegrPlot(BT,ST,'PlotType','CV');
%        
%       %Look at the kind of fit information returned by 
%       %lasso.
%       disp('lasso');
%       S
%       %PQSQ without trimming
%       disp('PQSQ without trimming');
%       SP
%       %PQSQ with trimming
%       disp('PQSQ with trimming');
%       ST
% 
%       %What variables are in the model corresponding to minimum 
%       %cross-validated MSE, and in the sparsest model within one 
%       %standard error of that minimum. 
%       disp('lasso');
%       S.PredictorNames(B(:,S.IndexMinMSE)~=0)
%       S.PredictorNames(B(:,S.Index1SE)~=0)
%       disp('PQSQRegularRegr without trimming');
%       SP.PredictorNames(BP(:,SP.IndexMinMSE)~=0)
%       SP.PredictorNames(BP(:,SP.Index1SE)~=0)
%       disp('PQSQRegularRegr with trimming');
%       ST.PredictorNames(BT(:,ST.IndexMinMSE)~=0)
%       ST.PredictorNames(BT(:,ST.Index1SE)~=0)
% 
%       %Fit the sparse model and examine residuals. 
%       %Lasso.
%       Xplus = [ones(size(X,1),1) X];
%       fitSparse = Xplus * [S.Intercept(S.Index1SE); B(:,S.Index1SE)];
%       a = 'lasso';
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
%       figure
%       plot(fitSparse,Y-fitSparse,'o');
%       %PQSQRegularRegr without trimming.
%       fitSparse = Xplus * [SP.Intercept(SP.Index1SE); BP(:,SP.Index1SE)];
%       a = 'PQSQRegularRegr without trimming';
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
%       figure
%       plot(fitSparse,Y-fitSparse,'o')
%       %PQSQRegularRegr with trimming.
%       fitSparse = Xplus * [ST.Intercept(ST.Index1SE); BT(:,ST.Index1SE)];
%       a = 'PQSQRegularRegr with trimming';
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
%       figure
%       plot(fitSparse,Y-fitSparse,'o')
% 
%       %Consider a slightly richer model. A model with 7 variables may be a 
%       %reasonable alternative.  Find the index for a corresponding fit.
%       a = 'lasso';
%       df7index = find(S.DF==7,1);
%       fitDF7 = Xplus * [S.Intercept(df7index); B(:,df7index)];
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
%       figure
%       plot(fitDF7,Y-fitDF7,'o')         
%       a = 'PQSQRegularRegr without trimming';
%       df7index = find(SP.DF==7,1);
%       fitDF7 = Xplus * [SP.Intercept(df7index); BP(:,df7index)];
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
%       figure
%       plot(fitDF7,Y-fitDF7,'o')         
%       a = 'PQSQRegularRegr with trimming';
%       df7index = find(ST.DF==7,1);
%       fitDF7 = Xplus * [ST.Intercept(df7index); BT(:,df7index)];
%       disp([a, '. Correlation coefficient of fitted values ',...
%           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
%       figure
%       plot(fitDF7,Y-fitDF7,'o')         
%         
%       %(2) Run lasso and PQSQ simulation of lasso on some random data
%       %with 250 predictors
%       %Data generation
%       n = 1000; p = 250;
%       X = randn(n,p);
%       beta = randn(p,1); beta0 = randn;
%       Y = beta0 + X*beta + randn(n,1);
%       lambda = 0:.01:.5;
%       %Run lasso
%       tic;
%       [B,S] = lasso(X,Y,'Lambda',lambda);
%       disp(['Time for lasso ',num2str(toc),' seconds']);
%       lassoPlot(B,S);
%       %Run PQSQ simulation of lasso with the same lambdas and without
%       %trimming
%       tic;
%       [BP,SP] = PQSQRegularRegr(X,Y,'Lambda',lambda);
%       disp(['Time for PQSQ simulation of lasso without trimming ',...
%            num2str(toc),' seconds']);
%       PQSQRegularRegrPlot(BP,SP);
%
%       %Compute PQSQ simulation of lasso with trimming and the same lambdas
%       tic;
%       [BT,ST] = PQSQRegularRegr(X,Y,'Lambda',lambda,'TrimmingPoint',-1);
%       disp(['Time for PQSQ simulation of lasso with trimming ',...
%            num2str(toc),' seconds']);
%       PQSQRegularRegrPlot(BT,ST);
%
%       %compare against OLS
%       %Lasso
%       figure
%       bls = [ones(size(X,1),1) X] \ Y;
%       plot(bls,[S.Intercept; B],'.');
%       %PQSQ without trimming
%       figure
%       plot(bls,[SP.Intercept; BP],'.');
%       %PQSQ with trimming
%       figure
%       plot(bls,[ST.Intercept; BT],'.');
%
%
%
%Inputs
%   X is numeric matrix with n rows and p columns. Each row represents one
%       observation, and each column represents one predictor (variable). 
%   Y is numeric vector of length n, where n is the number of rows of X.
%       Y(i) is the response to row i of X.
%
%   IMPORTANT NOTES: data matrix X and response vector Y are standardized
%       anyway.
%
%   Name, Value is one or more pairs of name and value. There are several
%       possible names with corresponding values: 
%       'Lambda' is vector of Lambda values. It will be returned in return
%           argument FitInfo in ascending order. The default is to have
%           PQSQRegularRegr generate a sequence of lambda values, based on
%           'NumLambda' and 'LambdaRatio'. PQSQRegularRegr will generate a
%           sequence, based on the values in X and Y, such that the largest
%           LAMBDA value is just sufficient to produce all zero
%           coefficients B in standard lasso. You may supply a vector of
%           real, non-negative values of lambda for PQSQRegularRegr to use,
%           in place of its default sequence. If you supply a value for
%           'Lambda' then values of 'NumLambda' and 'LambdaRatio' are
%           ignored. 
%       'NumLambda' is the number of lambda values to use, if the parameter
%           'Lambda' is not supplied (default 100). It is ignored if
%           'Lambda' is supplied. PQSQRegularRegr may return fewer fits
%           than specified by 'NumLambda' if the residual error of the fits
%           drops below a threshold percentage of the variance of Y.
%       'LambdaRatio' is ratio between the minimum value and maximum value
%           of lambda to generate, if the  parameter "Lambda" is not
%           supplied. Legal range is [0,1). Default is 0.0001. If
%           'LambdaRatio' is zero, PQSQRegularRegr will generate its
%           default sequence of lambda values but replace the smallest
%           value in this sequence with the value zero. 'LambdaRatio' is
%           ignored if 'Lambda' is supplied.
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. Default: {}
%       'Weights' is vector of observation weights. It must be a vector of
%           non-negative values, of the same length as columns of X. At
%           least two values must be positive. Default (1/N)*ones(N,1).
%       'Epsilon' is real number between 0 and 1 which specify the radius
%           of black hole r. The meaning of this argument is the fraction of
%           coefficients which has to be put to zero for unregularized
%           problem. Procedure of coefficients putting to zero is:
%           1. calculate regression coefficients for unregularized problem.
%           2. identify all coefficients which less than r.
%           3. if there is no small coefficients then stop.
%           4. remove predictors which are corresponded to small
%               coefficients and go to step 1.
%           If number of removed coefficients is less than number of
%           predictors multiplied by Epsilon, then increase r.
%           If number of removed coefficients is greater than number of
%           predictors multiplied by Epsilon, then decrease r.
%           Default is 0.01 (1% of coefficients has to be put to zero).
%       'Majorant' is handle of majorant function or cell array with two
%           elements, first of which is handle of majorant function and the
%           second is weight of this function in linear combination.
%           Several pairs of 'Majorant', Values can be specified. If
%           argument is handle of majorant function then weight is equal to
%           one. Weight can be changed by argument 'MajorantWeight' which
%           is appear after current 'Majorant' argument but before next one. 
%           There are several special values of 'Majorant':
%           {'elasticnet', num} is imitation of
%               elastic net with parameter alpha equals num. Num must be
%               real number between zero and one inclusively. It is
%               equivalent to consequence of two pairs of majorant
%               parameters:  
%               'Majorant', {@L1, num}, Majorant', {@L2,num1}
%               where num1 = 1-num or to four pairs:
%               'Majorant', @L1, 'MajorantWeight', num, 'Majorant', @L2,
%               'MajorantWeight', num1 
%           'lasso' is simplest imitation of lasso. It is
%               equivalent to pair 'Majorant', @L1. 
%           'ridge' is simplest imitation of ridge regression. It is
%               equivalent to pair 'Majorant', @L2.
%       'MajorantWeight' is new weight for the last specified majorant. It
%           must be non-negative value.
%       'Intervals' is specification of intervals for PQSQ approximation of
%           majorant function or linear combination of majorant functions.
%           Intervals must be array of intervals boundaries. In this case
%           it has to be row vector. The first element must be zero. All
%           other elements must be sorted in ascending order.
%           Default is empty.
%           If you supply a value for 'Intervals' then values for
%           'IntervalsByQudratic', 'IntervalsByOptimal',
%           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
%           'IntervalsByAccuracyMajorant' are ignored
%       'IntervalsByQudratic' specifies the required number of intervals.
%           The value N must be positive integer. Intervals boundaries are
%           calculated as TrimmingPoint*((0:N)/N).^2;
%           This option is not supplied by default.
%           This option is ignored if 'Intervals' is supplied.
%           This option is also ignored if 'IntervalsByOptimal',
%           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
%           'IntervalsByAccuracyMajorant' is supplied later than it. 
%       'IntervalsByOptimal' specifies the positive integer N which is used
%           for calculations of the optimal positions of N intervals
%           boundaries. The most computationally expansive way of intervals
%           specification. The procedure of boundaries search optimise the
%           maximum of difference between majorant function and PQSQ
%           approximation of it.
%           This option is not supplied by default.
%           This option is ignored if 'Intervals' is supplied.
%           This option is also ignored if 'IntervalsByQudratic',
%           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
%           'IntervalsByAccuracyMajorant' is supplied later than it. 
%       'IntervalsByAccuracy' is a positive number which specify the
%           required accuracy of PQSQ approximation of majorant function.
%           In this case all intervals exclude last are created to have
%           specified value of maximum of difference between majorant
%           function and PQSQ approximation of it. The last interval can
%           has maximal difference that is less then required.
%           This option is not supplied by default.
%           This option is ignored if 'Intervals' is supplied.
%           This option is also ignored if 'IntervalsByOptimal',
%           'IntervalsByQudratic' or 'IntervalsByAccuracyMajorant'
%           is supplied later than it. 
%       'IntervalsByAccuracyTrimming' is a positive number which define
%           the required accuracy of PQSQ approximation of majorant
%           function as product of specified value and TrimmingPoint.
%           In this case all intervals exclude last are created to have
%           specified value of maximum of difference between majorant
%           function and PQSQ approximation of it. The last interval can
%           has maximal difference that is less then required.
%           This option is not supplied by default.
%           This option is ignored if 'Intervals' is supplied.
%           This option is also ignored if 'IntervalsByOptimal',
%           'IntervalsByQudratic', 'IntervalsByAccuracy', or
%           'IntervalsByAccuracyMajorant' is supplied later than it. 
%       'IntervalsByAccuracyMajorant' is a positive number which define
%           the required accuracy of PQSQ approximation of majorant
%           function as product of specified value and value of majorant
%           function of TrimmingPoint. In this case all intervals exclude
%           last are created to have specified value of maximum of
%           difference between majorant function and PQSQ approximation of
%           it. The last interval can has maximal difference that is less
%           then required.
%           Default value is 0.02.
%           This option is ignored if 'Intervals' is supplied.
%           This option is also ignored if 'IntervalsByQudratic',
%           'IntervalsByOptimal', 'IntervalsByAccuracy' or
%           'IntervalsByAccuracyTrimming' is supplied later than it. 
%       'TrimmingPoint' is a real number. If it is positive number then it
%           is the trimming point. If it is negative value then it is minus
%           multiplier for default trimming point value.
%           By default it is maximum of absolute value of unregularised
%           regression coefficient. Such value of trimming point assumes
%           the trimming of regularisation term. To avoid trimming it is
%           necessary to specify another value or use pair 
%           'TrimmingPoint', -2.
%       'CV' If present, indicates the method used to compute MSE. When
%           'CV' is a positive integer K, PQSQRegularRegr uses K-fold cross-
%           validation.  Set 'CV' to a cross-validation partition, created
%           using CVPARTITION, to use other forms of cross-validation. You
%           cannot use a 'Leaveout' partition with PQSQRegularRegr. When
%           'CV' is 'resubstitution', PQSQRegularRegr uses X and Y both to
%           fit the model and to estimate the mean squared errors, without
%           cross-validation.   
%           The default is 'resubstitution'.
%       'MCReps' is a positive integer indicating the number of Monte-Carlo
%           repetitions for cross-validation.  The default value is 1. If
%           'CV' is a cvpartition of type 'holdout', then 'MCReps' must be
%           greater than one. Otherwise it is ignored.
%       'Options' is a structure that contains options specifying whether
%           to conduct cross-validation evaluations in parallel, and
%           options specifying how to use random numbers when computing
%           cross validation partitions. This argument can be created by a
%           call to STATSET. CROSSVAL uses the following fields: 
%               'UseParallel'
%               'UseSubstreams'
%               'Streams'
%           For information on these fields see PARALLELSTATS.
%           NOTE: If supplied, 'Streams' must be of length one.
%Return values:
%   B is the fitted coefficients for each model. B will have dimension PxL,
%       where P = size(X,2) is the number of predictors, and 
%       L = length(lambda). 
%   FitInfo is a structure that contains information about the sequence of
%       model fits corresponding to the columns of B. STATS contains the
%       following fields:
%       'Intercept' is the intercept term for each model. Dimension is 1xL.
%       'Lambda' is the sequence of lambda penalties used, in ascending
%           order. Dimension is 1xL. 
%       'Regularization' is cell matrix. Each row of matrix corresponds to
%           regularization in form {func, weight} where weight is column
%           with normalized weights alpha of regularization terms, func is column
%           with function handles for majorant function.
%       'Intervals' is row array of intervals without last (Inf) value.
%       'BlackHoleRadius' is radius of black hole.
%       'DF' is the number of nonzero coefficients in B for each value of
%           lambda. Dimension is 1xL. 
%       'MSE' is the mean squared error of the fitted model for each value
%           of lambda. If cross-validation was performed, the values for
%           'MSE' represent Mean Prediction Squared Error for each value of
%           lambda, as calculated by cross-validation. Otherwise, 'MSE' is
%           the mean sum of squared residuals obtained from the model with
%           B and FitInfo.Intercept. 
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. 
%
%If cross-validation was performed, FitInfo also includes the following
%fields: 
%       'SE' is a row vector which contains the standard error of MSE for
%           each lambda, as calculated during cross-validation. 
%           Dimension 1xL.
%       'LambdaMinMSE' is the lambda value with minimum MSE. Scalar.
%       'Lambda1SE' is the largest lambda such that MSE is within one
%           standard error of the minimum. Scalar. 
%       'IndexMinMSE' is the index of Lambda with value LambdaMinMSE.
%       'Index1SE' is the index of Lambda with value Lambda1SE.



    %Sanity-check of X and Y
    %X must be real valued matrix without Infs and NaNs
    if ~isreal(X) || ~all(isfinite(X(:))) || isscalar(X) || length(size(X))~=2
        error('Incorrect value for argument "X". It must be real valued matrix without Infs and NaNs');
    end
    
    %Define dimensions
    [n, m] = size(X);
    
    %Y must be real valued vector without Infs and NaNs and with number of
    %elements equals to n
    if ~isreal(Y) || ~all(isfinite(Y)) || numel(Y)~=n
        error(['Incorrect value for argument "Y". It must be real valued',...
            'vector without Infs and NaNs and with number of elements',...
            'equals to number of rows in X']);
    end
    %Transform to column vector.
    Y=Y(:);
        
    %Get optional parameters
    lambda = [];
    nLambda = 100;
    LRdefault = 0.00001;
    lRatio = LRdefault;
    predNames = {};
    weights = [];
    epsilon = 0.01;
    majorants = cell(100,2);
    nMajorants = 0;
    intervals = [];
    intervalsMeth = 'IntervalsByAccuracyMajorant';
    intervalsValue = 0.02;
    trimmingPoint = -2;
    cv = 'resubstitution';
    cvReps = 1;
    options = [];
    
    %Search Name-Value pairs
    for i=1:2:length(varargin)
        strTmp = varargin{i};
        if strcmpi(strTmp,'Lambda')
            lambda = varargin{i+1};
        elseif strcmpi(strTmp,'NumLambda')
            nLambda = varargin{i+1};
        elseif strcmpi(strTmp,'LambdaRatio')
            lRatio = varargin{i+1};
        elseif strcmpi(strTmp,'PredictorNames')
            predNames = varargin{i+1};
        elseif strcmpi(strTmp,'Weights')
            weights = varargin{i+1};
        elseif strcmpi(strTmp,'Epsilon')
            epsilon = varargin{i+1};
        elseif strcmpi(strTmp,'Majorant')
            tmp = varargin{i+1};
            if ischar(tmp)
               %Special values of majorant 
                if strcmpi(tmp,'lasso')
                    nMajorants = nMajorants + 1;
                    majorants(nMajorants,:) = {@L1, 1};
                elseif strcmpi(tmp,'ridge')
                    nMajorants = nMajorants + 1;
                    majorants(nMajorants,:) = {@L2, 1};
                else
                    error('Incorrect string in "Majorant" argument'); 
                end
            elseif iscell(tmp)
                if length(tmp)~=2
                    error('Cell array in "Majorant" argument must contains two elements');
                end
                t = tmp{1};
                tt = tmp{2};
                %tt (Alpha) must be positive real less than 1
                if ~isreal(tt) || tt<=0 || tt>=1
                    error(['Incorrect value of the second element of',...
                        ' cell array in "Majorant" term. It must be real',...
                        ' between 0 and 1 inclusively']);
                end
                if ischar(t)
                    if strcmpi(t,'elasticnet')
                        nMajorants = nMajorants + 2;
                        majorants(nMajorants-1,:) = {@L1, tt};
                        majorants(nMajorants,:) = {@L2, 1-tt};
                    end
                elseif isa(t,'function_handle')
                    nMajorants = nMajorants + 1;
                    majorants(nMajorants,:) = {t, tt};
                else
                    error(['Incorrect value of the first element of',...
                        ' cell array in "Majorant" term. It must be',...
                        ' string "elasticnet" or function handle']);
                end
            elseif isa(tmp,'function_handle')
                nMajorants = nMajorants + 1;
                majorants(nMajorants,:) = {tmp, 1};
            else
                error(['Incorrect value of the first element of',...
                    ' value in "Majorant" term. It must be',...
                    ' cell array or function handle']);
            end
        elseif strcmpi(strTmp,'MajorantWeight')
            tt = varargin{i+1};
            %It must be non-negative real value
            if ~isnumeric(tt) || tt<=0
                error(['Incorrect value in "MajorantWeight" term.',...
                    ' It must be non-negative real value.']);
            end
            %At least one Majorant must be specified before
            if nMajorants<1
                error(['At least one "Majorant" must be specified',...
                    'before "MajorantWeight" term.']);
            end
            majorants{nMajorants,2} = tt;
        elseif strcmpi(strTmp,'Intervals')
            intervals = varargin{i+1};
        elseif strcmpi(strTmp,'IntervalsByQudratic')
            intervalsMeth = strTmp;
            intervalsValue = varargin{i+1};
        elseif strcmpi(strTmp,'IntervalsByOptimal')
            intervalsMeth = strTmp;
            intervalsValue = varargin{i+1};
        elseif strcmpi(strTmp,'IntervalsByAccuracy')
            intervalsMeth = strTmp;
            intervalsValue = varargin{i+1};
        elseif strcmpi(strTmp,'IntervalsByAccuracyTrimming')
            intervalsMeth = strTmp;
            intervalsValue = varargin{i+1};
        elseif strcmpi(strTmp,'IntervalsByAccuracyMajorant')
            intervalsMeth = strTmp;
            intervalsValue = varargin{i+1};
        elseif strcmpi(strTmp,'TrimmingPoint')
            trimmingPoint = varargin{i+1};
        elseif strcmpi(strTmp,'CV')
            cv = varargin{i+1};
        elseif strcmpi(strTmp,'MCReps')
            cvReps = varargin{i+1};
        elseif strcmpi(strTmp,'Options')
            options = varargin{i+1};
        else
            error('Incorrect name or type of "Name, Value" pair');
        end
    end
    
    %Sanity-check of parameters and redefine all necessary values

    if ~isreal(epsilon) || ~isscalar(epsilon) || epsilon>1 || epsilon<0
        error(['Incorrect value for argument "Epsilon". ',...
            'It must be a real scalar between zero and one inclusevely']);
    end
    
    %weights
    if isempty(weights)
        weights = ones(1,n);
    else
        %Weights must be a vector of nonnegative finite reals with at least two
        %values greater than zero and with number of elements equal to number
        %of rows in X. 
        if ~isreal(weights) || ~isfinite(weights) || sum(weights<0)>0 ||...
                sum(weights>0)<2 || numel(weights)~=n
            error(['Incorrect value for argument "Weights". It must be ',...
                'a vector of nonnegative finite reals with at least two',...
                'values greater than zero and with number of elements equal',...
                ' to number of rows in X.']);
        end
        weights = weights(:)';
    end
    %Normalise weights
    weights = weights/sum(weights);

    %Centralize
    meanX = weights*X;
    Xs = bsxfun(@minus,X,meanX);
    meanY = weights*Y;
    Ys = Y-meanY;
    %Standardize
    SX = sqrt(weights*(Xs.^2));
    SX(SX==0) = 1;
    SY = sqrt(weights*(Ys.^2));
    Xs = bsxfun(@rdivide, Xs, SX);
    Ys = Ys/SY;
    %Weighted version of X
    XW = bsxfun(@times, Xs, weights')';
    %The first matrix in SLAE
    M = XW*Xs;
    %The right hand side for SLAE
    R = XW*Ys;

    %Lambda
    if isempty(lambda)
        %nLambda be positive integer.
        if ~isreal(nLambda) || ~isfinite(nLambda) || nLambda < 1
            error('Incorrect value for argument "NumLambda". It must be positive integer');
        else
            nLambda = floor(nLambda);
        end

        % LambdaRatio should be real in [0,1).
        if ~isreal(lRatio) || lRatio <0 || lRatio >= 1
            error('Incorrect value for argument "LambdaRatio". It must be real in [0,1).');
        end
        
        %Calculate lambda maximal (provide all coefficient is zero for
        %lasso)
        lambdaMax = max(abs(R))*10;
        
        if nLambda==1
            lambda = lambdaMax;
        else
            % Fill in a number "nLambda" of smaller values, on a log scale.
            if lRatio==0
                    lRatio = LRdefault;
                    addZeroLambda = true;
            else
                addZeroLambda = false;
            end
            lambdaMin = lambdaMax * lRatio;
            loghi = log(lambdaMax);
            loglo = log(lambdaMin);
            logrange = loghi - loglo;
            interval = -logrange/(nLambda-1);
            lambda = exp(loghi:interval:loglo);
            if addZeroLambda
                lambda(end) = 0;
            else
                lambda(end) = lambdaMin;
            end
        end

    else
        %Lambda must be vector of nonnegative real values
        if ~isreal(lambda) || any(lambda < 0)
            error('Incorrect value for argument Lambda. Lambda must be vector of nonnegative real values.');
        end
    end
    lambda = sort(lambda(:),1)';
    nLambda = size(lambda,2);
    
    %PredictorNames is a cell array of strings with m elements
    if ~isempty(predNames) 
        if ~iscellstr(predNames) || length(predNames) ~= m
            error('Incorrect value for argument PredictorNames. It must be a cell array of strings with m elements.');
        else
            predNames = predNames(:)';
        end
    end

    %Solve problem without restiction to obtain information for coefficient
    %values. ????????
    x = M\R;

    %Select epsilon
    tmp = sort(abs(x));
    t = round(m*epsilon);
    if t == m
        %In this case we select the average of two greatest coefficients
        epsilon = (tmp(end)+tmp(end-1))/2;
    elseif t == 0
        %In this case we take 0.99 of minimal regression coefficient
        epsilon = tmp(1)*0.99;
    else
        %Search of epsilon for user specified fraction of dead coefficients
        epsilon = searchEpsilon(M, R, t, tmp(end), tmp(1));
    end
    
    % cvReps
    if ~isscalar(cvReps) || ~isreal(cvReps) || ~isfinite(cvReps) || cvReps < 1
        error(['If the parameter "CV" is a "holdout" partition, "MCReps"'...
            ' must be greater than one. If "CV" is "resubstitution"'...
            ' or a "resubstitution" type partition, "MCReps"'...
            'must be one (the default).']);
    end
    cvReps = fix(cvReps);
    
    %Cross-validation
    if isnumeric(cv) && isscalar(cv) && (cv==round(cv)) && (0<cv)
        %cv is a kfold value. Create a cvpartition.
        if (cv > n)
            error(['Incorrect number in a "CV" parameters: number of folds',...
                ' cannot be greater than number of points']);
        end
        cv = cvpartition(size(Xs,1),'Kfold',cv);
    elseif isa(cv,'cvpartition')
        if strcmpi(cv.Type,'resubstitution')
            cv = 'resubstitution';
        elseif strcmpi(cv.Type,'leaveout')
            error('Type "leaveout" of cvpartition in parameter "CV" is forbidden.');
        elseif strcmpi(cv.Type,'holdout') && cvReps <= 1
            error('MCReps must be greater than 1 for "holdout" type of cvpartition');
        end
    elseif strncmpi(cv,'resubstitution',length(cv))
        %We assume that cv can contain the part of this word only
        cv = 'resubstitution';
    else
        error('Incorrect type of value in "CV" parameter.');
    end
    if strcmp(cv,'resubstitution') && cvReps ~= 1
        error('MCReps must be 1 for "resubstitution" value of "CV" parameter');
    end
    if isa(cv,'cvpartition')
        if (cv.N ~= n) || (min(cv.TrainSize) < 2)
            %Number of elements in cv must be the same as in data matrix
            %Number of cases in each train set must be at least 2.
            error(['One of test sets is too small or number of ',...
                'cases in cvpartition is not the same as in data matrix']);
        end
    end
    
    %Majorants
    if nMajorants == 0
        %There is no specified majorants. It means that we use 'lasso':
        nMajorants = 1;
        majorants(nMajorants,:) = {@L1, 1};
    end
    
    %Normalize majorants
    tmp = sum([majorants{:, 2}]);
    for k=1:nMajorants
        majorants{k, 2} = majorants{k, 2} / tmp;
    end
    
    %Calculate intervals
    if isempty(intervals)
        %There is no specified intervals. We must calculate it.
        %Firstly we need to know the trimming point
        if trimmingPoint<0 
            %trimmingPoint contains minus multiplier for maximum of
            %regression coefficients absolute value
            trimmingPoint = -trimmingPoint*max(abs(x));
        end
        
        %Now we can start cration of intervals in accordance with specified
        %method and value

        if strcmpi(intervalsMeth,'IntervalsByQudratic')
            if ~isreal(intervalsValue) || intervalsValue<1 ...
                || floor(intervalsValue)~=intervalsValue
                error(['For term "IntervalsByQudratic" value must be ',...
                    'positive integer']);
            end
            intervals = trimmingPoint*((0:intervalsValue)/intervalsValue).^2;
        elseif strcmpi(intervalsMeth,'IntervalsByOptimal')
            if ~isreal(intervalsValue) || intervalsValue<1 ...
                || floor(intervalsValue)~=intervalsValue
                error(['For term "IntervalsByOptimal" value must be ',...
                    'positive integer']);
            end
            intervals = optimalIntervals( trimmingPoint, intervalsValue,...
                @(y)majorantF(y,majorants,nMajorants));
        elseif strcmpi(intervalsMeth,'IntervalsByAccuracy')
            if ~isreal(intervalsValue) || intervalsValue<=0
                error(['For term "IntervalsByAccuracy" value must be ',...
                    'positive real']);
            end
            intervals = intervalsByAccuracy( trimmingPoint, intervalsValue,...
                @(y)majorantF(y,majorants,nMajorants));
        elseif strcmpi(intervalsMeth,'IntervalsByAccuracyTrimming')
            if ~isreal(intervalsValue) || intervalsValue<=0
                error(['For term "IntervalsByAccuracyTrimming" value ',...
                    'must be positive real']);
            end
            intervals = intervalsByAccuracy( trimmingPoint,...
                intervalsValue*trimmingPoint,...
                @(y)majorantF(y,majorants,nMajorants));
        elseif strcmpi(intervalsMeth,'IntervalsByAccuracyMajorant')
            if ~isreal(intervalsValue) || intervalsValue<=0
                error(['For term "IntervalsByAccuracyMajorant" value ',...
                    'must be positive real']);
            end
            intervals = intervalsByAccuracy( trimmingPoint,...
                intervalsValue*majorantF(trimmingPoint,majorants,nMajorants),...
                @(y)majorantF(y,majorants,nMajorants));
        end
    else
        %Check the quality of intervals
        if ~isreal(intervals) || ~all(intervals(2:end)) || intervals(1)~=0 || size(intervals,1)>1
            error(['Value of term "Intervals" must contain row vector',...
                ' of real numbers. The first element must be equal ',...
                'to zero. All other elementa must be positive.']);
        end
        intervals = sort(intervals);
    end
    
    %Now we have intervals. We need to calculate all coefficients for PQSQ
    %function
    potFunc = struct();
    potFunc.Intervals = [intervals, Inf];
    [potFunc.A,potFunc.B] = ...
            computeABcoefficients(intervals,...
            @(y)majorantF(y,majorants,nMajorants));
    
    %Preallocate data to return
    B = zeros(m,nLambda);
    FitInfo = struct();
    FitInfo.Intercept = [];
    FitInfo.Lambda = lambda;
    FitInfo.BlackHoleRadius = epsilon;
    FitInfo.Regularization = majorants(1:nMajorants,:);
    FitInfo.Intervals = intervals;
    FitInfo.DF = [];
    FitInfo.MSE = [];
    FitInfo.PredictorNames = predNames;
    %If cross validation is used
    if ~strcmp(cv,'resubstitution')
        FitInfo.SE = [];
        FitInfo.LambdaMinMSE = 0;
        FitInfo.Lambda1SE = [];
        FitInfo.IndexMinMSE = 0;
        FitInfo.Index1SE = [];
    end
    
    %Main lambda loop
    for k=1:nLambda
        %Solve problem
        B(:,k) = fitModel(M, R, lambda(k), x, potFunc, epsilon);
    end
    
    %1. Calculate number of zero coefficients
    FitInfo.DF = sum(B~=0);
    %2. Calculate mean squared error
    FitInfo.MSE = weights*((bsxfun(@minus,Y,(Xs*B))*SY).^2);
    %3. Denormalisiation of regression coefficietns
    B = bsxfun(@rdivide, B*SY, SX');
    %4. Recalculate denormalized intercepts
    FitInfo.Intercept = meanY - meanX*B;
    
    %Perform cross-validation if it is required
    if ~strcmp(cv,'resubstitution')
        %Form function to use in crossval
        cvfun = @(Xtrain,Ytrain,Xtest,Ytest) modelFitAndPredict( ...
            Xtrain,Ytrain,Xtest,Ytest, ...
            lambda, potFunc, epsilon);
        %Perform cross-validation
        cvMSE = crossval(cvfun,[weights(:) X],Y, ...
            'Partition',cv,'Mcreps',cvReps,'Options',options);
        %Calculate and save statistics of CV
        FitInfo.MSE = mean(cvMSE);
        FitInfo.SE = std(cvMSE) / sqrt(size(cvMSE,1));
        [~, FitInfo.IndexMinMSE] = min(FitInfo.MSE);
        FitInfo.LambdaMinMSE = lambda(FitInfo.IndexMinMSE);
        tmp = FitInfo.MSE(FitInfo.IndexMinMSE)...
            + FitInfo.SE(FitInfo.IndexMinMSE);
        ind = find((FitInfo.MSE(FitInfo.IndexMinMSE:end)<=tmp),1,'last');
        if ~isempty(ind)
            ind = ind + FitInfo.IndexMinMSE - 1;
            FitInfo.Lambda1SE = lambda(ind);
            FitInfo.Index1SE = ind;
        end
    end    
end

function b = fitModel(M, R, lambda, b, pFunc, epsilon)
%fitModel fits model for specified lambda.
%Inputs
%   M is matrix X'*X, where X is data matrix (with weights)
%   R is vector of right hand side of SLAE
%`  lambda is specified value of lambda
%   b is original values of regression coefficients
%   pFunc is PQSQ potential function structure
%   epsilon is threshold to set coeeficient to zero.
%Returns
%   fitted regression coefficients.
    
    %Get size
    L = size(b,1);
    %Form muliindeces from 'previous' step
    qOld = repmat(-1,L,1);
    indOld = abs(b)<epsilon;
    %Main loop of fitting
    while true
        %Form new multiindeces
        d = abs(b);
        [~, q] = histc(d,pFunc.Intervals);
        ind = d<epsilon;
        %Stop if new multiindex is the same as previous
        if ~any(q(:)-qOld(:)) && ~any(ind-indOld)
            break;
        end
        qOld = q;
        indOld = ind;
        %Calculate diagonal of regulariser matrix
        d = lambda*pFunc.A(q);
        %Remove too small coefficients
        A = M;
        RR = R;
        A(ind,:) = 0;
        A(:,ind) = 0;
        d(ind) = 1;
        RR(ind) = 0;
        %Solve new SLAE
        b = (A+diag(d))\RR;
    end
end

function [A,B] = computeABcoefficients(intervals, potential_function_handle)
%computeABcoefficients calculates the coefficients a and b for
%quadratic fragments of potential function.
%   intervals is the 1-by-K matrix of intervals' boudaries without final
%       infinit boundary.
%   potential_function_handle is a handle of majorant function.

    %Get dimensions of intervals
    p = size(intervals,2);

    %Preallocate memory
    A = zeros(1,p);
    B = zeros(1,p);

    %Calculate value of function all boundaries
    pxk = potential_function_handle(intervals);
    sxk = intervals.^2;

    A(1:p-1) = (pxk(1:p-1)-pxk(2:p))./(sxk(1:p-1)-sxk(2:p));
    B(1:p-1) = (pxk(2:p).*sxk(1:p-1)-pxk(1:p-1).*sxk(2:p))./...
        (sxk(1:p-1)-sxk(2:p));
    B(p) = pxk(p);
end

function epsilon = searchEpsilon(M, R, toKill, re, rb)
%searchEpsilon search the value of black hole radius which kill toKill
%coefficients for the unregularized regression
%Inputs:
%   M is matrix of linear regression SLAE
%   R is right hand side of linear regression SLAE
%   toKill is number of coefficients to kill
%   re and rb are highest and lowest possible values of regression
%       coefficients.
    %Create copy of M and R
    ne = length(R)-toKill;
    nb = -toKill;
    %Number of consecutive left or right choices of radius
    epss = sqrt(eps()); %Difference which is less than epss is zero
    r = 0;
    l = 0;
    while true
        %Calculate candidate to check
        if r == 3
            rc = (nb*rb/2-ne*re)/(nb/2-ne);
            r = 0;
        elseif l == 3
            rc = (nb*rb-ne*re/2)/(nb-ne/2);
            l = 0;
        else
            rc = (nb*rb-ne*re)/(nb-ne);
        end
        %Calculate number of nonzero coefficients for this radius
        nc = numberOfKilled(M, R, rc)-toKill;
        if nc == 0
            epsilon = rc;
            return;
        elseif nc*nb>0
            rb = rc;
            nb = nc;
            r = r+1;
            l = 0;
        else
            re = rc;
            ne = nc;
            l = l+1;
            r = 0;
        end
        if re-rb<epss
            epsilon = re;
            return;
        end
    end
end

function killed = numberOfKilled(A, RR, r)
%numberOfKilled calculates number of killed coefficients of unregularised
%linear regression for SLAE A*b=RR and black hole radius r
    %Solve the first SLAE
    n = length(RR);
    b = A\RR;
    oldKilled = 0;
    while true
        %Form new multiindeces
        d = abs(b);
        ind = d<r;
        killed = sum(ind);
        %Stop if new multiindex is the same as previous
        if oldKilled==killed || killed == n
            break;
        end
        oldKilled = killed;
        %Remove too small coefficients
        A(ind,:) = 0;
        A(:,ind) = 0;
        RR(ind) = 0;
        
        %Solve new SLAE
        b = (A+diag(ind))\RR;
    end
end

function res = majorantF(y,majorants,nMajorants)
%majorantF processes all nMajorants functions specified in majorants and
%calculates weighted sum of reslts.
%Inputes
%   y is vector of input arguments for functions in majorants
%   majorants is nMajorants-by-2 cell matrix. Elements of the first column
%       are the function handles and elements in the second column are
%       weights of functions.
%   nMajorants is number of rows in matrix majorants to process
%Output
%   res is array of the same size as y

    %Create array for results with the same sizes as y
    res = zeros(size(y));
    
    %Process majorants function by function
    for k = 1:nMajorants
        f = majorants{k,1};
        res = res + majorants{k,2}*f(y);
    end
end

function intervals = optimalIntervals( trimming, num, func )
%optimalIntervals searchs the num intervals with optimal borders (the same
%maximum difference between marginal function and PQSQ approximation of it)
%   
%Input features
%   Itimming is the trimming point (upper border of the last interval)
%   num is required number of intervals
%   func is the handle of majorant function.
%
%Output feature
%   set of optimal intervals

    %Prepare the first two points for method of false position
    %Test accurauracy 0.1*trimming and then decide
    accBeg = 0.1*trimming;
    interv = intervalsByAccuracy( trimming, accBeg, func );
    if length(interv)-1>num
        %Accuracy is too small
        accEnd = accBeg;
        while length(interv)-1>num
            accBeg = accEnd;
            accEnd = accEnd*2;
            interv = intervalsByAccuracy( trimming, accEnd, func );
        end
        intervals = interv;
    else
        %Accuracy is too rough
        while length(interv)-1<=num
            accEnd = accBeg;
            intervals = interv;
            accBeg = accBeg/2;
            interv = intervalsByAccuracy( trimming, accBeg, func );
        end
    end
    %Search appropriate accuracy
    while accEnd-accBeg>0.0005*(accEnd+accBeg)
        acc = (accEnd+accBeg)/2;
        interv = intervalsByAccuracy( trimming, acc, func );
        if length(interv)-1>num
            accBeg = acc;
        else
            accEnd = acc;
            intervals = interv;
        end        
    end
end

function intervals = intervalsByAccuracy( trimming, accuracy, func )
%intervalsByAccuracy calculates intervals for specified trimming point,
%accuracy and majorant function.
%   trimming is the point of trimming and the uper value of the one before
%       last interval
%   accuracy is the required accuracy of function func approximation by
%       PQSQ potential.
%   func is function to be approximated.

    %Create empty array of intervals with the first value zero
    intervals = zeros(1,100);
    %Initiate some variables for search
    fend = func(trimming);  %Value of function at the end
    epss = sqrt(eps(accuracy)); %Difference which is less than epss is zero
    p = 2;   %Number of currently searched interval
    while true
        %Loop of number of intervals search
        rm1 = intervals(p-1);
        fm1 = func(rm1);
        %Start from the remainder of interval
        re = trimming;
        
        %Calculate coefficients for this interval
        a = (fm1 - fend)/(rm1^2-re^2);
        b = fend - a*re^2;
        
        %Calculate accuracy
        %accEnd = accuracyCalculator(rm1, re, a, b, func, 0.001);
        accEnd = fminbnd(@(x)a*x^2+b-func(x), rm1, re);
        accEnd = func(accEnd) - a*accEnd^2-b;
        
        %Check for the stop condition
        if accEnd<accuracy
%            disp(accEnd);
            intervals(p) = trimming;
            break;
        end
        
        %Initialise auhiliary variables
        accBeg = -accuracy; % there is no error for interval with zero length
        rb = rm1;
        accEnd = accEnd - accuracy;
        r = 0;
        l = 0;
        %Loop of border search
        while re-rb>accuracy/100
            %Candidate calculation
            if r == 3
                rc = (accEnd*rb/2-accBeg*re)/(accEnd/2-accBeg);
                r = 0;
            elseif l == 3
                rc = (accEnd*rb-accBeg*re/2)/(accEnd-accBeg/2);
                l = 0;
            else
                rc = (accEnd*rb-accBeg*re)/(accEnd-accBeg);
            end
            %Calculate accuracy. Calculate coefficients
            fc = func(rc);
            a = (fm1 - fc)/(rm1^2-rc^2);
            b = fc - a*rc^2;
            %Calculate accuracy. Point of maximal difference
            accCand = fminbnd(@(x)a*x^2+b-func(x), rm1, rc);
            %Calculate accuracy. Value of accuracy
            accCand = func(accCand) - a*accCand^2-b - accuracy;
            if abs(accCand)<epss
                rb = rc;
                accBeg = accCand;
                break
            elseif accCand<0
                rb = rc;
                accBeg = accCand;
                r = r+1;
                l = 0;
            else
                re = rc;
                accEnd = accCand;
                l = l+1;
                r = 0;
            end
        end
        %Add new interval into list
        if -accBeg<accEnd
            intervals(p) = rb;
        else
            intervals(p) = re;
        end
        p = p+1;
    end
    intervals = intervals(1:p);
end

function res = modelFitAndPredict(Xtrain,Ytrain,Xtest,Ytest, ...
        lambda, potFunc, epsilon)
    %Extract weights from training set
    trainWeights = Xtrain(:,1);
    Xtrain = Xtrain(:,2:end);
    trainWeights = trainWeights/sum(trainWeights);
    %Centralize and standardize Xtrain, Ytrain, Xtest and Ytest
    meanX = trainWeights'*Xtrain;
    Xs = bsxfun(@minus,Xtrain,meanX);
    meanY = trainWeights'*Ytrain;
    Ys = Ytrain-meanY;
    
    %Standardize
    SX = sqrt(trainWeights'*(Xs.^2));
    SX(SX==0) = 1;
    SY = sqrt(trainWeights'*(Ys.^2));
    Xs = bsxfun(@rdivide, Xs, SX);
    Ys = Ys/SY;
    %Weighted version of X
    XW = bsxfun(@times, Xs, trainWeights)';
    %The first matrix in SLAE
    M = XW*Xs;
    %The right hand side for SLAE
    R = XW*Ys;
    %Solve unregularized problem
    x = M\R;
    %Preallocate array for coefficients
    nLambda = length(lambda);
    B = zeros(size(Xtrain,2),nLambda);
    %Fit models
    for k=1:nLambda
        %Solve problem
        B(:,k) = fitModel(M, R, lambda(k), x, potFunc, epsilon);
    end
    %Extract weights from training set
    testWeights = Xtest(:,1);
    Xtest = Xtest(:,2:end);
    %standardize Xtest and centralize Ytest
    Xtest = bsxfun(@rdivide, bsxfun(@minus,Xtest,meanX), SX);
    Ytest = Ytest - meanY;
    %Calculate fitted value and renormalize it 
    yFitted = (Xtest * B) * SY;
    res = testWeights'*(bsxfun(@minus,Ytest,yFitted).^2) / sum(testWeights);
end