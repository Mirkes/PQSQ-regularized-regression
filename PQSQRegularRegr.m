function [B, FitInfo] = PQSQRegularRegr(X, Y, varargin)
%PQSQRegularRegr calculates PQSQ regularization of linear regression.
%Syntax:
%   B = PQSQRegularRegr(X, Y)
%   B = PQSQRegularRegr(X, Y, Name, Value)
%   [B, FitInfo] = PQSQRegularRegr(X, Y)
%   [B, FitInfo] = PQSQRegularRegr(X, Y, Name, Value)
%Inputs
%   X is numeric matrix with n rows and p columns. Each row represents one
%       observation, and each column represents one predictor (variable). 
%   Y is numeric vector of length n, where n is the number of rows of X.
%       Y(i) is the response to row i of X. 
%   Name, Value is one or more pairs of name and value. There are several
%       possible names with corresponding values: 
%       'Lambda' is vector of Lambda values. It will be returned in return
%           argument FitInfo in descending order. The default is to have
%           PQSQRegularRegr generate a sequence of lambda values, based on
%           'NumLambda' and 'LambdaRatio'. PQSQRegularRegr will generate a
%           sequence, based on the values in X and Y, such that the largest
%           LAMBDA value is just sufficient to produce all zero
%           coefficients B in standard lasso. You may supply a vector of
%           real, non-negative values of lambda for PQSQRegularRegr to use,
%           in place of its default sequence. If you supply a value for
%           'Lambda', 'NumLambda' and 'LambdaRatio' are ignored.
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
%       'Standardize' is indicator whether to scale X prior to fitting the
%           model sequence. This affects whether the regularization is
%           applied to the coefficients on the standardized scale or the
%           original scale. The results are always presented on the
%           original data scale. Possible values are true (any nonzero
%           number) and false (zero). Default is TRUE, do scale X. Note: X
%           and Y are always centred. 
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. Default: {}
%       'Weights' is vector of observation weights. It must be a vector of
%           non-negative values, of the same length as columns of X. At
%           least two values must be positive. Default (1/N)*ones(N,1).
%       'Epsilon' is positive value which specify minimal nonzero value of
%           regression coefficient. It means that attribute with absolute
%           value of regression coefficient which is less than 'epsilon' is
%           removed from regressions (coefficient becomes zero). There are
%           four possible ways to spesify epsilon:
%           positive value means that it is 'epsilon'.
%           zero means that maximal by absolute value OLS regression
%               coefficient divided by 32 is used as epsilon.  
%           negative value E means that lambda * (-E) is used as epsilon.
%           infinit (Nan or Inf) value mean that maximal by absolute value
%               OLS regression coefficient divided by 32 and bultiplied by
%               lambda is used as epsilon
%           Default is 0.
%       'Regular' is description of one regularisation term in equation
%           PQSQ regilarization regression. If this parameter is omitted
%           then 'lasso' is used. There are several special values of this
%           argument and general case for customization: 
%           Cell array C is a method to specify arbitrary term. Elements of
%               array have following meaning: 
%               C(1) is alpha for defined term, string 'elasticnet' or
%                  'elasticnet1'. Let us consider numeric C(1). 
%               C(2) is handle of majorant function, for example, L1 or L2.
%                   If this argument is omitted then L1 function is used. 
%               C(3) is array of intervals boundaries or positive integer
%                   number. If C(3) is empty, then it is interpreted as 5.
%                   If C(3) is array of intervals boundaries, then it is
%                   row vector The first element must be zero. All other
%                   elements must be sorted in ascending order. If C(3) is
%                   array of intervals then element C(4) is ignored. If
%                   C(3) is positive number then it is interpreted as
%                   number of intervals p. Let us M is specified trimming
%                   threshold. In this case intervals boundaries are
%                   calculated by following rule: the first element is
%                   zero, all other borders are calculated as r(i) =
%                   M*(i-1)^2/p^2. Default value is 5. 
%               C(4) is interpreted if C(3) is positive integer number
%                   only. In this case C(4) must be positive real number
%                   which interpreted as multiplier ? for trimming
%                   threshold definition: trimming threshold M is product
%                   of delta and maximal value of regression coefficients
%                   for OLS method. Default value is 1.
%           {'elasticnet', num} is imitation of elastic net with parameter
%               alpha equals num and without trimming. Num must be real
%               number between zero and one exclusively. It is equivalent
%               to consequence of two arrays: {num, @L1, 5, 2} and {1-num,
%               @L2, 1, 2}.
%           {'elasticnet1', num} is imitation of elastic net with parameter
%               alpha equals num and without trimming on ridge term and
%               with possible trimming in lasso term. Num must be real
%               number between zero and one exclusively. It is equivalent
%               to consequence of two arrays: {num, @L1, 5, 1} and {1-num,
%               @L2, 1, 2}.
%           {'elasticnet2', num} is imitation of elastic net with parameter
%               alpha equals num and without trimming on lasso term and
%               with possible trimming in ridge term. Num must be real
%               number between zero and one exclusively. It is equivalent
%               to consequence of two arrays: {num, @L1, 5, 2} and {1-num,
%               @L2, 1, 1}.
%           {'elasticnet3', num} is imitation of elastic net with parameter
%               alpha equals num and with possible trimming. Num must be
%               real number between zero and one exclusively. It is
%               equivalent to consequence of two arrays: {num, @L1, 5, 1}
%               and {1-num, @L2, 1, 1}.
%           'lasso' is simplest imitation of lasso without trimming. It is
%               equivalent to array {1, @L1, 5, 2} or {1, @L1, [], 2}. 
%           'lasso1' is simplest imitation of lasso with possible trimming.
%               It is equivalent to array {1, @L1, 5, 1} or {1, @L1}. 
%           'ridge' is simplest imitation of ridge regression without
%               trimming. It is equivalent to array {1, @L2, 1, 2}. 
%           'ridge1' is simplest imitation of ridge regression with
%               possible trimming. It is equivalent to array {1, @L2, 1, 1}. 
%Return values:
%   B is the fitted coefficients for each model. B will have dimension PxL,
%       where P = size(X,2) is the number of predictors, and 
%       L = length(lambda). 
%   FitInfo is a structure that contains information about the sequence of
%       model fits corresponding to the columns of B. STATS contains the
%       following fields:
%       'Intercept' is the intercept term for each model. Dimension is 1xL.
%       'Lambda' is the sequence of lambda penalties used, in ascending
%       order. Dimension is 1xL. 
%       'Regularization' is cell matrix. Each row of matrix corresponds to
%           regularization in form {C1, C2, C3} where C1 is column with
%           normalized weight alpha of regularization term, C2 is column
%           with function handle for majorant function and C3 is column
%           with set of intervals.
%       'DF' is the number of nonzero coefficients in B for each value of
%           lambda. Dimension is 1xL. 
%       'MSE' is the mean squared error of the fitted model for each value
%           of lambda. Otherwise, 'MSE' is the mean sum of squared
%           residuals obtained from the model with B and FitInfo.Intercept.
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. 
%       'Epsilon' is array of values epsilon which is used for each lambda.



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
        error(['Incorrect value for argument "X". It must be real valued',...
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
    standardize = true;
    predNames = {};
    weights = [];
    epsilon = 0;
    regular = {};
    
    %Search Name-Value pairs
    for i=1:2:length(varargin)
        if strcmpi(varargin{i},'Lambda')
            lambda = varargin{i+1};
        elseif strcmpi(varargin{i},'NumLambda')
            nLambda = varargin{i+1};
        elseif strcmpi(varargin{i},'LambdaRatio')
            lRatio = varargin{i+1};
        elseif strcmpi(varargin{i},'Standardize')
            standardize = varargin{i+1};
        elseif strcmpi(varargin{i},'PredictorNames')
            predNames = varargin{i+1};
        elseif strcmpi(varargin{i},'Weights')
            weights = varargin{i+1};
        elseif strcmpi(varargin{i},'Epsilon')
            epsilon = varargin{i+1};
        elseif strcmpi(varargin{i},'Regular')
            regular = {regular, varargin{i+1}};
        end
    end
    
    %Sanity-check of parameters and redefine all necessary values

    if ~isreal(epsilon) || ~isscalar(epsilon)
        error('Incorrect value for argument "Epsilon". It must be a real scalar');
    end
    
    %weights
    if isempty(weights)
        weights = ones(n,1);
    else
        %Weights must be a vector of nonnegative finite reals with at least two
        %values greater than zero and with number of elements equal to number
        %of rows in X. 
        if ~isreal(weights) || ~isfinite(weights) || sum(weights<0)>0 || sum(weights>0)<2 || numel(weights)~=n
            error(['Incorrect value for argument "Weights". It must be ',...
                'a vector of nonnegative finite reals with at least two',...
                'values greater than zero and with number of elements equal',...
                ' to number of rows in X.']);
        end
        weights = weights(:);
    end
    %Normalise
    weights = weights/sum(weights);
    
    %Standardize must be logical or numerical scalar
    if  ~isscalar(standardize) || ...
            (~islogical(standardize) && ~isnumeric(standardize))
        error(['Incorrect value for argument "Standardize". It must be ',...
            'a scalar of logical or numeric type']);
    end
    
    if ~islogical(standardize)
        standardize = standardize~=0;
    end
    
    %Centralize
    meanX = weights'*X;
    X = bsxfun(@minus,X,meanX);
    meanY = weights'*Y;
    Y = Y-meanY;
    
    %Standardize if necessary
    if standardize
        SX = sqrt(weights'*(X.^2));
        SX(SX==0) = 1;
        SY = sqrt(weights'*(Y.^2));
    else
        SX = ones(1,m);
        SY = 1;
    end
    
    %Standardize
    X = bsxfun(@rdivide, X, SX);
    Y = Y/SY;
    %Weighted version of X
    XW = bsxfun(@times, X, weights)';
    %The first matrix in SLAE
    M = XW*X;
    %The right hand side for SLAE
    R = XW*Y;

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
        %lasso
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
        %lambda = sort(lambda(:),1,'descend')';
        nLambda = size(lambda,2);
    end
    
    %PredictorNames is a cell array of strings with m elements
    if ~isempty(predNames) 
        if ~iscellstr(predNames) || length(predNames) ~= m
            error('Incorrect value for argument PredictorNames. It must be a cell array of strings with m elements.');
        else
            predNames = predNames(:)';
        end
    end

    %Preallocate data to return
    B = zeros(m,nLambda);
    FitInfo = struct();
    FitInfo.Intercept = zeros(1, nLambda);
    FitInfo.Lambda = lambda;
    FitInfo.Epsilon = zeros(1, nLambda);
    FitInfo.Regularization = {};
    FitInfo.DF = zeros(1, nLambda);
    FitInfo.MSE = zeros(1, nLambda);
    FitInfo.PredictorNames = predNames;

    %Solve problem without restiction to obtain information for coefficient
    %values.
    x = M\R;

    %Create all regularization terms
    if isempty(regular)
        regular = {{},'lasso'};
    end;
    
    xMax = max(abs(x));
    while ~isempty(regular)
        tmp = regular{2};
        %It is one term
        if isempty(tmp)
            continue;
        end
        if ischar(tmp)
            if strcmpi(tmp,'lasso')
                FitInfo.Regularization = [FitInfo.Regularization;...
                    {1, @L1, (xMax * 2) * ((0:5)/5).^2}];
            elseif strcmpi(tmp,'lasso1')
                FitInfo.Regularization = [FitInfo.Regularization;...
                    {1, @L1, xMax * ((0:5)/5).^2}];
            elseif strcmpi(tmp,'ridge')
                FitInfo.Regularization = [FitInfo.Regularization;...
                    {1, @L2, (xMax * 2) * (0:1)}];
            elseif strcmpi(tmp,'ridge1')
                FitInfo.Regularization = [FitInfo.Regularization;...
                    {1, @L2, xMax * (0:1)}];
            end
        elseif iscell(tmp)
            t = tmp{1};
            tt = tmp{2};
            if ischar(t)
                %tt (Alpha) must be positive real less than 1
                if ~isreal(tt) || tt<=0 || tt>=1
                    error('Incorrect value for "weight" in "elasticnetX" "Regular" term. It must be positive real not greater than 1');
                end
                if strcmpi(t,'elasticnet')
                    FitInfo.Regularization = [FitInfo.Regularization;...
                        {tt, @L1, (xMax * 2) * ((0:5)/5).^2};...
                        {1-tt, @L2, (xMax * 2) * (0:1)}];
                elseif strcmpi(t,'elasticnet1')
                    FitInfo.Regularization = [FitInfo.Regularization;...
                        {tt, @L1, xMax * ((0:5)/5).^2};...
                        {1-tt, @L2, (xMax * 2) * (0:1)}];
                elseif strcmpi(t,'elasticnet2')
                    FitInfo.Regularization = [FitInfo.Regularization;...
                        {tt, @L1, (xMax * 2) * ((0:5)/5).^2};...
                        {1-tt, @L2, xMax * (0:1)}];
                elseif strcmpi(t,'elasticnet3')
                    FitInfo.Regularization = [FitInfo.Regularization;...
                        {tt, @L1, xMax * ((0:5)/5).^2};...
                        {1-tt, @L2, xMax * (0:1)}];
                end
            elseif isreal(t)
                %tt must be function handler
                if ~isa(tt,'function_handle')
                    error('Incorrect value of "Regular" argument. It must be function handler');
                end
                if length(tmp)==2
                    FitInfo.Regularization = [FitInfo.Regularization;...
                        {t, tt, xMax * ((0:5)/5).^2}];
                else
                    ttt = tmp{3};
                    if isempty(ttt) || isscalar(ttt)
                        tttt = tmp{4};
                        if ~isreal(tttt) || tttt<=0
                            error(['Incorrect the fourth argument in cell'...
                                ' array in "Regular" argument. It must be '...
                                'non negative real number']);
                        end
                        if isempty(ttt)
                            ttt = 5;
                        else
                            if ttt<=0
                                error(['Incorrect the third argument in'...
                                    ' cell array in "Regular" argument.'...
                                    'It must be positive integer number'...
                                    ' or array of intervals']);
                            end
                            ttt = floor(ttt);
                        end
                        FitInfo.Regularization = [FitInfo.Regularization;...
                            {t, tt, (tttt * xMax) * ((0:ttt)/ttt).^2}];
                    elseif isvector(ttt) && isreal(ttt) && ttt(1)==0
                        FitInfo.Regularization = [FitInfo.Regularization;...
                            {t, tt, ttt}];
                    else
                        error(['Incorrect the third argument in'...
                            ' cell array in "Regular" argument.'...
                            'It must be positive integer number'...
                            ' or array of intervals']);
                    end
                end
                
            else
                error('Incorrect argument for "Regular" value.');
            end
            
        else
            error('Incorrect argument for "Regular" value.');
        end
        regular = regular{1};
    end
    
    %Rrecalculate weights of regularizators
    nReg = size(FitInfo.Regularization,1);
    t = 0;
    for k=1:nReg
        t = t + FitInfo.Regularization{k,1};
    end
    %coplete renormalization and form array of potential functions
    tt=struct();
    tt.Alpha = 1;
    tt.Intervals  = [1, 2];
    tt.A = tt.Intervals;
    tt.B = tt.Intervals;
    potFunc(nReg) = tt;
    for k=1:nReg
        tt.Alpha = FitInfo.Regularization{k,1}/t;
        FitInfo.Regularization{k,1} = tt.Alpha;
        ttt = FitInfo.Regularization{k,3};
        tt.Intervals  = [ttt, inf];
        [tt.A,tt.B] = ...
            computeABcoefficients(ttt, FitInfo.Regularization{k,2});
        potFunc(k) = tt;
    end
    %Prepare epsilon
    if epsilon == 0
        epsilon = xMax/32;
        %epsilon = ;
    elseif ~isfinite(epsilon)
        epsilon = -xMax/32;
    end
    
    %Main lambda loop
    for k=1:nLambda
        %Threshold to suppress coefficients
        if epsilon>0
            th = epsilon;
        else
            th = -epsilon*min([1,lambda(k)]);
        end
        
        %threshold is defined as middle of the smallest interval
        %th = potFunc(1).Intervals(2)/2;
        %threshold is defined as the smallest coefficient in simple linear
        %regression (such that for lambda=0 we would have all parameters
        %th = min(abs(x))*(1-0.1);
        
        %display(sprintf('epsilon = %f',th));
        %Store threshold
        FitInfo.Epsilon(k) = th;
        %Solve problem
        B(:,k) = fitModel(M, R, lambda(k), x, potFunc, th);
    end
    
    %Restore values of coefficients and intercepts
    %Calculate number of zero coefficients
    FitInfo.DF = sum(B~=0);
    %Calculate mean squared error
    FitInfo.MSE = weights'*((bsxfun(@minus,Y,bsxfun(@plus,FitInfo.Intercept,X*B))*SY).^2);
    %3. Denormalisiation of regression coefficietns
    B = bsxfun(@rdivide, B*SY, SX');
    %4. Recalculate intercepts
    FitInfo.Intercept = meanY - meanX*B;
end

function b = fitModel(M, R, lambda, b, pFunc, epsilon)
%fitModel fits model for specified lambda.
%Inputs
%   M is matrix X'*X, where X is data matrix (with weights)
%   R is vector of right hand side of SLAE
%`  lambda is specified value of lambda
%   b is original values of regression coefficients
%   pFunc is array of PQSQ potential function structures
%   epsilon is threshold to set coeeficient to zero.
%Returns
%   fitted regression coefficients.
    
    %Get size
    L = size(b,1);
    nR = numel(pFunc);
    %Form muliindeces from 'previous' step
    qOld = repmat(-1,L,nR);
    q = qOld;
    indOld = abs(b)<epsilon;
    %Main loop of fitting
    count = 0;
    while true
        %Form new multiindeces
        d = abs(b);
        for k=1:nR
            q(:,k) = discretize(d,pFunc(k).Intervals);
        end
        ind = d<epsilon;
        %Stop if new multiindex is the same as previous
        if ~any(q(:)-qOld(:)) && ~any(ind-indOld)
            break;
        end
        qOld = q;
        indOld = ind;
        %Calculate diagonal of regulariser matrix
        d = (lambda*pFunc(1).Alpha)*pFunc(1).A(q(:,1));
        for k=2:nR
            d = d + (lambda*pFunc(k).Alpha)*pFunc(k).A(q(:,k));
        end
        %Remove too small coefficients
        A = M;
        RR = R;
        A(ind,:) = 0;
        A(:,ind) = 0;
        d(ind) = 1;
        RR(ind) = 0;
        %Solve new SLAE
        b = (A+diag(d))\RR;
        count = count+1;
    end
    %display(sprintf('N of iterations %i',count));
end

function [A,B] = computeABcoefficients(intervals, potential_function_handle)
%PQSQR_computeABcoefficients calculates the coefficients a and b for
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