function [B, FitInfo] = lassoPQSQ(X, Y, varargin)
%lassoPQSQ calculates PQSQ approximation for lasso regression.
%Syntax
%   B = lassoPQSQ(X, Y)
%   B = lassoPQSQ(X, Y, Name, Value)
%   [B, FitInfo] = lassoPQSQ(X, Y)
%   [B, FitInfo] = lassoPQSQ(X, Y, Name, Value)
%
%Inputs
%   X is numeric matrix with n rows and p columns. Each row represents one
%       observation, and each column represents one predictor (variable). 
%   Y is numeric vector of length n, where n is the number of rows of X.
%       Y(i) is the response to row i of X.
%   Name, Value is one or more pairs of name and value. There are several
%       possible names with corresponding values:
%       'Intervals', intervals serves to specify user defined intervals.
%           intervals is row vector The first element must be zero. By
%           default is created by usage of 'number_of_intervals' and
%           'intshrinkage'. Maximal value M is maximum of absolute value of
%           coefficients for OLS without any penalties multiplied by
%           'intshrinkage'. All other boreders are calcualted as r(i) =
%           M*i^2/p^2, where p is 'number_of_intervals'. 
%       'Number_of_intervals' specifies the number of intervals to
%           automatic interval calculation. Default value is 5. 
%       'intshrinkage', delta serves to specify delta which is coefficient
%           for intervals shrinkage (see argument delta in
%           defineIntervals). Default value is 1 (no shrinkage).
%       'Trimming' is value of trimming. If this value is not specified
%           then maximal absolute value of regression coefficients wothout
%           restriction is used. For standardized data theoretical maximum
%           of regression coefficient is 1. It means that for standardized
%           data 'trimming' is reasonably restricted by 1. Default empty
%       'Epsilon' is positive value which spesify minimal nonzero value of
%           regression coefficient. It means that attribute with absolute
%           value of regression coefficient which is less than 'epsilon' is
%           removed from regressions (coefficient becomes zero). There are
%           three possible ways to spesify epsilon:
%               positive value means that it is 'epsilon'.
%               zero means that r(1)/2 is used as epsilon. r(1) is right
%                   border of the first interval.
%               negative value means that lambda*r(1)/2 is used as epsilon.
%           Default 0.
%       'potential' is majorant function for PQSQ. By default it is L1.
%       'Alpha' is elastic net mixing value, or the relative balance
%           between L2 and L1 penalty (default 1, range (0,1]). Alpha=1 ==>
%           lasso, otherwise elastic net. Alpha near zero ==> nearly ridge
%           regression. 
%       'Lambda' is vector of Lambda values. Will be returned in return
%           argument FitInfo in descending order. The default is to have
%           lassoPQSQ generate a sequence of lambda values, based on
%           'NumLambda' and 'LambdaRatio'. lassoPQSQ will generate a
%           sequence, based on the values in X and Y, such that the largest
%           LAMBDA value is just sufficient to produce all zero
%           coefficients B in standard lasso. You may supply a vector of
%           real, non-negative values of  lambda for lassoPQSQ to use, in
%           place of its default sequence. If you supply a value for
%           'Lambda', 'NumLambda' and  'LambdaRatio' are ignored. 
%       'NumLambda' is the number of lambda values to use, if the parameter
%           'Lambda' is not supplied (default 100). Ignored if 'Lambda' is
%           supplied.  lassoPQSQ may return fewer fits than specified by
%           'NumLambda' if the residual error of the fits drops below a
%           threshold percentage of the variance of Y.
%       'LambdaRatio' is ratio between the minimum value and maximum value
%           of lambda to generate, if the  parameter "Lambda" is not
%           supplied.  Legal range is [0,1). Default is 0.0001. If
%           'LambdaRatio' is zero, lassoPQSQ will generate its default
%           sequence of lambda values but replace the smallest value in
%           this sequence with the value zero. 'LambdaRatio' is ignored if
%           'Lambda' is supplied. 
%       'Standardize' is indicator whether to scale X prior to fitting the
%           model sequence. This affects whether the regularization is
%           applied to the coefficients on the standardized scale or the
%           original scale. The results are always presented on the
%           original data scale. possible values true (any nonzero number)
%           and false (zero). Default is TRUE, do scale X. 
%                      Note: X and Y are always centered.
%       'PredictorNames' is a cell array of names for the predictor
%           variables, in the order in which they appear in X. Default: {}
%       'Weights' is vector of observation weights.  Must be a vector of
%           non-negative values, of the same length as columns of X.  At
%           least two values must be positive. (default (1/N)*ones(N,1)). 
%
%Return values:
%   B is the fitted coefficients for each model. B will have dimension PxL,
%       where P = size(X,2) is the number of predictors, and 
%       L =  length(lambda). 
%   FitInfo is a structure that contains information about the sequence of
%       model fits corresponding to the columns of B. STATS contains the
%       following fields: 
%       'Intercept' is the intercept term for each model. Dimension 1xL.
%       'Lambda' is the sequence of lambda penalties used, in ascending
%           order. Dimension 1xL.
%       'Alpha' is the elastic net mixing value that is used.
%       'DF' is the number of nonzero coefficients in B for each value of
%           lambda. Dimension 1xL. 
%       'MSE' is the mean squared error of the fitted model for each value
%       of lambda. Otherwise, 'MSE' is the mean sum of squared residuals
%       obtained from the model with B and FitInfo.Intercept.
%       'PredictorNames' is a cell array of names for the predictor
%       variables, in the order in which they appear in X.



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
    intervals = [];
    numOfInt = 5;
    delta = 1;
    func = @L1;
    alpha = 1;
    lambda = [];
    nLambda = 100;
    lRatio = 0.00001;
    standardize = true;
    predNames = {};
    weights = [];
    eps = 0;
    trimming = [];
    
    %Search Name-Value pairs
    for i=1:2:length(varargin)
        if strcmpi(varargin{i},'intervals')
            intervals = varargin{i+1};
        elseif strcmpi(varargin{i},'number_of_intervals')
            numOfInt = varargin{i+1};
        elseif strcmpi(varargin{i},'intshrinkage')
            delta = varargin{i+1};
        elseif strcmpi(varargin{i},'potential')
            func = varargin{i+1};
        elseif strcmpi(varargin{i},'Alpha')
            alpha = varargin{i+1};
        elseif strcmpi(varargin{i},'Lambda')
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
        elseif strcmpi(varargin{i},'Trimming')
            trimming = varargin{i+1};
        elseif strcmpi(varargin{i},'Epsilon')
            eps = varargin{i+1};
        end
    end
    
    %Sanity-check of parameters and redefine all necessary values

    if ~isempty(trimming) && (~isreal(trimming) || trimming<0)
        error(['Incorrect value for argument "Trimming". It must be a positive scalar']);
    end
        
    if ~isreal(eps) || ~isscalar(eps)
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
    
    %Func must be function handler
    if ~isa(func,'function_handle')
        error('Incorrect value in "potential" argument. It must be function handler');
    end
    
    %Alpha must be positive real not greater than 1
    if ~isreal(alpha) || alpha<=0 || alpha > 1
        error('Incorrect value for "Alpha". It must be positive real not greater than 1');
    end

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
        lambdaMax = max(abs(R))/alpha*10;
        
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
        lambda = sort(lambda(:),1,'descend');
        nLambda = size(lambda,1);
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
    FitInfo.Alpha = alpha;
    FitInfo.DF = zeros(1, nLambda);
    FitInfo.MSE = zeros(1, nLambda);
    FitInfo.PredictorNames = predNames;

    %Solve problem without restiction to obtain information for coefficient
    %values.
    x = M\R;
    
    if isempty(intervals)
        %Function has to create intervals by automatic way
        %numOfInt must be positive integer scalar
        if ~isreal(numOfInt) || ~isfinite(numOfInt) || numOfInt < 1
            error(['Incorrect value of "number_of_intervals" argument' ...
                'It must be positive integer scalar']);
        else
            numOfInt = floor(numOfInt);
        end
        %delta has to be positive real scalar
        if ~isreal(delta) || ~isfinite(delta) || delta < 0
            error(['Incorrect value of "intshrinkage" argument' ...
                'It must be positive real scalar']);
        end
        if isempty(trimming)
            trimming = max(abs(x));
        end
            
        pFunc = definePotentialFunction(trimming, numOfInt, func, delta);
    else
        %intervals must contains non nerative values in ascending order.
        %The first value must be zero.
        if intervals(1)~=0 || ~all(isfinite(intervals)) ...
                || any((intervals(2:end)-intervals(1:end-1))<=0)
            error(['Incorrect values in argument intervals: intervals must'...
                ' contains non negative values in ascending order.'...
                ' The first value must be zero.']);
        end
        pFunc.intervals = [intervals, Inf(size(X,2),1)];
        [pFunc.A,pFunc.B] = ...
            computeABcoefficients(intervals, func);
    end
    
    %Main lambda loop
    for k=1:nLambda
        %Threshold to suppress coefficients
        if eps==0
            th = pFunc.intervals(2)/2;
        elseif eps>0
            th = eps;
        else
            th = pFunc.intervals(2)/2*min([1,lambda(k)]);
        end
        %Solve problem
        B(:,k) = fitModel(M, R, lambda(k), alpha, x, pFunc, th);
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

function b = fitModel(M, R, lambda, alpha, b, pFunc, eps)
%fitModel fits model for specified lambda.
%Inputs
%   M is matrix X'*X, where X is data matrix (with weights)
%   R is vector of right hand side of SLAE
%`  lambda is specified value of lambda
%   alpha is elastic net mixing value
%   b is original values of regression coefficients
%   pFunc is PQSQ potential function structure
%Returns
%   fitted regression coefficients.
    
    %Get size
    L = size(b,1);
    %Form muliindeces from 'previous' step
    qOld = repmat(-1,L,1);
    indOld = abs(b)<eps;

    %Main loop of fitting
    while true
        %Form new multiindeces
        q = discretize(abs(b),pFunc.intervals);
        ind = abs(b)<eps;
        %Stop if new multiindex is the same as previous
        if ~any(q-qOld) && ~any(ind-indOld)
            break;
        end
        qOld = q;
        indOld = ind;
        %Calculate diagonal of regulariser matrix
        d = lambda*(alpha*(pFunc.A(q)-1)+1);
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

function potentialFunction = definePotentialFunction( x, number_of_intervals, potential_function_handle, delta )
%definePotentialFunction defines "uniform in square" intervals for trimming
%threshold x and specified number_of_intervals.
%   x is upper boundary of the interval last but one.
%   number_of_intervals is required number of intervals.
%   potential_function_handle is function handler for coefficients
%       calculation.
%   delta is coefficient of shrinkage which is greater than 0 ang not
%       greater than 1.
%Output argument potentialFunction is structure with three fields:
%   intervals is matrix m-by-number_of_intervals. Each row contains
%       number_of_intervals values of thresholds for intervals and one
%       additional value Inf
%   A and B are the m-by-number_of_intervals matrices with quadratic
%       functions coefficients

    if nargin<4 
        delta = 1;
    end
    
    p=number_of_intervals-1;
    
    %intervals is the product of row and maximal coefficient multiplied by delta:
    intervals = (x * delta) * ((0:p)/p).^2;
    
    potentialFunction.intervals = [intervals, Inf(1)];
    potentialFunction.sqint = potentialFunction.intervals.^2;
    [potentialFunction.A,potentialFunction.B] = ...
        computeABcoefficients(intervals, potential_function_handle);
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