# PQSQ-regularized-regression
The method of generalized regression regularization through PQSQ technique is presented. This approach includes lasso, elastic net and ridge regression as partial case.

Current MatLab implementation includes several files:

L1 is majorant function for L_1 norm.

L1_5 is majorant function for L_1 + L_2 norm.

L2 is majorant function for L_2 norm.

LLog is natural logarithm majorant function.

LSqrt is square root or L_0.5 quasi norm majorant function.

PQSQRegularRegr calculates PQSQ regularization of linear regression. See [description](https://github.com/Mirkes/PQSQ-regularized-regression#Lab1)

PQSQRegularRegrPlot plots coefficient values or goodness of fit of PQSQ regularised regression fits. See [description]( https://github.com/Mirkes/PQSQ-regularized-regression #Lab2)
fastRegularisedRegression perform feature selection for regression which is regularized by Tikhonov regularization (ridge regression) with automated selection of the optimal value of regularization parameter Alpha. See [description]( https://github.com/Mirkes/PQSQ-regularized-regression #Lab3)

<a name="Lab1">Description of PQSQ regularised regression</a>
<pre>
%PQSQRegularRegr calculates PQSQ regularization of linear regression.
Syntax:
   B = PQSQRegularRegr(X, Y)
   B = PQSQRegularRegr(X, Y, Name, Value)
   [B, FitInfo] = PQSQRegularRegr(X, Y)
   [B, FitInfo] = PQSQRegularRegr(X, Y, Name, Value)

Examples:
       %(1) Run the lasso and PQSQ lasso simulation on data obtained from
       the 1985 %Auto Imports Database  of the UCI repository.  
       %http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

       load imports-85;

       %Extract Price as the response variable and extract non-categorical
       %variables related to auto construction and performance
       %
       X = X(~any(isnan(X(:,1:16)),2),:);
       Y = X(:,16);
       Y = log(Y);
       X = X(:,3:15);
       predictorNames = {'wheel-base' 'length' 'width' 'height' ...
           'curb-weight' 'engine-size' 'bore' 'stroke' 'compression-ratio' ...
           'horsepower' 'peak-rpm' 'city-mpg' 'highway-mpg'};
 
       %Compute the default sequence of lasso fits. Fix time.
       tic;
       [B,S] = lasso(X,Y,'CV',10,'PredictorNames',predictorNames);
       disp(['Time for lasso ',num2str(toc),' seconds']);
 
       %Compute PQSQ simulation of lasso without trimming
       tic;
       [BP,SP] = PQSQRegularRegr(X,Y,'CV',10,...
           'PredictorNames',predictorNames);
       disp(['Time for PQSQ simulation of lasso without trimming ',...
            num2str(toc),' seconds']);
 
       %Compute PQSQ simulation of lasso with trimming
       tic;
       [BT,ST] = PQSQRegularRegr(X,Y,'CV',10,...
           'TrimmingPoint',-1,'PredictorNames',predictorNames);
       disp(['Time for PQSQ simulation of lasso with trimming ',...
            num2str(toc),' seconds']);
 
 
       %Display a trace plot of the fits.
       %Lasso
       lassoPlot(B,S);
       %PQSQ without trimming
       PQSQRegularRegrPlot(BP,SP);
       %PQSQ with trimming
       PQSQRegularRegrPlot(BT,ST);
 
       %Display the sequence of cross-validated predictive MSEs.
       %Lasso
       lassoPlot(B,S,'PlotType','CV');
       %PQSQ without trimming
       PQSQRegularRegrPlot(BP,SP,'PlotType','CV');
       %PQSQ with trimming
       PQSQRegularRegrPlot(BT,ST,'PlotType','CV');
        
       %Look at the kind of fit information returned by 
       %lasso.
       disp('lasso');
       S
       %PQSQ without trimming
       disp('PQSQ without trimming');
       SP
       %PQSQ with trimming
       disp('PQSQ with trimming');
       ST
 
       %What variables are in the model corresponding to minimum 
       %cross-validated MSE, and in the sparsest model within one 
       %standard error of that minimum. 
       disp('lasso');
       S.PredictorNames(B(:,S.IndexMinMSE)~=0)
       S.PredictorNames(B(:,S.Index1SE)~=0)
       disp('PQSQRegularRegr without trimming');
       SP.PredictorNames(BP(:,SP.IndexMinMSE)~=0)
       SP.PredictorNames(BP(:,SP.Index1SE)~=0)
       disp('PQSQRegularRegr with trimming');
       ST.PredictorNames(BT(:,ST.IndexMinMSE)~=0)
       ST.PredictorNames(BT(:,ST.Index1SE)~=0)
 
       %Fit the sparse model and examine residuals. 
       %Lasso.
       Xplus = [ones(size(X,1),1) X];
       fitSparse = Xplus * [S.Intercept(S.Index1SE); B(:,S.Index1SE)];
       a = 'lasso';
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
       figure
       plot(fitSparse,Y-fitSparse,'o');
       %PQSQRegularRegr without trimming.
       fitSparse = Xplus * [SP.Intercept(SP.Index1SE); BP(:,SP.Index1SE)];
       a = 'PQSQRegularRegr without trimming';
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
       figure
       plot(fitSparse,Y-fitSparse,'o')
       %PQSQRegularRegr with trimming.
       fitSparse = Xplus * [ST.Intercept(ST.Index1SE); BT(:,ST.Index1SE)];
       a = 'PQSQRegularRegr with trimming';
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitSparse,Y-fitSparse))]);
       figure
       plot(fitSparse,Y-fitSparse,'o')
 
       %Consider a slightly richer model. A model with 7 variables may be a 
       %reasonable alternative.  Find the index for a corresponding fit.
       a = 'lasso';
       df7index = find(S.DF==7,1);
       fitDF7 = Xplus * [S.Intercept(df7index); B(:,df7index)];
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
       figure
       plot(fitDF7,Y-fitDF7,'o')         
       a = 'PQSQRegularRegr without trimming';
       df7index = find(SP.DF==7,1);
       fitDF7 = Xplus * [SP.Intercept(df7index); BP(:,df7index)];
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
       figure
       plot(fitDF7,Y-fitDF7,'o')         
       a = 'PQSQRegularRegr with trimming';
       df7index = find(ST.DF==7,1);
       fitDF7 = Xplus * [ST.Intercept(df7index); BT(:,df7index)];
       disp([a, '. Correlation coefficient of fitted values ',...
           'vs. residuals ',num2str(corr(fitDF7,Y-fitDF7))]);
       figure
       plot(fitDF7,Y-fitDF7,'o')         
         
       %(2) Run lasso and PQSQ simulation of lasso on some random data
       %with 250 predictors
       %Data generation
       n = 1000; p = 250;
       X = randn(n,p);
       beta = randn(p,1); beta0 = randn;
       Y = beta0 + X*beta + randn(n,1);
       lambda = 0:.01:.5;
       %Run lasso
       tic;
       [B,S] = lasso(X,Y,'Lambda',lambda);
       disp(['Time for lasso ',num2str(toc),' seconds']);
       lassoPlot(B,S);
       %Run PQSQ simulation of lasso with the same lambdas and without
       %trimming
       tic;
       [BP,SP] = PQSQRegularRegr(X,Y,'Lambda',lambda);
       disp(['Time for PQSQ simulation of lasso without trimming ',...
            num2str(toc),' seconds']);
       PQSQRegularRegrPlot(BP,SP);

       %Compute PQSQ simulation of lasso with trimming and the same lambdas
       tic;
       [BT,ST] = PQSQRegularRegr(X,Y,'Lambda',lambda,'TrimmingPoint',-1);
       disp(['Time for PQSQ simulation of lasso with trimming ',...
            num2str(toc),' seconds']);
       PQSQRegularRegrPlot(BT,ST);

       %compare against OLS
       %Lasso
       figure
       bls = [ones(size(X,1),1) X] \ Y;
       plot(bls,[S.Intercept; B],'.');
       %PQSQ without trimming
       figure
       plot(bls,[SP.Intercept; BP],'.');
       %PQSQ with trimming
       figure
       plot(bls,[ST.Intercept; BT],'.');



Inputs
   X is numeric matrix with n rows and p columns. Each row represents one
       observation, and each column represents one predictor (variable). 
   Y is numeric vector of length n, where n is the number of rows of X.
       Y(i) is the response to row i of X.

   IMPORTANT NOTES: data matrix X and response vector Y are standardized
       anyway.

   Name, Value is one or more pairs of name and value. There are several
       possible names with corresponding values: 
       'Lambda' is vector of Lambda values. It will be returned in return
           argument FitInfo in ascending order. The default is to have
           PQSQRegularRegr generate a sequence of lambda values, based on
           'NumLambda' and 'LambdaRatio'. PQSQRegularRegr will generate a
           sequence, based on the values in X and Y, such that the largest
           LAMBDA value is just sufficient to produce all zero
           coefficients B in standard lasso. You may supply a vector of
           real, non-negative values of lambda for PQSQRegularRegr to use,
           in place of its default sequence. If you supply a value for
           'Lambda' then values of 'NumLambda' and 'LambdaRatio' are
           ignored. 
       'NumLambda' is the number of lambda values to use, if the parameter
           'Lambda' is not supplied (default 100). It is ignored if
           'Lambda' is supplied. PQSQRegularRegr may return fewer fits
           than specified by 'NumLambda' if the residual error of the fits
           drops below a threshold percentage of the variance of Y.
       'LambdaRatio' is ratio between the minimum value and maximum value
           of lambda to generate, if the  parameter "Lambda" is not
           supplied. Legal range is [0,1). Default is 0.0001. If
           'LambdaRatio' is zero, PQSQRegularRegr will generate its
           default sequence of lambda values but replace the smallest
           value in this sequence with the value zero. 'LambdaRatio' is
           ignored if 'Lambda' is supplied.
       'PredictorNames' is a cell array of names for the predictor
           variables, in the order in which they appear in X. Default: {}
       'Weights' is vector of observation weights. It must be a vector of
           non-negative values, of the same length as columns of X. At
           least two values must be positive. Default (1/N)*ones(N,1).
       'Epsilon' is real number between 0 and 1 which specify the radius
           of black hole r. The meaning of this argument is the fraction of
           coefficients which has to be put to zero for unregularized
           problem. Procedure of coefficients putting to zero is:
           1. calculate regression coefficients for unregularized problem.
           2. identify all coefficients which less than r.
           3. if there is no small coefficients then stop.
           4. remove predictors which are corresponded to small
               coefficients and go to step 1.
           If number of removed coefficients is less than number of
           predictors multiplied by Epsilon, then increase r.
           If number of removed coefficients is greater than number of
           predictors multiplied by Epsilon, then decrease r.
           Default is 0.01 (1% of coefficients has to be put to zero).
       'Majorant' is handle of majorant function or cell array with two
           elements, first of which is handle of majorant function and the
           second is weight of this function in linear combination.
           Several pairs of 'Majorant', Values can be specified. If
           argument is handle of majorant function then weight is equal to
           one. Weight can be changed by argument 'MajorantWeight' which
           is appear after current 'Majorant' argument but before next one. 
           There are several special values of 'Majorant':
           {'elasticnet', num} is imitation of
               elastic net with parameter alpha equals num. Num must be
               real number between zero and one inclusively. It is
               equivalent to consequence of two pairs of majorant
               parameters:  
               'Majorant', {@L1, num}, Majorant', {@L2,num1}
               where num1 = 1-num or to four pairs:
               'Majorant', @L1, 'MajorantWeight', num, 'Majorant', @L2,
               'MajorantWeight', num1 
           'lasso' is simplest imitation of lasso. It is
               equivalent to pair 'Majorant', @L1. 
           'ridge' is simplest imitation of ridge regression. It is
               equivalent to pair 'Majorant', @L2.
       'MajorantWeight' is new weight for the last specified majorant. It
           must be non-negative value.
       'Intervals' is specification of intervals for PQSQ approximation of
           majorant function or linear combination of majorant functions.
           Intervals must be array of intervals boundaries. In this case
           it has to be row vector. The first element must be zero. All
           other elements must be sorted in ascending order.
           Default is empty.
           If you supply a value for 'Intervals' then values for
           'IntervalsByQudratic', 'IntervalsByOptimal',
           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
           'IntervalsByAccuracyMajorant' are ignored
       'IntervalsByQudratic' specifies the required number of intervals.
           The value N must be positive integer. Intervals boundaries are
           calculated as TrimmingPoint*((0:N)/N).^2;
           This option is not supplied by default.
           This option is ignored if 'Intervals' is supplied.
           This option is also ignored if 'IntervalsByOptimal',
           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
           'IntervalsByAccuracyMajorant' is supplied later than it. 
       'IntervalsByOptimal' specifies the positive integer N which is used
           for calculations of the optimal positions of N intervals
           boundaries. The most computationally expansive way of intervals
           specification. The procedure of boundaries search optimise the
           maximum of difference between majorant function and PQSQ
           approximation of it.
           This option is not supplied by default.
           This option is ignored if 'Intervals' is supplied.
           This option is also ignored if 'IntervalsByQudratic',
           'IntervalsByAccuracy', 'IntervalsByAccuracyTrimming' or
           'IntervalsByAccuracyMajorant' is supplied later than it. 
       'IntervalsByAccuracy' is a positive number which specify the
           required accuracy of PQSQ approximation of majorant function.
           In this case all intervals exclude last are created to have
           specified value of maximum of difference between majorant
           function and PQSQ approximation of it. The last interval can
           has maximal difference that is less then required.
           This option is not supplied by default.
           This option is ignored if 'Intervals' is supplied.
           This option is also ignored if 'IntervalsByOptimal',
           'IntervalsByQudratic' or 'IntervalsByAccuracyMajorant'
           is supplied later than it. 
       'IntervalsByAccuracyTrimming' is a positive number which define
           the required accuracy of PQSQ approximation of majorant
           function as product of specified value and TrimmingPoint.
           In this case all intervals exclude last are created to have
           specified value of maximum of difference between majorant
           function and PQSQ approximation of it. The last interval can
           has maximal difference that is less then required.
           This option is not supplied by default.
           This option is ignored if 'Intervals' is supplied.
           This option is also ignored if 'IntervalsByOptimal',
           'IntervalsByQudratic', 'IntervalsByAccuracy', or
           'IntervalsByAccuracyMajorant' is supplied later than it. 
       'IntervalsByAccuracyMajorant' is a positive number which define
           the required accuracy of PQSQ approximation of majorant
           function as product of specified value and value of majorant
           function of TrimmingPoint. In this case all intervals exclude
           last are created to have specified value of maximum of
           difference between majorant function and PQSQ approximation of
           it. The last interval can has maximal difference that is less
           then required.
           Default value is 0.02.
           This option is ignored if 'Intervals' is supplied.
           This option is also ignored if 'IntervalsByQudratic',
           'IntervalsByOptimal', 'IntervalsByAccuracy' or
           'IntervalsByAccuracyTrimming' is supplied later than it. 
       'TrimmingPoint' is a real number. If it is positive number then it
           is the trimming point. If it is negative value then it is minus
           multiplier for default trimming point value.
           By default it is maximum of absolute value of unregularised
           regression coefficient. Such value of trimming point assumes
           the trimming of regularisation term. To avoid trimming it is
           necessary to specify another value or use pair 
           'TrimmingPoint', -2.
       'CV' If present, indicates the method used to compute MSE. When
           'CV' is a positive integer K, LASSO uses K-fold cross-
           validation.  Set 'CV' to a cross-validation partition, created
           using CVPARTITION, to use other forms of cross-validation. You
           cannot use a 'Leaveout' partition with PQSQRegularRegr. When
           'CV' is 'resubstitution', PQSQRegularRegr uses X and Y both to
           fit the model and to estimate the mean squared errors, without
           cross-validation.   
           The default is 'resubstitution'.
       'MCReps' is a positive integer indicating the number of Monte-Carlo
           repetitions for cross-validation.  The default value is 1. If
           'CV' is a cvpartition of type 'holdout', then 'MCReps' must be
           greater than one. Otherwise it is ignored.
       'Options' is a structure that contains options specifying whether
           to conduct cross-validation evaluations in parallel, and
           options specifying how to use random numbers when computing
           cross validation partitions. This argument can be created by a
           call to STATSET. CROSSVAL uses the following fields: 
               'UseParallel'
               'UseSubstreams'
               'Streams'
           For information on these fields see PARALLELSTATS.
           NOTE: If supplied, 'Streams' must be of length one.
Return values:
   B is the fitted coefficients for each model. B will have dimension PxL,
       where P = size(X,2) is the number of predictors, and 
       L = length(lambda). 
   FitInfo is a structure that contains information about the sequence of
       model fits corresponding to the columns of B. STATS contains the
       following fields:
       'Intercept' is the intercept term for each model. Dimension is 1xL.
       'Lambda' is the sequence of lambda penalties used, in ascending
           order. Dimension is 1xL. 
       'Regularization' is cell matrix. Each row of matrix corresponds to
           regularization in form {func, weight} where weight is column
           with normalized weights alpha of regularization terms, func is column
           with function handles for majorant function.
       'Intervals' is row array of intervals without last (Inf) value.
       'BlackHoleRadius' is radius of black hole.
       'DF' is the number of nonzero coefficients in B for each value of
           lambda. Dimension is 1xL. 
       'MSE' is the mean squared error of the fitted model for each value
           of lambda. If cross-validation was performed, the values for
           'MSE' represent Mean Prediction Squared Error for each value of
           lambda, as calculated by cross-validation. Otherwise, 'MSE' is
           the mean sum of squared residuals obtained from the model with
           B and FitInfo.Intercept. 
       'PredictorNames' is a cell array of names for the predictor
           variables, in the order in which they appear in X. 

If cross-validation was performed, FitInfo also includes the following
fields: 
       'SE' is a row vector which contains the standard error of MSE for
           each lambda, as calculated during cross-validation. 
           Dimension 1xL.
       'LambdaMinMSE' is the lambda value with minimum MSE. Scalar.
       'Lambda1SE' is the largest lambda such that MSE is within one
           standard error of the minimum. Scalar. 
       'IndexMinMSE' is the index of Lambda with value LambdaMinMSE.
       'Index1SE' is the index of Lambda with value Lambda1SE.
</pre>

<a name="Lab2">Description of function to plot PQSQ regularised regression fits</a>
<pre>
%PQSQRegularRegrPlot plots coefficient values or goodness of fit of PQSQ
   regularised regression fits.

   [AXH, FIGH] = PQSQRegularRegrPlot(B, PLOTDATA) creates a Trace Plot
       showing the sequence of coefficient values B produced by a
       PQSQRegularRegr. B is a P by nLambda matrix of coefficients, with
       each column of B representing a set of coefficients estimated by
       using a single penalty term Lambda.  AXH is an axes handle that
       gives access to the axes used to plot the coefficient values B.
       FIGH is a handle to the figure window. 

   [AXH, FIGH] = PQSQRegularRegrPlot(B) plots all the coefficient values
       contained in B against the L1-norm of B. The L1-norm is the  sum of
       the absolute value of  all the coefficients. The plot is also
       annotated with the number of non-zero coefficients of B ("df"),
       displayed along the top axis of  the plot.

   [AXH, FIGH] = PQSQRegularRegrPlot(B,PLOTDATA) creates a plot with
       contents dependent on the type of PLOTDATA.

           If PLOTDATA is a vector, then PQSQRegularRegrPlot(B,PLOTDATA)
           uses the values of PLOTDATA to form the x-axis of the plot,
           rather than the L1-norm of B. In this case, PLOTDATA  must have
           the same length as the number of columns as B. 

           If PLOTDATA is a struct, then
           PQSQRegularRegrPlot(B,PLOTDATA,'PlotType',val) allows you to
           control aspects of the plot depending on the value of the
           optional argument 'PlotType'. 

           The possible values for 'PlotType' are:
           'L1' The x-axis of the plot is formed from the L1-norm of
                the coefficients in B. This is the default plot. 

           'Lambda' The x-axis of the plot is formed from the values of
                    the field named 'Lambda' in PLOTDATA.

           'CV' A different kind of plot is produced showing, for each
                lambda, an estimate of the goodness of fit on new data
                for the model fitted by PQSQRegularRegr with that
                value of lambda, plus error bars for the estimate. 
                For fits performed by PQSQRegularRegrPlot, the goodness of
                fit is MSE, or mean squared prediction error. The 'CV'
                plot also indicates the value for lambda with the minimum
                cross-validated measure of goodness of fit,  and the
                greatest lambda (thus sparsest model) that is  within one
                standard error of the minimum goodness of fit. The 'CV'
                plot type is valid only if PLOTDATA was produced by a call
                to PQSQRegularRegrPlot with cross-validation enabled. 

   PQSQRegularRegrPlot(...,'Name',val). Following pairs of Name, val are
       possible:

       'PredictorNames' contains a cell array of strings to label each of
           the coefficients of B. Default: {'B1', 'B2', ...}. If
           PredictorNames is of length less than the number of rows of B,
           the remaining labels will be padded with default values. 

       'XScale' is 'linear' for linear x-axis (Default),
                   'log' for logarithmic scale on the x-axis.

       'Parent' contains a axes in which to draw the plot. 

Examples:
       %(1) Run the lasso and PQSQ lasso simulation on data obtained from
       the 1985 %Auto Imports Database  of the UCI repository.  
       %http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

       load imports-85;

       %Extract Price as the response variable and extract non-categorical
       %variables related to auto construction and performance
       %
       X = X(~any(isnan(X(:,1:16)),2),:);
       Y = X(:,16);
       Y = log(Y);
       X = X(:,3:15);
       predictorNames = {'wheel-base' 'length' 'width' 'height' ...
           'curb-weight' 'engine-size' 'bore' 'stroke' 'compression-ratio' ...
           'horsepower' 'peak-rpm' 'city-mpg' 'highway-mpg'};
 
       %Compute the default sequence of lasso fits. Fix time.
       tic;
       [B,S] = lasso(X,Y,'CV',10,'PredictorNames',predictorNames);
       disp(['Time for lasso ',num2str(toc),' seconds']);
 
       %Compute PQSQ simulation of lasso without trimming
       tic;
       [BP,SP] = PQSQRegularRegr(X,Y,'CV',10,...
           'PredictorNames',predictorNames);
       disp(['Time for PQSQ simulation of lasso without trimming ',...
            num2str(toc),' seconds']);
 
       %Compute PQSQ simulation of lasso with trimming
       tic;
       [BT,ST] = PQSQRegularRegr(X,Y,'CV',10,...
           'TrimmingPoint',-1,'PredictorNames',predictorNames);
       disp(['Time for PQSQ simulation of lasso with trimming ',...
            num2str(toc),' seconds']);
 
 
       %Display a trace plot of the fits.
       %Lasso
       lassoPlot(B,S);
       %PQSQ without trimming
       PQSQRegularRegrPlot(BP,SP);
       %PQSQ with trimming
       PQSQRegularRegrPlot(BT,ST);
 
       %Display the sequence of cross-validated predictive MSEs.
       %Lasso
       lassoPlot(B,S,'PlotType','CV');
       %PQSQ without trimming
       PQSQRegularRegrPlot(BP,SP,'PlotType','CV');
       %PQSQ with trimming
       PQSQRegularRegrPlot(BT,ST,'PlotType','CV');
</pre>

<a name="Lab3">Description of function for automatic evaluation of optimal parameter of Tikhonov regularisation for linear regression</a>
<pre>
%fastRegularisedRegression perform feature selection for regression which
is regularized by Tikhonov regularization (ridge regression) with
automated selection of the optimal value of regularization parameter
Alpha.

Inputs:
   X is numeric matrix with n rows and p columns. Each row represents one
       observation, and each column represents one predictor (variable). 
   Y is numeric vector of length n, where n is the number of rows of X.
       Y(i) is the response to row i of X.

   IMPORTANT NOTES: data matrix X and response vector Y are standardized
       anyway but values of coefficients and intercept are calculated for
       the original values of X and Y

   Name, Value is one or more pairs of name and value. There are several
       possible names with corresponding values: 
       'Weights' is vector of observation weights. It must be a vector of
           non-negative values, of the same length as columns of X. At
           least two values must be positive. Default (1/N)*ones(N,1).
       'PredictorNames' is a cell array of names for the predictor
           variables, in the order in which they appear in X. Default: {}
       'CV', 'AlphaCV', 'FSCV'. If 'CV' is presented, then it indicates
           the method used to compute the final quality statistics (MSE,
           MAD or MAR). If 'AlphaCV' is presented, then it indicates the
           method used to compute accuracy (MSE, MAD or MAR) for searching
           of Alpha. If 'FSCV' is presented, then it indicates  the method
           used to compute accuracy (MSE, MAD or MAR) for feature
           selection. When parameter value is a positive integer K,
           fastRegularisedRegression uses K-fold cross-validation. Set
           parameter to a cross-validation partition, created using
           CVPARTITION, to use other forms of cross-validation. You cannot
           use a 'Leaveout' partition with fastRegularisedRegression. When
           parameter value is 'resubstitution', fastRegularisedRegression
           uses X and Y both to fit the model and to estimate the
           accuracy, without cross-validation.  
           The default is 
              'resubstitution' for 'CV'
               10 for 'AlphaCV'
               'resubstitution' for 'FSCV'
           'AlphaCV' parameter is ignored if value of parameter 'Alpha' is
           not 'CV'. 
       'MCReps', 'AlphaMCReps', 'FSMCReps' is a positive integer
           indicating the number of Monte-Carlo repetitions for
           cross-validation. If 
           'CV' ('AlphaCV', 'FSCV') is a cvpartition of type 'holdout',
           then 'MCReps' ('AlphaMCReps', 'FSMCReps') must be greater than
           one. Otherwise it is ignored. 
           The default value is 1. 
       'Options', 'AlphaOptions', 'FSOptions' is a structure that contains
           options specifying whether to conduct cross-validation
           evaluations in parallel, and options specifying how to use
           random numbers when computing cross validation partitions. This
           argument can be created by a call to STATSET. CROSSVAL uses the
           following fields:  
               'UseParallel'
               'UseSubstreams'
               'Streams'
           For information on these fields see PARALLELSTATS.
           NOTE: If supplied, 'Streams' must be of length one.
       'Regularize' is one of the following values:
           'Full' is used for full regularization by addition of identity
               matrix multiplied by 'Alpha'.
           'Partial' is used for partial regularization by substitution of
               'Alpha' for all eigenvalues which are less than 'Alpha'.
           Default value is 'Full'.
       'Alpha' is one of the following values:
           'Coeff' means that alpha is minimal which provide that all
               coefficients are less than or equal to 1. 
           'Iterative' means usage of iterative method with recalculation
               alpha.
           'CV' means usage of cross-validation to select alpha with
               smallest value of specified criterion. In this case
               behaviour of function can be customized by parameters
               'AlphaCV', 'AlphaMCReps', and 'AlphaOptions'. 
           nonnegative real number is fixed value of alpha. 
           negative real number -N is used for regularisation on base of
               condition number. If 'Regularize' has value 'Full' or
               omitted then
                   alpha = (maxE - N*minE)/(N-1);
               where maxE and minE are maximal and minimal singular
               values. If 'Regularize' has value 'Partial' then all
               eigenvalues which are less than alpha are substituted by
               alpha with alpha = maxE/N;
           Default is 'CV'
       'Criterion' is criterion which is used to search optimal 'Alpha'
           and for statistics calculation. Value of this parameter can
           be:
           'MSE' for mean squared error. 
           'MAD' for maximal absolute deviation.
           'MAR' for maximal absolute residual.
           handle of function with two arguments:
               res = fun(residuals, weights),
               where residual is n-by-k (k>=1) matrix of residuals'
               absolute values and weights is n-by-1 vector of weights.
               All weights are anyway normalised.
               res must be 1-by-k vector. 
               For example, for MSE function is
               function res = MSE(residuals, weights)
                   res = (weights'*(residuals.^2))/(1-sum(weights.^2));
               end
           For CV statistics is calculated for each fold (test set) and
           then is averaged among folds (test sets).
           Default value is 'MSE'.
       'FS' is feature selection method. It must be one of string:
           'Forward' means forward feature selection method.
           'Backward' means backward feature selection method (Feature
               elimination).
           'Sensitivity' means backward feature selection on base of
               feature sensitivity.

Output variable is structure with following fields:
   PredictorNames is a cell array of names for the predictor
           variables, in the order in which they appear in X. 
   MultiValue - array which contains the values of multicollinearity
       detection criteria: 1. VIF, 2. Corr, 3. Condition number
   MultiDecision is Boolean array of decision of existence of
       multicollinearity (true means that multicollinearity is found).
   Alphas is vector of values of regularization parameter for all sets of
       used input features.
   Criterion is name of used criterion for alpha selection and statistics
       calculation.
   Statistics is the vector of statistics specified by Criterion for all
       sets of used input features. This data are calculated by usage of
       all data for fitting and testing
   StatisticsCV is the vector of statistics specified by Criterion for all
       sets of used input features. This value is calculated for
       cross-validation only ('CV' is not 'resubstitution').
   SE is the vector standard errors of mean of Statistics for all sets
       of used input features. This value is calculated for
       cross-validation only ('CV' is not 'resubstitution').
   Intercept is vector of intercepts for all sets of used input features.
   Coefficients is matrix of regression coefficients with one column for
       each set of used input features. 
</pre>
