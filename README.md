# PQSQ-regularized-regression
The method of generalized regression regularization through PQSQ technique is presented. This approach includes lasso, elastic net and ridge regression as particular case.

Current MatLab implementation includes four files:
L1 is majorant function for L_1 norm
L1_5 is majorant function for L_1 + L_2 norm
L2 is majorant function for L_2 norm

lassoPQSQ is function of partial PQSQ-regularized-regression implementation.

<pre>

Syntax
   B = lassoPQSQ(X, Y)
   B = lassoPQSQ(X, Y, Name, Value)
   [B, FitInfo] = lassoPQSQ(X, Y)
   [B, FitInfo] = lassoPQSQ(X, Y, Name, Value)

Inputs
   X is numeric matrix with n rows and p columns. Each row represents one
       observation, and each column represents one predictor (variable). 
   Y is numeric vector of length n, where n is the number of rows of X.
       Y(i) is the response to row i of X.
   Name, Value is one or more pairs of name and value. There are several
       possible names with corresponding values:
       'Intervals', intervals serves to specify user defined intervals.
           intervals is row vector The first element must be zero. By
           default is created by usage of 'number_of_intervals' and
           'intshrinkage'. Maximal value M is maximum of absolute value of
           coefficients for OLS without any penalties multiplied by
           'intshrinkage'. All other boreders are calcualted as r(i) =
           M*i^2/p^2, where p is 'number_of_intervals'. 
       'Number_of_intervals' specifies the number of intervals to
           automatic interval calculation. Default value is 5. 
       'intshrinkage', delta serves to specify delta which is coefficient
           for intervals shrinkage (see argument delta in
           defineIntervals). Default value is 1 (no shrinkage).
       'Trimming' is value of trimming. If this value is not specified
           then maximal absolute value of regression coefficients wothout
           restriction is used. For standardized data theoretical maximum
           of regression coefficient is 1. It means that for standardized
           data 'trimming' is reasonably restricted by 1. Default empty
       'Epsilon' is positive value which spesify minimal nonzero value of
           regression coefficient. It means that attribute with absolute
           value of regression coefficient which is less than 'epsilon' is
           removed from regressions (coefficient becomes zero). There are
           three possible ways to spesify epsilon:
               positive value means that it is 'epsilon'.
               zero means that r(1)/2 is used as epsilon. r(1) is right
                   border of the first interval.
               negative value means that lambda*r(1)/2 is used as epsilon.
           Default 0.
       'potential' is majorant function for PQSQ. By default it is L1.
       'Alpha' is elastic net mixing value, or the relative balance
           between L2 and L1 penalty (default 1, range (0,1]). Alpha=1 ==>
           lasso, otherwise elastic net. Alpha near zero ==> nearly ridge
           regression. 
       'Lambda' is vector of Lambda values. Will be returned in return
           argument FitInfo in descending order. The default is to have
           lassoPQSQ generate a sequence of lambda values, based on
           'NumLambda' and 'LambdaRatio'. lassoPQSQ will generate a
           sequence, based on the values in X and Y, such that the largest
           LAMBDA value is just sufficient to produce all zero
           coefficients B in standard lasso. You may supply a vector of
           real, non-negative values of  lambda for lassoPQSQ to use, in
           place of its default sequence. If you supply a value for
           'Lambda', 'NumLambda' and  'LambdaRatio' are ignored. 
       'NumLambda' is the number of lambda values to use, if the parameter
           'Lambda' is not supplied (default 100). Ignored if 'Lambda' is
           supplied.  lassoPQSQ may return fewer fits than specified by
           'NumLambda' if the residual error of the fits drops below a
           threshold percentage of the variance of Y.
       'LambdaRatio' is ratio between the minimum value and maximum value
           of lambda to generate, if the  parameter "Lambda" is not
           supplied.  Legal range is [0,1). Default is 0.0001. If
           'LambdaRatio' is zero, lassoPQSQ will generate its default
           sequence of lambda values but replace the smallest value in
           this sequence with the value zero. 'LambdaRatio' is ignored if
           'Lambda' is supplied. 
       'Standardize' is indicator whether to scale X prior to fitting the
           model sequence. This affects whether the regularization is
           applied to the coefficients on the standardized scale or the
           original scale. The results are always presented on the
           original data scale. possible values true (any nonzero number)
           and false (zero). Default is TRUE, do scale X. 
                      Note: X and Y are always centered.
       'PredictorNames' is a cell array of names for the predictor
           variables, in the order in which they appear in X. Default: {}
       'Weights' is vector of observation weights.  Must be a vector of
           non-negative values, of the same length as columns of X.  At
           least two values must be positive. (default (1/N)*ones(N,1)). 

Return values:
   B is the fitted coefficients for each model. B will have dimension PxL,
       where P = size(X,2) is the number of predictors, and 
       L =  length(lambda). 
   FitInfo is a structure that contains information about the sequence of
       model fits corresponding to the columns of B. STATS contains the
       following fields: 
       'Intercept' is the intercept term for each model. Dimension 1xL.
       'Lambda' is the sequence of lambda penalties used, in ascending
           order. Dimension 1xL.
       'Alpha' is the elastic net mixing value that is used.
       'DF' is the number of nonzero coefficients in B for each value of
           lambda. Dimension 1xL. 
       'MSE' is the mean squared error of the fitted model for each value
       of lambda. Otherwise, 'MSE' is the mean sum of squared residuals
       obtained from the model with B and FitInfo.Intercept.
       'PredictorNames' is a cell array of names for the predictor
       variables, in the order in which they appear in X.
</pre>
