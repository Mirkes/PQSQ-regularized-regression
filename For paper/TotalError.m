function err = TotalError(coef, X, Y, lambda)
%TotalError calculates lasso function of interest for centralized data
%matrix X and response vector Y, vector of regression coefficients coef and
%weight of lasso regularization term lambda.
%   X is data matrix with rows correspond to data points.
%   Y is column vector of responses
%   coef is column vector of regression coefficients
    %MSE calculation
    err = sum((Y - X*coef).^2)/size(Y,1);
    %Add lasso term
    err = err + lambda*sum(abs(coef));
end