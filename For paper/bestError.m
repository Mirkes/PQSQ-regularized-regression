function err = bestError( FitInfo, nonZero )
%bestError serches the minimal value of FitInfo.MSE for elements which
%corresponds to FitInfo.DF==nonZero (with the same number of nonzero
%coefficients).
    ind = FitInfo.DF == nonZero;
    if any(ind)
        %There is at leas one lambda with nonZero nonzero coefficients
        err = min(FitInfo.MSE(ind));
    else
        %Absolutely impossible value as a sign of absence
        err = -1;
    end
end