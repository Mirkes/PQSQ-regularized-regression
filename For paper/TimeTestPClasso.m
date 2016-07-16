%Time Test for lasso and PQSQRegularRegr
%Use 100 of starts for each method to provide better accuracy of
%measurements.
tic;
for k=1:100
    [B,FitInfo] = lasso(Prostate(:,1:8),Prostate(:,9));
end
toc/100
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Prostate(:,1:8),Prostate(:,9),...
        'Lambda',FitInfo.Lambda);
end
toc/100
tic;
for k=1:100
    [Pq1B,Pq1FitInfo] = PQSQRegularRegr(Prostate(:,1:8),Prostate(:,9),...
        'Lambda',FitInfo.Lambda,'Regular','lasso1');
end
toc/100