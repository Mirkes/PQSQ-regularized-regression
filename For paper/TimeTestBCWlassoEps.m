%Time Test for lasso and PQSQRegularRegr
%Use 100 of starts for each method to provide better accuracy of
%measurements. For standard lasso we use 1 start because it is slow enough
tic;
for k=1:1
    [B,FitInfo] = lasso(Breast(:,2:end),Breast(:,1));
end
toc/1
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',FitInfo.Lambda,'Epsilon',Inf);
end
toc/100
tic;
for k=1:100
    [Pq1B,Pq1FitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',FitInfo.Lambda,'Regular','lasso1','Epsilon',Inf);
end
toc/100