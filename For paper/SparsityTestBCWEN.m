%Test standard lasso
[B,FitInfo] = lasso(Breast(:,2:end),Breast(:,1),'Alpha',0.5);
%Test PQSQRegularRegr lasso without trimming
[PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
    'Lambda',FitInfo.Lambda,'Regular',{'elasticnet',0.5});
%Test PQSQRegularRegr lasso with possible trimming
[Pq1B,Pq1FitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
    'Lambda',FitInfo.Lambda,'Regular',{'elasticnet3',0.5});
%Define array for results
sparsity = zeros(9,3);
for k=1:32 %k is number of nonzero coefficients plus 1
    %Standard lasso
    sparsity(k,1) = bestError( FitInfo, k-1 );
    sparsity(k,2) = bestError( PqFitInfo, k-1 );
    sparsity(k,3) = bestError( Pq1FitInfo, k-1 );
end