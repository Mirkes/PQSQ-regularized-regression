%Test standard lasso
[B,FitInfo] = lasso(Prostate(:,1:8),Prostate(:,9),'Alpha',0.5);
%Test PQSQRegularRegr elastic net without trimming
[PqB,PqFitInfo] = PQSQRegularRegr(Prostate(:,1:8),Prostate(:,9),...
    'Regular',{'elasticnet',0.5});
%Test PQSQRegularRegr elastic net with possible trimming
[Pq1B,Pq1FitInfo] = PQSQRegularRegr(Prostate(:,1:8),Prostate(:,9),...
    'Regular',{'elasticnet3',0.5});
%Define array for results
sparsity = zeros(9,3);
for k=1:9 %k is number of nonzero coefficients plus 1
    %Standard lasso
    sparsity(k,1) = bestError( FitInfo, k-1 );
    sparsity(k,2) = bestError( PqFitInfo, k-1 );
    sparsity(k,3) = bestError( Pq1FitInfo, k-1 );
end