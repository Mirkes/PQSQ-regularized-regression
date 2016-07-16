%Time Test for elastic net and PQSQRegularRegr
%Use 100 of starts for each method to provide better accuracy of
%measurements.
tic;
for k=1:1
    [B,FitInfo] = lasso(Breast(:,2:end),Breast(:,1),'Alpha',0.5);
end
toc/1
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',FitInfo.Lambda,'Regular',{'elasticnet',0.5});
end
toc/100
tic;
for k=1:100
    [Pq1B,Pq1FitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',FitInfo.Lambda,'Regular',{'elasticnet3',0.5});
end
toc/100