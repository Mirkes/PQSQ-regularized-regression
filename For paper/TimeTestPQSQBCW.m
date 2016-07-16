%Time Test of different PQSQ regularized regressions
%Pseudo lasso
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular','lasso1');
end
toc/100
%pseudo elastic net with alpha = 0.5
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{'elasticnet3',0.5});
end
toc/100
%pseudo ridge
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{1,@L2,5,1});
end
toc/100
%Unnamed 3/2
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{1,@L1_5,5,1});
end
toc/100
%Unnamed log
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{1,@LLog,5,1});
end
toc/100
%Unnamed sqrt
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{1,@LSqrt,5,1});
end
toc/100
%All together
tic;
for k=1:100
    [PqB,PqFitInfo] = PQSQRegularRegr(Breast(:,2:end),Breast(:,1),...
        'Lambda',lambda,'Regular',{'elasticnet3',0.5},...
        'Regular',{0.5,@L1_5,5,1},'Regular',{0.5,@LLog,5,1},...
        'Regular',{0.5,@LSqrt,5,1});
end
toc/100