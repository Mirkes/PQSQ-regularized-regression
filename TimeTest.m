%T1me Test
tic;
for k=1:1
%    [B,FitInfo] = lasso(ProstNorm(:,1:8),ProstNorm(:,9));
    [B,FitInfo] = lasso(Breast(:,2:end),Breast(:,1));
end
toc
tic;
for k=1:100
%    [PqB,PqFitInfo] = lassoPQSQ(ProstNorm(:,1:8),ProstNorm(:,9));
    [B,FitInfo] = lassoPQSQ(Breast(:,2:end),Breast(:,1),'epsilon',-1,'trimming',1);
end
toc/100
tic;
for k=1:100
%    [PqB,PqFitInfo] = lassoPQSQ(ProstNorm(:,1:8),ProstNorm(:,9));
    [B,FitInfo] = lassoPQSQ(Breast(:,2:end),Breast(:,1),'epsilon',-1);
end
toc/100