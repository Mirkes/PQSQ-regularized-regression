%Test of lasso
%Create array for results
res = zeros(100,4);
%Calculate solution by lasso and fix lambdas
[B,FitInfo] = lasso(ProstateNorm(:,1:8),ProstateNorm(:,9));
lambda = FitInfo.Lambda;
%Lasso uses ascending order of lambda and lassoPQSQ and PQSQRegularRegr
%use ascending one. To uniformity it is necessary to flip lasso results.
res(:,1) = flip(lambda');
res(:,2) = flip(TotalErrors(B,ProstateNorm(:,1:8),ProstateNorm(:,9)...
    ,lambda)');
lambda = flip(lambda);
[B3,FitInfo3] = lassoPQSQ(ProstateNorm(:,1:8),ProstateNorm(:,9),...
    'Lambda',lambda,'trimming',1);
res(:,3) = TotalErrors(B3,ProstateNorm(:,1:8),ProstateNorm(:,9),lambda)';
[B4,FitInfo4] = PQSQRegularRegr(ProstateNorm(:,1:8),ProstateNorm(:,9),...
        'Lambda',lambda);
res(:,4) = TotalErrors(B4,ProstateNorm(:,1:8),ProstateNorm(:,9),lambda)';