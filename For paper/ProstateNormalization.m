%Normalisation of prostate cancer database
%Centralization
ProstateNorm = bsxfun(@minus, Prostate, mean(Prostate));
%Rescaling
ProstateNorm = bsxfun(@rdivide, ProstateNorm, sqrt(sum(ProstateNorm.^2)/size(Prostate,1)));