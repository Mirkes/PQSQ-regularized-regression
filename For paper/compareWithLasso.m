tic; [BPQ,FIPQ] = PQSQRegularRegr(Z,Y,'Lambda',FI.Lambda*scaleLambda); toc;
tic; [BPQ1,FIPQ1] = PQSQRegularRegr(Z,Y,'Lambda',FI.Lambda*scaleLambda,'Regular','lasso1'); toc;
if showallplots==1
figure;
plot(B+rand(n1,n2)/(max(max(abs(B)))*1000),BPQ+rand(n1,n2)/(max(max(abs(B)))*1000),'ko');
figure;
nonzeroparameters_number;
end