prostate = load('breast.txt');
X = prostate(:,2:end); Y = prostate(:,1); Z = zscore(X);
display(sprintf('Breast Cancer(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

prostate = load('prostate.txt');
X = prostate(:,1:8); Y = prostate(:,9); Z = zscore(X);
display(sprintf('\nProstate Cancer(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);


enb = load('enb.txt');
X = enb(:,1:8); Y = enb(:,9); Z = zscore(X);
display(sprintf('\nENB(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

ffires = load('ffires.txt');
X = ffires(:,1:8); Y = ffires(:,10); Z = zscore(X);
display(sprintf('\nForestFires(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

rnd1000_250 = load('rnd1000_250.txt');
X = rnd1000_250(:,2:end); Y = rnd1000_250(:,1); Z = zscore(X);
display(sprintf('\nrnd1000_250(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

parkinson = load('parkinson.txt');
X = parkinson(:,3:end); Y = parkinson(:,1); Z = zscore(X);
display(sprintf('\nParkinson(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);