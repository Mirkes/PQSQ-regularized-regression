breast = load('breast.txt');
X = breast(:,2:end); Y = zscore(breast(:,1)); Z = zscore(X);
display(sprintf('Breast Cancer(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

prostate = load('prostate.txt');
X = prostate(:,1:8); Y = zscore(prostate(:,9)); Z = zscore(X);
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
%X = parkinson(:,3:end); Y = parkinson(:,1); Z = zscore(X);
X = log10(parkinson(:,3:end)+1); Y = zscore(parkinson(:,1)); Z = zscore(X);
display(sprintf('\nParkinson(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

crime = load('crime.txt');
X = crime(:,1:99); Y = zscore(crime(:,100)); Z = zscore(X);
display(sprintf('\nCrime(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

crime_small = load('crime.txt');
ind = [1:4:1994];
X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X);
display(sprintf('\nCrime reduced(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);


crime_small = load('crime.txt');
ind = [1:3:1994];
X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X);
display(sprintf('\nCrime reduced(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

crime_small = load('crime.txt');
ind = [1:5:1994];
X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X);
display(sprintf('\nCrime reduced(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

crime_small = load('crime.txt');
ind = [1:10:2000];
X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X);
display(sprintf('\nCrime reduced(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);

crime_small = load('crime.txt');
ind = [1:20:2000]; X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X);
display(sprintf('\nCrime reduced(%i,%i)',size(X,1),size(X,2)));
display('MATLAB lasso:');
tic; [B,FI] = lasso(Z,Y); toc; lassoPlot(B,FI);