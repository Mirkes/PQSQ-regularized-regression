showallplots=0;

figure;

display(sprintf('========\nBreast cancer:'));
breast = load('..\AZ_tests\breast.txt');
X = breast(:,2:end); Y = zscore(breast(:,1)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,1);
compareMSE; title('Breast cancer (47 obj, 31 attr)','FontSize',14);

display(sprintf('\n\n========\nProstate cancer :'));
prostate = load('..\AZ_tests\prostate.txt');
X = prostate(:,1:8); Y = zscore(prostate(:,9)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,2);
compareMSE; title('Prostate cancer (97 obj, 8 attr)','FontSize',14);

display(sprintf('\n\n========\nENB:'));
enb = load('..\AZ_tests\enb.txt');
X = enb(:,[1:8]); Y = zscore(enb(:,9)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,3);
compareMSE; title('ENB (768 obj, 8 attr)','FontSize',14);

display(sprintf('\n\n========\nParkinson:'));
parkinson = load('..\AZ_tests\parkinson.txt');
X = log10(parkinson(:,3:end)+1); Y = zscore(parkinson(:,1)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,4);
compareMSE; title('Parkinson dataset (5875 obj, 26 attr)','FontSize',14);

display(sprintf('\n\n========\nCrime:'));
crime = load('..\AZ_tests\crime.txt');
X = crime(:,1:99); Y = zscore(crime(:,100)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,5);
compareMSE; title('Crime (1994 obj, 100 attr)','FontSize',14);

display(sprintf('\n\n========\nCrime small:'));
ind = (1:10:2000); X = crime(ind,1:99); Y = zscore(crime(ind,100)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,6);
compareMSE; title('Crime reduced (200 obj, 100 attr)','FontSize',14);

display(sprintf('\n\n========\nForest fires:'));
ffires = load('..\AZ_tests\ffires.txt');
X = ffires(:,1:8); Y = zscore(ffires(:,10)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,7);
compareMSE; title('Forest fires (517 obj, 8 attr)','FontSize',14);

display(sprintf('\n\n========\nRandom regression (1000 obj, 250 attr):'));
rnd1000_250 = load('..\AZ_tests\rnd1000_250.txt');
X = rnd1000_250(:,2:end); Y = zscore(rnd1000_250(:,1)); Z = zscore(X); tic; [B,FI] = lasso(Z,Y); toc;
scaleLambdaRange; compareWithLasso; 
%figure; 
subplot(4,2,8);
compareMSE; title('Random regression (1000x250)','FontSize',14);


showallplots=1;