n1 = size(B,1); n2 = size(B,2); 

logPlot = 0;
useMedianInsteadOfMin = 0;

maxv = 0;
minv = 1e6;
mse1 = zeros(1); mse1k = zeros(1);
mse2 = zeros(1); mse2k = zeros(1);
mse3 = zeros(1); mse3k = zeros(1);

c1 = 1;
c2 = 1;
c3 = 1;
for k=0:n1
    ind = FI.DF == k;
    err = FI.MSE(ind);
    if(size(err,2)>0)
    for i=1:size(err,2)
        if logPlot
            semilogy(k,err(i),'bx','MarkerSize',10); hold on;
        else
            plot(k,err(i),'bx','MarkerSize',10); hold on;
        end
        maxv = max(maxv,err(i));
        minv = min(minv,err(i));
    end
        mse1k(c1) = k;
        if useMedianInsteadOfMin
            mse1(c1) = median(err);
        else
            mse1(c1) = min(err);
        end
        c1 = c1+1;
    end
    
    ind = FIPQ.DF == k;
    err = FIPQ.MSE(ind);
    if(size(err,2)>0)
    for i=1:size(err,2)
        if logPlot        
            semilogy(k,err(i),'rs','MarkerSize',10); hold on;
        else
            plot(k,err(i),'rs','MarkerSize',10); hold on;
        end;
        maxv = max(maxv,err(i));
        minv = min(minv,err(i));
    end
        mse2k(c2) = k;
        if useMedianInsteadOfMin
            mse2(c2) = median(err);
        else
            mse2(c2) = min(err);
        end
        c2 = c2+1;
    end
    
    ind = FIPQ1.DF == k;
    err = FIPQ1.MSE(ind);
    if(size(err,2)>0)
    for i=1:size(err,2)
        if logPlot        
            semilogy(k,err(i),'m+','MarkerSize',10); hold on;
        else
            plot(k,err(i),'m+','MarkerSize',10); hold on;
        end;
        maxv = max(maxv,err(i));
        minv = min(minv,err(i));
    end
        mse3k(c3) = k;
        if useMedianInsteadOfMin
            mse3(c3) = median(err);
        else
            mse3(c3) = min(err);
        end
        c3 = c3+1;
    end
end

plot(mse1k,mse1,'b-','LineWidth',1.5);
plot(mse2k,mse2,'r-','LineWidth',1.5);
plot(mse3k,mse3,'m-','LineWidth',1.5);

t = minv-(maxv-minv)*5;
t = max(0,t);
ylim([t maxv]);

set(gca,'FontSize',10);
ylabel('MSE','FontSize',10);
xlabel('Number of non-zero parameters','FontSize',10);