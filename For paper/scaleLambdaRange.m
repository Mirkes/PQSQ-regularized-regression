scaleLambda = 1;
n1 = size(B,1); n2 = size(B,2);
maxLambda = max(FI.Lambda);
for i=1:100
    [BPQtest,FIPQtest] = PQSQRegularRegr(Z,Y,'Lambda',maxLambda*scaleLambda);
    nz = sum(BPQtest(:)~=0);
    if nz==0
        break;
    end
    scaleLambda = scaleLambda*1.5;
end

display(sprintf('scaleLambda = %f',scaleLambda));

