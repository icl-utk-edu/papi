% Compute an Inner Product (c = a * x) 
% on elements sized from 50 to 500,
% in steps of 50. 
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
fprintf(1,'\nPAPI Inner Product Test');
fprintf(1,'\n%12s %12s %12s %12s %12s\n', 'n', 'ops', '2n', 'difference', '% error')
i=0;
for n=50:50:500,
    a=rand(1,n);x=rand(n,1);
    i=i+1;
    flops(0);
    c=a*x;
    count(i)=flops;
    fprintf(1,'%12d %12d %12d %12d %12.2g\n',n,count(i),2*n,count(i) - 2*n, (1.0 - ((2*n) / (count(i)))) * 100)
end