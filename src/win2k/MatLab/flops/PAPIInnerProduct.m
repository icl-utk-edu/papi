function PAPIInnerProduct

% Compute an Inner Product (c = a * x) 
% on elements sized from 50 to 500,
% in steps of 50. 
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
% - mflops/s

fprintf(1,'\nPAPI Inner Product Test');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n', 'difference', '% error', 'mflops')
for n=50:50:500,
    a=rand(1,n);x=rand(n,1);
    flops(0);
    c=a*x;
    [count,mflops]=flops;
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.2f\n',n,count,2*n,count - 2*n, (1.0 - ((2*n) / count)) * 100,mflops)
end