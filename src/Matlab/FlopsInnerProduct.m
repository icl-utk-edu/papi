function FlopsInnerProduct

% Compute an Inner Product (c = a * x) 
% on elements sized from 50 to 500,
% in steps of 50.
%
% Use the PAPI flops call to measure the floating point operations performed.
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
% - mflops/s

fprintf(1,'\nPAPI Inner Product Test');
fprintf(1,'\nUsing flops');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n', 'difference', '% error', 'mflops')
for n=50:50:500,
    a=rand(1,n);x=rand(n,1);
    flops(0);
    c=a*x;
    [ops, mflops] = flops;
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.2f\n',n,ops,2*n,ops - 2*n, (1.0 - ((2*n) / ops)) * 100,mflops)
end
