function FlopsMatrixMatrix

% Compute a Matrix Matrix multiply 
% on square arrays sized from 50 to 500,
% in steps of 50. 
%
% Use the PAPI flops call to measure the floating point operations performed.
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
% - mflops/s

fprintf(1,'\nPAPI Matrix Matrix Multiply Test');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n^3', 'difference', '% error', 'mflops')
for n=50:50:500,
    a=rand(n);b=rand(n);c=rand(n);
    flops(0);
    c=c+a*b;
    [count,mflops]=flops;
    fprintf(1,'%12d %12d %12d %12d %12.2f %12.2f\n',n,count,2*n^3,count - 2*n^3, (1.0 - ((2*n^3) / count)) * 100,mflops)
end