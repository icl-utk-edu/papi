% Compute a Matrix Matrix multiply 
% on square arrays sized from 50 to 500,
% in steps of 50. 
% For each size, display:
% - number of floating point operations
% - theoretical number of operations
% - difference
% - per cent error
% - mflops/s
fprintf(1,'\nPAPI Matrix Matrix Multiply Test');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n^3', 'difference', '% error', 'mflops')
i=0;
for n=50:50:500,
    a=rand(n);b=rand(n);c=rand(n);
    i=i+1;
    flops(0);
    c=c+a*b;
    [count(i),mflops]=floprate;
    fprintf(1,'%12d %12d %12d %12d %12.2g %12.2g\n',n,count(i),2*n^3,count(i) - 2*n^3, (1.0 - ((2*n^3) / (count(i)))) * 100,mflops)
end