fprintf(1,'\nPAPI Matrix Matrix Multiply Test');
fprintf(1,'\n%12s %12s %12s %12s %12s %12s\n', 'n', 'ops', '2n^3', 'difference', '% error', 'mflops')
i=0;
for n=50:50:500,
    a=rand(n);b=rand(n);c=rand(n);
    i=i+1;
    PAPI_start_flops;
    c=c+a*b;
    [count(i),mflops]=PAPI_count_flops;
    fprintf(1,'%12d %12d %12d %12d %12.2g %12.2g\n',n,count(i),2*n^3,count(i) - 2*n^3, (1.0 - ((2*n^3) / (count(i)))) * 100,mflops)
end