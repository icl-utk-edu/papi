function FlopsSampler(n) 

% A Sampler of Matlab functions that consume Floating Point Operations
% in increasing order of floating point intensity.
%
%        FlopsSampler(n) - where n == array or vector size
%

fprintf(1,'\nCounts Using PAPI\n');
fprintf(1,'\n%24s %12s %14s %12s\n',  'Operations', 'n', 'fl pt ops', 'Mflop/s' )
s1=rand(1,1);s2=rand(1,1);
x=rand(n,1);y=rand(n,1);
a=rand(n); 
b=a;
c=a*a';

fprintf(1,'%25s', 'calling PAPI flops')
flops(0);
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'dot product')
flops(0);
x'*y; 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'matrix vector')
flops(0);
a*x; 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'random matrix')
flops(0);
a=rand(n);
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'chol(a)')
flops(0);
chol(c); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'lu(a)')
flops(0);
lu(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'x=a\y')
flops(0);
x=a\y; 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'condest(a)')
flops(0);
condest(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'qr(a)')
flops(0);
qr(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'matrix multiply')
flops(0);
a*b; 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'inv(a)')
flops(0);
inv(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'svd(a)')
flops(0);
svd(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'cond(a)')
flops(0);
cond(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'hess(a)')
flops(0);
hess(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'eig(a)')
flops(0);
eig(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', '[u,s,v]=svd(a)')
flops(0);
[u,s,v]=svd(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 'pinv(a)')
flops(0);
pinv(a);
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', 's=gsvd(a)')
flops(0);
s=gsvd(a,b);
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', '[x,e]=eig(a)')
flops(0);
[x,e]=eig(a); 
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

fprintf(1,'%25s', ' [u,v,x,c,s]=gsvd(a,b)')
flops(0);
[u,v,x,c,s]=gsvd(a,b);
[ops,mflops]=flops;
fprintf(1,'%12d %14d %12.2f\n', n, ops, mflops )

