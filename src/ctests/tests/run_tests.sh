#!/bin/sh

ALLTESTS="zero_omp zero_smp zero_shmem profile_pthreads overflow_pthreads zero_pthreads avail zero sprofile cost johnmay2 johnmay inherit clockres first second third fourth fifth overflow profile nineth tenth native eventname case1 case2"
x=0;

echo "PAPI library test suite: All logs in <test>.stdout and <test>.stderr";

for i in $ALLTESTS;
do
if [ -x $i ]; then
echo -n "Running $i: ";
./$i > $i.stdout 2> $i.stderr
if [ $? -ne 0 ]; then
echo "FAILED";
x=1;
else
echo "PASSED";
fi;
else
echo "Skipping $i...";
fi;
done
