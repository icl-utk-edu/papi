#!/bin/sh

FTESTS="ftests/avail ftests/case1 ftests/case2 ftests/clockres ftests/eventname ftests/fmatrixlowpapi ftests/fmatrixpapi"
CTESTS="tests/zero_omp tests/zero_smp tests/zero_shmem tests/profile_pthreads tests/overflow_pthreads tests/zero_pthreads tests/avail tests/zero tests/sprofile tests/cost tests/johnmay2 tests/johnmay tests/inherit tests/clockres tests/first tests/second tests/third tests/fourth tests/fifth tests/overflow tests/profile tests/nineth tests/tenth tests/native tests/eventname tests/case1 tests/case2"
ALLTESTS="$FTESTS $CTESTS"
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
