#!/bin/sh

# File:    run_cat_tests.c
# CVS:     $Id$
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu
# Mods:    Dan Terpstra
#          terpstra@cs.utk.edu

export AIXTHREAD_SCOPE=S
if [ "X$1" = "X-v" ]; then
  shift ; TESTS_QUIET=""
else
  TESTS_QUIET="TESTS_QUIET"
fi

CTESTS=`find ctests -perm -u+x -type f`;
FTESTS=`find ftests -perm -u+x -type f`;
ALLTESTS="$CTESTS $FTESTS";
x=0;
CWD=`pwd`

echo "Platform:"
uname -a

echo ""
echo "The following test cases will be run:";
echo $ALLTESTS;

echo "";
echo "Running Catamount C Tests";
echo ""

for i in $CTESTS;
do
if [ -x $i ]; then
if [ "$i" = "ctests/timer_overflow" ]; then
  echo Skipping test $i, it takes too long...
else
echo -n "Running $i: ";
#./$i $TESTS_QUIET
yod -sz 1 $i $TESTS_QUIET
fi;
fi;
done

echo ""
echo "Running Catamount Fortran Tests";
echo ""

for i in $FTESTS;
do
if [ -x $i ]; then
echo -n "Running $i: ";
#./$i $TESTS_QUIET
yod -sz 1 $i $TESTS_QUIET
fi;
done
