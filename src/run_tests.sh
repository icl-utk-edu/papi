#!/bin/sh

# File:    papi.c
# CVS:     $Id$
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu

# The export below breaks Solaris, this needs to be fixed somehow
#export AIXTHREAD_SCOPE=S
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
echo "Running C Tests";
echo ""

for i in $CTESTS;
do
if [ -x $i ]; then
if [ "$i" = "ctests/timer_overflow" ]; then
  echo Skipping test $i, it takes too long...
else
if [ "$i" = "ctests/shlib" ]; then
  echo -n "Running $i: ";
  LD_LIBRARY_PATH=.
  export LD_LIBRARY_PATH
  LIBPATH=.
  export LIBPATH
  ./$i $TESTS_QUIET
else
echo -n "Running $i: ";
./$i $TESTS_QUIET
fi;
fi;
fi;
done

echo ""
echo "Running Fortran Tests";
echo ""

for i in $FTESTS;
do
if [ -x $i ]; then
echo -n "Running $i: ";
./$i $TESTS_QUIET
fi;
done
