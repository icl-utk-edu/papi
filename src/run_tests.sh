#!/bin/sh

# File:    papi.c
# CVS:     $Id$
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu

CTESTS=`find tests -perm -u+x -type f`;
FTESTS=`find ftests -perm -u+x -type f`;
ALLTESTS="$FTESTS $CTESTS";
x=0;
CWD=`pwd`

echo "The following test cases will be run";
echo $ALLTESTS;
echo "\n";

echo "Running C Tests\n";
for i in $CTESTS;
do
if [ -x $i ]; then
echo -n "Running $i: ";
$CWD/$i TESTS_QUIET
fi;
done

echo "\n\nRunning Fortran Tests\n";
for i in $FTESTS;
do
if [ -x $i ]; then
echo -n "Running $i: ";
$CWD/$i TESTS_QUIET
fi;
done
