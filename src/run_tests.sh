#!/bin/sh

# File:    papi.c
# CVS:     $Id$
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    <your name here>
#          <your email address>

CTESTS=`find tests -perm -u+x -type f`;
FTESTS=`find ftests -perm -u+x -type f`;
ALLTESTS="$FTESTS $CTESTS";
x=0;

echo "PAPI library test suite: All logs in <test>.stdout and <test>.stderr";
echo "";
echo "The following test cases will be run";
echo $ALLTESTS;
echo "";

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
