#!/bin/sh

# File:    papi.c
# CVS:     $Id$
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu

CTESTS=`find tests -perm -u+x -type f`;
#FTESTS=`find ftests -perm -u+x -type f`;
ALLTESTS="$FTESTS $CTESTS";
x=0;

echo "The following test cases will be run";
echo $ALLTESTS;
echo "";

for i in $ALLTESTS;
do
if [ -x $i ]; then
./$i TESTS_QUIET
fi;
done
