#!/bin/sh 
# this is the configuration that goes into a fedora rpm
#./configure --with-debug --with-components="coretemp example infiniband lustre mx net" $1 
./configure --with-debug --with-components="coretemp example lustre mx net" $1 
