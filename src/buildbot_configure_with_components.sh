#!/bin/sh 
# this is the configuration that goes into a fedora rpm
#./configure --with-debug --with-components="coretemp example infiniband lustre mx net" $1 
if [ -f components/cuda/Makefile.cuda ]; then
	if [ -f components/nvml/Makefile.nvml ]; then
		./configure --with-components="coretemp example lustre mx net cuda nvml" $1
	else
		./configure --with-components="coretemp example lustre mx net cuda" $1
	fi
else
	./configure --with-components="coretemp example lustre mx net" $1
fi
