# File: ctests/Makefile
INCLUDE = -I.. -I../testlib
PAPILIB=$(LIBRARY)
CC	= gcc
CC_R	= $(CC) -pthread
CFLAGS	= -g -O -Wall
testlibdir=../testlib
TESTLIB= $(testlibdir)/libtestlib.a

include Makefile.recipies

.PHONY : install

install: default
	@echo "C tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ctests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ctests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ctests \;
	-chmod go+rx $(DATADIR)/ctests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/ctests \;
	-cp Makefile.target $(DATADIR)/ctests/Makefile
	-cat Makefile.recipies >> $(DATADIR)/ctests/Makefile
