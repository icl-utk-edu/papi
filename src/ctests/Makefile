# File: ctests/Makefile

include Makefile.target

INCLUDE = -I../testlib -I../validation_tests -I.. -I.

testlibdir= ../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o
CLOCKCORE= $(testlibdir)/clockcore.o

validationlibdir= ../validation_tests
TESTFLOPS= $(validationlibdir)/flops_testcode.o
TESTINS= $(validationlibdir)/instructions_testcode.o
TESTCYCLES = $(validationlibdir)/busy_work.o
DISPLAYERROR= $(validationlibdir)/display_error.o

include Makefile.recipies

.PHONY : install

install: default
	@echo "C tests (DATADIR) being installed in: \"$(datadir)\""; 
	-mkdir -p $(datadir)/ctests
	-chmod go+rx $(datadir)
	-chmod go+rx $(datadir)/ctests
	-find . -perm -100 -type f -exec cp {} $(datadir)/ctests \;
	-chmod go+rx $(datadir)/ctests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(datadir)/ctests \;
	-cp Makefile.target $(datadir)/ctests/Makefile
	-cat Makefile.recipies >> $(datadir)/ctests/Makefile
