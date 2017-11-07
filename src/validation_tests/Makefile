# File: validation_tests/Makefile

include Makefile.target

INCLUDE = -I../testlib -I.. -I.
PAPILIB= ../$(LIBRARY)
testlibdir= ../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o
CLOCKCORE= $(testlibdir)/clockcore.o
EXTRALIB = -lrt

include Makefile.recipies

.PHONY : install

install:	all
	@echo "Validation tests (DATADIR) being installed in: \"$(datadir)\"";
	-mkdir -p $(datadir)/validation_tests
	-chmod go+rx $(datadir)
	-chmod go+rx $(datadir)/validation_tests
	-find . -perm -100 -type f -exec cp {} $(datadir)/validation_tests \;
	-chmod go+rx $(datadir)/validation_tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(datadir)/validation_tests \;
	-cp Makefile.target $(datadir)/validation_tests/Makefile
	-cat Makefile.recipies >> $(datadir)/validation_tests/Makefile

