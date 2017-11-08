# File: ftests/Makefile

include Makefile.target

INCLUDE = -I../testlib -I. -I..
FFLAGS	= $(CFLAGS) -ffixed-line-length-132
testlibdir=../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o

ifneq ($(F77),)
include Makefile.recipies

install: default
	@echo "Fortran tests (DATADIR) being installed in: \"$(datadir)\""; 
	-mkdir -p $(datadir)/ftests
	-chmod go+rx $(datadir)
	-chmod go+rx $(datadir)/ftests
	-find . -perm -100 -type f -exec cp {} $(datadir)/ftests \;
	-chmod go+rx $(datadir)/ftests/*
	-find . -name "*.[Ffh]" -type f -exec cp {} $(datadir)/ftests \;
	-cp Makefile.target $(datadir)/ftests/Makefile
	-cat Makefile.recipies >> $(datadir)/ftests/Makefile

else
all:
	@echo "Install Fortran compiler to build and run Fortran tests"

install:
	@echo "No Fortran tests to install."
endif
