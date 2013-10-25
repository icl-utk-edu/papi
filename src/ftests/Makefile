# File: ftests/Makefile
# CVS:  $Id$
INCLUDE = -I../testlib
LIBRARY = -lpapi
PAPILIB = $(LIBRARY)
CC	= gcc
F77	= gfortran
CFLAGS	= -g -O 
FFLAGS	= -g -O -ffixed-line-length-132

TESTLIB = ../testlib/libtestlib.a

include Makefile.recipies

install: default
	@echo "Fortran tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ftests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ftests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ftests \;
	-chmod go+rx $(DATADIR)/ftests/*
	-find . -name "*.[Ffh]" -type f -exec cp {} $(DATADIR)/ftests \;
	-cp Makefile.target $(DATADIR)/ftests/Makefile
	-cat Makefile.recipies >> $(DATADIR)/ftests/Makefile
