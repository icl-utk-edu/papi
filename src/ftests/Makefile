# File: ftests/Makefile

include Makefile.target
INCLUDE = -I../testlib -I. -I..

cray_compilers := ftn% crayftn%
flang_compilers := flang% armflang% amdflang%
gnu_compilers := gfortran%
intel_compilers := ifort% ifx%
nvidia_compilers := nvfortran% pgi77%
ifneq ($(filter $(cray_compilers), $(notdir $(F77))),)
    FFLAGS += -N 132
else ifneq ($(filter $(flang_compilers), $(notdir $(F77))),)
    FFLAGS += -ffixed-line-length=132
else ifneq ($(filter $(gnu_compilers), $(notdir $(F77))),)
    FFLAGS += -ffixed-line-length-132
else ifneq ($(filter $(intel_compilers), $(notdir $(F77))),)
    FFLAGS += -extend-source 132
else ifneq ($(filter $(nvidia_compilers), $(notdir $(F77))),)
    FFLAGS += -Mextend
endif

testlibdir=../testlib
TESTLIB= $(testlibdir)/libtestlib.a
DOLOOPS= $(testlibdir)/do_loops.o

ifeq ($(ENABLE_FORTRAN_TESTS),yes)
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

else
all:
	@echo "Install Fortran compiler to build and run Fortran tests"

install:
	@echo "No Fortran tests to install."

clean:

distclean clobber:
	rm -f Makefile.target
endif
