# File: ftests/Makefile
# CVS:  $Id$
INCLUDE = -I../testlib -I.
LIBRARY = -lpapi
PAPILIB = $(LIBRARY)
CC	= gcc
F77	= gfortran
CFLAGS	= -g -O 
FFLAGS	= -g -O -ffixed-line-length-132

ALL = strtest zero zeronamed first second tenth description fdmemtest accum highlevel cost \
	  case1 case2 clockres eventname fmatrixlowpapi fmatrixpapi fmatrixpapi2 flops fmultiplex1 \
      johnmay2 fmultiplex2 avail 
TESTLIB = ../testlib/libtestlib.a

default ftests ftest: $(FTEST_TARGETS)

all: $(ALL)

clockres: clockres.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) clockres.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) avail.F  $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) eventname.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) case1.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) case2.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o case2

fdmemtest: fdmemtest.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fdmemtest.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o fdmemtest

fmatrixlowpapi: fmatrixlowpapi.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixlowpapi.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixpapi.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixpapi2.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) strtest.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F $(TESTLIB)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) flops.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o flops

description: description.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) description.F $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o description

accum: accum.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) accum.F $(TESTLIB) $(PAPILIB) -o accum $(LDFLAGS)

highlevel: highlevel.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) highlevel.F $(TESTLIB) $(PAPILIB) -o highlevel $(LDFLAGS)

zero: zero.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) zero.F $(TESTLIB) $(PAPILIB) -o zero $(LDFLAGS)

zeronamed: zeronamed.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) zeronamed.F $(TESTLIB) $(PAPILIB) -o zeronamed $(LDFLAGS)

first: first.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) first.F $(TESTLIB) $(PAPILIB) -o first $(LDFLAGS)

second: second.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) second.F $(TESTLIB) $(PAPILIB) -o second $(LDFLAGS)

tenth: tenth.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) tenth.F $(TESTLIB) $(PAPILIB) -o tenth $(LDFLAGS)

cost: cost.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) cost.F $(TESTLIB) $(PAPILIB) -o cost $(LDFLAGS)

johnmay2: johnmay2.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) johnmay2.F $(TESTLIB) $(PAPILIB) -o johnmay2 $(LDFLAGS)

fmultiplex1: fmultiplex1.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmultiplex1.F $(TESTLIB) $(PAPILIB) -o fmultiplex1 $(LDFLAGS)

fmultiplex2: fmultiplex2.F $(TESTLIB) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmultiplex2.F $(TESTLIB) $(PAPILIB) -o fmultiplex2 $(LDFLAGS)

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

install: $(FTEST_TARGETS)
	@echo "Fortran tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ftests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ftests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ftests \;
	-chmod go+rx $(DATADIR)/ftests/*
	-find . -name "*.F" -type f -exec cp {} $(DATADIR)/ftests \;
