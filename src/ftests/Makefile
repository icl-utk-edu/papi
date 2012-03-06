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
UTILOBJS = ../testlib/do_loops.o ../testlib/dummy.o ftests_util.o 

default ftests ftest: $(FTEST_TARGETS)

all: $(ALL)

clockres: clockres.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) clockres.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) avail.F  $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) eventname.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) case1.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) case2.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

fdmemtest: fdmemtest.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fdmemtest.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fdmemtest

fmatrixlowpapi: fmatrixlowpapi.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixlowpapi.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixpapi.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmatrixpapi2.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) strtest.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) flops.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o flops

description: description.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) description.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o description

accum: accum.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) accum.F $(UTILOBJS) $(PAPILIB) -o accum $(LDFLAGS)

highlevel: highlevel.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) highlevel.F $(UTILOBJS) $(PAPILIB) -o highlevel $(LDFLAGS)

zero: zero.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) zero.F $(UTILOBJS) $(PAPILIB) -o zero $(LDFLAGS)

zeronamed: zeronamed.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) zeronamed.F $(UTILOBJS) $(PAPILIB) -o zeronamed $(LDFLAGS)

first: first.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) first.F $(UTILOBJS) $(PAPILIB) -o first $(LDFLAGS)

second: second.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) second.F $(UTILOBJS) $(PAPILIB) -o second $(LDFLAGS)

tenth: tenth.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) tenth.F $(UTILOBJS) $(PAPILIB) -o tenth $(LDFLAGS)

cost: cost.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) cost.F $(UTILOBJS) $(PAPILIB) -o cost $(LDFLAGS)

johnmay2: johnmay2.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) johnmay2.F $(UTILOBJS) $(PAPILIB) -o johnmay2 $(LDFLAGS)

fmultiplex1: fmultiplex1.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmultiplex1.F $(UTILOBJS) $(PAPILIB) -o fmultiplex1 $(LDFLAGS)

fmultiplex2: fmultiplex2.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) fmultiplex2.F $(UTILOBJS) $(PAPILIB) -o fmultiplex2 $(LDFLAGS)

ftests_util.o: ftests_util.F fpapi_test.h 
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) -c ftests_util.F

do_loops.o: ../testlib/do_loops.c ../testlib/papi_test.h ../testlib/test_utils.h
	cd ../testlib && $(MAKE)

dummy.o: ../testlib/dummy.c
	cd ../testlib && $(MAKE)

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
