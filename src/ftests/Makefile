TESTS 	= strtest zero first  second tenth description fdmemtest\
          accum highlevel cost johnmay2 native \
	  case1 case2 clockres eventname fmatrixlowpapi \
	  fmatrixpapi fmatrixpapi2 flops fmultiplex1 fmultiplex2 avail 
UTILOBJS = do_loops.o dummy.o ftests_util.o 
PAPILIB	= ../$(LIBRARY)
INCLUDE = -I. -I.. -I../ctests
 
default: $(TESTS)

attach:

p4:

ear:

range:

mpi:

shared:

multiplex:

omp:

smp: 

pthreads:

shmem:

multiplex_and_pthreads: 

serial:  $(TESTS)

clockres: clockres.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) clockres.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) avail.F  $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) eventname.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) case1.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) case2.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

fdmemtest: fdmemtest.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fdmemtest.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fdmemtest

fmatrixlowpapi: fmatrixlowpapi.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi2.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) strtest.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F $(UTILOBJS)  $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) flops.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o flops

description: description.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) description.F $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o description

accum: accum.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) accum.F $(UTILOBJS) $(PAPILIB) -o accum $(LDFLAGS)

highlevel: highlevel.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) highlevel.F $(UTILOBJS) $(PAPILIB) -o highlevel $(LDFLAGS)

zero: zero.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) zero.F $(UTILOBJS) $(PAPILIB) -o zero $(LDFLAGS)

first: first.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) first.F $(UTILOBJS) $(PAPILIB) -o first $(LDFLAGS)

second: second.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) second.F $(UTILOBJS) $(PAPILIB) -o second $(LDFLAGS)

tenth: tenth.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) tenth.F $(UTILOBJS) $(PAPILIB) -o tenth $(LDFLAGS)

cost: cost.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) cost.F $(UTILOBJS) $(PAPILIB) -o cost $(LDFLAGS)

johnmay2: johnmay2.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) johnmay2.F $(UTILOBJS) $(PAPILIB) -o johnmay2 $(LDFLAGS)

native: native.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) native.F $(UTILOBJS) $(PAPILIB) -o native $(LDFLAGS)

fmultiplex1: fmultiplex1.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex1.F $(UTILOBJS) $(PAPILIB) -o fmultiplex1 $(LDFLAGS)

fmultiplex2: fmultiplex2.F $(UTILOBJS) $(PAPILIB)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex2.F $(UTILOBJS) $(PAPILIB) -o fmultiplex2 $(LDFLAGS)

ftests_util.o: ftests_util.F fpapi_test.h 
	$(F77) $(FTOPTFLAGS) -c $(FFLAGS) $(INCLUDE) ftests_util.F

do_loops.o: ../ctests/do_loops.c ../ctests/papi_test.h ../ctests/test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c ../ctests/do_loops.c

dummy.o: ../ctests/dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c ../ctests/dummy.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(TESTS)

install: $(TARGETS)
	@echo "Fortran tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ftests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ftests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ftests \;
	-chmod go+rx $(DATADIR)/ftests/*
	-find . -name "*.F" -type f -exec cp {} $(DATADIR)/ftests \;
