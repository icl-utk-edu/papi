INCLUDE = -I.. -I../ctests
TESTS 	= zero first  second tenth description \
          accum highlevel cost johnmay2 native \
	  avail case1 case2 clockres eventname fmatrixlowpapi \
	  fmatrixpapi fmatrixpapi2 flops fmultiplex1 fmultiplex2
PAPILIB=../ctests/do_loops.o ../ctests/dummy.o ftests_util.o ../$(LIBRARY)
FTEST_DEPENDENCE=$(PAPILIB) fpapi_test.h ../fpapi.h ../ctests/test_utils.h

default: $(TESTS)

mpi:

shared:

multiplex:

omp:

smp: 

pthreads:

shmem:

multiplex_and_pthreads: 

serial:  $(TESTS)


clockres: clockres.F $(FTEST_DEPENDENCE) 
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) clockres.F $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) avail.F  $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) eventname.F $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(FTEST_DEPENDENCE) 
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) case1.F $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) case2.F $(PAPILIB) $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi.F $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi2.F $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) strtest.F $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F $(FTEST_DEPENDENCE)  
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) flops.F $(PAPILIB) $(LDFLAGS) -o flops

description: description.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) description.F $(PAPILIB) $(LDFLAGS) -o description

accum: accum.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) accum.F $(PAPILIB) -o accum $(LDFLAGS)

highlevel: highlevel.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) highlevel.F $(PAPILIB) -o highlevel $(LDFLAGS)

zero: zero.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) zero.F $(PAPILIB) -o zero $(LDFLAGS)

first: first.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) first.F $(PAPILIB) -o first $(LDFLAGS)

second: second.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) second.F $(PAPILIB) -o second $(LDFLAGS)

tenth: tenth.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) tenth.F $(PAPILIB) -o tenth $(LDFLAGS)

cost: cost.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) cost.F $(PAPILIB) -o cost $(LDFLAGS)

johnmay2: johnmay2.F $(FTEST_DEPENDENCE)
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) johnmay2.F $(PAPILIB) -o johnmay2 $(LDFLAGS)

native: native.F $(FTEST_DEPENDENCE) 
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) native.F $(PAPILIB) -o native $(LDFLAGS)

fmultiplex1: fmultiplex1.F $(FTEST_DEPENDENCE) 
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex1.F $(PAPILIB) -o fmultiplex1 $(LDFLAGS)


fmultiplex2: fmultiplex2.F $(FTEST_DEPENDENCE) 
	$(F77) $(FTOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex2.F $(PAPILIB) -o fmultiplex2 $(LDFLAGS)

ftests_util.o: ftests_util.F fpapi_test.h ../fpapi.h ../ctests/test_utils.h
	$(F77) $(FTOPTFLAGS) -c $(FFLAGS) $(INCLUDE) ftests_util.F

do_loops.o: do_loops.F fpapi_test.h ../fpapi.h ../ctests/test_utils.h
	$(F77) $(FFLAGS) $(INCLUDE) $(FTOPTFLAGS) -c do_loops.F

dummy.o: dummy.F
	$(F77) $(FFLAGS) $(INCLUDE) $(FTOPTFLAGS) -c dummy.F

clean:
	rm -f *.o *.stderr *.stdout core *~ $(TESTS)

install: $(TARGETS)
	@if [ \"$(PREFIX)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make PREFIX=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(PREFIX)\""; 
	-mkdir -p $(PREFIX)/share/papi/ftests
	-chmod go+rx $(PREFIX)/share/papi/ftests
	-find . -perm -100 -type f -exec cp {} $(PREFIX)/share/papi/ftests \;
	-chmod go+rx $(PREFIX)/share/papi/ftests/*
	-find . -name "*.F" -type f -exec cp {} $(PREFIX)/share/papi/ftests \;
