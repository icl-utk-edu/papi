INCLUDE = -I..
TESTS 	= zero first \
	  second third fourth fifth nineth tenth profile description \
          accum highlevel cost johnmay johnmay2 native profile \
	  avail case1 case2 clockres eventname fmatrixlowpapi \
	  fmatrixpapi fmatrixpapi2 flops fmultiplex1 fmultiplex2

FTEST_DEPENDENCE=ftests_util.o
PAPILIB=ftests_util.o ../$(LIBRARY)

default: $(TESTS)

ftests_util.o: ftests_util.F
	@if [ $(SUBSTR) = tru64-alpha ]; then $(F77) -g2 $(FOPTFLAGS) -c $(FFLAGS) $(INCLUDE) ftests_util.F; \
	else if [ $(SUBSTR) = solaris-ultra ]; then $(F77) -c $(FFLAGS) $(INCLUDE) ftests_util.F;  \
	else $(F77) $(FOPTFLAGS) -c $(FFLAGS) $(INCLUDE) ftests_util.F; fi; fi

addAIXevents.o: addAIXevents.c
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) -c addAIXevents.c

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) -c do_loops.c

clockres: clockres.F $(FTEST_DEPENDENCE) 
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) clockres.F $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) avail.F  $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) eventname.F $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(FTEST_DEPENDENCE) 
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) case1.F $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) case2.F $(PAPILIB) $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi.F $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi2.F $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) strtest.F $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F $(FTEST_DEPENDENCE)  
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) flops.F $(PAPILIB) $(LDFLAGS) -o flops

description: description.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) description.F $(PAPILIB) $(LDFLAGS) -o description

accum: accum.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) accum.F $(PAPILIB) -o accum $(LDFLAGS)

highlevel: highlevel.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) highlevel.F $(PAPILIB) -o highlevel $(LDFLAGS)

zero: zero.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) zero.F $(PAPILIB) -o zero $(LDFLAGS)

first: first.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) first.F $(PAPILIB) -o first $(LDFLAGS)

second: second.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) second.F $(PAPILIB) -o second $(LDFLAGS)

third: third.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) third.F $(PAPILIB) -o third $(LDFLAGS)

fourth: fourth.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fourth.F $(PAPILIB) -o fourth $(LDFLAGS)

fifth: fifth.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fifth.F $(PAPILIB) -o fifth $(LDFLAGS)

nineth: nineth.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) nineth.F $(PAPILIB) -o nineth $(LDFLAGS)

tenth: tenth.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) tenth.F $(PAPILIB) -o tenth $(LDFLAGS)

cost: cost.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) cost.F $(PAPILIB) -o cost $(LDFLAGS)

johnmay: johnmay.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) johnmay.F $(PAPILIB) -o johnmay $(LDFLAGS)

johnmay2: johnmay2.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) johnmay2.F $(PAPILIB) -o johnmay2 $(LDFLAGS)

native: native.F $(FTEST_DEPENDENCE) addAIXevents.o
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) native.F $(PAPILIB) addAIXevents.o -o native $(LDFLAGS)

profile: profile.F $(FTEST_DEPENDENCE)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) profile.F $(PAPILIB) -o profile $(LDFLAGS)

fmultiplex1: fmultiplex1.F $(FTEST_DEPENDENCE) do_loops.o
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex1.F $(PAPILIB) do_loops.o -o fmultiplex1 $(LDFLAGS)


fmultiplex2: fmultiplex2.F $(FTEST_DEPENDENCE) do_loops.o
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmultiplex2.F $(PAPILIB) do_loops.o -o fmultiplex2 $(LDFLAGS)

clean:
	rm -f *.o *.stderr *.stdout core *~ $(TESTS)

install: default
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/ftests
	chmod go+rx $(DESTDIR)/ftests
	find . -perm -100 -type f -exec cp {} $(DESTDIR)/ftests \;
	chmod go+rx $(DESTDIR)/ftests/*
	find . -name "*.F" -type f -exec cp {} $(DESTDIR)/ftests \;
