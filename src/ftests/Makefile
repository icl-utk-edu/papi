CC = gcc
FOR_USED = -DFORTRANUNDERSCORE
CFLAGS = -O -DONT_WRAP -g $(FOR_USED)

F77 = f77

INCLUDE = -I..
TESTS 	= zero first \
	  second third fourth fifth nineth tenth profile description \
          accum highlevel cost johnmay johnmay2 native profile \
	  avail case1 case2 clockres eventname fmatrixlowpapi \
	  fmatrixpapi fmatrixpapi2 flops strtest

FTEST_DEPENDENCE=ftests_wrappers.o ftests_util.o
PAPILIB=ftests_wrappers.o ftests_util.o ../libpapi.a -lcpc

default: $(TESTS)

ftests_wrappers.o: ftests_wrappers.c
	$(CC) $(INCLUDE) $(CFLAGS) -c ftests_wrappers.c -o ftests_wrappers.o

ftests_util.o: ftests_util.F
	$(F77) $(NOOPT) -c $(FFLAGS) $(INCLUDE) ftests_util.F

clockres: clockres.F $(FTEST_DEPENDENCE) 
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) clockres.F $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F $(FTEST_DEPENDENCE)  
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) avail.F  $(PAPILIB) $(LDFLAGS) -o avail

eventname: eventname.F $(FTEST_DEPENDENCE)  
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) eventname.F $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F $(FTEST_DEPENDENCE) 
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case1.F $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F $(FTEST_DEPENDENCE)  
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case2.F $(PAPILIB) $(LDFLAGS) -o case2

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
	$(F77) $(INCLUDE) description.F $(PAPILIB) -o description

accum: accum.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) accum.F $(PAPILIB) -o accum

highlevel: highlevel.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) highlevel.F $(PAPILIB) -o highlevel

zero: zero.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) zero.F $(PAPILIB) -o zero

first: first.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) first.F $(PAPILIB) -o first

second: second.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) second.F $(PAPILIB) -o second

third: third.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) third.F $(PAPILIB) -o third

fourth: fourth.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) fourth.F $(PAPILIB) -o fourth

fifth: fifth.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) fifth.F $(PAPILIB) -o fifth

nineth: nineth.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) nineth.F $(PAPILIB) -o nineth

tenth: tenth.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) tenth.F $(PAPILIB) -o tenth

cost: cost.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) cost.F $(PAPILIB) -o cost

johnmay: johnmay.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) johnmay.F $(PAPILIB) -o johnmay

johnmay2: johnmay2.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) johnmay2.F $(PAPILIB) -o johnmay2

native: native.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) native.F $(PAPILIB) -o native

profile: profile.F $(FTEST_DEPENDENCE)
	$(F77) $(INCLUDE) profile.F $(PAPILIB) -o profile

clean:
	rm -f *.o $(TESTS) core *~

install: default
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/ftests
	chmod go+rx $(DESTDIR)/ftests
	find . -perm -100 -type f -exec cp {} $(DESTDIR)/ftests \;
	chmod go+rx $(DESTDIR)/ftests/*
	find . -name "*.F" -type f -exec cp {} $(DESTDIR)/ftests \;
