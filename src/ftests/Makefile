INCLUDE = -I..
TESTS 	= avail case1 case2 clockres eventname fmatrixlowpapi \
          fmatrixpapi fmatrixpapi2 flops strtest
PAPILIB	= ../libpapi.a

default: $(TESTS)

ftests_util.o: ftests_util.F
	$(F77) $(NOOPT) -c $(FFLAGS) $(INCLUDE) ftests_util.F

clockres: clockres.F ftests_util.o $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) clockres.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o clockres

avail: avail.F ftests_util.o $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) avail.F $(PAPILIB) $(LDFLAGS) ftests_util.o -o avail

eventname: eventname.F ftests_util.o $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) eventname.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.F ftests_util.o $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case1.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.F ftests_util.o $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case2.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F ftests_util.o $(PAPILIB)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F ftests_util.o $(PAPILIB)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o fmatrixpapi

fmatrixpapi2: fmatrixpapi2.F $(PAPILIB)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi2.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o fmatrixpapi2

strtest: strtest.F $(PAPILIB)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) strtest.F $(PAPILIB) $(LDFLAGS) -o strtest

flops: flops.F ftests_util.o $(PAPILIB)
	$(F77) $(FOPTFLAGS) $(FFLAGS) $(INCLUDE) flops.F ftests_util.o $(PAPILIB) $(LDFLAGS) -o flops

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
