INCLUDE = -I..
TESTS 	= avail case1 case2 clockres eventname fmatrixlowpapi \
          fmatrixpapi flops
PAPILIB	= ../libpapi.a

default: $(TESTS)

clockres: clockres.F $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) clockres.F -L.. -lpapi $(LDFLAGS) -o clockres

avail: avail.F $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) avail.F -L.. -lpapi $(LDFLAGS) -o avail

eventname: eventname.F $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) eventname.F -L.. -lpapi $(LDFLAGS) -o eventname

case1: case1.F $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case1.F -L.. -lpapi $(LDFLAGS) -o case1

case2: case2.F $(PAPILIB)
	$(F77) $(NOOPT) $(FFLAGS) $(INCLUDE) case2.F -L.. -lpapi $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F $(PAPILIB)
	$(F77) $(OPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F -L.. -lpapi $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F $(PAPILIB)
	$(F77) $(OPTFLAGS) $(FFLAGS) $(INCLUDE) fmatrixpapi.F -L.. -lpapi $(LDFLAGS) -o fmatrixpapi

flops: flops.F $(PAPILIB)
	$(F77) $(OPTFLAGS) $(FFLAGS) $(INCLUDE) flops.F -L.. -lpapi $(LDFLAGS) -o flops

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
