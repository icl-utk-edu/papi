INCLUDE = -I..
TARGETS = avail case1 case2 clockres eventname fmatrixlowpapi \
          fmatrixpapi

default: $(TARGETS)

clockres: clockres.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) clockres.F ../libpapi.a $(LDFLAGS) -o clockres

avail: avail.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) avail.F ../libpapi.a $(LDFLAGS) -o avail

eventname: eventname.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) eventname.F ../libpapi.a $(LDFLAGS) -o eventname

case1: case1.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) case1.F ../libpapi.a $(LDFLAGS) -o case1

case2: case2.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) case2.F ../libpapi.a $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) fmatrixlowpapi.F ../libpapi.a $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F ../libpapi.a
	$(F77) $(FFLAGS) $(INCLUDE) fmatrixpapi.F ../libpapi.a $(LDFLAGS) -o fmatrixpapi

clean:
	rm -f *.o $(TARGETS) core *~
