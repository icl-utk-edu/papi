TARGETS = avail case1 case2 clockres eventname fmatrixlowpapi \
          fmatrixpapi
ALL: $(TARGETS)
default: $(TARGETS)


clockres: clockres.F ../libpapi.a
	$(F77) $(FFLAGS) clockres.F ../libpapi.a $(LDFLAGS) -o clockres

avail: avail.F ../libpapi.a
	$(F77) $(FFLAGS) avail.F ../libpapi.a $(LDFLAGS) -o avail

eventname: eventname.F ../libpapi.a
	$(F77) $(FFLAGS) eventname.F ../libpapi.a $(LDFLAGS) -o eventname

case1: case1.F ../libpapi.a
	$(F77) $(FFLAGS) case1.F ../libpapi.a $(LDFLAGS) -o case1

case2: case2.F ../libpapi.a
	$(F77) $(FFLAGS) case2.F ../libpapi.a $(LDFLAGS) -o case2

fmatrixlowpapi: fmatrixlowpapi.F ../libpapi.a
	$(F77) $(FFLAGS) fmatrixlowpapi.F ../libpapi.a $(LDFLAGS) -o fmatrixlowpapi

fmatrixpapi: fmatrixpapi.F ../libpapi.a
	$(F77) $(FFLAGS) fmatrixpapi.F ../libpapi.a $(LDFLAGS) -o fmatrixpapi

clean:
	rm -f *.o $(TARGETS) core *~
