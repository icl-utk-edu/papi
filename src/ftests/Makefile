INCLUDE = -I..
TESTS 	= avail case1 case2 clockres eventname fmatrixlowpapi \
          fmatrixpapi

default: $(TESTS)

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
	rm -f *.o $(TESTS) core *~

install: default
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/ftests
	chmod go+rx $(DESTDIR)/ftests
	find . -perm -1 -type f -exec cp {} $(DESTDIR)/ftests \;
	cp fmt1 fmt2 $(DESTDIR)/ftests
	chmod go+rx $(DESTDIR)/ftests/*
