LIBRARY   = libpapi.a
SHLIB     = libpapi.so
OBJECTS   = papi_fwrappers.o papi.o papi_hl.o $(SUBSTR).o extras.o $(MISCOBJS)
HEADERS   = papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h

all: showconf $(LIBS) fpapi.h examples

showconf:
	@echo; 
	@echo "Host architecture: $(DESCR)"; 
	@echo "Host substrate   : $(SUBSTR).c"; 
	@echo "Host examples    : $(TARGETS)";
	@echo

static: $(LIBRARY)

$(LIBRARY): $(OBJECTS) 
	ar ruv $(LIBRARY) $(OBJECTS) 

shared: $(SHLIB)

$(SHLIB): $(OBJECTS) 
	-$(CC_SHR) -o $(SHLIB) $(OBJECTS)

papi_fwrappers.o: papi_fwrappers.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(FOR_USED) -c papi_fwrappers.c -o papi_fwrappers.o
papi.o: papi.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi.c -o papi.o
papi_hl.o: papi_hl.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_hl.c -o papi_hl.o
extras.o: extras.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c extras.c -o extras.o
$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o
genpapifdef: genpapifdef.c $(LIBRARY)
	$(CC) $(CFLAGS) $(OPTFLAGS) genpapifdef.c -o genpapifdef $(LIBRARY) $(LDFLAGS)
fpapi.h: genpapifdef
	./genpapifdef > fpapi.h

examples: 
	cd tests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" OPTFLAGS="$(OPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" NOOPT="$(NOOPT)" $(TARGETS)
	cd ftests; $(MAKE) F77="$(F77)" OPTFLAGS="$(OPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" NOOPT="$(NOOPT)" 


libclean:
	rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations

clean: libclean
	rm -rf $(LIBRARY) $(SHLIB) $(SHLIB).1 $(SHLIB).1.0 genpapifdef
	cd tests; $(MAKE) clean
	cd ftests; $(MAKE) clean

dist:
	chmod -R 644 *
	chgrp -R papi *
	cd ..; tar cvf papisrc.tar src; gzip papisrc.tar

install: all
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/lib
	chmod go+rx $(DESTDIR)/lib
	mkdir -p $(DESTDIR)/include
	chmod go+rx $(DESTDIR)/include
	-cp libpapi.a libpapi.so $(DESTDIR)/lib
	chmod go+r $(DESTDIR)/lib/*
	cp papi.h papiStdEventDefs.h $(DESTDIR)/include
	chmod go+r $(DESTDIR)/include/*
	cd ../man; $(MAKE) DESTDIR=$(DESTDIR) install
	cd tests; $(MAKE) DESTDIR=$(DESTDIR) TARGETS="$(TARGETS)" install
	cd ftests; $(MAKE) DESTDIR=$(DESTDIR) TARGETS="$(TARGETS)" install
	cp $(SUBSTR).README $(DESTDIR)