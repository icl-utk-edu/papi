LIBRARY   = libpapi.a
SHLIB     = libpapi.so
SOURCES	  = $(MISCSRCS) papi.c papi_hl.c $(SUBSTR).c extras.c \
	    papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c
OBJECTS   = $(MISCOBJS) papi.o papi_hl.o $(SUBSTR).o extras.o \
	    papi_fwrappers_.o papi_fwrappers__.o PAPI_FWRAPPERS.o
HEADERS   = $(MISCHDRS) papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h

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
	-$(CC_SHR) $(CFLAGS) $(OPTFLAGS) $(SOURCES) $(MISCOBJS) $(MISCLIBS) -o $@

papi_fwrappers_.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANUNDERSCORE papi_fwrappers.c > papi_fwrappers_.c
papi_fwrappers__.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANDOUBLEUNDERSCORE papi_fwrappers.c > papi_fwrappers__.c
PAPI_FWRAPPERS.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANCAPS papi_fwrappers.c > PAPI_FWRAPPERS.c

papi_fwrappers_.o: papi_fwrappers_.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_fwrappers_.c -o papi_fwrappers_.o
papi_fwrappers__.o: papi_fwrappers__.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_fwrappers__.c -o papi_fwrappers__.o
PAPI_FWRAPPERS.o: PAPI_FWRAPPERS.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c PAPI_FWRAPPERS.c -o PAPI_FWRAPPERS.o
papi.o: papi.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi.c -o papi.o
papi_hl.o: papi_hl.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_hl.c -o papi_hl.o
extras.o: extras.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c extras.c -o extras.o
$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o
genpapifdef: genpapifdef.c $(LIBRARY)
	(CC) $(CFLAGS) $(OPTFLAGS) genpapifdef.c -o genpapifdef libpapi.a $(LDFLAGS)
fpapi.h: genpapifdef
	./genpapifdef > fpapi.h

examples: tests ftests

test: examples
	tests/zero

fulltest: examples
	sh run_tests.sh

tests: null
	cd tests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" OPTFLAGS="$(OPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" NOOPT="$(NOOPT)" $(TARGETS)

ftests: null
	cd ftests; $(MAKE) F77="$(F77)" OPTFLAGS="$(OPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" NOOPT="$(NOOPT)" 


libclean:
	rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c

clean: libclean
	cd tests; $(MAKE) clean
	cd ftests; $(MAKE) clean

clobber: clean
	rm -f $(LIBRARY) $(SHLIB) fpapi.h
null:

dist: 
	$(MAKE) -f Makefile.$(SUBSTR) install DESTDIR=`pwd`/papi-$(SUBSTR)
	tar cfv ./papi-$(SUBSTR).tar ./papi-$(SUBSTR)
	gzip ./papi-$(SUBSTR).tar
	rm -rf ./papi-$(SUBSTR)

install: all
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/lib
	chmod go+rx $(DESTDIR)/lib
	mkdir -p $(DESTDIR)/include
	chmod go+rx $(DESTDIR)/include
	-cp libpapi.a libpapi.so $(DESTDIR)/lib
	chmod go+r $(DESTDIR)/lib/*
	cp fpapi.h papi.h papiStdEventDefs.h $(DESTDIR)/include
	chmod go+r $(DESTDIR)/include/*
	cd ../man; $(MAKE) DESTDIR=$(DESTDIR) install
	cd tests; $(MAKE) DESTDIR=$(DESTDIR) TARGETS="$(TARGETS)" install
	cd ftests; $(MAKE) DESTDIR=$(DESTDIR) TARGETS="$(TARGETS)" install
	cp $(SUBSTR).README $(DESTDIR)
