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
	rm -rf $(OBJECTS) core rii_files genpapifdef *~

clean: libclean
	rm -rf $(LIBRARY) $(SHLIB) $(SHLIB).1 $(SHLIB).1.0 genpapifdef
	cd tests; $(MAKE) clean
	cd ftests; $(MAKE) clean

dist:
	chmod -R 644 *
	chgrp -R papi *
	cd ..; tar cvf papisrc.tar src; gzip papisrc.tar
