LIBRARY   = libpapi.a
FLIBRARY  = libfpapi.a
PTHRLIB   = libpapi_r.a
SMPTHRLIB = libpapi_smp.a
OMPTHRLIB = libpapi_omp.a
SHLIB     = libpapi.so
OBJECTS   = papi.o papi_hl.o $(SUBSTR).o extras.o 
FOBJECTS   = papi.o papi_hl.o $(SUBSTR).o extras.o papi_fwrappers.o
PTHROBS   = papi_r.o papi_hl_r.o $(SUBSTR)_r.o extras_r.o
SMPTHROBS = papi_smp.o papi_hl_smp.o $(SUBSTR)_smp.o extras_smp.o 
HEADERS   = papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h

all: showconf $(TARGETS) examples

showconf:
	@echo; 
	@echo "Host architecture: $(DESCR)"; 
	@echo "Host substrate   : $(SUBSTR).c"; 
	@echo "Host libraries   : $(TARGETS)";
	@echo

serial: $(LIBRARY)

pthreads: $(PTHRLIB)

smp: $(SMPTHRLIB)

$(LIBRARY): $(OBJECTS) $(MISCOBJS)
	ar ruv $(LIBRARY) $(OBJECTS) $(MISCOBJS)

$(FLIBRARY): $(FOBJECTS) $(MISCOBJS) 
	ar ruv $(FLIBRARY) $(FOBJECTS) $(MISCOBJS)

$(PTHRLIB): $(PTHROBS) $(MISCOBJS)
	ar ruv $(PTHRLIB) $(PTHROBS) $(MISCOBJS)

$(SMPTHRLIB): $(SMPTHROBS) $(MISCOBJS)
	ar ruv $(SMPTHRLIB) $(SMPTHROBS) $(MISCOBJS)

$(OMPTHRLIB): $(OMPTHROBS) $(MISCOBJS)
	ar ruv $(OMPTHRLIB) $(OMPTHROBS) $(MISCOBJS)

# No Threads

papi_fwrappers.o: papi_fwrappers.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(FOR_USED) -c papi_fwrappers.c -o papi_fwrappers.o
papi.o: papi.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi.c -o papi.o
papi_hl.o: papi_hl.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_hl.c -o papi_hl.o
extras.o: extras.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c extras.c -o extras.o
$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o

# PThreads

papi_r.o: papi.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c papi.c -o papi_r.o
papi_hl_r.o: papi_hl.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c papi_hl.c -o papi_hl_r.o
extras_r.o: extras.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c extras.c -o extras_r.o
$(SUBSTR)_r.o: $(SUBSTR).c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c $(SUBSTR).c -o $(SUBSTR)_r.o

# Native SMP Threads

papi_smp.o: papi.c $(HEADERS)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) -DSMPTHREADS -c papi.c -o papi_smp.o
papi_hl_smp.o: papi_hl.c $(HEADERS)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) -DSMPTHREADS -c papi_hl.c -o papi_hl_smp.o
extras_smp.o: extras.c $(HEADERS)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) -DSMPTHREADS -c extras.c -o extras_smp.o
$(SUBSTR)_smp.o: $(SUBSTR).c $(HEADERS)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) -DSMPTHREADS -c $(SUBSTR).c -o $(SUBSTR)_smp.o

shared: library
	rm -f *.o
	$(CC) -c $(SHCFLGS) papi.c papi_hl.c extras.c $(SUBSTR).c
	$(CC) $(SHLFLGS) $(OBJECTS)
	ln -s $(SHLIB).1 $(SHLIB)
	ln -s $(SHLIB) $(SHLIB).1.0

examples: 
	cd tests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" OPTFLAGS="$(OPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS) -I.." LDFLAGS="$(LDFLAGS)" NOOPT="$(NOOPT)" $(TARGETS)

libclean:
	rm -rf $(SMPTHROBS) $(PTHROBS) $(OBJECTS) $(MISCOBJS) core rii_files *~

clean: libclean
	rm -rf $(SMPTHRLIB) $(PTHRLIB) $(LIBRARY) $(SHLIB) $(SHLIB).1 $(SHLIB).1.0 
	cd tests; $(MAKE) clean

dist:
	chmod -R 644 *
	chgrp -R papi *
	cd ..; tar cvf papisrc.tar src; gzip papisrc.tar
