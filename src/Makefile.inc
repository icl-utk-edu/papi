PAPI_SRCDIR = $(PWD)	
SOURCES	  = $(MISCSRCS) papi.c papi_internal.c papi_hl.c extras.c multiplex.c \
    papi_fwrappers.c papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c \
    papi_data.c threads.c $(MEMSUBSTR)-memory.c $(SUBSTR).c
OBJECTS   = $(MISCOBJS) papi.o papi_internal.o papi_hl.o extras.o multiplex.o \
    papi_fwrappers.o papi_fwrappers_.o papi_fwrappers__.o PAPI_FWRAPPERS.o \
    papi_data.o threads.o $(MEMSUBSTR)-memory.o $(SUBSTR).o $(ALLOCATE)
HEADERS   = $(MISCHDRS) papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h
LIBCFLAGS = $(CFLAGS) -DSUBSTRATE=\"$(SUBSTR).h\"  
FHEADERS= fpapi.h f77papi.h f90papi.h

all: showconf $(LIBS) genpapifdef $(FHEADERS) examples

showconf:
		@echo; 
		@echo "Host architecture: $(DESCR)"; 
		@echo "Host substrate   : $(SUBSTR).c"; 
		@echo "Host examples    : $(TARGETS)";
		@if [ \"$(DESTDIR)\" = \"\" ]; then echo "Installation root: <unset> (You should do a make DESTDIR=/usr/local or equivalent)"; else echo "Installation root: $(DESTDIR)"; fi 
		@echo

static: $(LIBRARY)

shared: $(SHLIB)

$(LIBRARY): $(OBJECTS) 
	rm -f $(LIBRARY)
	ar $(ARG64) ruv $(LIBRARY) $(OBJECTS) 

$(SHLIB): $(SOURCES) $(SHLIBOBJS) 
	$(CC_SHR) $(LIBCFLAGS) $(OPTFLAGS) $(SOURCES) $(SHLIBOBJS) -o $@ $(SHLIBDEPS)

papi_fwrappers_.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANUNDERSCORE papi_fwrappers.c > papi_fwrappers_.c 

papi_fwrappers__.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANDOUBLEUNDERSCORE papi_fwrappers.c > papi_fwrappers__.c

PAPI_FWRAPPERS.c: papi_fwrappers.c $(HEADERS)
	$(CPP) -DFORTRANCAPS papi_fwrappers.c > PAPI_FWRAPPERS.c

papi_fwrappers.o: papi_fwrappers.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_fwrappers.c -o papi_fwrappers.o

papi_fwrappers_.o: papi_fwrappers_.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_fwrappers_.c -o papi_fwrappers_.o 

papi_fwrappers__.o: papi_fwrappers__.c $(HEADERS) 
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_fwrappers__.c -o papi_fwrappers__.o

PAPI_FWRAPPERS.o: PAPI_FWRAPPERS.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c PAPI_FWRAPPERS.c -o PAPI_FWRAPPERS.o

papi.o: papi.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi.c -o papi.o 

papi_internal.o: papi_internal.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_internal.c -o papi_internal.o 

papi_data.o: papi_data.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_data.c -o papi_data.o 

threads.o: threads.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c threads.c -o threads.o 

papi_hl.o: papi_hl.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_hl.c -o papi_hl.o 

$(MEMSUBSTR)-memory.o: $(MEMSUBSTR)-memory.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c $(MEMSUBSTR)-memory.c -o $(MEMSUBSTR)-memory.o

allocate.o: allocate.c
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c allocate.c -o allocate.o

extras.o: extras.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c extras.c -o extras.o 

multiplex.o: multiplex.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c multiplex.c -o multiplex.o 

$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o 

genpapifdef: genpapifdef.c $(LIBRARY)
	$(CC) $(CFLAGS) $(OPTFLAGS) genpapifdef.c -o genpapifdef $(LIBRARY) $(LDFLAGS)

fpapi.h: genpapifdef
	./genpapifdef -c > fpapi.h

f77papi.h: genpapifdef
	./genpapifdef -f77 > f77papi.h

f90papi.h: genpapifdef
	./genpapifdef -f90 > f90papi.h

examples: tests ftests

test: examples
	ctests/zero

fulltest: examples
	sh run_tests.sh

tests: null
	cd ctests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" CFLAGS="$(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" $(TARGETS) TOOLPATH="$(TOOLPATH)"

# LIBCFLAGS='$(CFLAGS)'
ftests: null
	cd ftests; $(MAKE) CC="$(CC)" F77="$(F77)" TOPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" CFLAGS="$(CFLAGS)" SUBSTR="$(SUBSTR)"

clean: native_clean
	rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c
	cd ctests; $(MAKE) clean
	cd ftests; $(MAKE) clean

clobber: clean
	rm -f $(LIBRARY) $(SHLIB) $(EXTRALIBS) $(FHEADERS)
null:

dist: 
	$(MAKE) -f Makefile.$(MSUBSTR) install DESTDIR=`pwd`/papi-$(SUBSTR)
	tar cfv ./papi-$(SUBSTR).tar ./papi-$(SUBSTR)
	gzip ./papi-$(SUBSTR).tar
	rm -rf ./papi-$(SUBSTR)

install: all install_tree native_install
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	-chmod go+rx $(DESTDIR)/lib
	-chmod go+rx $(DESTDIR)/include
	-cp $(LIBRARY) $(SHLIB) $(DESTDIR)/lib
	-chmod go+r $(DESTDIR)/lib/*
	-cp $(FHEADERS) papi.h papiStdEventDefs.h $(DESTDIR)/include
	-chmod go+r $(DESTDIR)/include/*
	-cd ../man; $(MAKE) DESTDIR=$(DESTDIR) install
	cd ctests; $(MAKE) DESTDIR=$(DESTDIR) CC="$(CC)" CC_R="$(CC_R)" MPICC="$(MPICC)" TOPTFLAGS="$(TOPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" LIBCFLAGS='$(LIBCFLAGS)' TARGETS="$(TARGETS)" LIBRARY="$(LIBRARY)" install
	cd ftests; $(MAKE) DESTDIR=$(DESTDIR) F77="$(F77)" CC="$(CC)" OPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" SUBSTR="$(SUBSTR)" CFLAGS="$(CFLAGS)" TARGETS="$(TARGETS)" LIBRARY="$(LIBRARY)" install
	-cp $(SUBSTR).README $(DESTDIR)

install_tree:
	-mkdir -p $(DESTDIR)/lib
	-mkdir -p $(DESTDIR)/include
	-mkdir -p $(DESTDIR)/man
	-mkdir -p $(DESTDIR)/ctests
	-mkdir -p $(DESTDIR)/ftests
