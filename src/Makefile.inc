PAPI_SRCDIR = $(PWD)	
SOURCES	  = $(MISCSRCS) papi.c papi_internal.c papi_hl.c extras.c multiplex.c \
    papi_fwrappers.c papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c \
    papi_data.c threads.c $(MEMSUBSTR)-memory.c $(SUBSTR).c papi_preset.c \
    papi_vector.c
OBJECTS   = $(MISCOBJS) papi.o papi_internal.o papi_hl.o extras.o multiplex.o \
    papi_fwrappers.o papi_fwrappers_.o papi_fwrappers__.o PAPI_FWRAPPERS.o \
    papi_data.o threads.o $(MEMSUBSTR)-memory.o $(SUBSTR).o papi_preset.o \
    papi_vector.o
HEADERS   = $(MISCHDRS) papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h \
    papi_preset.h threads.h papi_protos.h papi_vector.h papi_vector_redefine.h
LIBCFLAGS = $(CFLAGS) -DSUBSTRATE=\"$(SUBSTR).h\"  
FHEADERS  = fpapi.h f77papi.h f90papi.h

all: showconf $(LIBS) genpapifdef $(FHEADERS) utils tests

showconf:
		@echo; 
		@echo "Host architecture: $(DESCR)"; 
		@echo "Host substrate   : $(SUBSTR).c"; 
		@echo "Host examples    : $(TARGETS)";
		@if [ \"$(PREFIX)\" = \"\" ]; then echo "Installation prefix: <unset> (You should do a make PREFIX=/usr/local or equivalent)"; else echo "Installation root: $(PREFIX)"; fi 
		@echo

static: $(LIBRARY)

shared: $(SHLIB)

$(LIBRARY): $(OBJECTS) 
	rm -f $(LIBRARY)
	ar $(ARG64) ruv $(LIBRARY) $(OBJECTS) 

$(SHLIB): $(SOURCES) $(SHLIBOBJS) 
	$(CC_SHR) $(LIBCFLAGS) $(OPTFLAGS) $(SOURCES) $(SHLIBOBJS) -o $@ $(SHLIBDEPS)
	-$(POST_BUILD)

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

extras.o: extras.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c extras.c -o extras.o 

papi_vector.o: papi_vector.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_vector.c -o papi_vector.o

papi_preset.o: papi_preset.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_preset.c -o papi_preset.o 

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

test: ctests
	ctests/zero

fulltest: tests
	sh run_tests.sh

tests: ctests ftests

utils: null
	-cd utils; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" CFLAGS="$(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" util TOOLPATH="$(TOOLPATH)"

ctests: null
	cd ctests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" CFLAGS="$(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" $(TARGETS) TOOLPATH="$(TOOLPATH)"

# LIBCFLAGS='$(CFLAGS)'
ftests: null
	-cd ftests; $(MAKE) CC="$(CC)" F77="$(F77)" TOPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" CFLAGS="$(CFLAGS)" SUBSTR="$(SUBSTR)"

clean: native_clean
	-rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c
	-cd ctests; $(MAKE) clean
	-cd ftests; $(MAKE) clean
	-cd utils;  $(MAKE) clean

clobber: clean native_clobber
	-rm -f $(LIBRARY) $(SHLIB) $(EXTRALIBS) $(FHEADERS) $(SPECIAL_REMOVE)
	-cd utils;  $(MAKE) clobber
null:

dist: 
	$(MAKE) -f Makefile.$(MSUBSTR) install-all PREFIX=`pwd`/papi-$(SUBSTR)
	tar cfv ./papi-$(SUBSTR).tar ./papi-$(SUBSTR)
	gzip ./papi-$(SUBSTR).tar
	rm -rf ./papi-$(SUBSTR)

install-all: install install-tests

install: install-tree install-man install-lib install-utils

install-lib: install-tree native_install
	cp $(FHEADERS) papi.h papiStdEventDefs.h $(PREFIX)/include
	cd $(PREFIX)/include; chmod go+r $(FHEADERS) papi.h papiStdEventDefs.h 
	cp $(LIBRARY) $(SHLIB) $(PREFIX)/lib
	chmod go+r $(PREFIX)/lib/$(LIBRARY) $(PREFIX)/lib/$(SHLIB)

install-man: install-tree
	cd ../man && $(MAKE) PREFIX=$(PREFIX) install

install-utils: install-tree
	cd utils; $(MAKE) PREFIX=$(PREFIX) CC="$(CC)" CC_R="$(CC_R)" MPICC="$(MPICC)" TOPTFLAGS="$(TOPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" LIBCFLAGS='$(LIBCFLAGS)' LIBRARY="$(LIBRARY)" install

install-tests: install-tree
	cd ctests; $(MAKE) PREFIX=$(PREFIX) CC="$(CC)" CC_R="$(CC_R)" MPICC="$(MPICC)" TOPTFLAGS="$(TOPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" LIBCFLAGS='$(LIBCFLAGS)' TARGETS="$(TARGETS)" LIBRARY="$(LIBRARY)" install
	cd ftests; $(MAKE) PREFIX=$(PREFIX) F77="$(F77)" CC="$(CC)" OPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" LIBRARY="$(LIBRARY)" SUBSTR="$(SUBSTR)" CFLAGS="$(CFLAGS)" TARGETS="$(TARGETS)" LIBRARY="$(LIBRARY)" install

install-tree:
	@if [ \"$(PREFIX)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make PREFIX=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(PREFIX)\""; 
	-mkdir -p $(PREFIX)/lib
	# chmod go+rx $(PREFIX)/lib
	-mkdir -p $(PREFIX)/include
	# chmod go+rx $(PREFIX)/include
	-mkdir -p $(PREFIX)/man
	# chmod go+rx $(PREFIX)/man
	-mkdir -p $(PREFIX)/share/papi
	# chmod go+rx $(PREFIX)/share/papi
	
