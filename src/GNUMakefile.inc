# Communicate all variables to submakes
export

PAPI_SRCDIR = $(PWD)	
SOURCES	  = $(MISCSRCS) papi.c papi_internal.c papi_hl.c extras.c multiplex.c \
    papi_fwrappers.c papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c \
    papi_data.c threads.c $(MEMSUBSTR)-memory.c $(SUBSTR).c papi_preset.c
OBJECTS   = papi.o papi_internal.o papi_hl.o extras.o multiplex.o \
    papi_fwrappers.o papi_fwrappers_.o papi_fwrappers__.o PAPI_FWRAPPERS.o \
    papi_data.o threads.o $(MEMSUBSTR)-memory.o $(SUBSTR).o papi_preset.o
HEADERS   = papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h papi_preset.h
LIBCFLAGS = $(CFLAGS) -DSUBSTRATE=\"$(SUBSTR).h\"  
FHEADERS  = fpapi.h f77papi.h f90papi.h
PREFIX    = /usr

all: showconf $(LIBS) genpapifdef $(FHEADERS) tests

showconf:
		@echo; 
		@echo "Host architecture: $(DESCR)"; 
		@echo "Host substrate   : $(SUBSTR).c"; 
		@echo "Host examples    : $(TARGETS)";
		@if [ \"$(PREFIX)\" = \"\" ]; then echo "Installation prefix: <unset> (You should do a make PREFIX=/usr/local or equivalent)"; else echo "Installation root: $(PREFIX)"; fi 
		@echo

static: $(LIBRARY)

shared: $(SHLIB)

$(LIBRARY): $(OBJECTS) $(MISCOBJS)
	rm -f $(LIBRARY)
	ar $(ARG64) ruv $@ $^

$(SHLIB): $(SOURCES)
	$(CC_SHR) $(LIBCFLAGS) $(OPTFLAGS) $^ -o $@ $(SHLIBDEPS)

papi_fwrappers.c: $(HEADERS)

papi_fwrappers_.c: papi_fwrappers.c
	$(CPP) -DFORTRANUNDERSCORE $< > $@

papi_fwrappers__.c: papi_fwrappers.c
	$(CPP) -DFORTRANDOUBLEUNDERSCORE $< > $@

PAPI_FWRAPPERS.c: papi_fwrappers.c
	$(CPP) -DFORTRANCAPS $< > $@

$(OBJECTS): %.o: %.c $(HEADERS) $(MISCHDRS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c $< 

genpapifdef: genpapifdef.c $(LIBRARY)
	$(CC) $(CFLAGS) $(OPTFLAGS) $< -o $@ $(LIBRARY) $(LDFLAGS)

fpapi.h: genpapifdef
	$< -c > $@

f77papi.h: genpapifdef
	$< -f77 > $@

f90papi.h: genpapifdef
	$< -f90 > $@

test: ctests
	ctests/zero

tests fulltest: ctests ftests
	sh run_tests.sh

ctests: null
	$(MAKE) -C ctests

ftests: null
	$(MAKE) -C ftests

clean: native_clean
	rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c
	$(MAKE) -C ctests clean
	$(MAKE) -C ftests clean

clobber: clean
	rm -f $(LIBRARY) $(SHLIB) $(EXTRALIBS) $(FHEADERS)

null:

dist: 
	$(MAKE) -f Makefile.$(MSUBSTR) install-all PREFIX=`pwd`/papi-$(SUBSTR)
	tar cfv ./papi-$(SUBSTR).tar ./papi-$(SUBSTR)
	gzip ./papi-$(SUBSTR).tar
	rm -rf ./papi-$(SUBSTR)

install-all: install-tree install install-man install-tests

install: install-tree native_install
	cp $(LIBRARY) $(SHLIB) $(PREFIX)/lib
	chmod go+r $(PREFIX)/lib/$(LIBRARY) $(PREFIX)/lib/$(SHLIB)
	cp $(FHEADERS) papi.h papiStdEventDefs.h $(PREFIX)/include
	cd $(PREFIX)/include; chmod go+r $(FHEADERS) papi.h papiStdEventDefs.h 

install-man: install-tree
	cd ../man; $(MAKE) PREFIX=$(PREFIX) install

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
	
