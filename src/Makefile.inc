PAPI_SRCDIR = $(PWD)	
SOURCES	  = $(MISCSRCS) papi.c papi_internal.c papi_hl.c extras.c multiplex.c \
    papi_fwrappers.c papi_fwrappers_.c papi_fwrappers__.c PAPI_FWRAPPERS.c \
    papi_data.c threads.c $(MEMSUBSTR)-memory.c $(SUBSTR).c papi_preset.c \
    papi_vector.c papi_memory.c $(COMPSRCS)
OBJECTS   = $(MISCOBJS) papi.o papi_internal.o papi_hl.o extras.o multiplex.o \
    papi_fwrappers.o papi_fwrappers_.o papi_fwrappers__.o PAPI_FWRAPPERS.o \
    papi_data.o threads.o $(MEMSUBSTR)-memory.o $(SUBSTR).o papi_preset.o \
    papi_vector.o papi_memory.o $(COMPOBJS)
HEADERS   = $(MISCHDRS) papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h \
    papi_preset.h threads.h papi_protos.h papi_vector.h \
    papi_memory.h config.h $(COMPHDRS)
LIBCFLAGS += -I. $(CFLAGS) -DSUBSTRATE=\"$(SUBSTR).h\" 
FHEADERS  = fpapi.h f77papi.h f90papi.h

all: showconf $(LIBS) utils tests

showconf:
		@echo; 
		@echo "Host architecture: $(DESCR)"; 
		@echo "Host substrate   : $(SUBSTR).c"; 
		@echo "Host examples    : $(TARGETS)";
		@echo "Installation DESTDIR: $(DESTDIR)"; 
		@echo "Installation PREFIX : $(PREFIX)"; 
		@echo "Installation EPREFIX: $(EPREFIX)"; 
		@echo "Installation INCDIR : $(INCDIR)"; 
		@echo "Installation LIBDIR : $(LIBDIR)"; 
		@echo "Installation BINDIR : $(BINDIR)"; 
		@echo "Installation MANDIR : $(MANDIR)"; 
		@echo "Installation DOCDIR : $(DOCDIR)"; 
		@echo "Installation DATADIR: $(DATADIR)"; 
		@echo

static: $(LIBRARY)

shared: $(SHLIB)

$(LIBRARY): $(OBJECTS) 
	rm -f $(LIBRARY)
	$(AR) $(ARG64) ruv $(LIBRARY) $(OBJECTS) 

$(SHLIB): $(SOURCES) $(SHLIBOBJS) 
	rm -f $(SHLIB)
	$(CC_SHR) $(LIBCFLAGS) $(OPTFLAGS) $(SOURCES) $(SHLIBOBJS) -o $@ $(LDFLAGS) $(SHLIBDEPS) 
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

papi_memory.o: papi_memory.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_memory.c -o papi_memory.o

papi_vector.o: papi_vector.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_vector.c -o papi_vector.o

papi_preset.o: papi_preset.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c papi_preset.c -o papi_preset.o 

multiplex.o: multiplex.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c multiplex.c -o multiplex.o 

$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o 

linux-mx.o: linux-mx.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c linux-mx.c -o linux-mx.o 

linux-acpi.o: linux-acpi.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c linux-acpi.c -o linux-acpi.o 

linux-net.o: linux-net.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c linux-net.c -o linux-net.o 

linux-lmsensors.o: linux-lmsensors.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c linux-lmsensors.c -o linux-lmsensors.o 


#
# This is all done by configure now
#
genpapifdef: genpapifdef.c papi.h papi_preset.h papi_data.c
	$(CC) -I. genpapifdef.c -o genpapifdef

fpapi.h: papi.h
	./genpapifdef -c > fpapi.h

f77papi.h: papi.h
	./genpapifdef -f77 > f77papi.h

f90papi.h: papi.h
	./genpapifdef -f90 > f90papi.h

test: ctests
	$(SETPATH) ctests/zero

fulltest: tests
	sh run_tests.sh

tests: $(TESTS)

utils: $(LIBRARY) null
	$(SETPATH) cd utils; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY="$(LINKLIB)" util

ctests: $(LIBRARY) null
	$(SETPATH) cd ctests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY="$(LINKLIB)" $(TARGETS)

ftests: $(LIBRARY) null
	$(SETPATH) cd ftests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" F77="$(F77)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="-I.. $(FFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY="$(LINKLIB)" $(TARGETS)

clean: native_clean
	-rm -rf $(OBJECTS) core rii_files genpapifdef *~ so_locations papi_fwrappers_.c	papi_fwrappers__.c PAPI_FWRAPPERS.c perfmon_events_table.h
	-cd ctests; $(MAKE) clean
	-cd ftests; $(MAKE) clean
	-cd utils;  $(MAKE) clean

clobber distclean: clean native_clobber
	-rm -f $(LIBRARY) $(SHLIB) $(EXTRALIBS) $(SPECIAL_REMOVE) Makefile config.h libpapi.so
	-cd utils;  $(MAKE) clobber
	-rm -f config.log config.status f77papi.h f90papi.h fpapi.h

null:

dist: 
	$(MAKE) -f Makefile.$(MSUBSTR) install-all PREFIX=`pwd`/papi-$(SUBSTR)
	tar cfv ./papi-$(SUBSTR).tar ./papi-$(SUBSTR)
	gzip ./papi-$(SUBSTR).tar
	rm -rf ./papi-$(SUBSTR)

install-all: install install-tests

install: install-lib install-man install-utils

install-lib: native_install
	@echo "Headers (INCDIR) being installed in: \"$(DESTDIR)$(INCDIR)\""; 
	-mkdir -p $(DESTDIR)$(INCDIR)
	-chmod go+rx $(DESTDIR)$(INCDIR)
	cp $(FHEADERS) papi.h papiStdEventDefs.h $(DESTDIR)$(INCDIR)
	cd $(DESTDIR)$(INCDIR); chmod go+r $(FHEADERS) papi.h papiStdEventDefs.h
	@echo "Libraries (LIBDIR) being installed in: \"$(DESTDIR)$(LIBDIR)\""; 
	-mkdir -p $(DESTDIR)$(LIBDIR)
	-chmod go+rx $(DESTDIR)$(LIBDIR)
	cp $(LIBRARY) $(DESTDIR)$(LIBDIR)
	chmod go+r $(DESTDIR)$(LIBDIR)/$(LIBRARY)
# copy SHLIB conditionally, since it isn't always made
	if [[ -r $(SHLIB) ]] ; then \
		cp -p $(SHLIB) $(DESTDIR)$(LIBDIR)/libpapi.so.$(PAPIVER).$(PAPIREV).$(PAPIAGE); \
		chmod go+r $(DESTDIR)$(LIBDIR)/libpapi.so.$(PAPIVER).$(PAPIREV).$(PAPIAGE) ; \
		cd $(DESTDIR)$(LIBDIR); ln -sf libpapi.so.$(PAPIVER).$(PAPIREV).$(PAPIAGE) libpapi.so.$(PAPIVER); \
		cd $(DESTDIR)$(LIBDIR); ln -sf libpapi.so.$(PAPIVER).$(PAPIREV).$(PAPIAGE) libpapi.so; \
	fi

install-man: 
	cd ../man && $(MAKE) DOCDIR=$(DESTDIR)$(DOCDIR) MANDIR=$(DESTDIR)$(MANDIR) install

install-utils: 
	$(SETPATH) cd utils; $(MAKE) BINDIR="$(DESTDIR)$(BINDIR)" CC="$(CC)" CC_R="$(CC_R)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY=$(LINKLIB) install
	
install-tests: 
	$(SETPATH) cd ctests; $(MAKE) DATADIR="$(DESTDIR)$(DATADIR)" CC="$(CC)" CC_R="$(CC_R)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY=$(LINKLIB) install
	$(SETPATH) cd ftests; $(MAKE) DATADIR="$(DESTDIR)$(DATADIR)" CC="$(CC)" CC_R="$(CC_R)" F77="$(F77)" CFLAGS="-I.. $(CFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" TOPTFLAGS="$(TOPTFLAGS)" FTOPTFLAGS="$(FTOPTFLAGS)" FFLAGS="-I.. $(FFLAGS)" NOOPT="$(NOOPT)" LDFLAGS="$(LDFLAGS) $(STATIC)" LIBRARY=$(LINKLIB) install
