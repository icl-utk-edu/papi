SUBSTR  = linux-x86
DESCR	= "Patched linux 2.x with Intel Pentium Pro, II or III"
TARGET	= libraries

CFLAGS  = -g -Wall
OPTFLAGS= 
#-O6 -mpentiumpro
SMPCFLGS= 
OMPCFLGS= 
LDFLAGS = 
CC_R	= cc
CC	= cc

LIBRARY = libpapi.a
PTHRLIB = libpapi_r.a
PTHRLIBS= -lpthread
SHLIB   = libpapi.so
OBJECTS = papi.o papi_hl.o $(SUBSTR).o extras.o
PTHROBS = papi_r.o papi_hl_r.o $(SUBSTR)_r.o extras_r.o
HEADERS = papi.h papi_internal.h papiStdEventDefs.h $(SUBSTR).h
TESTS	= serial
# pthreads smp omp 

all: showconf $(TARGET) examples

showconf:
	@echo; echo "Host architecture: $(DESCR)"; 
	@echo "Host substrate   : $(SUBSTR).c"; echo

libraries: $(LIBRARY) $(PTHRLIB) 

$(LIBRARY): $(OBJECTS)
	ar ruv $(LIBRARY) $(OBJECTS)

$(PTHRLIB): $(PTHROBS)
	ar ruv $(PTHRLIB) $(PTHROBS)

papi.c papi_hl.c extras.c: papi.h papi_internal.h

$(SUBSTR).c: papi.h papi_internal.h $(SUBSTR).h

papi_r.o: papi.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c papi.c -o papi_r.o
papi_hl_r.o: papi_hl.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c papi_hl.c -o papi_hl_r.o
extras_r.o: extras.c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c extras.c -o extras_r.o
$(SUBSTR)_r.o: $(SUBSTR).c $(HEADERS)
	$(CC_R) $(CFLAGS) $(OPTFLAGS) -DPTHREADS -c $(SUBSTR).c -o $(SUBSTR)_r.o
papi.o: papi.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi.c -o papi.o
papi_hl.o: papi_hl.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c papi_hl.c -o papi_hl.o
extras.o: extras.c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c extras.c -o extras.o
$(SUBSTR).o: $(SUBSTR).c $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(SUBSTR).c -o $(SUBSTR).o

shared: library
	rm -f *.o
	$(CC) -c $(SHCFLGS) papi.c papi_hl.c extras.c $(SUBSTR).c
	$(CC) $(SHLFLGS) $(OBJECTS)
	ln -s $(SHLIB).1 $(SHLIB)
	ln -s $(SHLIB) $(SHLIB).1.0

examples: 
	cd tests; $(MAKE) CC="$(CC)" CC_R="$(CC_R)" OPTFLAGS="$(OPTFLAGS)" SMPCFLGS="$(SMPCFLGS)" OMPCFLGS="$(OMPCFLGS)" CFLAGS="$(CFLAGS) -I.." LDFLAGS="$(LDFLAGS)" PTHRLIBS="$(PTHRLIBS)" $(TESTS)

libclean:
	rm -rf $(PTHROBS) $(OBJECTS) core *~

clean: libclean
	rm -rf $(PTHRLIB) $(LIBRARY) $(SHLIB) $(SHLIB).1 $(SHLIB).1.0 
	cd tests; $(MAKE) clean

dist:
	chmod -R 644 *
	chgrp -R papi *
	cd ..; tar cvf papisrc.tar src; gzip papisrc.tar
