INCLUDE = -I.
LIBRARY = -lpapi
PAPILIB = $(LIBRARY)
CC	= gcc
CC_R	= $(CC) -pthread
CFLAGS	= -g -O -Wall

UTILSRCS:= do_loops.c test_utils.c dummy.c clockcore.c
UTILOBJS:=$(UTILSRCS:.c=.o)
ifneq ($(F77),)
UTILSRCS+= ftests_util.F
UTILOBJS+= ftests_util.o
endif

all:	$(UTILOBJS)
	$(AR) $(ARG64) rv libtestlib.a $(UTILOBJS)

do_loops.o: do_loops.c papi_test.h test_utils.h
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c do_loops.c

clockcore.o: clockcore.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c clockcore.c

dummy.o: dummy.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c dummy.c

test_utils.o: test_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c test_utils.c

ftests_util.o: ftests_util.F fpapi_test.h 
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) -c ftests_util.F

clean:
	rm -f *.o *genmod.f90 *genmod.mod *.stderr *.stdout core *~ $(ALL) libtestlib.a libtestlib.so

install: 
	@echo "Papi testlib (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/testlib
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/testlib
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/testlib \;
	-chmod go+rx $(DATADIR)/testlib/*
	-find . -name "*.[chaF]" -type f -exec cp {} $(DATADIR)/testlib \;
	-cp Makefile.target $(DATADIR)/testlib/Makefile
