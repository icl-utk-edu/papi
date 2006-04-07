OMP	= 
SMP	= 
SHMEM	= 
PTHREADS= 
MPX	= 
MPXPTHR	= 
MPI	= 

BLRTS_GNU_ROOT = /bgl/BlueLight/ppcfloor/blrts-gnu
BGLSYS_ROOT    = /bgl/BlueLight/ppcfloor/bglsys
#PREFIX         = /home/noeth2/bgl-papi/papi3/bglsys
PREFIX         = ../..
LIBRARY = libpapi.a
 
CC      = $(SILENT) $(BLRTS_GNU_ROOT)/bin/powerpc-bgl-blrts-gnu-gcc
F77     = $(SILENT) $(BLRTS_GNU_ROOT)/bin/powerpc-bgl-blrts-gnu-g77
CFLAGS  = -D_BGL -g -gdwarf-2 -O2 -Wall -I$(BGLSYS_ROOT)/include
FFLAGS  = -g -gdwarf-2 -O2 -Wall -I$(BGLSYS_ROOT)/include -Dlinux

SERIAL  = hummermult.rts hummermult-inf.rts test_bgl_perfctr.rts \
	cachetest.rts cachetest-vn.rts mpi_bcast_papi.rts test_bgl_perfctr-overf.rts
### This functionality is not present in BG/L: test_bgl_perfctr-overf.rts
BROKEN	= inherit pernode
UTILOBJS= 
ALL 	= $(SERIAL) $(MPI)
#ALL 	= $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL) 
INCLUDE = -I../.. #-I/bgl/BlueLight/ppcfloor/bglsys/include
PAPILIB	= ../../$(LIBRARY) #/bgl/BlueLight/ppcfloor/bglsys/lib/librts.rts.a /bgl/BlueLight/ppcfloor/bglsys/lib/libbgl_perfctr.rts.a /bgl/BlueLight/ppcfloor/bglsys/lib/libdevices.440.a
LDFLAGS = -L$(BGLSYS_ROOT)/lib -lbgl_perfctr.rts -lmpich.rts -lmsglayer.rts -lrts.rts -ldevices.rts

.SUFFIXES: .rts.o
.c.rts.o:
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

all: $(ALL)

mpi: $(MPI)

multiplex_and_pthreads: $(MPXPTHR) $(MPX) $(PTHREADS)

multiplex: $(MPX) 

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

run: 
	sh ./run_tests.sh

mpi_bcast_papi.rts: mpi_bcast_papi.rts.o $(PAPILIB)
	$(CC) -o $@ $^ $(LIBS_MPI) $(LIBS_MPI) $(LDFLAGS) $(PAPILIB)

mpi_bcast_papi.rts.o: mpi_bcast_papi.c
	$(CC) -c -o $@ $(CFLAGS) $(INCLUDE) $< $

cachetest.rts: cachetest.rts.o $(PAPILIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

cachetest-vn.rts: cachetest-vn.rts.o $(PAPILIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

cachetest-vn.rts.o: cachetest.c
	$(CC) -DVN $(CFLAGS) $(INCLUDE) -c $< -o $@

hummermult-inf.rts: hummermult-inf.rts.o hummer-loops.rts.o $(PAPILIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

hummermult.rts: hummermult.rts.o hummer-loops.rts.o $(PAPILIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

test_bgl_perfctr.rts: test_bgl_perfctr.rts.o 
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

test_bgl_perfctr-overf.rts: test_bgl_perfctr-overf.rts.o
	$(CC) -o $@ $^ $(LDFLAGS) $(PAPILIB)

install: $(TARGETS)
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	-mkdir -p $(DESTDIR)/examples/papi-2-3-4/bgl_tests
	-chmod go+rx $(DESTDIR)/examples/papi-2-3-4/bgl_tests
	-find . -perm -100 -type f -exec cp {} $(DESTDIR)/examples/papi-2-3-4/bgl_tests \;
	-chmod go+rx $(DESTDIR)/examples/papi-2-3-4/bgl_tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DESTDIR)/examples/papi-2-3-4/bgl_tests \;
