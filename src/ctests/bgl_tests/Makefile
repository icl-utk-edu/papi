BLRTS_GNU_ROOT = /bgl/BlueLight/ppcfloor/blrts-gnu
BGLSYS_ROOT    = /bgl/BlueLight/ppcfloor/bglsys
PREFIX         = ../..
LIBRARY = libpapi.a

SERIAL  = hummermult.rts hummermult-inf.rts test_bgl_perfctr.rts \
        cachetest.rts cachetest-vn.rts mpi_bcast_papi.rts test_bgl_perfctr-overf.rts
### This functionality is not present in BG/L: test_bgl_perfctr-overf.rts
BROKEN  = inherit pernode
UTILOBJS=../do_loops.o ../test_utils.o ../dummy.o

ALL     = $(SERIAL)
INCLUDE = -I.. -I../.. -I$(BGLSYS_ROOT)/include
PAPILIB = ../../$(LIBRARY)

.SUFFIXES: .rts.o
.c.rts.o:
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $< -o $@

bgl_test: $(ALL)

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

../do_loops.o: ../do_loops.c
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $< -o $@

../test_utils.o: ../test_utils.c
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $< -o $@

../dummy.o: ../dummy.c
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $< -o $@

mpi_bcast_papi.rts: mpi_bcast_papi.rts.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $< $(LIBS_MPI) $(LIBS_MPI) $(LDFLAGS) $(PAPILIB) $(UTILOBJS)

mpi_bcast_papi.rts.o: mpi_bcast_papi.c
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) $< -o $@

cachetest.rts: cachetest.rts.o $(PAPILIB)
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)

cachetest-vn.rts: cachetest-vn.rts.o $(PAPILIB)
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)

cachetest-vn.rts.o: cachetest.c
	$(CC) -DVN $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -c $< -o $@

hummermult-inf.rts: hummermult-inf.rts.o hummer-loops.rts.o $(PAPILIB)
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)

hummermult.rts: hummermult.rts.o hummer-loops.rts.o $(PAPILIB)
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)

test_bgl_perfctr.rts: test_bgl_perfctr.rts.o
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)

test_bgl_perfctr-overf.rts: test_bgl_perfctr-overf.rts.o
	$(CC) $(TOPTFLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(PAPILIB)


