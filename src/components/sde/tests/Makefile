NAME=sde
include ../../Makefile_comp_tests.target
INCLUDE += -I../interface -I..
ifeq ($(notdir $(F77)),gfortran)
	FFLAGS +=-ffree-form -ffree-line-length-none
else
	FFLAGS +=-free
endif
FFLAGS +=-g
CFLAGS +=-O3

LDFLAGS +=-L.. -lrt
SDE_F08_API=$(datadir)/sde_F.o

SDE_LIBS=libXandria.so libGamum.so libRecorder.so

EXTRA_SRC=../interface/papi_sde_interface.c

TESTS = Minimal_Test sde_test_f08

sde_tests: $(SDE_LIBS) $(TESTS)

weak_symbols.o: $(EXTRA_SRC)
	$(CC) -Wall -c $< -o $@

sde_test_f08: sde_test_f08.F90 $(UTILOBJS) $(PAPILIB) libXandria.so libGamum.so libRecorder.so
	$(F77) $< -o $@ $(INCLUDE) $(FFLAGS) libXandria.so libGamum.so libRecorder.so $(PAPILIB) $(LDFLAGS)

libXandria.so: Xandria.F90 weak_symbols.o
	$(F77) -shared -Wall -fPIC $(FFLAGS) $(INCLUDE) -o $@ $^ $(SDE_F08_API)

libGamum.so: Gamum.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^

libRecorder.so: Recorder.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^

Minimal_Test: Minimal_Test.c
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f *.o *.so *.mod

mrproper: clean
	rm -f $(TESTS) $(SDE_LIBS)


