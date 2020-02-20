NAME=sde
include ../../Makefile_comp_tests.target
INCLUDE += -I../interface -I..
ifeq ($(notdir $(F77)),gfortran)
	FFLAGS +=-ffree-form -ffree-line-length-none
else
	FFLAGS +=-free
endif
FFLAGS +=-g
#CFLAGS +=-g

LDFLAGS +=-L..
SDE_F08_API=$(datadir)/sde_F.o

EXTRA_SRC=../interface/papi_sde_interface.c

TESTS = Minimal_Test Simple_Test Simple2_Test Recorder_Test sde_test_f08

sde_tests: $(TESTS)

weak_symbols.o: $(EXTRA_SRC)
	$(CC) -Wall -c $< -o $@

################################################################################
## Minimal test
prfx=Minimal

Minimal_Test: $(prfx)/Minimal_Test.c
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(PAPILIB) $(LDFLAGS)

################################################################################
## Simple test
prfx=Simple

$(prfx)/libSimple.so: $(prfx)/Simple_Lib.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^

Simple_Test: $(prfx)/Simple_Driver.c $(prfx)/libSimple.so
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS) $(PAPILIB) $(LDFLAGS) -lm

################################################################################
## Simple2 test
prfx=Simple2

$(prfx)/libSimple2.so: $(prfx)/Simple2_Lib.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^

Simple2_Test: $(prfx)/Simple2_Driver.c $(prfx)/libSimple2.so
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS) $(PAPILIB) $(LDFLAGS) -lm

################################################################################
## Recorder test
prfx=Recorder

$(prfx)/libRecorder.so: $(prfx)/Lib_With_Recorder.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^

Recorder_Test: $(prfx)/Recorder_Driver.c $(prfx)/libRecorder.so
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS) $(PAPILIB) $(LDFLAGS) -lm

################################################################################
## Advanced test
prfx=Advanced_C+FORTRAN
rcrd_prfx=Recorder

sde_test_f08: $(prfx)/sde_test_f08.F90 $(UTILOBJS) $(PAPILIB) $(prfx)/libXandria.so $(prfx)/libGamum.so $(rcrd_prfx)/libRecorder.so
	$(F77) $< -o $@ $(INCLUDE) $(FFLAGS) $(prfx)/libXandria.so $(prfx)/libGamum.so $(rcrd_prfx)/libRecorder.so $(PAPILIB) $(LDFLAGS)

$(prfx)/libXandria.so: $(prfx)/Xandria.F90 weak_symbols.o
	$(F77) -shared -Wall -fPIC $(FFLAGS) $(INCLUDE) -o $@ $^ $(SDE_F08_API)

$(prfx)/libGamum.so: $(prfx)/Gamum.c weak_symbols.o
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o $@ $^


LIBS=Advanced_C+FORTRAN/libXandria.so Advanced_C+FORTRAN/libGamum.so Recorder/libRecorder.so Simple/libSimple.so Simple2/libSimple2.so

clean:
	rm -f *.o *.mod $(LIBS)

mrproper: clean
	rm -f $(TESTS)


