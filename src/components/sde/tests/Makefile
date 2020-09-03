NAME=sde
include ../../Makefile_comp_tests.target
INCLUDE += -I../sde_lib -I..
ifeq ($(notdir $(F77)),gfortran)
	FFLAGS +=-ffree-form -ffree-line-length-none
else ifeq ($(notdir $(F77)),flang)
	FFLAGS +=-ffree-form
else
	FFLAGS +=-free
endif
FFLAGS +=-g
#CFLAGS +=-g

SDELIB =-L../sde_lib -lsde
LDFLAGS += -Llib -L$(datadir) -lpapi
SDE_F08_API=$(datadir)/sde_F.o

TESTS = Minimal_Test Simple_Test Simple2_Test Simple2_NoPAPI_Test Recorder_Test Created_Counter_Test Overflow_Test sde_test_f08

sde_tests: libsde $(TESTS)

.PHONY: libsde
libsde: ../sde_lib/sde_lib.c ../sde_lib/sde_common.c
		make -C ../sde_lib

################################################################################
## Minimal test
prfx=Minimal

Minimal_Test: $(prfx)/Minimal_Test.c
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(SDELIB) $(UTILOBJS) $(LDFLAGS)

################################################################################
## Simple test
prfx=Simple

libSimple.so: $(prfx)/Simple_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Simple_Test: $(prfx)/Simple_Driver.c libSimple.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS) -lSimple $(SDELIB) -lm

################################################################################
## Simple2 test
prfx=Simple2

libSimple2.so: $(prfx)/Simple2_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Simple2_Test: $(prfx)/Simple2_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS) -lSimple2 $(SDELIB) -lm

Simple2_NoPAPI_Test: $(prfx)/Simple2_NoPAPI_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) -Llib -lSimple2 $(SDELIB) -lm

################################################################################
## Recorder test
prfx=Recorder

libRecorder.so: $(prfx)/Lib_With_Recorder.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Recorder_Test: $(prfx)/Recorder_Driver.c libRecorder.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS) -lRecorder $(SDELIB) -lm


################################################################################
## Created Counter test
prfx=Created_Counter

libCreated_Counter.so: $(prfx)/Lib_With_Created_Counter.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Created_Counter_Test: $(prfx)/Created_Counter_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS) -lCreated_Counter $(SDELIB) -lm
Overflow_Test: $(prfx)/Overflow_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS) -lCreated_Counter $(SDELIB) -lm

################################################################################
## Advanced test
prfx=Advanced_C+FORTRAN
rcrd_prfx=Recorder

sde_test_f08: $(prfx)/sde_test_f08.F90 $(UTILOBJS) $(PAPILIB) libXandria.so libGamum.so libRecorder.so
	$(F77) $< -o $@ $(INCLUDE) $(FFLAGS) $(UTILOBJS) $(LDFLAGS) -lXandria -lGamum -lRecorder $(SDELIB)

libXandria.so: $(prfx)/Xandria.F90
	$(F77) -shared -Wall -fPIC $(FFLAGS) $(INCLUDE) -o lib/$@ $^ $(SDE_F08_API)

libGamum.so: $(prfx)/Gamum.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

clean:
	rm -f *.o *.mod lib/*.so

mrproper: clean
	rm -f $(TESTS)


