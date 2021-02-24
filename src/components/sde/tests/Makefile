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

LDFLAGS += -Llib -L$(datadir) -lpapi

SDELIBOBJ=sde_lib.o sde_common.o
SDE_F08_API=../sde_F.F90

TESTS = Minimal_Test Simple_Test Simple2_Test Simple2_NoPAPI_Test Recorder_Test Created_Counter_Test Overflow_Test sde_test_f08

sde_tests: sdelibobj $(TESTS)

sdelibobj: ../sde_lib/sde_lib.c ../sde_lib/sde_common.c
	$(CC) -c -Bdynamic -fPIC -shared -D_GNU_SOURCE -fvisibility=hidden -Wextra -Wall -O2 $(INCLUDE) ../sde_lib/sde_lib.c -o sde_lib.o
	$(CC) -c -Bdynamic -fPIC -shared -D_GNU_SOURCE -fvisibility=hidden -Wextra -Wall -O2 $(INCLUDE) ../sde_lib/sde_common.c -o sde_common.o


################################################################################
## Minimal test
prfx=Minimal

Minimal_Test: $(prfx)/Minimal_Test.c sdelibobj
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(SDELIBOBJ) $(UTILOBJS) $(LDFLAGS)

################################################################################
## Simple test
prfx=Simple

libSimple.so: $(prfx)/Simple_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $(SDELIBOBJ) $^

Simple_Test: $(prfx)/Simple_Driver.c libSimple.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lSimple $(LDFLAGS) -lm

################################################################################
## Simple2 test
prfx=Simple2

libSimple2.so: $(prfx)/Simple2_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $(SDELIBOBJ) $^

Simple2_Test: $(prfx)/Simple2_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lSimple2 $(LDFLAGS) -lm

Simple2_NoPAPI_Test: $(prfx)/Simple2_NoPAPI_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) -Llib -lSimple2 -lm -ldl

################################################################################
## Recorder test
prfx=Recorder

libRecorder.so: $(prfx)/Lib_With_Recorder.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $(SDELIBOBJ) $^

Recorder_Test: $(prfx)/Recorder_Driver.c libRecorder.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lRecorder $(LDFLAGS) -lm


################################################################################
## Created Counter test
prfx=Created_Counter

libCreated_Counter.so: $(prfx)/Lib_With_Created_Counter.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $(SDELIBOBJ) $^

Created_Counter_Test: $(prfx)/Created_Counter_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCreated_Counter $(LDFLAGS) -lm
Overflow_Test: $(prfx)/Overflow_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCreated_Counter $(LDFLAGS) -lm

################################################################################
## Advanced test
prfx=Advanced_C+FORTRAN
rcrd_prfx=Recorder

libXandria.so: $(prfx)/Xandria.F90
	$(F77) -shared -Wall -fPIC $(FFLAGS) $(INCLUDE) -o lib/$@ $(SDE_F08_API) $(SDELIBOBJ) $^

libGamum.so: $(prfx)/Gamum.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $(SDELIBOBJ) $^

sde_test_f08: $(prfx)/sde_test_f08.F90 $(UTILOBJS) $(PAPILIB) libXandria.so libGamum.so libRecorder.so
	$(F77) $< -o $@ $(INCLUDE) $(FFLAGS) $(UTILOBJS) -lXandria -lGamum -lRecorder $(LDFLAGS)

################################################################################
## Cleaning

clean:
	rm -f *.o *.mod lib/*.so

mrproper: clean
	rm -f $(TESTS)


