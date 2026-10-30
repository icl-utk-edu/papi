NAME=sysdetect

include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

ifneq ($(MPICC),)
    ifeq ($(NO_MPI_TESTS),)
        MPITESTS = query_device_mpi
    else
        MPITESTS =
    endif
endif

FTESTS =
ifeq ($(ENABLE_FORTRAN_TESTS),yes)
    cray_compilers := ftn% crayftn%
    flang_compilers := flang% armflang% amdflang%
    gnu_compilers := gfortran%
    intel_compilers := ifort% ifx%
    nvidia_compilers := nvfortran% pgf%
    ifneq ($(filter $(cray_compilers), $(notdir $(F77))),)
        F90FLAGS += $(FFLAGS) -ffree
    else ifneq ($(filter $(flang_compilers), $(notdir $(F77))),)
        F90FLAGS += $(FFLAGS) -ffree-form
    else ifneq ($(filter $(gnu_compilers), $(notdir $(F77))),)
        F90FLAGS += $(FFLAGS) -ffree-form -ffree-line-length-none
    else ifneq ($(filter $(intel_compilers), $(notdir $(F77))),)
        F90FLAGS += $(FFLAGS) -free
    else ifneq ($(filter $(nvidia_compilers), $(notdir $(F77))),)
        F90FLAGS += $(FFLAGS) -Mfree
    endif
    F90 ?= $(F77)
    FTESTS = query_device_simple_f
endif

TESTS = query_device_simple \
        $(FTESTS)           \
        $(MPITESTS)

sysdetect_tests: $(TESTS)

query_device_simple: query_device_simple.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o query_device_simple query_device_simple.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi: query_device_mpi.o $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(CFLAGS) $(INCLUDE) -o query_device_mpi query_device_mpi.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi.o: query_device_mpi.c
	$(MPICC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

query_device_simple_f: query_device_simple_f.F
	$(F90) $(F90FLAGS) -I../../.. -o $@ $< $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
