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

ifneq ($(F77),)
    FTESTS = query_device_simple_f
else
    FTESTS =
endif

ifeq ($(notdir $(F77)),gfortran)
    FFLAGS +=-ffree-form -ffree-line-length-none
else ifeq ($(notdir $(F77)),flang)
    FFLAGS +=-ffree-form
else
    FFLAGS +=-free
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
	$(F77) $(FFLAGS) -I../../.. -o $@ $< $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
