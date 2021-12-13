NAME=sysdetect

include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = query_device_simple \
        query_device_mpi

sysdetect_tests: $(TESTS)

query_device_simple: query_device_simple.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o query_device_simple query_device_simple.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi: query_device_mpi.o $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(CFLAGS) $(INCLUDE) -o query_device_mpi query_device_mpi.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi.o: query_device_mpi.c
	$(MPICC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

clean:
	rm -f $(TESTS) *.o
