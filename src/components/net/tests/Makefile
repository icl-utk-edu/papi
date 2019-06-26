NAME=net
include ../../Makefile_comp_tests.target

TESTS = net_list_events net_values_by_code net_values_by_name mpi_net_test

net_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

net_list_events: net_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_code: net_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_code.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_name: net_values_by_name.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_name.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

mpi_net_test: mpi_net_test.o $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(INCLUDE) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o

