NAME=infiniband
include ../../Makefile_comp_tests

TESTS = infiniband_list_events infiniband_values_by_code 

infiniband_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

infiniband_list_events: infiniband_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

infiniband_values_by_code: infiniband_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o

