include ../../Makefile_comp_tests

TESTS = appio_list_events net_values_by_code net_values_by_name

net_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

appio_list_events: appio_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_code: net_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_code.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_name: net_values_by_name.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_name.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)


clean:
	rm -f $(TESTS) *.o

