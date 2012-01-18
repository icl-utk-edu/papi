include ../../Makefile_comp_tests

TESTS = appio_list_events appio_values_by_code appio_values_by_name appio_test_read_write appio_test_pthreads appio_test_fread_fwrite

appio_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

appio_list_events: appio_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_values_by_code: appio_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_values_by_code.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_values_by_name: appio_values_by_name.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_values_by_name.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_read_write: appio_test_read_write.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_read_write.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_fread_fwrite: appio_test_fread_fwrite.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_fread_fwrite.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

appio_test_pthreads: appio_test_pthreads.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ appio_test_pthreads.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lpthread

clean:
	rm -f $(TESTS) *.o

