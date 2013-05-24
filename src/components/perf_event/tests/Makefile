include ../../Makefile_comp_tests

TESTS = perf_event_offcore_response perf_event_system_wide perf_event_user_kernel


perf_event_tests: $(TESTS)


event_name_lib.o:	event_name_lib.c event_name_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c event_name_lib.c


perf_event_offcore_response.o:	perf_event_offcore_response.c event_name_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_offcore_response.c

perf_event_offcore_response:	perf_event_offcore_response.o event_name_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_offcore_response perf_event_offcore_response.o event_name_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_system_wide.o:	perf_event_system_wide.c
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_system_wide.c

perf_event_system_wide:	perf_event_system_wide.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_system_wide perf_event_system_wide.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_user_kernel.o:	perf_event_user_kernel.c event_name_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_user_kernel.c

perf_event_user_kernel:	perf_event_user_kernel.o event_name_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_user_kernel perf_event_user_kernel.o event_name_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 



install:
	@echo "perf_event tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/perf_event/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/perf_event/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/perf_event/tests \;
	-chmod go+rx $(DATADIR)/perf_event/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/perf_event/tests \;

clean:
	rm -f $(TESTS) *.o *~





