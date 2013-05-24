include ../../Makefile_comp_tests

TESTS = perf_event_system_wide


perf_event_tests: $(TESTS)


perf_event_system_wide.o:	perf_event_system_wide.c
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_system_wide.c

perf_event_system_wide:	perf_event_system_wide.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_system_wide perf_event_system_wide.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 



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





