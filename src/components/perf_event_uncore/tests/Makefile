include ../../Makefile_comp_tests

TESTS = perf_event_uncore perf_event_uncore_multiple perf_event_uncore_nogran


perf_event_uncore_tests: $(TESTS)

perf_event_uncore_lib.o:	perf_event_uncore_lib.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore_lib.c


perf_event_uncore.o:	perf_event_uncore.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore.c

perf_event_uncore:	perf_event_uncore.o $(UTILOBJS) $(PAPILIB) perf_event_uncore_lib.o
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore perf_event_uncore.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_uncore_multiple.o:	perf_event_uncore_multiple.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore_multiple.c

perf_event_uncore_multiple:	perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_multiple perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_uncore_nogran.o:	perf_event_uncore_nogran.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore_nogran.c

perf_event_uncore_nogran:	perf_event_uncore_nogran.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_nogran perf_event_uncore_nogran.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 



install:
	@echo "perf_event_uncore tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/perf_event_uncore/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/perf_event_uncore/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/perf_event_uncore/tests \;
	-chmod go+rx $(DATADIR)/perf_event_uncore/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/perf_event_uncore/tests \;

clean:
	rm -f $(TESTS) *.o *~





