NAME=perf_event_uncore
include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = perf_event_uncore perf_event_uncore_multiple \
	perf_event_amd_northbridge perf_event_uncore_cbox


perf_event_uncore_tests: $(TESTS)

perf_event_uncore_lib.o:	perf_event_uncore_lib.c perf_event_uncore_lib.h
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_uncore_lib.c


perf_event_amd_northbridge.o:	perf_event_amd_northbridge.c
	$(CC) $(CFLAGS) $(INCLUDE) -c perf_event_amd_northbridge.c

perf_event_amd_northbridge:	perf_event_amd_northbridge.o
	$(CC) $(LFLAGS) -o perf_event_amd_northbridge perf_event_amd_northbridge.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)


perf_event_uncore.o:	perf_event_uncore.c perf_event_uncore_lib.h
	$(CC) $(LFLAGS) $(INCLUDE) -c perf_event_uncore.c

perf_event_uncore:	perf_event_uncore.o $(UTILOBJS) $(PAPILIB) perf_event_uncore_lib.o
	$(CC) $(CFLAGS) -o perf_event_uncore perf_event_uncore.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_uncore_multiple.o:	perf_event_uncore_multiple.c perf_event_uncore_lib.h
	$(CC) $(LFLAGS) $(INCLUDE) -c perf_event_uncore_multiple.c

perf_event_uncore_multiple:	perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_multiple perf_event_uncore_multiple.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


perf_event_uncore_cbox.o:	perf_event_uncore_cbox.c perf_event_uncore_lib.h
	$(CC) $(LFLAGS) $(INCLUDE) -c perf_event_uncore_cbox.c

perf_event_uncore_cbox:	perf_event_uncore_cbox.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o perf_event_uncore_cbox perf_event_uncore_cbox.o perf_event_uncore_lib.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 



clean:
	rm -f $(TESTS) *.o *~

