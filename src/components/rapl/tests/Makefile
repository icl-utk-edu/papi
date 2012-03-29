include ../../Makefile_comp_tests

TESTS = rapl_basic rapl_busy

rapl_tests: $(TESTS)

rapl_basic.o:	rapl_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_basic.c -o rapl_basic.o -DBASIC_TEST=1

rapl_busy.o:	rapl_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_basic.c -o rapl_busy.o -DBUSY_TEST=1

rapl_basic: rapl_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_basic rapl_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rapl_busy: rapl_busy.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_busy rapl_busy.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o





