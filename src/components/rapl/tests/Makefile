NAME=rapl
include ../../Makefile_comp_tests

TESTS = rapl_basic rapl_busy rapl_wraparound rapl_overflow

rapl_tests: $(TESTS)

rapl_overflow.o:	rapl_overflow.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_overflow.c

rapl_overflow:	rapl_overflow.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_overflow rapl_overflow.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rapl_basic.o:	rapl_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_basic.c -o rapl_basic.o -DBASIC_TEST=1

rapl_busy.o:	rapl_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_basic.c -o rapl_busy.o -DBUSY_TEST=1

rapl_wraparound.o:	rapl_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c rapl_basic.c -o rapl_wraparound.o -DWRAP_TEST=1

rapl_basic: rapl_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_basic rapl_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rapl_busy: rapl_busy.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_busy rapl_busy.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rapl_wraparound: rapl_wraparound.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rapl_wraparound rapl_wraparound.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o *~





