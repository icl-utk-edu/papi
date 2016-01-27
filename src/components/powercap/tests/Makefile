NAME=powercap
include ../../Makefile_comp_tests

TESTS = powercap_basic powercap_busy

powercap_tests: $(TESTS)

powercap_basic.o:	powercap_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c powercap_basic.c -o powercap_basic.o -DBASIC_TEST=1

powercap_busy.o:	powercap_basic.c
	$(CC) $(CFLAGS) $(INCLUDE) -c powercap_basic.c -o powercap_busy.o 

powercap_basic: powercap_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o powercap_basic powercap_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

powercap_busy: powercap_busy.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o powercap_busy powercap_busy.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o *~





