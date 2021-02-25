NAME=powercap_ppc
include ../../Makefile_comp_tests.target

TESTS = powercap_ppc_basic powercap_ppc_limit

powercap_ppc_tests: $(TESTS)

powercap_ppc_basic.o:	powercap_ppc_basic.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c powercap_ppc_basic.c -o powercap_ppc_basic.o

powercap_ppc_basic: powercap_ppc_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_ppc_basic powercap_ppc_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


powercap_ppc_limit.o:	powercap_ppc_limit.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c powercap_ppc_limit.c -o powercap_ppc_limit.o

powercap_ppc_limit: powercap_ppc_limit.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o powercap_ppc_limit powercap_ppc_limit.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o *~
