include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = coretemp_basic

example_tests: $(TESTS)

coretemp_basic: coretemp_basic.o
	$(CC) $(CFLAGS) $(INCLUDE) -o coretemp_basic coretemp_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


clean:
	rm -f $(TESTS) *.o





