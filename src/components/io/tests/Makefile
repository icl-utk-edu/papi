NAME=io
include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = io_basic io_multiple_components

io_tests: $(TESTS)

io_basic: io_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o io_basic io_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

io_multiple_components: io_multiple_components.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o io_multiple_components io_multiple_components.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o
