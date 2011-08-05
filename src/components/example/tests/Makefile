include ../../Makefile_comp_tests

CC = gcc
CFLAGS = 

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = HelloWorld

example_tests: $(TESTS)

HelloWorld: HelloWorld.o
	$(CC) $(CFLAGS) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


clean:
	rm -f $(TESTS) *.o





