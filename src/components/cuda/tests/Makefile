include ../../Makefile_comp_tests

CC = nvcc
CFLAGS = -arch=sm_13

%.o:%.cu
	$(CC) $(INCLUDE) $(CFLAGS) -c -o $@ $<

TESTS = HelloWorld

cuda_tests: $(TESTS)

HelloWorld: HelloWorld.o
	$(CC) $(INCLUDE) $(CFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o





