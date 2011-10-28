include ../../Makefile_comp_tests

NVCC = nvcc
NVCFLAGS = -arch=sm_13

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

cuda_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS) $(PAPILIB)
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o





