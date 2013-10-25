NAME=cuda
include ../../Makefile_comp_tests
include ../Makefile.cuda

NVCC = $(CUDA_DIR)/bin/nvcc
NVCFLAGS = -arch=sm_13
CUDALIBS = -L$(CUDRV_DIR)/lib64 -L$(CUDA_DIR)/lib64 -L$(CUPTI_DIR)/lib64 -lcudart -lcupti -lcuda 

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

cuda_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CC) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

clean:
	rm -f $(TESTS) *.o





