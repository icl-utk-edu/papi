NAME=nvml
include ../../Makefile_comp_tests
include ../Makefile.nvml

NVCC = $(CUDA_DIR)/bin/nvcc
NVCFLAGS = -L$(NVML_INCDIR)
CUDALIBS = -L$(NVML_LIBDIR) -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lnvidia-ml

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

nvml_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CC) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

clean:
	rm -f $(TESTS) *.o





