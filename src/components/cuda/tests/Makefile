NAME=cuda
include ../../Makefile_comp_tests
include ../Makefile.cuda

TESTS = HelloWorld simpleMultiGPU simpleMultiGPU_no_counters simpleMultiGPU_cupti_only
cuda_tests: $(TESTS) 

CUDA_DIR ?= $(CUDA_PATH)
NVCC = $(CUDA_DIR)/bin/nvcc
NVCFLAGS = -g 
INCLUDE += -I$(CUDA_DIR)/include -I$(CUPTI_DIR)/include 
CUDALIBS = -L$(CUDRV_DIR)/lib64 -L$(CUDA_DIR)/lib64 -L$(CUPTI_DIR)/lib64 -lcudart -lcupti -lcuda 

default: HelloWorld simpleMultiGPU_papi simpleMultiGPU_no_counters simpleMultiGPU_cupti_only

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CC) -Wl,-rpath,../../../libpfm4/lib $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

simpleMultiGPU: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

simpleMultiGPU_no_counters: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(INCLUDE) -DNO_COUNTERS -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

simpleMultiGPU_cupti_only: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(INCLUDE) -DCUPTI_ONLY -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

clean:
	rm -f $(TESTS) *.o


run: 
	( cd ~/icl/papi/papi/src/ && make )
	( make HelloWorld )
	./HelloWorld
	( cd ~/icl/papi/papi/src/ && make )
	( make simpleMultiGPU )
	./simpleMultiGPU
