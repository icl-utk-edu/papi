NAME=cuda
include ../../Makefile_comp_tests.target

CUDA_DIR ?= /opt/cuda
CUPTI_DIR ?= $(CUDA_DIR)/extras/CUPTI
CUDRV_DIR ?= $(CUDA_DIR)

TESTS = HelloWorld simpleMultiGPU simpleMultiGPU_no_counters 
cuda_tests: $(TESTS) 

CUDA_DIR ?= $(CUDA_PATH)
NVCC = $(CUDA_DIR)/bin/nvcc 
NVCFLAGS = -g -ccbin='$(CC)'
INCLUDE += -I$(CUDA_DIR)/include -I$(CUPTI_DIR)/include 
CUDALIBS = -L$(CUDRV_DIR)/lib64 -L$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64/stubs -L$(CUPTI_DIR)/lib64 -lcudart -lcupti -lcuda 
PAPILIB += -L../../../libpfm4/lib -lpfm
default: $(TESTS)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -E -c -o $@ $<

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_no_counters: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DNO_COUNTERS -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_cupti_only: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DCUPTI_ONLY -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

nvlink_bandwidth: nvlink_bandwidth.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

nvlink_bw_plus: nvlink_bw_plus.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cuda_event_only_test: cuda_event_only_test.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaTest_cupti_only: cudaTest_cupti_only.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

likeComp_cupti_only: likeComp_cupti_only.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

nvlink_all: nvlink_all.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

nvlink_bandwidth_cupti_only: nvlink_bandwidth_cupti_only.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

# The following program is taken from the cuda sample directory; /sw/peak/cuda/9.2.148/samples/1_Utilities/
nvidia_sample_bw_test: nvidia_sample_bw_test.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -I./sample_inc -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

# The following program is taken from the cuda sample directory; /sw/peak/cuda/9.2.148/samples/1_Utilities/
p2pBandwidthLatencyTest: p2pBandwidthLatencyTest.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -I./sample_inc -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o cuda_ld_preload_example.so nvlink_bandwidth nvlink_all


# Extra example to show LD_PRELOAD can be used to insert PAPI CUDA into a pre-existing binary
cuda_ld_preload_example.so: simpleMultiGPU_no_counters cuda_ld_preload_example.c
	gcc -Wall -fPIC -shared -o cuda_ld_preload_example.so $(LDFLAGS) $(INCLUDE) cuda_ld_preload_example.c ../../../libpapi.so

run: 
	( cd ~/icl/papi/papi/src/ && make )
	( make HelloWorld )
	./HelloWorld
	( cd ~/icl/papi/papi/src/ && make )
	( make simpleMultiGPU )
	./simpleMultiGPU
