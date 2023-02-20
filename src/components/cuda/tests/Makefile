NAME=cuda
include ../../Makefile_comp_tests.target

PAPI_CUDA_ROOT ?= /opt/cuda
PAPI_CUPTI_ROOT ?= $(PAPI_CUDA_ROOT)/extras/CUPTI
CUDRV_DIR ?= $(PAPI_CUDA_ROOT)

TESTS = cupti_multi_kernel_launch_monitoring HelloWorld simpleMultiGPU simpleMultiGPU_no_counters
cuda_tests: $(TESTS) 

PAPI_CUDA_ROOT ?= $(CUDA_PATH)
NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc 
NVCFLAGS = -g -ccbin='$(CC)' $(NVPPC64LEFLAGS)
INCLUDE += -I$(PAPI_CUDA_ROOT)/include -I$(PAPI_CUPTI_ROOT)/include 
CUDALIBS = -L$(CUDRV_DIR)/lib64 -L$(PAPI_CUDA_ROOT)/lib64 -L$(PAPI_CUDA_ROOT)/lib64/stubs -L$(PAPI_CUPTI_ROOT)/lib64 -lcudart -lcupti -lcuda 
PAPILIB += -L../../../libpfm4/lib -lpfm
default: $(TESTS)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -E -c -o $@ $<

cupti_multi_kernel_launch_monitoring: cupti_multi_kernel_launch_monitoring.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o cupti_multi_kernel_launch_monitoring cupti_multi_kernel_launch_monitoring.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_multipass_event_fail: test_multipass_event_fail.c
	$(NVCC) $(NVCFLAGS) -o test_multipass_event_fail test_multipass_event_fail.c $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_CUPTI11: HelloWorld_CUPTI11.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld_CUPTI11 HelloWorld_CUPTI11.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_NP_Ctx: HelloWorld_NP_Ctx.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld_NP_Ctx HelloWorld_NP_Ctx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_CUPTI11: simpleMultiGPU_CUPTI11.cu $(UTILOBJS)
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
