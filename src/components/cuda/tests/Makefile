NAME=cuda
include ../../Makefile_comp_tests.target

PAPI_CUDA_ROOT ?= $(shell dirname $(shell dirname $(shell which nvcc)))

TESTS = HelloWorld simpleMultiGPU \
		pthreads cudaOpenMP concurrent_profiling \
		test_multi_read_and_reset test_multipass_event_fail \
		test_2thr_1gpu_not_allowed

TESTS_NOCTX = concurrent_profiling_noCuCtx pthreads_noCuCtx \
			  cudaOpenMP_noCuCtx HelloWorld_noCuCtx \
			  simpleMultiGPU_noCuCtx

NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc
NVCC_VERSION := $(shell nvcc --version | grep -oP '(?<=release )\d+\.\d+')
ifeq ($(shell echo "$(NVCC_VERSION) >= 11.6" | bc), 1)
    NVCFLAGS := -arch=native
else
    NVCFLAGS :=
endif

PAPI_FLAG = -DPAPI    # Comment this line for tests to run without PAPI profiling
NVCFLAGS += -g -ccbin='$(CC)' $(PAPI_FLAG)
INCLUDE += -I$(PAPI_CUDA_ROOT)/include
CUDALIBS = -L$(PAPI_CUDA_ROOT)/lib64 -lcudart -lcuda
PAPILIB += -L../../../libpfm4/lib -lpfm

cuda_tests: $(TESTS) $(TESTS_NOCTX)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -E -c -o $@ $<

test_multi_read_and_reset: test_multi_read_and_reset.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multi_read_and_reset test_multi_read_and_reset.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling: concurrent_profiling.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o concurrent_profiling concurrent_profiling.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling_noCuCtx: concurrent_profiling_noCuCtx.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o concurrent_profiling_noCuCtx concurrent_profiling_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads: pthreads.o
	$(NVCC) $(NVCFLAGS) -o pthreads pthreads.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads_noCuCtx: pthreads_noCuCtx.o
	$(NVCC) $(NVCFLAGS) -o pthreads_noCuCtx pthreads_noCuCtx.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP: cudaOpenMP.o
	$(NVCC) $(NVCFLAGS) -o cudaOpenMP cudaOpenMP.o -lgomp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP.o: cudaOpenMP.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c cudaOpenMP.cu -Xcompiler -fopenmp

cudaOpenMP_noCuCtx: cudaOpenMP_noCuCtx.o
	$(NVCC) $(NVCFLAGS) -o cudaOpenMP_noCuCtx cudaOpenMP_noCuCtx.o -lgomp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP_noCuCtx.o: cudaOpenMP_noCuCtx.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c cudaOpenMP_noCuCtx.cu -Xcompiler -fopenmp

test_multipass_event_fail: test_multipass_event_fail.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multipass_event_fail test_multipass_event_fail.o $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(LDFLAGS) $(CUDALIBS)

test_2thr_1gpu_not_allowed: test_2thr_1gpu_not_allowed.o
	$(NVCC) $(NVCFLAGS) -o test_2thr_1gpu_not_allowed test_2thr_1gpu_not_allowed.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_noCuCtx: HelloWorld_noCuCtx.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld_noCuCtx HelloWorld_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_noCuCtx: simpleMultiGPU_noCuCtx.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f *.o $(TESTS) $(TESTS_NOCTX)
