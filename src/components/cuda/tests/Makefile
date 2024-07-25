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
NVCC_VERSION := $(shell $(NVCC) --version | grep -oP '(?<=release )\d+\.\d+')
ifeq ($(shell echo "$(NVCC_VERSION) >= 11.6" | bc), 1)
    NVCFLAGS := -arch=native
else
    NVCFLAGS :=
endif

PAPI_FLAG = -DPAPI    # Comment this line for tests to run without PAPI profiling
NVCFLAGS += -g -ccbin='$(CC)' $(PAPI_FLAG)
CFLAGS += -g $(PAPI_FLAG)
INCLUDE += -I$(PAPI_CUDA_ROOT)/include
CUDALIBS = -L$(PAPI_CUDA_ROOT)/lib64 -lcudart -lcuda
PAPILIB += -L../../../libpfm4/lib -lpfm

cuda_tests: $(TESTS) $(TESTS_NOCTX)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -E -c -o $@ $<

test_multi_read_and_reset: test_multi_read_and_reset.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o test_multi_read_and_reset test_multi_read_and_reset.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling: concurrent_profiling.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o concurrent_profiling concurrent_profiling.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling_noCuCtx: concurrent_profiling_noCuCtx.o $(UTILOBJS)
	$(CXX) $(CFLAGS) -o concurrent_profiling_noCuCtx concurrent_profiling_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads: pthreads.o
	$(CC) $(CFLAGS) -o pthreads pthreads.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads_noCuCtx: pthreads_noCuCtx.o
	$(CC) $(CFLAGS) -o pthreads_noCuCtx pthreads_noCuCtx.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP: cudaOpenMP.o
	$(CC) $(CFLAGS) -o cudaOpenMP cudaOpenMP.o -lgomp -fopenmp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP_noCuCtx: cudaOpenMP_noCuCtx.o
	$(CC) $(CFLAGS) -o cudaOpenMP_noCuCtx cudaOpenMP_noCuCtx.o -lgomp -fopenmp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_multipass_event_fail: test_multipass_event_fail.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o test_multipass_event_fail test_multipass_event_fail.o $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(LDFLAGS) $(CUDALIBS)

test_2thr_1gpu_not_allowed: test_2thr_1gpu_not_allowed.o
	$(CC) $(CFLAGS) -o test_2thr_1gpu_not_allowed test_2thr_1gpu_not_allowed.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_noCuCtx: HelloWorld_noCuCtx.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o HelloWorld_noCuCtx HelloWorld_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o simpleMultiGPU simpleMultiGPU.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_noCuCtx: simpleMultiGPU_noCuCtx.o $(UTILOBJS)
	$(CC) $(CFLAGS) -o simpleMultiGPU_noCuCtx simpleMultiGPU_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f *.o $(TESTS) $(TESTS_NOCTX)
