NAME=cuda_range
include ../../Makefile_comp_tests.target

PAPI_CUDA_RANGE_ROOT ?= $(shell dirname $(shell dirname $(shell which nvcc)))
nvcc = $(PAPI_CUDA_RANGE_ROOT)/bin/nvcc

# Set flags for the NVCC compiler.
NVCCFLAGS += -g -ccbin=$(CC)
# Following logic from PAPI PR #251.
ifeq ($(BUILD_SHARED_LIB), yes)
    NVCCFLAGS += -Xcompiler -fpic
endif

# Set includes for the CUDA driver, CUDA runtime, and CUPTI APIs.
INCLUDE += -I$(PAPI_CUDA_RANGE_ROOT)/include -I$(PAPI_CUDA_RANGE_ROOT)/extras/CUPTI/include

# Set libs for the CUDA runtime and CUPTI dynamic shared objects.
CUDALIBS = -L$(PAPI_CUDA_RANGE_ROOT)/lib64 -lcudart -L$(PAPI_CUDA_RANGE_ROOT)/extras/CUPTI/lib64 -lcupti -lcuda

list_of_cuda_range_tests = test_cuda_range_floating_point_operations \
                           test_cuda_range_hello_world               \
                           test_cuda_range_2thr_1gpu_not_allowed     \
                           test_cuda_range_no_user_context           \
                           test_cuda_range_pthreads                  \
                           test_cuda_range_multiple_pass_events
cuda_range_tests: $(list_of_cuda_range_tests)

cuda_range_tests_helper.o: cuda_range_tests_helper.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $^

test_cuda_range_floating_point_operations: test_cuda_range_floating_point_operations.cu
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

test_cuda_range_hello_world: test_cuda_range_hello_world.cu cuda_range_tests_helper.o
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) cuda_range_tests_helper.o  $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

test_cuda_range_2thr_1gpu_not_allowed: test_cuda_range_2thr_1gpu_not_allowed.cu
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

test_cuda_range_no_user_context: test_cuda_range_no_user_context.cu cuda_range_tests_helper.o
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) cuda_range_tests_helper.o $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

test_cuda_range_pthreads: test_cuda_range_pthreads.cu cuda_range_tests_helper.o
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) cuda_range_tests_helper.o $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

test_cuda_range_multiple_pass_events: test_cuda_range_multiple_pass_events.cu cuda_range_tests_helper.o
	$(nvcc) $(NVCCFLAGS) -Xcompiler $(INCLUDE) cuda_range_tests_helper.o $< -o $@ $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

clean:
	rm -f $(list_of_cuda_range_tests) *.o
