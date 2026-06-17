NAME=nvml
include ../../Makefile_comp_tests.target
# -fverbose-asm -Wa,-adhln=benchSANVML.s

PAPI_NVML_ROOT ?= $(PAPI_CUDA_ROOT)
PAPI_NVML_INC ?= $(PAPI_CUDA_ROOT)/include
PAPI_NVML_LIB ?= $(PAPI_CUDA_ROOT)/lib64/stubs

NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc
NVCFLAGS = -I$(PAPI_NVML_INC) -ccbin='$(CC)' $(NVPPC64LEFLAGS)
CUDALIBS = -L$(PAPI_NVML_LIB) -L$(PAPI_CUDA_ROOT)/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++
PAPILIB := ../../../libpapi.a  -ldl

NVCC_VERSION := $(shell $(NVCC) --version | grep -oP '(?<=release )\d+\.\d+')

# Check to see if the Cuda Toolkit version is either 12.8 or 12.9 such that
# we can add -Wno-deprecated-gpu-targets to avoid compilation warning
CUDA_TOOLKIT_EQUAL_TO_12_8_OR_12_9 := $(shell echo "$(NVCC_VERSION)" | awk '{print ($$1 == 12.8 || $$1 == 12.9)}')
ifeq ($(CUDA_TOOLKIT_EQUAL_TO_12_8_OR_12_9), 1)
    NVCFLAGS += -Wno-deprecated-gpu-targets
endif

%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = HelloWorld nvml_power_limiting_test

nvml_tests: $(TESTS) 

nvml_tests_helper.o: nvml_tests_helper.c
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $^

HelloWorld: HelloWorld.o nvml_tests_helper.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o HelloWorld HelloWorld.o nvml_tests_helper.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 

nvml_power_limiting_test: nvml_power_limiting_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limiting_test nvml_power_limiting_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

nvml_power_limit_read_test: nvml_power_limit_read_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limit_read_test nvml_power_limit_read_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

benchSANVML: benchSANVML.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o benchSANVML benchSANVML.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 

clean:
	rm -f $(TESTS) *.o
	rm -f benchSANVML
	rm -f nvmlcap_plot results*

FORCE:
