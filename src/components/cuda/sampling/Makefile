# Set defaults if these environment variables are not set yet.
PAPI_CUDA_ROOT ?= /opt/cuda
PAPI_CUDA_INC ?= $(PAPI_CUDA_ROOT)/include
PAPI_CUPTI_INC ?= $(PAPI_CUDA_ROOT)/extras/CUPTI/include
PAPI_CUDA_LIBS ?= $(PAPI_CUDA_ROOT)/lib64
PAPI_CUDA_STUBS ?= $(PAPI_CUDA_ROOT)/lib64/stubs
PAPI_CUPTI_LIBS ?= $(PAPI_CUDA_ROOT)/extras/CUPTI/lib64
CFG = -I$(PAPI_CUDA_INC)  -I$(PAPI_CUPTI_INC)
LDG = -L$(PAPI_CUDA_LIBS) -L$(PAPI_CUDA_STUBS)  -L$(PAPI_CUPTI_LIBS)
NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc

all:
	@make lib
	@make papi_sampling
	@make sample
	@make cpy

lib: activity.c
	@gcc -fPIC -shared -o libactivity.so  $(CFG) $(LDG) activity.c -lcuda -lcupti

papi_sampling: gpu_activity.c
	@echo 'char* ld_prld="LD_PRELOAD='`pwd`'/sampling/libactivity.so";' > path.h
	@gcc -I. gpu_activity.c -o ../../../utils/papi_sampling_cuda

sample: test/matmul.cu
	@$(NVCC) test/matmul.cu $(CFG) $(LDG) -o test/matmul -lcuda 

cpy:
	cp ../../../utils/papi_sampling_cuda  test

clean:
	rm libactivity.so ../../../utils/papi_sampling_cuda test/matmul test/sass* test/papi_sampling_cuda
