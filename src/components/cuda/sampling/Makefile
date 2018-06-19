# include ../Makefile.cuda

CUDA_DIR ?= /opt/cuda
CUPTI_DIR ?= $(CUDA_DIR)/extras/CUPTI
CFG = -I$(CUDA_DIR)/include  -I$(CUPTI_DIR)/include
LDG = -L$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64/stubs  -L$(CUPTI_DIR)/lib64
NVCC = $(CUDA_DIR)/bin/nvcc

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
