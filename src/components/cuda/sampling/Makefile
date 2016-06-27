include ../Makefile.cuda
CFG = -I$(CUDA_DIR)/include  -I$(CUPTI_DIR)/include
LDG = -L$(CUDA_DIR)/lib64  -L$(CUPTI_DIR)/lib64
comp= $(CUDA_DIR)/bin/nvcc

all:
	@make lib
	@make papi_sampling
	@make sample
	@make cpy

lib: activity.c
	@gcc -fPIC -shared -o libactivity.so  $(CFG) $(LDG) activity.c -lcuda -lcupti

papi_sampling: gpu_activity.c
	@gcc -I. gpu_activity.c -o ../../../utils/papi_sampling_cuda

sample: test/matmul.cu
	@$(comp) test/matmul.cu $(CFG) $(LDG) -o test/matmul -lcuda 

cpy:
	cp ../../../utils/papi_sampling_cuda  test

clean:
	rm libactivity.so ../../../utils/papi_sampling_cuda test/matmul test/sass* test/papi_sampling_cuda
