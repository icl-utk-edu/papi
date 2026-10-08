NAME=nvml_utils
include ../../Makefile_comp_tests
# -fverbose-asm -Wa,-adhln=benchSANVML.s

PAPI_NVML_INC ?= $(PAPI_NVML_ROOT)/include
PAPI_NVML_LIB ?= $(PAPI_NVML_ROOT)/lib64/stubs

NVCC = $(PAPI_NVML_ROOT)/bin/nvcc
NVCFLAGS = -I$(PAPI_NVML_INC) -ccbin='$(CC)' $(NVPPC64LEFLAGS)
CUDALIBS = -L$(PAPI_NVML_LIB) -L$(PAPI_NVML_ROOT)/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++
PAPILIB := ../../../libpapi.a  -ldl

NVML_UTILS = nvmlcap_plot
nvml_utils_tests: $(NVML_UTILS)

%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -c -o $@ $<

nvmlcap_plot: nvmlcap_plot.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvmlcap_plot nvmlcap_plot.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

clean:
	rm -f $(NVML_UTILS) *.o *.gnuplot *.tsv
