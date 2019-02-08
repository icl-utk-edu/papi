NAME=nvml
include ../../Makefile_comp_tests
include ../Makefile.nvml
# -fverbose-asm -Wa,-adhln=benchSANVML.s

NVCC = $(CUDA_DIR)/bin/nvcc
NVCFLAGS = -L$(NVML_INCDIR) -ccbin=$(CC)
CUDALIBS = -L$(NVML_LIBDIR) -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++
PAPILIB := ../../../libpapi.a  -ldl

%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = HelloWorld nvml_power_limiting_test

nvml_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 

nvml_power_limiting_test: nvml_power_limiting_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limiting_test nvml_power_limiting_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)
	cp nvml_power_limiting_test ~/nvml_power_limiting_test

nvml_power_limit_read_test: nvml_power_limit_read_test.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvml_power_limit_read_test nvml_power_limit_read_test.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)
	cp nvml_power_limit_read_test ~/nvml_power_limit_read_test

benchSANVML: benchSANVML.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o benchSANVML benchSANVML.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) 


nvmlcap_plot: nvmlcap_plot.o $(UTILOBJS) 
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -o nvmlcap_plot nvmlcap_plot.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)


clean:
	rm -f $(TESTS) *.o
	rm -f benchSANVML
	rm -f nvmlcap_plot results*

FORCE:
