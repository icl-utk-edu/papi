include ../../Makefile_comp_tests
include ../Makefile.nvml

NVCC = $(CUDA_DIR)/bin/nvcc
NVCFLAGS = -arch=sm_13
CUDALIBS = -L$(NVML_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lnvidia-ml

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

nvml_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(CC) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

install:
	@echo "NVML tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/nvml/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/nvml/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/nvml/tests \;
	-chmod go+rx $(DATADIR)/nvml/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/nvml/tests \;

clean:
	rm -f $(TESTS) *.o





