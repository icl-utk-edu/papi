include ../../Makefile_comp_tests
include ../Makefile.cuda

NVCC = nvcc
NVCFLAGS = -arch=sm_13
CUDALIBS = -L/usr/lib64 -L$(CUDA_DIR)/lib64 -L$(CUPTI_DIR)/lib64 -lcuda -lcudart -lcupti

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

cuda_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS)

install:
	@echo "CUDA tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/cuda/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/cuda/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/cuda/tests \;
	-chmod go+rx $(DATADIR)/cuda/tests/*
	-find . -name "*.cu" -type f -exec cp {} $(DATADIR)/cuda/tests \;

clean:
	rm -f $(TESTS) *.o





