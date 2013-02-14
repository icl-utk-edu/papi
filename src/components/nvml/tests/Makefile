include ../../Makefile_comp_tests

NVCC = nvcc
NVCFLAGS = -arch=sm_13

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

TESTS = HelloWorld

nvml_tests: $(TESTS) 

HelloWorld: HelloWorld.o $(UTILOBJS) $(PAPILIB)
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

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





