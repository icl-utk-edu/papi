NAME=rocm
include ../../Makefile_comp_tests.target
INCLUDE += -I/opt/rocm/hip/include
INCLUDE += -I$(PAPI_ROCM_ROOT)/hsa/include/hsa
INCLUDE += -I$(PAPI_ROCM_ROOT)/rocprofiler/include
INCLUDE += -I$(PAPI_ROCM_ROOT)/include
LDFLAGS = -ldl -g

linux-rocm.o: components/rocm/linux-rocm.c $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c components/rocm/linux-rocm.c -o linux-rocm.o


%.o:%.c
	@echo "INCLUDE=" $(INCLUDE)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

%.o:%.cpp
	@echo "INCLUDE=" $(INCLUDE)
	@echo "CFLAGS=" $(CFLAGS)
	g++ $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = 

rocm_tests: $(TESTS)

rocm_command_line: rocm_command_line.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rocm_command_line rocm_command_line.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rocm_standalone_report_all: rocm_standalone_report_all.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rocm_standalone_report_all rocm_standalone_report_all.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o
