# ***NOTE*** The Environment Variable PAPI_ROCM_ROOT must be defined for 
# programs to compile correctly. one typical location is /opt/rocm, but
# contact your sysadmin if you cannot find it.
NAME=rocm_smi
include ../../Makefile_comp_tests.target
PAPI_ROCM_ROOT ?= /opt/rocm
HIPCC ?= $(PAPI_ROCM_ROOT)/bin/hipcc

INCLUDE += -I$(PAPI_ROCMSMI_ROOT)/include
INCLUDE += -I$(PAPI_ROCM_ROOT)/include
INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocm_smi
INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hip
INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hsa
INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocprofiler
INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocblas
LDFLAGS = -ldl -g -pthread

%.o:%.c
	@echo "INCLUDE=" $(INCLUDE)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

%.o:%.cpp
	@echo "INCLUDE=" $(INCLUDE)
	@echo "CFLAGS=" $(CFLAGS)
	g++ $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = rocm_command_line rocm_smi_all power_monitor_rocm rocm_smi_writeTests
TESTS_LONG = rocmsmi_example

rocm_smi_tests: $(TESTS)
rocm_smi_tests_long: $(TESTS_LONG)

# Note: We compile .o separately from the executable link; some versions of hipcc
#       have trouble managing libraries if we try to do both in a single step.

rocm_command_line.o: rocm_command_line.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@ 

rocm_command_line: rocm_command_line.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rocm_smi_all.o: rocm_smi_all.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocm_smi_all: rocm_smi_all.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

power_monitor_rocm.o: power_monitor_rocm.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

power_monitor_rocm: power_monitor_rocm.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rocmsmi_example.o: rocmsmi_example.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocmsmi_example: rocmsmi_example.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -L$(PAPI_ROCM_ROOT)/lib/rocblas -lrocblas

rocm_smi_writeTests.o: rocm_smi_writeTests.cpp $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

rocm_smi_writeTests: rocm_smi_writeTests.o $(UTILOBJS) $(PAPILIB)
	$(HIPCC) $(CFLAGS) $(INCLUDE) -o $@ $< $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) $(TESTS_LONG) *.o

checkpath: 
	echo PAPI_ROCM_ROOT = $(PAPI_ROCM_ROOT)
	echo HIP_PATH = $(HIP_PATH)
	echo HIPCC = $(HIPCC)
	echo INCLUDE = $(INCLUDE)
	echo LDFLAGS = $(LDFLAGS)
	echo CFLAGS = $(CFLAGS)

