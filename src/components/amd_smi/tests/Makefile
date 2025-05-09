# ***NOTE*** The Environment Variable PAPI_ROCM_ROOT must be defined for
# programs to compile correctly. one typical location is /opt/rocm, but
# contact your sysadmin if you cannot find it.

# Set the default PAPI_ROCM_ROOT if not already set in the environment
PAPI_ROCM_ROOT ?= /opt/rocm-6.4.0 # Updated to match your log, or ensure your env var is set
HIP_PATH := $(PAPI_ROCM_ROOT)
HIPCC := $(HIP_PATH)/bin/hipcc

# --- DIAGNOSIS STEP: Comment out the include temporarily ---
# include ../../Makefile_comp_tests.target

# --- Manually define variables expected from Makefile_comp_tests.target ---
# For C++ specific flags, you might need CXXFLAGS
CXXFLAGS ?= $(CFLAGS) -std=c++17 # Add -std=c++17 if not already in CFLAGS from an outer makefile
CFLAGS ?= -DPAPI_NUM_COMP=3
OPTFLAGS ?= -O2
UTILOBJS ?= ../../../testlib/libtestlib.a
PAPILIB ?= ../../../libpapi.a

# Base includes that were part of your amdsmi_example compilation log
BASE_INCLUDES_FROM_LOG ?= -I. -I../../.. -I../../../testlib -I../../../validation_tests

# Specific PAPI include path
PAPI_DIR_INCLUDE ?= /home/dwoun/apps/papi/include # Your path

# Consolidate INCLUDE paths
EFFECTIVE_INCLUDE := $(BASE_INCLUDES_FROM_LOG)
EFFECTIVE_INCLUDE += -I$(PAPI_DIR_INCLUDE)
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocm_smi # For amdsmi.h if needed by amdsmi_example
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hip
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hsa
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocprofiler
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocblas

# Common Linker Flags
COMMON_LDFLAGS = -ldl -g -lpthread

# Specific linker flags for AMD SMI tests
# Assuming the library is libamdsmi_runtime.so. Adjust if it's libamdsmi.so
# Also ensure that PAPI_ROCM_ROOT/lib is the correct path for this library.
LDFLAGS_AMDSMI = $(UTILOBJS) $(PAPILIB) -L$(PAPI_ROCM_ROOT)/lib -lamd_smi $(COMMON_LDFLAGS)

# Specific linker flags for GEMM example
LDFLAGS_GEMM = $(UTILOBJS) $(PAPILIB) -L$(PAPI_ROCM_ROOT)/lib -lrocblas $(COMMON_LDFLAGS)

# Default target
all: amdsmi_example amdsmi_basics gemm

# ------------------------
# Pattern rules
# ------------------------

# Generic rule for .c to .o compilation (if you still have pure C files)
# Note: If amdsmi_basics.c was indeed C++, this rule won't be used for it after renaming.
%.o: %.c
	@echo "Compiling C source $< to $@..."
	@echo "HIPCC: $(HIPCC)"
	@echo "CFLAGS: $(CFLAGS) OPTFLAGS: $(OPTFLAGS)"
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(CFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -c $< -o $@

# Generic rule for .cpp to .o compilation
%.o: %.cpp
	@echo "Compiling C++ source $< to $@..."
	@echo "HIPCC: $(HIPCC)"
	@echo "CXXFLAGS: $(CXXFLAGS) OPTFLAGS: $(OPTFLAGS)" # Use CXXFLAGS here
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(CXXFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -c $< -o $@

# ------------------------
# AMD SMI example
# ------------------------

# amdsmi_example.o will use the generic %.o: %.cpp rule
# amdsmi_example.o: amdsmi_example.cpp # This explicit rule is fine but not strictly necessary if generic one exists

amdsmi_example: amdsmi_example.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	# Pass CXXFLAGS for linking if they contain necessary flags like -std=c++17 that hipcc might need for consistency
	$(HIPCC) $(CXXFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_example.o $(LDFLAGS_AMDSMI)

# Assuming amdsmi_basics.c has been renamed to amdsmi_basics.cpp
# amdsmi_basics.o will use the generic %.o: %.cpp rule
# amdsmi_basics.o: amdsmi_basics.cpp # This explicit rule is fine but not strictly necessary

amdsmi_basics: amdsmi_basics.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CXXFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_basics.o $(LDFLAGS_AMDSMI)

# ------------------------
# GEMM example
# ------------------------

# gemm.o will use the generic %.o: %.cpp rule
# gemm.o: gemm.cpp # This explicit rule is fine but not strictly necessary

gemm: gemm.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_GEMM: $(LDFLAGS_GEMM)"
	$(HIPCC) $(CXXFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ gemm.o $(LDFLAGS_GEMM)

# ------------------------
# Cleanup and diagnostics
# ------------------------

clean:
	@echo "Cleaning up..."
	rm -f amdsmi_example amdsmi_example.o \
	      amdsmi_basics amdsmi_basics.o \
	      gemm gemm.o *.exe

checkpath:
	@echo "--- Variables ---"
	@echo "PAPI_ROCM_ROOT = $(PAPI_ROCM_ROOT)"
	@echo "HIPCC = $(HIPCC)"
	@echo "INCLUDES = $(EFFECTIVE_INCLUDE)"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "OPTFLAGS = $(OPTFLAGS)"
	@echo "UTILOBJS = $(UTILOBJS)"
	@echo "PAPILIB = $(PAPILIB)"
	@echo "COMMON_LDFLAGS = $(COMMON_LDFLAGS)"
	@echo "LDFLAGS_AMDSMI = $(LDFLAGS_AMDSMI)"
	@echo "LDFLAGS_GEMM = $(LDFLAGS_GEMM)"

.PHONY: all clean checkpath amdsmi_example amdsmi_basics gemm