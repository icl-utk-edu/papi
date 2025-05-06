# ***NOTE*** The Environment Variable PAPI_ROCM_ROOT must be defined for
# programs to compile correctly. one typical location is /opt/rocm, but
# contact your sysadmin if you cannot find it.

# Set the default PAPI_ROCM_ROOT if not already set in the environment
PAPI_ROCM_ROOT ?= /apps/rocm/rocm-6.3.2
HIP_PATH := $(PAPI_ROCM_ROOT)
HIPCC := $(HIP_PATH)/bin/hipcc

# --- DIAGNOSIS STEP: Comment out the include temporarily ---
# include ../../Makefile_comp_tests.target

# --- Manually define variables expected from Makefile_comp_tests.target ---
# These values are based on your successful amdsmi_example compilation log:
CFLAGS ?= -DPAPI_NUM_COMP=3
OPTFLAGS ?= -O2
UTILOBJS ?= ../../../testlib/libtestlib.a
PAPILIB ?= ../../../libpapi.a
# Base includes that were part of your amdsmi_example compile line (before PAPI/ROCm specifics)
BASE_INCLUDES_FROM_LOG ?= -I. -I../../.. -I../../../testlib -I../../../validation_tests

# Specific PAPI include path
PAPI_DIR_INCLUDE ?= /home/dwoun/apps/papi/include

# Consolidate INCLUDE paths
EFFECTIVE_INCLUDE := $(BASE_INCLUDES_FROM_LOG)
EFFECTIVE_INCLUDE += -I$(PAPI_DIR_INCLUDE) # Add PAPI include
# Add ROCm specific paths
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocm_smi
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hip
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/hsa
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocprofiler
EFFECTIVE_INCLUDE += -I$(PAPI_ROCM_ROOT)/include/rocblas # For GEMM

# Common Linker Flags
COMMON_LDFLAGS = -ldl -g -lpthread

# Specific linker flags for amdsmi_example
LDFLAGS_AMDSMI = $(UTILOBJS) $(PAPILIB) $(COMMON_LDFLAGS)

# Specific linker flags for gemm_example
LDFLAGS_GEMM = $(UTILOBJS) $(PAPILIB) -L$(PAPI_ROCM_ROOT)/lib -lrocblas $(COMMON_LDFLAGS)

# Default target
all: amdsmi_example gemm_example

# Generic Rule for .cpp to .o compilation (e.g., for amdsmi_example.o from amdsmi_example.cpp)
# This rule is used when the target is X.o and prerequisite is X.cpp
%.o: %.cpp
	@echo "Compiling $< to $@ using pattern rule..."
	@echo "HIPCC: $(HIPCC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OPTFLAGS: $(OPTFLAGS)"
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(CFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -c $< -o $@

# Rule for amdsmi_example.o (uses the pattern rule above)
amdsmi_example.o: amdsmi_example.cpp

# Rule for linking amdsmi_example
amdsmi_example: amdsmi_example.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_example.o $(LDFLAGS_AMDSMI)

# Rule for gemm_example.o from gemm.cpp (NEEDS AN EXPLICIT RECIPE)
# Here, the target object name (gemm_example.o) doesn't match the source stem (gemm.cpp)
# for the simple %.o: %.cpp pattern rule.
gemm_example.o: gemm.cpp
	@echo "Compiling $< (gemm.cpp) to produce $@ (gemm_example.o) with explicit rule..."
	@echo "HIPCC: $(HIPCC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OPTFLAGS: $(OPTFLAGS)"
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(CFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -c $< -o $@

# Rule for linking gemm_example
gemm_example: gemm_example.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_GEMM: $(LDFLAGS_GEMM)"
	$(HIPCC) $(CFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ gemm_example.o $(LDFLAGS_GEMM)

clean:
	@echo "Cleaning up..."
	rm -f amdsmi_example amdsmi_example.o gemm_example gemm_example.o *.exe

checkpath:
	@echo "--- Variables (with Makefile_comp_tests.target potentially commented out) ---"
	@echo "PAPI_ROCM_ROOT = $(PAPI_ROCM_ROOT)"
	@echo "HIP_PATH = $(HIP_PATH)"
	@echo "HIPCC = $(HIPCC)"
	@echo "PAPI_DIR_INCLUDE = $(PAPI_DIR_INCLUDE)"
	@echo "--- Manually set CFLAGS, OPTFLAGS, etc. (if include is commented) ---"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "OPTFLAGS = $(OPTFLAGS)"
	@echo "UTILOBJS = $(UTILOBJS)"
	@echo "PAPILIB = $(PAPILIB)"
	@echo "BASE_INCLUDES_FROM_LOG = $(BASE_INCLUDES_FROM_LOG)"
	@echo "--- Effective values ---"
	@echo "EFFECTIVE_INCLUDE = $(EFFECTIVE_INCLUDE)"
	@echo "COMMON_LDFLAGS = $(COMMON_LDFLAGS)"
	@echo "LDFLAGS_AMDSMI = $(LDFLAGS_AMDSMI)"
	@echo "LDFLAGS_GEMM = $(LDFLAGS_GEMM)"

.PHONY: all clean checkpath amdsmi_example gemm_example