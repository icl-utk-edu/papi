# ROCm + PAPI build settings
# Set PAPI_AMDSMI_ROOT to your ROCm install prefix (e.g., /opt/rocm or /opt/rocm-6.4.0).
# Used to find AMD SMI / HIP / ROCm headers and libraries. You can override on the command line.
PAPI_AMDSMI_ROOT ?= /opt/rocm-6.4.0
HIP_PATH := $(PAPI_AMDSMI_ROOT)
HIPCC := $(HIP_PATH)/bin/hipcc

# Toolchain / flags
CFLAGS   ?= -DPAPI_NUM_COMP=3
CSTD     ?= -std=c11
HIPFLAGS ?= $(CFLAGS) -std=c++17
OPTFLAGS ?= -O2

# Libraries in the PAPI tree (override if your layout differs)
UTILOBJS ?= ../../../testlib/libtestlib.a
PAPILIB  ?= ../../../libpapi.a

# Include paths (adjust if your PAPI install lives elsewhere)
BASE_INCLUDES_FROM_LOG ?= -I. -I../../.. -I../../../testlib -I../../../validation_tests
PAPI_DIR_INCLUDE ?= ../../../include

EFFECTIVE_INCLUDE := $(BASE_INCLUDES_FROM_LOG)
EFFECTIVE_INCLUDE += -I$(PAPI_DIR_INCLUDE)
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include/rocm_smi
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include/hip
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include/hsa
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include/rocprofiler
EFFECTIVE_INCLUDE += -I$(PAPI_AMDSMI_ROOT)/include/rocblas

# Linker flags
COMMON_LDFLAGS  = -ldl -g -lpthread
LDFLAGS_AMDSMI  = $(UTILOBJS) $(PAPILIB) -L$(PAPI_AMDSMI_ROOT)/lib -lamd_smi $(COMMON_LDFLAGS)
LDFLAGS_GEMM    = $(UTILOBJS) $(PAPILIB) -L$(PAPI_AMDSMI_ROOT)/lib -lrocblas $(COMMON_LDFLAGS)

# AMD SMI tests to build with 'make tests'
TESTS_AMDSMI := amdsmi_energy_monotonic amdsmi_ctx_conflict

# Default target
all: amdsmi_hello amdsmi_basics amdsmi_gemm $(TESTS_AMDSMI)

# ------------------------
# Compile rules
# ------------------------

# .c -> .o
%.o: %.c
	@echo "Compiling C source $< to $@..."
	@echo "HIPCC: $(HIPCC)"
	@echo "CFLAGS: $(CFLAGS) CSTD: $(CSTD) OPTFLAGS: $(OPTFLAGS)"
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -c $< -o $@

# amdsmi_gemm.c is compiled as HIP
amdsmi_gemm.o: amdsmi_gemm.c
	@echo "Compiling HIP source $< to $@..."
	@echo "HIPCC: $(HIPCC)"
	@echo "HIPFLAGS: $(HIPFLAGS) OPTFLAGS: $(OPTFLAGS)"
	@echo "INCLUDES: $(EFFECTIVE_INCLUDE)"
	$(HIPCC) $(HIPFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -x hip -c $< -o $@

# ------------------------
# Link rules
# ------------------------

# amdsmi_hello replaces the older amdsmi_example
amdsmi_hello: amdsmi_hello.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_hello.o $(LDFLAGS_AMDSMI)

# Back-compat alias: allow 'make amdsmi_example' to build amdsmi_hello
amdsmi_example: amdsmi_hello
	@true

amdsmi_basics: amdsmi_basics.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_basics.o $(LDFLAGS_AMDSMI)

# GEMM sample linked against rocBLAS
amdsmi_gemm: amdsmi_gemm.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_GEMM: $(LDFLAGS_GEMM)"
	$(HIPCC) $(HIPFLAGS) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_gemm.o $(LDFLAGS_GEMM)

# Tests
amdsmi_energy_monotonic: amdsmi_energy_monotonic.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_energy_monotonic.o $(LDFLAGS_AMDSMI)

amdsmi_ctx_conflict: amdsmi_ctx_conflict.o $(UTILOBJS) $(PAPILIB)
	@echo "Linking $@..."
	@echo "Using LDFLAGS_AMDSMI: $(LDFLAGS_AMDSMI)"
	$(HIPCC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(EFFECTIVE_INCLUDE) -o $@ amdsmi_ctx_conflict.o $(LDFLAGS_AMDSMI)

$(UTILOBJS):
	$(MAKE) -C ../../../testlib libtestlib.a

# Convenience meta-target
tests: $(TESTS_AMDSMI)

# ------------------------
# Cleanup and diagnostics
# ------------------------

clean:
	@echo "Cleaning up..."
	rm -f \
	  amdsmi_hello amdsmi_hello.o \
	  amdsmi_basics amdsmi_basics.o \
	  amdsmi_gemm amdsmi_gemm.o \
	  amdsmi_energy_monotonic amdsmi_energy_monotonic.o \
	  amdsmi_ctx_conflict amdsmi_ctx_conflict.o \
	  *.exe

checkpath:
	@echo "--- Variables ---"
	@echo "PAPI_AMDSMI_ROOT = $(PAPI_AMDSMI_ROOT)"
	@echo "HIPCC = $(HIPCC)"
	@echo "INCLUDES = $(EFFECTIVE_INCLUDE)"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "CSTD = $(CSTD)"
	@echo "HIPFLAGS = $(HIPFLAGS)"
	@echo "OPTFLAGS = $(OPTFLAGS)"
	@echo "UTILOBJS = $(UTILOBJS)"
	@echo "PAPILIB = $(PAPILIB)"
	@echo "COMMON_LDFLAGS = $(COMMON_LDFLAGS)"
	@echo "LDFLAGS_AMDSMI = $(LDFLAGS_AMDSMI)"
	@echo "LDFLAGS_GEMM = $(LDFLAGS_GEMM)"
	@echo "TESTS_AMDSMI = $(TESTS_AMDSMI)"

.PHONY: all clean checkpath tests \
	amdsmi_hello amdsmi_example amdsmi_basics amdsmi_gemm \
	amdsmi_energy_monotonic amdsmi_ctx_conflict
