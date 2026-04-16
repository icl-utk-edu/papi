NAME=template
include ../../Makefile_comp_tests.target

ROCP_SDK_INCL=-I$(PAPI_ROCP_SDK_ROOT)/include     \
              -I$(PAPI_ROCP_SDK_ROOT)/include/hsa \
              -I$(PAPI_ROCP_SDK_ROOT)/hsa/include \
			  -I$(PAPI_ROCP_SDK_ROOT)/include/rocprofiler-sdk

# If ROCprofiler_SDK is installed at a different
# path than the rest of ROCm, then the path to
# ROCm should be specified via PAPI_ROCM_ROOT
ifneq ($(PAPI_ROCM_ROOT),)
ROCP_SDK_INCL+=-I$(PAPI_ROCM_ROOT)/include     \
              -I$(PAPI_ROCM_ROOT)/include/hsa \
              -I$(PAPI_ROCM_ROOT)/hsa/include \
              -I$(PAPI_ROCM_ROOT)/include/rocprofiler-sdk
endif

AMDCXX   ?= amdclang++
CFLAGS    = $(OPTFLAGS)
INCFLAGS += $(INCLUDE) $(ROCP_SDK_INCL)
CPPFLAGS += $(INCFLAGS)
LDFLAGS  += $(PAPILIB) $(TESTLIB) $(UTILOBJS)
LDFLAGS  += -L$(PAPI_ROCP_SDK_ROOT)/lib -lrocprofiler-sdk-roctx

GPUARCH = $(shell rocm_agent_enumerator 2>/dev/null | grep -v "gfx000" | head -1)
ifneq ($(GPUARCH),)
    ARCHFLAG=--offload-arch=$(GPUARCH)
endif
GPUFLAGS=$(ARCHFLAG) --hip-link --rtlib=compiler-rt -unwindlib=libgcc

TESTS = simple advanced two_eventsets simple_sampling roctx_pause_resume
template_tests: $(TESTS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPTFLAGS) -D__HIP_PLATFORM_AMD__ -c -o $@ $<

kernel.o: kernel.cpp
	$(AMDCXX) -D__HIP_ROCclr__=1 -O2 -g -DNDEBUG $(INCFLAGS) $(ARCHFLAG) -W -Wall -Wextra -Wshadow -o kernel.o -x hip -c kernel.cpp

helper_rocp_sdk.o: helper_rocp_sdk.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $^

simple: simple.o kernel.o helper_rocp_sdk.o
	$(AMDCXX) -O2 -g -DNDEBUG $(GPUFLAGS) simple.o kernel.o helper_rocp_sdk.o -o simple $(LDFLAGS)

advanced: advanced.o kernel.o helper_rocp_sdk.o
	$(AMDCXX) -O2 -g -DNDEBUG $(GPUFLAGS) advanced.o kernel.o helper_rocp_sdk.o -o advanced $(LDFLAGS)

two_eventsets: two_eventsets.o kernel.o helper_rocp_sdk.o
	$(AMDCXX) -O2 -g -DNDEBUG $(GPUFLAGS) two_eventsets.o kernel.o helper_rocp_sdk.o -o two_eventsets $(LDFLAGS)

simple_sampling: simple_sampling.o kernel.o helper_rocp_sdk.o
	$(AMDCXX) -O2 -g -DNDEBUG $(GPUFLAGS) simple_sampling.o kernel.o helper_rocp_sdk.o -o simple_sampling $(LDFLAGS) -pthread

roctx_pause_resume.o: roctx_pause_resume.cpp
	$(AMDCXX) -O2 -g -DNDEBUG $(INCFLAGS) -x hip -D__HIP_PLATFORM_AMD__ -c roctx_pause_resume.cpp

roctx_pause_resume: roctx_pause_resume.o kernel.o helper_rocp_sdk.o
	$(AMDCXX) -O2 -g -DNDEBUG $(GPUFLAGS) roctx_pause_resume.o kernel.o helper_rocp_sdk.o -o roctx_pause_resume $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
