NAME=template
include ../../Makefile_comp_tests.target

CFLAGS    = $(OPTFLAGS)
CPPFLAGS += $(INCLUDE)
LDFLAGS  += $(PAPILIB) $(TESTLIB) $(UTILOBJS)

TESTS = simple advanced two_eventsets simple_sampling
#TESTS = simple_sampling
template_tests: $(TESTS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPTFLAGS) -c -o $@ $<

kernel.o: kernel.cpp
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -D__HIP_ROCclr__=1 -O2 -g -DNDEBUG -std=gnu++17 --offload-arch=gfx90a --offload-arch=gfx90a -W -Wall -Wextra -Wshadow -o kernel.o -x hip -c kernel.cpp

simple: simple.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG --offload-arch=gfx90a --offload-arch=gfx90a --hip-link --rtlib=compiler-rt -unwindlib=libgcc simple.o kernel.o -o simple $(PAPI_ROCP_SDK_ROOT)/lib/libamdhip64.so.6 $(LDFLAGS)

advanced: advanced.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG --offload-arch=gfx90a --offload-arch=gfx90a --hip-link --rtlib=compiler-rt -unwindlib=libgcc advanced.o kernel.o -o advanced $(PAPI_ROCP_SDK_ROOT)/lib/libamdhip64.so.6 $(LDFLAGS)

two_eventsets: two_eventsets.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG --offload-arch=gfx90a --offload-arch=gfx90a --hip-link --rtlib=compiler-rt -unwindlib=libgcc two_eventsets.o kernel.o -o two_eventsets $(PAPI_ROCP_SDK_ROOT)/lib/libamdhip64.so.6 $(LDFLAGS)

simple_sampling: simple_sampling.o kernel.o
	$(PAPI_ROCP_SDK_ROOT)/bin/amdclang++ -O2 -g -DNDEBUG --offload-arch=gfx90a --offload-arch=gfx90a --hip-link --rtlib=compiler-rt -unwindlib=libgcc simple_sampling.o kernel.o -o simple_sampling $(PAPI_ROCP_SDK_ROOT)/lib/libamdhip64.so.6 $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
