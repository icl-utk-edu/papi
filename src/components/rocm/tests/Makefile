NAME = rocm
include ../../Makefile_comp_tests.target

CC       = $(PAPI_ROCM_ROOT)/hip/bin/hipcc
CXX      = $(PAPI_ROCM_ROOT)/hip/bin/hipcc
CPPFLAGS = -I$(PAPI_ROCM_ROOT)/hip/include      \
           -I$(PAPI_ROCM_ROOT)/hsa/include/hsa  \
           -I../../..                           \
           -I../../../testlib
LDFLAGS  = -L../../../libpfm4/lib -lpfm                    \
           -L../../.. -lpapi                               \
           ../../../testlib/test_utils.o                   \
           -L$(PAPI_ROCM_ROOT)/lib -lhsa-runtime64         \
           -fopenmp                                        \
           -Wl,-rpath="$(PWD)/../../../libpfm4/lib"        \
           -Wl,-rpath="$(PWD)/../../../"                   \
           -Wl,-rpath="$(PWD)/libpfm4/lib"                 \
           -Wl,-rpath="$(PWD)"
CXXFLAGS = -g -O0 -fopenmp

rocm_tests: ALL

ALL: sample_single_kernel_monitoring       \
     sample_single_thread_monitoring       \
     sample_multi_kernel_monitoring        \
     sample_multi_thread_monitoring        \
     sample_overflow_monitoring            \
     intercept_single_kernel_monitoring    \
     intercept_single_thread_monitoring    \
     intercept_multi_kernel_monitoring     \
     intercept_multi_thread_monitoring     \
     hl_sample_single_kernel_monitoring    \
     hl_sample_single_thread_monitoring    \
     hl_intercept_single_kernel_monitoring \
     hl_intercept_single_thread_monitoring \
     hl_intercept_multi_thread_monitoring

sample_single_kernel_monitoring: sample_single_kernel_monitoring.o matmul.o
sample_single_thread_monitoring: sample_single_thread_monitoring.o single_thread_monitoring.o matmul.o
sample_multi_thread_monitoring: sample_multi_thread_monitoring.o multi_thread_monitoring.o matmul.o
sample_multi_kernel_monitoring: sample_multi_kernel_monitoring.o multi_kernel_monitoring.o matmul.o
sample_overflow_monitoring: sample_overflow_monitoring.o matmul.o
intercept_single_kernel_monitoring: intercept_single_kernel_monitoring.o matmul.o
intercept_single_thread_monitoring: intercept_single_thread_monitoring.o single_thread_monitoring.o matmul.o
intercept_multi_thread_monitoring: intercept_multi_thread_monitoring.o multi_thread_monitoring.o matmul.o
intercept_multi_kernel_monitoring: intercept_multi_kernel_monitoring.o multi_kernel_monitoring.o matmul.o
hl_sample_single_kernel_monitoring: hl_sample_single_kernel_monitoring.o matmul.o
hl_sample_single_thread_monitoring: hl_sample_single_thread_monitoring.o matmul.o
hl_intercept_single_kernel_monitoring: hl_intercept_single_kernel_monitoring.o matmul.o
hl_intercept_single_thread_monitoring: hl_intercept_single_thread_monitoring.o matmul.o
hl_intercept_multi_thread_monitoring: hl_intercept_multi_thread_monitoring.o matmul.o

clean:
	rm -f sample_single_thread_monitoring
	rm -f sample_single_kernel_monitoring
	rm -f sample_multi_kernel_monitoring
	rm -f sample_multi_thread_monitoring
	rm -f sample_overflow_monitoring
	rm -f intercept_single_thread_monitoring
	rm -f intercept_single_kernel_monitoring
	rm -f intercept_multi_kernel_monitoring
	rm -f intercept_multi_thread_monitoring
	rm -f hl_sample_single_kernel_monitoring
	rm -f hl_sample_single_thread_monitoring
	rm -f hl_intercept_single_kernel_monitoring
	rm -f hl_intercept_single_thread_monitoring
	rm -f hl_intercept_multi_thread_monitoring
	rm -f *.o
