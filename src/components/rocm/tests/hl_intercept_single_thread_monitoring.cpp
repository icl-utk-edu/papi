/**
 * @file   hl_intercepte_single_thread_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#include "common.h"
#include <omp.h>

int quiet;

int main(int argc, char *argv[])
{
    int papi_errno = PAPI_OK;
    hipError_t hip_errno = hipSuccess;
    quiet = tests_quiet(argc, argv);

    test_skip(__FILE__, __LINE__, "", papi_errno);

    setenv("HSA_TOOLS_LIB", "librocprofiler64.so", 0);
    setenv("ROCP_TOOL_LIB", "libpapi.so", 0);
    setenv("ROCP_HSA_INTERCEPT", "1", 1);

    /* NOTE: following a HIP call is made before PAPI_library_init.
     *       In such cases, ROCP_TOOL_LIB should be set to libpapi.so.
     *       This allows PAPI to set the rocprofiler environment through
     *       hsa_init() triggered by the HIP call */
    int dev_count;
    hip_errno = hipGetDeviceCount(&dev_count);
    if (hip_errno != hipSuccess) {
        test_fail(__FILE__, __LINE__, "hipGetDeviceCount", hip_errno);
    }

    omp_set_num_threads(dev_count);

#define NUM_EVENTS (2)
    const char *events[NUM_EVENTS] = {
        "rocm:::SQ_INSTS_VALU",
        "rocm:::SQ_WAVES",
    };

    char event_list[512] = { 0 };
    unsigned i, off = 0;
    for (i = 0; i < dev_count; ++i) {
        unsigned j;
        for (j = 0; j < NUM_EVENTS; ++j) {
            char event[64] = { 0 };
            sprintf(event, "%s:device=%d,", events[j], i);
            strncpy(event_list + off, event, strlen(event));
            off += strlen(event);
        }
    }

    event_list[off - 1] = 0;
    setenv("PAPI_EVENTS", event_list, 1);
    setenv("PAPI_HL_THREAD_MULTIPLE", "0", 1);

    papi_errno = PAPI_hl_region_begin("matmul_intercept");
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_hl_region_begin", papi_errno);
    }

#pragma omp parallel
    {
    int thread_num = omp_get_thread_num();
    hipStream_t stream;
    hipSetDevice(thread_num);
    hip_errno = hipStreamCreate(&stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamCreate", hip_errno);
    }

    void *handle;
    hip_do_matmul_init(&handle);
    hip_do_matmul_work(handle, stream);
    hip_errno = hipStreamSynchronize(stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamSynchronize", hip_errno);
    }
    hip_do_matmul_cleanup(&handle);
    }

    papi_errno = PAPI_hl_region_end("matmul_intercept");
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_hl_region_end", papi_errno);
    }

    PAPI_hl_stop();
    test_hl_pass(__FILE__);
    return 0;
}
