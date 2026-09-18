#ifndef CUDA_RANGE_PROFILER_HPP
#define CUDA_RANGE_PROFILER_HPP

// POSIX standard headers.
#include <dlfcn.h>

// Internal headers.
#ifdef __cplusplus
extern "C" {
#endif
#include "papi_debug.h"
#ifdef __cplusplus
}
#endif

#define CHECK_SNPRINTF_CALL(snprintf_call, type, maxLength)                                                                  \
do {                                                                                                                         \
    int strLen = snprintf_call;                                                                                              \
    if (strLen < 0 || (type) strLen >= maxLength) {                                                                          \
        SUBDBG("The snprintf call (%s) failed to fully write additional arguments into the buffer.\n", #snprintf_call);      \
        return PAPI_EBUF;                                                                                                    \
    }                                                                                                                        \
} while (0)

#define CHECK_INTERNAL_FUNC_CALL(internal_call)                                                                              \
do {                                                                                                                         \
    int papi_errno = internal_call;                                                                                          \
    if (papi_errno != PAPI_OK) {                                                                                             \
        SUBDBG("The call %s failed with error code %d.\n", #internal_call, papi_errno);                                      \
        return papi_errno;                                                                                                   \
    }                                                                                                                        \
} while (0)

#define CHECK_DRIVER_API_CALL(driver_api_call)                                                                               \
do                                                                                                                           \
{                                                                                                                            \
    CUresult status = driver_api_call;                                                                                       \
    if (status != CUDA_SUCCESS) {                                                                                            \
        SUBDBG("The call %s failed with error code %d.\n", #driver_api_call, status);                                        \
        return PAPI_ESYS;                                                                                                    \
    }                                                                                                                        \
} while (0)

#define CHECK_RUNTIME_API_CALL(runtime_api_call)                                                                             \
do                                                                                                                           \
{                                                                                                                            \
    cudaError_t status = runtime_api_call;                                                                                   \
    if (status != cudaSuccess) {                                                                                             \
        SUBDBG("The call %s failed with error code %d.\n", #runtime_api_call, status);                                       \
        return PAPI_ESYS;                                                                                                    \
    }                                                                                                                        \
} while (0)

#define CHECK_CUPTI_API_CALL(cupti_api_call)                                                                                 \
do                                                                                                                           \
{                                                                                                                            \
    CUptiResult status = cupti_api_call;                                                                                     \
    if (status != CUPTI_SUCCESS) {                                                                                           \
        SUBDBG("The call %s failed with error code %d.\n", #cupti_api_call, status);                                         \
        return PAPI_ESYS;                                                                                                    \
    }                                                                                                                        \
} while (0)

#define ASSIGN_AND_CHECK_DLSYM_API_CALL(function_ptr, typecast, handle, symbol)                                              \
do {                                                                                                                         \
    function_ptr = (typecast) dlsym(handle, symbol);                                                                         \
    if (function_ptr == NULL) {                                                                                              \
        SUBDBG("The call to dlsym failed due to -- %s.\n", dlerror());                                                       \
        return PAPI_ESYS;                                                                                                    \
    }                                                                                                                        \
} while(0)


/**
 * \brief Params for cupti_metric_info_t
 */
typedef struct cupti_metric_info_t
{
    /// [in] The type of CUPTI metric (i.e. counter, ratio, throughput).
    CUpti_MetricType cupti_metric_type;
    /// [in] The integer position in the std::map.
    size_t name_id;
    /// [in] The statistics for a metric (i.e. avg, max, min, sum).
    std::vector<std::string> stats;
    /// [in] The submetrics for a metric (i.e. peak_sustatined, peak_sustained_active, peak_sustained_elapsed,
    /// per_cycle_active, per_cycle_elapsed, per_second, pct_of_peak_sustained_active, pct_of_peak_sustained_elapsed).
    std::vector<std::string> submetrics;
    /// [in] The device indices the metric appears on. 
    std::vector<std::string> device_ids;
    /// [in] The description of the metric.
    std::string description;
    /// [in] The number of passes required for collection.
    size_t num_passes_for_collection;
} cupti_metric_info_t;

/**
 * \brief Params for native_event_info_t
 */
typedef struct native_event_info_t
{
    /// [in] Index corresponding to stat value.
    size_t stat;
    /// [in] Index corresponding to submetric value.
    size_t submetric;
    /// [in] Index corresponding to NVIDIA device.
    size_t device;
    /// [in] Flagged bits indicating if stat, submetric, or device qualifier exists for the event.  
    size_t flags;
    /// [in] Index corresponding to CUPTI metricname.
    size_t name_id;
} native_event_info_t;

#endif // CUDA_RANGE_PROFILER_HPP
