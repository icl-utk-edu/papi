#ifndef CUDA_TESTS_HELPER_H
#define CUDA_TESTS_HELPER_H
#include <stdio.h>
#include <stdlib.h>

void add_cuda_native_events(int EventSet, const char *cuda_native_event_name, int *num_events_successfully_added, char **events_successfully_added, int *numMultipassEvents);
int determine_if_device_is_enabled(int device_idx);
void enumerate_and_store_cuda_native_events(char ***cuda_native_event_names, int *total_event_count, int *cuda_device_index);

// Define to handle suppress print output for the cuda component tests
#define PRINT(global_suppress_output, format, args...)                          \
{                                                                               \
    if (!global_suppress_output) {                                              \
        fprintf(stderr, format, ## args);                                       \
    }                                                                           \
}                                                                               \

// Define to handle memory allocation checks
#define check_memory_allocation_call(var)                                       \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

// Define to handle PAPI api calls
#define check_papi_api_call(apiFuncCall)                                       \
do {                                                                           \
    int papi_errno = apiFuncCall;                                              \
    if (papi_errno != PAPI_OK) {                                               \
        test_fail(__FILE__, __LINE__, #apiFuncCall, papi_errno);               \
    }                                                                          \
} while (0)

// Define's to handle Cuda API calls
#define check_cuda_runtime_api_call(apiFuncCall)                               \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "Call to %s on line %d failed with error code %d.\n",  \
                #apiFuncCall, __LINE__, _status);                              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define check_cuda_driver_api_call(apiFuncCall)                                \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "Call to %s on line %d failed with error code %d.\n",  \
                #apiFuncCall, __LINE__, _status);                              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)
#endif // CUDA_TESTS_HELPER_H
