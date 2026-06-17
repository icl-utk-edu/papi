#ifndef NVML_TESTS_HELPER_H
#define NVML_TESTS_HELPER_H

void enumerate_and_store_nvml_native_events(char ***nvml_native_event_names_arg, int *total_event_count_arg, int *nvidia_device_index_arg);

// Define to handle memory allocation checks
#define check_memory_allocation_call(var)                                       \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

// Define to handle suppress print output for the cuda component tests
#define PRINT(global_suppress_output, format, args...)                          \
{                                                                               \
    if (!global_suppress_output) {                                              \
        fprintf(stdout, format, ## args);                                       \
    }                                                                           \
}

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

#endif // NVML_TESTS_HELPER_H
