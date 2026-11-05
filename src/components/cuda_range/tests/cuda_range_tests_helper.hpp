#ifndef CUDA_RANGE_TESTS_HELPER_HPP
#define CUDA_RANGE_TESTS_HELPER_HPP

// C++ STL headers.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// CTK headers.
#include <cuda.h>

// Define to handle memory allocation checks.
#define CHECK_MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                                  \
    if (var == NULL) {                                                                \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",                  \
                __FILE__, __LINE__);                                                  \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
} while (0)

// Define to handle PAPI API calls.
#define CHECK_PAPI_API_CALL(api_function_call)                                       \
do {                                                                                 \
    int papi_errno = api_function_call;                                              \
    if (papi_errno != PAPI_OK) {                                                     \
        test_fail(__FILE__, __LINE__, #api_function_call, papi_errno);               \
    }                                                                                \
} while (0)

// Define to handle CUDA runtime API calls.
#define CHECK_CUDA_RUNTIME_API_CALL(api_function_call)                               \
do {                                                                                 \
    cudaError_t _status = api_function_call;                                         \
    if (_status != cudaSuccess) {                                                    \
        fprintf(stderr, "Call to %s on line %d failed with error code %d.\n",        \
                #api_function_call, __LINE__, _status);                              \
        exit(EXIT_FAILURE);                                                          \
    }                                                                                \
} while (0)


// Define to handle CUDA driver API calls.
#define CHECK_CUDA_DRIVER_API_CALL(api_function_call)                                \
do {                                                                                 \
    CUresult _status = api_function_call;                                            \
    if (_status != CUDA_SUCCESS) {                                                   \
        fprintf(stderr, "Call to %s on line %d failed with error code %d.\n",        \
                #api_function_call, __LINE__, _status);                              \
        exit(EXIT_FAILURE);                                                          \
    }                                                                                \
} while (0)

// Define to handle CUPTI API calls.
#define CHECK_CUPTI_API_CALL(api_function_call)                                      \
do {                                                                                 \
    CUptiResult _status = api_function_call;                                         \
    if (_status != CUPTI_SUCCESS) {                                                  \
        fprintf(stderr, "Call to %s on line %d failed with error code %d.\n",        \
                #api_function_call, __LINE__, _status);                              \
        exit(EXIT_FAILURE);                                                          \
    }                                                                                \
} while (0)

void get_cuda_range_native_event_name(int cuda_range_cmp_index, int &device_index, std::vector<std::string> &cuda_range_native_event_names);
void get_cuda_range_native_event_name(int cuda_range_cmp_index, std::string &cuda_range_native_event_name);

#endif // CUDA_RANGE_TESTS_HELPER_HPP
