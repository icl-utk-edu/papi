//
//	Copyright 2022 Hewlett Packard Enterprise Development LP
//

//
// This software contains source code provided by NVIDIA Corporation
//
// According to the Nvidia EULA (CUDA Toolkit v11.6.1)
// https://docs.nvidia.com/cuda/eula/index.html
//
// Chapter 1.1 License
// 1.1.1 License Grant
// Subject to the terms of this Agreement, NVIDIA hereby grants you a non-exclusive,
// non-transferable license, without the right to sublicense (except as expressly
// provided in this Agreement) to:
//   1. Install and use the SDK,
//   2. Modify and create derivative works of sample source code delivered in the SDK, and
//   3. Distribute those portions of the SDK that are identified in this Agreement as
//      distributable, as incorporated in object code format into a software application
//      that meets the distribution requirements indicated in this Agreement.
//
// 1.1.2 Distribution Requirements
// These are the distribution requirements for you to exercise the distribution grant:
//    1. Your application must have material additional functionality, beyond the included
//       portions of the SDK.
//    2. The distributable portions of the SDK shall only be accessed by your application.
//    3. The following notice shall be included in modifications and derivative works of
//       sample source code distributed: “This software contains source code provided by
//       NVIDIA Corporation.”
//    4. Unless a developer tool is identified in this Agreement as distributable, it is
//       delivered for your internal use only.
//    5. The terms under which you distribute your application must be consistent with the
//       terms of this Agreement, including (without limitation) terms relating to the license
//       grant and license restrictions and protection of NVIDIA’s intellectual property rights.
//       Additionally, you agree that you will protect the privacy, security and legal rights of
//       your application users.
//    6. You agree to notify NVIDIA in writing of any known or suspected distribution or use of
//       the SDK not in compliance with the requirements of this Agreement, and to enforce the
//       terms of your agreements with respect to distributed SDK.
//

//
// This program acts as a simple multi-kernel launch PAPI_read monitoring test.  It works by
// creating an eventset to track the number of warps launched on device 0, taking various
// read measurements before and after CUDA invocations to inspect the counter behavior.
// Note: the GPU kernel used in this test is adapted from the NVIDIA CUPTI callback_profiling
// sample code and has been instrumented with little concern for overall performance.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_version.h>

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#define PAPI_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    int _status = apiFuncCall;                                                 \
    if (_status != PAPI_OK) {                                                  \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, PAPI_strerror(_status));     \
        test_fail(__FILE__, __LINE__, #apiFuncCall, _status);                  \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        test_fail(__FILE__, __LINE__, #apiFuncCall, _status);                  \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        test_fail(__FILE__, __LINE__, #apiFuncCall, _status);                  \
    }                                                                          \
} while (0)

__global__
void vecAddKernel(const int* A, const int* B, int* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

static void
initVec(int* vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = i;
}

static void
cleanUp(int* h_A, int* h_B, int* h_C, int* d_A, int* d_B, int* d_C) {
    if (d_A)
        RUNTIME_API_CALL(cudaFree(d_A));
    if (d_B)
        RUNTIME_API_CALL(cudaFree(d_B));
    if (d_C)
        RUNTIME_API_CALL(cudaFree(d_C));

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
}

static void
vectorAdd() {
    int N = 32000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int* h_A, * h_B, * h_C;
    int* d_A, * d_B, * d_C;
    int i, sum;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (0) printf("Launching kernel: blocks %d, thread/block %d\n",
        blocksPerGrid, threadsPerBlock);

    vecAddKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    if (cudaGetLastError() != cudaSuccess) {
        printf("Vector addition kernel execution failed.\n");
    }

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i) {
        sum = h_A[i] + h_B[i];
        if (h_C[i] != sum) {
            fprintf(stderr, "Error: result verification failed.\n");
        }
    }

    cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
}

int main(int argc, char **argv) {
    int quiet = tests_quiet(argc, argv);

    int retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed\n" );
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed", retval);
    }

    if (!quiet) {
        printf( "PAPI version: %d.%d.%d.%d\n\n",
            PAPI_VERSION_MAJOR(PAPI_VERSION), PAPI_VERSION_MINOR(PAPI_VERSION),
            PAPI_VERSION_REVISION(PAPI_VERSION), PAPI_VERSION_INCREMENT(PAPI_VERSION));
    }

    // Determine number of CUDA devices
    int deviceCount = 0;
    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("There are no CUDA supported devices.\n");
        test_fail(__FILE__, __LINE__, "There are no CUDA supported devices", 0);
    }

    // Set up device managment contexts and report characteristics
    const int target_device = 0;
    CUdevice device;
    CUcontext ctx, poppedCtx;
    char deviceName[64];
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    int runtimeVersion = 0, driverVersion = 0;
    DRIVER_API_CALL(cuDeviceGet(&device, target_device));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 64, device));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    DRIVER_API_CALL(cuCtxCreate(&ctx, 0, device));  // Automatically pushes new context on stack.
    DRIVER_API_CALL(cuCtxPopCurrent(&poppedCtx));   // Take it off stack.

    RUNTIME_API_CALL(cudaRuntimeGetVersion(&runtimeVersion));
    RUNTIME_API_CALL(cudaDriverGetVersion(&driverVersion));

    if (!quiet) {
        printf("CUDA device information:\n");
        printf("\tCUDA Device %d: %s : computeCapability %d.%d runtimeVersion %d.%d driverVersion %d.%d\n",
            target_device, deviceName, computeCapabilityMajor, computeCapabilityMinor,
            runtimeVersion/1000, (runtimeVersion%100)/10, driverVersion/1000, (driverVersion%100)/10);
    }

    // Find CUDA component
    int cid = -1;
    int cids = PAPI_num_components();
    for (int i = 0; i < cids && cid < 0; ++i) {
        PAPI_component_info_t *component = (PAPI_component_info_t*) PAPI_get_component_info(i);
        if (component == NULL) {
            fprintf(stderr, "PAPI_get_component_info(%d) failed.\n", i);
            test_fail(__FILE__, __LINE__, "PAPI_get_component_info() failed", 0);
        }
        if (!strcmp("cuda", component->name)) { // Found it
            cid = i;
        }
    }
    if (cid < 0) {
        fprintf(stderr, "Failed to find cuda component.\n");
        test_fail(__FILE__, __LINE__, "Failed to find cuda component", 0);
    }

    // Setup PAPI counters
    int EventSet = PAPI_NULL;
    PAPI_API_CALL(PAPI_create_eventset(&EventSet));
    PAPI_API_CALL(PAPI_assign_eventset_component(EventSet, cid));

    if (!quiet) {
        printf("\nAdding warps launched counter to target device %d:\n", target_device);
    }
    char cuda_event[50];
    // Account for counter name difference between legacy event API and current profiler API
#if CUPTI_API_VERSION < 13
    if (computeCapabilityMajor < 7 || (computeCapabilityMajor == 7 && computeCapabilityMinor == 0)) {
        sprintf(cuda_event, "cuda:::event:warps_launched:device=%d", target_device);
    } else {
        test_fail(__FILE__, __LINE__, "Incompatible cupti version", 0);
    }
#else
    if (computeCapabilityMajor < 7) {
        sprintf(cuda_event, "cuda:::event:warps_launched:device=%d", target_device);
    } else {
        sprintf(cuda_event, "cuda:::sm__warps_launched.sum:device=%d", target_device);
    }
#endif

    // Try adding counter to event set
    DRIVER_API_CALL(cuCtxSetCurrent(ctx));
    retval = PAPI_add_named_event(EventSet, cuda_event);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_named_event failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
        test_fail(__FILE__, __LINE__, "PAPI_add_named_event failed", retval);
    }
    if (!quiet) {
        printf("\tAdd event successful: '%s'\n", cuda_event);
    }

    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_start failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }

    // Set up multi-read test buffers/variables
    const int MAX_TESTS = 10;
    int numTests = 0;
    bool successful;
    long long values[1] = {-1};
    long long testResults[MAX_TESTS] = {-1};

    if (!quiet) {
        printf("\nStarting tests:\n");
    }

    // Test read prior to invoking any CUDA code,
    // counter value should be zero.
    retval = PAPI_read(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_read failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }
    successful = (values[0] == 0);
    if (successful) {
        if (!quiet) printf("\tTest 1 passed\n");
    } else {
        test_fail(__FILE__, __LINE__, "Test 1 failed", 0);
    }
    testResults[numTests++] = values[0];


    // Test read after invoking CUDA code,
    // counter value should be non-zero.
    vectorAdd();
    retval = PAPI_read(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_read failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }
    successful = (values[0] != 0);
    if (successful) {
        if (!quiet) printf("\tTest 2 passed\n");
    } else {
        test_fail(__FILE__, __LINE__, "Test 2 failed", 0);
    }
    testResults[numTests++] = values[0];


    // Test read after no additional CUDA code invocations,
    // counter value should be equal to previous measurement.
    retval = PAPI_read(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_read failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }
    successful = (values[0] == testResults[numTests - 1]);
    if (successful) {
        if (!quiet) printf("\tTest 3 passed\n");
    } else {
        test_fail(__FILE__, __LINE__, "Test 3 failed", 0);
    }
    testResults[numTests++] = values[0];


    // Test read after invoking same CUDA code twice,
    // counter value should be 3x previous measurement.
    vectorAdd();
    vectorAdd();
    retval = PAPI_read(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_read failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }
    successful = (values[0] == (3 * testResults[numTests - 1]));
    if (successful) {
        if (!quiet) printf("\tTest 4 passed\n");
    } else {
        test_fail(__FILE__, __LINE__, "Test 4 failed", 0);
    }
    testResults[numTests++] = values[0];


    // Test read from PAPI stop after invoking same CUDA code three more times,
    // counter value should be 2x previous measurement.
    vectorAdd();
    vectorAdd();
    vectorAdd();
    retval = PAPI_stop(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_stop failed, retval=%d [%s].\n", retval, PAPI_strerror(retval));
    }
    successful = (values[0] == (2 * testResults[numTests - 1]));
    if (successful) {
        if (!quiet) printf("\tTest 5 passed\n");
    } else {
        test_fail(__FILE__, __LINE__, "Test 5 failed", 0);
    }
    testResults[numTests++] = values[0];


    if (!quiet) {
        printf("\nCounter values from tests:\n");
        for (int test = 0; test < numTests; ++test) {
            printf("\tTest %d: %s = %lld\n", test, cuda_event, testResults[test]);
        }
    }

    PAPI_API_CALL(PAPI_cleanup_eventset(EventSet));
    PAPI_API_CALL(PAPI_destroy_eventset(&EventSet));
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
