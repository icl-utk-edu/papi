#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "papi.h"
#include "papi_test.h"
#include "gpu_work.h"

// Currently there is only Cuda preset support for the A100 and H100
#define NUM_SUPPORTED_PRESET_DEVS 2
const char *deviceNames[NUM_SUPPORTED_PRESET_DEVS] = {"A100", "H100"};

int main()
{
    int quietTests = 0;
    char *quietEnv = getenv("PAPI_CUDA_TEST_QUIET");
    if (quietEnv) {
        quietTests = (int) strtol(quietEnv, NULL, 10);
    }

    int devCount;
    cudaError_t cudaErr = cudaGetDeviceCount(&devCount);
    if (cudaErr != cudaSuccess) {
        printf("Failed cudaGetDeviceCount: %d\n", cudaErr);
        exit(1);
    }

    if (!quietTests) {
        printf("Total number of devices on the machine: %d\n", devCount);
    }

    char devIdxsThatSupportCudaPresets[PAPI_MAX_STR_LEN] = "";
    int devIdx, firstDevIdxThatSupportsCudaPresets = -1;
    for (devIdx = 0; devIdx < devCount; devIdx++) {
        cudaDeviceProp prop;
        cudaErr = cudaGetDeviceProperties(&prop, devIdx);
        if (cudaErr != cudaSuccess) {
            printf("Failed cudaGetDeviceProperties: %d\n", cudaErr);
        }

        int i;
        for (i = 0; i < NUM_SUPPORTED_PRESET_DEVS; i++) {
            if (strstr(prop.name, deviceNames[i])) {
                // Store the first device index that supports Cuda component presets, as this
                // index will be used to create the Cuda context
                if (firstDevIdxThatSupportsCudaPresets == -1) {
                    firstDevIdxThatSupportsCudaPresets = devIdx;
                }

                // Store all device indexes that support Cuda component presets
                int strLen = snprintf(devIdxsThatSupportCudaPresets + strlen(devIdxsThatSupportCudaPresets), PAPI_MAX_STR_LEN, "%d,", devIdx);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write supported device index for Cuda component presets.\n");
                    exit(1);
                }
            }
        }
    }

    if (strlen(devIdxsThatSupportCudaPresets) == 0) {
        fprintf(stderr, "Devices found on the machine do not support Cuda component presets. Currently only the A100 and H100 are supported.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    else {
        if (quietTests == 0) {
            devIdxsThatSupportCudaPresets[strlen(devIdxsThatSupportCudaPresets) - 1] = '\0';
            printf("Device indexes which support Cuda component presets: %s.\n", devIdxsThatSupportCudaPresets);
        }
    }

    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    const char *cmpName = "cuda";
    int cidx = PAPI_get_component_index(cmpName);
    if (cidx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index", papi_errno);
    }

    // Obtain the starting point for the Cuda component presets and initialize the component
    int eventCode = 0 | PAPI_PRESET_MASK;
    papi_errno = PAPI_enum_cmp_event(&eventCode, PAPI_ENUM_FIRST, cidx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", papi_errno);
    }

    if (quietTests == 0) {
        printf("\nCuda Component Presets: \n");
        printf("---------------------------------------------------------\n");
    }

    int numCudaPresets = 0, *cudaPresetCodes = NULL;
    char **cudaPresetNames = NULL;
    do {
        cudaPresetCodes = (int *) realloc(cudaPresetCodes, (1 + numCudaPresets) * sizeof(int));
        if (cudaPresetCodes == NULL) {
            fprintf(stderr, "Failed to allocate memory for cudaPresetCodes.\n");
            exit(1);
        }   
        cudaPresetCodes[numCudaPresets] = eventCode; 

        cudaPresetNames = (char **) realloc(cudaPresetNames, (1 + numCudaPresets) * sizeof(char *));
        if (cudaPresetNames == NULL) {
            fprintf(stderr, "Failed to allocate memory for the array cudaPresetNames.\n");
            exit(1);
        }
        cudaPresetNames[numCudaPresets] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char)); 
        if (cudaPresetNames[numCudaPresets] == NULL) {
            fprintf(stderr, "Failed to allocate memory for index %d in the array cudaPresetNames.\n", numCudaPresets);
            exit(1);
        }

        papi_errno = PAPI_event_code_to_name(cudaPresetCodes[numCudaPresets], cudaPresetNames[numCudaPresets]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", papi_errno);
        } 

        // Listing the Cuda component presets    
        if (quietTests == 0) {
            printf("- %s\n", cudaPresetNames[numCudaPresets]);
        }

        // Increment the total number of presets assigned
        numCudaPresets++;
    } while(PAPI_enum_cmp_event(&eventCode, PAPI_ENUM_EVENTS, cidx) == PAPI_OK);

    // Create a Cuda context for a device index that supports Cuda component presets
    CUcontext ctx;
    unsigned int flags = 0;
    CUresult cuErr = cuCtxCreate(&ctx, flags, firstDevIdxThatSupportsCudaPresets); 
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "Call to cuCtxCreate failed with error code %d.\n", cuErr);
        exit(1);
    }  

    int EventSet = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    if (quietTests == 0) {
        printf("\nAttempting to add, start, and stop Cuda component presets\n");
        printf("---------------------------------------------------------\n");
    }

    int idx;
    for (idx = 0; idx < numCudaPresets; idx++) {
        papi_errno = PAPI_add_event(EventSet, cudaPresetCodes[idx]);
        if (papi_errno != PAPI_OK) {
            if (papi_errno == PAPI_ENOEVNT) {
                if (quietTests == 0) {
                    printf("Cuda preset %s unable to be added to the EventSet. This is most likely due to the preset not being available for the GPU you are running on.\n", cudaPresetNames[idx]);
                }
                continue;
            }
            test_fail(__FILE__, __LINE__, "PAPI_add_event", papi_errno);
        }

        papi_errno = PAPI_start(EventSet);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
        } 

        // Do work
        int matrixSize = 100000;
        VectorAddSubtract(matrixSize, quietTests);

        long long counters;
        papi_errno = PAPI_stop(EventSet, &counters);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
        }
        if (quietTests == 0) {
            printf("Counter values for %s: %lld\n", cudaPresetNames[idx], counters);
        }

        papi_errno = PAPI_cleanup_eventset(EventSet);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
        } 
    }

    papi_errno = PAPI_destroy_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    // Destroy the created Cuda context
    cuErr = cuCtxDestroy(ctx);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "Call to cuCtxDestroy failed with error code %d.\n", cuErr);
        exit(1);
    }

    // Free allocated memory
    for (idx = 0; idx < numCudaPresets; idx++) {
        free(cudaPresetNames[idx]);
    }
    free(cudaPresetNames);
    free(cudaPresetCodes);

    test_pass(__FILE__);

    return 0; 
}
