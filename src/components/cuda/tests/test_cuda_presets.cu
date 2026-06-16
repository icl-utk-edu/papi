/**
* @file test_cuda_presets.cu
* @brief This test attempts to add, start, and stop cuda component presets for the GA100
*        and GH200. Note that this test only works with the Perfworks Metrics API.
*
*/


// Standard library headers
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Cuda Toolkit headers
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <cuda_runtime_api.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"
#include "gpu_work.h"

static void print_help_message(void)
{
    printf("./test_cuda_presets\n"
           "Notes:\n"
           "1. This test attempts to add the cuda component presets for the GA100 and GH200. If neither are present on the system"
           " the test will be skipped.\n"
           "2. Only the first occurrence for the GA100 and GH200 will be profiled with their respective presets.\n"
           "3. Must be using the Perfworks Metrics API.\n");
}

int main(int argc, char **argv)
{
    int suppressOutput = 0;
    char *user_defined_suppressOutput = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppressOutput) {
        suppressOutput = (int) strtol(user_defined_suppressOutput, (char**) NULL, 10);
    }   
    PRINT(suppressOutput, "Running the cuda component test test_cuda_presets.cu\n");

    if (argc > 1) {
        print_help_message();
        exit(EXIT_SUCCESS);
    }

    char *papi_cuda_api = getenv("PAPI_CUDA_API");
    if (papi_cuda_api != NULL) {
        fprintf(stderr, "The test test_cuda_presets only works with the Perfworks Metrics API. Unset the environment variable PAPI_CUDA_API.\n");
        test_skip(__FILE__, __LINE__, "", 0); 
    } 

    // Determine the number of Cuda capable devices
    int numDevicesOnMachine = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&numDevicesOnMachine) );
    // No devices detected on the machine, exit
    if (numDevicesOnMachine < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run. Exiting.\n");
        exit(EXIT_FAILURE);
    }

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    profilerInitializeParams.pPriv = NULL;
    check_cupti_api_call( cuptiProfilerInitialize(&profilerInitializeParams) );

    PRINT(suppressOutput, "\nSearching for the first occurrences of GA100 and GH200 on the system...\n");
    std::map<std::string, int> deviceMap;
    std::vector<std::string> devicesSupportingCudaPresets = {"GA100", "GH100"};
    int devIdx;
    for (devIdx = 0; devIdx < numDevicesOnMachine; devIdx++) {
        CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
        getChipNameParams.pPriv = NULL;
        getChipNameParams.deviceIndex = devIdx;
        check_cupti_api_call( cuptiDeviceGetChipName(&getChipNameParams) );
        const char *chipName = getChipNameParams.pChipName;

        int i;
        for (i = 0; i < devicesSupportingCudaPresets.size(); i++) {
            if (devicesSupportingCudaPresets[i].compare(chipName) == 0 && deviceMap.count(chipName) == 0) {
                PRINT(suppressOutput, "%s found at index %d. Storing...\n", chipName, devIdx);
                deviceMap[chipName] = devIdx;
            }
        }
    }

    if (deviceMap.empty() == true) {
        fprintf(stderr, "Neither a GA100 or GH200 were detected on the system. Skipping test.\n");
        test_skip(__FILE__, __LINE__, "", 0); 
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__,__LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(suppressOutput, "\nPAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION)); 


    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) ); 
    for (auto pair = deviceMap.begin(); pair != deviceMap.end(); pair++) {
        CUcontext sessionCtx = NULL;
        int flags = 0;
        CUdevice device = pair->second;
        #if defined(CUDA_TOOLKIT_GE_13)
        check_cuda_driver_api_call( cuCtxCreate(&sessionCtx, (CUctxCreateParams*)0, flags, device) );
        #else
        check_cuda_driver_api_call( cuCtxCreate(&sessionCtx, flags, device) );
        #endif

        std::vector<std::string> presets;
        // GA100 presets
        if (pair->first.compare("GA100") == 0) {
            presets = {"PAPI_CUDA_FP16_FMA", "PAPI_CUDA_BF16_FMA", "PAPI_CUDA_FP32_FMA", "PAPI_CUDA_FP64_FMA", "PAPI_CUDA_FP_FMA"}; 
        }
        // GH200 presets
        else {
            presets = {"PAPI_CUDA_FP16_FMA", "PAPI_CUDA_BF16_FMA", "PAPI_CUDA_FP32_FMA", "PAPI_CUDA_FP64_FMA", "PAPI_CUDA_FP_FMA", "PAPI_CUDA_FP8_OPS"};
        }

        PRINT(suppressOutput, "Attempting to add, start, and stop cuda component presets"
              " for the NVIDIA %s located at index %d\n", pair->first.c_str(), pair->second);
        PRINT(suppressOutput, "---------------------------------------------------------\n");
        for (std::string preset : presets) {
            check_papi_api_call( PAPI_add_named_event(EventSet, preset.c_str()));

            check_papi_api_call( PAPI_start(EventSet) );

            // Launch kernel
            int matrixSize = 100000;
            VectorAddSubtract(matrixSize, suppressOutput);

            long long counterValue = 0;
            check_papi_api_call( PAPI_stop(EventSet, &counterValue) );
            PRINT(suppressOutput, "The preset %s produced the counter value %lld\n", preset.c_str(), counterValue);

            check_papi_api_call( PAPI_cleanup_eventset(EventSet) );
        }
        PRINT(suppressOutput, "---------------------------------------------------------\n\n");

        check_cuda_driver_api_call( cuCtxDestroy(sessionCtx) );
    }

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0; 
}
