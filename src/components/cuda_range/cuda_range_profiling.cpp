// C++ standard library headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <set>

// POSIX standard headers.
#include <dlfcn.h>
#include <dirent.h>

// CTK headers.
#include <cuda_runtime_api.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <cupti_range_profiler.h>

// Internal headers.
#include "cuda_range_profiling.h"
#include "cuda_range_profiling.hpp"
#include "papi.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "papi_debug.h"
#include "papi_internal.h"
#include "papi_memory.h"
#ifdef __cplusplus
}
#endif

// 0 bits are left to be used with uint32_t. Therefore, if another qualifier is added bits will need
// to be pulled from NAMEID_WIDTH or implement uint64_t.
#define EVENTS_WIDTH    (sizeof(uint32_t) * 8)
#define STAT_WIDTH      ( 3 ) // 2^3 = 8
#define DEVICE_WIDTH    ( 6 ) // 2^6 = 64
#define SUBMETRIC_WIDTH ( 5 ) // 2^5 = 32
#define QLMASK_WIDTH    ( 3 ) // 2^2 = 4
#define NAMEID_WIDTH    ( 15 ) // 2^15 = 32768
#define UNUSED_WIDTH    (EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH - STAT_WIDTH - SUBMETRIC_WIDTH)
#define STAT_SHIFT      (EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH)
#define SUBMETRIC_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH - SUBMETRIC_WIDTH)
#define DEVICE_SHIFT    (EVENTS_WIDTH - UNUSED_WIDTH - STAT_WIDTH - SUBMETRIC_WIDTH - DEVICE_WIDTH)
#define QLMASK_SHIFT    (DEVICE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT    (QLMASK_SHIFT - NAMEID_WIDTH)
#define STAT_MASK       ((0xFFFFFFFF >> (EVENTS_WIDTH - STAT_WIDTH)) << STAT_SHIFT)
#define DEVICE_MASK     ((0xFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define SUBMETRIC_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - SUBMETRIC_WIDTH)) << SUBMETRIC_SHIFT)
#define QLMASK_MASK     ((0xFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK     ((0xFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define SUBMETRIC_FLAG  (0x4)
#define STAT_FLAG       (0x2)
#define DEVICE_FLAG     (0x1)
#define NOQUAL_FLAG     (0x0)

// CUPTI Profiler Host API.
typedef CUptiResult ( *cuptiProfilerHostInitialize_t ) (CUpti_Profiler_Host_Initialize_Params *pParams);
cuptiProfilerHostInitialize_t cuptiProfilerHostInitializePtr;
typedef CUptiResult ( *cuptiProfilerHostGetMaxNumHardwareMetricsPerPass_t ) (CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params *pParams);
cuptiProfilerHostGetMaxNumHardwareMetricsPerPass_t cuptiProfilerHostGetMaxNumHardwareMetricsPerPassPtr;
typedef CUptiResult ( *cuptiProfilerHostGetBaseMetrics_t ) (CUpti_Profiler_Host_GetBaseMetrics_Params *pParams);
cuptiProfilerHostGetBaseMetrics_t cuptiProfilerHostGetBaseMetricsPtr;
typedef CUptiResult ( *cuptiProfilerHostGetSubMetrics_t ) (CUpti_Profiler_Host_GetSubMetrics_Params *pParams);
cuptiProfilerHostGetSubMetrics_t cuptiProfilerHostGetSubMetricsPtr;
typedef CUptiResult ( *cuptiProfilerHostGetMetricProperties_t ) (CUpti_Profiler_Host_GetMetricProperties_Params *pParams);
cuptiProfilerHostGetMetricProperties_t cuptiProfilerHostGetMetricPropertiesPtr;
typedef CUptiResult ( *cuptiProfilerHostGetSupportedChips_t ) (CUpti_Profiler_Host_GetSupportedChips_Params *pParams);
cuptiProfilerHostGetSupportedChips_t cuptiProfilerHostGetSupportedChipsPtr;
typedef CUptiResult ( *cuptiProfilerHostConfigAddMetrics_t ) (CUpti_Profiler_Host_ConfigAddMetrics_Params *pParams);
cuptiProfilerHostConfigAddMetrics_t cuptiProfilerHostConfigAddMetricsPtr; 
typedef CUptiResult ( *cuptiProfilerHostGetConfigImageSize_t ) (CUpti_Profiler_Host_GetConfigImageSize_Params *pParams);
cuptiProfilerHostGetConfigImageSize_t cuptiProfilerHostGetConfigImageSizePtr;
typedef CUptiResult ( *cuptiProfilerHostGetConfigImage_t ) (CUpti_Profiler_Host_GetConfigImage_Params *pParams);
cuptiProfilerHostGetConfigImage_t cuptiProfilerHostGetConfigImagePtr;
typedef CUptiResult ( *cuptiProfilerHostGetNumOfPasses_t ) (CUpti_Profiler_Host_GetNumOfPasses_Params *pParams);
cuptiProfilerHostGetNumOfPasses_t cuptiProfilerHostGetNumOfPassesPtr;
typedef CUptiResult ( *cuptiProfilerHostEvaluateToGpuValues_t ) (CUpti_Profiler_Host_EvaluateToGpuValues_Params *pParams);
cuptiProfilerHostEvaluateToGpuValues_t cuptiProfilerHostEvaluateToGpuValuesPtr;
typedef CUptiResult ( *cuptiProfilerHostDeinitialize_t ) (CUpti_Profiler_Host_Deinitialize_Params *pParams);
cuptiProfilerHostDeinitialize_t cuptiProfilerHostDeinitializePtr;

// CUPTI Range Profiling API.
typedef CUptiResult ( *cuptiRangeProfilerEnable_t ) (CUpti_RangeProfiler_Enable_Params *pParams);
cuptiRangeProfilerEnable_t cuptiRangeProfilerEnablePtr;
typedef CUptiResult ( *cuptiRangeProfilerGetCounterDataSize_t ) (CUpti_RangeProfiler_GetCounterDataSize_Params *pParams);
cuptiRangeProfilerGetCounterDataSize_t cuptiRangeProfilerGetCounterDataSizePtr;
typedef CUptiResult ( *cuptiRangeProfilerCounterDataImageInitialize_t ) (CUpti_RangeProfiler_CounterDataImage_Initialize_Params *pParams);
cuptiRangeProfilerCounterDataImageInitialize_t cuptiRangeProfilerCounterDataImageInitializePtr;
typedef CUptiResult ( *cuptiRangeProfilerSetConfig_t ) (CUpti_RangeProfiler_SetConfig_Params *pParams);
cuptiRangeProfilerSetConfig_t cuptiRangeProfilerSetConfigPtr;
typedef CUptiResult ( *cuptiRangeProfilerStart_t ) (CUpti_RangeProfiler_Start_Params *pParams);
cuptiRangeProfilerStart_t cuptiRangeProfilerStartPtr;
typedef CUptiResult ( *cuptiRangeProfilerPushRange_t ) (CUpti_RangeProfiler_PushRange_Params *pParams);
cuptiRangeProfilerPushRange_t cuptiRangeProfilerPushRangePtr;
typedef CUptiResult ( *cuptiRangeProfilerPopRange_t ) (CUpti_RangeProfiler_PopRange_Params *pParams);
cuptiRangeProfilerPopRange_t cuptiRangeProfilerPopRangePtr;
typedef CUptiResult ( *cuptiRangeProfilerStop_t ) (CUpti_RangeProfiler_Stop_Params *pParams);
cuptiRangeProfilerStop_t cuptiRangeProfilerStopPtr;
typedef CUptiResult ( *cuptiRangeProfilerDecodeData_t ) (CUpti_RangeProfiler_DecodeData_Params *pParams);
cuptiRangeProfilerDecodeData_t cuptiRangeProfilerDecodeDataPtr;
typedef CUptiResult ( *cuptiRangeProfilerDisable_t ) (CUpti_RangeProfiler_Disable_Params *pParams);
cuptiRangeProfilerDisable_t cuptiRangeProfilerDisablePtr;
typedef CUptiResult ( *cuptiRangeProfilerCounterDataGetRangeInfo_t) (CUpti_RangeProfiler_CounterData_GetRangeInfo_Params *pParams);
cuptiRangeProfilerCounterDataGetRangeInfo_t cuptiRangeProfilerCounterDataGetRangeInfoPtr;

// CUPTI API Misc.
typedef CUptiResult ( *cuptiProfilerGetCounterAvailability_t ) ( CUpti_Profiler_GetCounterAvailability_Params *pParams );
cuptiProfilerGetCounterAvailability_t cuptiProfilerGetCounterAvailabilityPtr;
typedef CUptiResult ( *cuptiDeviceGetChipName_t ) (CUpti_Device_GetChipName_Params *pParams);
cuptiDeviceGetChipName_t cuptiDeviceGetChipNamePtr;
typedef CUptiResult ( *cuptiProfilerInitialize_t ) (CUpti_Profiler_Initialize_Params *pParams);
cuptiProfilerInitialize_t cuptiProfilerInitializePtr;
typedef CUptiResult ( *cuptiProfilerDeInitialize_t ) (CUpti_Profiler_DeInitialize_Params *pParams);
cuptiProfilerDeInitialize_t cuptiProfilerDeInitializePtr;

// CUDA Runtime API.
typedef cudaError_t ( *cudaGetDeviceCount_t ) (int *count);
cudaGetDeviceCount_t cudaGetDeviceCountPtr;
typedef cudaError_t ( *cudaRuntimeGetVersion_t ) (int *runtimeVersion);
cudaRuntimeGetVersion_t cudaRuntimeGetVersionPtr;

// CUDA Driver API.
 typedef CUresult (*cuInit_t) (unsigned int flags);
cuInit_t cuInitPtr;
typedef CUresult ( *cuCtxCreate_t ) (CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int  flags, CUdevice dev);
cuCtxCreate_t cuCtxCreatePtr;
typedef CUresult ( *cuCtxDestroy_t ) (CUcontext ctx);
cuCtxDestroy_t cuCtxDestroyPtr;
typedef CUresult ( *cuCtxGetCurrent_t ) (CUcontext *pctx);
cuCtxGetCurrent_t cuCtxGetCurrentPtr;
typedef CUresult ( *cuCtxPopCurrent_t ) (CUcontext *pctx);
cuCtxPopCurrent_t cuCtxPopCurrentPtr;
typedef CUresult ( *cuCtxPushCurrent_t ) (CUcontext ctx);
cuCtxPushCurrent_t cuCtxPushCurrentPtr;
typedef CUresult ( *cuCtxGetDevice_v2_t ) (CUdevice *device, CUcontext ctx);
cuCtxGetDevice_v2_t cuCtxGetDevice_v2Ptr;

// CUDA driver, runtime, and CUPTI handles for dlsym.
void *dl_driver_handle = NULL;
void *dl_runtime_handle = NULL;
void *dl_cupti_handle = NULL;

// Reserves an NVIDIA device for profiling.
int64_t device_bitmask;

// CUPTI metric enumeration.
std::map<std::string, cupti_metric_info_t> cupti_metrics;
std::vector<std::string> cupti_metrics_keys;

// CUPTI profiling and metric metadata.
class CuptiProfile;
thread_local std::map<int, CuptiProfile> cupti_profile_per_device;
thread_local std::map<int, CuptiProfile> cupti_profile_metric_descriptions;

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name Workflow to load function pointers
 *  @{
 */

/**
 * @brief Search and load CUDA and CUPTI shared objects
 *        from the system paths.
 *
 * @param *so_names_to_search
 *   Varying names of the shared object we want to search for.
 */
void *search_and_load_from_system_paths(std::vector<std::string> &so_names_to_search)
{
    for (std::string so_name : so_names_to_search) {
        void *so = dlopen(so_name.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (so) {
            return so;
        }
    }

    return NULL;
}

/**
 * @brief Overloaded function to search and load shared objects by
 *        the full path.
 *
 * @param *full_path_to_shared_object
 *   Explicit path to the shared object.
 */
void *search_and_load_shared_libraries(const char *full_path_to_shared_object)
{
    void *so = dlopen(full_path_to_shared_object, RTLD_NOW | RTLD_GLOBAL);

    return so;
}

/**@class
 * @brief Search and load CUDA and CUPTI shared objects.
 *
 * @param *so_main_name
 *   The name of the shared object e.g. libcudart. This is used
 *   to select the standard_sub_paths to use.
 * @param *parent_path
 *   The main path we will use to search for the shared objects.
 * @param &so_names_to_search
 *   Varying names of the shared object we want to search for. 
 */
void *search_and_load_shared_libraries(std::string &so_main_name, const char *parent_path, std::vector<std::string> &so_names_to_search_for)
{
    void *so = NULL;
    std::vector<std::string> standard_sub_paths;
    // Standard subpaths for the libcudart shared object.
    if (so_main_name.compare("libcudart")  == 0) {
        standard_sub_paths.push_back("/lib64/");
    }
    // Standard subpaths for the libcupti shared object.
    else if (so_main_name.compare("libcupti") == 0) {
        standard_sub_paths.push_back("/extras/CUPTI/lib64/");
        standard_sub_paths.push_back("/lib64/");
    }
    // Provided shared object is not accounted for, should never occur in production code.
    else {
        SUBDBG("The provided .so name (%s) is not accounted for. Enter a proper .so name or update the conditional workflow.\n", so_main_name.c_str());
        return NULL;
    }

    std::string path_to_shared_object;
    for (std::string sub_path : standard_sub_paths) {
        // Construct path to search for dl names.
        std::string directory_path_to_search = parent_path + sub_path;

        DIR *dir = opendir(directory_path_to_search.c_str());
        if (dir == NULL) {
            SUBDBG("Directory path %s could not be opened. Continuing to next subpath if one exists for the shared object.\n",
                   directory_path_to_search.c_str());
            continue;
        }

        int status = 0;
        for (std::string so_name : so_names_to_search_for) {
            struct dirent *dirEntry = NULL;
            while( ( dirEntry = readdir(dir) ) != NULL ) {
                int result = -1;
                std::string nameOfDirEntry = dirEntry->d_name;
                // The so name contains .so or .so.1; therefore, we look for an exact string match (e.g. libcupti.so == libcupti.so).
                if (so_name.find(".so") != std::string::npos) {
                    result = so_name.compare(nameOfDirEntry);
                }
                // The so name does not contain .so or .so.1; therefore, we look for a substring match (e.g. libcupti in libcupti.so.2025.3.1).
                else {
                    result = so_name.compare(0, so_name.length(), nameOfDirEntry);
                }

                if (result == 0) {
                    path_to_shared_object = directory_path_to_search + nameOfDirEntry;

                    status = closedir(dir);
                    if (status != 0) {
                        SUBDBG("Failed to close directory from path %s. Continuing.\n");
                    }

                    goto open;
                }
            }
            // Reset the position of the directory stream.
            rewinddir(dir);
        }
        status = closedir(dir);
        if (status != 0) {
            SUBDBG("Failed to close directory from path %s. Continuing.\n");
        }
    }

    return NULL;
  exit:
    return so;
  open:
    so = dlopen(path_to_shared_object.c_str(), RTLD_NOW | RTLD_GLOBAL);
    goto exit;
}

/**
  * @brief Load the Driver API functions used internally in the cuda_range
  *        component.
  *
  *        The libcuda shared library is searched for using the shared library names
  *        libcuda.so, libcuda.so.1, and a catch all libcuda. Once the libcuda shared
  *        library is found the used Driver API functions are dlsym'd.
*/
int load_cuda_driver_function_pointers(void)
{
    std::vector<std::string> so_names_to_search = {"libcuda.so", "libcuda.so.1", "libcuda"};

    char *papi_cuda_range_driver = std::getenv("PAPI_CUDA_RANGE_DRIVER");
    if (papi_cuda_range_driver != NULL) {
        dl_driver_handle = search_and_load_shared_libraries(papi_cuda_range_driver);
        if (dl_driver_handle != NULL) {
            goto load_functions;
        }
        else {
            SUBDBG("PAPI_CUDA_RANGE_DRIVER was set, but did not result in successfully loading the libcuda shared object."
                   " Set PAPI_CUDA_RANGE_DRIVER to a valid libcuda shared object.\n");
            return PAPI_ESYS;
        }
    }
    else {
        SUBDBG("PAPI_CUDA_RANGE_DRIVER was not set. Falling back to dlopen to search for the libcuda shared object.\n");
    }

    dl_driver_handle = search_and_load_from_system_paths(so_names_to_search);
    if (dl_driver_handle == NULL) {
        SUBDBG("Failed to load a libcudart shared object. Try setting PAPI_CUDA_RANGE_RUNTIME.\n");
    }

    dl_driver_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);

  load_functions:
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuInitPtr, cuInit_t, dl_driver_handle, "cuInit");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxCreatePtr, cuCtxCreate_t, dl_driver_handle, "cuCtxCreate_v4");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxDestroyPtr, cuCtxDestroy_t, dl_driver_handle, "cuCtxDestroy");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxGetCurrentPtr, cuCtxGetCurrent_t, dl_driver_handle, "cuCtxGetCurrent");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxPopCurrentPtr, cuCtxPopCurrent_t, dl_driver_handle, "cuCtxPopCurrent");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxPushCurrentPtr, cuCtxPushCurrent_t, dl_driver_handle, "cuCtxPushCurrent");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuCtxGetDevice_v2Ptr, cuCtxGetDevice_v2_t, dl_driver_handle, "cuCtxGetDevice_v2");

    return PAPI_OK;
}

/**
  * @brief Inform the system the dl_driver_handle object is no longer needed by
  *        the application.
*/
void unload_cuda_driver_function_pointers(void)
{
    int retval = dlclose(dl_driver_handle);
    if (retval != 0) {
        // dlclose can fail, but this function is called at PAPI_shutdown which has a return
        // type of void. Therefore, an error code here would never make it back to the user.
        SUBDBG("Failed to close the handle associated with the driver function pointers due to -- %s.\n", dlerror());
    }

    cuCtxCreatePtr       = nullptr;
    cuCtxDestroyPtr      = nullptr;
    cuCtxGetCurrentPtr   = nullptr;
    cuCtxPopCurrentPtr   = nullptr;
    cuCtxPushCurrentPtr  = nullptr;
    cuCtxGetDevice_v2Ptr = nullptr;
    cuInitPtr            = nullptr;

    return;
}

/**
  * @brief Load the Runtime API functions used internally in the cuda_range
  *        component.
  *
  *        The libcudart shared library is searched for using the shared library names
  *        libcudart.so, libcudart.so.1, and a catch all libcudart. Once the libcudart shared
  *        library is found the used Runtime API functions are dlsym'd.
*/
int load_cuda_runtime_function_pointers(void)
{
    // The libcudart shared object names that we will search for.
    // Note that, the libcudart without .so should always be the last entry
    // in the vector.
    std::vector<std::string> so_names_to_search = {"libcudart.so", "libcudart.so.1", "libcudart"};

    char *papi_cuda_range_runtime = std::getenv("PAPI_CUDA_RANGE_RUNTIME"), *papi_cuda_range_root = std::getenv("PAPI_CUDA_RANGE_ROOT");
    // If a user set PAPI_CUDA_RANGE_RUNTIME with a path to a shared object, attempt to load it (takes precedent over PAPI_CUDA_RANGE_ROOT)
    if (papi_cuda_range_runtime != NULL) {
        dl_runtime_handle = search_and_load_shared_libraries(papi_cuda_range_runtime);
        if (dl_runtime_handle != NULL) {
            goto load_functions;
        }    
        else {
            SUBDBG("PAPI_CUDA_RANGE_RUNTIME was set, but did not result in successfully loading the libcudart shared object."
                   " Set PAPI_CUDA_RANGE_RUNTIME to a valid libcudart shared object.\n");
            return PAPI_ESYS;
        }    
    }    
    else {
        SUBDBG("PAPI_CUDA_RANGE_RUNTIME was not set. Falling back to PAPI_CUDA_RANGE_ROOT to search for the libcudart shared object.\n");
    }    

    if (papi_cuda_range_root != NULL) {
        std::string so_name = "libcudart";
        dl_runtime_handle = search_and_load_shared_libraries(so_name, papi_cuda_range_root, so_names_to_search);
        if (dl_runtime_handle != NULL) {
            goto load_functions;
        }    
        else {
            SUBDBG("PAPI_CUDA_RANGE_ROOT was set, but did not result in successfully loading the libcudart shared object."
                   " Falling back to dlopen to search for the libcudart shared object.\n");
        }    
    }    
    else {
        SUBDBG("PAPI_CUDA_RANGE_ROOT was not set. Falling back to dlopen to search for the libcudart shared object.\n");
    }    

    // If PAPI_CUDA_RANGE_RUNTIME was not set and PAPI_CUDA_RANGE_ROOT was either not set or did not result in the libcudart shared object
    // being successfully loaded then use dlopen and follow the search logic used by the dynamic linker.
    dl_runtime_handle = search_and_load_from_system_paths(so_names_to_search);
    if (dl_runtime_handle == NULL) {
        SUBDBG("Failed to load a libcudart shared object. Try setting PAPI_CUDA_RANGE_RUNTIME or PAPI_CUDA_RANGE_ROOT.\n");
        return PAPI_ESYS;
    }    

  load_functions:
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cudaGetDeviceCountPtr, cudaGetDeviceCount_t, dl_runtime_handle, "cudaGetDeviceCount");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cudaRuntimeGetVersionPtr, cudaRuntimeGetVersion_t, dl_runtime_handle, "cudaRuntimeGetVersion");

    return PAPI_OK;
}

/**
  * @brief Inform the system the dl_runtime_handle object is no longer needed by
  *        the application.
*/
void unload_cuda_runtime_function_pointers(void)
{
    int retval = dlclose(dl_runtime_handle);
    if (retval != 0) { 
        // dlclose can fail, but this function is called at PAPI_shutdown which has a return
        // type of void. Therefore, an error code here would never make it back to the user.
        SUBDBG("Failed to close the handle associated with the runtime function pointers due to -- %s.\n", dlerror());
    }    

    cudaGetDeviceCountPtr    = nullptr;
    cudaRuntimeGetVersionPtr = nullptr;

    return;
}

/**
  * @brief Load the CUPTI API functions used internally in the cuda_range
  *        component.
  *
  *        The libcupti shared library is searched for using the shared library names
  *        libcupti.so, libcupti.so.1, and a catch all libcupti. Once the libcupti shared
  *        library is found the used Runtime API functions are dlsym'd.
*/
int load_cupti_function_pointers(void)
{
    // The libcupti shared object names that we will search for.
    // Note that, the libcupti without .so should always be the last entry
    // in the vector.
    std::vector<std::string> so_names_to_search = {"libcupti.so", "libcupti.so.1", "libcupti"};

    char *papi_cuda_range_cupti = std::getenv("PAPI_CUDA_RANGE_CUPTI"), *papi_cuda_range_root = std::getenv("PAPI_CUDA_RANGE_ROOT");
    // If a user set PAPI_CUDA_RANGE_CUPTI with a path to a shared object, attempt to load it (takes precedent over PAPI_CUDA_RANGE_ROOT)
    if (papi_cuda_range_cupti != NULL) {
        dl_cupti_handle = search_and_load_shared_libraries(papi_cuda_range_cupti); 
        if (dl_cupti_handle != NULL) {
            goto load_functions;
        }    
        else {
            SUBDBG("PAPI_CUDA_RANGE_CUPTI was set, but did not result in successfully loading the libcupti shared object."
                   " Set PAPI_CUDA_RANGE_CUPTI to a valid libcupti shared object.\n");
            return PAPI_ESYS;
        }    
    }    
    else {
        SUBDBG("PAPI_CUDA_RANGE_CUPTI was not set. Falling back to PAPI_CUDA_RANGE_ROOT to search for the libcupti shared object.\n");
    }    

    if (papi_cuda_range_root != NULL) {
        std::string so_name = "libcupti";
        dl_cupti_handle = search_and_load_shared_libraries(so_name, papi_cuda_range_root, so_names_to_search);
        if (dl_cupti_handle != NULL) {
            goto load_functions;
        }    
        else {
            SUBDBG("PAPI_CUDA_RANGE_ROOT was set, but did not result in successfully loading the libcupti shared object."
                   " Falling back to dlopen to search for the libcupti shared object.\n");
        }    
    }    
    else {
        SUBDBG("PAPI_CUDA_RANGE_ROOT was not set. Falling back to dlopen to search for the libcudart shared object.\n");
    }    

    // If PAPI_CUDA_RANGE_CUPTI was not set and PAPI_CUDA_RANGE_ROOT was either not set or did not result in the libcupti shared object
    // being successfully loaded then use dlopen and follow the search logic used by the dynamic linker.
    dl_cupti_handle = search_and_load_from_system_paths(so_names_to_search);
    if (dl_cupti_handle) {
        SUBDBG("Failed to load a libcupti shared object. Try setting PAPI_CUDA_RANGE_CUPTI or PAPI_CUDA_RANGE_ROOT.\n");
        return PAPI_ESYS;
    }    

  load_functions:
    // CUPTI Profiler Host API.
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostInitializePtr, cuptiProfilerHostInitialize_t, dl_cupti_handle, "cuptiProfilerHostInitialize");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetMaxNumHardwareMetricsPerPassPtr, cuptiProfilerHostGetMaxNumHardwareMetricsPerPass_t,
                                    dl_cupti_handle, "cuptiProfilerHostGetMaxNumHardwareMetricsPerPass");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetBaseMetricsPtr, cuptiProfilerHostGetBaseMetrics_t, dl_cupti_handle, "cuptiProfilerHostGetBaseMetrics");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetSubMetricsPtr, cuptiProfilerHostGetSubMetrics_t, dl_cupti_handle, "cuptiProfilerHostGetSubMetrics");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetMetricPropertiesPtr, cuptiProfilerHostGetMetricProperties_t, dl_cupti_handle, "cuptiProfilerHostGetMetricProperties");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetSupportedChipsPtr, cuptiProfilerHostGetSupportedChips_t, dl_cupti_handle, "cuptiProfilerHostGetSupportedChips");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostConfigAddMetricsPtr, cuptiProfilerHostConfigAddMetrics_t, dl_cupti_handle, "cuptiProfilerHostConfigAddMetrics");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetConfigImageSizePtr, cuptiProfilerHostGetConfigImageSize_t, dl_cupti_handle, "cuptiProfilerHostGetConfigImageSize");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetConfigImagePtr, cuptiProfilerHostGetConfigImage_t, dl_cupti_handle, "cuptiProfilerHostGetConfigImage");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostGetNumOfPassesPtr, cuptiProfilerHostGetNumOfPasses_t, dl_cupti_handle, "cuptiProfilerHostGetNumOfPasses");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostEvaluateToGpuValuesPtr, cuptiProfilerHostEvaluateToGpuValues_t, dl_cupti_handle, "cuptiProfilerHostEvaluateToGpuValues");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerHostDeinitializePtr, cuptiProfilerHostDeinitialize_t, dl_cupti_handle, "cuptiProfilerHostDeinitialize");
    
    // CUPTI Range Profiling API.
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerEnablePtr, cuptiRangeProfilerEnable_t, dl_cupti_handle, "cuptiRangeProfilerEnable");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerGetCounterDataSizePtr, cuptiRangeProfilerGetCounterDataSize_t, dl_cupti_handle, "cuptiRangeProfilerGetCounterDataSize");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerCounterDataImageInitializePtr, cuptiRangeProfilerCounterDataImageInitialize_t, dl_cupti_handle, "cuptiRangeProfilerCounterDataImageInitialize");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerSetConfigPtr, cuptiRangeProfilerSetConfig_t, dl_cupti_handle, "cuptiRangeProfilerSetConfig");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerStartPtr, cuptiRangeProfilerStart_t, dl_cupti_handle, "cuptiRangeProfilerStart");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerPushRangePtr, cuptiRangeProfilerPushRange_t, dl_cupti_handle, "cuptiRangeProfilerPushRange");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerPopRangePtr, cuptiRangeProfilerPopRange_t, dl_cupti_handle, "cuptiRangeProfilerPopRange");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerStopPtr, cuptiRangeProfilerStop_t, dl_cupti_handle, "cuptiRangeProfilerStop");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerDecodeDataPtr, cuptiRangeProfilerDecodeData_t, dl_cupti_handle, "cuptiRangeProfilerDecodeData");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerDisablePtr, cuptiRangeProfilerDisable_t, dl_cupti_handle, "cuptiRangeProfilerDisable");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiRangeProfilerCounterDataGetRangeInfoPtr, cuptiRangeProfilerCounterDataGetRangeInfo_t, dl_cupti_handle, "cuptiRangeProfilerCounterDataGetRangeInfo");

    // General CUPTI API.
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerGetCounterAvailabilityPtr, cuptiProfilerGetCounterAvailability_t, dl_cupti_handle, "cuptiProfilerGetCounterAvailability");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiDeviceGetChipNamePtr, cuptiDeviceGetChipName_t, dl_cupti_handle, "cuptiDeviceGetChipName");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerInitializePtr, cuptiProfilerInitialize_t, dl_cupti_handle, "cuptiProfilerInitialize");
    ASSIGN_AND_CHECK_DLSYM_API_CALL(cuptiProfilerDeInitializePtr, cuptiProfilerDeInitialize_t, dl_cupti_handle, "cuptiProfilerDeInitialize");

    return PAPI_OK;
}

/**
  * @brief Inform the system the dl_cupti_handle object is no longer needed by
  *        the application.
*/
void unload_cupti_function_pointers(void)
{
    int retval = dlclose(dl_cupti_handle);
    if (retval != 0) { 
        // dlclose can fail, but this function is called at PAPI_shutdown which has a return
        // type of void. Therefore, an error code here would never make it back to the user.
        SUBDBG("Failed to close the handle associated with the cupti function pointers due to -- %s.\n", dlerror());
    }    

    // CUPTI Profiler Host API.
    cuptiProfilerHostInitializePtr                      = nullptr;
    cuptiProfilerHostGetMaxNumHardwareMetricsPerPassPtr = nullptr;
    cuptiProfilerHostGetBaseMetricsPtr                  = nullptr;
    cuptiProfilerHostGetSubMetricsPtr                   = nullptr;
    cuptiProfilerHostGetMetricPropertiesPtr             = nullptr;
    cuptiProfilerHostGetSupportedChipsPtr               = nullptr;
    cuptiProfilerHostConfigAddMetricsPtr                = nullptr;
    cuptiProfilerHostGetConfigImageSizePtr              = nullptr;
    cuptiProfilerHostGetConfigImagePtr                  = nullptr;
    cuptiProfilerHostGetNumOfPassesPtr                  = nullptr;
    cuptiProfilerHostEvaluateToGpuValuesPtr             = nullptr;
    cuptiProfilerHostDeinitializePtr                    = nullptr;

    // CUPTI Range Profiling API.
    cuptiRangeProfilerEnablePtr                         = nullptr;
    cuptiRangeProfilerGetCounterDataSizePtr             = nullptr;
    cuptiRangeProfilerCounterDataImageInitializePtr     = nullptr;
    cuptiRangeProfilerSetConfigPtr                      = nullptr;
    cuptiRangeProfilerStartPtr                          = nullptr;
    cuptiRangeProfilerPushRangePtr                      = nullptr;
    cuptiRangeProfilerPopRangePtr                       = nullptr;
    cuptiRangeProfilerStopPtr                           = nullptr;
    cuptiRangeProfilerDecodeDataPtr                     = nullptr;
    cuptiRangeProfilerDisablePtr                        = nullptr;
    cuptiRangeProfilerCounterDataGetRangeInfoPtr        = nullptr;

    // CUPTI API Misc.
    cuptiProfilerGetCounterAvailabilityPtr              = nullptr;
    cuptiDeviceGetChipNamePtr                           = nullptr;
    cuptiProfilerInitializePtr                          = nullptr;
    cuptiProfilerDeInitializePtr                        = nullptr;

    return;
}

/**
 *  @}
 ******************************************************************************/

std::string cuda_range_error_message;
/**
 * @brief Set the error message that will appear in the PAPI_component_info_t member
 *        variable disabled_reason.
 *
 * @param &message_to_set
 *   The error message to set.
 */
void cuda_range_set_last_err_msg(std::string &message_to_set)
{
    cuda_range_error_message = message_to_set;
}

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   C++ class for cuda_range component
 *  @{
 */

class CuptiProfile
{
private:
    CUcontext m_context = nullptr;
    bool m_context_ownership = false;
    bool m_all_passes_submitted = false;
    CUpti_Profiler_Host_Object *m_host_object = nullptr;
    CUpti_RangeProfiler_Object *m_range_profiler_object = nullptr;
    std::string m_chip_name;
    size_t m_samples_read = 0;
    
    std::vector<uint8_t> m_counter_availability_image;
    std::vector<uint8_t> m_config_image;
    std::vector<uint8_t> m_counter_data_image;

    std::vector<size_t> m_event_insertion_order;
    std::vector<const char *> m_cupti_metric_names;
    std::vector<double> m_metric_values;

public:
    // Constructors and Destructor.
    CuptiProfile(CUcontext context, bool context_ownership);
    CuptiProfile(CuptiProfile &&obj);
    ~CuptiProfile(void);

    // Setters.
    void set_cupti_metric_names(const std::vector<std::string> &cupti_metric_names);
    void set_event_insertion_order(const std::vector<size_t> &event_insertion_order);
    
    // Getters.
    bool get_all_passes_submitted(void); 

    // Member functions for metric enumeration and metric metadata.
    int base_metrics(size_t metric_type, const char ***base_metric_names, size_t &number_of_base_metrics);
    int sub_metrics(size_t metric_type, const char *base_metric_name, const char ***sub_metric_names, size_t &number_of_sub_metrics);
    int metric_properties(const char *cupti_metric_name, std::string &description);
    int number_of_passes(size_t &number_of_passes);

    // Member functions for metric profiling.
    int create_config_image(void);
    int enable_range_profiler(void);
    int create_counter_data_image(void);
    int config(void);
    int start_profiling(void);
    int push_range(std::string &range_name);
    int pop_range(void);
    int stop_profiling(void);
    int decode_data(void);
    int evaluate_counter_data(void);
    void calculate(cuda_range_ctx_t ctx);
    int disable_range_profiler(void);

    // Member functions for both metric enumeration/metadata and profiling.
    int counter_availability(void);
    int chip_name(int device_index);
    int host_initialize(void);
    int host_deinitialize(void);
    void destroy_context(void);
};

/**
 * @brief CuptiProfile constructor. Sets the private member variables
 *        m_context and m_context_ownership.
 *
 * @param context
 *   A CUDA context to be used for profiling.
 * @param context_ownership
 *   True if we internally created the CUDA context and false if a user
 *   created the CUDA context.
 */
inline CuptiProfile::CuptiProfile(CUcontext context, bool context_ownership)
{
    m_context = context;
    m_context_ownership = context_ownership;
}

/**
 * @brief CuptiProfile move constructor. Transfers ownerhship of
 *        m_context and m_context_ownership.
 * 
 * @param &&obj
 *   The CuptiProfile object to take ownership of.
 */
inline CuptiProfile::CuptiProfile(CuptiProfile &&obj)
{
    // Transfer ownership.
    m_context = obj.m_context;
    m_context_ownership = obj.m_context_ownership;

    // Leave obj in a valid state.
    obj.m_context = nullptr;
    obj.m_context_ownership = false;
}

/**
 * @brief CuptiProfile destructor. The allocated CUPTI metric names
 *        are freed and the internal created context's are destroy.
 */
inline CuptiProfile::~CuptiProfile(void)
{
    // Free memory allocated in set_cupt_metric_names.
    for (size_t i = 0; i < m_cupti_metric_names.size(); i++) {
        free((char*) m_cupti_metric_names[i]);
    }
    m_cupti_metric_names.clear();

    destroy_context();
}

/**
 * @brief A setter member function to set the private member variable
 *        m_cupti_metric_names.
 *
 *        This setter is called in cuda_range_store_added_native_events
 *        to set the user added cuda_range native events that are to
 *        be used for profiling.
 * 
 * @param &cupti_metric_names
 *   A vector of CUPTI metric names in the format of basename.rollup.submetric
 *   to be used for profiling.
 */
inline void CuptiProfile::set_cupti_metric_names(const std::vector<std::string> &cupti_metric_names)
{
    // Free the cached user added events.
    if (m_cupti_metric_names.size() > 0) {
        for (size_t i = 0; i < m_cupti_metric_names.size(); i++) {
            free((char*) m_cupti_metric_names[i]);
        }
        m_cupti_metric_names.clear();
    }

    // Cache the user added events.
    for (std::string cupti_metric_name : cupti_metric_names) {
        const char *c_cupti_metric_name = strdup(cupti_metric_name.c_str());
        m_cupti_metric_names.push_back(c_cupti_metric_name);
    }
}

/**
 * @brief A setter member function to set the private member variable
 *        m_event_insertion_order.
 *
 *        This setter is called in cuda_range_store_added_native_events
 *        to set the order of the user added cuda_range native events. This is done
 *        to properly map the evaluated counter data values into the returned
 *        long long values array for PAPI_read and PAPI_stop.
 * 
 * @param &event_insertion_order
 *   A vector containing the order of added cuda_range native events. 
 */
inline void CuptiProfile::set_event_insertion_order(const std::vector<size_t> &event_insertion_order)
{
    m_event_insertion_order = event_insertion_order;
}

/**
 * @brief A getter member function to return the private member
 *        variable m_all_passes_submitted.
 *
 *        This getter is called in cuda_range_decode_and_evaluate_counter_data
 *        to determine if a counter data image has successfully completed all passes. 
 */
inline bool CuptiProfile::get_all_passes_submitted(void)
{
    return m_all_passes_submitted;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostGetBaseMetrics.
 *
 * @param metric_type
 *   The metric type (counter, ratio, or throughput) 
 * @param ***base_metric_names
 *   Stores the list of supported base metric names for the chip.
 * @param &number_of_base_metrics
 *   Stores the number of supported base metrics for the chip.  
 */
inline int CuptiProfile::base_metrics(size_t metric_type, const char ***base_metric_names, size_t &number_of_base_metrics)
{
    CUpti_Profiler_Host_GetBaseMetrics_Params get_base_metrics_params {};
    /// [in]
    get_base_metrics_params.structSize = CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE;
    /// [in]
    get_base_metrics_params.pPriv = nullptr;
    /// [in]
    get_base_metrics_params.pHostObject = m_host_object;
    /// [in]
    get_base_metrics_params.metricType = (CUpti_MetricType) metric_type;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetBaseMetricsPtr(&get_base_metrics_params) );
    *base_metric_names = get_base_metrics_params.ppMetricNames;
    number_of_base_metrics = get_base_metrics_params.numMetrics;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostGetSubMetrics.
 *
 * @param metric_type
 *   The metric type (counter, ratio, or throughput) 
 * @param ***sub_metric_names
 *   Stores the list of supported sub-metric names for the chip.
 * @param &number_of_base_metrics
 *   Stores the number of supported sub-metrics for the chip.
 */
inline int CuptiProfile::sub_metrics(size_t metric_type, const char *base_metric_name, const char ***sub_metric_names, size_t &number_of_sub_metrics)
{
    CUpti_Profiler_Host_GetSubMetrics_Params get_sub_metrics_params {};
    /// [in]
    get_sub_metrics_params.structSize = CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE;
    /// [in]
    get_sub_metrics_params.pPriv = nullptr;
    /// [in]
    get_sub_metrics_params.pHostObject = m_host_object;
    /// [in]
    get_sub_metrics_params.metricType = (CUpti_MetricType) metric_type;
    /// [in]
    get_sub_metrics_params.pMetricName = base_metric_name;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetSubMetricsPtr(&get_sub_metrics_params) );
    *sub_metric_names = get_sub_metrics_params.ppSubMetrics;
    number_of_sub_metrics = get_sub_metrics_params.numOfSubmetrics;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostGetMetricProperties.
 *
 * @param cupti_metric_name
 *   The CUPTI metric name the properties will be listed.
 * @param &description
 *   Stores the short description for the CUPTI metric name..
 */
inline int CuptiProfile::metric_properties(const char *cupti_metric_name, std::string &description)
{
    CUpti_Profiler_Host_GetMetricProperties_Params get_metric_properties_params {};
    /// [in]
    get_metric_properties_params.structSize = CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE;
    /// [in]
    get_metric_properties_params.pPriv = nullptr;
    /// [in]
    get_metric_properties_params.pHostObject = m_host_object;
    /// [in]
    get_metric_properties_params.pMetricName = cupti_metric_name;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetMetricPropertiesPtr(&get_metric_properties_params) );
    description = get_metric_properties_params.pDescription;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostGetNumOfPasses.
 *
 * @param number_of_passes
 *   Store the number of passes required for profiling the scheduled
 *   metrics in the config image.
 */
inline int CuptiProfile::number_of_passes(size_t &number_of_passes)
{
    CUpti_Profiler_Host_GetNumOfPasses_Params get_num_of_passes_params {};
    /// [in]
    get_num_of_passes_params.structSize = CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE;
    /// [in]
    get_num_of_passes_params.pPriv = nullptr;
    /// [in]
    get_num_of_passes_params.configImageSize = m_config_image.size();
    /// [in]
    get_num_of_passes_params.pConfigImage = m_config_image.data();
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetNumOfPassesPtr(&get_num_of_passes_params) );
    number_of_passes = get_num_of_passes_params.numOfPasses;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI calls:
 *        1. cuptiProfilerHostConfigAddMetrics
 *        2. cuptiProfilerHostGetConfigImageSize
 *        3. cuptiProfilerHostGetConfigImage
 *
 *        This member function sets the private member variable
 *        m_config_image. 
 */
inline int CuptiProfile::create_config_image(void)
{
    CUpti_Profiler_Host_ConfigAddMetrics_Params config_add_metrics_params {};
    /// [in]
    config_add_metrics_params.structSize = CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE;
    /// [in]
    config_add_metrics_params.pPriv = nullptr;
    /// [in]
    config_add_metrics_params.pHostObject = m_host_object;
    /// [in]
    config_add_metrics_params.ppMetricNames = m_cupti_metric_names.data();
    /// [in]
    config_add_metrics_params.numMetrics = m_cupti_metric_names.size();
    CHECK_CUPTI_API_CALL(cuptiProfilerHostConfigAddMetricsPtr(&config_add_metrics_params));

    CUpti_Profiler_Host_GetConfigImageSize_Params get_config_image_size_params {};
    /// [in]
    get_config_image_size_params.structSize = CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE;
    /// [in]
    get_config_image_size_params.pPriv = nullptr;
    /// [in]
    get_config_image_size_params.pHostObject = m_host_object;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetConfigImageSizePtr(&get_config_image_size_params) );
    m_config_image.resize(get_config_image_size_params.configImageSize, 0);

    CUpti_Profiler_Host_GetConfigImage_Params get_config_image_params {};
    /// [in]
    get_config_image_params.structSize = CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE;
    /// [in]
    get_config_image_params.pPriv = nullptr;
    /// [in]
    get_config_image_params.pHostObject = m_host_object;
    /// [in]
    get_config_image_params.configImageSize = m_config_image.size();
    get_config_image_params.pConfigImage = m_config_image.data();
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetConfigImagePtr( &get_config_image_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerEnable.
 *
 *        This member function sets the private member variable
 *        m_range_profiler_object. 
 */
inline int CuptiProfile::enable_range_profiler(void)
{
    CUpti_RangeProfiler_Enable_Params enable_params {};
    /// [in]
    enable_params.structSize = CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE;
    /// [in]
    enable_params.pPriv = nullptr;
    /// [in]
    enable_params.ctx = m_context;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerEnablePtr(&enable_params) );
    m_range_profiler_object = enable_params.pRangeProfilerObject;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI calls:
 *        1. cuptiRangeProfilerGetCounterDataSize
 *        2. cuptiRangeProfilerCounterDataImageInitialize
 *
 *        This member function sets the private member variable
 *        m_counter_data_image.
 */
inline int CuptiProfile::create_counter_data_image(void)
{
    CUpti_RangeProfiler_GetCounterDataSize_Params get_counter_data_size_params {};
    /// [in]
    get_counter_data_size_params.structSize = CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE;
    /// [in]
    get_counter_data_size_params.pPriv = nullptr;
    /// [in]
    get_counter_data_size_params.pRangeProfilerObject = m_range_profiler_object;
    /// [in]
    get_counter_data_size_params.pMetricNames = m_cupti_metric_names.data();
    /// [in]
    get_counter_data_size_params.numMetrics = m_cupti_metric_names.size();
    /// [in]
    get_counter_data_size_params.maxNumOfRanges = 1;
    /// [in]
    get_counter_data_size_params.maxNumRangeTreeNodes = 1;
    CHECK_CUPTI_API_CALL(cuptiRangeProfilerGetCounterDataSizePtr(&get_counter_data_size_params));
    m_counter_data_image.resize(get_counter_data_size_params.counterDataSize, 0);

    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initialize_counter_data_image_params {};
    /// [in]
    initialize_counter_data_image_params.structSize = CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    /// [in]
    initialize_counter_data_image_params.pPriv = nullptr;
    /// [in]
    initialize_counter_data_image_params.pRangeProfilerObject = m_range_profiler_object;
    /// [in]
    initialize_counter_data_image_params.pCounterData = m_counter_data_image.data();
    /// [in]
    initialize_counter_data_image_params.counterDataSize = m_counter_data_image.size();
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerCounterDataImageInitializePtr(&initialize_counter_data_image_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerSetConfig.
 *
 *        This member function sets the private member variable
 *        m_range_profiler_object. 
 */
inline int CuptiProfile::config(void)
{
    CUpti_RangeProfiler_SetConfig_Params set_config_params {};
    /// [in]
    set_config_params.structSize = CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE;
    /// [in]
    set_config_params.pPriv = nullptr;
    /// [in]
    set_config_params.pRangeProfilerObject = m_range_profiler_object;
    /// [in]
    set_config_params.configSize = m_config_image.size();
    /// [in]
    set_config_params.pConfig = m_config_image.data();
    /// [in]
    set_config_params.counterDataImageSize = m_counter_data_image.size();
    /// [in]
    set_config_params.pCounterDataImage = m_counter_data_image.data();
    /// [in]
    set_config_params.range = CUPTI_UserRange;
    /// [in]
    set_config_params.replayMode = CUPTI_UserReplay;
    /// [in]
    set_config_params.maxRangesPerPass = 1;
    /// [in]
    set_config_params.numNestingLevels = 1;
    /// [in]
    set_config_params.minNestingLevel = 1;
    /// [in]
    set_config_params.passIndex = 0;
    /// [in]
    set_config_params.targetNestingLevel = 1;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerSetConfigPtr(&set_config_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerStart.
 */
inline int CuptiProfile::start_profiling()
{
    CUpti_RangeProfiler_Start_Params start_range_profiler {};
    /// [in]
    start_range_profiler.structSize = CUpti_RangeProfiler_Start_Params_STRUCT_SIZE;
    /// [in]
    start_range_profiler.pPriv = nullptr;
    /// [in]
    start_range_profiler.pRangeProfilerObject = m_range_profiler_object;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerStartPtr(&start_range_profiler) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerPushRange.
 * 
 * @param &range_name
 *   Name of the range to be profiled.
 */
inline int CuptiProfile::push_range(std::string &range_name)
{
    CUpti_RangeProfiler_PushRange_Params push_range_params {};
    /// [in]
    push_range_params.structSize = CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE;
    /// [in]
    push_range_params.pPriv = NULL;
    /// [in]
    push_range_params.pRangeProfilerObject = m_range_profiler_object;
    /// [in]
    push_range_params.pRangeName = range_name.c_str();
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerPushRangePtr(&push_range_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerPopRange.
 */
inline int CuptiProfile::pop_range(void)
{
    CUpti_RangeProfiler_PopRange_Params pop_range_params {};
    /// [in]
    pop_range_params.structSize = CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE;
    /// [in]
    pop_range_params.pPriv = nullptr;
    /// [in]
    pop_range_params.pRangeProfilerObject = m_range_profiler_object;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerPopRangePtr(&pop_range_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerStop.
 */
inline int CuptiProfile::stop_profiling(void)
{
    CUpti_RangeProfiler_Stop_Params stop_params {};
    /// [in]
    stop_params.structSize = CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE;
    /// [in]
    stop_params.pPriv = nullptr;
    /// [in]
    stop_params.pRangeProfilerObject = m_range_profiler_object;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerStopPtr(&stop_params) );
    m_all_passes_submitted = stop_params.isAllPassSubmitted;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerDecodeData.
 */
inline int CuptiProfile::decode_data(void)
{
    CUpti_RangeProfiler_DecodeData_Params decode_data_params {};
    /// [in]
    decode_data_params.structSize = CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE;
    /// [in]
    decode_data_params.pPriv = nullptr;
    /// [in]
    decode_data_params.pRangeProfilerObject = m_range_profiler_object;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerDecodeDataPtr(&decode_data_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostEvaluateToGpuValues.
 */
inline int CuptiProfile::evaluate_counter_data(void)
{
    m_metric_values.resize(m_cupti_metric_names.size(), 0.0);
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate_to_gpu_values_params {};
    /// [in]
    evaluate_to_gpu_values_params.structSize = CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE;
    /// [in]
    evaluate_to_gpu_values_params.pPriv = nullptr;
    /// [in]
    evaluate_to_gpu_values_params.pHostObject = m_host_object;
    /// [in]
    evaluate_to_gpu_values_params.pCounterDataImage = m_counter_data_image.data();
    /// [in]
    evaluate_to_gpu_values_params.counterDataImageSize = m_counter_data_image.size();
    /// [in]
    evaluate_to_gpu_values_params.rangeIndex = 0;
    /// [in]
    evaluate_to_gpu_values_params.ppMetricNames = m_cupti_metric_names.data();
    /// [in]
    evaluate_to_gpu_values_params.numMetrics = m_cupti_metric_names.size();
    /// [in/out]
    evaluate_to_gpu_values_params.pMetricValues = m_metric_values.data();
    CHECK_CUPTI_API_CALL( cuptiProfilerHostEvaluateToGpuValuesPtr(&evaluate_to_gpu_values_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that determines
 *        the rollup metric appended to the CUPTI metric name and
 *        calculates the counter value to be stored in the array
 *        for PAPI_read and PAPI_stop.
 *
 *        The first PAPI_read/PAPI_stop will store the initial obtained value
 *        into the array. Subsequent PAPI_read/PAPI_stop calls will then
 *        store the counter value based on the stat qualifier (i.e. avg,
 *        max, min, and sum). In the case of avg, it is a running average.
 *
 * @param ctx
 *   A structure containing the state, event ids, counters, and number
 *   of events.
 */
inline void CuptiProfile::calculate(cuda_range_ctx_t ctx)
{
    for (size_t i = 0; i < m_cupti_metric_names.size(); i++) {
        size_t pos = m_event_insertion_order[i];
        long long metric_value_ll = static_cast<long long>(m_metric_values[i]);
        // PAPI_read has previously not been called. Set initial counter values.
        if (m_samples_read == 0) {
            ctx->counters[pos] = metric_value_ll;
        }
        // PAPI_read has previously been called. Update counter values. 
        else {
            // Calculate the running avg.
            if (strstr(m_cupti_metric_names[i], "avg")) {
                ctx->counters[pos] = ( (m_samples_read * ctx->counters[pos]) + metric_value_ll ) / (m_samples_read + 1);
            }
            // Calculate the max across all reads.
            else if (strstr(m_cupti_metric_names[i], "max")) {
                ctx->counters[pos] = std::max(metric_value_ll, ctx->counters[pos]);
            }
            // Calculate the min across all reads.
            else if (strstr(m_cupti_metric_names[i], "min")) {
                ctx->counters[pos] = std::min(metric_value_ll, ctx->counters[pos]);
            }
            // Calculate the sum across all reads.
            else if (strstr(m_cupti_metric_names[i], "sum")) {
                ctx->counters[pos] += metric_value_ll;
            }
            // A rollup metric does not exist for the name.
            // Note, this will occur for CUPTI_METRIC_TYPE_RATIO.
            else {
                ctx->counters[pos] = metric_value_ll;
            }
        }
    }
    m_samples_read++;

    return;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiRangeProfilerDisable.
 */
inline int CuptiProfile::disable_range_profiler(void)
{
    CUpti_RangeProfiler_Disable_Params disable_params {};
    /// [in]
    disable_params.structSize = CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE;
    /// [in]
    disable_params.pPriv = nullptr;
    /// [in]
    disable_params.pRangeProfilerObject = m_range_profiler_object;
    CHECK_CUPTI_API_CALL( cuptiRangeProfilerDisablePtr(&disable_params) );
    m_range_profiler_object = nullptr;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerGetCounterAvailability.
 *
 *        This member function sets the private member variable
 *        m_counter_availability_image.
 */
inline int CuptiProfile::counter_availability(void)
{
    CUpti_Profiler_GetCounterAvailability_Params get_counter_availability_params {};
    /// [in]
    get_counter_availability_params.structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE;
    /// [in]
    get_counter_availability_params.pPriv = nullptr;
    /// [in]
    get_counter_availability_params.ctx = m_context;
    /// [in]
    get_counter_availability_params.pCounterAvailabilityImage = nullptr;
    /// [in]
    #if CUDART_VERSION >= 13010
    get_counter_availability_params.bAllowDeviceLevelCounters = 1;
    #endif
    CHECK_CUPTI_API_CALL( cuptiProfilerGetCounterAvailabilityPtr(&get_counter_availability_params) );
    m_counter_availability_image.resize(get_counter_availability_params.counterAvailabilityImageSize);
    /// [in]
    get_counter_availability_params.pCounterAvailabilityImage = m_counter_availability_image.data();
    /// [in/out]
    get_counter_availability_params.counterAvailabilityImageSize = m_counter_availability_image.size();
    CHECK_CUPTI_API_CALL( cuptiProfilerGetCounterAvailabilityPtr(&get_counter_availability_params) );

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiDeviceGetChipName.
 *
 *        This member function sets the private member variable
 *        m_chip_name.
 *
 * @param device_index
 *   An NVIDIA device index.
 */
inline int CuptiProfile::chip_name(int device_index)
{
    CUpti_Device_GetChipName_Params get_chip_name_params {};
    /// [in]
    get_chip_name_params.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
    /// [in]
    get_chip_name_params.pPriv = nullptr;
    /// [in]
    get_chip_name_params.deviceIndex = device_index;
    CHECK_CUPTI_API_CALL( cuptiDeviceGetChipNamePtr(&get_chip_name_params) );
    m_chip_name = get_chip_name_params.pChipName;
 
    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostInitialize.
 *
 *        This member function sets the private member variable
 *        m_host_object.
 */
inline int CuptiProfile::host_initialize(void)
{
    CUpti_Profiler_Host_Initialize_Params initializeParams {};
    /// [in] 
    initializeParams.structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;
    /// [in]
    initializeParams.pPriv = nullptr;
    /// [in]
    initializeParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    /// [in]
    initializeParams.pChipName = m_chip_name.c_str();
    /// [in]
    initializeParams.pCounterAvailabilityImage = m_counter_availability_image.data();
    /// [in]
    #if CUDART_VERSION >= 13020
    // Only valid for PM sampling and can be set to nullptr for CUPTI Range Profiler.
    initializeParams.pSinglePassMetricSetName = nullptr;
    #endif
    CHECK_CUPTI_API_CALL( cuptiProfilerHostInitializePtr(&initializeParams) );
    m_host_object = initializeParams.pHostObject;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that wraps
 *        the CUPTI call cuptiProfilerHostDeinitialize.
 *
 *        This member function sets the private member variable
 *        m_host_object back to a nullptr.
 */
inline int CuptiProfile::host_deinitialize(void)
{
    CUpti_Profiler_Host_Deinitialize_Params hostDeinitializeParams {};
    /// [in]
    hostDeinitializeParams.structSize = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE;
    /// [in]
    hostDeinitializeParams.pPriv = nullptr;
    /// [in]
    hostDeinitializeParams.pHostObject = m_host_object;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostDeinitializePtr(&hostDeinitializeParams) );
    m_host_object = nullptr;

    return PAPI_OK;
}

/**
 * @brief A member function of the class CuptiProfile that
 *        destroys OWNED CUDA contexts.
 *
 *        Normally we can leave this to the destructor; however, the global class destructor
 *        is not called until the end of int main. Therefore, if PAPI_shutdown is called
 *        before the last } cuCtxDestroyPtr is set equal to nullptr and will lead
 *        to a segmentation fault in the destructor.      
 */
inline void CuptiProfile::destroy_context(void)
{
    if (m_context_ownership == true && m_context != nullptr) {
        cuCtxDestroyPtr(m_context);
        m_context_ownership = false;
        m_context = nullptr;
    }

    return;
}

/**
 *  @}
 ******************************************************************************/

/**
 *  @}
 ******************************************************************************/

/***************************************************************************//**
 *  @name   Helper functions for the PAPI instrumentation
 *  @{
 */

/**
  * @brief A wrapper for the CUPTI call cuptiProfilerInitialize.
*/
int initialize_cupti_profiler(void)
{
    CUpti_Profiler_Initialize_Params profiler_initialize_params {};
    /// [in]
    profiler_initialize_params.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    /// [in]
    profiler_initialize_params.pPriv = nullptr;
    CHECK_CUPTI_API_CALL( cuptiProfilerInitializePtr(&profiler_initialize_params) );

    return PAPI_OK;
}

/**
  * @brief A wrapper for the CUPTI call cuptiDeviceGetChipName.
  *
  * @param device_index
  *   Index of an NVIDIA device (e.g. 0, 1, 2, 3, ..).
  * @param &device_chip_name
  *   Stores the chipname for the specificed device index.
*/
int get_device_chip_name(int device_index, std::string &chip_name)
{
    CUpti_Device_GetChipName_Params get_chip_name_params {};
    /// [in]
    get_chip_name_params.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
    /// [in]
    get_chip_name_params.pPriv = nullptr;
    /// [in]
    get_chip_name_params.deviceIndex = device_index;
    CHECK_CUPTI_API_CALL( cuptiDeviceGetChipNamePtr(&get_chip_name_params) );
    chip_name = get_chip_name_params.pChipName;

    return PAPI_OK;
}

/**
  * @brief For each device on the system, get the CUPTI metrics.
  *
  *        The base metrics will be stored as keys in an std::map.
  *        The value associated with the base metric keys will be a
  *        structure containing vectors to store the rollup metrics,
  *        sub metrics, and devices associated with the key.
  *
  * @param device_count
  *   Number of NVIDIA devices detected on the system.
*/
int enumerate_and_collect_metrics_per_device(int device_count)
{
    // Check for a user context on the calling CPU thread.
    CUcontext context = nullptr;
    CHECK_DRIVER_API_CALL( cuCtxGetCurrentPtr(&context) );
    if (context != nullptr) {
        // Pop the context from the calling CPU thread.
        CUcontext popped_context;
        CHECK_DRIVER_API_CALL( cuCtxPopCurrentPtr(&popped_context) );
    }

    std::map<std::string, std::vector<std::string>> cached_device_metrics;
    for (int device_index = 0; device_index < device_count; device_index++) {
         std::string chip_name {};
         CHECK_INTERNAL_FUNC_CALL( get_device_chip_name(device_index, chip_name) );

        // CUPTI metrics have not been obtained for the current chip name.
        if (cached_device_metrics.count(chip_name) == 0) {
            CUcontext context;
            unsigned int flags = 0;
            CHECK_DRIVER_API_CALL( cuCtxCreatePtr(&context, (CUctxCreateParams*)0, flags, device_index) );

            CuptiProfile profile = CuptiProfile(context, true);

            CHECK_INTERNAL_FUNC_CALL( profile.counter_availability() );

            CHECK_INTERNAL_FUNC_CALL( profile.chip_name(device_index) );

            CHECK_INTERNAL_FUNC_CALL( profile.host_initialize() );

            for (size_t metric_type = 0; metric_type < CUPTI_METRIC_TYPE__COUNT; metric_type++) {
                const char **base_metric_names = nullptr;
                size_t number_of_base_metrics = 0;
                CHECK_INTERNAL_FUNC_CALL( profile.base_metrics(metric_type, &base_metric_names, number_of_base_metrics) );

                for (size_t base_metric_index = 0; base_metric_index < number_of_base_metrics; base_metric_index++) {
                    std::string base_metric_name = base_metric_names[base_metric_index];
                    cupti_metrics[base_metric_name].name_id = cupti_metrics.size();
                    cupti_metrics[base_metric_name].cupti_metric_type = (CUpti_MetricType) metric_type;
                    cupti_metrics[base_metric_name].device_ids.push_back(std::to_string(device_index));

                    const char **sub_metric_names = nullptr;
                    size_t number_of_sub_metrics = 0;
                    CHECK_INTERNAL_FUNC_CALL( profile.sub_metrics(metric_type, base_metric_name.c_str(), &sub_metric_names, number_of_sub_metrics) );

                    for (size_t sub_metric_index = 0; sub_metric_index < number_of_sub_metrics; sub_metric_index++) {
                        std::string sub_metric_name = sub_metric_names[sub_metric_index];
                        // Erase the leading period.
                        sub_metric_name.erase(0, 1);
                        // Handle counter metrics. Note, counter metrics can appear as .rollup or .rollup.submetric.
                        if (metric_type == CUPTI_METRIC_TYPE_COUNTER) {
                            int period_count = std::count (sub_metric_name.begin(), sub_metric_name.end(), '.');
                            // Only .rollup appeared.
                            if (period_count == 0) {
                                if (std::find(cupti_metrics[base_metric_name].stats.begin(), cupti_metrics[base_metric_name].stats.end(), sub_metric_name) == cupti_metrics[base_metric_name].stats.end()) {
                                    cupti_metrics[base_metric_name].stats.push_back(sub_metric_name);
                                }
                            }
                            // .rollup.submetric appeared.
                            else {
                                std::istringstream input(sub_metric_name);
                                std::string rollup;
                                std::getline(input, rollup, '.');
                                if (std::find(cupti_metrics[base_metric_name].stats.begin(), cupti_metrics[base_metric_name].stats.end(), rollup) == cupti_metrics[base_metric_name].stats.end()) {
                                    cupti_metrics[base_metric_name].stats.push_back(rollup);
                                }

                                std::string submetric;
                                std::getline(input, submetric, '.');
                                if (std::find(cupti_metrics[base_metric_name].submetrics.begin(), cupti_metrics[base_metric_name].submetrics.end(), submetric) == cupti_metrics[base_metric_name].submetrics.end()) {
                                    cupti_metrics[base_metric_name].submetrics.push_back(submetric);
                                }
                            }
                        }
                        // Handle ratio metrics. Note, ratio metrics appear as .submetric.
                        else if (metric_type == CUPTI_METRIC_TYPE_RATIO) {
                            if (std::find(cupti_metrics[base_metric_name].submetrics.begin(), cupti_metrics[base_metric_name].submetrics.end(), sub_metric_name) == cupti_metrics[base_metric_name].submetrics.end()) {
                                cupti_metrics[base_metric_name].submetrics.push_back(sub_metric_name);
                            }
                        }
                        // Handle throughput metrics. Note, throughput metrics appear as .rollup.submetric.
                        else if (metric_type == CUPTI_METRIC_TYPE_THROUGHPUT) {
                            std::istringstream input(sub_metric_name);
                            std::string rollup;
                            std::getline(input, rollup, '.');
                            if (std::find(cupti_metrics[base_metric_name].stats.begin(), cupti_metrics[base_metric_name].stats.end(), rollup) == cupti_metrics[base_metric_name].stats.end()) {
                                cupti_metrics[base_metric_name].stats.push_back(rollup);
                            }

                            std::string submetric;
                            std::getline(input, submetric, '.');
                            if (std::find(cupti_metrics[base_metric_name].submetrics.begin(), cupti_metrics[base_metric_name].submetrics.end(), submetric) == cupti_metrics[base_metric_name].submetrics.end()) {
                                cupti_metrics[base_metric_name].submetrics.push_back(submetric);
                            }
                        }
                        // CUpti_MetricType is not implemented.
                        else {
                            SUBDBG("NVIDIA has added a new CUpti_MetricType (%u).\n", metric_type);
                            return PAPI_EBUG;
                        }
                    }

                    // Cache the base_metric_name based on the device.
                    // We do this such that we can save on initialization time
                    // if repeated devices exist (i.e. 4 * A100's) on a system.
                    cached_device_metrics[chip_name].push_back(base_metric_name);
                }
            }
            CHECK_INTERNAL_FUNC_CALL( profile.host_deinitialize() );
        }
        // CUPTI metrics have been obtained for the current chip name.
        else {
            for (std::string base_metric_name : cached_device_metrics[chip_name]) {
                cupti_metrics[base_metric_name].device_ids.push_back(std::to_string(device_index));
            }
        }
    }

    // Store the maps keys as a vector.
    for (const std::pair<std::string, cupti_metric_info_t> pair : cupti_metrics) {
        cupti_metrics_keys.push_back(pair.first);
    }

    // Push the user's context back onto the calling CPU thread.
    if (context != nullptr) {
        CHECK_DRIVER_API_CALL( cuCtxPushCurrentPtr(context) );
    }

    return PAPI_OK;
}

/**
  * @brief A wrapper for the CUPTI call cuptiProfilerHostGetMaxNumHardwareMetricsPerPass.
  *
  * @param device_chip_name
  *   Name of the NVIDIA chip.
  * @param &max_number_of_hardware_metrics
  *   Stores the max number of hardware metrics per pass for the NVIDIA chip.
*/
int get_max_num_hardware_metrics_per_pass(std::string &device_chip_name, int &max_number_of_hardware_metrics)
{
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params get_max_num_hardware_metrics_per_pass_params {};
    /// [in]
    get_max_num_hardware_metrics_per_pass_params.structSize = CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params_STRUCT_SIZE;
    /// [in]
    get_max_num_hardware_metrics_per_pass_params.pPriv = nullptr;
    /// [in]
    get_max_num_hardware_metrics_per_pass_params.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    /// [in]
    get_max_num_hardware_metrics_per_pass_params.pChipName = device_chip_name.c_str();
    /// [in]
    get_max_num_hardware_metrics_per_pass_params.pCounterAvailabilityImage = nullptr;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetMaxNumHardwareMetricsPerPassPtr(&get_max_num_hardware_metrics_per_pass_params) );
    max_number_of_hardware_metrics = get_max_num_hardware_metrics_per_pass_params.maxMetricsPerPass;

    return PAPI_OK;
}

/**
  * @brief A wrapper for the CUPTI call cuptiProfilerHostGetSupportedChips.
  *
  * @param &number_of_supported_chips
  *   Stores the number of supported chips.
  * @param *const **supported_chip_names
  *   Stores the list of supported chips.
*/
int get_supported_chips(int &number_of_supported_chips, const char *const **supported_chip_names)
{
    CUpti_Profiler_Host_GetSupportedChips_Params get_supported_chip_params {};
    /// [in]
    get_supported_chip_params.structSize = CUpti_Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE;
    /// [in]
    get_supported_chip_params.pPriv = nullptr;
    /// [in/out]
    get_supported_chip_params.numChips = 0;
    CHECK_CUPTI_API_CALL( cuptiProfilerHostGetSupportedChipsPtr(&get_supported_chip_params) );
    number_of_supported_chips = get_supported_chip_params.numChips;
    *supported_chip_names = get_supported_chip_params.ppChipNames;

    return PAPI_OK;
}

/**
  * @brief Verify is all NVIDIA chips on the system are supported.
  *
  * @param device_count
  *   Number of NVIDIA devices on the system.
*/
int check_for_chips_not_supported_on_the_system(int device_count)
{
    int number_of_supported_chips = 0;
    const char *const *supported_chip_names = nullptr;
    CHECK_INTERNAL_FUNC_CALL( get_supported_chips(number_of_supported_chips, &supported_chip_names) );

    std::map<size_t, std::string> chips_not_supported;
    for (size_t chip_index = 0; chip_index < static_cast<size_t>(device_count); chip_index++) {
        std::string chip_name;
        CHECK_INTERNAL_FUNC_CALL( get_device_chip_name(chip_index, chip_name) );

        int chip_supported = false;
        for (int supp_chip_index = 0; supp_chip_index < number_of_supported_chips; supp_chip_index++) {
            if (chip_name.compare(supported_chip_names[supp_chip_index]) == 0) {
                chip_supported = true;
            }
        }

        if (chip_supported == false) {
            chips_not_supported[chip_index] = chip_name;
        }
    }

    if (chips_not_supported.empty() == false) {
        std::string first_formatter = " - ";
        std::string second_formatter = ", ";
        std::stringstream ss;
        ss << "The cuda_range component does not support devices (index - chipname): ";
        for (auto entry : chips_not_supported) {
            ss << entry.first << first_formatter << entry.second << second_formatter;
        }
        std::string formatted_error_message = ss.str();

        // Remove the trailing second_formatter (", ").
        int starting_index = formatted_error_message.length() - second_formatter.length();
        formatted_error_message.erase(starting_index, second_formatter.length());
        formatted_error_message += ". A possible reason is because of the Cuda Toolkit in use.";

        cuda_range_set_last_err_msg(formatted_error_message);
        return PAPI_ECMP;
    }

    return PAPI_OK;
}

/**
  * @brief Use the stat, submetric, and device qualifiers
  *        to bit mask an event id.
  *
  * @param *native_event_info
  *   Structure containing integer values for stat, submetric, device,
  *   flags, and name_id.
  * @param *event_id
  *   Stores the bit masked event id.
*/
int encode_evt_id(native_event_info_t *native_event_info, uint32_t *event_id)
{
    *event_id  = (uint32_t)(native_event_info->stat     << STAT_SHIFT);
    *event_id |= (uint32_t)(native_event_info->submetric << SUBMETRIC_SHIFT);
    *event_id |= (uint32_t)(native_event_info->device   << DEVICE_SHIFT);
    *event_id |= (uint32_t)(native_event_info->flags    << QLMASK_SHIFT);
    *event_id |= (uint32_t)(native_event_info->name_id   << NAMEID_SHIFT);

    return PAPI_OK;
}

/**
  * @brief Decode the bit masked event id to get the stat,
  *        submetric, and device qualifier indexes.
  *
  * @param *event_id
  *   Stores the bit masked event id.
  * @param *native_event_info
  *   Structure containing integer values for stat, submetric, device,
  *   flags, and name_id.
*/
int decode_evt_id(uint32_t event_id, native_event_info_t *native_event_info)
{
    native_event_info->stat          = (uint32_t)((event_id & STAT_MASK) >> STAT_SHIFT);
    native_event_info->submetric     = (uint32_t)((event_id & SUBMETRIC_MASK) >> SUBMETRIC_SHIFT);
    native_event_info->device        = (uint32_t)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    native_event_info->flags         = (uint32_t)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    native_event_info->name_id        = (uint32_t)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    return PAPI_OK;
}

/**
  * @brief Formats the qualifier vectors (stat, submetric, or device id) for a CUPTI metric name
  *        into a comma separated list.
  *
  * @param qualifiers
  *   A list of valid qualifiers for a CUPTI metric name.
  * @param &formatted_qualifiers
  *   Stores the comma separated list of qualifiers.
*/
void format_qualifiers_for_code_to_info(std::vector<std::string> qualifiers, std::string &formatted_qualifiers)
{
    for (auto const &entry : qualifiers) {
        // Add a comma.
        if (entry != qualifiers.back()) {
            formatted_qualifiers += entry + ", ";
        }
        // Do not add a comma - last element.
        else {
            formatted_qualifiers += entry;
        }
    }

    return;
}

/**
  * @brief Take a user provided cuda_range native event name and
  *        parse it for the CUPTI base metric name.
  *
  * @param *cuda_range_native_event_name
  *   A user provided cuda_range native event name.
  * @param *cupti_base_name
  *   Stores the cupti_base_name.
  * @param string_length
  *   Maximum legnth the CUPTI base metric name can be.
*/
int parse_native_event_name_for_cupti_base_name(const char *cuda_range_native_event_name, char *cupti_base_name, int string_length)
{
    int length_of_base_name = 0;
    for (int c = 0; cuda_range_native_event_name[c] != ':' && cuda_range_native_event_name[c] != '\0'; c++) {
        length_of_base_name++;
    }

    int string_len = snprintf(cupti_base_name, string_length, "%.*s", length_of_base_name, cuda_range_native_event_name);
    if (string_len < 0 || string_len >= string_length) {
        SUBDBG("Failed to fully write the base metric name into the buffer.");
        return PAPI_EBUF;
    }

    return PAPI_OK;
}

/**
  * @brief Take a user provided cuda_range native event name and
  *        parse the qualifiers.
  *
  * @param *cuda_range_native_event_name
  *   A user provided cuda_range native event name.
  * @param *qualifier_name
  *   Name of the qualifier to search for.
  * @param qualifier_value
  *   Stores the value associated with the qualifier.
*/
int parse_native_event_name_for_qualifiers(const char *cuda_range_native_event_name, const char *qualifier_name, std::string &qualifier_value)
{
    const char *position_of_qualifier = strstr(cuda_range_native_event_name, qualifier_name);
    if (position_of_qualifier != nullptr) {
        const char *beginning_of_qualifier_value = position_of_qualifier + strlen(qualifier_name);
        for (int c = 0; beginning_of_qualifier_value[c] != ':' && beginning_of_qualifier_value[c] != '\0'; c++) {
             qualifier_value += beginning_of_qualifier_value[c];
        }
    }

    return PAPI_OK;
}

/**
  * @brief Get the qualifier name for the qualifier index.
  *
  * @param decoded_flags
  *   The resulting flags from the encoded cuda_range native event.
  * @param qualifier_bitmask
  *   The specific flag we want to check against to see if it is present.
  * @param decoded_qualifier_index
  *   The encoded qualifier index corresponding to a vector entry.
  * @param &vector_of_qualifiers
  *   The vector to index into using decoded_qualifier_index.
  * @param required
  *   True or false -- if the qualifier is required we add a default.
  * @param qualifier_name
  *   Stores the resulting qualifier value.
*/
int obtain_qualifier_name(int decoded_flags, int qualifier_bitmask, size_t decoded_qualifier_index, std::vector<std::string> &vector_of_qualifiers, bool required, std::string &qualifier_name)
{
    // Qualifier is present in cuda_range native event code. 
    if (decoded_flags & qualifier_bitmask) {
        if (decoded_qualifier_index < vector_of_qualifiers.size()) {
            qualifier_name = vector_of_qualifiers[decoded_qualifier_index];
        }
        else {
            SUBDBG("Invalid cuda_range native event code. The decoded qualifier index (%d) does not index (index is %d)"
                   " into the vector (size is %u).", decoded_qualifier_index, vector_of_qualifiers.size()); 
            return PAPI_ENOEVNT;
        }
    }
    // Qualifier is not present in the cuda_range native_event code, but is required.
    else if (required) {
        qualifier_name = vector_of_qualifiers[0];
    }
    
    return PAPI_OK;
}

/**
  * @brief A helper function to take a user provided native event code and
  *        convert it to the corresponding cuda_range native event name. 
  *
  * @param native_event_code
  *   A user provided native event code.
  * @param &base_name
  *   Stores the corresponding base name.
  * @param &stat_name
  *   Stores the corresponding stat name.
  * @param &sub_metric_name
  *   Stores the corresponding sub-metric name.
  * @param &device_id
  *   Stores the corresponding device id.
*/
int native_event_code_to_native_event_name(unsigned int native_event_code, std::string &base_name, std::string &stat_name, std::string &sub_metric_name, std::string &device_id)
{   
    native_event_info_t native_event_info;
    decode_evt_id(native_event_code, &native_event_info);

    // Native event base name is present in the map.
    if ((size_t) native_event_info.name_id < cupti_metrics_keys.size()) {
        base_name = cupti_metrics_keys[native_event_info.name_id];
    }   
    // Native event base name is not present in the map.
    else {
        SUBDBG("The eventcode provided does not index into (index is %u) the vector of keys (size of vector is %u)."
               " Therefore, the basename is not valid.\n", native_event_info.name_id, cupti_metrics_keys.size());
        return PAPI_ENOEVNT;
    }

    cupti_metric_info_t cupti_metric_info = cupti_metrics[base_name];
    int papi_errno = PAPI_OK;
    switch(cupti_metric_info.cupti_metric_type) {
        // Handle CUPTI counter metrics.
        case CUPTI_METRIC_TYPE_COUNTER:
            papi_errno = obtain_qualifier_name(native_event_info.flags, STAT_FLAG, native_event_info.stat, cupti_metric_info.stats,
                                               true, stat_name);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            papi_errno = obtain_qualifier_name(native_event_info.flags, SUBMETRIC_FLAG, native_event_info.submetric, cupti_metric_info.submetrics,
                                               false, sub_metric_name);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            break;
        // Handle CUPTI ratio metrics.
        case CUPTI_METRIC_TYPE_RATIO:
            // Stat qualifier is present and is not supported for CUPTI ratio metrics.
            if (native_event_info.flags & STAT_FLAG) {
                SUBDBG("CUPTI ratio metrics do not support a stat qualifier.\n");
                return PAPI_ENOEVNT;
            }

            papi_errno = obtain_qualifier_name(native_event_info.flags, SUBMETRIC_FLAG, native_event_info.submetric, cupti_metric_info.submetrics,
                                               true, sub_metric_name);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            break;
        // Handle CUPTI throughput metrics.
        case CUPTI_METRIC_TYPE_THROUGHPUT:
            papi_errno = obtain_qualifier_name(native_event_info.flags, STAT_FLAG, native_event_info.stat, cupti_metric_info.stats,
                                               true, stat_name);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }
            
            papi_errno = obtain_qualifier_name(native_event_info.flags, SUBMETRIC_FLAG, native_event_info.submetric, cupti_metric_info.submetrics,
                                               true, sub_metric_name);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            break;
        // The CUpti_MetricType is not accounted for.
        default:
            SUBDBG("The CUpti_MetricType (%d) is not valid for the current workflow. Possible reason is the"
                   "  addition of a new CUpti_MetricType.\n", cupti_metric_info.cupti_metric_type);
            return PAPI_EBUG;
    }

    // Device qualifier is present.
    if (native_event_info.flags & DEVICE_FLAG) {
        // Device is present in the vector.
        if ((size_t) native_event_info.device < cupti_metric_info.device_ids.size()) {
            device_id = cupti_metric_info.device_ids[native_event_info.device];
        }
        // Device is not present in the vector.
        else {
            SUBDBG("The eventcode provided does not index into (index is %d) the vector of devices (size of vector is %d)."
                   " Therefore, the device is not valid.\n", native_event_info.device, cupti_metric_info.device_ids.size());
            return PAPI_ENOEVNT;
        }
    }
    // Device qualifier is not present -- use default.
    else {
        device_id = cupti_metric_info.device_ids[0];
    }

    return PAPI_OK;
}

/**
  * @brief Reserve an NVIDIA device.
  *
  *        The cuda_range component only allows an NVIDIA device
  *        to be used on a single thread. Therefore, internally
  *        the NVIDIA device will be "reserved" for use. 
  *
  * @param device_index
  *   The index of the device to be reserved.
*/
int cuda_range_reserve_device(int device_index)
{
    uint64_t local_bitmask = 0; 
    local_bitmask |= (1 << device_index);

    if (local_bitmask & device_bitmask) {
        SUBDBG("The cuda_range component only supports one thread per device and device %d is already in use.\n", device_index);
        return PAPI_ECNFLCT;
    }    

    _papi_hwi_lock(_cuda_range_lock);
    device_bitmask |= local_bitmask;
    _papi_hwi_unlock(_cuda_range_lock);

    return PAPI_OK;
}

/**
  * @brief Un-reserve the NVIDIA device reserved by the call
  *        cuda_range_reserve_device.
  *
  * @param device_index
  *   The index of the device to be un-reserved.
*/
int cuda_range_unreserve_device(int device_index)
{
    uint64_t local_bitmask = 0; 
    local_bitmask |= (1 << device_index);

    if ((local_bitmask & device_bitmask) != local_bitmask) {
        SUBDBG("Device %d is not reserved and should be.\n", device_index);
        return PAPI_EBUG;
    }    

    _papi_hwi_lock(_cuda_range_lock);
    device_bitmask ^= local_bitmask;
    _papi_hwi_unlock(_cuda_range_lock);

    return PAPI_OK;
}

typedef struct added_native_events_data_t
{
    std::vector<std::string> cupti_metric_names;
    std::vector<size_t> event_insertion_order;
} added_native_events_data_t;

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name PAPI instrumentation
 *  @{
 */

/** 
  * @brief Initial initialization of the cuda_range component.
  *
  *        The following is done within this function call:
  *        1. Load the function pointers for the Cuda Driver,
  *        Cuda Runtime, and CUPTI APIs.
  *        2. Verify NVIDIA devices exist on the machine.
  *        3. Initialize the CUPTI Profiler interface.
*/
extern "C" int initialize_cuda_range_component(void)
{
    int papi_errno = load_cuda_driver_function_pointers();
    if (papi_errno != PAPI_OK) {
        std::string error_message = "Unable to load the Cuda Driver API's. Try setting PAPI_CUDA_RANGE_ROOT or PAPI_CUDA_RANGE_DRIVER.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    papi_errno = load_cuda_runtime_function_pointers();
    if (papi_errno != PAPI_OK) {
        std::string error_message = "Unable to load the Cuda Runtime API's. Try setting PAPI_CUDA_RANGE_ROOT or PAPI_CUDA_RANGE_RUNTIME.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    papi_errno = load_cupti_function_pointers();
    if (papi_errno != PAPI_OK) {
        std::string error_message = "Unable to load the CUPTI API's. Try setting PAPI_CUDA_RANGE_ROOT or PAPI_CUDA_RANGE_CUPTI.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    int device_count = 0;
    CHECK_RUNTIME_API_CALL( cudaGetDeviceCountPtr(&device_count) );
    // From documentation, cudaGetDeviceCount always succeeds; therefore,
    // we need to verify device_count is not 0.
    if (device_count == 0) {
        std::string error_message = "No NVIDIA devices detected on the machine.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    // During initial development of the cuda_range component, using Cuda Toolkit 12.9 resulted in
    // cuptiProfilerHostGetMetricProperties returning the error CUPTI_ERROR_INVALID_METRIC_NAME.
    // However, Cuda Toolkit 13.0 does not have this error occur. Therefore, we restrict users to
    // Cuda Toolkit 13.0.
    int cuda_runtime_version = 0, minimum_cuda_runtime_version = 13000;
    CHECK_RUNTIME_API_CALL( cudaRuntimeGetVersionPtr(&cuda_runtime_version) );
    if (cuda_runtime_version < minimum_cuda_runtime_version) {
        std::stringstream ss;
        ss << "The cuda_range component requires a Cuda Toolkit >= 13.0. Detected Cuda Toolkit in use is "
           << cuda_runtime_version / 1000 << "."
           << (cuda_runtime_version % 1000) / 10
           << ".";
        std::string error_message = ss.str();

        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    papi_errno = initialize_cupti_profiler();
    if (papi_errno != PAPI_OK) {
        std::string error_message = "Unable to initialize the CUPTI profiler interface. A possible reason is"
                                   " a mismatched Cuda Toolkit and NVIDIA architecture. Try setting PAPI_CUDA_RANGE_ROOT"
                                   " to a newer Cuda Toolkit version.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    papi_errno = check_for_chips_not_supported_on_the_system(device_count);
    if (papi_errno != PAPI_OK) {
        return PAPI_ECMP;
    }

    CHECK_DRIVER_API_CALL( cuInitPtr(0) );

    papi_errno = enumerate_and_collect_metrics_per_device(device_count);
    if (papi_errno != PAPI_OK) {
        std::string error_message = "Failed to create map containing cuda_range native events.";
        cuda_range_set_last_err_msg(error_message);
        return PAPI_ECMP;
    }

    return PAPI_OK;
}

/**
  * @brief For each device the maximum number of hardware metrics
  *        that can be scheduled in a single pass for a chip are retrieved.
  *
  *        Note 1: In the case a system has heterogeneous NVIDIA
  *        architectures (i.e. H100 and V100) the maximum will be
  *        taken between them. 
  *
  *        Note 2: Per NVIDIA, while this represents a theoretical upper limit,
  *        practical constraints may prevent reaching this threshold for a specfic
  *        set of metrics. Furthermore, the maximum achievable value is contingent
  *        upon the characteristics and architecture of the chip in question.
  *
  * @param *maximum_number_of_counters
  *   Stores the maximum number of counters across all NVIDIA devivces on the system.. 
*/
extern "C" int get_the_maximum_number_of_hardware_metrics_per_device(int *maximum_number_of_counters)
{
    int device_count = 0; 
    CHECK_RUNTIME_API_CALL( cudaGetDeviceCountPtr(&device_count) );

    std::vector<int> number_of_counters_per_device;
    for (size_t device_index = 0; device_index < static_cast<size_t>(device_count); device_index++) {
        std::string deviceChipName;
        int retval = get_device_chip_name(device_index, deviceChipName);
        if (retval != PAPI_OK) {
            return retval;
        }    

        int maxNumberOfHardwareMetrics = 0; 
        retval = get_max_num_hardware_metrics_per_pass(deviceChipName, maxNumberOfHardwareMetrics);
        if (retval != PAPI_OK) {
            return retval;
        }    

        number_of_counters_per_device.push_back(maxNumberOfHardwareMetrics);
    }    
 
    *maximum_number_of_counters = *std::max_element(number_of_counters_per_device.begin(), number_of_counters_per_device.end());
 
    return PAPI_OK;
}

/**
  * @brief Take a user provided native event code and enumerate to the
  *        next native event code.
  *
  *        Implementation only accounts for PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS,
  *        and PAPI_ENUM_NTV_UMASKS modifiers.
  *
  * @param *native_event_code
  *   A user provided native event code.
  * @param modifier
  *   Value to indicate the enumeration of events, i.e. PAPI_ENUM_FIRST.
*/
extern "C" int cuda_range_event_enum(uint32_t *native_event_code, int modifier)
{
    int papi_errno;

    std::string base_name;
    native_event_info_t native_event_info;
    switch(modifier) {
        case PAPI_ENUM_FIRST:
            native_event_info.stat = 0;
            native_event_info.submetric = 0;
            native_event_info.device = 0;
            native_event_info.flags = NOQUAL_FLAG;
            native_event_info.name_id = 0;
            encode_evt_id(&native_event_info, native_event_code);
            papi_errno = PAPI_OK;
            break;
        case PAPI_ENUM_EVENTS:
            decode_evt_id(*native_event_code, &native_event_info);
            // Enumeration does not exceed map.
            if (native_event_info.name_id + 1 < cupti_metrics.size()) {
                native_event_info.stat = 0;
                native_event_info.submetric = 0;
                native_event_info.device = 0;
                native_event_info.flags = NOQUAL_FLAG;
                native_event_info.name_id++;
                encode_evt_id(&native_event_info, native_event_code);
                papi_errno = PAPI_OK;
                break;
            }
            // Enumeration exceeds map.
            else {
                SUBDBG("Attempted to exceed the size of the map. Therefore, no metrics are left to enumerate.\n");
                papi_errno = PAPI_ENOEVNT;
                break;
            }

            papi_errno = PAPI_ENOEVNT;
            break;
        case PAPI_NTV_ENUM_UMASKS:
            decode_evt_id(*native_event_code, &native_event_info);

            base_name = cupti_metrics_keys[native_event_info.name_id];
            if (native_event_info.flags == NOQUAL_FLAG && cupti_metrics[base_name].stats.size() > 0) {
                native_event_info.stat = 0;
                native_event_info.submetric = 0;
                native_event_info.device = 0;
                native_event_info.flags = STAT_FLAG;
                encode_evt_id(&native_event_info, native_event_code);
                papi_errno = PAPI_OK;
                break;
            }

            if ((native_event_info.flags == NOQUAL_FLAG || native_event_info.flags == STAT_FLAG) && cupti_metrics[base_name].submetrics.size() > 0) {
                native_event_info.stat = 0;
                native_event_info.submetric = 0;
                native_event_info.device = 0;
                native_event_info.flags = SUBMETRIC_FLAG;
                encode_evt_id(&native_event_info, native_event_code);
                papi_errno = PAPI_OK;
                break;
            }

            if (native_event_info.flags == SUBMETRIC_FLAG && cupti_metrics[base_name].device_ids.size() > 0) {
                native_event_info.stat = 0;
                native_event_info.submetric = 0;
                native_event_info.device = 0;
                native_event_info.flags = DEVICE_FLAG;
                encode_evt_id(&native_event_info, native_event_code);
                papi_errno = PAPI_OK;
                break;
            }

            papi_errno = PAPI_ENOEVNT;
            break;
        default:
            SUBDBG("The provided modifier is not supported in the cuda_range component.\n");
            papi_errno = PAPI_EINVAL;
    }   

    return papi_errno;
}

/**
  * @brief Take a user provided cuda_range native event name and
  *        convert it to the corresponding cuda_range native event
  *        code.
  *
  * @param *native_event_name
  *   A user provided cuda_range native event name.
  * @param *native_event_code
  *   Stores the corresponding cuda_range native event code.
*/
extern "C" int cuda_range_native_event_name_to_native_event_code(const char *native_event_name, uint32_t *native_event_code)
{
    native_event_info_t native_event_info = {};

    // Parse the base name.
    char base_metric_name[PAPI_2MAX_STR_LEN];
    CHECK_INTERNAL_FUNC_CALL( parse_native_event_name_for_cupti_base_name(native_event_name, base_metric_name, PAPI_MAX_STR_LEN) );

    std::map<std::string, cupti_metric_info_t>::iterator eventEntry = cupti_metrics.find(base_metric_name);
    // Base name exists.
    if (eventEntry != cupti_metrics.end()) {
        native_event_info.name_id = std::distance(cupti_metrics.begin(), eventEntry);    
    }
    // Base name does not exist.
    else {
        SUBDBG("The provided base name (%s) is not valid.\n", base_metric_name);
        return PAPI_ENOEVNT;
    }

    std::string value_of_qualifier;
    cupti_metric_info_t cupti_metric_info = eventEntry->second;

    // Parse for the stat qualifier.
    const char *name_of_stat_qualifier = ":stat=";
    CHECK_INTERNAL_FUNC_CALL( parse_native_event_name_for_qualifiers(native_event_name, name_of_stat_qualifier, value_of_qualifier) );
    // Stat qualifier found.
    if (value_of_qualifier.empty() == false) {
        std::vector<std::string>::iterator stats_iter = std::find(cupti_metric_info.stats.begin(), cupti_metric_info.stats.end(), value_of_qualifier);
        // User provided a valid stat, update from index 0 (default).
        if (stats_iter != cupti_metric_info.stats.end()) {
            native_event_info.stat = std::distance(cupti_metric_info.stats.begin(), stats_iter);
            native_event_info.flags = STAT_FLAG;
        }
        // User did not provide a valid stat.
        else {
            SUBDBG("The provided stat (%s) is not a valid option.\n", value_of_qualifier.c_str());
            return PAPI_ENOEVNT;
        }
    }
    value_of_qualifier.clear();

    // Parse for the submetric qualifier.
    const char *name_of_sub_qualifier = ":submetric=";
    CHECK_INTERNAL_FUNC_CALL( parse_native_event_name_for_qualifiers(native_event_name, name_of_sub_qualifier, value_of_qualifier) );
    // Submetric qualifier found.
    if (value_of_qualifier.empty() == false) {
        std::vector<std::string>::iterator sub_metrics_iter  = std::find(cupti_metric_info.submetrics.begin(), cupti_metric_info.submetrics.end(), value_of_qualifier);
        // User provided a valid submetric, update from index 0 (default).
        if (sub_metrics_iter != cupti_metric_info.submetrics.end()) {
            native_event_info.submetric = std::distance(cupti_metric_info.submetrics.begin(), sub_metrics_iter);
            native_event_info.flags |= SUBMETRIC_FLAG;
        }
        // User did not provide a valid submetric.
        else {
            SUBDBG("The provided submetric (%s) is not a valid option.\n", value_of_qualifier.c_str());
            return PAPI_ENOEVNT;
        }
    }
    value_of_qualifier.clear();

    // Parse for the device qualifier.
    const char *name_of_dev_qualifier = ":device=";
    CHECK_INTERNAL_FUNC_CALL( parse_native_event_name_for_qualifiers(native_event_name, name_of_dev_qualifier, value_of_qualifier) );
    // User provided a device qualifier.
    if (value_of_qualifier.empty() == false) {
        std::vector<std::string>::iterator device_ids_iter = std::find(cupti_metric_info.device_ids.begin(), cupti_metric_info.device_ids.end(), value_of_qualifier);
        // User provided a valid device, update from index 0 (default).
        if (device_ids_iter != cupti_metric_info.device_ids.end()) {
            native_event_info.device = std::distance(cupti_metric_info.device_ids.begin(), device_ids_iter);
            native_event_info.flags |= DEVICE_FLAG;
        }
        // User did not provide a valid device. 
        else {
            SUBDBG("The provided device (%s) is not a valid option.\n", value_of_qualifier.c_str());
            return PAPI_ENOEVNT; 
        }
    }

    encode_evt_id(&native_event_info, native_event_code);

    return PAPI_OK;
}

/**
  * @brief Take a user provided cuda_range native event code and
  *        convert it to the corresponding cuda_range native event
  *        name.
  *
  * @param *native_event_code
  *   A user provided cuda_range native event code.
  * @param *native_event_name
  *   Stores the corresponding cuda_range native event name.
*/
extern "C" int cuda_range_native_event_code_to_native_event_name(unsigned int native_event_code, char *native_event_name, int len)
{
    std::string base_name, stat_name, sub_metric_name, device_id;
    CHECK_INTERNAL_FUNC_CALL( native_event_code_to_native_event_name(native_event_code, base_name, stat_name, sub_metric_name, device_id) );

    std::string cuda_range_native_event_name = base_name;
    // Concatenate stat qualifier.
    if (stat_name.empty() == false) {
        cuda_range_native_event_name += ":stat=" + stat_name;
    }

    // Concatenate submetric qualifier.
    if (sub_metric_name.empty() == false) {
        cuda_range_native_event_name += ":submetric=" + sub_metric_name;
    }

    // Concatenate device qualifier.
    if (device_id.empty() == false) {
        cuda_range_native_event_name += ":device=" + device_id;
    }

    CHECK_SNPRINTF_CALL( snprintf(native_event_name, len, "%s", cuda_range_native_event_name.c_str()),
                         int, len );

    return PAPI_OK;
}

/**
  * @brief Take a user provided cuda_range native event code and
  *        fill the member variables of the PAPI_event_info_t structure.
  *
  * @param *native_event_code
  *   A user provided cuda_range native event code.
  * @param *info
  *   A filled PAPI_event_info_t structure for the user provided cuda_range native
  *   event code.
*/
extern "C" int cuda_range_event_code_to_info(uint32_t native_event_code, PAPI_event_info_t *info)
{
    // Check for a user context on the calling CPU thread.
    CUcontext context = nullptr;
    CHECK_DRIVER_API_CALL( cuCtxGetCurrentPtr(&context) );
    if (context != nullptr) {
        // Pop the context from the calling CPU thread.
        CUcontext popped_context;
        CHECK_DRIVER_API_CALL( cuCtxPopCurrentPtr(&popped_context) );
    }

    native_event_info_t native_event_info;
    decode_evt_id(native_event_code, &native_event_info);

    std::string base_metric_name = cupti_metrics_keys[native_event_info.name_id];
    cupti_metric_info_t &cupti_metric_info = cupti_metrics[base_metric_name];
    int device_id_int = std::stoi(cupti_metric_info.device_ids[0]);

    // Description has not been stored for the CUPTI metric.
    if (cupti_metric_info.description.empty() == true) {
        // Class entry for device has yet to be instantiated.
        if (cupti_profile_metric_descriptions.count(device_id_int) == 0) {
            CUcontext context;
            unsigned int flags = 0;
            CHECK_DRIVER_API_CALL( cuCtxCreatePtr(&context, (CUctxCreateParams*)0, flags, device_id_int) );

            bool own_context = true;
            cupti_profile_metric_descriptions.emplace(device_id_int, CuptiProfile(context, own_context));
            CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).counter_availability() );
            CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).chip_name(device_id_int) );
        }

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).host_initialize() );

        std::string cupti_metric_name {};
        // CUPTI metric is of type Counter or Throughput.
        if (cupti_metric_info.cupti_metric_type == CUPTI_METRIC_TYPE_COUNTER ||
            cupti_metric_info.cupti_metric_type == CUPTI_METRIC_TYPE_THROUGHPUT) {
            cupti_metric_name = base_metric_name + "." + cupti_metric_info.stats[0] + "." + cupti_metric_info.submetrics[0];
        }
        // CUPTI metric is of type Ratio.
        else {
            cupti_metric_name = base_metric_name + "." + cupti_metric_info.submetrics[0];
        }
        cupti_profile_metric_descriptions.at(device_id_int).set_cupti_metric_names({cupti_metric_name}); 

        std::string description {};
        CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).metric_properties(base_metric_name.c_str(), description) );
        description += ".";

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).create_config_image() );

        size_t number_of_passes = 0;
        CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).number_of_passes(number_of_passes) );
        description += " The number of passes required for collection is " + std::to_string(number_of_passes) + ".";
        cupti_metric_info.description = description;

        // Avoid continously adding metrics to the current CUpti_Profiler_Host_Object by destroying it.
        CHECK_INTERNAL_FUNC_CALL( cupti_profile_metric_descriptions.at(device_id_int).host_deinitialize() ); 
    }

    int string_length;
    switch (native_event_info.flags) {
        case (NOQUAL_FLAG):
        {
            string_length = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s", base_metric_name.c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write metric name in case 0.\n");
                return PAPI_EBUF;
            }
            string_length = snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s", cupti_metric_info.description.c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write long description in case 0.\n");
                return PAPI_EBUF;
            }
            break;
        }
        case DEVICE_FLAG:
        {
            string_length = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:device=%s", base_metric_name.c_str(), cupti_metric_info.device_ids[0].c_str() );
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write metric name in case DEVICE_FLAG.\n");
                return PAPI_EBUF;
            }

            std::string devices_formatted;
            format_qualifiers_for_code_to_info(cupti_metric_info.device_ids, devices_formatted);

            string_length = snprintf(info->long_descr, PAPI_HUGE_STR_LEN, "masks:Mandatory device qualifier [%s]",
                              devices_formatted.c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write long description in case DEVICE_FLAG.\n");
                return PAPI_EBUF;
            }
            break;
        }
        case STAT_FLAG:
        {
            string_length = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:stat=%s", base_metric_name.c_str(), cupti_metric_info.stats[0].c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write metric name in case STAT_FLAG.\n");
                return PAPI_EBUF;
            }

            std::string stats_formatted;
            format_qualifiers_for_code_to_info(cupti_metric_info.stats, stats_formatted);

            string_length = snprintf(info->long_descr, PAPI_HUGE_STR_LEN, "masks:Mandatory stat qualifier [%s]",
                              stats_formatted.c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write long description in case STAT_FLAG.\n");
                return PAPI_EBUF;
            }
            break;
        }
        case SUBMETRIC_FLAG:
        {
            string_length = snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:submetric=%s", base_metric_name.c_str(), cupti_metric_info.submetrics[0].c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write metric name in case SUBMETRIC_FLAG.\n");
                return PAPI_EBUF;
            }

            std::string sub_metrics_formatted;
            format_qualifiers_for_code_to_info(cupti_metric_info.submetrics, sub_metrics_formatted);

            string_length = snprintf(info->long_descr, PAPI_HUGE_STR_LEN, "masks:Mandatory submetric qualifier [%s]",
                              sub_metrics_formatted.c_str());
            if (string_length < 0 || string_length >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write long description in case SUBMETRIC_FLAG.\n");
                return PAPI_EBUF;
            }
            break;
        }
        default:
            break;
    }

    // Push the user's context back onto the calling CPU thread.
    if (context != nullptr) {
        CHECK_DRIVER_API_CALL( cuCtxPushCurrentPtr(context) );
    }

    return PAPI_OK;
}

/**
  * @brief Collect the user added cuda_range native events and the user created CUDA contexts.
  *
  *        Note: If a user did not have a CUDA context on the calling CPU thread then
  *        one will be created based on # from :device=#.
  *
  * @param *event_codes
  *   User added cuda_range native event codes.
  * @param number_of_events
  *   Number of user added native events.
*/
extern "C" int cuda_range_store_added_native_events(uint32_t *event_codes, int number_of_events)
{
    // Collect the cuda_range native event data i.e. names and event order.
    std::map<int, added_native_events_data_t> cuda_range_native_event_data;
    for (size_t event_index = 0; event_index < static_cast<size_t>(number_of_events); event_index++) {
        std::string base_name, stat_name, sub_metric_name, device_id;
        CHECK_INTERNAL_FUNC_CALL( native_event_code_to_native_event_name(event_codes[event_index], base_name, stat_name, sub_metric_name, device_id) );

        std::string cupti_metric_name = base_name;
        // Concatenate stat (i.e. rollup in CUPTI terminology).
        if (stat_name.empty() == false) {
            cupti_metric_name += "." + stat_name;
        }    
        // Concatenate submetric.
        if (sub_metric_name.empty() == false) {
            cupti_metric_name += "." + sub_metric_name;
        }

        int device_id_int = std::stoi(device_id);
        cuda_range_native_event_data[device_id_int].cupti_metric_names.push_back(cupti_metric_name);
        cuda_range_native_event_data[device_id_int].event_insertion_order.push_back(event_index);
    }

    // Store the collected cuda_range native events for profiling.
    for (auto pair : cuda_range_native_event_data) {
        // No corresponding key entry for the current device.
        if (cupti_profile_per_device.count(pair.first) == 0) {
            bool own_context = false;
            CUdevice device;
            CUcontext context;
            CHECK_DRIVER_API_CALL( cuCtxGetCurrentPtr(&context) );
            // The user created a CUDA context on the calling CPU thread.
            if (context != nullptr) {
                SUBDBG("A user context (%p) was found on the calling CPU thread.\n", context);
                CHECK_DRIVER_API_CALL( cuCtxGetDevice_v2Ptr(&device, context) );
                if (device != pair.first) {
                    SUBDBG("The device index for the Cuda context on the calling CPU thread does not match"
                           " the device index assigned to the device qualifier.\n");
                    return PAPI_ECMP;
                }    
            }    
            // The user did not create a CUDA context on the calling CPU thread.
            else {
                SUBDBG("A user context was not found on the calling CPU thread. One will be created for device id %d\n", pair.first);
                CHECK_DRIVER_API_CALL( cuCtxCreatePtr(&context, (CUctxCreateParams*)0, 0, pair.first) );
                char *pop_context = getenv("PAPI_CUDA_RANGE_INTERNAL_VERIFY_CONTEXTS");
                if (pop_context == nullptr) {
                    CHECK_DRIVER_API_CALL( cuCtxPopCurrentPtr(&context) );
                }
                own_context = true;
            }

            cupti_profile_per_device.emplace(pair.first, CuptiProfile(context, own_context));
        }
        // Corresponding key entry for the current device found.
        else {
            SUBDBG("The device (%d) already has a cuda context stored.\n", pair.first);
        }

        cupti_profile_per_device.at(pair.first).set_cupti_metric_names(pair.second.cupti_metric_names);
        cupti_profile_per_device.at(pair.first).set_event_insertion_order(pair.second.event_insertion_order);
    }

    return PAPI_OK;
}

/**
  * @brief Start profiling.
  *
  *        Create and initialize the profiler host object (CUpti_Profiler_Host_Object),
  *        create a config image for the metrics added to the profiler host object,
  *        enable Range Profiling, set the config, and start profiling.
*/
extern "C" int cuda_range_start_profiling(void)
{
    int number_of_devices_on_system = 0;
    CHECK_RUNTIME_API_CALL( cudaGetDeviceCountPtr(&number_of_devices_on_system) );
    if (number_of_devices_on_system < 0) {
        SUBDBG("No NVIDIA devices detected on the machine.");
        return PAPI_ECMP;
    }

    for (int device_index = 0; device_index < number_of_devices_on_system; device_index++) {
        if (cupti_profile_per_device.count(device_index) == 0) {
            SUBDBG("No entry detected for device %d. Continuing.\n", device_index);
            continue;
        }
        SUBDBG("Entry detected for device %d. Proceeding to start profiling.\n", device_index);

        CHECK_INTERNAL_FUNC_CALL( cuda_range_reserve_device(device_index) );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).counter_availability() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).chip_name(device_index) );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).host_initialize() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).create_config_image() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).enable_range_profiler() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).create_counter_data_image() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).config() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).start_profiling() );

        std::string range_name = "PAPI_CUDA_RANGE_DEVICE_" + std::to_string(device_index);
        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).push_range(range_name) );
    }

    return PAPI_OK;
}

/**
  * @brief Read the counter values.
  *
  *        The profiling data stored in the hardware to the counter data image
  *        will be decoded and then evaluated based on the range index stored
  *        in the counter data.
  *        
  * @param ctx
  *   A structure containing the state, event ids, counters, and number
  *   of events.
  * @param **counter_values
  *   Stores the decoded and evaluated counter data.
*/
extern "C" int cuda_range_decode_and_evaluate_counter_data(cuda_range_ctx_t ctx, long long **counter_values)
{ 
    int number_of_devices_on_system = 0; 
    CHECK_RUNTIME_API_CALL( cudaGetDeviceCountPtr(&number_of_devices_on_system) );
    if (number_of_devices_on_system < 0) { 
        SUBDBG("No NVIDIA devices detected on the machine.");
        return PAPI_ECMP;
    }    

    bool all_passes_submitted_for_devices = true;
    for (int device_index = 0; device_index < number_of_devices_on_system; device_index++) {
        // No user added events for the device.
        if (cupti_profile_per_device.count(device_index) == 0) { 
            SUBDBG("No entry detected for device %d. Continuing.\n", device_index);
            continue;
        }
        // Avoid continous overwrites of initial all passes submitted collection.
        // Example: A user is on a multiple device system and creates a PAPI event set. The PAPI event set
        // contains a cuda_range native event on device 0 which requires a single pass AND a cuda_range native event on
        // device 1 which requires 4 passes. In this case a user will need to call PAPI_read 4 times to get the proper value
        // for the device 1 cuda_range native event. In doing so, the device 0 cuda_range native event will be contiously
        // overwrote; therefore, a user can avoid overwrites by setting the environemnt variable 
        // PAPI_CUDA_RANGE_CONTINUE_ON_ALL_PASSES.
        if (getenv("PAPI_CUDA_RANGE_CONTINUE_ON_ALL_PASSES") &&
            cupti_profile_per_device.at(device_index).get_all_passes_submitted()) {
            SUBDBG("PAPI_CUDA_RANGE_CONTINUE_ON_ALL_PASSES set; therefore, continuing for device %d as all passes are submitted.\n");
            continue;
        }
        SUBDBG("Entry detected for device %d. Proceeding to decode and evaluate to gpu values.\n", device_index);

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).pop_range() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).stop_profiling() );

        // All passes have been submitted to the device for collection.
        if (cupti_profile_per_device.at(device_index).get_all_passes_submitted()) {
            CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).decode_data() );

            CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).evaluate_counter_data() );

            cupti_profile_per_device.at(device_index).calculate(ctx);

            CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).create_counter_data_image() );
        }
        // All passes have not been submitted to the device for collection.
        else {
            all_passes_submitted_for_devices = false;
        }

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).start_profiling() );

        std::string range_name = "PAPI_CUDA_RANGE_DEVICE_" + std::to_string(device_index);
        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).push_range(range_name) );
    }
    *counter_values = ctx->counters;

    return all_passes_submitted_for_devices == true ?  PAPI_OK : PAPI_EALLPASSES_NOT_SUBMITTED;
}

/**
  * @brief Stop profiling.
  *
  *        CUPTI Range Profiling will be stopped and disabled.
  *        Along with the CUpti_Profiler_Host_Object being 
  *        deinitialized and destroyed.
*/
extern "C" int cuda_range_stop_profiling(void)
{
    int number_of_devices_on_system = 0; 
    CHECK_RUNTIME_API_CALL( cudaGetDeviceCountPtr(&number_of_devices_on_system) );
    if (number_of_devices_on_system < 0) { 
        SUBDBG("No NVIDIA devices detected on the machine.");
        return PAPI_ECMP;
    } 

    for (int device_index = 0; device_index < number_of_devices_on_system; device_index++) {
        if (cupti_profile_per_device.count(device_index) == 0) {
            SUBDBG("No entry detected for device %d. Continuing.\n", device_index);
            continue;
        }
        SUBDBG("Entry detected for device %d. Proceeding to stop profiling.\n", device_index);

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).stop_profiling() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).disable_range_profiler() );

        CHECK_INTERNAL_FUNC_CALL( cupti_profile_per_device.at(device_index).host_deinitialize() );

        CHECK_INTERNAL_FUNC_CALL( cuda_range_unreserve_device(device_index) );
    }

    return PAPI_OK;
}

/**
  * @brief Reset the counter values to zero.
  *
  * @param ctx
  *   A structure containing the state, event ids, counters, and number
  *   of events. 
*/
extern "C" int cuda_range_reset_counters(cuda_range_ctx_t ctx)
{
    memset(ctx->counters, 0, sizeof(ctx->counters) * ctx->num_events);

    return PAPI_OK;
}

/**
  * @brief Shutdown the cuda_range component.
  *
  *        Unload, CUPTI CUDA driver, and CUDA runtime
  *        function pointers.
*/
extern "C" int cuda_range_unload_function_pointers_and_shutdown(void)
{
    // Destroy OWNED CUDA contexts for cupti_profile_per_device.
    // See member function documentation for more details.
    for (auto& pair : cupti_profile_per_device)
        pair.second.destroy_context();

    // Destroy OWNED CUDA contexts for cupti_profile_metric_descriptions.
    // See member function documentation for more details.
    for (auto& pair : cupti_profile_metric_descriptions)
        pair.second.destroy_context();

    unload_cupti_function_pointers();

    unload_cuda_runtime_function_pointers();

    unload_cuda_driver_function_pointers();

    return PAPI_OK;
}

/**
  * @brief Return the cuda_range error message set by
  *        cuda_range_set_last_err_msg.
*/
extern "C" const char *cuda_range_get_last_err_msg(void)
{
    return cuda_range_error_message.c_str();
}

/**
 *  @}
 ******************************************************************************/
