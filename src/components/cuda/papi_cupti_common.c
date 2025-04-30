/**
 * @file    papi_cupti_common.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include <dlfcn.h>
#include <link.h>
#include <libgen.h>
#include <dirent.h>
#include <papi.h>
#include "papi_memory.h"

#include "cupti_config.h"
#include "papi_cupti_common.h"

static void *dl_drv, *dl_rt;

void *dl_cupti;

unsigned int _cuda_lock;

/* cuda driver function pointers */
CUresult ( *cuCtxGetCurrentPtr ) (CUcontext *);
CUresult ( *cuCtxSetCurrentPtr ) (CUcontext);
CUresult ( *cuCtxDestroyPtr ) (CUcontext);
CUresult ( *cuCtxCreatePtr ) (CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult ( *cuCtxGetDevicePtr ) (CUdevice *);
CUresult ( *cuDeviceGetPtr ) (CUdevice *, int);
CUresult ( *cuDeviceGetCountPtr ) (int *);
CUresult ( *cuDeviceGetNamePtr ) (char *, int, CUdevice);
CUresult ( *cuDevicePrimaryCtxRetainPtr ) (CUcontext *pctx, CUdevice);
CUresult ( *cuDevicePrimaryCtxReleasePtr ) (CUdevice);
CUresult ( *cuInitPtr ) (unsigned int);
CUresult ( *cuGetErrorStringPtr ) (CUresult error, const char** pStr);
CUresult ( *cuCtxPopCurrentPtr ) (CUcontext * pctx);
CUresult ( *cuCtxPushCurrentPtr ) (CUcontext pctx);
CUresult ( *cuCtxSynchronizePtr ) ();
CUresult ( *cuDeviceGetAttributePtr ) (int *, CUdevice_attribute, CUdevice);

/* cuda runtime function pointers */
cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
cudaError_t ( *cudaGetDevicePtr ) (int *);
const char *( *cudaGetErrorStringPtr ) (cudaError_t);
cudaError_t ( *cudaSetDevicePtr ) (int);
cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
cudaError_t ( *cudaFreePtr ) (void *);
cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

/* cupti function pointer */
CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );

/**@class load_cuda_sym
 * @brief Search for a variation of the shared object libcuda.
 */
static int load_cuda_sym(void)
{
    int soNamesToSearchCount = 3;
    const char *soNamesToSearchFor[] = {"libcuda.so", "libcuda.so.1", "libcuda"};

    dl_drv = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
    if (!dl_drv) {
        ERRDBG("Loading installed libcuda.so failed. Check that cuda drivers are installed.\n");
        goto fn_fail;
    }

    cuCtxSetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxSetCurrent");
    cuCtxGetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxGetCurrent");
    cuCtxDestroyPtr              = DLSYM_AND_CHECK(dl_drv, "cuCtxDestroy");
    cuCtxCreatePtr               = DLSYM_AND_CHECK(dl_drv, "cuCtxCreate");
    cuCtxGetDevicePtr            = DLSYM_AND_CHECK(dl_drv, "cuCtxGetDevice");
    cuDeviceGetPtr               = DLSYM_AND_CHECK(dl_drv, "cuDeviceGet");
    cuDeviceGetCountPtr          = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetCount");
    cuDeviceGetNamePtr           = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr  = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRelease");
    cuInitPtr                    = DLSYM_AND_CHECK(dl_drv, "cuInit");
    cuGetErrorStringPtr          = DLSYM_AND_CHECK(dl_drv, "cuGetErrorString");
    cuCtxPopCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxPushCurrent");
    cuCtxSynchronizePtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxSynchronize");
    cuDeviceGetAttributePtr      = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetAttribute");

    Dl_info info;
    dladdr(cuCtxSetCurrentPtr, &info);
    LOGDBG("CUDA driver library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cuda_sym(void)
{
    if (dl_drv) {
        dlclose(dl_drv);
        dl_drv = NULL;
    }
    cuCtxSetCurrentPtr           = NULL;
    cuCtxGetCurrentPtr           = NULL;
    cuCtxDestroyPtr              = NULL;
    cuCtxCreatePtr               = NULL;
    cuCtxGetDevicePtr            = NULL;
    cuDeviceGetPtr               = NULL;
    cuDeviceGetCountPtr          = NULL;
    cuDeviceGetNamePtr           = NULL;
    cuDevicePrimaryCtxRetainPtr  = NULL;
    cuDevicePrimaryCtxReleasePtr = NULL;
    cuInitPtr                    = NULL;
    cuGetErrorStringPtr          = NULL;
    cuCtxPopCurrentPtr           = NULL;
    cuCtxPushCurrentPtr          = NULL;
    cuCtxSynchronizePtr          = NULL;
    cuDeviceGetAttributePtr      = NULL;
    return PAPI_OK;
}

/**@class search_and_load_shared_objects
 * @brief Search and load Cuda shared objects.
 *
 * @param *parentPath
 *   The main path we will use to search for the shared objects. 
 * @param *soMainName
 *   The name of the shared object e.g. libcudart. This is used
 *   to select the standardSubPaths to use.
 * @param *soNamesToSearchFor[]
 *   Varying names of the shared object we want to search for.
 * @param soNamesToSearchCount
 *   Total number of names in soNamesToSearchFor.
 */
void *search_and_load_shared_objects(const char *parentPath, const char *soMainName, const char *soNamesToSearchFor[], int soNamesToSearchCount)
{
    const char *standardSubPaths[3];
    // Case for when we want to search explicit subpaths for a shared object
    if (soMainName != NULL) {
        if (strcmp(soMainName, "libcudart") == 0) {
            standardSubPaths[0] = "%s/lib64/";
            standardSubPaths[1] = NULL;
        }
        else if (strcmp(soMainName, "libcupti") == 0) {
            standardSubPaths[0] = "%s/extras/CUPTI/lib64/";
            standardSubPaths[1] = "%s/lib64/";
            standardSubPaths[2] = NULL;
        }
        else if (strcmp(soMainName, "libnvperf_host") == 0) {
            standardSubPaths[0] = "%s/extras/CUPTI/lib64/";
            standardSubPaths[1] = "%s/lib64/";
            standardSubPaths[2] = NULL;
        }
    }
    // Case for when a user provides an exact path e.g. PAPI_CUDA_RUNTIME
    // and we do not want to search subpaths
    else{
        standardSubPaths[0] = "%s/";
        standardSubPaths[1] = NULL;     
    }

    char pathToSharedLibrary[PAPI_HUGE_STR_LEN], directoryPathToSearch[PAPI_HUGE_STR_LEN];
    void *so = NULL;
    char *soNameFound;
    int i, strLen;
    for (i = 0; standardSubPaths[i] != NULL; i++) {
        // Create path to search for dl names
        int strLen = snprintf(directoryPathToSearch, PAPI_HUGE_STR_LEN, standardSubPaths[i], parentPath);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            ERRDBG("Failed to fully write path to search for dlnames.\n");
            return NULL;
        }   

        DIR *dir = opendir(directoryPathToSearch);
        if (dir == NULL) {
            ERRDBG("Directory path could not be opened.\n");
            continue;
        }

        int j;
        for (j = 0; j < soNamesToSearchCount; j++) {
            struct dirent *dirEntry;
            while( ( dirEntry = readdir(dir) ) != NULL ) {
                int result;
                char *p = strstr(soNamesToSearchFor[j], "so");
                // Check for an exact match of a shared object name (.so and .so.1 case)
                if (p) {
                    result = strcmp(dirEntry->d_name, soNamesToSearchFor[j]);
                }
                // Check for any match of a shared object name (we could not find .so and .so.1)
                else {
                    result = strncmp(dirEntry->d_name, soNamesToSearchFor[j], strlen(soNamesToSearchFor[j]));
                }

                if (result == 0) {
                    soNameFound = dirEntry->d_name;
                    goto found;
                }
            }
            // Reset the position of the directory stream
            rewinddir(dir);
        }
    }

  exit:
    return so;
  found:
    // Construct path to shared library
    strLen = snprintf(pathToSharedLibrary, PAPI_HUGE_STR_LEN, "%s%s", directoryPathToSearch, soNameFound);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        ERRDBG("Failed to fully write constructed path to shared library.\n");
        return NULL;
    }
    so = dlopen(pathToSharedLibrary, RTLD_NOW | RTLD_GLOBAL);
   
    goto exit; 
}

/**@class search_and_load_from_system_paths
 * @brief A simple wrapper to try and search and load
 *        Cuda shared objects from system paths.
 *
 * @param *soNamesToSearchFor[]
 *   Varying names of the shared object we want to search for.
 * @param soNamesToSearchCount
 *   Total number of names in soNamesToSearchFor.
 */
void *search_and_load_from_system_paths(const char *soNamesToSearchFor[], int soNamesToSearchCount)
{
    void *so = NULL;
    int i;
    for (i = 0; i < soNamesToSearchCount; i++) {
        so = dlopen(soNamesToSearchFor[i], RTLD_NOW | RTLD_GLOBAL);
        if (so) {
            return so;
        }   
    }

    return so; 
}

/**@class load_cudart_sym
 * @brief Search for a variation of the shared object libcudart.
 *        Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_RUNTIME, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect a variation of the shared object libcudart from PAPI_CUDA_RUNTIME or it is not set,
 *    we will search the path defined with PAPI_CUDA_ROOT; as this is supposed to always be set.
 * 3. If we fail to collect a variation of the shared object libcudart from steps 1 and 2, then we will search the linux
 *    default directories listed by /etc/ld.so.conf. As a note, updating the LD_LIBRARY_PATH is
 *    advised for this option.
 * 4. We use dlopen to search for a variation of the shared object libcudart.
 *    If this fails, then we failed to find a variation of the shared object
 *    libcudart.
 */
static int load_cudart_sym(void)
{
    int soNamesToSearchCount = 3;
    const char *soNamesToSearchFor[] = {"libcudart.so", "libcudart.so.1", "libcudart"};

    // If a user set PAPI_CUDA_RUNTIME with a path, then search it for the shared object (takes precedent over PAPI_CUDA_ROOT)
    char *papi_cuda_runtime = getenv("PAPI_CUDA_RUNTIME");
    if (papi_cuda_runtime) {
        dl_rt = search_and_load_shared_objects(papi_cuda_runtime, NULL, soNamesToSearchFor, soNamesToSearchCount);
    }

    char *soMainName = "libcudart";
    // If a user set PAPI_CUDA_ROOT with a path and we did not already find the shared object, then search it for the shared object
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_rt) {
        dl_rt = search_and_load_shared_objects(papi_cuda_root, soMainName, soNamesToSearchFor, soNamesToSearchCount);
    }

    // Last ditch effort to find a variation of libcudart, see dlopen manpages for how search occurs
    if (!dl_rt) {
        dl_rt = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
        if (!dl_rt) {
            ERRDBG("Loading libcudart shared library failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cudaGetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaGetDevice");
    cudaGetDeviceCountPtr      = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceCount");
    cudaGetDevicePropertiesPtr = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceProperties");
    cudaGetErrorStringPtr      = DLSYM_AND_CHECK(dl_rt, "cudaGetErrorString");
    cudaDeviceGetAttributePtr  = DLSYM_AND_CHECK(dl_rt, "cudaDeviceGetAttribute");
    cudaSetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaSetDevice");
    cudaFreePtr                = DLSYM_AND_CHECK(dl_rt, "cudaFree");
    cudaDriverGetVersionPtr    = DLSYM_AND_CHECK(dl_rt, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr   = DLSYM_AND_CHECK(dl_rt, "cudaRuntimeGetVersion");

    Dl_info info;
    dladdr(cudaGetDevicePtr, &info);
    LOGDBG("CUDA runtime library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cudart_sym(void)
{
    if (dl_rt) {
        dlclose(dl_rt);
        dl_rt = NULL;
    }
    cudaGetDevicePtr           = NULL;
    cudaGetDeviceCountPtr      = NULL;
    cudaGetDevicePropertiesPtr = NULL;
    cudaGetErrorStringPtr      = NULL;
    cudaDeviceGetAttributePtr  = NULL;
    cudaSetDevicePtr           = NULL;
    cudaFreePtr                = NULL;
    cudaDriverGetVersionPtr    = NULL;
    cudaRuntimeGetVersionPtr   = NULL;
    return PAPI_OK;
}

/**@class load_cupti_common_sym
 * @brief Search for a variation of the shared object libcupti.
 *        Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_CUPTI, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect a variation of the shared object libcupti from PAPI_CUDA_CUPTI or it is not set,
 *    we will search the path defined with PAPI_CUDA_ROOT; as this is supposed to always be set.
 * 3. If we fail to collect a variation of the shared object libcupti from steps 1 and 2, then we will search the linux
 *    default directories listed by /etc/ld.so.conf. As a note, updating the LD_LIBRARY_PATH is
 *    advised for this option.
 * 4. We use dlopen to search for a variation of the shared object libcupti.
 *    If this fails, then we failed to find a variation of the shared object
 *    libcupti.
 */
static int load_cupti_common_sym(void)
{
    int soNamesToSearchCount = 3;
    const char  *soNamesToSearchFor[] = {"libcupti.so", "libcupti.so.1", "libcupti"};

    // If a user set PAPI_CUDA_CUPTI with a path, then search it for the shared object (takes precedent over PAPI_CUDA_ROOT)
    char *papi_cuda_cupti = getenv("PAPI_CUDA_CUPTI");
    if (papi_cuda_cupti) {
        dl_cupti = search_and_load_shared_objects(papi_cuda_cupti, NULL, soNamesToSearchFor, soNamesToSearchCount);
    }

    char *soMainName = "libcupti";
    // If a user set PAPI_CUDA_ROOT with a path and we did not already find the shared object, then search it for the shared object
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_cupti) {
        dl_cupti = search_and_load_shared_objects(papi_cuda_root, soMainName, soNamesToSearchFor, soNamesToSearchCount);
    }

    // Last ditch effort to find a variation of libcupti, see dlopen manpages for how search occurs
    if (!dl_cupti) {
        dl_cupti = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
        if (!dl_cupti) {
            ERRDBG("Loading libcupti.so failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiGetVersion");

    Dl_info info;
    dladdr(cuptiGetVersionPtr, &info);
    LOGDBG("CUPTI library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cupti_common_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiGetVersionPtr = NULL;
    return PAPI_OK;
}

static int util_load_cuda_sym(void)
{
    int papi_errno;
    papi_errno = load_cuda_sym();
    papi_errno += load_cudart_sym();
    papi_errno += load_cupti_common_sym();
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }
    else
        return PAPI_OK;
}

int cuptic_shutdown(void)
{
    unload_cuda_sym();
    unload_cudart_sym();
    unload_cupti_common_sym();
    return PAPI_OK;
}

static int util_dylib_cu_runtime_version(void)
{
    int runtimeVersion;
    cudaArtCheckErrors(cudaRuntimeGetVersionPtr(&runtimeVersion), return PAPI_EMISC );
    return runtimeVersion;
}

static int util_dylib_cupti_version(void)
{
    unsigned int cuptiVersion;
    cuptiCheckErrors(cuptiGetVersionPtr(&cuptiVersion), return PAPI_EMISC );
    return cuptiVersion;
}


/** @class cuptic_device_get_count
  * @brief Get total number of gpus on the machine that are compute
  *        capable..
  * @param *num_gpus 
  *    Collect the total number of gpus.
*/
int cuptic_device_get_count(int *num_gpus)
{
    cudaError_t cuda_err;

    /* find the total number of compute-capable devices */
    cuda_err = cudaGetDeviceCountPtr(num_gpus);
    if (cuda_err != cudaSuccess) {
        cuptic_disabled_reason_set(cudaGetErrorStringPtr(cuda_err));
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int get_gpu_compute_capability(int dev_num, int *cc)
{
    int cc_major, cc_minor;
    cudaError_t cuda_errno;
    cuda_errno = cudaDeviceGetAttributePtr(&cc_major, cudaDevAttrComputeCapabilityMajor, dev_num);
    if (cuda_errno != cudaSuccess) {
        cuptic_disabled_reason_set(cudaGetErrorStringPtr(cuda_errno));
        return PAPI_EMISC;
    }
    cuda_errno = cudaDeviceGetAttributePtr(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev_num);
    if (cuda_errno != cudaSuccess) {
        cuptic_disabled_reason_set(cudaGetErrorStringPtr(cuda_errno));
        return PAPI_EMISC;
    }
    *cc = cc_major * 10 + cc_minor;
    return PAPI_OK;
}

typedef enum {GPU_COLLECTION_UNKNOWN, GPU_COLLECTION_ALL_PERF, GPU_COLLECTION_MIXED, GPU_COLLECTION_ALL_EVENTS, GPU_COLLECTION_ALL_CC70} gpu_collection_e;

static int util_gpu_collection_kind(gpu_collection_e *coll_kind)
{
    int papi_errno = PAPI_OK;
    static gpu_collection_e kind = GPU_COLLECTION_UNKNOWN;
    if (kind != GPU_COLLECTION_UNKNOWN) {
        goto fn_exit;
    }

    int total_gpus;
    papi_errno = cuptic_device_get_count(&total_gpus);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    int i, cc;
    int count_perf = 0, count_evt = 0, count_cc70 = 0;
    for (i=0; i<total_gpus; i++) {
        papi_errno = get_gpu_compute_capability(i, &cc);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        if (cc == 70) {
            ++count_cc70;
        }
        if (cc >= 70) {
            ++count_perf;
        }
        if (cc <= 70) {
            ++count_evt;
        }
    }
    if (count_cc70 == total_gpus) {
        kind = GPU_COLLECTION_ALL_CC70;
        goto fn_exit;
    }
    if (count_perf == total_gpus) {
        kind = GPU_COLLECTION_ALL_PERF;
        goto fn_exit;
    }
    if (count_evt == total_gpus) {
        kind = GPU_COLLECTION_ALL_EVENTS;
        goto fn_exit;
    }
    kind = GPU_COLLECTION_MIXED;

fn_exit:
    *coll_kind = kind;
    return papi_errno;
}

const char *cuptic_disabled_reason_g;

/** @class cuptic_disabled_reason_set
  * @brief Updating the current Cuda context.
  * @param *msg
  *    Cuda error message.
*/
void cuptic_disabled_reason_set(const char *msg)
{
    cuptic_disabled_reason_g = msg;
}

void cuptic_disabled_reason_get(const char **pmsg)
{
    *pmsg = cuptic_disabled_reason_g;
}

int cuptic_init(void)
{
    int papi_errno = util_load_cuda_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to load CUDA library functions.");
        goto fn_exit;
    }

    gpu_collection_e kind;
    papi_errno = util_gpu_collection_kind(&kind);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
 
    if (kind == GPU_COLLECTION_MIXED) {
        cuptic_disabled_reason_set("No support for systems with mixed compute capabilities, such as CC < 7.0 and CC > 7.0 GPUS.");
        papi_errno = PAPI_ECMP;
        goto fn_exit;
    }
fn_exit:
    return papi_errno;
}

int cuptic_is_runtime_perfworks_api(void)
{
    static int is_perfworks_api = -1;
    if (is_perfworks_api != -1) {
        goto fn_exit;
    }
    char *papi_cuda_110_cc70_perfworks_api = getenv("PAPI_CUDA_110_CC_70_PERFWORKS_API");

    gpu_collection_e gpus_kind;
    int papi_errno = util_gpu_collection_kind(&gpus_kind);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    unsigned int cuptiVersion = util_dylib_cupti_version();

    if (gpus_kind == GPU_COLLECTION_ALL_CC70 && 
        (cuptiVersion == CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION || util_dylib_cu_runtime_version() == 11000))
    {
        if (papi_cuda_110_cc70_perfworks_api != NULL) {
            is_perfworks_api = 1;
            goto fn_exit;
        }
        else {
            is_perfworks_api = 0;
            goto fn_exit;
        }
    }

    if ((gpus_kind == GPU_COLLECTION_ALL_PERF || gpus_kind == GPU_COLLECTION_ALL_CC70) && cuptiVersion >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION) {
        is_perfworks_api = 1;
        goto fn_exit;
    } else {
        is_perfworks_api = 0;
        goto fn_exit;
    }

fn_exit:
    return is_perfworks_api;
}

int cuptic_is_runtime_events_api(void)
{
    static int is_events_api = -1;
    if (is_events_api != -1) {
        goto fn_exit;
    }

    gpu_collection_e gpus_kind;
    int papi_errno = util_gpu_collection_kind(&gpus_kind);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /*
     * See cupti_config.h: When NVIDIA removes the events API add a check in the following condition
     * to check the `util_dylib_cupti_version()` is also <= CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION.
     */
    if ((gpus_kind == GPU_COLLECTION_ALL_EVENTS || gpus_kind == GPU_COLLECTION_ALL_CC70)) {
        is_events_api = 1;
        goto fn_exit;
    } else {
        is_events_api = 0;
        goto fn_exit;
    }
fn_exit:
    return is_events_api;
}

struct cuptic_info {
    CUcontext ctx;
};


/** @class cuptic_ctxarr_create
  * @brief Allocate memory for pinfo.
  * @param *pinfo
  *    Instance of a struct that holds read count, running, cuptic_t
  *    and cuptip_gpu_state_t.
*/
int cuptic_ctxarr_create(cuptic_info_t *pinfo)
{
    COMPDBG("Entering.\n");
    int total_gpus, papi_errno;

    /* retrieve total number of compute-capable devices */
    papi_errno = cuptic_device_get_count(&total_gpus);
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }
  
    /* allocate memory */ 
    *pinfo = (cuptic_info_t) papi_calloc (total_gpus, sizeof(*pinfo));
    if (*pinfo == NULL) {
        return PAPI_ENOMEM;
    }

    return PAPI_OK;
}


/** @class cuptic_ctxarr_update_current
  * @brief Updating the current Cuda context.
  * @param info
  *    Struct that contains a Cuda context, that can be indexed into based
  *    on device id.
  * @param evt_dev_id
  *    Device id from an appended device qualifier (e.g. :device=#).
*/
int cuptic_ctxarr_update_current(cuptic_info_t info, int evt_dev_id)
{
    CUcontext pctx;
    CUresult cuda_err;
    CUdevice dev_id;

    // If a Cuda context already exists, get it
    cuda_err = cuCtxGetCurrentPtr(&pctx);
    if (cuda_err != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }

    // Get the Device ID for the existing Cuda context
    if (pctx != NULL) {
        cuda_err = cuCtxGetDevicePtr(&dev_id);
        if (cuda_err != CUDA_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    // A context is not stored for the :device=# qualifier
    if (info[evt_dev_id].ctx == NULL) {
        // Cuda context was not found or a user did not provide an appropriate Cuda context for the
        // device qualifier id that was supplied
        if (pctx == NULL || dev_id != evt_dev_id) {
            // If multiple devices are found on the machine, then we need to call cudaSetDevice
            SUBDBG("A Cuda context was not found. Therefore, one is created for device: %d\n", evt_dev_id);
            cudaArtCheckErrors(cudaSetDevicePtr(evt_dev_id), return PAPI_EMISC);
            cudaArtCheckErrors(cudaFreePtr(0), return PAPI_EMISC);

            cudaCheckErrors(cuCtxGetCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
            cudaCheckErrors(cuCtxPopCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
        }
        // Cuda context was found
        else {
            SUBDBG("A cuda context was found for device: %d\n", evt_dev_id);
            cudaCheckErrors(cuCtxGetCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
        }
    }
    // If the Cuda context has changed for a device keep the first one seen, but output a warning
    else if (pctx != NULL){
        if (evt_dev_id == dev_id) {
            if (info[dev_id].ctx != pctx) {
                ERRDBG("Warning: cuda context for device %d has changed from %p to %p\n", dev_id, info[dev_id].ctx, pctx);
            }
        }
    }

    return PAPI_OK;
}

int cuptic_ctxarr_get_ctx(cuptic_info_t info, int gpu_idx, CUcontext *ctx)
{
    *ctx = info[gpu_idx].ctx;
    return PAPI_OK;
}

int cuptic_ctxarr_destroy(cuptic_info_t *pinfo)
{
    papi_free(*pinfo);
    *pinfo = NULL;
    return PAPI_OK;
}

/* Functions based on bitmasking to detect gpu exclusivity */
typedef int64_t gpu_occupancy_t;
static gpu_occupancy_t global_gpu_bitmask;

static int _devmask_events_get(cuptiu_event_table_t *evt_table, gpu_occupancy_t *bitmask)
{
    gpu_occupancy_t acq_mask = 0;
    long i;
    for (i = 0; i < evt_table->count; i++) {
        acq_mask |= (1 << evt_table->cuda_devs[i]);
    }
    *bitmask = acq_mask;

    return PAPI_OK;
}

int cuptic_device_acquire(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if (bitmask & global_gpu_bitmask) {
        return PAPI_ECNFLCT;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask |= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}

int cuptic_device_release(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if ((bitmask & global_gpu_bitmask) != bitmask) {
        return PAPI_EMISC;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask ^= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}

/** @class cuptiu_dev_set
  * @brief For a Cuda native event, set the device ID.
  *
  * @param *bitmap
  *   Device map.
  * @param i
  *   Device ID.
*/
int cuptiu_dev_set(cuptiu_bitmap_t *bitmap, int i)
{
    *bitmap |= (1ULL << i);
    return PAPI_OK;
}

/** @class cuptiu_dev_check
  * @brief For a Cuda native event, check for a valid device ID.
  *
  * @param *bitmap
  *   Device map.
  * @param i
  *   Device ID.
*/
int cuptiu_dev_check(cuptiu_bitmap_t bitmap, int i)
{
    return (bitmap & (1ULL << i));
}
