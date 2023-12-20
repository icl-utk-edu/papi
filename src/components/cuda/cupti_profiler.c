/**
 * @file    cupti_profiler.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <dlfcn.h>
#include <papi.h>
#include "papi_memory.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include "cupti_common.h"
#include "cupti_profiler.h"
#include "lcuda_debug.h"

typedef struct byte_array_s         byte_array_t;
typedef struct cuptip_gpu_state_s   cuptip_gpu_state_t;
typedef struct list_metrics_s       list_metrics_t;
typedef struct NVPA_MetricsContext  NVPA_MetricsContext;
typedef NVPW_CUDA_MetricsContext_Create_Params MCCP_t;
enum running_e           {False, True};
enum collection_method_e {SpotValue, RunningMin, RunningMax, RunningSum};

static void *dl_nvpw;
static int num_gpus;
static list_metrics_t *avail_events;

static int load_cupti_perf_sym(void);
static int unload_cupti_perf_sym(void);
static int load_nvpw_sym(void);
static int unload_nvpw_sym(void);
static int initialize_cupti_profiler_api(void);
static int finalize_cupti_profiler_api(void);
static int initialize_perfworks_api(void);
static int get_chip_name(int dev_num, char* chipName);
static int init_all_metrics(void);
static int find_same_chipname(int gpu_id);
static void free_all_enumerated_metrics(void);
static int event_name_tokenize(const char *name, char *nv_name, int *gpuid);
static int retrieve_metric_details(NVPA_MetricsContext *pMetricsContext, const char *nv_name,
                                   char *description, int *numDep, NVPA_RawMetricRequest **pRMR);
static int check_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count,
                            NVPA_RawMetricRequest *rmr, int *num_pass);
static enum collection_method_e get_event_collection_method(const char *evt_name);

static int nvpw_cuda_metricscontext_create(cuptip_control_t state);
static int nvpw_cuda_metricscontext_destroy(cuptip_control_t state);
static int add_events_per_gpu(cuptip_control_t state, cuptiu_event_table_t *event_names);
static int control_state_validate(cuptip_control_t state);

static int get_event_names_rmr(cuptip_gpu_state_t *gpu_ctl);
static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl);
static int metric_get_config_image(cuptip_gpu_state_t *gpu_ctl);
static int metric_get_counter_data_prefix_image(cuptip_gpu_state_t *gpu_ctl);
static int create_counter_data_image(cuptip_gpu_state_t *gpu_ctl);
static int reset_cupti_prof_config_images(cuptip_gpu_state_t *gpu_ctl);
static int begin_profiling(cuptip_gpu_state_t *gpu_ctl);
static int end_profiling(cuptip_gpu_state_t *gpu_ctl);
static int get_measured_values(cuptip_gpu_state_t *gpu_ctl);

NVPA_Status ( *NVPW_GetSupportedChipNamesPtr ) (NVPW_GetSupportedChipNames_Params* params);
NVPA_Status ( *NVPW_CUDA_MetricsContext_CreatePtr ) (NVPW_CUDA_MetricsContext_Create_Params* params);
NVPA_Status ( *NVPW_MetricsContext_DestroyPtr ) (NVPW_MetricsContext_Destroy_Params * params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_BeginPtr ) (NVPW_MetricsContext_GetMetricNames_Begin_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_EndPtr ) (NVPW_MetricsContext_GetMetricNames_End_Params* params);
NVPA_Status ( *NVPW_InitializeHostPtr ) (NVPW_InitializeHost_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_BeginPtr ) (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_EndPtr ) (NVPW_MetricsContext_GetMetricProperties_End_Params* p);
NVPA_Status ( *NVPW_CUDA_RawMetricsConfig_CreatePtr ) (NVPW_CUDA_RawMetricsConfig_Create_Params*);

NVPA_Status ( *NVPW_RawMetricsConfig_DestroyPtr ) (NVPW_RawMetricsConfig_Destroy_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_BeginPassGroupPtr ) (NVPW_RawMetricsConfig_BeginPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_EndPassGroupPtr ) (NVPW_RawMetricsConfig_EndPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_AddMetricsPtr ) (NVPW_RawMetricsConfig_AddMetrics_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GenerateConfigImagePtr ) (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetConfigImagePtr ) (NVPW_RawMetricsConfig_GetConfigImage_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_CreatePtr ) (NVPW_CounterDataBuilder_Create_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_DestroyPtr ) (NVPW_CounterDataBuilder_Destroy_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_AddMetricsPtr ) (NVPW_CounterDataBuilder_AddMetrics_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_GetCounterDataPrefixPtr ) (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params);
NVPA_Status ( *NVPW_CounterData_GetNumRangesPtr ) (NVPW_CounterData_GetNumRanges_Params* params);
NVPA_Status ( *NVPW_Profiler_CounterData_GetRangeDescriptionsPtr ) (NVPW_Profiler_CounterData_GetRangeDescriptions_Params* params);
NVPA_Status ( *NVPW_MetricsContext_SetCounterDataPtr ) (NVPW_MetricsContext_SetCounterData_Params* params);
NVPA_Status ( *NVPW_MetricsContext_EvaluateToGpuValuesPtr ) (NVPW_MetricsContext_EvaluateToGpuValues_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetNumPassesPtr ) (NVPW_RawMetricsConfig_GetNumPasses_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_SetCounterAvailabilityPtr ) (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr ) (NVPW_RawMetricsConfig_IsAddMetricsPossible_Params* params);

NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_BeginPtr ) (NVPW_MetricsContext_GetCounterNames_Begin_Params* pParams);
NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_EndPtr ) (NVPW_MetricsContext_GetCounterNames_End_Params* pParams);

CUptiResult ( *cuptiDeviceGetChipNamePtr ) (CUpti_Device_GetChipName_Params* params);
CUptiResult ( *cuptiProfilerInitializePtr ) (CUpti_Profiler_Initialize_Params* params);
CUptiResult ( *cuptiProfilerDeInitializePtr ) (CUpti_Profiler_DeInitialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializePtr ) (CUpti_Profiler_CounterDataImage_Initialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializeScratchBufferPtr ) (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params);
CUptiResult ( *cuptiProfilerBeginSessionPtr ) (CUpti_Profiler_BeginSession_Params* params);
CUptiResult ( *cuptiProfilerSetConfigPtr ) (CUpti_Profiler_SetConfig_Params* params);
CUptiResult ( *cuptiProfilerBeginPassPtr ) (CUpti_Profiler_BeginPass_Params* params);
CUptiResult ( *cuptiProfilerEnableProfilingPtr ) (CUpti_Profiler_EnableProfiling_Params* params);
CUptiResult ( *cuptiProfilerPushRangePtr ) (CUpti_Profiler_PushRange_Params* params);
CUptiResult ( *cuptiProfilerPopRangePtr ) (CUpti_Profiler_PopRange_Params* params);
CUptiResult ( *cuptiProfilerDisableProfilingPtr ) (CUpti_Profiler_DisableProfiling_Params* params);
CUptiResult ( *cuptiProfilerEndPassPtr ) (CUpti_Profiler_EndPass_Params* params);
CUptiResult ( *cuptiProfilerFlushCounterDataPtr ) (CUpti_Profiler_FlushCounterData_Params* params);
CUptiResult ( *cuptiProfilerUnsetConfigPtr ) (CUpti_Profiler_UnsetConfig_Params* params);
CUptiResult ( *cuptiProfilerEndSessionPtr ) (CUpti_Profiler_EndSession_Params* params);
CUptiResult ( *cuptiProfilerGetCounterAvailabilityPtr ) (CUpti_Profiler_GetCounterAvailability_Params* params);
CUptiResult ( *cuptiFinalizePtr ) (void);

#define NVPW_CALL( call, handleerror ) \
    do {  \
        NVPA_Status _status = (call);  \
        LOGCUPTICALL("\t" #call "\n");  \
        if (_status != NVPA_STATUS_SUCCESS) {  \
            ERRDBG("NVPA Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

static int load_cupti_perf_sym(void)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    if (dl_cupti == NULL) {
        ERRDBG("libcupti.so should already be loaded.\n");
        goto fn_fail;
    }

    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndSession");
    cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerGetCounterAvailability");
    cuptiFinalizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiFinalize");

fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static int unload_cupti_perf_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiDeviceGetChipNamePtr                                  = NULL;
    cuptiProfilerInitializePtr                                 = NULL;
    cuptiProfilerDeInitializePtr                               = NULL;
    cuptiProfilerCounterDataImageCalculateSizePtr              = NULL;
    cuptiProfilerCounterDataImageInitializePtr                 = NULL;
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = NULL;
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr    = NULL;
    cuptiProfilerBeginSessionPtr                               = NULL;
    cuptiProfilerSetConfigPtr                                  = NULL;
    cuptiProfilerBeginPassPtr                                  = NULL;
    cuptiProfilerEnableProfilingPtr                            = NULL;
    cuptiProfilerPushRangePtr                                  = NULL;
    cuptiProfilerPopRangePtr                                   = NULL;
    cuptiProfilerDisableProfilingPtr                           = NULL;
    cuptiProfilerEndPassPtr                                    = NULL;
    cuptiProfilerFlushCounterDataPtr                           = NULL;
    cuptiProfilerUnsetConfigPtr                                = NULL;
    cuptiProfilerEndSessionPtr                                 = NULL;
    cuptiProfilerGetCounterAvailabilityPtr                     = NULL;
    cuptiFinalizePtr                                           = NULL;
    return PAPI_OK;
}

static int load_nvpw_sym(void)
{
    COMPDBG("Entering.\n");
    char dlname[] = "libnvperf_host.so";
    char lookup_path[PATH_MAX];

    char *papi_cuda_perfworks = getenv("PAPI_CUDA_PERFWORKS");
    if (papi_cuda_perfworks) {
        sprintf(lookup_path, "%s/%s", papi_cuda_perfworks, dlname);
        dl_nvpw = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
    }

    const char *standard_paths[] = {
        "%s/extras/CUPTI/lib64/%s",
        "%s/lib64/%s",
        NULL,
    };

    if (linked_cudart_path && !dl_nvpw) {
        dl_nvpw = cuptic_load_dynamic_syms(linked_cudart_path, dlname, standard_paths);
    }

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_nvpw) {
        dl_nvpw = cuptic_load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
    }

    if (!dl_nvpw) {
        dl_nvpw = dlopen(dlname, RTLD_NOW | RTLD_GLOBAL);
        if (!dl_nvpw) {
            ERRDBG("Loading libnvperf_host.so failed.\n");
            goto fn_fail;
        }
    }

    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_GetSupportedChipNames");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_End");
    NVPW_CUDA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_SetCounterAvailability");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_IsAddMetricsPossible");
    NVPW_MetricsContext_GetCounterNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_End");

    Dl_info info;
    dladdr(NVPW_GetSupportedChipNamesPtr, &info);
    LOGDBG("NVPW library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_nvpw_sym(void)
{
    if (dl_nvpw) {
        dlclose(dl_nvpw);
        dl_nvpw = NULL;
    }
    NVPW_GetSupportedChipNamesPtr                     = NULL;
    NVPW_CUDA_MetricsContext_CreatePtr                = NULL;
    NVPW_MetricsContext_DestroyPtr                    = NULL;
    NVPW_MetricsContext_GetMetricNames_BeginPtr       = NULL;
    NVPW_MetricsContext_GetMetricNames_EndPtr         = NULL;
    NVPW_InitializeHostPtr                            = NULL;
    NVPW_MetricsContext_GetMetricProperties_BeginPtr  = NULL;
    NVPW_MetricsContext_GetMetricProperties_EndPtr    = NULL;
    NVPW_CUDA_RawMetricsConfig_CreatePtr              = NULL;
    NVPW_RawMetricsConfig_DestroyPtr                  = NULL;
    NVPW_RawMetricsConfig_BeginPassGroupPtr           = NULL;
    NVPW_RawMetricsConfig_EndPassGroupPtr             = NULL;
    NVPW_RawMetricsConfig_AddMetricsPtr               = NULL;
    NVPW_RawMetricsConfig_GenerateConfigImagePtr      = NULL;
    NVPW_RawMetricsConfig_GetConfigImagePtr           = NULL;
    NVPW_CounterDataBuilder_CreatePtr                 = NULL;
    NVPW_CounterDataBuilder_DestroyPtr                = NULL;
    NVPW_CounterDataBuilder_AddMetricsPtr             = NULL;
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr   = NULL;
    NVPW_CounterData_GetNumRangesPtr                  = NULL;
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = NULL;
    NVPW_MetricsContext_SetCounterDataPtr             = NULL;
    NVPW_MetricsContext_EvaluateToGpuValuesPtr        = NULL;
    NVPW_RawMetricsConfig_GetNumPassesPtr             = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr   = NULL;
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr     = NULL;
    NVPW_MetricsContext_GetCounterNames_BeginPtr      = NULL;
    NVPW_MetricsContext_GetCounterNames_EndPtr        = NULL;
    return PAPI_OK;
}

static int initialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE, NULL };
    papi_errno = cuptiProfilerInitializePtr(&profilerInitializeParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: cuptiProfilerInitialize failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int finalize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = { CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE, NULL };
    papi_errno = cuptiProfilerDeInitializePtr(&profilerDeInitializeParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI Error %d: cuptiProfilerDeInitialize failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int initialize_perfworks_api(void)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    NVPW_InitializeHost_Params perfInitHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE, NULL };
    papi_errno = NVPW_InitializeHostPtr(&perfInitHostParams);
    if (papi_errno != NVPA_STATUS_SUCCESS) {
        ERRDBG("NVPW Error %d: NVPW_InitializeHostPtr failed.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int get_chip_name(int dev_num, char* chipName)
{
    int papi_errno;
    CUpti_Device_GetChipName_Params getChipName = {
        .structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .deviceIndex = 0
    };
    getChipName.deviceIndex = dev_num;
    papi_errno = cuptiDeviceGetChipNamePtr(&getChipName);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get chip name for device %d\n", papi_errno, dev_num);
        return PAPI_EMISC;
    }
    strcpy(chipName, getChipName.pChipName);
    return PAPI_OK;
}

struct byte_array_s {
    int      size;
    uint8_t *data;
};

struct cuptip_gpu_state_s {
    int                    gpu_id;
    cuptiu_event_table_t  *event_names;
    int                    rmr_count;
    NVPA_RawMetricRequest *rmr;
    MCCP_t                *pmetricsContextCreateParams;
    byte_array_t           counterDataImagePrefix;
    byte_array_t           configImage;
    byte_array_t           counterDataImage;
    byte_array_t           counterDataScratchBuffer;
    byte_array_t           counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams;
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams;
};

struct cuptip_control_s {
    cuptip_gpu_state_t *gpu_ctl;
    int                 read_count;
    enum running_e      running;
    cuptic_info_t       info;
};

struct list_metrics_s {
    char chip_name[32];
    MCCP_t *pmetricsContextCreateParams;
    int num_metrics;
    cuptiu_event_table_t *nv_metrics;
};

static int event_name_tokenize(const char *name, char *nv_name, int *gpuid)
{
    if (nv_name == NULL) {
        return PAPI_EINVAL;
    }

    int numchars;
    const char token[] = ":device=";
    const int tok_len = 8;
    char *rest;

    char *getdevstr = strstr(name, token);
    if (getdevstr == NULL) {
        ERRDBG("Event name does not contain device number.\n");
        return PAPI_EINVAL;
    }
    getdevstr += tok_len;
    *gpuid = strtol(getdevstr, &rest, 10);
    numchars = strlen(name) - strlen(getdevstr) - tok_len;
    memcpy(nv_name, name, numchars);
    nv_name[numchars] = '\0';

    return PAPI_OK;
}

static int add_events_per_gpu(cuptip_control_t state, cuptiu_event_table_t *event_names)
{
    COMPDBG("Entering.\n");
    int i, gpu_id, papi_errno = PAPI_OK;
    char nvName[PAPI_MAX_STR_LEN];
    cuptiu_event_t *evt_rec;
    for (i=0; i < num_gpus; i++) {
        papi_errno = cuptiu_event_table_create(sizeof(cuptiu_event_t), &(state->gpu_ctl[i].event_names));
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
    for (i = 0; i < (int) event_names->count; i++) {
        papi_errno = cuptiu_event_table_get_item(event_names, i, &evt_rec);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        papi_errno = event_name_tokenize(evt_rec->name, (char*) &nvName, &gpu_id);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        if (gpu_id < 0 || gpu_id > num_gpus) {
            papi_errno = PAPI_EINVAL;
            goto fn_exit;
        }
        cuptiu_event_table_insert_record(state->gpu_ctl[gpu_id].event_names, evt_rec->name, evt_rec->evt_code, i);
        LOGDBG("Adding event gpu %d name %s with code %d at pos %d\n", gpu_id, evt_rec->name, evt_rec->evt_code, i);
    }
fn_exit:
    return papi_errno;
}

static int retrieve_metric_details(NVPA_MetricsContext *pMetricsContext, const char *nv_name,
                                   char *description, int *numDep, NVPA_RawMetricRequest **pRMR)
{
    COMPDBG("Entering.\n");
    int num_dep, i, len;
    NVPA_Status nvpa_err;

    if (nv_name == NULL || description == NULL) {
        return PAPI_EINVAL;
    }

    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
        .pMetricName = nv_name,
    };
    nvpa_err = NVPW_MetricsContext_GetMetricProperties_BeginPtr(&getMetricPropertiesBeginParams);

    if (nvpa_err != NVPA_STATUS_SUCCESS || getMetricPropertiesBeginParams.ppRawMetricDependencies == NULL) {
        strcpy(description, "Could not get description.");
        return PAPI_EINVAL;
    }

    for (num_dep = 0; getMetricPropertiesBeginParams.ppRawMetricDependencies[num_dep] != NULL; num_dep++) {;}

    NVPA_RawMetricRequest *rmr = (NVPA_RawMetricRequest *) papi_calloc(num_dep, sizeof(NVPA_RawMetricRequest));
    if (rmr == NULL) {
        return PAPI_ENOMEM;
    }

    for (i = 0; i < num_dep; i++) {
        rmr[i].pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[i]);
        rmr[i].isolated = 1;
        rmr[i].keepInstances = 1;
        rmr[i].structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    }

    len = snprintf(description, PAPI_2MAX_STR_LEN, "%s. Units=(%s)",
                getMetricPropertiesBeginParams.pDescription,
                getMetricPropertiesBeginParams.pDimUnits);
    if (len > PAPI_2MAX_STR_LEN) {
        ERRDBG("String formatting exceeded max string length.\n");
        return PAPI_ENOMEM;
    }
    *numDep = num_dep;
    *pRMR = rmr;
    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
    };
    NVPW_CALL( NVPW_MetricsContext_GetMetricProperties_EndPtr(&getMetricPropertiesEndParams), return PAPI_EMISC );
    return PAPI_OK;
}

static int get_event_names_rmr(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    NVPA_RawMetricRequest *all_rmr=NULL;
    int count_raw_metrics = 0;
    unsigned int i;
    int j, k, num_dep;
    NVPA_RawMetricRequest *temp;
    char nv_name[PAPI_MAX_STR_LEN];
    int gpuid;
    cuptiu_event_t *evt_rec;
    for (i = 0; i < gpu_ctl->event_names->count; i++) {
        papi_errno = cuptiu_event_table_get_item(gpu_ctl->event_names, i, &evt_rec);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        papi_errno = event_name_tokenize(evt_rec->name, (char *) &nv_name, &gpuid);

        papi_errno = retrieve_metric_details(gpu_ctl->pmetricsContextCreateParams->pMetricsContext, nv_name, evt_rec->desc, &num_dep, &temp);
        if (papi_errno != PAPI_OK) {
            papi_errno = PAPI_ENOEVNT;
            goto fn_exit;
        }

        all_rmr = (NVPA_RawMetricRequest *) papi_realloc(all_rmr, (count_raw_metrics + num_dep) * sizeof(NVPA_RawMetricRequest));
        if (all_rmr == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        for (j = 0; j < num_dep; j++) {
            k = j + count_raw_metrics;
            all_rmr[k].structSize = temp[j].structSize;
            all_rmr[k].pPriv = NULL;
            all_rmr[k].pMetricName = strdup(temp[j].pMetricName);
            all_rmr[k].keepInstances = 1;
            all_rmr[k].isolated = 1;
            papi_free((void *) temp[j].pMetricName);
        }
        count_raw_metrics += num_dep;
        papi_free(temp);
    }
    gpu_ctl->rmr = all_rmr;
    gpu_ctl->rmr_count = count_raw_metrics;
fn_exit:
    return papi_errno;
}

static int check_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count, NVPA_RawMetricRequest *rmr, int *num_pass)
{
    COMPDBG("Entering.\n");
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
        .maxPassCount = 1,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
        .pRawMetricRequests = rmr,
        .numMetricRequests = rmr_count,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), goto fn_fail );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = {
        .structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_GetNumPassesPtr(&rawMetricsConfigGetNumPassesParams), goto fn_fail );

    int numNestingLevels = 1, numIsolatedPasses, numPipelinedPasses;
    numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;

    *num_pass = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

    if (*num_pass > 1) {
        ERRDBG("Metrics requested requires multiple passes to profile.\n");
        return PAPI_EMULPASS;
    }

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int nvpw_cuda_metricscontext_create(cuptip_control_t state)
{
    int gpu_id, found, papi_errno = PAPI_OK;
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            gpu_ctl->pmetricsContextCreateParams = state->gpu_ctl[found].pmetricsContextCreateParams;
            continue;
        }
        MCCP_t *pMCCP = (MCCP_t *) papi_calloc(1, sizeof(MCCP_t));
        if (pMCCP == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        pMCCP->structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE;
        pMCCP->pChipName = avail_events[gpu_id].chip_name;
        NVPW_CALL( NVPW_CUDA_MetricsContext_CreatePtr(pMCCP), goto fn_fail );
        gpu_ctl->pmetricsContextCreateParams = pMCCP;
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static int nvpw_cuda_metricscontext_destroy(cuptip_control_t state)
{
    int gpu_id, found, papi_errno = PAPI_OK;
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            gpu_ctl->pmetricsContextCreateParams = NULL;
            continue;
        }
        if (gpu_ctl->pmetricsContextCreateParams->pMetricsContext) {
            NVPW_MetricsContext_Destroy_Params mCDP = {
                .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
            };
            NVPW_CALL( NVPW_MetricsContext_DestroyPtr(&mCDP), goto fn_fail );
            papi_free(gpu_ctl->pmetricsContextCreateParams);
            gpu_ctl->pmetricsContextCreateParams = NULL;
        }
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static int control_state_validate(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    int gpu_id, papi_errno = PAPI_OK, passes;
    cuptip_gpu_state_t *gpu_ctl;

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->event_names->count == 0) {
            continue;
        }

        papi_errno = get_event_names_rmr(gpu_ctl);

        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
            .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = avail_events[gpu_id].chip_name,
        };
        NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), goto fn_fail );

        papi_errno = check_num_passes(nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                               gpu_ctl->rmr_count, gpu_ctl->rmr, &passes);

        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
            .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        };
        NVPW_CALL( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), goto fn_fail );
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static int get_counter_availability(cuptip_gpu_state_t *gpu_ctl)
{
    int papi_errno;
    /* Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data */
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
        .structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pCounterAvailabilityImage = NULL,
    };
    papi_errno = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get size.\n", papi_errno);
        return PAPI_EMISC;
    }
    /* Allocate sized counterAvailabilityImage */
    gpu_ctl->counterAvailabilityImage.size = getCounterAvailabilityParams.counterAvailabilityImageSize;
    gpu_ctl->counterAvailabilityImage.data = (uint8_t *) papi_malloc(gpu_ctl->counterAvailabilityImage.size);
    if (gpu_ctl->counterAvailabilityImage.data == NULL) {
        return PAPI_ENOMEM;
    }
    /* Initialize counterAvailabilityImage */
    getCounterAvailabilityParams.pCounterAvailabilityImage = gpu_ctl->counterAvailabilityImage.data;
    papi_errno = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get bytes.\n", papi_errno);
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

static int metric_get_config_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
        .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
        .pChipName = avail_events[gpu_ctl->gpu_id].chip_name,
    };
    NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), goto fn_fail );

    if( gpu_ctl->counterAvailabilityImage.data != NULL) {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {
            .structSize = NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
            .pCounterAvailabilityImage = gpu_ctl->counterAvailabilityImage.data,
        };
        NVPW_CALL( NVPW_RawMetricsConfig_SetCounterAvailabilityPtr(&setCounterAvailabilityParams), goto fn_fail );
    };

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .maxPassCount = 1,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .pRawMetricRequests = gpu_ctl->rmr,
        .numMetricRequests = gpu_ctl->rmr_count,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), goto fn_fail );

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), goto fn_fail );

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_GenerateConfigImagePtr(&generateConfigImageParams), goto fn_fail );

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), goto fn_fail );

    gpu_ctl->configImage.size = getConfigImageParams.bytesCopied;
    gpu_ctl->configImage.data = (uint8_t *) papi_calloc(gpu_ctl->configImage.size, sizeof(uint8_t));
    if (gpu_ctl->configImage.data == NULL) {
        ERRDBG("calloc gpu_ctl->configImage.data failed!");
        return PAPI_ENOMEM;
    }

    getConfigImageParams.bytesAllocated = gpu_ctl->configImage.size;
    getConfigImageParams.pBuffer = gpu_ctl->configImage.data;
    NVPW_CALL( NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), goto fn_fail );

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int metric_get_counter_data_prefix_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        .structSize = NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pChipName = avail_events[gpu_ctl->gpu_id].chip_name,
    };
    NVPW_CALL( NVPW_CounterDataBuilder_CreatePtr(&counterDataBuilderCreateParams), goto fn_fail );

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .pRawMetricRequests = gpu_ctl->rmr,
        .numMetricRequests = gpu_ctl->rmr_count,
    };
    NVPW_CALL( NVPW_CounterDataBuilder_AddMetricsPtr(&addMetricsParams), goto fn_fail );

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        .structSize = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), goto fn_fail );

    gpu_ctl->counterDataImagePrefix.size = getCounterDataPrefixParams.bytesCopied;
    gpu_ctl->counterDataImagePrefix.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataImagePrefix.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataImagePrefix.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataImagePrefix.data failed!");
        return PAPI_ENOMEM;
    }

    getCounterDataPrefixParams.bytesAllocated = gpu_ctl->counterDataImagePrefix.size;
    getCounterDataPrefixParams.pBuffer = gpu_ctl->counterDataImagePrefix.data;
    NVPW_CALL( NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), goto fn_fail );

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        .structSize = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
    };
    NVPW_CALL( NVPW_CounterDataBuilder_DestroyPtr(&counterDataBuilderDestroyParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int create_counter_data_image(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    gpu_ctl->counterDataImageOptions = (CUpti_Profiler_CounterDataImageOptions) {
        .structSize = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataPrefix = gpu_ctl->counterDataImagePrefix.data,
        .counterDataPrefixSize = gpu_ctl->counterDataImagePrefix.size,
        .maxNumRanges = 1,
        .maxNumRangeTreeNodes = 1,
        .maxRangeNameLength = 64,
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &gpu_ctl->counterDataImageOptions,
    };
    CUPTI_CALL( cuptiProfilerCounterDataImageCalculateSizePtr(&calculateSizeParams), goto fn_fail );

    gpu_ctl->initializeParams = (CUpti_Profiler_CounterDataImage_Initialize_Params) {
        .structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &gpu_ctl->counterDataImageOptions,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
    };

    gpu_ctl->counterDataImage.size = calculateSizeParams.counterDataImageSize;
    gpu_ctl->counterDataImage.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataImage.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataImage.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataImage.data failed!\n");
        return PAPI_ENOMEM;
    }

    gpu_ctl->initializeParams.pCounterDataImage = gpu_ctl->counterDataImage.data;
    CUPTI_CALL( cuptiProfilerCounterDataImageInitializePtr(&gpu_ctl->initializeParams), goto fn_fail );

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = gpu_ctl->initializeParams.pCounterDataImage,
    };
    CUPTI_CALL( cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr(&scratchBufferSizeParams), goto fn_fail );

    gpu_ctl->counterDataScratchBuffer.size = scratchBufferSizeParams.counterDataScratchBufferSize;
    gpu_ctl->counterDataScratchBuffer.data = (uint8_t *) papi_calloc(gpu_ctl->counterDataScratchBuffer.size, sizeof(uint8_t));
    if (gpu_ctl->counterDataScratchBuffer.data == NULL) {
        ERRDBG("calloc gpu_ctl->counterDataScratchBuffer.data failed!\n");
        return PAPI_ENOMEM;
    }

    gpu_ctl->initScratchBufferParams = (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params) {
        .structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = gpu_ctl->initializeParams.pCounterDataImage,
        .counterDataScratchBufferSize = gpu_ctl->counterDataScratchBuffer.size,
        .pCounterDataScratchBuffer = gpu_ctl->counterDataScratchBuffer.data,
    };
    CUPTI_CALL( cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&gpu_ctl->initScratchBufferParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int reset_cupti_prof_config_images(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    papi_free(gpu_ctl->counterDataImagePrefix.data);
    papi_free(gpu_ctl->configImage.data);
    papi_free(gpu_ctl->counterDataImage.data);
    papi_free(gpu_ctl->counterDataScratchBuffer.data);
    papi_free(gpu_ctl->counterAvailabilityImage.data);
    gpu_ctl->counterDataImagePrefix.data = NULL;
    gpu_ctl->configImage.data = NULL;
    gpu_ctl->counterDataImage.data = NULL;
    gpu_ctl->counterDataScratchBuffer.data = NULL;
    gpu_ctl->counterAvailabilityImage.data = NULL;
    gpu_ctl->counterDataImagePrefix.size = 0;
    gpu_ctl->configImage.size = 0;
    gpu_ctl->counterDataImage.size = 0;
    gpu_ctl->counterDataScratchBuffer.size = 0;
    gpu_ctl->counterAvailabilityImage.size = 0;
    return PAPI_OK;
}

static int begin_profiling(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("Entering.\n");
    byte_array_t *configImage = &(gpu_ctl->configImage);
    byte_array_t *counterDataScratchBuffer = &(gpu_ctl->counterDataScratchBuffer);
    byte_array_t *counterDataImage = &(gpu_ctl->counterDataImage);

    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        .structSize = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .counterDataImageSize = counterDataImage->size,
        .pCounterDataImage = counterDataImage->data,
        .counterDataScratchBufferSize = counterDataScratchBuffer->size,
        .pCounterDataScratchBuffer = counterDataScratchBuffer->data,
        .range = CUPTI_UserRange,
        .replayMode = CUPTI_UserReplay,
        .maxRangesPerPass = 1,
        .maxLaunchesPerPass = 1,
    };
    CUPTI_CALL( cuptiProfilerBeginSessionPtr(&beginSessionParams), goto fn_fail );

    CUpti_Profiler_SetConfig_Params setConfigParams = {
        .structSize = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pConfig = configImage->data,
        .configSize = configImage->size,
        .minNestingLevel = 1,
        .numNestingLevels = 1,
        .passIndex = 0,
        .targetNestingLevel = 1,
    };
    CUPTI_CALL( cuptiProfilerSetConfigPtr(&setConfigParams), goto fn_fail );

    CUpti_Profiler_BeginPass_Params beginPassParams = {
        .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerBeginPassPtr(&beginPassParams), goto fn_fail );

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        .structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerEnableProfilingPtr(&enableProfilingParams), goto fn_fail );

    char rangeName[64];
    sprintf(rangeName, "PAPI_Range_%d", gpu_ctl->gpu_id);
    CUpti_Profiler_PushRange_Params pushRangeParams = {
        .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pRangeName = (const char*) &rangeName,
        .rangeNameLength = 100,
    };
    CUPTI_CALL( cuptiProfilerPushRangePtr(&pushRangeParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int end_profiling(cuptip_gpu_state_t *gpu_ctl)
{

    COMPDBG("EndProfiling. dev = %d\n", gpu_ctl->gpu_id);
    (void) gpu_ctl;

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        .structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerDisableProfilingPtr(&disableProfilingParams), goto fn_fail );

    CUpti_Profiler_PopRange_Params popRangeParams = {
        .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerPopRangePtr(&popRangeParams), goto fn_fail );

    CUpti_Profiler_EndPass_Params endPassParams = {
        .structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerEndPassPtr(&endPassParams), goto fn_fail );

    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        .structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), goto fn_fail );

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        .structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerUnsetConfigPtr(&unsetConfigParams), goto fn_fail );

    CUpti_Profiler_EndSession_Params endSessionParams = {
        .structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL( cuptiProfilerEndSessionPtr(&endSessionParams), goto fn_fail );

    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int get_measured_values(cuptip_gpu_state_t *gpu_ctl)
{
    COMPDBG("eval_metric_values. dev = %d\n", gpu_ctl->gpu_id);
    if (!gpu_ctl->counterDataImage.size) {
        ERRDBG("Counter Data Image is empty!\n");
        return PAPI_EINVAL;
    }
    int i, papi_errno = PAPI_OK;
    int numMetrics = gpu_ctl->event_names->count;

    int dummy;
    char **metricNames = (char**) papi_calloc(numMetrics, sizeof(char *));
    if (metricNames == NULL) {
        ERRDBG("calloc metricNames failed.\n");
        return PAPI_ENOMEM;
    }
    cuptiu_event_t *evt_rec;
    for (i = 0; i < numMetrics; i++) {
        papi_errno = cuptiu_event_table_get_item(gpu_ctl->event_names, i, &evt_rec);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        papi_errno = event_name_tokenize(evt_rec->name, evt_rec->desc, &dummy);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        metricNames[i] = (char *) &(evt_rec->desc);
        LOGDBG("Setting metric name %s\n", metricNames[i]);
    }

    double *gpuValues = (double*) papi_malloc(numMetrics * sizeof(double));
    if (gpuValues == NULL) {
        ERRDBG("malloc gpuValues failed.\n");
        return PAPI_ENOMEM;
    }

    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        .structSize = NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
        .pCounterDataImage = gpu_ctl->counterDataImage.data,
        .rangeIndex = 0,
        .isolated = 1,
    };
    NVPW_CALL( NVPW_MetricsContext_SetCounterDataPtr(&setCounterDataParams), goto fn_fail );
    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
        .structSize = NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = gpu_ctl->pmetricsContextCreateParams->pMetricsContext,
        .numMetrics = numMetrics,
        .ppMetricNames = (const char* const*) metricNames,
        .pMetricValues = gpuValues,
    };
    NVPW_CALL( NVPW_MetricsContext_EvaluateToGpuValuesPtr(&evalToGpuParams), goto fn_fail );
    papi_free(metricNames);
    for (i = 0; i < (int) gpu_ctl->event_names->count; i++) {
        papi_errno = cuptiu_event_table_get_item(gpu_ctl->event_names, i, &evt_rec);
        if (papi_errno != PAPI_OK) {
            papi_free(gpuValues);
            goto fn_exit;
        }
        evt_rec->value = gpuValues[i];
    }
    papi_free(gpuValues);
fn_exit:
    return papi_errno;
fn_fail:
    return PAPI_EMISC;
}

/* List metrics API */
static int find_same_chipname(int gpu_id)
{
    int i;
    for (i = 0; i < gpu_id; i++) {
        if (!strcmp(avail_events[gpu_id].chip_name, avail_events[i].chip_name)) {
            return i;
        }
    }
    return -1;
}

static int init_all_metrics(void)
{
    int gpu_id, papi_errno = PAPI_OK;
    avail_events = (list_metrics_t *) papi_calloc(num_gpus, sizeof(list_metrics_t));
    if (avail_events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_exit;
    }
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        papi_errno = get_chip_name(gpu_id, avail_events[gpu_id].chip_name);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
    int found;
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            avail_events[gpu_id].pmetricsContextCreateParams = avail_events[found].pmetricsContextCreateParams;
            continue;
        }
        MCCP_t *pMCCP = (MCCP_t *) papi_calloc(1, sizeof(MCCP_t));
        if (pMCCP == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        pMCCP->structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE;
        pMCCP->pChipName = avail_events[gpu_id].chip_name;
        NVPW_CALL( NVPW_CUDA_MetricsContext_CreatePtr(pMCCP), goto fn_fail );

        avail_events[gpu_id].pmetricsContextCreateParams = pMCCP;
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int cuptip_event_enum(cuptiu_event_table_t *all_evt_names)
{
    int gpu_id, i, found, listsubmetrics = 1, papi_errno = PAPI_OK;
    if (avail_events[0].nv_metrics != NULL) {
        /* Already eumerated for 1st device? Then exit... */
        goto fn_exit;
    }
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        LOGDBG("Getting metric names for gpu %d\n", gpu_id);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            avail_events[gpu_id].num_metrics = avail_events[found].num_metrics;
            avail_events[gpu_id].nv_metrics = avail_events[found].nv_metrics;
            continue;
        }
        /* If same chip_name not found, get all the details for this gpu */

        NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = {
            .structSize = NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = avail_events[gpu_id].pmetricsContextCreateParams->pMetricsContext,
            .hidePeakSubMetrics = !listsubmetrics,
            .hidePerCycleSubMetrics = !listsubmetrics,
            .hidePctOfPeakSubMetrics = !listsubmetrics,
        };
        NVPW_CALL( NVPW_MetricsContext_GetMetricNames_BeginPtr(&getMetricNameBeginParams), goto fn_fail );

        avail_events[gpu_id].num_metrics = getMetricNameBeginParams.numMetrics;

        papi_errno = cuptiu_event_table_create_init_capacity(avail_events[gpu_id].num_metrics, sizeof(cuptiu_event_t), &(avail_events[gpu_id].nv_metrics));
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        for (i = 0; i < avail_events[gpu_id].num_metrics; i++) {
            papi_errno = cuptiu_event_table_insert_record(avail_events[gpu_id].nv_metrics,
                                      getMetricNameBeginParams.ppMetricNames[i],
                                      i, 0);
            if (papi_errno != PAPI_OK) {
                goto fn_exit;
            }
        }

        NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = {
            .structSize = NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = avail_events[gpu_id].pmetricsContextCreateParams->pMetricsContext,
        };
        NVPW_CALL( NVPW_MetricsContext_GetMetricNames_EndPtr((NVPW_MetricsContext_GetMetricNames_End_Params *) &getMetricNameEndParams), goto fn_fail );

    }
    char evt_name[PAPI_2MAX_STR_LEN];
    cuptiu_event_t *find = NULL;
    cuptiu_event_t *evt_rec = NULL;
    int len;
    unsigned int curr = all_evt_names->count;
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        for (i = 0; i < avail_events[gpu_id].num_metrics; i++) {
            papi_errno = cuptiu_event_table_get_item(avail_events[gpu_id].nv_metrics, i, &evt_rec);
            if (papi_errno != PAPI_OK) {
                goto fn_exit;
            }
            len = snprintf(evt_name, PAPI_2MAX_STR_LEN, "%s:device=%d", evt_rec->name, gpu_id);
            if (len > PAPI_2MAX_STR_LEN) {
                ERRDBG("String formatting exceeded maximum length.\n");
                papi_errno = PAPI_ENOMEM;
                goto fn_exit;
            }
            if (cuptiu_event_table_find_name(all_evt_names, evt_name, &find) == PAPI_ENOEVNT) {
                papi_errno = cuptiu_event_table_insert_record(all_evt_names, evt_name, curr, 0);
                if (papi_errno != PAPI_OK) {
                    goto fn_exit;
                }
                ++curr;
            }
        }
    }
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int cuptip_event_name_to_descr(const char *evt_name, char *description)
{
    int papi_errno, numdep, gpu_id, passes;
    char nv_name[PAPI_MAX_STR_LEN];
    cuptiu_event_t *evt_rec = NULL;
    NVPA_RawMetricRequest *temp;
    papi_errno = event_name_tokenize(evt_name, nv_name, &gpu_id);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = cuptiu_event_table_find_name(avail_events[gpu_id].nv_metrics, nv_name, &evt_rec);
    if (papi_errno != PAPI_OK) {
        ERRDBG("Event name not found in avail_events array.\n");
        goto fn_exit;
    }
    char *desc = evt_rec->desc;
    if (desc[0] == '\0') {
        papi_errno = retrieve_metric_details(avail_events[gpu_id].pmetricsContextCreateParams->pMetricsContext,
                                             nv_name, desc, &numdep, &temp);
        if (papi_errno == PAPI_OK) {
            NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
                .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
                .pChipName = avail_events[gpu_id].chip_name,
            };
            NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), goto fn_fail );

            papi_errno = check_num_passes(nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                                   numdep, temp, &passes);

            NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
                .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
            };
            NVPW_CALL( NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), goto fn_fail );

            snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " Numpass=%d", passes);
            if (passes > 1) {
                snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " (multi-pass not supported)");
            }

            const char *token_sw_evt = "sass";
            if (strstr(nv_name, token_sw_evt) != NULL) {
                snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " (SW event)");
            }
        }
        papi_free(temp);
    }
    strcpy(description, desc);
fn_exit:
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static void free_all_enumerated_metrics(void)
{
    COMPDBG("Entering.\n");
    int gpu_id, found;
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams;
    if (avail_events == NULL) {
        return;
    }
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            avail_events[gpu_id].num_metrics = 0;
            avail_events[gpu_id].nv_metrics = NULL;
            avail_events[gpu_id].pmetricsContextCreateParams = NULL;
            continue;
        }
        if (avail_events[gpu_id].pmetricsContextCreateParams->pMetricsContext) {
            metricsContextDestroyParams = (NVPW_MetricsContext_Destroy_Params) {
                .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pMetricsContext = avail_events[gpu_id].pmetricsContextCreateParams->pMetricsContext,
            };
            NVPW_CALL(NVPW_MetricsContext_DestroyPtr(&metricsContextDestroyParams), );
        }
        papi_free(avail_events[gpu_id].pmetricsContextCreateParams);
        avail_events[gpu_id].pmetricsContextCreateParams = NULL;

        if (avail_events[gpu_id].nv_metrics) {
            cuptiu_event_table_destroy( &(avail_events[gpu_id].nv_metrics) );
        }
    }
    papi_free(avail_events);
    avail_events = NULL;
}

/* CUPTI Profiler component API functions */
int cuptip_init(void)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;

    papi_errno = load_cupti_perf_sym();
    papi_errno += load_nvpw_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to load CUDA library functions.");
        goto fn_fail;
    }

    papi_errno = cuptic_device_get_count(&num_gpus);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    if (num_gpus <= 0) {
        cuptic_disabled_reason_set("No GPUs found on system.");
        goto fn_fail;
    }

    papi_errno = initialize_cupti_profiler_api();
    papi_errno += initialize_perfworks_api();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to initialize CUPTI profiler libraries.");
        goto fn_fail;
    }
    papi_errno = init_all_metrics();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    papi_errno = cuInitPtr(0);
    if (papi_errno != CUDA_SUCCESS) {
        cuptic_disabled_reason_set("Failed to initialize CUDA driver API.");
        goto fn_fail;
    }
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

int cuptip_control_create(cuptiu_event_table_t *event_names, cuptic_info_t thr_info, cuptip_control_t *pstate)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK, gpu_id;
    cuptip_control_t state = (cuptip_control_t) papi_calloc (1, sizeof(struct cuptip_control_s));
    if (state == NULL) {
        return PAPI_ENOMEM;
    }
    state->gpu_ctl = (cuptip_gpu_state_t *) papi_calloc(num_gpus, sizeof(cuptip_gpu_state_t));
    if (state->gpu_ctl == NULL) {
        return PAPI_ENOMEM;
    }
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        state->gpu_ctl[gpu_id].gpu_id = gpu_id;
    }

    /* Register the user created cuda context for the current gpu if not already known */
    papi_errno = cuptic_ctxarr_update_current(thr_info);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = nvpw_cuda_metricscontext_create(state);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = add_events_per_gpu(state, event_names);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = control_state_validate(state);
    state->info = thr_info;

fn_exit:
    *pstate = state;
    return papi_errno;
}

int cuptip_control_destroy(cuptip_control_t *pstate)
{
    COMPDBG("Entering.\n");
    cuptip_control_t state = *pstate;
    int i, j;
    int papi_errno = nvpw_cuda_metricscontext_destroy(state);
    for (i = 0; i < num_gpus; i++) {
        reset_cupti_prof_config_images( &(state->gpu_ctl[i]) );
        cuptiu_event_table_destroy( &(state->gpu_ctl[i].event_names) );
        for (j = 0; j < state->gpu_ctl[i].rmr_count; j++) {
            papi_free((void *) state->gpu_ctl[i].rmr[j].pMetricName);
        }
        papi_free(state->gpu_ctl[i].rmr);
    }
    papi_free(state->gpu_ctl);
    papi_free(state);
    *pstate = NULL;
    return papi_errno;
}

int cuptip_control_start(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    cuptip_gpu_state_t *gpu_ctl;
    CUcontext userCtx, ctx;
    CUDA_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );
    if (userCtx == NULL) {
        CUDART_CALL( cudaFreePtr(NULL), goto fn_fail_misc );
        CUDA_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );
    }
    int gpu_id;
    int papi_errno = PAPI_OK;
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->event_names->count == 0) {
            continue;
        }
        LOGDBG("Device num %d: event_count %d, rmr count %d\n", gpu_id, gpu_ctl->event_names->count, gpu_ctl->rmr_count);
        papi_errno = cuptic_device_acquire(state->gpu_ctl[gpu_id].event_names);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Profiling same gpu from multiple event sets not allowed.\n");
            return papi_errno;
        }
        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        CUDA_CALL( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );
        papi_errno = get_counter_availability(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Error getting counter availability image.\n");
            return papi_errno;
        }
        /* CUPTI profiler host configuration */
        papi_errno = metric_get_config_image(gpu_ctl);
        papi_errno += metric_get_counter_data_prefix_image(gpu_ctl);
        papi_errno += create_counter_data_image(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Failed to create CUPTI profiler state for gpu %d\n", gpu_id);
            goto fn_fail;
        }
        papi_errno = begin_profiling(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Failed to start profiling for gpu %d\n", gpu_id);
            goto fn_fail;
        }
    }
    state->running = True;
fn_exit:
    CUDA_CALL( cuCtxSetCurrentPtr(userCtx), goto fn_fail_misc );
    return papi_errno;
fn_fail:
    papi_errno = PAPI_ECMP;
    goto fn_exit;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int cuptip_control_stop(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    cuptip_gpu_state_t *gpu_ctl;
    CUcontext userCtx = NULL, ctx = NULL;
    CUDA_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );
    if (userCtx == NULL) {
        CUDART_CALL( cudaFreePtr(NULL), goto fn_fail_misc );
        CUDA_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );
    }
    int gpu_id;
    int papi_errno = PAPI_OK;
    if (state->running == False) {
        ERRDBG("Profiler is already stopped.\n");
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->event_names->count == 0) {
            continue;
        }
        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        CUDA_CALL( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );
        papi_errno = end_profiling(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            ERRDBG("Failed to stop profiling on gpu %d\n", gpu_id);
            goto fn_fail;
        }
        papi_errno = cuptic_device_release(state->gpu_ctl[gpu_id].event_names);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }
    state->running = False;
fn_exit:
    CUDA_CALL( cuCtxSetCurrentPtr(userCtx), goto fn_fail_misc );
    return papi_errno;
fn_fail:
    goto fn_exit;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

static enum collection_method_e get_event_collection_method(const char *evt_name)
{
    if (strstr(evt_name, ".sum") != NULL) {
        return RunningSum;
    }
    else if (strstr(evt_name, ".min") != NULL) {
        return RunningMin;
    }
    else if (strstr(evt_name, ".max") != NULL) {
        return RunningMax;
    }
    else {
        return SpotValue;
    }
}

int cuptip_control_read(cuptip_control_t state, long long *values)
{
    COMPDBG("Entering.\n");
    int papi_errno, gpu_id, i;
    cuptip_gpu_state_t *gpu_ctl = NULL;
    CUcontext userCtx = NULL, ctx = NULL;
    CUDA_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc);
    if (userCtx == NULL) {
        CUDART_CALL( cudaFreePtr(NULL), goto fn_fail_misc );
        CUDART_CALL( cuCtxGetCurrentPtr(&userCtx), goto fn_fail_misc );
    }
    unsigned int evt_pos;
    long long val;
    cuptiu_event_t *evt_rec = NULL;
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_ctl = &(state->gpu_ctl[gpu_id]);
        if (gpu_ctl->event_names->count == 0) {
            continue;
        }

        papi_errno = cuptic_ctxarr_get_ctx(state->info, gpu_id, &ctx);
        CUDA_CALL( cuCtxSetCurrentPtr(ctx), goto fn_fail_misc );

        CUpti_Profiler_PopRange_Params popRangeParams = {
            .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        CUPTI_CALL( cuptiProfilerPopRangePtr(&popRangeParams), goto fn_fail_misc );

        CUpti_Profiler_EndPass_Params endPassParams = {
            .structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        CUPTI_CALL( cuptiProfilerEndPassPtr(&endPassParams), goto fn_fail_misc );

        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
            .structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        CUPTI_CALL( cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), goto fn_fail_misc );

        papi_errno = get_measured_values(gpu_ctl);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        for (i = 0; i < (int) gpu_ctl->event_names->count; i++) {
            papi_errno = cuptiu_event_table_get_item(gpu_ctl->event_names, i, &evt_rec);
            if (papi_errno != PAPI_OK) {
                goto fn_exit;
            }
            evt_pos = evt_rec->evt_pos;
            val = (long long) evt_rec->value;

            if (state->read_count == 0) {
                values[evt_pos] = val;
            }
            else {
                switch (get_event_collection_method(evt_rec->name)) {
                    case RunningSum:
                        values[evt_pos] += val;
                        break;
                    case RunningMin:
                        values[evt_pos] = values[evt_pos] < val ? values[evt_pos] : val;
                        break;
                    case RunningMax:
                        values[evt_pos] = values[evt_pos] > val ? values[evt_pos] : val;
                        break;
                    default:
                        values[evt_pos] = val;
                        break;
                }
            }
        }

        CUPTI_CALL( cuptiProfilerCounterDataImageInitializePtr(&gpu_ctl->initializeParams), goto fn_fail_misc );
        CUPTI_CALL( cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&gpu_ctl->initScratchBufferParams), goto fn_fail_misc );

        CUpti_Profiler_BeginPass_Params beginPassParams = {
            .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
        };
        CUPTI_CALL( cuptiProfilerBeginPassPtr(&beginPassParams), goto fn_fail_misc );

        char rangeName[64];
        sprintf(rangeName, "PAPI_Range_%d", gpu_ctl->gpu_id);
        CUpti_Profiler_PushRange_Params pushRangeParams = {
            .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .ctx = NULL,
            .pRangeName = (const char*) &rangeName,
            .rangeNameLength = 100,
        };
        CUPTI_CALL( cuptiProfilerPushRangePtr(&pushRangeParams), goto fn_fail_misc );

    }
    state->read_count++;
fn_exit:
    CUDA_CALL( cuCtxSetCurrentPtr(userCtx), );
    return papi_errno;
fn_fail_misc:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int cuptip_control_reset(cuptip_control_t state)
{
    COMPDBG("Entering.\n");
    state->read_count = 0;
    return PAPI_OK;
}

int cuptip_shutdown(void)
{
    COMPDBG("Entering.\n");
    free_all_enumerated_metrics();
    finalize_cupti_profiler_api();
    unload_nvpw_sym();
    unload_cupti_perf_sym();
    return PAPI_OK;
}
