/**
 * @file    linux-cuda.c
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 08/2019, to make counters accumulate.)
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 2018, to use batch reads and support nvlink metrics.
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief This implements a PAPI component that enables PAPI-C to
 *  access hardware monitoring counters for NVIDIA CUDA GPU devices
 *  through the CUPTI library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

//-----------------------------------------------------------------------------
// A basic assumption here (and in other components) is that we put as much of
// the computational load of this component into the initialization stage and
// the "adding" stage for events (update_control), because users are likely not
// measuring performance at those times, but may well be reading these events
// when performance matters. So we want the read operation lightweight, but we
// can remember tables and such at startup and when servicing a PAPI_add().
//-----------------------------------------------------------------------------

#include <dlfcn.h>
#include <limits.h>
#include <float.h> // For DBL_MAX. 
#include <sys/stat.h>
#include <dirent.h>

// NOTE: We can't use extended directories; these include files have includes.
#include <cupti.h>
#include <cuda_runtime_api.h>

#include <cuda.h>

#if CUPTI_API_VERSION >= 13
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#endif 

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

// We use a define so we can use it as a static array dimension. 
// This winds up defining the maximum number of events in a PAPI eventset.
// Increase as needed, but 512 is probably sufficient!
#define PAPICUDA_MAX_COUNTERS 512

// Hash Table Size. Note in Cupti-11 we can have ~120,000 events per device.
// This causes a performance problem in looking up events by name; especially
// if we have 2 or more devices. This hash eliminates that issue.
// 32768; 233K events hashed: inUse=99.927%, avgChain=7.02, maxChain=20.
#define CUDA11_HASH_SIZE 32768

// Will include code to time the multi-pass metric elimination code;
// Only applies to Legacy CUpti, not CUDA11. In CUDA11 we do not eliminate 
// multi-pass events.
// #define TIME_MULTIPASS_ELIM

// CUDA metrics use events that are not "exposed" by the normal enumeration of
// performance events; they are only identified by an event number (no name, 
// no description). But we have to track them for accumulating purposes, they
// get zeroed after a read too. This #define will add code that produces a
// full event report, showing the composition of metrics, etc. This is a 
// diagnostic aid, it is possible these unenumerated events will vary by 
// GPU version or model, should some metric seem to be misbehaving.
// Only applies to Legacy CUpti, not CUDA11.
// #define PRODUCE_EVENT_REPORT

// For the same unenumerated events, for experimentation and diagnostics, this
// #define will add all the unumerated events as PAPI events that can be
// queried individually.  The "event" will show up as something like
// "cuda:::unenum_event:0x16000001:device=0", with a description "Unenumerated
// Event used in a metric". But this allows you to add it to a PAPI_EventSet
// and see how it behaves under different test kernels. 
// Only applies to Legacy CUpti, not CUDA11.
// #define EXPOSE_UNENUMERATED_EVENTS 

// An experimental alternative.
// #define PAPICUDA_KERNEL_REPLAY_MODE

// CUDA metrics can require events that do not appear in the 
// enumerated event lists. A table of these tracks these for
// cumulative valuing (necessary because a read of any counter
// zeros it).

// Some structures apply only to Legacy CUpti, not CUDA11.
// See below for cuda11 structures.
typedef struct cuda_all_events {
   CUpti_EventID  eventId;
   int            deviceNum;
   int            idx;              // -1 if unenumerated, otherwise idx into enumerated events.
   int            nonCumulative;    // 0=cumulative. 1=do not; spot value or constant. 
   long unsigned int cumulativeValue;
} cuda_all_events_t;

/* Store the name and description for an event */
typedef struct cuda_name_desc {
    char        name[PAPI_MAX_STR_LEN];
    char        description[PAPI_2MAX_STR_LEN];
    uint16_t        numMetricEvents;        // 0=event, if a metric, size of metricEvents array below.
    CUpti_EventID   *metricEvents;          // NULL for cuda events, an array of member events if a metric.
    CUpti_MetricValueKind MV_Kind;          // eg. % or counter or rate, etc. Needed to compute metric from individual events.
} cuda_name_desc_t;

/* For a device, store device description */
typedef struct cuda_device_desc {
    CUdevice    cuDev;
    int         deviceNum;
    char        deviceName[PAPI_MIN_STR_LEN];
    int         CC_Major;                   /* Compute Capability Major */
    int         CC_Minor;
    uint32_t    maxDomains;                 /* number of domains per device */
    CUpti_EventDomainID *domainIDArray;     /* Array[maxDomains] of domain IDs */
    uint32_t    *domainIDNumEvents;         /* Array[maxDomains] of num of events in that domain */

    int         cc_le70;                 /* <= 7.0 can use legacy. */
    int         cc_ge70;                 /* >= 7.0 can use profiler. */

#if CUPTI_API_VERSION >= 13
    CUcontext   cuContext;                      // context created during cuda11_add_native_events.
    CUcontext   sessionCtx;                     // context created for profiling session.
    char        cuda11_chipName[PAPI_MIN_STR_LEN];
    uint8_t*    cuda11_ConfigImage;             // Part 1 of an 'eventset' for NV PerfWorks.
    int         cuda11_ConfigImageSize;
    uint8_t*    cuda11_CounterDataPrefixImage;  // Part 2 of an 'eventset' for NV PerfWorks.
    int         cuda11_CounterDataPrefixImageSize;
    uint8_t*    cuda11_CounterDataImage;        // actual data from an 'eventset' for NV PerfWorks.
    int         cuda11_CounterDataImageSize;
    uint8_t*    cuda11_CounterDataScratchBuffer;
    int         cuda11_CounterDataScratchBufferSize;
    char        cuda11_range_name[32];          // Name of the only range we have.

    // Parameters init and used in _cuda11_start().
    int  cuda11_RMR_count;
    NVPA_RawMetricRequest *cuda11_RMR;
    int  cuda11_numMetricNames;
    int  *cuda11_ValueIdx;
    int  *cuda11_MetricIdx;
    char **cuda11_MetricNames;  
    CUpti_Profiler_BeginSession_Params beginSessionParams;
    CUpti_Profiler_SetConfig_Params setConfigParams;
    CUpti_Profiler_PushRange_Params pushRangeParams;
    NVPW_CUDA_MetricsContext_Create_Params *pMetricsContextCreateParams;
    int ownsMetricsContext;   

    // Parameters init in _cuda11_build_profiling_structures and used
    // in every _cuda11_read to reset the counter images (buffers).
    CUpti_Profiler_CounterDataImageOptions                         cuda11_CounterDataOptions;
    CUpti_Profiler_CounterDataImage_Initialize_Params              cuda11_CounterDataInitParms;
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params cuda11_CounterScratchInitParms;
#endif 
} cuda_device_desc_t;

// Contains device list, pointer to device description, and the list of all available events.
typedef struct cuda_context {
    int         deviceCount;
    cuda_device_desc_t *deviceArray;
    uint32_t    availEventSize;
    CUpti_ActivityKind *availEventKind;
    int         *availEventDeviceNum;
    uint32_t    *availEventIDArray;
    uint32_t    *availEventIsBeingMeasuredInEventset;
    cuda_name_desc_t *availEventDesc;
    uint32_t    numAllEvents;
    cuda_all_events_t *allEvents;
} cuda_context_t;

// For each active cuda context (one measuring something) we also track the
// cuda device number it is on. We track in separate arrays for each reading
// method.  cuda metrics and nvlink metrics require multiple events to be read,
// these are then arithmetically combined to produce the metric value. The
// allEvents array stores all the actual events; i.e. metrics are decomposed 
// to their individual events and both enumerated and metric events are stored,
// so we can perform an analysis of how to read with cuptiEventGroupSetsCreate().

typedef struct cuda_active_cucontext_s {
    CUcontext cuCtx;
    int deviceNum;

    uint32_t      ctxActiveCount;                               // Count of entries in ctxActiveEvents.
    uint32_t      ctxActiveEvents    [PAPICUDA_MAX_COUNTERS];   // index into gctrl->activeEventXXXX arrays, so we can store values.

    uint32_t      allEventsCount;                               // entries in allEvents array.
    CUpti_EventID allEvents          [PAPICUDA_MAX_COUNTERS];   // allEvents, including sub-events of metrics. (no metric Ids in here).
    long unsigned int allEventValues     [PAPICUDA_MAX_COUNTERS];   // aggregated event values.

    CUpti_EventGroupSets *eventGroupSets;                       // Built during add, to save time not doing it at read.
} cuda_active_cucontext_t;

// Control structure tracks array of active contexts and active events
// in the order the user requested them; along with associated values
// values and types (to save lookup time).
typedef struct cuda_control {
    uint32_t    countOfActiveCUContexts;
    cuda_active_cucontext_t *arrayOfActiveCUContexts[PAPICUDA_MAX_COUNTERS];
    uint32_t    activeEventCount;
    int         activeEventIndex            [PAPICUDA_MAX_COUNTERS];    // index into gctxt->availEventXXXXX arrays.
    long long   activeEventValues           [PAPICUDA_MAX_COUNTERS];    // values we will return.
    CUpti_MetricValueKind activeEventKind   [PAPICUDA_MAX_COUNTERS];    // For metrics: double, uint64, % or throughput. Needed to compute metric from individual events.
    uint64_t    cuptiStartTimestampNs;                                  // needed to compute duration for some metrics.
    uint64_t    cuptiReadTimestampNs;                                   // ..
} cuda_control_t;

#if CUPTI_API_VERSION >= 13
//*****************************************************************************
// CUDA 11 structures.
//*****************************************************************************
enum {SpotValue, RunningMin, RunningMax, RunningSum};
typedef struct {
    int     deviceNum;                      // idx to gctxt->deviceArray[].
    char*   nv_name;                        // The nvidia name.
    char*   papi_name;                      // The papi name (with :device=i). PAPI_MAX_STR_LEN.
    int     detailsDone;                    // If details are already done.
    char*   description;                    
    char*   dimUnits;                    
    double  gpuBurstRate;
    double  gpuSustainedRate;
    int     passes;
    int     inEventSet;
    int     numRawMetrics;
    int     treatment;              // see enum above; SpotValue, etc.
    NVPA_RawMetricRequest* rawMetricRequests;
    double  cumulativeValue;        // cumulativeValue. 
} cuda11_eventData;

// Hash Table Entry; to look up by name.
typedef struct cuda11_hash_entry_s {
   int idx;                                        // The entry that matches this hash.
   void *next;                                     // next entry that matches this hash, or NULL.
} cuda11_hash_entry_t;
#endif

// file handles used to access cuda libraries with dlopen
static void *dl1 = NULL;
static void *dl2 = NULL;
static void *dl3 = NULL;

#if CUPTI_API_VERSION >= 13
static void *dl4 = NULL;
#endif

static char cuda_main[]=PAPI_CUDA_MAIN;
static char cuda_runtime[]=PAPI_CUDA_RUNTIME;
static char cuda_cupti[]=PAPI_CUDA_CUPTI;
#if CUPTI_API_VERSION >= 13
static char cuda_perfworks[]=PAPI_CUDA_PERFWORKS;
#endif

static int cuda_version=0;
static int cuda_runtime_version=0;
static uint32_t cupti_runtime_version=0;

#if CUPTI_API_VERSION >= 13
// The following structure sizes change from version 10 to version 11.
static int GetChipName_Params_STRUCT_SIZE=0;
static int Profiler_SetConfig_Params_STRUCT_SIZE=0;
static int Profiler_EndPass_Params_STRUCT_SIZE=0;
static int Profiler_FlushCounterData_Params_STRUCT_SIZE=0;
#endif

/* The PAPI side (external) variable as a global */
papi_vector_t _cuda_vector;

/* Global variable for hardware description, event and metric lists */
static cuda_context_t *global_cuda_context = NULL;

/* This global variable points to the head of the control state list */
static cuda_control_t *global_cuda_control = NULL;


#if CUPTI_API_VERSION >= 13
// This global variable tracks all cuda11 metrics.
static int cuda11_numEvents = 0;       // actual number of events in array.
static int cuda11_maxEvents = 0;       // allocated space for events in array.
static cuda11_eventData** cuda11_AllEvents;
static cuda11_hash_entry_t* cuda11_NameHashTable[CUDA11_HASH_SIZE];

// prototypes for cuda->cuda11 hand offs.
static int _cuda11_init_control_state(hwd_control_state_t * ctrl);
static int _cuda11_update_control_state(hwd_control_state_t * ctrl,
    NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx);
static int _cuda11_ntv_enum_events(unsigned int *EventCode, int modifier);
static int _cuda11_ntv_name_to_code(const char *nameIn, unsigned int *out);
static int _cuda11_ntv_code_to_name(unsigned int EventCode, char *name, int len);

static int _cuda11_add_native_events(cuda_context_t * gctxt);
static void _cuda11_cuda_vector(void);
#endif

#define DEBUG_CALLS 0

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

#define CHECK_PRINT_EVAL( checkcond, str, evalthis )                        \
    do {                                                                    \
        int _cond = (checkcond);                                            \
        if (_cond) {                                                        \
            SUBDBG("error: condition %s failed: %s.\n", #checkcond, str);   \
            evalthis;                                                       \
        }                                                                   \
    } while (0)

#define CUDA_CALL( call, handleerror )                                                          \
    do {                                                                                        \
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CUDA_CALL %s\n",                             \
             __FILE__, __func__, __LINE__, #call);                                              \
        cudaError_t _status = (call);                                                           \
        if (_status != cudaSuccess) {                                                           \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status);               \
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CUDA error: function %s failed error %d.\n", \
                __FILE__, __func__, __LINE__, #call, _status);                                  \
            handleerror;                                                                        \
        }                                                                                       \
    } while (0)

#define CU_CALL( call, handleerror )                                                                \
    do {                                                                                            \
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CU_CALL %s\n",                                   \
            __FILE__, __func__, __LINE__, #call);                                                   \
        CUresult _status = (call);                                                                  \
        if (_status != CUDA_SUCCESS) {                                                              \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status);                   \
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CU error: function %s failed with error %d.\n", \
                    __FILE__, __func__, __LINE__, #call, _status);                                  \
            {handleerror;}                                                                          \
        }                                                                                           \
    } while (0)


#define CUPTI_CALL(call, handleerror)                                                               \
    do {                                                                                            \
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CUPTI_CALL %s\n",                                \
            __FILE__, __func__, __LINE__, #call);                                                   \
        CUptiResult _status = (call);                                                               \
        if (_status != CUPTI_SUCCESS) {                                                             \
            const char *errstr;                                                                     \
            (*cuptiGetResultStringPtr)(_status, &errstr);                                           \
            SUBDBG("error: function %s failed with error %s.\n", #call, errstr);                    \
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i CUpti error: function %s failed with error %d (%s).\n", \
                    __FILE__, __func__, __LINE__, #call, _status, errstr);                          \
            {handleerror;}                                                                          \
        }                                                                                           \
    } while (0)

#if CUPTI_API_VERSION >= 13
#define NVPW_CALL(call, handleerror)                                                                \
    do {                                                                                            \
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i NVPW_CALL %s\n",                                 \
            __FILE__, __func__, __LINE__, #call);                                                   \
        NVPA_Status _status = (call);                                                               \
        if (_status != NVPA_STATUS_SUCCESS) {                                                       \
            SUBDBG("error: NVPW function %s failed with error %d.\n", #call, _status);              \
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i PerfWork error: function %s failed with error %d.\n",         \
                    __FILE__, __func__, __LINE__, #call, _status);                                  \
            {handleerror;}                                                                          \
        }                                                                                           \
    } while (0)
#endif

#define BUF_SIZE (32 * PATH_MAX)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

/* Function prototypes */

// Sort into ascending order, by eventID and device.
static int ascAllEvents(const void *A, const void *B) { 
    cuda_all_events_t *a = (cuda_all_events_t*) A;
    cuda_all_events_t *b = (cuda_all_events_t*) B;
    if (a->eventId < b->eventId) return(-1);
    if (a->eventId > b->eventId) return( 1);
    if (a->deviceNum < b->deviceNum) return(-1);
    if (a->deviceNum > b->deviceNum) return( 1);
    return(0);
}

static int _cuda_cleanup_eventset(hwd_control_state_t * ctrl);

/* ******  CHANGE PROTOTYPES TO DECLARE CUDA LIBRARY SYMBOLS AS WEAK  **********
 *  This is done so that a version of PAPI built with the cuda component can   *
 *  be installed on a system which does not have the cuda libraries installed. *
 *                                                                             *
 *  If this is done without these prototypes, then all papi services on the    *
 *  system without the cuda libraries installed will fail.  The PAPI libraries *
 *  contain references to the cuda libraries which are not installed.  The     *
 *  load of PAPI commands fails because the cuda library references can not be *
 *  resolved.                                                                  *
 *                                                                             *
 *  This also defines pointers to the cuda library functions that we call.     *
 *  These function pointers will be resolved with dlopen/dlsym calls at        *
 *  component initialization time.  The component then calls the cuda library  *
 *  functions through these function pointers.                                 *
 *******************************************************************************/
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  static CUresult( *funcname##Ptr ) funcsig;
DECLARECUFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARECUFUNC(cuCtxSetCurrent, (CUcontext));
DECLARECUFUNC(cuCtxDestroy, (CUcontext));
DECLARECUFUNC(cuCtxCreate, (CUcontext *pctx, unsigned int flags, CUdevice dev));
DECLARECUFUNC(cuCtxGetDevice, (CUdevice *));
DECLARECUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUFUNC(cuDeviceGetCount, (int *));
DECLARECUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUFUNC(cuDevicePrimaryCtxRetain, (CUcontext *pctx, CUdevice));
DECLARECUFUNC(cuDevicePrimaryCtxRelease, (CUdevice));
DECLARECUFUNC(cuInit, (unsigned int));
DECLARECUFUNC(cuGetErrorString, (CUresult error, const char** pStr));
DECLARECUFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARECUFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARECUFUNC(cuCtxSynchronize, ());
DECLARECUFUNC(cuDeviceGetAttribute, (int *, CUdevice_attribute, CUdevice));

#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARECUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  static cudaError_t( *funcname##Ptr ) funcsig;
DECLARECUDAFUNC(cudaGetDevice, (int *));
DECLARECUDAFUNC(cudaSetDevice, (int));
// DECLARECUDAFUNC(cudaGetDeviceProperties, (struct cudaDeviceProp* prop, int  device));
DECLARECUDAFUNC(cudaDeviceGetAttribute, (int *value, enum cudaDeviceAttr attr, int device));
DECLARECUDAFUNC(cudaFree, (void *));
DECLARECUDAFUNC(cudaDriverGetVersion, (int *));
DECLARECUDAFUNC(cudaRuntimeGetVersion, (int *));

#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARECUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  static CUptiResult( *funcname##Ptr ) funcsig;
/* CUptiResult CUPTIAPIWEAK cuptiDeviceEnumEventDomains( CUdevice, size_t *, CUpti_EventDomainID * ); */
/* CUptiResult( *cuptiDeviceEnumEventDomainsPtr )( CUdevice, size_t *, CUpti_EventDomainID * ); */
DECLARECUPTIFUNC(cuptiDeviceEnumMetrics, (CUdevice device, size_t * arraySizeBytes, CUpti_MetricID * metricArray));
DECLARECUPTIFUNC(cuptiDeviceGetEventDomainAttribute, (CUdevice device, CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiDeviceGetNumMetrics, (CUdevice device, uint32_t * numMetrics));
DECLARECUPTIFUNC(cuptiEventGroupGetAttribute, (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiEventGroupReadEvent, (CUpti_EventGroup eventGroup, CUpti_ReadEventFlags flags, CUpti_EventID event, size_t * eventValueBufferSizeBytes, uint64_t * eventValueBuffer));
DECLARECUPTIFUNC(cuptiEventGroupSetAttribute, (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t valueSize, void *value));
DECLARECUPTIFUNC(cuptiEventGroupSetDisable, (CUpti_EventGroupSet * eventGroupSet));
DECLARECUPTIFUNC(cuptiEventGroupSetEnable, (CUpti_EventGroupSet * eventGroupSet));
DECLARECUPTIFUNC(cuptiEventGroupSetsCreate, (CUcontext context, size_t eventIdArraySizeBytes, CUpti_EventID * eventIdArray, CUpti_EventGroupSets ** eventGroupPasses));
DECLARECUPTIFUNC(cuptiMetricCreateEventGroupSets, (CUcontext context, size_t metricIdArraySizeBytes, CUpti_MetricID * metricIdArray, CUpti_EventGroupSets ** eventGroupPasses));
DECLARECUPTIFUNC(cuptiEventGroupSetsDestroy, (CUpti_EventGroupSets * eventGroupSets));
DECLARECUPTIFUNC(cuptiMetricGetRequiredEventGroupSets, (CUcontext ctx, CUpti_MetricID metricId, CUpti_EventGroupSets **thisEventGroupSet));
DECLARECUPTIFUNC(cuptiGetTimestamp, (uint64_t * timestamp));
DECLARECUPTIFUNC(cuptiGetVersion, (uint32_t * version));
DECLARECUPTIFUNC(cuptiMetricEnumEvents, (CUpti_MetricID metric, size_t * eventIdArraySizeBytes, CUpti_EventID * eventIdArray));
DECLARECUPTIFUNC(cuptiMetricGetAttribute, (CUpti_MetricID metric, CUpti_MetricAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiMetricGetNumEvents, (CUpti_MetricID metric, uint32_t * numEvents));
DECLARECUPTIFUNC(cuptiMetricGetValue, (CUdevice device, CUpti_MetricID metric, size_t eventIdArraySizeBytes, CUpti_EventID * eventIdArray, size_t eventValueArraySizeBytes, uint64_t * eventValueArray, uint64_t timeDuration, CUpti_MetricValue * metricValue));
DECLARECUPTIFUNC(cuptiSetEventCollectionMode, (CUcontext context, CUpti_EventCollectionMode mode));
DECLARECUPTIFUNC(cuptiDeviceEnumEventDomains, (CUdevice, size_t *, CUpti_EventDomainID *));
DECLARECUPTIFUNC(cuptiDeviceGetNumEventDomains, (CUdevice, uint32_t *));
DECLARECUPTIFUNC(cuptiEventDomainEnumEvents, (CUpti_EventDomainID, size_t *, CUpti_EventID *));
DECLARECUPTIFUNC(cuptiEventDomainGetAttribute, (CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiEventDomainGetNumEvents, (CUpti_EventDomainID, uint32_t *));
DECLARECUPTIFUNC(cuptiEventGetAttribute, (CUpti_EventID, CUpti_EventAttribute, size_t *, void *));
DECLARECUPTIFUNC(cuptiEventGroupAddEvent, (CUpti_EventGroup, CUpti_EventID));
DECLARECUPTIFUNC(cuptiEventGroupCreate, (CUcontext, CUpti_EventGroup *, uint32_t));
DECLARECUPTIFUNC(cuptiEventGroupDestroy, (CUpti_EventGroup));
DECLARECUPTIFUNC(cuptiEventGroupDisable, (CUpti_EventGroup));
DECLARECUPTIFUNC(cuptiEventGroupEnable, (CUpti_EventGroup));
DECLARECUPTIFUNC(cuptiEventGroupReadAllEvents, (CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *, size_t *));
DECLARECUPTIFUNC(cuptiEventGroupResetAllEvents, (CUpti_EventGroup));
DECLARECUPTIFUNC(cuptiGetResultString, (CUptiResult result, const char **str));
DECLARECUPTIFUNC(cuptiEnableKernelReplayMode, ( CUcontext context ));
DECLARECUPTIFUNC(cuptiDisableKernelReplayMode, ( CUcontext context ));

#if CUPTI_API_VERSION >= 13
// Functions for perfworks profiler.
// cuptiDeviceGetChipName relies on cupti_target.h, not in legacy Cuda distributions.
DECLARECUPTIFUNC(cuptiDeviceGetChipName, (CUpti_Device_GetChipName_Params* params));
DECLARECUPTIFUNC(cuptiProfilerInitialize, (CUpti_Profiler_Initialize_Params* params));
DECLARECUPTIFUNC(cuptiProfilerDeInitialize, (CUpti_Profiler_DeInitialize_Params* params));
DECLARECUPTIFUNC(cuptiProfilerCounterDataImageCalculateSize, (CUpti_Profiler_CounterDataImage_CalculateSize_Params* params));
DECLARECUPTIFUNC(cuptiProfilerCounterDataImageInitialize, (CUpti_Profiler_CounterDataImage_Initialize_Params* params));
DECLARECUPTIFUNC(cuptiProfilerCounterDataImageCalculateScratchBufferSize, (CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params));
DECLARECUPTIFUNC(cuptiProfilerCounterDataImageInitializeScratchBuffer, (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params));

DECLARECUPTIFUNC(cuptiProfilerBeginSession, (CUpti_Profiler_BeginSession_Params* params));
DECLARECUPTIFUNC(cuptiProfilerSetConfig, (CUpti_Profiler_SetConfig_Params* params));
DECLARECUPTIFUNC(cuptiProfilerBeginPass, (CUpti_Profiler_BeginPass_Params* params));
DECLARECUPTIFUNC(cuptiProfilerEnableProfiling, (CUpti_Profiler_EnableProfiling_Params* params));
DECLARECUPTIFUNC(cuptiProfilerPushRange, (CUpti_Profiler_PushRange_Params* params));
DECLARECUPTIFUNC(cuptiProfilerPopRange, (CUpti_Profiler_PopRange_Params* params));
DECLARECUPTIFUNC(cuptiProfilerDisableProfiling, (CUpti_Profiler_DisableProfiling_Params* params));
DECLARECUPTIFUNC(cuptiProfilerEndPass, (CUpti_Profiler_EndPass_Params* params));
DECLARECUPTIFUNC(cuptiProfilerFlushCounterData, (CUpti_Profiler_FlushCounterData_Params* params));
DECLARECUPTIFUNC(cuptiProfilerUnsetConfig, (CUpti_Profiler_UnsetConfig_Params* params));
DECLARECUPTIFUNC(cuptiProfilerEndSession, (CUpti_Profiler_EndSession_Params* params));
DECLARECUPTIFUNC(cuptiProfilerGetCounterAvailability, (CUpti_Profiler_GetCounterAvailability_Params* params));

#define NVPWAPIWEAK __attribute__( ( weak ) )
#define DECLARENVPWFUNC(fname, fsig) NVPA_Status NVPWAPIWEAK fname fsig; static NVPA_Status( *fname##Ptr ) fsig;

DECLARENVPWFUNC(NVPW_GetSupportedChipNames, (NVPW_GetSupportedChipNames_Params* params));
DECLARENVPWFUNC(NVPW_CUDA_MetricsContext_Create, (NVPW_CUDA_MetricsContext_Create_Params* params));
DECLARENVPWFUNC(NVPW_MetricsContext_Destroy, (NVPW_MetricsContext_Destroy_Params * params));
DECLARENVPWFUNC(NVPW_MetricsContext_GetMetricNames_Begin, (NVPW_MetricsContext_GetMetricNames_Begin_Params* params));
DECLARENVPWFUNC(NVPW_MetricsContext_GetMetricNames_End, (NVPW_MetricsContext_GetMetricNames_End_Params* params));
DECLARENVPWFUNC(NVPW_InitializeHost, (NVPW_InitializeHost_Params* params));
DECLARENVPWFUNC(NVPW_MetricsContext_GetMetricProperties_Begin, (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p));
DECLARENVPWFUNC(NVPW_MetricsContext_GetMetricProperties_End, (NVPW_MetricsContext_GetMetricProperties_End_Params* p));
DECLARENVPWFUNC(NVPW_CUDA_RawMetricsConfig_Create, (NVPW_CUDA_RawMetricsConfig_Create_Params*));

// Already defined in nvperf_host.h. I don't use these, I just need the pointer.
// DECLARENVPWFUNC(NVPA_RawMetricsConfig_Create, (NVPA_RawMetricsConfigOptions*, NVPA_RawMetricsConfig**));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_Destroy, (NVPW_RawMetricsConfig_Destroy_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_BeginPassGroup, (NVPW_RawMetricsConfig_BeginPassGroup_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_EndPassGroup, (NVPW_RawMetricsConfig_EndPassGroup_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_AddMetrics, (NVPW_RawMetricsConfig_AddMetrics_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_GenerateConfigImage, (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_GetConfigImage, (NVPW_RawMetricsConfig_GetConfigImage_Params* params));
DECLARENVPWFUNC(NVPW_CounterDataBuilder_Create, (NVPW_CounterDataBuilder_Create_Params* params));
DECLARENVPWFUNC(NVPW_CounterDataBuilder_Destroy, (NVPW_CounterDataBuilder_Destroy_Params* params));
DECLARENVPWFUNC(NVPW_CounterDataBuilder_AddMetrics, (NVPW_CounterDataBuilder_AddMetrics_Params* params));
DECLARENVPWFUNC(NVPW_CounterDataBuilder_GetCounterDataPrefix, (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params));
DECLARENVPWFUNC(NVPW_CounterData_GetNumRanges, (NVPW_CounterData_GetNumRanges_Params* params));
DECLARENVPWFUNC(NVPW_Profiler_CounterData_GetRangeDescriptions, (NVPW_Profiler_CounterData_GetRangeDescriptions_Params* params));
DECLARENVPWFUNC(NVPW_MetricsContext_SetCounterData, (NVPW_MetricsContext_SetCounterData_Params* params));
DECLARENVPWFUNC(NVPW_MetricsContext_EvaluateToGpuValues, (NVPW_MetricsContext_EvaluateToGpuValues_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_GetNumPasses, (NVPW_RawMetricsConfig_GetNumPasses_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_SetCounterAvailability, (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params));
DECLARENVPWFUNC(NVPW_RawMetricsConfig_IsAddMetricsPossible, (NVPW_RawMetricsConfig_IsAddMetricsPossible_Params* params));

DECLARENVPWFUNC(NVPW_MetricsContext_GetCounterNames_Begin, (NVPW_MetricsContext_GetCounterNames_Begin_Params* pParams));
DECLARENVPWFUNC(NVPW_MetricsContext_GetCounterNames_End, (NVPW_MetricsContext_GetCounterNames_End_Params* pParams));
#endif

/*****************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 *****************************************************************************/

//-----------------------------------------------------------------------------
// This function returns the number of Nvidia devices in the system.
// We search the file system for /sys/class/drm/card?/device/vendor. These must
// be card0, card1, etc. When we cannot open a file we stop looking. If they
// can be opened and return a line it will be a string 0xhhhh as a hex vendor
// ID. See the website  https://pci-ids.ucw.cz, particularly 
// https://pci-ids.ucw.cz/read/PC and  https://pci-ids.ucw.cz/read/PD/
// for a list. 0x10de is the vendor ID for Nvidia.
// The /sys/class/drm/card?/device/class, if present, must begin 0x03. This 
// indicates a "Display Controller" (i.e. GPU) but on Nvidia we have seen both
//  0x030200 and 0x030000, so we only match the beginning.
// 
// If your devices are not found; double check what the system is saying
// manually; e.g. 
// >cat /sys/class/drm/card0/device/vendor
// >cat /sys/class/drm/card0/device/class
// 
// Note we DO have cuDeviceGetCount(), but this requires cuInit() to be run;
// and we don't want to do that before delayed_init. It causes problems for 
// higher level tool vendors that use PAPI underneath. Without cuInit(); I 
// know of no other way to check for Nvidia GPUs present in the system.
//-----------------------------------------------------------------------------
static int _cuda_count_dev_sys(void)
{
    char vendor_id[64]="/sys/class/drm/card%i/device/vendor";
    char class_id[64]="/sys/class/drm/card%i/device/class";
    char filename[64];
    uint32_t myVendor = 0x10de;                     // The NVIDIA GPU vendor ID.
    char line[16];
    size_t bytes;
    int card;
    long int devID;

    int totalDevices=0;                             // Reset, in case called more than once.

    for (card=0; card<64; card++) {
        sprintf(filename, vendor_id, card);         // make a name for myself.
        FILE *fcard = fopen(filename, "r");         // Open for reading.
        if (fcard == NULL) {                        // Failed to open,
            break;
        }

        bytes=fread(line, 1, 6, fcard);             // read six bytes.
        fclose(fcard);                              // Always close it (avoid mem leak).
        if (bytes != 6) {                           // If we did not read 6,
            continue;                               // skip this one, vendor id is malformed.
        }

        line[bytes]=0;                              // Ensure null termination.
        devID = strtol(line, NULL, 16);             // convert base 16 to long int. Handles '0xhhhh'. NULL=Don't need 'endPtr'.
        if (devID != myVendor) continue;            // Not the droid I am looking for.

        // Right vendor. Look for some class.
        sprintf(filename, class_id, card);          // make a name for myself.
        fcard = fopen(filename, "r");               // Open for reading.
        if (fcard == NULL) {                        // Failed to open,
            continue;                               // skip this one if no class file found.
        }

        // expecting 8 bytes for class; but some have a '0xa0' at the end and read nine bytes.
        // e.g. '0x030200'. So I read 4, I only care if it starts with '0x03'; a Display Controller.
        bytes=fread(line, 1, 4, fcard);             // read 1 byte x 4.
        fclose(fcard);                              // Always close it (avoid mem leak).
        if (bytes < 4) {                            // If we did not read enough to match,
            continue;                               // skip this one if class text is too short.
        }

        line[bytes]=0;                              // Ensure null termination.
        if (strncasecmp("0x03", line, 4) != 0) continue;    // Not a Display Controller.

        // Found one.
        totalDevices++;                             // count it.
    } // end loop through possible cards.

    return(totalDevices);
} // end __cuda_count_dev_sys

static int _cuda_count_dev_proc(void)
{
    const char *proc_dir = "/proc/driver/nvidia/gpus";

    struct stat proc_stat;
    int err = stat(proc_dir, &proc_stat);
    if (err) {
        return 0;
    }

    DIR *dir = opendir(proc_dir);
    if (dir == NULL) {
        return 0;
    }

    int count = 0;
    struct dirent *dentry;
    while ((dentry = readdir(dir)) != NULL) {
        if (dentry->d_name[0] == '.') {
            continue;
        }
        ++count;
    }

    closedir(dir);

    return count;
}

static int _cuda_init_private(void);

/*
 * Check for the initialization step and does it if needed
 */
static int
_cuda_check_n_initialize(papi_vector_t *vector)
{
  if (!vector->cmp_info.initialized) {
      return _cuda_init_private();
  }
  return PAPI_OK;
}

#define DO_SOME_CHECKING(vectorp) do {           \
  int err = _cuda_check_n_initialize(vectorp);   \
  if (PAPI_OK != err) return err;                \
} while(0)

//-----------------------------------------------------------------------------
// Binary Search of gctxt->allEvents[gctxt->numAllEvents]. Returns idx, or -1.
// No thread locks, the event array is static after init_component().
//-----------------------------------------------------------------------------
static int _search_all_events(cuda_context_t * gctxt, CUpti_EventID id, int device) {
    cuda_all_events_t *list =gctxt->allEvents; 
    int lower=0, upper=gctxt->numAllEvents-1, middle=(lower+upper)/2;

    while (lower <= upper) {
        if (list[middle].eventId < id || 
            (list[middle].eventId == id && list[middle].deviceNum < device)) {
            lower = middle+1;
        } else {
            if (list[middle].eventId > id || 
                (list[middle].eventId == id && list[middle].deviceNum > device)) {
                upper = middle -1;
            } else {
                if (list[middle].eventId == id && list[middle].deviceNum == device) {
                    return(middle);
                }
            }
        }

        // We have a new lower or new upper.
        if (lower > upper) break;
        middle = (lower+upper)/2;
    }

    return(-1);
} // end search_all_events.


/*
 * Link the necessary CUDA libraries to use the cuda component.  If any of them can not be found, then
 * the CUDA component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the CUDA component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */

#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name );                \
    if ( dlerror()!=NULL ) {                                                \
        int strErr;                                                         \
        strErr = snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,   \
        "A CUDA required function '%s' was not found in lib '%s'.",         \
        name, #dllib);                                                      \
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;        \
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;                 \
        return ( PAPI_ENOSUPP );                                            \
    }
#define DLSYM_AND_CHECK_nvperf( dllib, name ) dlsym( dllib, name );         \
    if ( dlerror()!=NULL ) {                                                \
        snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,   \
        "A required function '%s' was not found in '%s'.",                  \
        name, nvperf_info.dli_fname);                                       \
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;        \
        return ( PAPI_ENOSUPP );                                            \
    }


/*
 * Link the necessary CUDA libraries to use the cuda component.  If any of them can not be found, then
 * the CUDA component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the CUDA component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */
static int _cuda_linkCudaLibraries(void)
{
    char path_lib[PATH_MAX];

    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        char* strCpy;
        strCpy=strncpy(_cuda_vector.cmp_info.disabled_reason, "The CUDA component does not support statically linking to libc.", PAPI_MAX_STR_LEN-1);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        return PAPI_ENOSUPP;
    }
    // Need to link in the cuda libraries, if any not found disable the component
    // getenv returns NULL if environment variable is not found.
    char *cuda_root = getenv("PAPI_CUDA_ROOT");

    dl1 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(cuda_main) > 0) {                                // If override given, it has to work.
        int strErr;
        dl1 = dlopen(cuda_main, RTLD_NOW | RTLD_GLOBAL);        // Try to open that path.
        if (dl1 == NULL) {
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_CUDA_MAIN override '%s' given in Rules.cuda not found.", cuda_main);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl1 == NULL) {                                          // No override,
        dl1 = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);     // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl1 == NULL && cuda_root != NULL) {                          // if root given, try it.
        int strErr=snprintf(path_lib, sizeof(path_lib)-2, "%s/lib64/libcuda.so", cuda_root);  // PAPI Root check.
        path_lib[sizeof(path_lib)-1]=0;
        if (strErr > (int) sizeof(path_lib)-2) HANDLE_STRING_ERROR;
        dl1 = dlopen(path_lib, RTLD_NOW | RTLD_GLOBAL);              // Try to open that path.
    }

    // Check for failure.
    if (dl1 == NULL) {
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libcuda.so not found.");
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);
    }

    // We have a dl1. (libcuda.so).

    cuCtxGetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxGetCurrent");
    cuCtxSetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxSetCurrent");
    cuDeviceGetPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr = DLSYM_AND_CHECK(dl1, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = DLSYM_AND_CHECK(dl1, "cuDevicePrimaryCtxRelease");
    cuInitPtr = DLSYM_AND_CHECK(dl1, "cuInit");
    cuGetErrorStringPtr = DLSYM_AND_CHECK(dl1, "cuGetErrorString");
    cuCtxPopCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPushCurrent");
    cuCtxDestroyPtr = DLSYM_AND_CHECK(dl1, "cuCtxDestroy");
    cuCtxCreatePtr  = DLSYM_AND_CHECK(dl1, "cuCtxCreate");
    cuCtxGetDevicePtr = DLSYM_AND_CHECK(dl1, "cuCtxGetDevice");
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl1, "cuCtxSynchronize");
    cuDeviceGetAttributePtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetAttribute");
    cuDevicePrimaryCtxRetainPtr = DLSYM_AND_CHECK(dl1, "cuDevicePrimaryCtxRetain");


    /* Need to link in the cuda runtime library, if not found disable the component */
    dl2 = NULL;                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(cuda_runtime) > 0) {                                // If override given, it has to work.
        dl2 = dlopen(cuda_runtime, RTLD_NOW | RTLD_GLOBAL);        // Try to open that path.
        if (dl2 == NULL) {
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_CUDA_RUNTIME override '%s' given in Rules.cuda not found.", cuda_runtime);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl2 == NULL) {                                          // No override,
        dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);   // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl2 == NULL && cuda_root != NULL) {                             // if root given, try it.
        int strErr=snprintf(path_lib, sizeof(path_lib)-2, "%s/lib64/libcudart.so", cuda_root);   // PAPI Root check.
        path_lib[sizeof(path_lib)-1]=0;
        if (strErr > (int) sizeof(path_lib)-2) HANDLE_STRING_ERROR; 
        dl2 = dlopen(path_lib, RTLD_NOW | RTLD_GLOBAL);                 // Try to open that path.
    }

    // Check for failure.
    if (dl2 == NULL) {
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libcudart.so not found.");
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);
    }

    // We have a dl2. (libcudart.so).

    cudaGetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaGetDevice");
    // cudaGetDevicePropertiesPtr = DLSYM_AND_CHECK(dl2, "cudaGetDeviceProperties");
    cudaDeviceGetAttributePtr = DLSYM_AND_CHECK(dl2, "cudaDeviceGetAttribute");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl2, "cudaFree");
    cudaDriverGetVersionPtr = DLSYM_AND_CHECK(dl2, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr = DLSYM_AND_CHECK(dl2, "cudaRuntimeGetVersion");

    dl3 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(cuda_cupti) > 0) {                                       // If override given, it MUST work.
        dl3 = dlopen(cuda_cupti, RTLD_NOW | RTLD_GLOBAL);               // Try to open that path.
        if (dl3 == NULL) {
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_CUDA_CUPTI override '%s' given in Rules.cuda not found.", cuda_cupti);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl3 == NULL) {                                          // If no override,
        dl3 = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);    // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl3 == NULL && cuda_root != NULL) {                                         // If ROOT given, it doesn't HAVE to work.
        int strErr=snprintf(path_lib, sizeof(path_lib)-2, "%s/extras/CUPTI/lib64/libcupti.so", cuda_root);   // PAPI Root check.
        path_lib[sizeof(path_lib)-1]=0;
        if (strErr > (int) sizeof(path_lib)-2) HANDLE_STRING_ERROR;
        dl3 = dlopen(path_lib, RTLD_NOW | RTLD_GLOBAL);                             // Try to open that path.
    }

    // Check for failure.
    if (dl3 == NULL) {
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libcupti.so not found.");
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);   // Not found on default paths.
    }
    // We have a dl3. (libcupti.so)

    /* The macro DLSYM_AND_CHECK results in the expansion example below */
    /* cuptiDeviceEnumEventDomainsPtr = dlsym( dl3, "cuptiDeviceEnumEventDomains" ); */
    /* if ( dlerror()!=NULL ) { strncpy( _cuda_vector.cmp_info.disabled_reason, "A CUDA required function was not found in dynamic libs", PAPI_MAX_STR_LEN ); return ( PAPI_ENOSUPP ); } */
    cuptiDeviceEnumMetricsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceEnumMetrics");
    cuptiDeviceGetEventDomainAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetEventDomainAttribute");
    cuptiDeviceGetNumMetricsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetNumMetrics");
    cuptiEventGroupGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupGetAttribute");
    cuptiEventGroupReadEventPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupReadEvent");
    cuptiEventGroupSetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetAttribute");
    cuptiMetricGetRequiredEventGroupSetsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetRequiredEventGroupSets");
    cuptiEventGroupSetDisablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetDisable");
    cuptiEventGroupSetEnablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetEnable");
    cuptiEventGroupSetsCreatePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetsCreate");
    cuptiEventGroupSetsDestroyPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetsDestroy");
    cuptiGetTimestampPtr = DLSYM_AND_CHECK(dl3, "cuptiGetTimestamp");
    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl3, "cuptiGetVersion");
    cuptiMetricEnumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricEnumEvents");
    cuptiMetricGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetAttribute");
    cuptiMetricGetNumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetNumEvents");
    cuptiMetricGetValuePtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetValue");
    cuptiMetricCreateEventGroupSetsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricCreateEventGroupSets");
    cuptiSetEventCollectionModePtr = DLSYM_AND_CHECK(dl3, "cuptiSetEventCollectionMode");
    cuptiDeviceEnumEventDomainsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceEnumEventDomains");
    cuptiDeviceGetNumEventDomainsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetNumEventDomains");
    cuptiEventDomainEnumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiEventDomainEnumEvents");
    cuptiEventDomainGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventDomainGetAttribute");
    cuptiEventDomainGetNumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiEventDomainGetNumEvents");
    cuptiEventGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGetAttribute");
    cuptiEventGroupAddEventPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupAddEvent");
    cuptiEventGroupCreatePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupCreate");
    cuptiEventGroupDestroyPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupDestroy");
    cuptiEventGroupDisablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupDisable");
    cuptiEventGroupEnablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupEnable");
    cuptiEventGroupReadAllEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupReadAllEvents");
    cuptiEventGroupResetAllEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupResetAllEvents");
    cuptiGetResultStringPtr = DLSYM_AND_CHECK(dl3, "cuptiGetResultString");
    cuptiEnableKernelReplayModePtr = DLSYM_AND_CHECK(dl3, "cuptiEnableKernelReplayMode");
    cuptiDisableKernelReplayModePtr = DLSYM_AND_CHECK(dl3, "cuptiEnableKernelReplayMode");

#if CUPTI_API_VERSION >= 13
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerDeInitialize");
// cuptiDeviceGetChipName relies on cupti_target.h, not in legacy Cuda distributions.
    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetChipName");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEndSession");

    dl4 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(cuda_perfworks) > 0) {                                       // If override given, it MUST work.
        dl4 = dlopen(cuda_perfworks, RTLD_NOW | RTLD_GLOBAL);               // Try to open that path.
        if (dl4 == NULL) {
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_CUDA_PERFWORKS override '%s' given in Rules.cuda not found.", cuda_perfworks);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl4 == NULL) {                                          // If no override,
        dl4 = dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL);    // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl4 == NULL && cuda_root != NULL) {                                         // If ROOT given, it doesn't HAVE to work.
        int strErr=snprintf(path_lib, sizeof(path_lib)-2, "%s/extras/CUPTI/lib64/libnvperf_host.so", cuda_root);   // PAPI Root check.
        path_lib[sizeof(path_lib)-1]=0;
        if (strErr > (int) sizeof(path_lib)-2) HANDLE_STRING_ERROR;
        dl4 = dlopen(path_lib, RTLD_NOW | RTLD_GLOBAL);                             // Try to open that path.
    }

    // Check for failure.
    if (dl4 == NULL) {
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libnvperf_host.so not found.");
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);   // Not found on default paths.
    }

    // We have a dl4. (libnvperf_host.so)
    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl4, "NVPW_GetSupportedChipNames");

    Dl_info dl1_i, dl2_i, dl3_i, nvperf_info;
    // requires address of any function within the library.
    dladdr(NVPW_GetSupportedChipNamesPtr, &nvperf_info);
    dladdr(cuptiProfilerInitializePtr, &dl3_i);
    dladdr(cudaGetDevicePtr, &dl2_i);
    dladdr(cuCtxGetCurrentPtr, &dl1_i);
    SUBDBG("CUDA driver library loaded='%s'\n", dl1_i.dli_fname);
    SUBDBG("CUDA runtime library loaded='%s'\n", dl2_i.dli_fname);
    SUBDBG("Cupti library loaded='%s'\n", dl3_i.dli_fname);
    SUBDBG("PerfWorks library loaded='%s'\n", nvperf_info.dli_fname);
    
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetMetricProperties_End");

    NVPW_CUDA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_EndPassGroup")
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_GetConfigImage");

    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_IsAddMetricsPossible");

    NVPW_MetricsContext_GetCounterNames_BeginPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_MetricsContext_GetCounterNames_End");
#endif

    CUDA_CALL((*cudaDriverGetVersionPtr)(&cuda_version), return PAPI_ENOSUPP);
    CUDA_CALL((*cudaRuntimeGetVersionPtr)(&cuda_runtime_version), return PAPI_ENOSUPP);
    CUPTI_CALL((*cuptiGetVersionPtr)(&cupti_runtime_version), return PAPI_ENOSUPP);

    SUBDBG("CUDA Compile Versions: driver=%d, runtime=%d, cupti=%d\n", CUDA_VERSION, CUDART_VERSION, CUPTI_API_VERSION);
    SUBDBG("CUDA Runtime Versions: driver=%d, runtime=%d, cupti=%d\n", cuda_version, cuda_runtime_version, cupti_runtime_version);

#if CUPTI_API_VERSION >= 13
    cuptiProfilerGetCounterAvailabilityPtr = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = NULL; 

    if (cuda_version >= 11000 && cuda_runtime_version >= 11000)
    {
        cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK_nvperf(dl3, "cuptiProfilerGetCounterAvailability");
        NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK_nvperf(dl4, "NVPW_RawMetricsConfig_SetCounterAvailability");
    }
    else
    {
        // We cannot run without them; it may be possible to just eliminate them but haven't tried that yet. -TC
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Cuda and cuda_runtime lib versions must be >=11.");
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP); // We do not currently support cuda 10.x. 
    }
#endif

    return (PAPI_OK);
} // END _cuda_linkCudaLibraries

static int _cuda_add_native_events(cuda_context_t * gctxt)
{
    SUBDBG("Entering\n");
    CUresult cuErr;
    int deviceNum, userDeviceNum;
    uint32_t domainNum, eventNum;
    cuda_device_desc_t *mydevice;
    char tmpStr[PAPI_MIN_STR_LEN];
    tmpStr[PAPI_MIN_STR_LEN - 1] = '\0';
    size_t tmpSizeBytes;
    int ii;
    CUptiResult cuptiError;
    cudaError_t cudaErr;
    uint32_t maxEventSize;
    int strErr;
    long long elim_ns = 0;
    (void) elim_ns;

    CUcontext currCuCtx;
    CUcontext userCuCtx = NULL;

    // Get any current user cuda context. If it fails, leave it
    // at NULL. (Also the function may work and return NULL).

    cuErr = (*cuCtxGetCurrentPtr)(&userCuCtx);
    cudaErr = (*cudaGetDevicePtr)(&userDeviceNum);
    if (0) fprintf(stderr, "%s:%s:%i cuCtxGetCurrent cuErr=%d userCuCtx=%p, cudaErr=%d, userDevice=%d.\n", __FILE__, __func__, __LINE__, cuErr, userCuCtx, cudaErr, userDeviceNum);

    /* How many CUDA devices do we have? */
    cuErr = (*cuDeviceGetCountPtr) (&gctxt->deviceCount);
    if(cuErr == CUDA_ERROR_NOT_INITIALIZED) {
        /* If CUDA not initialized, initialize CUDA and retry the device list */
        /* This is required for some of the PAPI tools, that do not call the init functions */
        if (0) fprintf(stderr, "%s:%s:%i Executing cuInit(0).\n", __FILE__, __func__, __LINE__);
        cuErr = (cuInitPtr) (0); // Try the init.
        if(cuErr != CUDA_SUCCESS) {     // If that failed, we are bailing.
            const char *errString="Unknown";
            if (cuGetErrorStringPtr) (*cuGetErrorStringPtr) (cuErr, &errString); // Read the string.
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cuDeviceGetCount() failed; error code=%d [%s].", cuErr, errString);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            _cuda_vector.cmp_info.initialized = 1;
            _cuda_vector.cmp_info.disabled = cuErr;
            return PAPI_ENOSUPP;
        } // end if cuInit(0) failed.

        // Get number of cuda devices again.
        cuErr = (*cuDeviceGetCountPtr) (&gctxt->deviceCount);
        if(cuErr != CUDA_SUCCESS) {
            const char *errString="Unknown";
            if (cuGetErrorStringPtr) (*cuGetErrorStringPtr) (cuErr, &errString); // Read the string.
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cuDeviceGetCount() failed; error code=%d [%s].", cuErr, errString);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            _cuda_vector.cmp_info.initialized = 1;
            _cuda_vector.cmp_info.disabled = cuErr;
            return(PAPI_EMISC);    
        } 
    } // end if CUDA was not initialized; try to init.
    
    if (cuErr != CUDA_SUCCESS) {
        const char *errString="Unknown";
        if (cuGetErrorStringPtr) (*cuGetErrorStringPtr) (cuErr, &errString); // Read the string.
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "Function cuDeviceGetCount() failed; error code=%d [%s].", cuErr, errString);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        _cuda_vector.cmp_info.initialized = 1;
        _cuda_vector.cmp_info.disabled = cuErr;
        return(PAPI_EMISC);    
    } 

    // We have the device count.
    if(gctxt->deviceCount == 0) {
        char* strCpy=strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA initialized but no CUDA devices found.", PAPI_MAX_STR_LEN);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i '%s'\n", __FILE__, __func__, __LINE__, _cuda_vector.cmp_info.disabled_reason);
        return PAPI_ENOSUPP;
    }
    SUBDBG("Found %d devices\n", gctxt->deviceCount);

    /* allocate memory for device information */
    gctxt->deviceArray = (cuda_device_desc_t *) papi_calloc(gctxt->deviceCount, sizeof(cuda_device_desc_t));
    if (!gctxt->deviceArray) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "Could not allocate %lu bytes of memory for CUDA device structure.", gctxt->deviceCount*sizeof(cuda_device_desc_t));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i '%s'\n", __FILE__, __func__, __LINE__, _cuda_vector.cmp_info.disabled_reason);
        return (PAPI_ENOMEM);
    }

    int total_le70=0, total_ge70=0;

    // For each device, get some device information.
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        mydevice = &gctxt->deviceArray[deviceNum];
        /* Get device id, name, compute capability for each device */
        cuErr = (*cuDeviceGetPtr) (&mydevice->cuDev, deviceNum);
        if (cuErr != CUDA_SUCCESS) {
            const char *errString=NULL;
            (*cuGetErrorStringPtr) (cuErr, &errString); // Read the string.
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cuDeviceGet() failed; error code=%d [%s].", cuErr, errString);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i '%s'\n", __FILE__, __func__, __LINE__, _cuda_vector.cmp_info.disabled_reason);
            return(PAPI_EMISC);    
        } 

        cuErr = (*cuDeviceGetNamePtr) ((char*) &mydevice->deviceName, PAPI_MIN_STR_LEN - 1, mydevice->cuDev);
        if (cuErr != CUDA_SUCCESS) {
            const char *errString=NULL;
            (*cuGetErrorStringPtr) (cuErr, &errString); // Read the string.
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cuDeviceGetName() failed; error code=%d [%s].", cuErr, errString);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i '%s'\n", __FILE__, __func__, __LINE__, _cuda_vector.cmp_info.disabled_reason);
            return(PAPI_EMISC);    
        } 

        mydevice->deviceName[PAPI_MIN_STR_LEN - 1] = '\0';                      // z-terminate it.

        // The routine cuptiDeviceGetNumEventDomains() is illegal for devices with compute
        // capability >= 7.5.  From the online manual
        // (https://docs.nvidia.com/cupti/Cupti/modules.html): Legacy CUPTI Profiling is not
        // supported on devices with Compute Capability 7.5 or higher (Turing+).  From
        // https://developer.nvidia.com/cuda-gpus#compute): We find the Quadro GTX 5000 (our first
        // failure) has a Compute Capability of 7.5.

        // Note: We use cudaDeviceGetAttribute() because it is consistent; the library routine knows
        // where to find the major and minor within the properties structure. If we use
        // cudaGetDeviceProperties; the returned structure depends on the current cuda driver
        // loaded; e.g.  9.2.88 and 11.2.0 return different structures. So if we compile PAPI with
        // 9.2.88, and run with 11.2.0, the major and minor we get (using the wrong structure
        // definition) is 1024,64 instead of 7,0. If the minor is needed,
        // cudaDevAttrComputeCapabilityMinor is the necessary attribute.

        cudaErr = (*cudaDeviceGetAttributePtr) (&mydevice->CC_Major, cudaDevAttrComputeCapabilityMajor, deviceNum); 
        cudaErr |= (*cudaDeviceGetAttributePtr) (&mydevice->CC_Minor, cudaDevAttrComputeCapabilityMinor, deviceNum);
        if (0) fprintf(stderr, "%s:%s:%i Compute Capability Major=%d\n", __FILE__, __func__, __LINE__, mydevice->CC_Major);
        if (0) fprintf(stderr, "%s:%s:%i Compute Capability Minor=%d\n", __FILE__, __func__, __LINE__, mydevice->CC_Minor);
        if (cudaErr != cudaSuccess) {
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cudaDeviceGetAttribute() error code=%d.", cudaErr);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            if (DEBUG_CALLS) fprintf(stderr, "%s:%s:%i '%s'\n", __FILE__, __func__, __LINE__, _cuda_vector.cmp_info.disabled_reason);
            return(PAPI_EMISC);    
        }

        // If profiler is available and we CAN use profiler, we do.
        // If profiler is unavailable we must be able to use Legacy.
        mydevice->cc_le70 = 0;
        mydevice->cc_ge70 = 0;

        if (0) fprintf(stderr, "%s:%s:%i device=%d name=%s  major=%d.\n", __FILE__, __func__, __LINE__, deviceNum,
            mydevice->deviceName, mydevice->CC_Major);

        if (mydevice->CC_Major < 7 || (mydevice->CC_Major == 7 && mydevice->CC_Minor == 0)) {
            mydevice->cc_le70 = 1;
        }

        if (mydevice->CC_Major >= 7) {
            mydevice->cc_ge70 = 1;
        }


        total_le70 += mydevice->cc_le70;
        total_ge70 += mydevice->cc_ge70;
    } // END per device.

#if CUPTI_API_VERSION >= 13
    // Profiler exists, use it if all devices can use it.
    if (total_ge70 == gctxt->deviceCount) {
        int ret = _cuda11_add_native_events(gctxt);
        if (ret == PAPI_OK) _cuda11_cuda_vector();   // reset function pointers.

        // this is to trick the final return from component_init(), to set
        // the  _cuda_vector.cmp_info.num_native_events correctly. 
        global_cuda_context->availEventSize = cuda11_numEvents;
        return(ret);
    } // Done with init_component if cuda11 worked.

    // Profile exists, but not all devices are >= 7.0, so must use legacy.
    if (total_le70 != gctxt->deviceCount) {
        // some devices are 7.5 and cannot use legacy. We cannot support this.
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "(2) Mixed compute capabilities, must use Legacy, but only %d of %d devices have CC<=7.0", total_le70, gctxt->deviceCount);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return(PAPI_ENOSUPP);    
    }

    // We'll be okay with legacy.

#endif 

    // If profiler existed, we are proceeding without it. But 
    // it may not, so we need to check again if Legacy will work.
     if (total_le70 != gctxt->deviceCount) {
        // some devices are 7.5 and cannot use legacy. We cannot support this.
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "(%d) Mixed compute capabilities, must use Legacy, but only %d of %d devices have CC<=7.0", CUPTI_API_VERSION, total_le70, gctxt->deviceCount);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return(PAPI_ENOSUPP);    
    }

    // It is safe to proceed with Legacy CUPTI interace.
    // For each device, get domains and domain-events counts.

    maxEventSize = 0;
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        mydevice = &gctxt->deviceArray[deviceNum];
        /* Get numeventdomains for each device */

        cuptiError=(*cuptiDeviceGetNumEventDomainsPtr) (mydevice->cuDev, &mydevice->maxDomains);
        if (cuptiError != CUPTI_SUCCESS) {
            const char *errstr;
            (*cuptiGetResultStringPtr)(cuptiError, &errstr);
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Function cuptiDeviceGetNumEventDomains() failed; error code=%d [%s].", cuptiError, errstr);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC);    
        }
            
        /* Allocate space to hold domain IDs */
        mydevice->domainIDArray = (CUpti_EventDomainID *) papi_calloc(
            mydevice->maxDomains, sizeof(CUpti_EventDomainID));

        if (!mydevice->domainIDArray) {
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for device domain ID array", mydevice->maxDomains*sizeof(CUpti_EventDomainID));
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return (PAPI_ENOMEM);
        }

        /* Put domain ids into allocated space */
        size_t domainarraysize = mydevice->maxDomains * sizeof(CUpti_EventDomainID);
        // enumerate domain ids into space.
        cuptiError=(*cuptiDeviceEnumEventDomainsPtr)(mydevice->cuDev, &domainarraysize, mydevice->domainIDArray);
        if (cuptiError != CUPTI_SUCCESS) {
            const char *errstr;
           (*cuptiGetResultStringPtr)(cuptiError, &errstr);
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "Function cuptiDeviceEnumEventDomains() failed; error code=%d [%s].", cuptiError, errstr);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC);    
        }

        /* Allocate space to hold domain event counts */
        mydevice->domainIDNumEvents = (uint32_t *) papi_calloc(mydevice->maxDomains, sizeof(uint32_t));
        if (!mydevice->domainIDArray) {
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "Could not allocate %lu bytes of memory for device domain ID array", mydevice->maxDomains*sizeof(CUpti_EventDomainID));
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return (PAPI_ENOMEM);
        }

        /* For each domain, get event counts in domainNumEvents[] */
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {     // For each domain,
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];  // .. make a copy of the domain ID.
            /* Get num events in domain */
            cuptiError=(*cuptiEventDomainGetNumEventsPtr) (domainID, &mydevice->domainIDNumEvents[domainNum]);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
               (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiEventDomaintGetNumEvents() failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC);    
            }

            maxEventSize += mydevice->domainIDNumEvents[domainNum];             // keep track of overall number of events.
        } // end for each domain.
    } // end of for each device.

    // Increase maxEventSize for metrics on this device.
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {               // for each device,
        uint32_t maxMetrics = 0;
        CUptiResult cuptiRet;
        mydevice = &gctxt->deviceArray[deviceNum];                                  // Get cuda_device_desc pointer.
        cuptiRet = (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics);   // Read the # metrics on this device.
        if (cuptiRet != CUPTI_SUCCESS || maxMetrics < 1) continue;                  // If no metrics, skip to next device.
        maxEventSize += maxMetrics;                                                 // make room for metrics we discover later.
    } // end for each device.

    /* Allocate space for all events and descriptors */
    gctxt->availEventKind = (CUpti_ActivityKind *) papi_calloc(maxEventSize, sizeof(CUpti_ActivityKind));
    if (!gctxt->availEventKind) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for availEventKind.", maxEventSize*sizeof(CUpti_ActivityKind));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    gctxt->availEventDeviceNum = (int *) papi_calloc(maxEventSize, sizeof(int));
    if (!gctxt->availEventDeviceNum) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for availEventDeviceNum.", maxEventSize*sizeof(int));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    gctxt->availEventIDArray = (CUpti_EventID *) papi_calloc(maxEventSize, sizeof(CUpti_EventID));
    if (!gctxt->availEventIDArray) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for availEventIDArray.", maxEventSize*sizeof(CUpti_EventID));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    gctxt->availEventIsBeingMeasuredInEventset = (uint32_t *) papi_calloc(maxEventSize, sizeof(uint32_t));
    if (!gctxt->availEventIsBeingMeasuredInEventset) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for availEventIsBeingMeasured.", maxEventSize*sizeof(uint32_t));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    gctxt->availEventDesc = (cuda_name_desc_t *) papi_calloc(maxEventSize, sizeof(cuda_name_desc_t));
    if (!gctxt->availEventDesc) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for availEventDesc.", maxEventSize*sizeof(cuda_name_desc_t));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    // Record all events on each device, and their descriptions.
    uint32_t idxEventArray = 0;
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {           // loop through each device.
        mydevice = &gctxt->deviceArray[deviceNum];                              // get a pointer to the cuda_device_desc struct.

        // For each domain, get and store event IDs, names, descriptions.
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {         // loop through the domains in this device.

            /* Get domain id */
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];      // get the domain id,
            uint32_t domainNumEvents = mydevice->domainIDNumEvents[domainNum];      // get the number of events in it.

            // SUBDBG( "For device %d domain %d domainID %d numEvents %d\n", mydevice->cuDev, domainNum, domainID, domainNumEvents );

            CUpti_EventID *domainEventIDArray =                                         // Make space for the events in this domain.
                (CUpti_EventID *) papi_calloc(domainNumEvents, sizeof(CUpti_EventID));  // ..
            if (!domainEventIDArray) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for domainEventIDArray.", domainNumEvents*sizeof(CUpti_EventID));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

            size_t domainEventArraySize = domainNumEvents * sizeof(CUpti_EventID);      // compute size of array we allocated.
            // Enumerate the events in the domain,
            cuptiError=(*cuptiEventDomainEnumEventsPtr)(domainID, &domainEventArraySize, domainEventIDArray);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiEventDomainEnumEvents() failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC);    
            }

            for(eventNum = 0; eventNum < domainNumEvents; eventNum++) {                 // Loop through the events in this domain.
                CUpti_EventID myeventCuptiEventId = domainEventIDArray[eventNum];       // .. get this event,
                gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_EVENT;       // .. record the kind,
                gctxt->availEventIDArray[idxEventArray] = myeventCuptiEventId;          // .. record the id,
                gctxt->availEventDeviceNum[idxEventArray] = deviceNum;                  // .. record the device number,

                tmpSizeBytes = PAPI_MAX_STR_LEN - 1 * sizeof(char);                     // .. compute size of name,
                cuptiError=(*cuptiEventGetAttributePtr)(myeventCuptiEventId,CUPTI_EVENT_ATTR_NAME, &tmpSizeBytes, tmpStr);                      // .. into tmpStr.
                if (cuptiError != CUPTI_SUCCESS) {
                    const char *errstr;
                    (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                    strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "Function cuptiEventGetAttribute(EVENT_NAME) failed; error code=%d [%s].", cuptiError, errstr);
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC);    
                }

                snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MAX_STR_LEN,   // record expanded name for papi user.
                    "event:%s:device=%d", tmpStr, deviceNum);
                gctxt->availEventDesc[idxEventArray].name[PAPI_MAX_STR_LEN - 1] = '\0'; // ensure null termination.
                char *nameTmpPtr = gctxt->availEventDesc[idxEventArray].name;           // For looping, get pointer to name.
                for(ii = 0; ii < (int) strlen(nameTmpPtr); ii++) {                      // Replace spaces with underscores.
                    if(nameTmpPtr[ii] == ' ') nameTmpPtr[ii] = '_';                     // ..
                }

                /* Save description in the native event array */
                tmpSizeBytes = PAPI_2MAX_STR_LEN - 1 * sizeof(char);                    // Most space to use for description.
                cuptiError=(*cuptiEventGetAttributePtr)(myeventCuptiEventId,
                    CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &tmpSizeBytes,
                    gctxt->availEventDesc[idxEventArray].description);
                if (cuptiError != CUPTI_SUCCESS) {
                    const char *errstr;
                    (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                    strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "Function cuptiEventGetAttribute(EVENT_NAME) failed; error code=%d [%s].", cuptiError, errstr);
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC);    
                }

                gctxt->availEventDesc[idxEventArray].description[PAPI_2MAX_STR_LEN - 1] = '\0'; // Ensure null terminator.
                gctxt->availEventDesc[idxEventArray].numMetricEvents = 0;                       // Not a metric.
                gctxt->availEventDesc[idxEventArray].metricEvents = NULL;                       // No space allocated.
                /* Increment index past events in this domain to start of next domain */
                idxEventArray++;                                                        // Bump total number of events.
            } // end of events in this domain.

            papi_free(domainEventIDArray);                                              // done with temp space.
        } // end of domain loop within device.
    } // end of device loop, for events.

    // Now we retrieve and store all METRIC info for each device; this includes
    // both cuda metrics and nvlink metrics.
    int firstMetricIdx = idxEventArray;
    int maxUnenumEvents = 0;
    int idxAllEvents = 0;
    cuda_all_events_t *localAllEvents = NULL; 

    SUBDBG("Checking for metrics\n");
    for (deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {

        // This is a sensitive protocol. We require a context on each device to
        // be able to eliminate multi-pass metrics. The problem is that we may
        // be working with an un-initialized (cuInit) or not, and we may be
        // working with an application's non-primary context or not, and we may
        // have to create a primary context, or not. We have tried doing a
        // cudaSetDevice(), which is supposed to create primary context if it
        // is not present -- it creates something, but it is unusable. The same
        // is true for cudaFree(). We have tried to just cuCtxCreate() our own,
        // that works if the application has not done a cuCtxCreate(), but it
        // fails if the application *has* done a cuCtxCreate().
        // cuDevicePrimaryCtxRetain() creates a context, but doesn't set it as
        // the current context. (The manual says it does, but experimentation
        // shows differently: if the current context is NOT the primary
        // context, it is not replaced.) We check to make sure the current context (userCtx) 
        // is not the same as the primary context, if they differ we push the primary, which
        // automatically does a cudaSetDevice(), it works, and we can pop it
        // later (restoring the application's context, if any). But only if we pushed it.

        // Get/create primary context for device, must later
        // cuDevicePrimaryCtxRelease(deviceNum). Does not make
        // context active; push to make it active.
       
        CU_CALL((*cuDevicePrimaryCtxRetainPtr) (&currCuCtx, deviceNum), 
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "cuDevicePrimaryCtxRetain failed.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC););

        if (currCuCtx != userCuCtx) { 
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), 
                int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "cuCtxPushCurrent() failed.");
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC););
        }

        uint32_t maxMetrics = 0, i, j;
        CUpti_MetricID *metricIdList = NULL;
        CUptiResult cuptiRet;
        mydevice = &gctxt->deviceArray[deviceNum];                                  // Get cuda_device_desc pointer.
        cuptiRet = (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics);   // Read the # metrics on this device.
        if (cuptiRet != CUPTI_SUCCESS || maxMetrics < 1) continue;                  // If no metrics, skip to next device.

        SUBDBG("Device %d: Checking each of the (maxMetrics) %d metrics\n", deviceNum, maxMetrics);

        // Make a temporary list of the metric Ids to add to the available named collectables.
        size_t size = maxMetrics * sizeof(CUpti_EventID);
        metricIdList = (CUpti_MetricID *) papi_calloc(maxMetrics, sizeof(CUpti_EventID));
        if (!metricIdList) {
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "Could not allocate %lu bytes of memory for metricIdList.", maxMetrics*sizeof(CUpti_EventID));
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            if (currCuCtx != userCuCtx) { 
                CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), 
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuCtxPopCurrent() failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
            }
            CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "cuDevicePrimaryCtxRelease failed.");
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC););
            return(PAPI_EMISC);
        }

        cuptiError=(*cuptiDeviceEnumMetricsPtr)(mydevice->cuDev, &size, metricIdList);  // Enumerate into metricIDList.
        if (cuptiError != CUPTI_SUCCESS) {
            const char *errstr;
            (*cuptiGetResultStringPtr)(cuptiError, &errstr);
            strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "Function cuptiDeviceEnumMetrics failed; error code=%d [%s].", cuptiError, errstr);
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            if (currCuCtx != userCuCtx) { 
                CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuCtxPopCurrent() failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
            }
            CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "cuDevicePrimaryCtxRelease failed.");
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC););
            return(PAPI_EMISC);
        }

        // Elimination loop for metrics we cannot support.
        for (i=0, j=0; i<maxMetrics; i++) {                                         // process each metric Id.
            size = PAPI_MIN_STR_LEN-1;                                              // Most bytes allowed to be written.
            cuptiError=(*cuptiMetricGetAttributePtr) (metricIdList[i], CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *) tmpStr);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricGetAttribute(METRIC_NAME)) failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), 
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }

                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);
            }

            // Note that 'size' also returned total bytes written.
            tmpStr[size] = '\0';

            // We reject anything requiring more than 1 set.

            #if defined(TIME_MULTIPASS_ELIM)
            long long start_ns = PAPI_get_real_nsec();
            #endif 

            CUpti_EventGroupSets *thisEventGroupSets = NULL;
            cuptiError=(*cuptiMetricCreateEventGroupSetsPtr) (
                currCuCtx,
                sizeof(CUpti_MetricID),
                &metricIdList[i],
                &thisEventGroupSets);

            // Some metric names fail cuptiMetricCreateEventGroupSets(), if they
            // do we just exclude them, same as if numSets>1. 
            if (cuptiError != CUPTI_SUCCESS) {
                  const char *errstr;
                  (*cuptiGetResultStringPtr) (cuptiError, &errstr);
                  if (0) fprintf(stderr, "%s:%s:%i metric '%s:device=%d' failed cuptiMetricCreateEventGroupSets() cuptiError=%d [%s].\n", __FILE__, __func__, __LINE__, tmpStr, deviceNum, cuptiError, errstr);
                continue;
            } else if (0) fprintf(stderr, "%s:%i cuptiMetricCreateEventGroupSets() success.\n", __FILE__, __LINE__);
            
            int numSets = 0;                                                        // # of sets (passes) required.
            if (thisEventGroupSets != NULL) {
                numSets=thisEventGroupSets->numSets;                                // Get sets if a grouping is necessary.
                cuptiError=(*cuptiEventGroupSetsDestroyPtr) (thisEventGroupSets);     // Done with this.
               if (cuptiError != CUPTI_SUCCESS) {
                  const char *errstr;
                  (*cuptiGetResultStringPtr) (cuptiError, &errstr);
                  strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                     "Function cuptiEventGroupSetsDestroy() failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);    
               } // else fprintf(stderr, "%s:%i cuptiEventGroupSetsDestroy() success.\n", __FILE__, __LINE__);
            } else {
               strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                  "Function cuptiMetricCreateEventGroupSets() returned an invalid NULL pointer.");
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
               if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);    
            }

            #if defined(TIME_MULTIPASS_ELIM)
            long long net_ns = PAPI_get_real_nsec() - start_ns;
            elim_ns += net_ns;
            #endif 

            if (numSets > 1) {                                                // skip this metric too many passes.
                // fprintf(stderr, "%s:%i skipping metric %s has %i sets.\n", __FILE__, __LINE__, tmpStr, numSets);
                continue;
            }

            metricIdList[j++] = metricIdList[i];                                    // we are compressing if we skipped any.
        } // end elimination loop.

        // Done with eliminations, the rest are valid.
        maxMetrics = j;                                                             // Change the number to process.

        // Eliminations accomplished, now add the valid metric Ids to the list.
        for(i = 0; i < maxMetrics; i++) {                                           // for each id,
            gctxt->availEventIDArray[idxEventArray] = metricIdList[i];              // add to the list of collectables.
            gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_METRIC;      // Indicate it is a metric.
            gctxt->availEventDeviceNum[idxEventArray] = deviceNum;                  // remember the device number.
            size = PAPI_MAX_STR_LEN;
            cuptiError=(*cuptiMetricGetAttributePtr) (metricIdList[i], CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *) tmpStr);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricGetAttribute(METRIC_NAME)) failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);    
            }

            if (size >= PAPI_MAX_STR_LEN) {                                         // Truncate if we don't have room for the name.
                gctxt->availEventDesc[idxEventArray].name[PAPI_MAX_STR_LEN - 1] = '\0';
            }

            size_t MV_KindSize = sizeof(CUpti_MetricValueKind);
            cuptiError=(*cuptiMetricGetAttributePtr)(metricIdList[i], CUPTI_METRIC_ATTR_VALUE_KIND, 
                &MV_KindSize, &gctxt->availEventDesc[idxEventArray].MV_Kind);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricGetAttribute(METRIC_KIND)) failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);    
            }

            snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MAX_STR_LEN,   // .. develop name for papi user in tmpStr.
                "metric:%s:device=%d", tmpStr, deviceNum);

            size = PAPI_2MAX_STR_LEN-1;                                             // Most bytes to return.
            cuptiError=(*cuptiMetricGetAttributePtr)(metricIdList[i], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, 
                &size, (uint8_t *) gctxt->availEventDesc[idxEventArray].description);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricGetAttribute(METRIC_LONG_DESC)) failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EMISC);    
            }

            // Note that 'size' also returned total bytes written.
            gctxt->availEventDesc[idxEventArray].description[size] = '\0';          // Always z-terminate.

            // Now we get all the sub-events of this metric.
            uint32_t numSubs;
            CUpti_MetricID itemId = metricIdList[i];                                //.. shortcut to metric id.
            cuptiError=(*cuptiMetricGetNumEventsPtr) (itemId, &numSubs);            // .. Get number of sub-events in metric.
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricGetNumEvents() failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EINVAL);    
            }

            size_t sizeBytes = numSubs * sizeof(CUpti_EventID);                     // .. compute size of array we need.
            CUpti_EventID *subEventIds = papi_malloc(sizeBytes);                    // .. Make the space.
            if (!subEventIds) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for subEventIds.", sizeBytes);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum), 
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return (PAPI_ENOMEM);
            }

            cuptiError=(*cuptiMetricEnumEventsPtr)(itemId, &sizeBytes, subEventIds);
            if (cuptiError != CUPTI_SUCCESS) {
                const char *errstr;
                (*cuptiGetResultStringPtr)(cuptiError, &errstr);
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Function cuptiMetricEnumEvents() failed; error code=%d [%s].", cuptiError, errstr);
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                if (currCuCtx != userCuCtx) { 
                    CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                        "cuCtxPopCurrent() failed.");
                        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                        return(PAPI_EMISC););
                }
                CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
                    int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "cuDevicePrimaryCtxRelease failed.");
                    _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                    if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                    return(PAPI_EMISC););
                return(PAPI_EINVAL);    
            }

            gctxt->availEventDesc[idxEventArray].metricEvents = subEventIds;        // .. Copy the array pointer for IDs.
            gctxt->availEventDesc[idxEventArray].numMetricEvents = numSubs;         // .. Copy number of elements in it.
            maxUnenumEvents += numSubs;                                             // .. a rough size for unenum event array.

            idxEventArray++;                                                        // count another collectable found.
        } // end maxMetrics loop.

        papi_free(metricIdList);                                                    // Done with this enumeration of metrics.

        if (currCuCtx != userCuCtx) { 
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "cuCtxPopCurrent() failed.");
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return(PAPI_EMISC););
        }

        CU_CALL((*cuDevicePrimaryCtxReleasePtr) (deviceNum),
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "cuDevicePrimaryCtxRelease failed.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC););
    } // end of device loop, for metrics.

    //-------------------------------------------------------------------------
    // The NVIDIA code, by design, zeros counters once events are read. PAPI
    // promises monotonically increasing performance counters. So we have to
    // accumulate the counters in-between reads. We make a new list here, 
    // which includes unenumerated events, so we can do that.
    //-------------------------------------------------------------------------
    // Build an all Events array. Over-specify the number of entries.
    localAllEvents = calloc(maxUnenumEvents+firstMetricIdx, sizeof(cuda_all_events_t));
    CHECK_PRINT_EVAL(localAllEvents == NULL, "Malloc failed", return (PAPI_ENOMEM));
    if (!localAllEvents) {
        strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "Could not allocate %lu bytes of memory for localAllEvents.", (maxUnenumEvents+firstMetricIdx)*sizeof(cuda_all_events_t));
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
        return (PAPI_ENOMEM);
    }

    unsigned int i,j;
    int k;
   
    // Begin by populating with all fixed events. 
    for (k=0; k<firstMetricIdx; k++) {
        localAllEvents[k].eventId = gctxt->availEventIDArray[k];         // CUpti EventID.
        localAllEvents[k].deviceNum = gctxt->availEventDeviceNum[k];     // Device.
        localAllEvents[k].idx = k;                                       // Index into table.
        localAllEvents[k].nonCumulative = 0;                             // flag if spot or constant event.
    }        
      
    idxAllEvents = firstMetricIdx;
    for (i = firstMetricIdx; i<idxEventArray; i++) {
        uint32_t numSubs=gctxt->availEventDesc[i].numMetricEvents;
        #ifdef PRODUCE_EVENT_REPORT
        fprintf(stderr, "%s %d SubEvents:\n", gctxt->availEventDesc[i].name, numSubs);
        #endif 
        // Look for any subEvents that are NOT in the array of raw events
        // we have already composed, and add to array. 
        // i=current metric idx, j is subEvent entry within it,
        // k index is search item within known events.
        for (j=0; j<numSubs; j++) {
            #ifdef PRODUCE_EVENT_REPORT
            fprintf(stderr, "    0x%08X = ",gctxt->availEventDesc[i].metricEvents[j]); 
            #endif 
            // search for this subEventId in our list.
            for (k=0; k<firstMetricIdx; k++) {
                if (gctxt->availEventIDArray[k] ==                  // existing event id
                    gctxt->availEventDesc[i].metricEvents[j] &&     // metric event and subEvent id,
                    gctxt->availEventDeviceNum[k] ==                // existing event device, 
                    gctxt->availEventDeviceNum[i]) break;           // subEvent device.
                }

             // If we found it, report but move on to next subEvent,
             // we do not need to add this to the list as a newly
             // discovered event.
            if (k < firstMetricIdx) {
                #ifdef PRODUCE_EVENT_REPORT
                fprintf(stderr, "%s\n", gctxt->availEventDesc[k].name);
                #endif 
                continue;
            }

            // Here, we did not find it in the list of normal events, so
            // we have to add it in.
            #ifdef PRODUCE_EVENT_REPORT
            fprintf(stderr, "?\n");      
            #endif 
            localAllEvents[idxAllEvents].eventId = 
                gctxt->availEventDesc[i].metricEvents[j];
            localAllEvents[idxAllEvents].deviceNum = 
                gctxt->availEventDeviceNum[i];
            localAllEvents[idxAllEvents].idx = -1;
            localAllEvents[idxAllEvents].nonCumulative = 0;
            idxAllEvents++;
        }
    } // end 'for each metric' search for unique subevents. 

    gctxt->availEventSize = idxEventArray;

    // We should have an array of all possible events, with duplicates.
    // Sort into ascending order by event #, device #.
    qsort(localAllEvents, idxAllEvents, sizeof(cuda_all_events_t), ascAllEvents);

    // condense to remove duplicates. i scans list,
    // j holds count (and index) of next unique eventId.
    j=1;
    for (k=1; k<idxAllEvents; k++) {
        if (localAllEvents[k].eventId == localAllEvents[k-1].eventId &&
            localAllEvents[k].deviceNum == localAllEvents[k-1].deviceNum) continue;
        // found a new EventId.
        localAllEvents[j].eventId   = localAllEvents[k].eventId;
        localAllEvents[j].deviceNum = localAllEvents[k].deviceNum;
        localAllEvents[j].idx       = localAllEvents[k].idx;
        j++;
    }

    gctxt->numAllEvents = j;
    gctxt->allEvents = localAllEvents;

    #ifdef PRODUCE_EVENT_REPORT
    fprintf(stderr, "\nFull Event Report:\n"); 
    for (k=0; k<(signed) j; k++) {
        fprintf(stderr, "Event=0x%08x  Device=%d Name=", 
        gctxt->allEvents[k].eventId, gctxt->allEvents[k].deviceNum);
        int idx=gctxt->allEvents[k].idx;
        if (idx >= 0) {                                             // If a known event,
            fprintf(stderr, "%s\n", gctxt->availEventDesc[idx].name);
        } else {
            fprintf(stderr, "?\n");
        }
    }

    fprintf(stderr, "UnEnumerated Events Total Usage: %d Unique: %d.\n", idxAllEvents-firstMetricIdx, j-firstMetricIdx);
    #endif 

#ifdef EXPOSE_UNENUMERATED_EVENTS /* To help with Unenum event exploration: */
    /* Reallocate space for all events and descriptors to make room. */
    maxEventSize += (j-firstMetricIdx); 
    gctxt->availEventKind = (CUpti_ActivityKind *) papi_realloc(gctxt->availEventKind, maxEventSize * sizeof(CUpti_ActivityKind));
            if (!gctxt->availEventKind) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for availEventKind.", maxEventSize * sizeof(CUpti_ActivityKind));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

    gctxt->availEventDeviceNum = (int *) papi_realloc(gctxt->availEventDeviceNum,maxEventSize * sizeof(int));
            if (!gctxt->availEventDeviceNum) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for availEventDeviceNum.", maxEventSize * sizeof(int));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

    gctxt->availEventIDArray = (CUpti_EventID *) papi_realloc(gctxt->availEventIDArray,maxEventSize * sizeof(CUpti_EventID));
            if (!gctxt->availEventIDArray) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for availEventIDArray.", maxEventSize * sizeof(CUpti_EventID));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

    gctxt->availEventIsBeingMeasuredInEventset = (uint32_t *) papi_realloc(gctxt->availEventIsBeingMeasuredInEventset,maxEventSize * sizeof(uint32_t));
            if (!gctxt->availEventIsBeingMeasuredInEventset) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for availEventIsBeingMeasured.", maxEventSize * sizeof(uint32_t));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

    gctxt->availEventDesc = (cuda_name_desc_t *) papi_realloc(gctxt->availEventDesc,maxEventSize * sizeof(cuda_name_desc_t));
            if (!gctxt->availEventDesc) {
                strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "Could not allocate %lu bytes of memory for availEventDesc.", maxEventSize * sizeof(cuda_name_desc_t));
                _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
                return (PAPI_ENOMEM);
            }

    for (k=0; k<(signed) j; k++) {
        int idx=gctxt->allEvents[k].idx;
        if (idx < 0) {
            gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_EVENT;           // .. record the kind,
            gctxt->availEventIDArray[idxEventArray] = gctxt->allEvents[k].eventId;      // .. record the id,
            gctxt->availEventDeviceNum[idxEventArray] = gctxt->allEvents[k].deviceNum;  // .. record the device number,
            // We have tried to get the names of these events using
            // cuptiEventGetAttribute() with CUPTI_EVENT_ATTR_NAME,
            // it just returns "event_name" for all them.
            snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MAX_STR_LEN,       // record expanded name for papi user.
                "unenum_event:0x%08X:device=%d", 
                gctxt->allEvents[k].eventId, 
                gctxt->allEvents[k].deviceNum);
            gctxt->availEventDesc[idxEventArray].name[PAPI_MAX_STR_LEN - 1] = '\0'; // ensure null termination.
            snprintf(gctxt->availEventDesc[idxEventArray].description, PAPI_2MAX_STR_LEN - 1,
            "Unenumerated Event used in a metric.");
            gctxt->availEventDesc[idxEventArray].numMetricEvents = 0;                       // Not a metric.
            gctxt->availEventDesc[idxEventArray].metricEvents = NULL;                       // No space allocated.
            idxEventArray++;                                                        // Bump total number of events.
        }
    } 

    gctxt->availEventSize = idxEventArray;
#endif /* END IF we should expose Unenumerated Events */

    #if defined(TIME_MULTIPASS_ELIM)
    fprintf(stderr, "%s:%i metric set>1 elimination usec=%lld for %d events.\n", __FILE__, __LINE__, (elim_ns+500)/1000, idxEventArray);
    #endif 

    // Restore user context, if we had one.
    if (userCuCtx != NULL) {
        CU_CALL((*cuCtxSetCurrentPtr) (userCuCtx),
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "cuCtxSetCurrent() failed.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC););
    } else {
        // If the application did not have a current context, restore their device number.
        CUDA_CALL((*cudaSetDevicePtr)(userDeviceNum), 
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "cudaSetDevice() failed.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;    
            return(PAPI_EMISC););
    }        

    return 0;
} // end _cuda_add_native_events


/*
  This routine tries to convert all CUPTI values to long long values.
  If the CUPTI value is an integer type, it is cast to long long.  If
  the CUPTI value is a percent, it is multiplied by 100 to return the
  integer percentage.  If the CUPTI value is a double, the value
  is cast to long long... this can be a severe truncation.
 */
static int _cuda_convert_metric_value_to_long_long(CUpti_MetricValue metricValue, CUpti_MetricValueKind valueKind, long long int *papiValue)
{
    union {
        long long ll;
        double fp;
    } tmpValue;

    SUBDBG("Try to convert the CUPTI metric value kind (index %d) to PAPI value (long long or double)\n", valueKind);
    switch (valueKind) {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        SUBDBG("Metric double %f\n", metricValue.metricValueDouble);
        tmpValue.ll = (long long)(metricValue.metricValueDouble);
        //CHECK_PRINT_EVAL(tmpValue.fp - metricValue.metricValueDouble > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    case CUPTI_METRIC_VALUE_KIND_UINT64:
        SUBDBG("Metric uint64 = %llu\n", (unsigned long long) metricValue.metricValueUint64);
        tmpValue.ll = (long long) (metricValue.metricValueUint64);
        CHECK_PRINT_EVAL(tmpValue.ll - metricValue.metricValueUint64 > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    case CUPTI_METRIC_VALUE_KIND_INT64:
        SUBDBG("Metric int64 = %lld\n", (long long) metricValue.metricValueInt64);
        tmpValue.ll = (long long) (metricValue.metricValueInt64);
        CHECK_PRINT_EVAL(tmpValue.ll - metricValue.metricValueInt64 > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    case CUPTI_METRIC_VALUE_KIND_PERCENT:
        SUBDBG("Metric percent = %f%%\n", metricValue.metricValuePercent);
        tmpValue.ll = (long long)(metricValue.metricValuePercent*100);
        //CHECK_PRINT_EVAL(tmpValue.ll - metricValue.metricValuePercent > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        SUBDBG("Metric throughput %llu bytes/sec\n", (unsigned long long) metricValue.metricValueThroughput);
        tmpValue.ll = (long long) (metricValue.metricValueThroughput);
        CHECK_PRINT_EVAL(tmpValue.ll - metricValue.metricValueThroughput > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        SUBDBG("Metric utilization level %u\n", (unsigned int) metricValue.metricValueUtilizationLevel);
        tmpValue.ll = (long long) (metricValue.metricValueUtilizationLevel);
        CHECK_PRINT_EVAL(tmpValue.ll - metricValue.metricValueUtilizationLevel > 1e-6, "Error converting metric\n", return (PAPI_EMISC));
        break;
    default:
        CHECK_PRINT_EVAL(1, "ERROR: unsupported metric value kind", return (PAPI_EINVAL));
        exit(-1);
    }

    *papiValue = tmpValue.ll;
    return (PAPI_OK);
} // end routine


/* ****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 **************************************************************************** */

/*
 * This is called whenever a thread is initialized.
 */
static int _cuda_init_thread(hwd_context_t * ctx)
{
    (void) ctx;
    SUBDBG("Entering\n");

    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
/* NOTE: only called by main thread (not by every thread) !!! Starting
   in CUDA 4.0, multiple CPU threads can access the same CUDA
   context. This is a much easier programming model then pre-4.0 as
   threads - using the same context - can share memory, data,
   etc. It's possible to create a different context for each
   thread. That's why CUDA context creation is done in
   CUDA_init_component() (called only by main thread) rather than
   CUDA_init() or CUDA_init_control_state() (both called by each
   thread). */
static int _cuda_init_component(int cidx)
{
    SUBDBG("Entering with component idx: %d\n", cidx);

    _cuda_vector.cmp_info.CmpIdx = cidx;

    _cuda_vector.cmp_info.num_native_events = -1;
    _cuda_vector.cmp_info.num_cntrs = -1;
    // num_mpx_cntrs must be >0 for _papi_hwi_assign_eventset() to work.
    _cuda_vector.cmp_info.num_mpx_cntrs = PAPICUDA_MAX_COUNTERS;

    // Count if we have any devices with vendor ID for Nvidia.
    int devices = _cuda_count_dev_sys();
    if (0) fprintf(stderr, "%s:%i Found %d Nvidia devices.\n", __func__, __LINE__, devices);
    if (devices < 1) {
        devices = _cuda_count_dev_proc();
        if (devices < 1) {
            _cuda_vector.cmp_info.initialized = 1;
            _cuda_vector.cmp_info.disabled = PAPI_ENOSUPP;
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                    "No Nvidia Devices Found.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            (void) strErr;
            return(PAPI_ENOSUPP);
        }
    }

    sprintf(_cuda_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    _cuda_vector.cmp_info.disabled = PAPI_EDELAY_INIT;

    PAPI_unlock(COMPONENT_LOCK);

    return PAPI_EDELAY_INIT;
} // END _cuda_init_component.

// This is the "delayed initialization", called when the application user of
// PAPI calls any API function. This prevents long initialization times and
// memory usage for systems where PAPI is configured with the cuda component
// but not all applications use the cuda component.
int _cuda_init_private(void)
{
    int rv, err = PAPI_OK;
    // The entire init, for cupti11, timed at 913 ms.
    // The entire init, for legalcy cupti, timed at 2376 ms.

    if (_cuda_vector.cmp_info.initialized) {
        // copy any previous disabled error code.
        err = _cuda_vector.cmp_info.disabled;
        goto cuda_init_private_exit;
    }

    long long ns;
    if (0) ns = -PAPI_get_real_nsec();
    SUBDBG("Private init with component idx: %d\n", _cuda_vector.cmp_info.CmpIdx);

    if(_cuda_linkCudaLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of CUDA libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        err = (PAPI_ENOSUPP);
        goto cuda_init_private_exit;
    }

    /* Create the structure */
    if(!global_cuda_context) {
        global_cuda_context = (cuda_context_t *) papi_calloc(1, sizeof(cuda_context_t));
        if (global_cuda_context == NULL) {
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "Could not allocate %lu bytes of memory for global_cuda_context.", sizeof(cuda_context_t));
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            err = (PAPI_ENOMEM);
            goto cuda_init_private_exit;
        }
    }

    /* Get list of all native CUDA events supported */
    rv = _cuda_add_native_events(global_cuda_context);
    if(rv != 0) {
        err = (rv);
        goto cuda_init_private_exit;
    }
    /* Export some information */
    _cuda_vector.cmp_info.num_native_events = global_cuda_context->availEventSize;
    _cuda_vector.cmp_info.num_cntrs = _cuda_vector.cmp_info.num_native_events;
    // We do NOT CHANGE num_mpx_cntrs. We set that to PAPICUDA_MAX_COUNTERS, which
    // is the maximum counters in an EventSet. The memory for that is already 
    // allocated; if we increase num_mpx_cntrs beyond that we can get a segfault
    // in the PAPI side of PAPI_cleanup_eventset. -TonyC
    err = PAPI_OK;

cuda_init_private_exit:
    _cuda_vector.cmp_info.initialized = 1;
    _cuda_vector.cmp_info.disabled = err;

    PAPI_unlock(COMPONENT_LOCK);

    // the entire init, for cupti11, timed at 913 ms.
    // the entire init, for legacy cupti, timed at 2376 ms.
    if (0) {
        ns += PAPI_get_real_nsec();
        fprintf(stderr, "%s:%s:%i Duration ns=%lld.\n", __FILE__, __func__, __LINE__, ns);
    }

    // We double check; if err != 0 and the disabled reason is null, we have a problem.
    if (err != 0 && strlen(_cuda_vector.cmp_info.disabled_reason) < 1) {
            int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
                "CUDA init failed. Code failed to record a reason.");
            _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
    }

    return (err);
} // end _cuda_init_private


/* Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int _cuda_init_control_state(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    DO_SOME_CHECKING(&_cuda_vector);
    #if CUPTI_API_VERSION >= 13
    // If the function pointer has changed, pass to cupti11 version.
    if (_cuda_vector.init_control_state != _cuda_init_control_state) {
        return(_cuda11_init_control_state(ctrl));
    }
    #endif

    cuda_context_t *gctxt = global_cuda_context;

    CHECK_PRINT_EVAL(!gctxt, "Error: The PAPI CUDA component needs to be initialized first", return (PAPI_ENOINIT));
    /* If no events were found during the initial component initialization, return error */
    if(global_cuda_context->availEventSize <= 0) {
        strncpy(_cuda_vector.cmp_info.disabled_reason, "ERROR CUDA: No events exist", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }
    /* If it does not exist, create the global structure to hold CUDA contexts and active events */
    if(!global_cuda_control) {
        global_cuda_control = (cuda_control_t *) papi_calloc(1, sizeof(cuda_control_t));
        global_cuda_control->countOfActiveCUContexts = 0;
        global_cuda_control->activeEventCount = 0;
    }

    return PAPI_OK;
} // end cuda_init_control_state

/* Triggered by eventset operations like add or remove.  For CUDA, needs to be
 * called multiple times from each separate CUDA context with the events to be
 * measured from that context.  For each context, create eventgroups for the
 * events.
 */

/* Note: NativeInfo_t is defined in papi_internal.h */
static int _cuda_update_control_state(hwd_control_state_t * ctrl,
    NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    SUBDBG("Entering with nativeCount %d\n", nativeCount);
    (void) ctx;
    DO_SOME_CHECKING(&_cuda_vector);

    cuda_control_t *gctrl = global_cuda_control;    // We don't use the passed-in parameter, we use a global.
    cuda_context_t *gctxt = global_cuda_context;    // We don't use the passed-in parameter, we use a global.
    int currDeviceNum;
    CUcontext currCuCtx;
    int eventContextIdx;
    CUcontext eventCuCtx;
    int index, ii, ee, cc;

    /* Return if no events */
    if(nativeCount == 0)
        return (PAPI_OK);

    // Get deviceNum.
    CUDA_CALL((*cudaGetDevicePtr) (&currDeviceNum), return (PAPI_EMISC));
    SUBDBG("currDeviceNum %d \n", currDeviceNum);

    // cudaFree(NULL) does nothing real, but initializes a new cuda context
    // if one does not exist. This prevents cuCtxGetCurrent() from failing.
    // If it returns an error, we ignore it.
    CUDA_CALL((*cudaFreePtr) (NULL), );
    CU_CALL((*cuCtxGetCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    SUBDBG("currDeviceNum %d cuCtx %p \n", currDeviceNum, currCuCtx);

    /* Handle user request of events to be monitored */
    for (ii = 0; ii < nativeCount; ii++) {                                  // For each event provided by caller,
        index              = nativeInfo[ii].ni_event;                       // Get the index of the event (in the global context).
        char *eventName    = gctxt->availEventDesc[index].name;             // Shortcut to name.
        int numMetricEvents= gctxt->availEventDesc[index].numMetricEvents;  // Get if this is an event (=0) or metric (>0).
        int eventDeviceNum = gctxt->availEventDeviceNum[index];             // Device number for this event.
        (void) eventName;                                                   // Useful in checkpoint and debug, don't warn if not used.

        /* if this event is already added continue to next ii, if not, mark it as being added */
        _papi_hwi_lock( COMPONENT_LOCK );
        if (gctxt->availEventIsBeingMeasuredInEventset[index] == 1) {       // If already being collected, skip it.
            SUBDBG("Skipping event %s which is already added\n", eventName);
            _papi_hwi_unlock( COMPONENT_LOCK );
            continue;
        } else {
            gctxt->availEventIsBeingMeasuredInEventset[index] = 1;          // If not being collected yet, flag it as being collected now.
        }

        /* Find context/control in papicuda, creating it if does not exist */
        for(cc = 0; cc < (int) gctrl->countOfActiveCUContexts; cc++) {              // Scan all active contexts.
            CHECK_PRINT_EVAL(cc >= PAPICUDA_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (PAPICUDA_MAX_COUNTERS)", 
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

            if(gctrl->arrayOfActiveCUContexts[cc]->deviceNum == eventDeviceNum) {   // If this cuda context is for the device for this event,
                eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;             // Remember that context.
                SUBDBG("Event %s device %d already has a cuCtx %p registered\n", eventName, eventDeviceNum, eventCuCtx);

                if(eventCuCtx != currCuCtx)                                         // If that is not our CURRENT context, push and make it so.
                    CU_CALL((*cuCtxSetCurrentPtr) (eventCuCtx),                     // .. Set as current.
                        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));     // .. .. on failure.
                break;                                                              // .. exit the loop.
            } // end if found.
        } // end loop through active contexts.

        if(cc == (int) gctrl->countOfActiveCUContexts) {                            // If we never found the context, create one.
            SUBDBG("Event %s device %d does not have a cuCtx registered yet...\n", eventName, eventDeviceNum);
            if(currDeviceNum != eventDeviceNum) {                           // .. If we need to switch to another device,
                CUDA_CALL((*cudaSetDevicePtr) (eventDeviceNum),             // .. .. set the device pointer to the event's device.
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC)); // .. .. .. (on failure).
                CUDA_CALL((*cudaFreePtr) (NULL), );                           // .. .. ignore any error; used to force init of a context..
                CU_CALL((*cuCtxGetCurrentPtr) (&eventCuCtx),                // .. .. So we can get a pointer to it.
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC)); // .. .. .. On failure.
            } else {                                                        // .. If we are already on the right device,
                eventCuCtx = currCuCtx;                                     // .. .. just get the current context.
            }

            gctrl->arrayOfActiveCUContexts[cc] = papi_calloc(1, sizeof(cuda_active_cucontext_t));   // allocate a structure.
            CHECK_PRINT_EVAL(gctrl->arrayOfActiveCUContexts[cc] == NULL, "Memory allocation for new active context failed", 
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_ENOMEM));
            gctrl->arrayOfActiveCUContexts[cc]->deviceNum = eventDeviceNum; // Fill in everything.
            gctrl->arrayOfActiveCUContexts[cc]->cuCtx = eventCuCtx;
            gctrl->arrayOfActiveCUContexts[cc]->allEventsCount = 0;         // All events read by this context on this device.
            gctrl->arrayOfActiveCUContexts[cc]->ctxActiveCount = 0;         // active events being read by this context on this device.
            gctrl->countOfActiveCUContexts++;
            SUBDBG("Added a new context deviceNum %d cuCtx %p ... now countOfActiveCUContexts is %d\n", eventDeviceNum, eventCuCtx, gctrl->countOfActiveCUContexts);
        } // end if we needed to create a new context.

        //---------------------------------------------------------------------
        // We found the context, or created it, and the index is in cc.
        //---------------------------------------------------------------------
        eventContextIdx = cc;
        cuda_active_cucontext_t *eventctrl = gctrl->arrayOfActiveCUContexts[eventContextIdx];   // get the context for this event.

        // We need to get all the events (or sub-events of a metric) and add
        // them to our list of all events. Note we only check if we exceed the
        // bounds of the allEvents[] array; everything added to any other array
        // results in at least ONE add to allEvents[], so it will fail before
        // or coincident with any other array. TC

        CUpti_EventID itemId = gctxt->availEventIDArray[index];                 // event (or metric) ID.

        if (numMetricEvents == 0) {                                             // Dealing with a simple event.
            eventctrl->allEvents[eventctrl->allEventsCount++] = itemId;         // add to aggregate list, count it.
            if (eventctrl->allEventsCount >= PAPICUDA_MAX_COUNTERS) {           // .. Fail if we exceed size of array.
                SUBDBG("Num events (generated by metric) exceeded PAPICUDA_MAX_COUNTERS\n");
                _papi_hwi_unlock( COMPONENT_LOCK ); return(PAPI_EINVAL);
            }
        } else {                                                                // dealing with a metric.
            // cuda events and metrics have already been skipped if duplicates,
            // but we can't say the same for sub-events of a metric. We need to
            // check we don't duplicate them in allEvents.

            for(ee = 0; ee < numMetricEvents; ee++) {                           // For each event retrieved,
                int aeIdx;
                CUpti_EventID myId = gctxt->availEventDesc[index].metricEvents[ee]; // collect the sub-event ID.

                for (aeIdx=0; aeIdx<(int) eventctrl->allEventsCount; aeIdx++) {     // loop through existing events.
                    if (eventctrl->allEvents[aeIdx] == myId) break;                 // break out if duplicate found.
                }

                if (aeIdx < (int) eventctrl->allEventsCount) continue;              // Don't add if already present.
                eventctrl->allEvents[eventctrl->allEventsCount++] = myId;           // add event to the all array.

                if (eventctrl->allEventsCount >= PAPICUDA_MAX_COUNTERS) {       // Fail if we exceed size of array.
                    SUBDBG("Num events (generated by metric) exceeded PAPICUDA_MAX_COUNTERS\n");
                    _papi_hwi_unlock( COMPONENT_LOCK ); return(PAPI_EINVAL);
                }
            } // end for each event in metric.
        } // end if we must process all sub-events of a metric.

        // Record index of this active event back into the nativeInfo
        // structure.

        nativeInfo[ii].ni_position = gctrl->activeEventCount;

        // Record index of this active event within this context. We need this
        // so after we read this context, we can move values (or compute
        // metrics and move values) into their proper position within the
        // activeValues[] array.

        eventctrl->ctxActiveEvents[eventctrl->ctxActiveCount++] =       // within this active_cucontext.
            gctrl->activeEventCount;                                    // ..

        // Record in internal gctrl arrays.
        // so we have a succinct list of active events and metrics; this will
        // be useful for performance especially on metrics, where we must
        // compose values.

        CHECK_PRINT_EVAL(gctrl->activeEventCount == PAPICUDA_MAX_COUNTERS - 1, "Exceeded maximum num of events (PAPI_MAX_COUNTERS)", return (PAPI_EMISC));
        gctrl->activeEventIndex[gctrl->activeEventCount] = index;
        gctrl->activeEventValues[gctrl->activeEventCount] = 0;
        gctrl->activeEventCount++;

        // EventGroupSets does an analysis to creates 'sets' of events that
        // can be collected simultaneously, i.e. the application must be
        // run once per set. CUpti calls these 'passes'. We don't allow
        // such combinations, there is no way to tell a PAPI user to run
        // their application multiple times.  WITHIN a single set are
        // EventGroups which are collected simultaneously but must be read
        // separately because each group applies to a separate domain.  So
        // we don't mind that; but we must exit with an invalid combination
        // if numsets > 1, indicating the most recent event requested
        // cannot be collected simultaneously with the others.

        // We destroy any existing eventGroupSets, and then create one for the
        // new set of events.

        SUBDBG("Create eventGroupSets for context (destroy pre-existing) (nativeCount %d, allEventsCount %d) \n", gctrl->activeEventCount, eventctrl->allEventsCount);
        if(eventctrl->allEventsCount > 0) {                                         // If we have events...
            // SUBDBG("Destroy previous eventGroupPasses for the context \n");
            if(eventctrl->eventGroupSets != NULL) {                                 // if we have a previous analysis;
                CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr)                         // .. Destroy it.
                    (eventctrl->eventGroupSets), 
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. If we can't, return error.
                eventctrl->eventGroupSets = NULL;                                   // .. Reset pointer.
            }

            size_t sizeBytes = (eventctrl->allEventsCount) * sizeof(CUpti_EventID); // compute bytes in the array.

            // SUBDBG("About to create eventGroupPasses for the context (sizeBytes %zu) \n", sizeBytes);
#ifdef PAPICUDA_KERNEL_REPLAY_MODE
            CUPTI_CALL((*cuptiEnableKernelReplayModePtr) (eventCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_ECMP));
            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr)
                (eventCuCtx, sizeBytes, eventctrl->allEvents,
                &eventctrl->eventGroupSets),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_ECMP));

#else // Normal operation.
            // Note: We no longer fail if this collection mode does not work. It will only work
            // on TESLA devices, and is desirable there (not restricted to the kernel). But it
            // is not available on other models (including GTX) and we shouldn't fail without it.
            CUPTI_CALL((*cuptiSetEventCollectionModePtr)
                (eventCuCtx,CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS), );

// CUPTI provides two routines to create EventGroupSets, one is used
// here cuptiEventGroupSetsCreate(), the other is for metrics, it will
// automatically collect the events needed for a metric. It is called
// cuptiMetricCreateEventGroupSets(). We have checked and these two routines
// produce groups of the same size with the same event IDs, and work equally.

            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr)
                (eventCuCtx, sizeBytes, eventctrl->allEvents,
                &eventctrl->eventGroupSets),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

            if (eventctrl->eventGroupSets->numSets > 1) {                       // If more than one pass is required,
                SUBDBG("Error occurred: The combined CUPTI events cannot be collected simultaneously ... try different events\n");
                _cuda_cleanup_eventset(ctrl);                                // Will do cuptiEventGroupSetsDestroy() to clean up memory.
                _papi_hwi_unlock( COMPONENT_LOCK ); return(PAPI_ECOMBO);
            } else  {
                SUBDBG("Created eventGroupSets. nativeCount %d, allEventsCount %d. Sets (passes-required) = %d) \n", gctrl->activeEventCount, eventctrl->allEventsCount, eventctrl->eventGroupSets->numSets);
            }

#endif // #if/#else/#endif on PAPICUDA_KERNEL_REPLAY_MODE

        } // end if we had any events.

        if(eventCuCtx != currCuCtx)                                                 // restore original caller context if we changed it.
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), 
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    }
    _papi_hwi_unlock( COMPONENT_LOCK );
    return (PAPI_OK);
} // end PAPI_update_control_state.


// Triggered by PAPI_start().
// For CUDA component, switch to each context and start all eventgroups.
static int _cuda_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    uint32_t ii, gg, cc;
    int saveDeviceNum = -1;

    SUBDBG("Reset all active event values\n");
    // Zeroing values for the local read.
    _papi_hwi_lock( COMPONENT_LOCK );
    for(ii = 0; ii < gctrl->activeEventCount; ii++) {
        gctrl->activeEventValues[ii] = 0;
    }

    // Zeroing cumulative values at start.
    for(ii = 0; ii < gctxt->numAllEvents; ii++) {
        gctxt->allEvents[ii].cumulativeValue = 0;
    }

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    CUPTI_CALL((*cuptiGetTimestampPtr) (&gctrl->cuptiStartTimestampNs),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {                    // For each context,
        int eventDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;     // .. get device number.
        CUcontext eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;       // .. get this context,
        SUBDBG("Set to device %d cuCtx %p \n", eventDeviceNum, eventCuCtx);
        if(eventDeviceNum != saveDeviceNum) {                                   // .. If we need to switch,
            CU_CALL((*cuCtxPushCurrentPtr) (eventCuCtx),                        // .. .. push current on stack, use this one.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));
        }

        CUpti_EventGroupSets *eventGroupSets =                                  // .. Shortcut to eventGroupSets for this context.
            gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;                 // ..
            CUpti_EventGroupSet *groupset = &eventGroupSets->sets[0];           // .. There can be only one set of groups.
            for(gg = 0; gg < groupset->numEventGroups; gg++) {                  // .. For each group within this groupset,
                uint32_t one = 1;
                CUPTI_CALL((*cuptiEventGroupSetAttributePtr) (                  // .. .. Say we want to profile all domains.
                    groupset->eventGroups[gg],
                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                    sizeof(uint32_t), &one),
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));     // .. .. on failure of call.
            } // end for each group.

            CUPTI_CALL((*cuptiEventGroupSetEnablePtr) (groupset),               // .. Enable all groups in set (start collecting).
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. .. on failure of call.

        if(eventDeviceNum != saveDeviceNum) {                                   // .. If we pushed a context,
            CU_CALL((*cuCtxPopCurrentPtr) (&eventCuCtx),                        // .. Pop it.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. .. on failure of call.
        }
    } // end of loop on all contexts.

    _papi_hwi_unlock( COMPONENT_LOCK );
    return (PAPI_OK);                                                           // We started all groups.
} // end routine.

// Triggered by PAPI_read().  For CUDA component, switch to each context, read
// all the eventgroups, and put the values in the correct places. Note that
// parameters (ctx, ctrl, flags) are all ignored. The design of this components
// doesn't pay attention to PAPI EventSets, because ONLY ONE is ever allowed
// for a component.  So instead of maintaining ctx and ctrl, we use global
// variables to keep track of the one and only eventset.  Note that **values is
// where we have to give PAPI the address of an array of the values we read (or
// composed).
// ALSO note, cuda resets all event counters to zero after a read, while PAPI
// promises monotonically increasing counters (from PAPI_start()). So we have
// to synthesize that.

static int _cuda_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    (void) flags;
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    uint32_t gg, i, j, cc;
    int saveDeviceNum;

    _papi_hwi_lock( COMPONENT_LOCK );
    // Get read time stamp
    CUPTI_CALL((*cuptiGetTimestampPtr)                                          // Read current timestamp.
        (&gctrl->cuptiReadTimestampNs),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));
    uint64_t durationNs = gctrl->cuptiReadTimestampNs -
                          gctrl->cuptiStartTimestampNs;                         // compute duration from start.
    gctrl->cuptiStartTimestampNs = gctrl->cuptiReadTimestampNs;                 // Change start to value just read.

    SUBDBG("Save current context, then switch to each active device/context and enable context-specific eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // Save Caller's current device number on entry.

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {                    // For each active context,
        cuda_active_cucontext_t *activeCuCtxt =
            gctrl->arrayOfActiveCUContexts[cc];                                 // A shortcut.
        int currDeviceNum = activeCuCtxt->deviceNum;                            // Get the device number.
        CUcontext currCuCtx = activeCuCtxt->cuCtx;                              // Get the actual CUcontext.

        SUBDBG("Set to device %d cuCtx %p \n", currDeviceNum, currCuCtx);
        if(currDeviceNum != saveDeviceNum) {                                    // If my current is not the same as callers,
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx),                         // .. Push the current, and replace with mine.
               _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));        // Note, cuCtxPushCurrent()  implicitly includes a cudaSetDevice().
        } else {                                                                // If my current IS the same as callers,
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx),                          // .. No push. Just set the current.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. .. on failure of call.
        }

        CU_CALL((*cuCtxSynchronizePtr) (),                                      // Block until device finishes all prior tasks.
            _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));           // .. on failure of call.
        CUpti_EventGroupSets *myEventGroupSets =  activeCuCtxt->eventGroupSets; // Make a copy of pointer to EventGroupSets.

        uint32_t numEvents, numInstances, numTotalInstances;
        size_t sizeofuint32num = sizeof(uint32_t);
        CUpti_EventDomainID groupDomainID;
        size_t groupDomainIDSize = sizeof(groupDomainID);
        CUdevice cudevice = gctxt->deviceArray[currDeviceNum].cuDev;            // Make a copy of the current device.

        // For each pass, we get the event groups that can be read together.
        // But since elsewhere, we don't allow events to be added that would
        // REQUIRE more than one pass, this will always be just ONE pass. So we
        // only need to loop over the groups.

        CUpti_EventGroupSet *groupset = &myEventGroupSets->sets[0];             // The one and only set.
        SUBDBG("Read events in this context\n");
        int AEIdx = 0;                                                          // we will be over-writing the allEvents array.

        for (gg = 0; gg < groupset->numEventGroups; gg++) {                     // process each eventgroup within the groupset.
            CUpti_EventGroup group = groupset->eventGroups[gg];                 // Shortcut to the group.

            CUPTI_CALL((*cuptiEventGroupGetAttributePtr)                        // Get 'groupDomainID' for this group.
                (group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                &groupDomainIDSize, &groupDomainID),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. on failure of call.

            // 'numTotalInstances' and 'numInstances are needed for scaling
            // the values retrieved. (Nvidia instructions and samples).
            CUPTI_CALL((*cuptiDeviceGetEventDomainAttributePtr)                 // Get 'numTotalInstances' for this domain.
                (cudevice,
                groupDomainID,
                CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                &sizeofuint32num,
                &numTotalInstances),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. on failure of call.

            CUPTI_CALL((*cuptiEventGroupGetAttributePtr)                        // Get 'numInstances' for this domain.
                (group,
                CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                &sizeofuint32num,
                &numInstances),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. on failure of call.

            CUPTI_CALL((*cuptiEventGroupGetAttributePtr)                        // Get 'numEvents' in this group.
                (group,
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &sizeofuint32num,
                &numEvents),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. on failure of call.

            // Now we will read all events in this group; aggregate the values
            // and then distribute them.  We do not calculate metrics here;
            // wait until all groups are read and all values are available.

            size_t resultArrayBytes        = sizeof(uint64_t) * numEvents * numTotalInstances;
            size_t eventIdArrayBytes       = sizeof(CUpti_EventID) * numEvents;
            size_t numCountersRead         = 2;

            CUpti_EventID *eventIdArray = (CUpti_EventID *) papi_malloc(eventIdArrayBytes);
            uint64_t *resultArray       = (uint64_t *)      papi_malloc(resultArrayBytes);
            uint64_t *aggrResultArray   = (uint64_t *)      papi_calloc(numEvents, sizeof(uint64_t));

            if (eventIdArray == NULL || resultArray == NULL || aggrResultArray == NULL) {
                fprintf(stderr, "%s:%i failed to allocate memory.\n", __FILE__, __LINE__);
                _papi_hwi_unlock( COMPONENT_LOCK );                             // .. on failure of malloc.
                return(PAPI_EMISC);
            }

            for (i=0; i<(resultArrayBytes/sizeof(uint64_t)); i++) resultArray[i]=0;

            CUPTI_CALL( (*cuptiEventGroupReadAllEventsPtr)                      // Read all events.
                (group, CUPTI_EVENT_READ_FLAG_NONE,                             // This flag is the only allowed flag.
                &resultArrayBytes, resultArray,
                &eventIdArrayBytes, eventIdArray,
                &numCountersRead),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. on failure of call.

            // Now (per Nvidia) we must sum up all domains for each event.
            // Arrangement of 2-d Array returned in resultArray:
            //    domain instance 0: event0 event1 ... eventN
            //    domain instance 1: event0 event1 ... eventN
            //    ...
            //    domain instance M: event0 event1 ... eventN
            // But we accumulate by column, event[0], event[1], etc.

            for (i = 0; i < numEvents; i++) {                                   // outer loop is column (event) we are on.
                for (j = 0; j < numTotalInstances; j++) {                       // inner loop is row (instance) we are on.
                    aggrResultArray[i] += resultArray[i + numEvents * j];       // accumulate the column.
                }
            }

            // We received an eventIdArray; note this is not necessarily in the
            // same order as we added them; CUpti can reorder them when sorting
            // them into groups.  However, the total number of events must be
            // the same, so now as we read each group, we just overwrite the
            // allEvents[] and allEventValues[] arrays. It doesn't make a
            // difference to cuptiGetMetricValue what order the events appear
            // in.

            // After all these groups are read, allEvents will be complete, and
            // we can use it to compute the metrics and move metric and event
            // values back into user order.

            for (i=0; i<numEvents; i++) {                                               // For each event in eventIdArray (just this group),
                CUpti_EventID myId = eventIdArray[i];                                   // shortcut for the event id within this group.
                activeCuCtxt->allEvents[AEIdx] = myId;                                  // Overwrite All Events id.
                activeCuCtxt->allEventValues[AEIdx++] = aggrResultArray[i];             // Overwrite all events value; increment position.
            } // end loop for each event.

            papi_free(eventIdArray);
            papi_free(resultArray);
            papi_free(aggrResultArray);
        } // end of an event group.

        // We have finished all event groups within this context; allEvents[]
        // and allEventValues[] are populated. Now we must update the
        // cumulative totals.
        for (i=0; i<(unsigned) AEIdx; i++) {
            CUpti_EventID myId = activeCuCtxt->allEvents[i]; 
            int myIdx = _search_all_events(gctxt, myId, currDeviceNum);
            if (myIdx < 0) {
                fprintf(stderr, "Failed to find event 0x%08X device=%d.\n", myId, currDeviceNum);
                continue;
            }

            if (gctxt->allEvents[myIdx].nonCumulative == 0) {
                long unsigned int temp = gctxt->allEvents[myIdx].cumulativeValue;
                gctxt->allEvents[myIdx].cumulativeValue += activeCuCtxt->allEventValues[i];
                if (gctxt->allEvents[myIdx].cumulativeValue < temp) {
                    if (0) fprintf(stderr, "%s:%s:%i temp=%ld, value=%ld, result=%ld.\n", __FILE__, __func__, __LINE__, 
                    temp, gctxt->allEvents[myIdx].cumulativeValue, activeCuCtxt->allEventValues[i]);
                }
                activeCuCtxt->allEventValues[i] = gctxt->allEvents[myIdx].cumulativeValue;
            }
        }
        // Now we compute metrics and move event values. We do that by looping
        // through the events assigned to this context, and we must back track
        // to the activeEventIdx[] and activeEventValues[] array in gctrl. We
        // have kept our indexes into that array, in ctxActive[].

        uint32_t ctxActiveCount =  activeCuCtxt->ctxActiveCount;                // Number of (papi user) events in this context.
        uint32_t *ctxActive =  activeCuCtxt->ctxActiveEvents;                   // index of each event in gctrl->activeEventXXXX.

        for (j=0; j<ctxActiveCount; j++) {                                      // Search for matching active event.
            uint32_t activeIdx, availIdx;

            activeIdx=ctxActive[j];                                             // get index into activeEventIdx.
            availIdx = gctrl->activeEventIndex[activeIdx];                      // Get the availEventIdx.
            CUpti_EventID thisEventId = gctxt->availEventIDArray[availIdx];     // Get the event ID (or metric ID).
            struct cuda_name_desc *myDesc=&(gctxt->availEventDesc[availIdx]);   // get pointer to the description.

            if (myDesc->numMetricEvents == 0) {                                 // If this is a simple cuda event (not a metric),
                int k;
                for (k=0; k<AEIdx; k++) {                                       // search the array for this event id.
                    if (activeCuCtxt->allEvents[k] == thisEventId) {            // If I found the event,
                        gctrl->activeEventValues[activeIdx] =                   // Record the value,
                            activeCuCtxt->allEventValues[k];
                        break;                                                  // break out of the search loop.
                    } // end if I found it.
                } // end search loop.

                continue;                                                       // Jump to next in ctxActiveCount.
            } else {                                                            // If I found a metric, I must compute it.
                CUpti_MetricValue myValue;                                      // Space for a return.
                CUPTI_CALL( (*cuptiMetricGetValuePtr)                           // Get the value,
                    (cudevice, thisEventId,                                     // device and metric Id,
                    AEIdx * sizeof(CUpti_EventID),                              // size of event list,
                    activeCuCtxt->allEvents,                                    // the event list.
                    AEIdx * sizeof(uint64_t),                                   // size of corresponding event values,
                    activeCuCtxt->allEventValues,                               // the event values.
                    durationNs, &myValue),                                      // duration (for rates), and where to return the value.
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));   // .. on failure of call.

                _cuda_convert_metric_value_to_long_long(                        // convert the value computed to long long and store it.
                    myValue, myDesc->MV_Kind,
                    &gctrl->activeEventValues[activeIdx]);
            }
        } // end loop on active events in this context.

        if(currDeviceNum != saveDeviceNum) {                                    // If we had to change the context from user's,
            CUDA_CALL((*cudaSetDevicePtr) (saveDeviceNum),                      // set the device pointer to the user's original.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. .. (on failure).
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),                         // .. pop the pushed context back to user's.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));       // .. .. on failure of call.
        }
    } // end of loop for each active context.

    *values = gctrl->activeEventValues;                                         // Return ptr to the list of computed values to user.
    _papi_hwi_unlock( COMPONENT_LOCK );                                         // Done with reading.
    return (PAPI_OK);
} // end of cuda_read().

/* Triggered by PAPI_stop() */
static int _cuda_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    cuda_control_t *gctrl = global_cuda_control;
    uint32_t cc, ss;
    int saveDeviceNum;

    SUBDBG("Save current context, then switch to each active device/context and disable eventgroups\n");
    _papi_hwi_lock( COMPONENT_LOCK );
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        SUBDBG("Set to device %d cuCtx %p \n", currDeviceNum, currCuCtx);
        if(currDeviceNum != saveDeviceNum) {
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        } else {
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        }

        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;
        for (ss=0; ss<currEventGroupSets->numSets; ss++) {                      // For each group in the set,
            CUpti_EventGroupSet groupset = currEventGroupSets->sets[ss];        // get the set,
            CUPTI_CALL((*cuptiEventGroupSetDisablePtr) (&groupset),             // disable the whole set.
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        }

        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum) {
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        } 

    }

    _papi_hwi_unlock( COMPONENT_LOCK );
    return (PAPI_OK);
} // end of cuda_stop.


/*
 * Disable and destroy the CUDA eventGroup
 */
static int _cuda_cleanup_eventset(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;                                                    // Don't need this parameter.
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    // cuda_active_cucontext_t *currctrl;
    uint32_t cc;
    int saveDeviceNum;
    unsigned int ui;
    CUcontext saveCtx;  

    SUBDBG("Save current device/context, then switch to each active device/context and enable eventgroups\n");
    _papi_hwi_lock( COMPONENT_LOCK );
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.
    CU_CALL((*cuCtxGetCurrentPtr) (&saveCtx),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        CUDA_CALL((*cudaSetDevicePtr) (currDeviceNum),
            _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));             // .. on failure of call.
        CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx),
            _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));             // .. on failure of call.
        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;

        //CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr) (currEventGroupPasses), return (PAPI_EMISC));
        (*cuptiEventGroupSetsDestroyPtr) (currEventGroupSets);
        gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets = NULL;
        papi_free( gctrl->arrayOfActiveCUContexts[cc] );
    }
    /* Restore saved context, device pointer */
    CU_CALL((*cuCtxSetCurrentPtr) (saveCtx),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.
    CUDA_CALL((*cudaSetDevicePtr) (saveDeviceNum),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.

    /* Record that there are no active contexts or events */
    for (ui=0; ui<gctrl->activeEventCount; ui++) {              // For each active event,
        int idx = gctrl->activeEventIndex[ui];                  // .. Get its index...
        gctxt->availEventIsBeingMeasuredInEventset[idx] = 0;    // .. No longer being measured.
    }

    gctrl->countOfActiveCUContexts = 0;
    gctrl->activeEventCount = 0;
    _papi_hwi_unlock( COMPONENT_LOCK );
    return (PAPI_OK);
} // end cuda_cleanup_eventset


/* Called at thread shutdown. Does nothing in the CUDA component. */
static int _cuda_shutdown_thread(hwd_context_t * ctx)
{
    SUBDBG("Entering\n");
    (void) ctx;

    return (PAPI_OK);
}

/* Triggered by PAPI_shutdown() and frees memory allocated in the CUDA component. */
static int _cuda_shutdown_component(void)
{
    SUBDBG("Entering\n");
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    int deviceNum;
    uint32_t i, cc;
    /* Free context */
    if(gctxt) {
        for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
            cuda_device_desc_t *mydevice = &gctxt->deviceArray[deviceNum];
            papi_free(mydevice->domainIDArray);
            papi_free(mydevice->domainIDNumEvents);
        }

        for (i=0; i<gctxt->availEventSize; i++) {                               // For every event in this context,
            struct cuda_name_desc *desc = &(gctxt->availEventDesc[i]);      // get a name description.
            if (desc->numMetricEvents > 0) {                                    // If we have any sub-events,
                papi_free(desc->metricEvents);                                  // .. Free the list of sub-events.
            }
        } // end for every available event.

        papi_free(gctxt->availEventIDArray);
        papi_free(gctxt->availEventDeviceNum);
        papi_free(gctxt->availEventKind);
        papi_free(gctxt->availEventIsBeingMeasuredInEventset);
        papi_free(gctxt->availEventDesc);
        papi_free(gctxt->deviceArray);
        papi_free(gctxt);
        global_cuda_context = gctxt = NULL;
    }
    /* Free control */
    if(gctrl) {
        for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
#ifdef PAPICUDA_KERNEL_REPLAY_MODE
            CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
            CUPTI_CALL((*cuptiDisableKernelReplayModePtr) (currCuCtx), return (PAPI_EMISC));
#endif
            if(gctrl->arrayOfActiveCUContexts[cc] != NULL)
                papi_free(gctrl->arrayOfActiveCUContexts[cc]);
        }
        papi_free(gctrl);
        global_cuda_control = gctrl = NULL;
    }

    // close the dynamic libraries needed by this component (opened in the init substrate call)
    if (dl1) {
        dlclose(dl1);
    }
    if (dl2) {
        dlclose(dl2);
    }
    if (dl3) {
        dlclose(dl3);
    }

    return (PAPI_OK);
} // end cuda_shutdown_component().


/* Triggered by PAPI_reset() but only if the EventSet is currently
 *  running. If the eventset is not currently running, then the saved
 *  value in the EventSet is set to zero without calling this
 *  routine.  */
static int _cuda_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    (void) ctx;
    (void) ctrl;
    cuda_control_t *gctrl = global_cuda_control;
    uint32_t gg, ii, cc, ss;
    int saveDeviceNum;

    SUBDBG("Reset all active event values\n");
    _papi_hwi_lock( COMPONENT_LOCK );
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    SUBDBG("Save current context, then switch to each active device/context and reset\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));                 // .. on failure of call.
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        if(currDeviceNum != saveDeviceNum) {
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), 
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        } else {
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        }

        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;
        for (ss=0; ss<currEventGroupSets->numSets; ss++) {
            CUpti_EventGroupSet groupset = currEventGroupSets->sets[ss];
            for(gg = 0; gg < groupset.numEventGroups; gg++) {
                CUpti_EventGroup group = groupset.eventGroups[gg];
                CUPTI_CALL((*cuptiEventGroupResetAllEventsPtr) (group),
                    _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));     // .. on failure of call.
            }
            CUPTI_CALL((*cuptiEventGroupSetEnablePtr) (&groupset),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        }
        if(currDeviceNum != saveDeviceNum) {
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),
                _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));         // .. on failure of call.
        }
    }

    _papi_hwi_unlock( COMPONENT_LOCK );
    return (PAPI_OK);
} // end cuda_reset().


/* This function sets various options in the component - Does nothing in the CUDA component.
    @param[in] ctx -- hardware context
    @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
    @param[in] option -- options to be set
*/
static int _cuda_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) code;
    (void) option;
    return (PAPI_OK);
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
static int _cuda_set_domain(hwd_control_state_t * ctrl, int domain)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain))
        return (PAPI_OK);
    else
        return (PAPI_EINVAL);
    return (PAPI_OK);
}


/* Enumerate Native Events.
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 */
static int _cuda_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    DO_SOME_CHECKING(&_cuda_vector);
    #if CUPTI_API_VERSION >= 13
    // If the function pointer has changed, pass to cupti11 version.
    if (_cuda_vector.ntv_enum_events != _cuda_ntv_enum_events) {
        return(_cuda11_ntv_enum_events(EventCode, modifier));
    }
    #endif

    switch (modifier) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return (PAPI_OK);
        break;
    case PAPI_ENUM_EVENTS:
        if (global_cuda_context == NULL) {
            return (PAPI_ENOEVNT);
        } else if (*EventCode < global_cuda_context->availEventSize - 1) {
            *EventCode = *EventCode + 1;
            return (PAPI_OK);
        } else {
            return (PAPI_ENOEVNT);
        }
        break;
    default:
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
}


static int _cuda_ntv_name_to_code(const char *nameIn, unsigned int *out)
{
    DO_SOME_CHECKING(&_cuda_vector);
    #if CUPTI_API_VERSION >= 13
    // If the function pointer has changed, pass to cupti11 version.
    if (_cuda_vector.ntv_name_to_code != _cuda_ntv_name_to_code) {
        return(_cuda11_ntv_name_to_code(nameIn, out));
    }
    #endif

    (void) nameIn;
    (void) out;
    // Not supported by legacy cuda component.
    return(PAPI_ECMP); 
}

/* Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int _cuda_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    DO_SOME_CHECKING(&_cuda_vector);
    #if CUPTI_API_VERSION >= 13
    // If the function pointer has changed, pass to cupti11 version.
    if (_cuda_vector.ntv_code_to_name != _cuda_ntv_code_to_name) {
        return(_cuda11_ntv_code_to_name(EventCode, name, len));
    }
    #endif

    // SUBDBG( "Entering EventCode %d\n", EventCode );
    unsigned int index = EventCode;
    cuda_context_t *gctxt = global_cuda_context;
    if(gctxt != NULL && index < gctxt->availEventSize) {
        strncpy(name, gctxt->availEventDesc[index].name, len);
    } else {
        return (PAPI_EINVAL);
    }
    // SUBDBG( "Exit: EventCode %d: Name %s\n", EventCode, name );
    return (PAPI_OK);
}


/* Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int _cuda_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
    // SUBDBG( "Entering\n" );
    unsigned int index = EventCode;
    cuda_context_t *gctxt = global_cuda_context;
    if(gctxt != NULL && index < gctxt->availEventSize) {
        strncpy(name, gctxt->availEventDesc[index].description, len);
    } else {
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
}


/* Vector that points to entry points for the component */
papi_vector_t _cuda_vector = {
    .cmp_info = {
                 /* default component information (unspecified values are initialized to 0) */
                 .name = "cuda",
                 .short_name = "cuda",
                 .version = "5.1",
                 .description = "CUDA events and metrics via NVIDIA CuPTI interfaces",
                 .num_mpx_cntrs = PAPICUDA_MAX_COUNTERS,
                 .num_cntrs = PAPICUDA_MAX_COUNTERS,
                 .default_domain = PAPI_DOM_USER,
                 .default_granularity = PAPI_GRN_THR,
                 .available_granularities = PAPI_GRN_THR,
                 .hardware_intr_sig = PAPI_INT_SIGNAL,
                 /* component specific cmp_info initializations */
                 .fast_real_timer = 0,
                 .fast_virtual_timer = 0,
                 .attach = 0,
                 .attach_must_ptrace = 0,
                 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
                 .initialized = 0,
    }
    ,
    /* sizes of framework-opaque component-private structures... these are all unused in this component */
    .size = {
             .context = 1,      /* sizeof( cuda_context_t ), */
             .control_state = 1,        /* sizeof( cuda_control_t ), */
             .reg_value = 1,    /* sizeof( cuda_register_t ), */
             .reg_alloc = 1,    /* sizeof( cuda_reg_alloc_t ), */
             }
    ,
    /* function pointers in this component */
    .start = _cuda_start,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .stop = _cuda_stop,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .read = _cuda_read,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    .reset = _cuda_reset,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .cleanup_eventset = _cuda_cleanup_eventset,      /* ( hwd_control_state_t * ctrl ) */

    .init_component = _cuda_init_component,  /* ( int cidx ) */
    .init_thread = _cuda_init_thread,        /* ( hwd_context_t * ctx ) */
    .init_control_state = _cuda_init_control_state,  /* ( hwd_control_state_t * ctrl ) */
    .update_control_state = _cuda_update_control_state,      /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */

    .ctl = _cuda_ctrl,       /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    .set_domain = _cuda_set_domain,  /* ( hwd_control_state_t * cntrl, int domain ) */
    .ntv_enum_events = _cuda_ntv_enum_events,        /* ( unsigned int *EventCode, int modifier ) */
    .ntv_name_to_code = _cuda_ntv_name_to_code,      /* ( unsigned char *name, int *code ) */
    .ntv_code_to_name = _cuda_ntv_code_to_name,      /* ( unsigned int EventCode, char *name, int len ) */
    .ntv_code_to_descr = _cuda_ntv_code_to_descr,    /* ( unsigned int EventCode, char *name, int len ) */
    .shutdown_thread = _cuda_shutdown_thread,        /* ( hwd_context_t * ctx ) */
    .shutdown_component = _cuda_shutdown_component,  /* ( void ) */
};

//-------------------------------------------------------------------------------------------------
// This routine is an adaptation from 'readMetricValue' in nvlink_bandwidth_cupti_only.cu; where
// it is shown to work. Note that a metric can consist of more than one event, so the number of
// events and the number of metrics does not have to match.
// 'eventGroup' should contain the events needed to read the
// 'numEvents' is the number of events needed to read to compute the metrics.
// 'metricId' is the array of METRICS, and
// 'numMetrics" is the number of them, and also applies to the arrays 'values' and 'myKinds'.
// 'dev is the CUDevice needed to compute the metric. We don't need to switch the context, that is
// already done by the caller so we are pointing at the correct context.
//-------------------------------------------------------------------------------------------------
void readMetricValue(CUpti_EventGroup eventGroup,
                    uint32_t numEvents,                         // array COLS in results,
                    uint64_t numTotalInstances,                 // array ROWS in results,
                    CUdevice dev,                               // current Device structure.
                    uint32_t numMetrics,
                    CUpti_MetricID *metricId,
                    CUpti_MetricValueKind *myKinds,
                    long long int *values,
                    uint64_t timeDuration)
{
    size_t bufferSizeBytes, numCountersRead;
    uint64_t *eventValueArray = NULL;
    CUpti_EventID *eventIdArray;
    size_t arraySizeBytes = 0;
    uint64_t *aggrEventValueArray = NULL;
    size_t aggrEventValueArraySize;
    uint32_t i = 0, j = 0;

    arraySizeBytes = sizeof(CUpti_EventID) * numEvents;
    bufferSizeBytes = sizeof(uint64_t) * numEvents * numTotalInstances;

    eventValueArray = (uint64_t *) malloc(bufferSizeBytes);

    eventIdArray = (CUpti_EventID *) malloc(arraySizeBytes);

    aggrEventValueArray = (uint64_t *) calloc(numEvents, sizeof(uint64_t));

    aggrEventValueArraySize = sizeof(uint64_t) * numEvents;

    CUPTI_CALL( (*cuptiEventGroupReadAllEventsPtr)
                (eventGroup, CUPTI_EVENT_READ_FLAG_NONE, &bufferSizeBytes,
                 eventValueArray, &arraySizeBytes, eventIdArray, &numCountersRead),
                return);

    // Arrangement of 2-d Array returned in eventValueArray:
    //    domain instance 0: event0 event1 ... eventN
    //    domain instance 1: event0 event1 ... eventN
    //    ...
    //    domain instance M: event0 event1 ... eventN
    // But we accumulate by column, event[0], event[1], etc.

    for (i = 0; i < numEvents; i++) {                   // outer loop is column (event) we are on.
        for (j = 0; j < numTotalInstances; j++) {       // inner loop is row (instance) we are on.
            aggrEventValueArray[i] += eventValueArray[i + numEvents * j];
        }
    }

    // After aggregation, we use the data to compose the metrics.
    for (i = 0; i < numMetrics; i++) {
        CUpti_MetricValue metricValue;
        CUPTI_CALL( (*cuptiMetricGetValuePtr)
                    (dev, metricId[i], arraySizeBytes, eventIdArray,
                     aggrEventValueArraySize, aggrEventValueArray,
                     timeDuration, &metricValue),
                    return);

        _cuda_convert_metric_value_to_long_long(metricValue, myKinds[i], &values[i]);
    }

    free(eventValueArray);
    free(eventIdArray);
} // end readMetricValue.


#if CUPTI_API_VERSION >= 13
//*************************************************************************************************
//-------------------------------------------------------------------------------------------------
// CUPTI 11 routines; adapted from Thomas Gruber PerfWorks code.
// NOTE: CUPTI 11 cannot query counters directly! It has ONLY metrics. The "counters" are just
//       names of a collection of values, e.g. 'dram__bytes_read", but cannot be read directly;
//       there are several metrics based upon each, e.g.  dram__bytes_read.avg,
//       dram__bytes_read.max, dram__bytes_read.min, dram__bytes_read.sum. So unlike CUPTI, there
//       are no counters v. metrics. Only metrics can be read. On the Titan V development GPUs,
//       we found 1417 "counters" and 6064 metrics; an average of 4.28 metrics per counter. Note
//       that all counters have at least the four metrics of .min,.max,.avg,.sum.
//-------------------------------------------------------------------------------------------------
//*************************************************************************************************

//*************************************************************************************************
// Simple string hashing management functions.
//*************************************************************************************************

//-----------------------------------------------------------------------------
// stringHash: returns unsigned long value for hashed string.  See djb2, Dan
// Bernstein, http://www.cse.yorku.ca/~oz/hash.html Empirically a fast well
// distributed hash, not theoretically explained.  On a test system with 1857
// events, this gets about a 65% density in a 2000 element table; 35% of slots
// have dups; max dups was 4.
//-----------------------------------------------------------------------------
static unsigned int stringHash(char *str)
{
  unsigned long hash = 5381;                             // seed value.
  int c;
  while ((c = (*str++))) {                               // ends when c == 0.
     hash = ((hash << 5) + hash) + c;                    // hash * 33 + c.
  }

  return (hash % CUDA11_HASH_SIZE);                      // compute index and exit.
} // end function.


//-----------------------------------------------------------------------------
// addNameHash: Given a string, hash it, and add to hash table.
//-----------------------------------------------------------------------------
static unsigned int addNameHash(char *key, int idx) 
{
    // need a new item no matter what.
    cuda11_hash_entry_t* newItem = calloc(1, sizeof(cuda11_hash_entry_t));
    newItem->idx = idx;
    // compute slot.
    unsigned int slot = stringHash(key);
    // make next item of new entry previous head.
    newItem->next = cuda11_NameHashTable[slot];
    // replace head with new item (that chains to previous head).
    cuda11_NameHashTable[slot] = newItem;
    return(slot); 
} // end routine.


//-----------------------------------------------------------------------------
// freeEntireNameHash: Deletes all alloced data. note head is just a pointer
// to a hash entry; not an entry itself.
//-----------------------------------------------------------------------------
static void freeEntireNameHash(void) 
{
    int i;
    cuda11_hash_entry_t *newHead;
    for (i=0; i<CUDA11_HASH_SIZE; i++) {
        while (cuda11_NameHashTable[i] != NULL) {
            newHead = cuda11_NameHashTable[i]->next;    
            free(cuda11_NameHashTable[i]);
            cuda11_NameHashTable[i] = newHead;
        }
    }
} // end routine.


//-----------------------------------------------------------------------------
// findNameHash: Returns the idx into cuda11_AllEvents[] or -1 if not found.
//-----------------------------------------------------------------------------
static int findNameHash(char *key) 
{
    int idx;
    // compute hash slot it should be in.
    unsigned int slot = stringHash(key);

    cuda11_hash_entry_t* check = cuda11_NameHashTable[slot];    
    while (check != NULL) {
        idx = check->idx;
        if (strcmp(cuda11_AllEvents[idx]->papi_name, key) == 0) {
            // found it.
            return(idx);
        }
        check = check->next;
    }

    // Failed to find a match.
    return(-1);
} // end routine.


//-------------------------------------------------------------------------------------------------
// Adjust the size of cuda11_AllEvents[] if needed.
//-------------------------------------------------------------------------------------------------
void cuda11_makeRoomAllEvents(void) {
    int oldSize = cuda11_maxEvents;
    if (cuda11_numEvents < cuda11_maxEvents) return;    // cuda11_numEvents is okay.
    // We go big here; typical is 115,000 events on Titan V, may have multiple devices.
    const int MEMORY_PAD = 16384;
    cuda11_maxEvents += MEMORY_PAD;
    cuda11_AllEvents = (cuda11_eventData**) papi_realloc(cuda11_AllEvents, (cuda11_maxEvents*sizeof(cuda11_eventData*)));
    if (!cuda11_AllEvents) {
        fprintf(stderr, "%s:%s:%i Memory failure; failed to allocate %i entries for cuda11_AllEvents.\n",
                __FILE__, __func__, __LINE__, cuda11_maxEvents);
        exit(-1);
    }

    // Clear added memory.
    memset(&cuda11_AllEvents[oldSize], 0, MEMORY_PAD*sizeof(cuda11_eventData*));
    return;
}

// free elements of cuda11_eventData structure.
static void free_cuda11_eventData_contents(cuda11_eventData* myEvent) 
{
    int i;
    for (i=0; i<myEvent->numRawMetrics; i++) {
        // The name had to be copied by strdup.
        char *name = (char*) myEvent->rawMetricRequests[i].pMetricName;
        if (name != NULL) free(name);
    }

    if (myEvent->rawMetricRequests) free(myEvent->rawMetricRequests);
    if (myEvent->papi_name  ) free(myEvent->papi_name  );
    if (myEvent->nv_name    ) free(myEvent->nv_name    );
    if (myEvent->description) free(myEvent->description);
    if (myEvent->dimUnits   ) free(myEvent->dimUnits   );
} // end routine.

// Find or create a MetricsContext in the device list.
// returns NULL if the creation fails. 
static NVPW_CUDA_MetricsContext_Create_Params* cuda11_getMetricsContextPtr(int dev) 
{
    int i;
    cuda_device_desc_t *mydevice;
    cuda_context_t *gctxt = global_cuda_context;
    mydevice = &gctxt->deviceArray[dev];

    // If we have it, just return the pointer.
    if (mydevice->pMetricsContextCreateParams != NULL) {
        return(mydevice->pMetricsContextCreateParams);
    }

    // We don't have it. We must create it. this takes ~20ms.
    
    NVPW_CUDA_MetricsContext_Create_Params *pMCCP;
    pMCCP = calloc(1, sizeof(NVPW_CUDA_MetricsContext_Create_Params));
    if (pMCCP == NULL) {
        if (0) fprintf(stderr, "%s:%s:%i failed to allocate memory.\n", __FILE__, __func__, __LINE__);
        return(NULL);
    }

    pMCCP->structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE;
    pMCCP->pChipName = mydevice->cuda11_chipName;
    NVPW_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(pMCCP), // LEAK
        if (0) fprintf(stderr, "%s:%s:%i failed to create.\n", __FILE__, __func__, __LINE__);
        return(NULL));

    // We created successfully. populate.
    mydevice->pMetricsContextCreateParams = pMCCP;
    mydevice->ownsMetricsContext = 1;

    // Now populate all other same chip name devices with this one.
    for (i=0; i<gctxt->deviceCount; i++) {
        if (i == dev) continue;
        cuda_device_desc_t *adevice;
        adevice = &gctxt->deviceArray[i];
        if (strcmp(mydevice->cuda11_chipName, adevice->cuda11_chipName) == 0) {
            // found a sister with same name, populate with my context.
            adevice->pMetricsContextCreateParams = pMCCP;
            adevice->ownsMetricsContext = 0;
        }
    }

    return(pMCCP);
} // end routine

// Destroys all MetricsContexts in the device list.
static int cuda11_destroyMetricsContexts(void) 
{
    int i;
    cuda_device_desc_t *mydevice;
    cuda_context_t *gctxt = global_cuda_context;

    for (i=0; i<gctxt->deviceCount; i++) {
        mydevice = &gctxt->deviceArray[i];
        // If I don't own it, just zero it.
        if (mydevice->ownsMetricsContext == 0) {
            mydevice->pMetricsContextCreateParams = NULL;
            continue;
        }

        // otherwise, destroy it.
        NVPW_MetricsContext_Destroy_Params MetricsContextDestroyParams;
        memset(&MetricsContextDestroyParams, 0,  NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE);
        MetricsContextDestroyParams.structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE;
        MetricsContextDestroyParams.pMetricsContext = mydevice->pMetricsContextCreateParams->pMetricsContext;
        NVPW_CALL((*NVPW_MetricsContext_DestroyPtr)(&MetricsContextDestroyParams),
            if (0) fprintf(stderr, "%s:%s:%i failed to destroy MetricsContextCreateParams.\n", __FILE__, __func__, __LINE__);
            );
        free (mydevice->pMetricsContextCreateParams);
        mydevice->pMetricsContextCreateParams = NULL;
        mydevice->ownsMetricsContext = 0;
    }

    return(PAPI_OK);
} // END routine.    

//-------------------------------------------------------------------------------------------------
// This routine is not complete on its own, it is a continuation of _cuda_add_native_events() once
// we discover we are cupti 11 (or later).
// This is an internal routine, it is the caller's responsibility to ensure thread safety.
//-------------------------------------------------------------------------------------------------
static int _cuda11_init_profiler(void)
{
    // Call to init the profiler.
    CUptiResult cuptiRet;
    CUpti_Profiler_Initialize_Params profilerInitializeParams;
    memset(&profilerInitializeParams, 0,  CUpti_Profiler_Initialize_Params_STRUCT_SIZE);
    profilerInitializeParams.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    cuptiRet = (*cuptiProfilerInitializePtr)(&profilerInitializeParams); // Mem leak in library.

    if (cuptiRet != CUPTI_SUCCESS) {
        const char *errstr;
        (*cuptiGetResultStringPtr)(cuptiRet, &errstr);
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "cuptiProfilerInitialize failed; error '%s'.", errstr);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);   // Override given but not found.
    }

    NVPW_InitializeHost_Params initializeHostParams;
    memset(&initializeHostParams, 0,  NVPW_InitializeHost_Params_STRUCT_SIZE);
    initializeHostParams.structSize = NVPW_InitializeHost_Params_STRUCT_SIZE;
    NVPA_Status nvpaRet=(*NVPW_InitializeHostPtr)(&initializeHostParams);       // Mem leak in library.
    if (nvpaRet != NVPA_STATUS_SUCCESS) {
        int strErr=snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "NVPW_IntializeHost failed; error %d.", nvpaRet);
        _cuda_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        return(PAPI_ENOSUPP);   // Override given but not found.
    }

    return(PAPI_OK);
} // END _cuda11_init_profiler

// Deal with structure changes between runtime versions 10,11.
#define CUpti_Device_GetChipName_Params_STRUCT_SIZE10 16
#define CUpti_Device_GetChipName_Params_STRUCT_SIZE11 32

#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE10 56
#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE11 58

#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE10 24
#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE11 41

#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10 24
#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11 40

// Accumulate event data and return it. Returns NULL on calloc failure, else papi_error is set.
// All of this code is necessary to figure out the number of passes. We use this to extend the
// description of the metric; with passes, dimUnits, and Accumulation type (set on enumeration). 
//  
// a variation of the is code could be used by cuda11_update_control_state() to determinine the
// number of passes required for several metrics together.
// 
//
static int cuda11_getMetricDetails(cuda11_eventData* thisEventData, char *pChipName, 
    NVPW_CUDA_MetricsContext_Create_Params* pMetricsContextCreateParams) 
{
    size_t numNestingLevels;
    size_t numIsolatedPasses;
    size_t numPipelinedPasses;
    size_t numOfPasses=0;
    int    i, numDep;

    // Don't repeat this exercise.
    if (thisEventData->detailsDone == 1) return(PAPI_OK);
    // No matter how it turns out, don't do it again.
    thisEventData->detailsDone=1;
    //----------------SECTION----------------
    // build structure needed for call.
    NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams;
    nvpw_metricsConfigCreateParams.structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE;
    nvpw_metricsConfigCreateParams.pPriv = NULL;
    nvpw_metricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    nvpw_metricsConfigCreateParams.pChipName = pChipName;
    NVPW_CALL( (*NVPW_CUDA_RawMetricsConfig_CreatePtr)(&nvpw_metricsConfigCreateParams),
            return(PAPI_ENOSUPP) );

    //----------------SECTION----------------
    // build structure needed for call.
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams;
    memset(&beginPassGroupParams, 0,  NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE); 
    beginPassGroupParams.structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE; 
    beginPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    NVPW_CALL((*NVPW_RawMetricsConfig_BeginPassGroupPtr)(&beginPassGroupParams),
              return(PAPI_ENOSUPP));

    // Note: Here is where nvidia simpleQuery.cpp example code calls GetRawMetricRequests, we are inlining it.
    //----------------SECTION----------------
    // Need to build a metric properties; contains pDescription, pDimUnits, and **ppRawMetricDependencies.
    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams;
    memset(&getMetricPropertiesBeginParams, 0,  NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE); 
    getMetricPropertiesBeginParams.structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE;
    getMetricPropertiesBeginParams.pMetricsContext = pMetricsContextCreateParams->pMetricsContext;
    getMetricPropertiesBeginParams.pMetricName     = thisEventData->nv_name;
    
    NVPW_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams),
              return(PAPI_ENOSUPP));

    // Fill in what we learned with that call.
    thisEventData->description = strdup(getMetricPropertiesBeginParams.pDescription);
    thisEventData->dimUnits = strdup(getMetricPropertiesBeginParams.pDimUnits);
    thisEventData->gpuBurstRate =       getMetricPropertiesBeginParams.gpuBurstRate;
    thisEventData->gpuSustainedRate =   getMetricPropertiesBeginParams.gpuSustainedRate;

    //----------------SECTION----------------
    // count the dependencies, and build an array of NVPA_RawMetricRequest
    // entries with them.  We remember this array with the event, it is
    // necessary for both testing the compatibility of events in an eventset,
    // and necessary for querying values. Note that these may also be the
    // values we need to enforce accumulating events.
    // See nvidia example simpleQuery.cpp:112-131. 
    // Programmer Note: These dependency names are just 18 char hex strings. 
    // At this writing, metrics have from 1 to 45 dependencies:
    // event 'dram__bytes.avg' has 4 dependencies.
    // 0: '0x1b6d0ab8e9f0135d'.
    // 1: '0x667e0015f33a459f'.
    // 2: '0xee48e1b9f1ebf302'.
    // 3: '0xf53385f81b35356b'.

    numDep = 0;
    while (getMetricPropertiesBeginParams.ppRawMetricDependencies[numDep] != NULL) numDep++;
    if (numDep == 0) return(PAPI_ENOSUPP);

    // make space for all the raw metrics.
    NVPA_RawMetricRequest* rawMetricRequests = (NVPA_RawMetricRequest*) calloc(numDep, sizeof(NVPA_RawMetricRequest));

    if (rawMetricRequests == NULL) return(PAPI_ENOMEM);

    // For each dependency, build a rawMetricRequest table entry.
    for (i = 0; i<numDep; i++) {
        rawMetricRequests[i].pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[i]);
        if (rawMetricRequests[i].pMetricName == NULL) return(PAPI_ENOMEM);
        rawMetricRequests[i].isolated = 1;
        rawMetricRequests[i].keepInstances = 1;
        }

    // Remember it in the event data.
    thisEventData->numRawMetrics=numDep;
    thisEventData->rawMetricRequests = rawMetricRequests;

    // Now cleanup after GetMetricProperties.
    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams;
    getMetricPropertiesEndParams.structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    getMetricPropertiesEndParams.pPriv           = NULL;
    getMetricPropertiesEndParams.pMetricsContext = pMetricsContextCreateParams->pMetricsContext;
    NVPW_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), 
              return(PAPI_ENOSUPP));

    //----------------SECTION----------------
    // Collect info on the dependencies. See nvidia example simpleQuery.cpp:158.
    //
    NVPW_RawMetricsConfig_IsAddMetricsPossible_Params isAddMetricsPossibleParams;
    isAddMetricsPossibleParams.structSize = NVPW_RawMetricsConfig_IsAddMetricsPossible_Params_STRUCT_SIZE;
    isAddMetricsPossibleParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    isAddMetricsPossibleParams.pRawMetricRequests = &rawMetricRequests[0];
    isAddMetricsPossibleParams.numMetricRequests = numDep;
    NVPW_CALL((*NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr)(&isAddMetricsPossibleParams),
              return(PAPI_ENOSUPP));
    
    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams;
    addMetricsParams.structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
    addMetricsParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests = numDep;
    NVPW_CALL((*NVPW_RawMetricsConfig_AddMetricsPtr)(&addMetricsParams),
              return(PAPI_ENOSUPP));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams;
    endPassGroupParams.structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
    endPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    NVPW_CALL((*NVPW_RawMetricsConfig_EndPassGroupPtr)(&endPassGroupParams),
              return(PAPI_ENOSUPP));

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams;
    rawMetricsConfigGetNumPassesParams.structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE;
    rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    NVPW_CALL((*NVPW_RawMetricsConfig_GetNumPassesPtr)(&rawMetricsConfigGetNumPassesParams),
              return(PAPI_ENOSUPP));

    // No Nesting of ranges in case of CUPTI_AutoRange, in AutoRange the range
    // is already at finest granularity of every kernel Launch so
    // numNestingLevels = 1.
    // That said, in PAPI we use CUPTI_UserRange; but we still have no nesting
    // because we only allow one PAPI_start() for an EventSet. 

    numNestingLevels = 1;
    numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;

    //----------------SECTION----------------
    // Compute the number of passes.

    numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;
    thisEventData->passes = (int) numOfPasses;

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams;
    rawMetricsConfigDestroyParams.structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
    rawMetricsConfigDestroyParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
    NVPW_CALL((*NVPW_RawMetricsConfig_DestroyPtr)((NVPW_RawMetricsConfig_Destroy_Params*) &rawMetricsConfigDestroyParams),
              return(PAPI_ENOSUPP));

    //----------------SECTION----------------
    // Modify description to include type and number of passes.
    char added[PAPI_MAX_STR_LEN];
    char copyDesc[PAPI_HUGE_STR_LEN];
    switch (thisEventData->treatment) {
     
        case SpotValue: 
            snprintf(added, PAPI_MAX_STR_LEN, ". Units=%s Passes=%d Accum=Spot", thisEventData->dimUnits, thisEventData->passes);
            break;

        case RunningSum:
            snprintf(added, PAPI_MAX_STR_LEN, ". Units=%s Passes=%d Accum=Sum", thisEventData->dimUnits, thisEventData->passes);
            break;

        case RunningMin:
            snprintf(added, PAPI_MAX_STR_LEN, ". Units=%s Passes=%d Accum=Min", thisEventData->dimUnits, thisEventData->passes);
            break;

        case RunningMax:
            snprintf(added, PAPI_MAX_STR_LEN, ". Units=%s Passes=%d Accum=Max", thisEventData->dimUnits, thisEventData->passes);
            break;
    }

    int olen = strlen(thisEventData->description);
    int alen = strlen(added);

    if ((olen+alen) >= PAPI_HUGE_STR_LEN) {
        olen = PAPI_HUGE_STR_LEN - alen;
    }


    // Truncate original description if necessary to make room for unuts, passes, accum.
    thisEventData->description[olen] = 0;
    // Create augmented description.
    snprintf(copyDesc, PAPI_HUGE_STR_LEN, "%s%s", thisEventData->description, added);
    // discard original description.
    free(thisEventData->description);
    // record augmented description. 
    thisEventData->description = strdup(copyDesc); 
    
    return(PAPI_OK);
} // end cuda11_getMetricDetails.


//-------------------------------------------------------------------------------------------------
// This routine is not complete on its own, it is a continuation of _cuda_add_native_events() once
// we discover we are cupti 11 (or later).
// This is an internal routine, it is the caller's responsibility to ensure thread safety.
//-------------------------------------------------------------------------------------------------
static int _cuda11_add_native_events(cuda_context_t * gctxt)
{
    (void) gctxt;
    int i, ret;
    int userDevice, deviceNum;
    cuda_device_desc_t *mydevice;
    CUresult cuErr; (void) cuErr;
    CUcontext userCtx;

    // Get deviceNum.
    CUDA_CALL((*cudaGetDevicePtr) (&userDevice), return (PAPI_EMISC));
    CU_CALL((*cuCtxGetCurrentPtr) (&userCtx),    return (PAPI_EMISC));

    // I saw this in other sample code; it may not be relevant. It worries
    // whether runtime version differs from the Header versions. To me that
    // is a compile issue, not something to correct at runtime. -Tony C.

    if (cuda_runtime_version < 11000) {
        if (0) fprintf(stderr, "%s:%s:%i Setting for cuda_runtime_version=%d (SIZE10)\n", __FILE__, __func__, __LINE__, cuda_runtime_version);
        GetChipName_Params_STRUCT_SIZE=CUpti_Device_GetChipName_Params_STRUCT_SIZE10;
        Profiler_SetConfig_Params_STRUCT_SIZE=CUpti_Profiler_SetConfig_Params_STRUCT_SIZE10;
        Profiler_EndPass_Params_STRUCT_SIZE=CUpti_Profiler_EndPass_Params_STRUCT_SIZE10;
        Profiler_FlushCounterData_Params_STRUCT_SIZE=CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10;
    } else {
        if (0) fprintf(stderr, "%s:%s:%i Setting for cuda_runtime_version=%d (SIZE11)\n", __FILE__, __func__, __LINE__, cuda_runtime_version);
        GetChipName_Params_STRUCT_SIZE=CUpti_Device_GetChipName_Params_STRUCT_SIZE11;
        Profiler_SetConfig_Params_STRUCT_SIZE=CUpti_Profiler_SetConfig_Params_STRUCT_SIZE11;
        Profiler_EndPass_Params_STRUCT_SIZE=CUpti_Profiler_EndPass_Params_STRUCT_SIZE11;
        Profiler_FlushCounterData_Params_STRUCT_SIZE=CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11;
    }

    // If any of these do not match the actual sizes, we have a mismatch between compile headers and actual library.
    if (CUpti_Device_GetChipName_Params_STRUCT_SIZE != GetChipName_Params_STRUCT_SIZE ||
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE != Profiler_SetConfig_Params_STRUCT_SIZE ||
        CUpti_Profiler_EndPass_Params_STRUCT_SIZE != Profiler_EndPass_Params_STRUCT_SIZE ||
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE != Profiler_FlushCounterData_Params_STRUCT_SIZE) {

        strncpy(_cuda_vector.cmp_info.disabled_reason, "Profiler structures do not match Compiled Version. Possibly wrong libraries found.", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }

    // Comparison for debugging if you want it.
    if (0) {
        fprintf(stderr, "%s:%s:%i Actual vs Set sizes:\n", __FILE__, __func__, __LINE__);
        fprintf(stderr, "%s:%s:%i GetChipName : %zd, %d\n", __FILE__, __func__, __LINE__, CUpti_Device_GetChipName_Params_STRUCT_SIZE, GetChipName_Params_STRUCT_SIZE); 
        fprintf(stderr, "%s:%s:%i SetConfig   : %zd, %d\n", __FILE__, __func__, __LINE__, CUpti_Profiler_SetConfig_Params_STRUCT_SIZE, Profiler_SetConfig_Params_STRUCT_SIZE);
        fprintf(stderr, "%s:%s:%i EndPass     : %zd, %d\n", __FILE__, __func__, __LINE__, CUpti_Profiler_EndPass_Params_STRUCT_SIZE, Profiler_EndPass_Params_STRUCT_SIZE);
        fprintf(stderr, "%s:%s:%i FlushCounter: %zd, %d\n", __FILE__, __func__, __LINE__, CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE, Profiler_FlushCounterData_Params_STRUCT_SIZE);
    }
  
    // We have to initialize the profiler to get the chip names,
    // And we need the chipnames to read the metrics we have.
    // This performs cuptiProfilerInitialize(), and NVPW_InitializeHost().

    ret = _cuda11_init_profiler();
    if (ret != PAPI_OK) return(ret);

    // Ensure the hashtable (all pointers) is cleared to empty state.
    memset(&cuda11_NameHashTable[0], 0, CUDA11_HASH_SIZE*sizeof(cuda11_hash_entry_t*));

    /* Create, and zero, initial allocation for CUDA11 Events.
     * Total size [bytes] for cuda11_AllEvents =
     *     NUM_EVENTS * 8 (sizeof(cuda11_eventData*)) * 96 (sizeof(cuda11_eventData))
     * `papi_component_avail` reported number of events by GPU:
     *     - V100-SXM2-32GB:    137,994
     *     - V100-PCIE-32GB:    141,834
     *     - A100-PCIE-40GB:    158,496
     *     - A100-SXM4-40/80GB: 263,040
     * Based on above, initial NUM_EVENTS of 160,000 should be reasonable.
     * Note: Allocation can still be expanded by cuda11_makeRoomAllEvents().
     */
    const int INIT_NUM_EVENTS = 160000;
    cuda11_maxEvents = gctxt->deviceCount * INIT_NUM_EVENTS;
    cuda11_AllEvents = (cuda11_eventData**) calloc(cuda11_maxEvents, sizeof(cuda11_eventData*));
    if (cuda11_AllEvents == NULL) {
        return PAPI_ENOMEM;
    }

    int *firstLast = calloc(2*gctxt->deviceCount, sizeof(int)); // space for first/last indices.

    for (deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        int strErr;
        mydevice = &gctxt->deviceArray[deviceNum];
        // CUpti_Device_GetChipName_Params relies on cupti_target.h, not in legacy cuda distributions.
        CUpti_Device_GetChipName_Params ChipNameParams;
        memset(&ChipNameParams, 0,  GetChipName_Params_STRUCT_SIZE);
        ChipNameParams.structSize = GetChipName_Params_STRUCT_SIZE;
        ChipNameParams.pPriv = NULL;
        ChipNameParams.deviceIndex=deviceNum;

        CUPTI_CALL((*cuptiDeviceGetChipNamePtr)(&ChipNameParams),);
        strErr = snprintf(mydevice->cuda11_chipName, PAPI_MIN_STR_LEN, ChipNameParams.pChipName);
        if (strErr > PAPI_MIN_STR_LEN) HANDLE_STRING_ERROR;
        mydevice->cuda11_chipName[PAPI_MIN_STR_LEN-1]=0;

        // Init session context; it will be updated in cuda11_start() if needed.
        mydevice->sessionCtx = NULL;

        // Get or create the Metrics Context. 
        NVPW_CUDA_MetricsContext_Create_Params *pMCCP = cuda11_getMetricsContextPtr(deviceNum);
        if (pMCCP == NULL) {
            return(PAPI_EMISC);
        }

        firstLast[0+(deviceNum<<1)] = cuda11_numEvents;
        firstLast[1+(deviceNum<<1)] = cuda11_numEvents;

        // figure out if this device has the same names as a previous device.
        for (i=0; i<deviceNum; i++) {
            // break if we find a match.
            if (strcmp(mydevice->cuda11_chipName, gctxt->deviceArray[i].cuda11_chipName) == 0) break;
        }

        // If there is a previous device with my chipName, I can use
        // the names it found. Note if deviceNum==0, i==0, Not less.
        // In testing; COPY took 35 ms (best time) for 114K events; and
        // using the NVPW routines takes 99 ms (best time). 
        if (i<deviceNum) {
            int idx;
            for (idx = firstLast[0+(i<<1)]; idx <= firstLast[1+(i<<1)]; idx++) {
                cuda11_makeRoomAllEvents();
                cuda11_eventData* prevEventData = cuda11_AllEvents[idx];
                cuda11_eventData* thisEventData = (cuda11_eventData*) calloc(1, sizeof(cuda11_eventData));
                if (thisEventData == NULL) return(PAPI_ENOMEM);

                thisEventData->nv_name = strdup(prevEventData->nv_name); // allocate and copy.  
                thisEventData->treatment = prevEventData->treatment;
                char PAPI_name[PAPI_MAX_STR_LEN];
                snprintf(PAPI_name, PAPI_MAX_STR_LEN,  "%s:device=%d", thisEventData->nv_name, deviceNum);
                thisEventData->papi_name = strdup(PAPI_name);
                thisEventData->deviceNum = deviceNum;
                cuda11_AllEvents[cuda11_numEvents]=thisEventData;
                addNameHash(thisEventData->papi_name, cuda11_numEvents); 
                firstLast[1+(deviceNum<<1)] = cuda11_numEvents;
                cuda11_numEvents++; 
            }
        } else {
            // Collect the names for this device.
            //----------------SECTION----------------
            // Collect Counter Names.
            // Actually, we don't need to collect counter names! counters cannot be
            // read directly, you MUST read a metric that consists of the counter
            // name + .sum, .avg, .min, .max, or for sum additional suffixes.
            // perfworks ONLY deals with metrics. You can collect these here, but
            // then you must decorate the name with these suffixes.  See
            // https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_metrics_api

            // This is the code to get counters, which we don't need.
    //      NVPW_MetricsContext_GetCounterNames_Begin_Params getCounterNames;
    //      getCounterNames.structSize = NVPW_MetricsContext_GetCounterNames_Begin_Params_STRUCT_SIZE;
    //      getCounterNames.pPriv = NULL;
    //      getCounterNames.pMetricsContext = pMCCP->pMetricsContext;
    //      NVPW_CALL((*NVPW_MetricsContext_GetCounterNames_BeginPtr)(&getCounterNames),
    //          return(PAPI_ENOSUPP));
    //      
    //      fprintf(stderr, "%s:%i Counters Found: %zu.\n", __func__, __LINE__, getCounterNames.numCounters);
    //      for (i=0; i< (int) getCounterNames.numCounters; i++) {
    //          fprintf(stderr, "%s:%i Counter name='%s'\n", __func__, __LINE__, getCounterNames.ppCounterNames[i]);
    //      }
    //      NVPW_MetricsContext_GetCounterNames_End_Params endCounterNames;
    //      endCounterNames.structSize = NVPW_MetricsContext_GetCounterNames_End_Params_STRUCT_SIZE;
    //      endCounterNames.pPriv = NULL;
    //      endCounterNames.pMetricsContext = MetricsContextCreateParams.pMetricsContext;
    //      NVPW_CALL((*NVPW_MetricsContext_GetCounterNames_EndPtr)(&endCounterNames),
    //          return(PAPI_ENOSUPP));
                  
            //----------------SECTION----------------
            // Collect Metric Names, and get Metrics Data.
            NVPW_MetricsContext_GetMetricNames_Begin_Params GetMetricNameBeginParams;
            memset(&GetMetricNameBeginParams, 0,  NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE);
            GetMetricNameBeginParams.structSize = NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE;
            GetMetricNameBeginParams.pMetricsContext = pMCCP->pMetricsContext;

            // in : if true, SKIPS enumerating \<metric\>.peak_{burst, sustained}
            //      Adds 45 seconds and 57,120 events per device. (TITAN V)
            GetMetricNameBeginParams.hidePeakSubMetrics = 0;
            // in : if true, SKIPS enumerating \<metric\>.per_{active,elapsed,region,frame}_cycle
            //      Adds 24 seconds and 20,990 events per device. (TITAN V)
            GetMetricNameBeginParams.hidePerCycleSubMetrics = 0;
            // in : if true, SKIPS enumerating \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
            //      Adds 50 seconds and 28,796 events per device. (TITAN V)
            GetMetricNameBeginParams.hidePctOfPeakSubMetrics = 0;
            // in : if true, SKIPS enumerating \<unit\>__throughput.pct_of_peak_sustained_elapsed even if hidePctOfPeakSubMetrics is true
            //      Adds 0 seconds and 24 events per device. (TITAN V)
            GetMetricNameBeginParams.hidePctOfPeakSubMetricsOnThroughputs=0;

            // This call alone takes about 54ms to complete.
            NVPW_CALL((*NVPW_MetricsContext_GetMetricNames_BeginPtr)(&GetMetricNameBeginParams),
                return(PAPI_EMISC));
            
            //----------------SECTION----------------
            //  We have the names of metrics, get details on them.
            for (i = 0; i < (int) GetMetricNameBeginParams.numMetrics; i++) {
                cuda11_eventData* thisEventData = (cuda11_eventData*) calloc(1, sizeof(cuda11_eventData));
                if (thisEventData == NULL) return(PAPI_ENOMEM);

                thisEventData->nv_name = strdup(GetMetricNameBeginParams.ppMetricNames[i]); // allocate and copy.  

                // We have the name; enough to specify the treatment.
                if      (strstr(thisEventData->nv_name, ".sum") != NULL) thisEventData->treatment = RunningSum;
                else if (strstr(thisEventData->nv_name, ".min") != NULL) thisEventData->treatment = RunningMin;
                else if (strstr(thisEventData->nv_name, ".max") != NULL) thisEventData->treatment = RunningMax;
                else thisEventData->treatment = SpotValue;

                char PAPI_name[PAPI_MAX_STR_LEN];
                snprintf(PAPI_name, PAPI_MAX_STR_LEN,  "%s:device=%d", thisEventData->nv_name, deviceNum);
                thisEventData->papi_name = strdup(PAPI_name);

                // Here is the place to qualify events to be included,
                // but all we've got at this point is the name and 
                // derived treatment. Note 'if (1)' is always true.
                if (1) {        // If it qualifies, add to list.
                    cuda11_makeRoomAllEvents();
                    thisEventData->deviceNum = deviceNum;
                    cuda11_AllEvents[cuda11_numEvents]=thisEventData;
                    addNameHash(thisEventData->papi_name, cuda11_numEvents); 
                    firstLast[1+(deviceNum<<1)] = cuda11_numEvents;
                    cuda11_numEvents++; 
                } else {        // If it failed to qualify, discard it.
                    free_cuda11_eventData_contents(thisEventData);
                    free(thisEventData);
                    thisEventData = NULL;
                } // end if we need to discard event data.
            } // end metrics loop.

            // Finish up this GetMetricNames.
            NVPW_MetricsContext_GetMetricNames_End_Params GetMetricNameEndParams;
            memset(&GetMetricNameEndParams, 0,  NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE);
            GetMetricNameEndParams.structSize = NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE;
            GetMetricNameEndParams.pMetricsContext = pMCCP->pMetricsContext;
            NVPW_CALL((*NVPW_MetricsContext_GetMetricNames_EndPtr)(&GetMetricNameEndParams),);
        }
    } // end for each device. 

    // Performance report for hash table efficiency.
    if (0) {
        int inUse=0, avgChain=0, maxChain=0;
        int i, j;
        for (i=0; i<CUDA11_HASH_SIZE; i++) {
            if (cuda11_NameHashTable[i] == NULL) continue;
            inUse++;
            j=1;
            cuda11_hash_entry_t *item = cuda11_NameHashTable[i];
            while (item->next != NULL) {j++; item=item->next;}
            avgChain += j;
            if (j > maxChain) maxChain = j;
        }

        fprintf(stderr, "%s:%s:%i, Hash Stats: inUse=%.3f%%, avgChain=%.2f, maxChain=%d.\n",
            __FILE__, __func__, __LINE__, ((100.*inUse)/(CUDA11_HASH_SIZE+0.)),
            (avgChain+0.)/(inUse+0.), maxChain);
    }

    free(firstLast);

    return(PAPI_OK);    
} // end _cuda11_add_native_events.


//-------------------------------------------------------------------------------------------------
// We don't do anything for a new thread initialization.
//-------------------------------------------------------------------------------------------------
static int _cuda11_init_thread(hwd_context_t * ctx)
{
    (void) ctx;
    SUBDBG("Entering\n");
    // needs work.
    return PAPI_OK;
} // end _cuda11_init_thread


//-------------------------------------------------------------------------------------------------
// Setup a counter control state.
// In general a control state holds the hardware info for an EventSet.
//-------------------------------------------------------------------------------------------------
static int _cuda11_init_control_state(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    // If no events were found during the initial component initialization, return error.
    if(global_cuda_context->availEventSize <= 0) {
        strncpy(_cuda_vector.cmp_info.disabled_reason, "ERROR CUDA: No events exist", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }

    // If it does not exist, create the global structure to hold CUDA contexts and active events.
    _papi_hwi_lock( COMPONENT_LOCK );
    if(!global_cuda_control) {
        global_cuda_control = (cuda_control_t *) papi_calloc(1, sizeof(cuda_control_t));
        global_cuda_control->countOfActiveCUContexts = 0;
        global_cuda_control->activeEventCount = 0;
    }

    _papi_hwi_unlock( COMPONENT_LOCK );
    return PAPI_OK;
} // end _cuda11_init_control_state

//-----------------------------------------------------------------------------
// userCtx is the active context. We can only push a different context.
static int _cuda11_build_profiling_structures(CUcontext userCtx) 
{
    cuda_context_t *gctxt = global_cuda_context;    // We don't use the passed-in parameter, we use a global.
    CUcontext popCtx;
    int dev;

    for (dev=0; dev < gctxt->deviceCount; dev++) {
        int ctxPushed=0;
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        // skip devices with no events to get, or no context.
        if (mydevice->cuda11_RMR_count == 0 || 
            mydevice->sessionCtx == NULL) continue;

        if (mydevice->sessionCtx != userCtx) {
            ctxPushed = 1;
            CU_CALL((*cuCtxPushCurrentPtr) (mydevice->sessionCtx),
                // On error,  
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }

        // Create the configImage. 
        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams;
        nvpw_metricsConfigCreateParams.structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE;
        nvpw_metricsConfigCreateParams.pPriv = NULL;
        nvpw_metricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        nvpw_metricsConfigCreateParams.pChipName = mydevice->cuda11_chipName;
        NVPW_CALL( (*NVPW_CUDA_RawMetricsConfig_CreatePtr)(&nvpw_metricsConfigCreateParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Note: The sample code sometimes creates params and then immediately
        // destroys them, but uses param structure elements later. But if I
        // destroy pRawMetricsConfig here, I get an error when I try to call
        // BeginPassGroup. I'm following standard coding practice and calling
        // the destroy function when I'm done using the structure.

        // creating params for beginPassGroup...
        NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams;
        memset(&beginPassGroupParams, 0,  NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE);  
        beginPassGroupParams.structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
        beginPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        
        // Actually calling BeginPassGroup.
        NVPW_CALL((*NVPW_RawMetricsConfig_BeginPassGroupPtr) 
            (&beginPassGroupParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Creating params for addMetrics.
        NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams;
        memset(&addMetricsParams, 0,  NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE);
        addMetricsParams.structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
        addMetricsParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;

        // passing in the rawMetricsRequests array.
        addMetricsParams.pRawMetricRequests = mydevice->cuda11_RMR;

        // and number of entries.
        addMetricsParams.numMetricRequests = mydevice->cuda11_RMR_count;

        // Executing the AddMetrics.
        NVPW_CALL( (*NVPW_RawMetricsConfig_AddMetricsPtr)
            (&addMetricsParams),
            // On error, 
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Creating params for EndPassGroup.
        NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams;
        memset(&endPassGroupParams, 0,  NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE);
        endPassGroupParams.structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;

        // passing pRawMetricsConfig also used above.
        endPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;

        // Actually Call EndPassGroup.
        NVPW_CALL( (*NVPW_RawMetricsConfig_EndPassGroupPtr)
            (&endPassGroupParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));
        
        // Build structure to call generateConfigImage.
        NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams;
        memset(&generateConfigImageParams, 0,  NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE);
        generateConfigImageParams.structSize = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE;
        generateConfigImageParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;

        // Actually call GenerateConfigImage.
        NVPW_CALL( (*NVPW_RawMetricsConfig_GenerateConfigImagePtr)
            (&generateConfigImageParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Image is built, but now we must get a copy. Build structure to getConfigImage.
        // Before we can get it, we need to know its size.
        NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams;
        memset(&getConfigImageParams, 0,  NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE);
        getConfigImageParams.structSize = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
        getConfigImageParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        getConfigImageParams.bytesAllocated = 0;
        getConfigImageParams.pBuffer = NULL;
        // Notice pBuffer=NULL and bytesAllocated=0. This is a sizing call.
        // bytes needed is reported in .bytesCopied.
        NVPW_CALL( (*NVPW_RawMetricsConfig_GetConfigImagePtr)
            (&getConfigImageParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // allocate memory for a vector of bytes.
        mydevice->cuda11_ConfigImage = calloc(getConfigImageParams.bytesCopied, sizeof(uint8_t));
        if (mydevice->cuda11_ConfigImage == NULL) {
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_ENOMEM);
        }
        mydevice->cuda11_ConfigImageSize = getConfigImageParams.bytesCopied;

        // sets size and pointer based on allocation.
        getConfigImageParams.bytesAllocated = getConfigImageParams.bytesCopied;
        getConfigImageParams.pBuffer = mydevice->cuda11_ConfigImage;

        // same call, get the actual image now. 
        NVPW_CALL( (*NVPW_RawMetricsConfig_GetConfigImagePtr)
            (&getConfigImageParams),
            // On error, 
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // We are done with pRawMetricsConfig, we destroy it here.
        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams; 
        memset(&rawMetricsConfigDestroyParams, 0,  NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE);
        rawMetricsConfigDestroyParams.structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
        rawMetricsConfigDestroyParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        NVPW_CALL((*NVPW_RawMetricsConfig_DestroyPtr) 
            ((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Get the CounterDataPrefixImage. 

        // Build structure to call CounterDataBuilder.
        // (return is counterDataBuilderCreateParams.pCounterDataBuilder)
        NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams;
        memset(&counterDataBuilderCreateParams, 0,  NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE);
        counterDataBuilderCreateParams.structSize = NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE;
        counterDataBuilderCreateParams.pChipName = mydevice->cuda11_chipName;

        // CounterDataBuilder_Create.
        NVPW_CALL( (*NVPW_CounterDataBuilder_CreatePtr)
            (&counterDataBuilderCreateParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Build the structure to call AddMetrics.
        NVPW_CounterDataBuilder_AddMetrics_Params CD_addMetricsParams;
        memset(&CD_addMetricsParams, 0, NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE);
        CD_addMetricsParams.structSize = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE;
        CD_addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
        CD_addMetricsParams.pRawMetricRequests = mydevice->cuda11_RMR;
        CD_addMetricsParams.numMetricRequests = mydevice->cuda11_RMR_count;
        // Call AddMetrics.
        NVPW_CALL((*NVPW_CounterDataBuilder_AddMetricsPtr) (&CD_addMetricsParams),
            // On error, 
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Build structure to call GetCounterDataPrefix.
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams;
        memset(&getCounterDataPrefixParams, 0,  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE);
        getCounterDataPrefixParams.structSize = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
        getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
        getCounterDataPrefixParams.bytesAllocated = 0;
        getCounterDataPrefixParams.pBuffer = NULL;

        // Just getting the size of the CounterDataPrefix here.
        NVPW_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)
            (&getCounterDataPrefixParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Allocate data.
        mydevice->cuda11_CounterDataPrefixImage = calloc(getCounterDataPrefixParams.bytesCopied, sizeof(uint8_t));
        if (mydevice->cuda11_CounterDataPrefixImage == NULL) return(PAPI_ENOMEM);
        mydevice->cuda11_CounterDataPrefixImageSize = getCounterDataPrefixParams.bytesCopied;

        getCounterDataPrefixParams.bytesAllocated = getCounterDataPrefixParams.bytesCopied;
        getCounterDataPrefixParams.pBuffer = mydevice->cuda11_CounterDataPrefixImage;

        // Same call, get actual image this time.
        NVPW_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)
            (&getCounterDataPrefixParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // We are done with the counterDataBuilder, destroy the params.
        NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams;
        memset(&counterDataBuilderDestroyParams, 0,  NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE);
        counterDataBuilderDestroyParams.structSize = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE;
        counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
        NVPW_CALL((*NVPW_CounterDataBuilder_DestroyPtr)
            ((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Create the CounterDataImage.         
        // See  $PAPI_CUDA_ROOT/extras/CUPTI/samples/userrange_profiling/simplecuda.cu
        // routine CreateCounterDataImage.

        // Create an options structure.
        memset(&mydevice->cuda11_CounterDataOptions, 0,  sizeof(CUpti_Profiler_CounterDataImageOptions));
        mydevice->cuda11_CounterDataOptions.structSize = sizeof(CUpti_Profiler_CounterDataImageOptions);
        mydevice->cuda11_CounterDataOptions.pCounterDataPrefix = mydevice->cuda11_CounterDataPrefixImage;
        mydevice->cuda11_CounterDataOptions.counterDataPrefixSize = mydevice->cuda11_CounterDataPrefixImageSize;
        mydevice->cuda11_CounterDataOptions.maxNumRanges = 1;
        mydevice->cuda11_CounterDataOptions.maxNumRangeTreeNodes = 1;
        mydevice->cuda11_CounterDataOptions.maxRangeNameLength = 64;

        // Use that to fill in a Calculate Size parameters.
        CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams;
        memset(&calculateSizeParams, 0,  CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE);
        calculateSizeParams.structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE;
        calculateSizeParams.pOptions = &mydevice->cuda11_CounterDataOptions;
        calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

        // Actually calculate the Size.
        CUPTI_CALL((*cuptiProfilerCounterDataImageCalculateSizePtr) (&calculateSizeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Create params for initialization.
        memset(&mydevice->cuda11_CounterDataInitParms, 0, CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE);
        mydevice->cuda11_CounterDataInitParms.structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
        mydevice->cuda11_CounterDataInitParms.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
        mydevice->cuda11_CounterDataInitParms.pOptions = &mydevice->cuda11_CounterDataOptions;
        mydevice->cuda11_CounterDataInitParms.counterDataImageSize = calculateSizeParams.counterDataImageSize;
        
        // Allocate space for the image.
        mydevice->cuda11_CounterDataImage = calloc(calculateSizeParams.counterDataImageSize, sizeof(uint8_t));
        if (mydevice->cuda11_CounterDataImage == NULL) {
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_ENOMEM);
        }

        mydevice->cuda11_CounterDataImageSize = calculateSizeParams.counterDataImageSize;
 
        mydevice->cuda11_CounterDataInitParms.pCounterDataImage = mydevice->cuda11_CounterDataImage;

        CUPTI_CALL((*cuptiProfilerCounterDataImageInitializePtr) (&mydevice->cuda11_CounterDataInitParms),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Params for calculating the size of the scratch buffer.
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams;
        memset(&scratchBufferSizeParams, 0,  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE);
        scratchBufferSizeParams.structSize = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE;
        scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
        scratchBufferSizeParams.pCounterDataImage = mydevice->cuda11_CounterDataInitParms.pCounterDataImage;

        // Calculate the size of the scratch buffer.
        CUPTI_CALL((*cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr) (&scratchBufferSizeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));
    
        // Allocate memory for it.
        mydevice->cuda11_CounterDataScratchBuffer = calloc(scratchBufferSizeParams.counterDataScratchBufferSize, sizeof(uint8_t));
        // Remember the size. 
        mydevice->cuda11_CounterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;

        // Params to initialize the scratch buffer.
        memset(&mydevice->cuda11_CounterScratchInitParms, 0,  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE);
        mydevice->cuda11_CounterScratchInitParms.structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE;
        mydevice->cuda11_CounterScratchInitParms.counterDataImageSize = mydevice->cuda11_CounterDataImageSize;
        mydevice->cuda11_CounterScratchInitParms.pCounterDataImage = mydevice->cuda11_CounterDataImage;
        mydevice->cuda11_CounterScratchInitParms.counterDataScratchBufferSize = mydevice->cuda11_CounterDataScratchBufferSize;
        mydevice->cuda11_CounterScratchInitParms.pCounterDataScratchBuffer = mydevice->cuda11_CounterDataScratchBuffer;

        CUPTI_CALL((*cuptiProfilerCounterDataImageInitializeScratchBufferPtr) (&mydevice->cuda11_CounterScratchInitParms),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // Restore previous context.
        if (ctxPushed) {
            ctxPushed = 0;
            CU_CALL((*cuCtxPopCurrentPtr) (&popCtx),
                // On error,
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }
    } // end for each device.

    return(PAPI_OK);
} // end _cuda11_build_profiling_structures


//-------------------------------------------------------------------------------------------------
// Triggered by eventset operations like add or remove.  For CUDA, needs to be
// called multiple times from each separate CUDA context with the events to be
// measured from that context.  For each context, create eventgroups for the
// events.
// Note: NativeInfo_t is defined in papi_internal.h.
//-------------------------------------------------------------------------------------------------
static int _cuda11_update_control_state(hwd_control_state_t * ctrl,
    NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    (void) ctrl;
    (void) nativeInfo;
    (void) nativeCount;
    (void) ctx;
    cuda_control_t *gctrl = global_cuda_control;    // We don't use the passed-in parameter, we use a global.
    cuda_context_t *gctxt = global_cuda_context;    // We don't use the passed-in parameter, we use a global.
    (void) gctrl;
    int dev, ii, userDevice;
    CUcontext userCtx;

    /* Return if no events */
    if(nativeCount == 0)
        return (PAPI_OK);

    // Get deviceNum.
    CUDA_CALL((*cudaGetDevicePtr) (&userDevice), return (PAPI_EMISC));
    SUBDBG("userDevice %d \n", userDevice);

    // cudaFree(NULL) does nothing real, but initializes a new cuda context
    // if one does not exist. This prevents cuCtxGetCurrent() from failing.
    // If it returns an error, we ignore it.
    CUDA_CALL((*cudaFreePtr) (NULL), );
    CU_CALL((*cuCtxGetCurrentPtr) (&userCtx),    return (PAPI_EMISC));
    SUBDBG("userDevice %d sessionCtx %p \n", userDevice, userCtx);

    _papi_hwi_lock( COMPONENT_LOCK );

    // This is the protocol devised by the PAPI team in 2015; see:
    // PAPI/src/components/cuda/tests/simpleMultiGPU.c.  If events from
    // multiple devices are to be in the event set, the user must create a
    // context for each device and make it active for PAPI_add. The context
    // must be the one used to launch the kernel on that device. We remember
    // the userCtx for current userDevice here; and use it during PAPI_start.

    if (userDevice >=0 && userDevice < gctxt->deviceCount) {
        gctxt->deviceArray[userDevice].sessionCtx = userCtx;
        if (0) fprintf(stderr, "%s:%s:%i userDevice=%d, setting sessionCtx=%p.\n", __FILE__, __func__, __LINE__, userDevice, userCtx);
    }
 
    // We need the rawMetricRequests, which we collected during event enumeration.
    // We also need the nvidia names of the metrics: cuda11_AllEvents[]->nv_name.
    // We need to assemble by device, and one array for all events on that device.
   
    int *deviceMetricCount =    calloc(gctxt->deviceCount, sizeof(int));
    int *deviceRawMetricCount = calloc(gctxt->deviceCount, sizeof(int));

    // Note, in cuda11, we do not use gctxt->availEventIsBeingMeasuredInEventset[idx].
    // Always reset to zero. We rebuild the eventset from scratch on every call.
    gctrl->activeEventCount=0;

    // First up: ensure every event is populated.
    cuda_device_desc_t *mydevice;
    for (ii = 0; ii < nativeCount; ii++) {
        int idx = nativeInfo[ii].ni_event;
        // skip if already initialized.
        if (cuda11_AllEvents[idx]->detailsDone == 1) continue;
        dev = cuda11_AllEvents[idx]->deviceNum;
        // get or create the appropriate Metrics Context.
        NVPW_CUDA_MetricsContext_Create_Params *pMCCP = cuda11_getMetricsContextPtr(dev);
        mydevice = &gctxt->deviceArray[dev];
        int err  = cuda11_getMetricDetails(cuda11_AllEvents[idx], mydevice->cuda11_chipName, pMCCP);
        if (err != PAPI_OK) {
            if (0) fprintf(stderr, "%s:%s:%i cuda11_getMetricDetails() failed, index=%d err=%d '%s'.\n",
                     __FILE__, __func__, __LINE__, idx, err, PAPI_strerror(err));
            return(err);
        }
    }

    for (ii = 0; ii < nativeCount; ii++) {                                  // For each event provided by caller,
        nativeInfo[ii].ni_position = gctrl->activeEventCount++; 
        int idx = nativeInfo[ii].ni_event;                                  // Get the index of the event (in the global context).
        gctrl->activeEventIndex[ii] = idx;                                  // Remember global index for this value.
        // Here we init the values for this event.
        switch (cuda11_AllEvents[idx]->treatment) {
            case SpotValue: cuda11_AllEvents[idx]->cumulativeValue = 0; break;
            case RunningSum: cuda11_AllEvents[idx]->cumulativeValue = 0; break;
            case RunningMin: cuda11_AllEvents[idx]->cumulativeValue = DBL_MAX; break;
            case RunningMax: cuda11_AllEvents[idx]->cumulativeValue = -DBL_MAX; break;
        }

        // Get the device we need to count it towards.                
        int eventDeviceNum = cuda11_AllEvents[idx]->deviceNum;              // Device number for this event.
        deviceRawMetricCount[eventDeviceNum] += cuda11_AllEvents[idx]->numRawMetrics;   // Add to raw metrics for device.
        deviceMetricCount[eventDeviceNum]++;                                // Add to metrics names for device.
    }        

    // Now same loop, but for each device, collect metrics names for that
    // device into a separate collection.

    for (dev=0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];

        // Free all the allocations the profiler needs, to rebuild them.
        if (mydevice->cuda11_ConfigImage) free(mydevice->cuda11_ConfigImage);
        mydevice->cuda11_ConfigImage = NULL;
        mydevice->cuda11_ConfigImageSize = 0;

        if (mydevice->cuda11_CounterDataPrefixImage) free(mydevice->cuda11_CounterDataPrefixImage); 
        mydevice->cuda11_CounterDataPrefixImage = NULL; 
        mydevice->cuda11_CounterDataPrefixImageSize = 0; 

        if (mydevice->cuda11_CounterDataImage) free(mydevice->cuda11_CounterDataImage); 
        mydevice->cuda11_CounterDataImage = NULL; 
        mydevice->cuda11_CounterDataImageSize = 0; 

        if (mydevice->cuda11_CounterDataScratchBuffer) free(mydevice->cuda11_CounterDataScratchBuffer); 
        mydevice->cuda11_CounterDataScratchBuffer = NULL; 
        mydevice->cuda11_CounterDataScratchBufferSize = 0; 

        if (mydevice->cuda11_RMR) free(mydevice->cuda11_RMR);
        mydevice->cuda11_RMR = NULL;
        mydevice->cuda11_RMR_count = 0;

        if (mydevice->cuda11_ValueIdx) free(mydevice->cuda11_ValueIdx);
        mydevice->cuda11_ValueIdx = NULL;

        if (mydevice->cuda11_MetricIdx) free(mydevice->cuda11_MetricIdx);
        mydevice->cuda11_MetricIdx = NULL;

        if (mydevice->cuda11_MetricNames) free(mydevice->cuda11_MetricNames);
        mydevice->cuda11_MetricNames = NULL;
        mydevice->cuda11_numMetricNames = 0;

        if (deviceRawMetricCount[dev] == 0) continue;

        // make some room.
        mydevice->cuda11_RMR_count = deviceRawMetricCount[dev];
        mydevice->cuda11_RMR=calloc(deviceRawMetricCount[dev], sizeof(NVPA_RawMetricRequest));
        if (mydevice->cuda11_RMR == NULL) return PAPI_ENOMEM;
        mydevice->cuda11_ValueIdx = calloc(deviceMetricCount[dev], sizeof(int));
        if (mydevice->cuda11_ValueIdx == NULL) return PAPI_ENOMEM;
        mydevice->cuda11_MetricIdx = calloc(deviceMetricCount[dev], sizeof(int));
        if (mydevice->cuda11_MetricIdx == NULL) return PAPI_ENOMEM;
        mydevice->cuda11_MetricNames = calloc(deviceMetricCount[dev], sizeof(char*));
        if (mydevice->cuda11_MetricNames == NULL) return PAPI_ENOMEM;
        mydevice->cuda11_numMetricNames = deviceMetricCount[dev];
        
        int evIdx, midx=0, idx=0;

        // We build two lists here. mydevice->cuda11_RMR[] is a list of all the
        // raw metric requests (look like 0x...) for all the events on this
        // device.  mydevice->cuda11_MetricNames[] is a list of all the name
        // level metrics we are trying to compute.
        for (ii =0; ii < nativeCount; ii++) {
            int index          = nativeInfo[ii].ni_event;                       // Get the index of the event (in the global context).
            if (cuda11_AllEvents[index]->deviceNum != dev) continue;            // Skip if not on current device.
            // Copy over to master list.
            mydevice->cuda11_ValueIdx[midx] = ii;                               // Position in user's list.
            mydevice->cuda11_MetricIdx[midx] = index;                           // Position in cuda11_AllEvents[].
            mydevice->cuda11_MetricNames[midx++] = cuda11_AllEvents[index]->nv_name;    // Nvidia name of metric.

            // NOTE: The sample code does not eliminate duplicates in the 
            //       list of raw metric events; since we don't know how the
            //       internal evaluation process works, we don't either.
            for (evIdx=0; evIdx < cuda11_AllEvents[index]->numRawMetrics; evIdx++) {
                mydevice->cuda11_RMR[idx] = cuda11_AllEvents[index]->rawMetricRequests[evIdx];
                if (0) fprintf(stderr, "%s:%s:%i ii=%d, index=%d, eventName=%s, RMR[%d].name='%s'.\n", 
                    __FILE__, __func__, __LINE__, ii, index, cuda11_AllEvents[index]->nv_name, idx, mydevice->cuda11_RMR[idx].pMetricName);
                idx++;
            }
        }

        // Check that added metrics can be evaluated in a single pass, else return error.
        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams;
        nvpw_metricsConfigCreateParams.structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE;
        nvpw_metricsConfigCreateParams.pPriv = NULL;
        nvpw_metricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        nvpw_metricsConfigCreateParams.pChipName = mydevice->cuda11_chipName;
        NVPW_CALL( (*NVPW_CUDA_RawMetricsConfig_CreatePtr)(&nvpw_metricsConfigCreateParams),
                    return(PAPI_ENOSUPP) );

        NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams;
        memset(&beginPassGroupParams, 0,  NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE);
        beginPassGroupParams.structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
        beginPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        NVPW_CALL((*NVPW_RawMetricsConfig_BeginPassGroupPtr)(&beginPassGroupParams),
                    return(PAPI_ENOSUPP));

        NVPW_RawMetricsConfig_AddMetrics_Params nvpw_rawmetricsconfig_addmetrics;
        nvpw_rawmetricsconfig_addmetrics.structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
        nvpw_rawmetricsconfig_addmetrics.pPriv = NULL;
        nvpw_rawmetricsconfig_addmetrics.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        nvpw_rawmetricsconfig_addmetrics.pRawMetricRequests = mydevice->cuda11_RMR;
        nvpw_rawmetricsconfig_addmetrics.numMetricRequests = mydevice->cuda11_RMR_count;
        NVPW_CALL( (*NVPW_RawMetricsConfig_AddMetricsPtr)(&nvpw_rawmetricsconfig_addmetrics),
                    return(PAPI_ENOSUPP));

        NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams;
        endPassGroupParams.structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
        endPassGroupParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        NVPW_CALL((*NVPW_RawMetricsConfig_EndPassGroupPtr)(&endPassGroupParams),
                    return(PAPI_ENOSUPP));

        NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams;
        rawMetricsConfigGetNumPassesParams.structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE;
        rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        NVPW_CALL((*NVPW_RawMetricsConfig_GetNumPassesPtr)(&rawMetricsConfigGetNumPassesParams),
                    return(PAPI_ENOSUPP));
        int numNestingLevels = 1, numIsolatedPasses, numPipelinedPasses, numOfPasses;
        numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
        numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;

        numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams;
        rawMetricsConfigDestroyParams.structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
        rawMetricsConfigDestroyParams.pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig;
        NVPW_CALL((*NVPW_RawMetricsConfig_DestroyPtr)((NVPW_RawMetricsConfig_Destroy_Params*) &rawMetricsConfigDestroyParams),
                    return(PAPI_ENOSUPP));
        if (numOfPasses > 1) {
            SUBDBG("error: Metrics requested requires multiple passes to profile.\n");
            return PAPI_EMULPASS;
        }

    } // end each device.    

    // Free temp allocations.
    if (deviceRawMetricCount) free(deviceRawMetricCount);       
    if (deviceMetricCount)    free(deviceMetricCount);

    // See $PAPI_CUPTI_ROOT/samples/extensions/src/profilerhost_util/Metric.cpp
    // See $PAPI_CUPTI_ROOT/samples/userrange_profiling/simplecuda.cu

    _papi_hwi_unlock( COMPONENT_LOCK );
    return(PAPI_OK);
} // end_cuda11_update_control_state


// Triggered by PAPI_start().
// For CUDA component, switch to each context and start all eventgroups.
static int _cuda11_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    uint32_t dev;
    int err, userDevice = -1;
    CUcontext userCtx;
    CUcontext popCtx;

    // NOTE: Zero cumulative values for start
    //       (work to be done)
    // NOTE: Zero values for the local read.
    //       (work to be done)

    _papi_hwi_lock( COMPONENT_LOCK );

    CUDA_CALL((*cudaGetDevicePtr) (&userDevice),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    CU_CALL((*cuCtxGetCurrentPtr) (&userCtx),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    err = _cuda11_build_profiling_structures(userCtx);
    if (err != PAPI_OK) {
        if (1) fprintf(stderr, "%s:%s:%i _cuda11_build_profiling_structures() failed; err=%d.\n",
            __FILE__, __func__, __LINE__, err);
        return(err);
    }

    CUPTI_CALL((*cuptiGetTimestampPtr) (&gctrl->cuptiStartTimestampNs),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    for (dev=0; dev < (unsigned) gctxt->deviceCount; dev++) {
        int ctxPushed = 0;
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        // skip devices that have no events to start.
        if (mydevice->sessionCtx == NULL ||
            mydevice->cuda11_ConfigImage == NULL) continue;

        if (mydevice->sessionCtx != userCtx) {
            ctxPushed = 1;
            CU_CALL((*cuCtxPushCurrentPtr) (mydevice->sessionCtx),
                // On error,
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }

        // set up parameter structures in mydevice.
        memset(&mydevice->beginSessionParams,       0, CUpti_Profiler_BeginSession_Params_STRUCT_SIZE);
        mydevice->beginSessionParams.structSize     =  CUpti_Profiler_BeginSession_Params_STRUCT_SIZE;

        memset(&mydevice->setConfigParams,          0, Profiler_SetConfig_Params_STRUCT_SIZE);
        mydevice->setConfigParams.structSize        =  Profiler_SetConfig_Params_STRUCT_SIZE;

        // We begin a session.
        // mydevice->beginSessionParams.ctx = NULL; // NULL uses current cuda context.
        mydevice->beginSessionParams.ctx = mydevice->sessionCtx; // set to current context for this device.
        mydevice->beginSessionParams.counterDataImageSize = mydevice->cuda11_CounterDataImageSize;
        mydevice->beginSessionParams.pCounterDataImage = mydevice->cuda11_CounterDataImage;
        mydevice->beginSessionParams.counterDataScratchBufferSize = mydevice->cuda11_CounterDataScratchBufferSize;
        mydevice->beginSessionParams.pCounterDataScratchBuffer = mydevice->cuda11_CounterDataScratchBuffer;
        mydevice->beginSessionParams.range =        CUPTI_UserRange;
        mydevice->beginSessionParams.replayMode =   CUPTI_UserReplay;
        mydevice->beginSessionParams.maxRangesPerPass = 1;
        mydevice->beginSessionParams.maxLaunchesPerPass = 1;
        CUPTI_CALL(
            (*cuptiProfilerBeginSessionPtr) (&mydevice->beginSessionParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        mydevice->setConfigParams.ctx = mydevice->sessionCtx;
        mydevice->setConfigParams.pConfig = mydevice->cuda11_ConfigImage;
        mydevice->setConfigParams.configSize = mydevice->cuda11_ConfigImageSize;
        mydevice->setConfigParams.passIndex = 0;
        mydevice->setConfigParams.minNestingLevel = 1;
        mydevice->setConfigParams.numNestingLevels = 1;
        CUPTI_CALL(
            (*cuptiProfilerSetConfigPtr) (&mydevice->setConfigParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Build necessary structures; including those for reading/stopping.
        // Some of these structures are empty; if we don't set anything in 
        // them, we don't store them in mydevice.
        CUpti_Profiler_BeginPass_Params beginPassParams;
        memset(&beginPassParams, 0,  CUpti_Profiler_BeginPass_Params_STRUCT_SIZE);
        beginPassParams.structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE;
        beginPassParams.ctx =  mydevice->sessionCtx;

        memset(&mydevice->pushRangeParams, 0,  CUpti_Profiler_PushRange_Params_STRUCT_SIZE);
        mydevice->pushRangeParams.structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE;
        mydevice->pushRangeParams.ctx = mydevice->sessionCtx;

        //---------------------------------------------------------------------------------------------
        // See $PAPI_CUPTI_ROOT/samples/userrange_profiling/simplecuda.cu circa line 247.  
        // At this point, the sample code loops through all passes calling the kernel, with
        // cuptiProfiler functions: 
        // beginPass, EnableProfiling, pushRange, KernelCall, PopRange, DisableProfiling, endPass.
        // We do the first three functions; kernel calls are up to the user, the final three
        // (PopRange, Disable, endPass) be done before we can read.

        // Empirical Note: We can't skip BeginPass / EndPass, it causes hang ups in the Read.
        CUPTI_CALL(
            (*cuptiProfilerBeginPassPtr) (&beginPassParams), 
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        CUpti_Profiler_EnableProfiling_Params enableProfilingParams;
        memset(&enableProfilingParams,    0, CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE);
        enableProfilingParams.structSize  =  CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE;
        enableProfilingParams.ctx  = mydevice->sessionCtx;

        CUPTI_CALL(
            (*cuptiProfilerEnableProfilingPtr) (&enableProfilingParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // We only need one range per device. 
        snprintf(mydevice->cuda11_range_name, sizeof(((cuda_device_desc_t*)0)->cuda11_range_name), "PAPI_Range_%d", dev);
        mydevice->pushRangeParams.pRangeName = &mydevice->cuda11_range_name[0];
        CUPTI_CALL(
            (*cuptiProfilerPushRangePtr) (&mydevice->pushRangeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Restore previous context.
        if (ctxPushed) {
            ctxPushed = 0;
            CU_CALL((*cuCtxPopCurrentPtr) (&popCtx),
                // On error,
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }
    } // end for each device.

    _papi_hwi_unlock(COMPONENT_LOCK);

    return(PAPI_OK);
} //end _cuda11_start


// Triggered by PAPI_read().  For CUDA component, switch to each context, read
// all the eventgroups, and put the values in the correct places. Note that
// parameters (ctx, ctrl, flags) are all ignored. The design of this components
// doesn't pay attention to PAPI EventSets, because ONLY ONE is ever allowed
// for a component.  So instead of maintaining ctx and ctrl, we use global
// variables to keep track of the one and only eventset.  Note that **values is
// where we have to give PAPI the address of an array of the values we read (or
// composed).
// ALSO note, cuda resets all event counters to zero after a read, while PAPI
// promises monotonically increasing counters (from PAPI_start()). So we have
// to synthesize that.


static int _cuda11_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    (void) values;
    (void) flags;
    cuda_control_t *gctrl = global_cuda_control;
    cuda_context_t *gctxt = global_cuda_context;
    (void) gctrl;
    int i, dev;
    int ctxPushed=0, userDevice = -1;
    CUcontext userCtx, popCtx;

    _papi_hwi_lock( COMPONENT_LOCK );

    CUDA_CALL((*cudaGetDevicePtr) (&userDevice),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    CU_CALL((*cuCtxGetCurrentPtr) (&userCtx),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    for (dev=0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        if (mydevice->cuda11_ConfigImage == NULL) continue;
        if (mydevice->sessionCtx == NULL) continue;

        if (mydevice->sessionCtx != userCtx) {
            ctxPushed = 1;
            CU_CALL((*cuCtxPushCurrentPtr) (mydevice->sessionCtx), 
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }
            
        //---------------------------------------------------------------------------------------------
        // See $PAPI_CUPTI_ROOT/samples/userrange_profiling/simplecuda.cu circa line 247.  

        // simplecuda.cu:266: CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams))
        CUpti_Profiler_PopRange_Params popRangeParams;
        memset(&popRangeParams, 0,  CUpti_Profiler_PopRange_Params_STRUCT_SIZE);
        popRangeParams.structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE;
        popRangeParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerPopRangePtr) (&popRangeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:269: CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams))
        CUpti_Profiler_EndPass_Params endPassParams;
        memset(&endPassParams, 0,  Profiler_EndPass_Params_STRUCT_SIZE);
        endPassParams.structSize = Profiler_EndPass_Params_STRUCT_SIZE;
        endPassParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerEndPassPtr) (&endPassParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:272: CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams))
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams;
        memset(&flushCounterDataParams, 0,  Profiler_FlushCounterData_Params_STRUCT_SIZE);
        flushCounterDataParams.structSize = Profiler_FlushCounterData_Params_STRUCT_SIZE;
        flushCounterDataParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerFlushCounterDataPtr) (&flushCounterDataParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Evaluation of metrics collected in counterDataImage.
        // See  $PAPI_CUDA_ROOT/extras/CUPTI/samples/extensions/src/profilerhost_util/Eval.cpp
        // NV::Metric::Eval::GetMetricGpuValues(chipName, counterDataImage, metricNames, metricNameValueMap)
        if (mydevice->cuda11_CounterDataImageSize == 0) { // exit with problem if no image.
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC);
        }

        // Note: This is a "metricsContext", not the same as cuGetCurrentContext.
        // Eval.cpp:PrintMetricValues:121 pChipName='GV100'
        // Eval.cpp:PrintMetricValues:122 Call: NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams)
        
/******* The commented out code here compiles and worked correctly in development, but it is
 ******* not necessary in PAPI, and eliminated for efficiency. We don't need the multiple
 ******* ranges or their names; we have only one range per device.

        // Not necessary, we always have 1 range. 
        // Eval.cpp:PrintMetricValues:130 Call: NVPW_CounterData_GetNumRanges(&getNumRangesParams)
        NVPW_CounterData_GetNumRanges_Params getNumRangesParams;
        memset(&getNumRangesParams, 0,  NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE);
        getNumRangesParams.structSize = NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE;
        getNumRangesParams.pCounterDataImage = mydevice->cuda11_CounterDataImage;
        CUPTI_CALL(
            (*NVPW_CounterData_GetNumRangesPtr) (&getNumRangesParams),
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx),);
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Eval.cpp:PrintMetricValues:132 numRanges=1
        fprintf(stderr, "%s:%s:%i dev=%d getNumRangesParams.numRanges=%zd.\n", 
            __FILE__, __func__, __LINE__, dev, getNumRangesParams.numRanges); // DEBUG.

        // sample code loops over ranges here. We only have 1 in test code.
        // Eval.cpp:PrintMetricValues:137 metricNames.size()=1
        // Eval.cpp:PrintMetricValues:148 metricNames[0]='fe__cycles_elapsed.sum'

        // Eval.cpp:PrintMetricValues:158 getRangeDescParams.rangeIndex=0
        // Eval.cpp:PrintMetricValues:159 Call: NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams)
        // Setup params for "GetRangeDescriptions".
        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams;
        memset(&getRangeDescParams, 0,  NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE);
        getRangeDescParams.structSize = NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE;
        getRangeDescParams.pCounterDataImage = mydevice->cuda11_CounterDataImage;
        getRangeDescParams.rangeIndex = 0;
        // Call GetRangeDescriptions, but just to get NUMBER of descriptions.
        NVPW_CALL((*NVPW_Profiler_CounterData_GetRangeDescriptionsPtr) (&getRangeDescParams),
            CU_CALL(
            (*cuCtxPopCurrentPtr) (&currCuCtx),);
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Now we know the number, allocate space for range descriptions.
        char **rangeDescriptions = calloc(getRangeDescParams.numDescriptions, sizeof(char*));
        getRangeDescParams.ppDescriptions = (const char **) rangeDescriptions;

        // Eval.cpp:PrintMetricValues:161 getRangeDescParams.numDescriptions=1
        // Eval.cpp:PrintMetricValues:166 Call: NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams)
        // Call Get Range Descriptions again, get actual descriptions this time.
        NVPW_CALL((*NVPW_Profiler_CounterData_GetRangeDescriptionsPtr) (&getRangeDescParams),
            CU_CALL(
            (*cuCtxPopCurrentPtr) (&currCuCtx),);
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        fprintf(stderr, "%s:%s:%i dev=%d numRangeDescriptions=%zd, rangeDescription[0]='%s'\n", 
            __FILE__, __func__, __LINE__, dev, getRangeDescParams.numDescriptions, rangeDescriptions[0]);
******
******   END OF working but unnecessary code.
*****/

        // Space to receive values from profiler.        
        double *gpuValues = calloc(mydevice->cuda11_numMetricNames, sizeof(double) );

        NVPW_CUDA_MetricsContext_Create_Params *pMCCP = cuda11_getMetricsContextPtr(dev);

        // Eval.cpp:PrintMetricValues:188 setCounterDataParams.rangeIndex=0
        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams;
        memset(&setCounterDataParams, 0,   NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE);
        setCounterDataParams.structSize =  NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE;
        setCounterDataParams.pMetricsContext = pMCCP->pMetricsContext;
        setCounterDataParams.pCounterDataImage = mydevice->cuda11_CounterDataImage;
        setCounterDataParams.isolated = 1;
        setCounterDataParams.rangeIndex = 0; // Note Eval.cpp:155 uses zero relative; we have only one range per device.

        NVPW_CALL((*NVPW_MetricsContext_SetCounterDataPtr) (&setCounterDataParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams;
        memset(&evalToGpuParams, 0,  NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE);
        evalToGpuParams.structSize = NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE;
        evalToGpuParams.pMetricsContext = pMCCP->pMetricsContext;

        evalToGpuParams.numMetrics = mydevice->cuda11_numMetricNames;
        evalToGpuParams.ppMetricNames = (const char* const *) (mydevice->cuda11_MetricNames);
        evalToGpuParams.pMetricValues = &gpuValues[0];

        // Eval.cpp:PrintMetricValues:197 evalToGpuParams.numMetrics=1, evalToGpuParams.metricNamePtrs[0]='fe__cycles_elapsed.sum'
        NVPW_CALL((*NVPW_MetricsContext_EvaluateToGpuValuesPtr) (&evalToGpuParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC);
        );

        // Re-initialize CUDA11 profiling images to prevent erroneous values in repeated PAPI_read operations.
        CUPTI_CALL((*cuptiProfilerCounterDataImageInitializePtr) (&mydevice->cuda11_CounterDataInitParms),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));
        CUPTI_CALL((*cuptiProfilerCounterDataImageInitializeScratchBufferPtr) (&mydevice->cuda11_CounterScratchInitParms),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock( COMPONENT_LOCK );
            return(PAPI_EMISC));

        // It would be preferable to not do the following if we are reading for
        // a PAPI_stop(), but we can't know that without modifying the PAPI
        // main code; that is where the call is made, before _cuda11_stop() is
        // called.  HERE, we must set up for a new read: BeginPass, PushRange.

        CUpti_Profiler_BeginPass_Params beginPassParams;
        memset(&beginPassParams, 0,  CUpti_Profiler_BeginPass_Params_STRUCT_SIZE);
        beginPassParams.structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE;
        beginPassParams.ctx = mydevice->sessionCtx;

        CUPTI_CALL(
            (*cuptiProfilerBeginPassPtr) (&beginPassParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK);
            return(PAPI_EMISC)
        );

        // We re-use the structure we build in _cuda11_start.
        CUPTI_CALL(
            (*cuptiProfilerPushRangePtr) (&mydevice->pushRangeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK);
            return(PAPI_EMISC)
        );

        // END restart section.

        // In Sample code, this is where it executes:
        // cuptiProfilerUnsetConfig;
        // cuptiProfilerEndSession;
        // cuptiProfilerDeInitialize, <<-- not done here.
        // and destroys the cuda context <<-- not done here.
        // We have relocated the ProfileDeInitialize to _cuda11_shutdown().  We
        // leave context management (the destroy) up to the application, but it
        // does mean the rest of this code does not require a cuda context.
        // simplecuda.cu:394: DRIVER_API_CALL(cuCtxDestroy(cuContext))

        // Accumulate values if necessary, and move the values 
        // retrieved to gctrl->activeEventValues[].
        for (i=0; i<mydevice->cuda11_numMetricNames; i++) {
            int aeIdx = mydevice->cuda11_MetricIdx[i];
            // Adjust for treatment.
            switch (cuda11_AllEvents[aeIdx]->treatment) {
                case SpotValue:
                    if (0) fprintf(stderr, "%s:%s:%i SpotValue.\n", __FILE__, __func__, __LINE__);
                    break;

                case RunningMin:
                    if (0) fprintf(stderr, "%s:%s:%i RunningMax.\n", __FILE__, __func__, __LINE__);
                    if (gpuValues[i] < cuda11_AllEvents[aeIdx]->cumulativeValue) {
                        cuda11_AllEvents[aeIdx]->cumulativeValue = gpuValues[i];
                    }
                    gpuValues[i] = cuda11_AllEvents[aeIdx]->cumulativeValue;
                    break;

                case RunningMax:
                    if (0) fprintf(stderr, "%s:%s:%i RunningMax.\n", __FILE__, __func__, __LINE__);
                    if (gpuValues[i] > cuda11_AllEvents[aeIdx]->cumulativeValue) {
                        cuda11_AllEvents[aeIdx]->cumulativeValue = gpuValues[i];
                    }
                    gpuValues[i] = cuda11_AllEvents[aeIdx]->cumulativeValue;
                    break;

                case RunningSum:
                    if (0) fprintf(stderr, "%s:%s:%i Adding %f to cumulative value %f.\n", __FILE__, __func__, __LINE__, gpuValues[i], cuda11_AllEvents[aeIdx]->cumulativeValue);
                    cuda11_AllEvents[aeIdx]->cumulativeValue += gpuValues[i];
                    gpuValues[i] = cuda11_AllEvents[aeIdx]->cumulativeValue;
                    break;
            }

            int userIdx = mydevice->cuda11_ValueIdx[i];
            gctrl->activeEventValues[userIdx] = (int64_t) (gpuValues[i]);
        }

        free(gpuValues);

        // Restore previous context.
        if (ctxPushed) {
            ctxPushed = 0;
            CU_CALL((*cuCtxPopCurrentPtr) (&popCtx),
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
        }
    } // end for each device.

    *values = gctrl->activeEventValues;     // Full list of computed values to user.
    _papi_hwi_unlock(COMPONENT_LOCK);
    return(PAPI_OK);
} // end _cuda11_read


// Triggered by PAPI_stop().
static int _cuda11_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    cuda_context_t *gctxt = global_cuda_context;
    int dev;
    int ctxPushed=0, userDevice = -1;
    CUcontext userCtx, popCtx;

    _papi_hwi_lock( COMPONENT_LOCK );

    CUDA_CALL((*cudaGetDevicePtr) (&userDevice),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    CU_CALL((*cuCtxGetCurrentPtr) (&userCtx),
        _papi_hwi_unlock( COMPONENT_LOCK ); return (PAPI_EMISC));

    if (0) fprintf(stderr, "%s:%s:%i userDevice=%d userCtx=%p.\n", __FILE__, __func__, __LINE__, userDevice, userCtx);

    for (dev=0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        if (mydevice->cuda11_ConfigImage == NULL) continue;
        if (mydevice->sessionCtx == NULL) continue;

        if (mydevice->sessionCtx != userCtx) {
            ctxPushed = 1;
            CU_CALL((*cuCtxPushCurrentPtr) (mydevice->sessionCtx), 
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
            if (0) fprintf(stderr, "%s:%s:%i userCtx=%p pushed for sessionCtx=%p.\n", __FILE__, __func__, __LINE__, userCtx, mydevice->sessionCtx);
        }

        // We need to shut down the profiler; every time we do a read in
        // cuda11, we have to shut it down (up to FlushCounterData), and then
        // start it up. It would be preferable to skip that restart in
        // _cuda11_read() if we knew it was part of a cuda_stop(), but that
        // requires modifying the PAPI main code.

        // simplecuda.cu:266: CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams))
        CUpti_Profiler_PopRange_Params popRangeParams;
        memset(&popRangeParams, 0,  CUpti_Profiler_PopRange_Params_STRUCT_SIZE);
        popRangeParams.structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE;
        popRangeParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerPopRangePtr) (&popRangeParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:267: CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams))
        CUpti_Profiler_DisableProfiling_Params disableProfilingParams;
        memset(&disableProfilingParams,    0, CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE);
        disableProfilingParams.structSize  =  CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE;
        disableProfilingParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerDisableProfilingPtr) (&disableProfilingParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:269: CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams))
        CUpti_Profiler_EndPass_Params endPassParams;
        memset(&endPassParams, 0,  Profiler_EndPass_Params_STRUCT_SIZE);
        endPassParams.structSize = Profiler_EndPass_Params_STRUCT_SIZE;
        endPassParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerEndPassPtr) (&endPassParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:272: CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams))
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams;
        memset(&flushCounterDataParams, 0,  Profiler_FlushCounterData_Params_STRUCT_SIZE);
        flushCounterDataParams.structSize = Profiler_FlushCounterData_Params_STRUCT_SIZE;
        flushCounterDataParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerFlushCounterDataPtr) (&flushCounterDataParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:274: CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams))
        CUpti_Profiler_UnsetConfig_Params unsetConfigParams;
        memset(&unsetConfigParams, 0,  CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE);
        unsetConfigParams.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE;
        unsetConfigParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerUnsetConfigPtr) (&unsetConfigParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // simplecuda.cu:276: CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams))
        CUpti_Profiler_EndSession_Params endSessionParams;
        memset(&endSessionParams, 0,  CUpti_Profiler_EndSession_Params_STRUCT_SIZE);
        endSessionParams.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE;
        endSessionParams.ctx = mydevice->sessionCtx;
        CUPTI_CALL(
            (*cuptiProfilerEndSessionPtr) (&endSessionParams),
            // On error,
            if (ctxPushed) CU_CALL((*cuCtxPopCurrentPtr) (&popCtx), );
            _papi_hwi_unlock(COMPONENT_LOCK); 
            return(PAPI_EMISC)
        );

        // Restore previous context.
        if (ctxPushed) {
            ctxPushed = 0;
            CU_CALL((*cuCtxPopCurrentPtr) (&popCtx),
                _papi_hwi_unlock( COMPONENT_LOCK );
                return(PAPI_EMISC));
            if (0) fprintf(stderr, "%s:%s:%i PopCurrent popCtx=%p.\n", __FILE__, __func__, __LINE__, popCtx);
        }
    } // end for each device.

    _papi_hwi_unlock(COMPONENT_LOCK);
    return(PAPI_OK);
} // end _cuda11_stop


// Disable and destroy the CUDA eventGroup
static int _cuda11_cleanup_eventset(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    int dev;
    (void) ctrl;
    cuda_context_t *gctxt = global_cuda_context;

    _papi_hwi_lock( COMPONENT_LOCK );
    for (dev=0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        // Free all the allocations the profiler needs, to rebuild them.
        if (mydevice->cuda11_ConfigImage) free(mydevice->cuda11_ConfigImage);
        mydevice->cuda11_ConfigImage = NULL;
        mydevice->cuda11_ConfigImageSize = 0;

        if (mydevice->cuda11_CounterDataPrefixImage) free(mydevice->cuda11_CounterDataPrefixImage); 
        mydevice->cuda11_CounterDataPrefixImage = NULL; 
        mydevice->cuda11_CounterDataPrefixImageSize = 0; 

        if (mydevice->cuda11_CounterDataImage) free(mydevice->cuda11_CounterDataImage); 
        mydevice->cuda11_CounterDataImage = NULL; 
        mydevice->cuda11_CounterDataImageSize = 0; 

        if (mydevice->cuda11_CounterDataScratchBuffer) free(mydevice->cuda11_CounterDataScratchBuffer); 
        mydevice->cuda11_CounterDataScratchBuffer = NULL; 
        mydevice->cuda11_CounterDataScratchBufferSize = 0; 

        if (mydevice->cuda11_RMR) free(mydevice->cuda11_RMR);
        mydevice->cuda11_RMR = NULL;

        mydevice->cuda11_numMetricNames = 0;

        if (mydevice->cuda11_ValueIdx) free(mydevice->cuda11_ValueIdx); 
        mydevice->cuda11_ValueIdx = NULL;
        
        if (mydevice->cuda11_MetricIdx) free(mydevice->cuda11_MetricIdx);
        mydevice->cuda11_MetricIdx = NULL;

        // cuda11_MetricNames is char**, but the individual pointers are not
        // alloced so they don't have to be released. 
        if (mydevice->cuda11_MetricNames) free(mydevice->cuda11_MetricNames); 
        mydevice->cuda11_MetricNames = NULL;
    }

    _papi_hwi_unlock( COMPONENT_LOCK );
    return(PAPI_OK);
} // end _cuda11_cleanup_eventset


// Called at thread shutdown. Does nothing in the CUDA component.
static int _cuda11_shutdown_thread(hwd_context_t * ctx)
{
    SUBDBG("Entering\n");
    (void) ctx;
    // nothing to do in cuda11.
    return (PAPI_OK);
} // end _cuda11_shutdown_thread 

// Triggered by PAPI_shutdown() and frees memory allocated in the CUDA component.
static int _cuda11_shutdown_component(void)
{
    cuda_context_t *gctxt = global_cuda_context;
    int i, dev;

    _papi_hwi_lock( COMPONENT_LOCK );

    // Release (for all devices) any EventSet allocations.
    _cuda11_cleanup_eventset(NULL);

    // Release the hash table entries.
    freeEntireNameHash();

    // destroy all metrics contexts.
    cuda11_destroyMetricsContexts();

    // release device specific allocs.
    for (dev = 0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice;
        mydevice = &gctxt->deviceArray[dev];

        if (mydevice->domainIDArray) free(mydevice->domainIDArray);
        mydevice->domainIDArray = NULL;

        if (mydevice->domainIDNumEvents) free(mydevice->domainIDNumEvents);
        mydevice->domainIDNumEvents = NULL;
    }

    // Release all memory in cuda11_AllEvents, then the array itself.
    for (i=0; i < cuda11_numEvents; i++) {
        // free all elements of one event.
        if (cuda11_AllEvents[i]) {
            free_cuda11_eventData_contents(cuda11_AllEvents[i]);
            free(cuda11_AllEvents[i]);
        }
    }

    // Free the whole table of pointers.
    free(cuda11_AllEvents); 

    if (global_cuda_control) free(global_cuda_control);

    // simplecuda.cu:392: CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams))
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams;
    memset(&profilerDeInitializeParams, 0,  CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE);
    profilerDeInitializeParams.structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE;
    CUPTI_CALL(
        (*cuptiProfilerDeInitializePtr) (&profilerDeInitializeParams),
        _papi_hwi_unlock(COMPONENT_LOCK); 
        return(PAPI_EMISC)
    );

    _papi_hwi_unlock( COMPONENT_LOCK );

    // close the dynamic libraries needed by this component (opened in the init substrate call)
    if (dl1) {
        dlclose(dl1);
    }
    if (dl2) {
        dlclose(dl2);
    }
    if (dl3) {
        dlclose(dl3);
    }
    if (dl4) {
        dlclose(dl4);
    }

    return(PAPI_OK);
} // end _cuda11_shutdown_component.


// Triggered by PAPI_reset() but only if the EventSet is currently running. If
// the eventset is not currently running, then the saved value in the EventSet
// is set to zero without calling this routine.
static int _cuda11_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    (void) ctx;
    (void) ctrl;

    cuda_context_t *gctxt = global_cuda_context;
    int i, dev;

    _papi_hwi_lock( COMPONENT_LOCK );
    for (dev=0; dev < gctxt->deviceCount; dev++) {
        cuda_device_desc_t *mydevice = &gctxt->deviceArray[dev];
        for (i=0; i<mydevice->cuda11_numMetricNames; i++) {
            int aeIdx = mydevice->cuda11_MetricIdx[i];
            // Adjust for treatment.
            switch (cuda11_AllEvents[aeIdx]->treatment) {
                case SpotValue: cuda11_AllEvents[aeIdx]->cumulativeValue = 0; break;
                case RunningSum: cuda11_AllEvents[aeIdx]->cumulativeValue = 0; break;
                case RunningMin: cuda11_AllEvents[aeIdx]->cumulativeValue = DBL_MAX; break;
                case RunningMax: cuda11_AllEvents[aeIdx]->cumulativeValue = -DBL_MAX; break;
            }
        }
    }

    _papi_hwi_unlock( COMPONENT_LOCK );
    return(PAPI_OK);
} // end _cuda11_reset


// This function sets various options in the component - Does nothing in the CUDA component.
//  @param[in] ctx -- hardware context
//  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
//  @param[in] option -- options to be set
static int _cuda11_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) code;
    (void) option;
    return (PAPI_OK);
} // end _cuda11_ctrl


// This function has to set the bits needed to count different domains
// In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
// By default return PAPI_EINVAL if none of those are specified
// and PAPI_OK with success
// PAPI_DOM_USER is only user context is counted
// PAPI_DOM_KERNEL is only the Kernel/OS context is counted
// PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
// PAPI_DOM_ALL   is all of the domains
static int _cuda11_set_domain(hwd_control_state_t * ctrl, int domain)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain))
        return (PAPI_OK);
    else
        return (PAPI_EINVAL);
    return (PAPI_OK);
} // end _cuda11_set_domain


// Enumerate Native Events.
//   @param EventCode is the event of interest
//   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
static int _cuda11_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    // SUBDBG( "Entering (get next event after %u)\n", *EventCode );
    switch (modifier) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return (PAPI_OK);
        break;
    case PAPI_ENUM_EVENTS:
        if (global_cuda_context == NULL) {
            return (PAPI_ENOEVNT);
        } else if (*EventCode < (unsigned) (cuda11_numEvents-1)) {
            *EventCode = *EventCode + 1;
            return (PAPI_OK);
        } else {
            return (PAPI_ENOEVNT);
        }
        break;
    default:
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
} // end _cuda11_ntv_enum_events

static int _cuda11_ntv_name_to_code(const char *nameIn, unsigned int *out)
{
    (void) out;
    char *myName = strstr(nameIn, ":::");
    if (myName != NULL) myName +=3;
    else myName = (char*) nameIn;

    if (0) fprintf(stderr, "%s:%s:%i on entry, name='%s', myName='%s'.\n", __FILE__, __func__, __LINE__, nameIn, myName);
    int myIdx = findNameHash(myName);
    if (myIdx < 0) return(PAPI_EINVAL);
    *out = (unsigned int) myIdx;
    if (0) fprintf(stderr, "%s:%s:%i Found, returning myIdx=%d, papi_name='%s'.\n", __FILE__, __func__, __LINE__, myIdx, cuda11_AllEvents[myIdx]->papi_name);
	return(PAPI_OK);
} // end _cuda11_ntv_name_to_code

// Takes a native event code and passes back the name
// @param EventCode is the native event code
// @param name is a pointer for the name to be copied to
// @param len is the size of the name string
static int _cuda11_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    SUBDBG( "Entering EventCode %d\n", EventCode );
    unsigned int index = EventCode;
    int dev = cuda11_AllEvents[index]->deviceNum;
    cuda_device_desc_t *mydevice;
    cuda_context_t *gctxt = global_cuda_context;
    mydevice = &gctxt->deviceArray[dev];

    NVPW_CUDA_MetricsContext_Create_Params *pMCCP = cuda11_getMetricsContextPtr(dev);

    int err  = cuda11_getMetricDetails(cuda11_AllEvents[index], 
                 mydevice->cuda11_chipName, pMCCP);

    if (err != PAPI_OK) {
        if (0) fprintf(stderr, "%s:%s:%i index=%d err=%d.\n", __FILE__, __func__, __LINE__, index, err);
        return(err);
    }

    if (cuda11_AllEvents != NULL && index < (unsigned int) cuda11_numEvents) {
        strncpy(name, cuda11_AllEvents[index]->papi_name, len);
    } else {
        return (PAPI_EINVAL);
    }
    // SUBDBG( "Exit: EventCode %d: Name %s\n", EventCode, name );
    return (PAPI_OK);
} // end _cuda11_ntv_code_to_name


// Takes a native event code and passes back the event description
// @param EventCode is the native event code
// @param descr is a pointer for the description to be copied to
// @param len is the size of the descr string
static int _cuda11_ntv_code_to_descr(unsigned int EventCode, char *desc, int len)
{
    SUBDBG( "Entering\n" );
    unsigned int index = EventCode;
    int dev = cuda11_AllEvents[index]->deviceNum;
    cuda_device_desc_t *mydevice;
    cuda_context_t *gctxt = global_cuda_context;
    mydevice = &gctxt->deviceArray[dev];

    NVPW_CUDA_MetricsContext_Create_Params *pMCCP = cuda11_getMetricsContextPtr(dev);

    int err  = cuda11_getMetricDetails(cuda11_AllEvents[index], 
                 mydevice->cuda11_chipName, pMCCP);

    if (err != PAPI_OK) {
        if (0) fprintf(stderr, "%s:%s:%i index=%d err=%d.\n", __FILE__, __func__, __LINE__, index, err);
        return(err);
    }

    if (cuda11_AllEvents != NULL && index < (unsigned int) cuda11_numEvents) {
        strncpy(desc, cuda11_AllEvents[index]->description, len);

    } else {
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
} // end _cuda11_ntv_code_to_descr

//-------------------------------------------------------------------------------------------------
// Change _cuda_vector functions to cuda11.
//-------------------------------------------------------------------------------------------------
static void _cuda11_cuda_vector(void)
{
    _cuda_vector.start = _cuda11_start;    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    _cuda_vector.stop = _cuda11_stop;      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    _cuda_vector.read = _cuda11_read;      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    _cuda_vector.reset = _cuda11_reset;    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    _cuda_vector.cleanup_eventset = _cuda11_cleanup_eventset;      /* ( hwd_control_state_t * ctrl ) */

//  _cuda_vector.init_component  is unchanged.
    _cuda_vector.init_thread = _cuda11_init_thread;        /* ( hwd_context_t * ctx ) */
    _cuda_vector.init_control_state = _cuda11_init_control_state;  /* ( hwd_control_state_t * ctrl ) */
    _cuda_vector.update_control_state = _cuda11_update_control_state;      /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */

    _cuda_vector.ctl = _cuda11_ctrl;       /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    _cuda_vector.set_domain = _cuda11_set_domain;  /* ( hwd_control_state_t * cntrl, int domain ) */
    _cuda_vector.ntv_enum_events = _cuda11_ntv_enum_events;        /* ( unsigned int *EventCode, int modifier ) */
    _cuda_vector.ntv_name_to_code = _cuda11_ntv_name_to_code;      /* ( unsigned char *name, int *code ) */
    _cuda_vector.ntv_code_to_name = _cuda11_ntv_code_to_name;      /* ( unsigned int EventCode, char *name, int len ) */
    _cuda_vector.ntv_code_to_descr = _cuda11_ntv_code_to_descr;    /* ( unsigned int EventCode, char *name, int len ) */
    _cuda_vector.shutdown_thread = _cuda11_shutdown_thread;        /* ( hwd_context_t * ctx ) */
    _cuda_vector.shutdown_component = _cuda11_shutdown_component;  /* ( void ) */
} // end _cuda11_cuda_vector 
#endif // CUPTI_API_VERSION >= 13
