/**
 * @file    linux-cuda.c
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
// the "adding" stage for events (update_control), becuase users are likely not
// measuring performance at those times, but may well be reading these events
// when performance matters. So we want the read operation lightweight, but we
// can remember tables and such at startup and when servicing a PAPI_add().
//-----------------------------------------------------------------------------

#include <dlfcn.h>
#include <cupti.h>
#include <cuda_runtime_api.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

// We use a define so we can use it as a static array dimension. Increase as needed.
#define PAPICUDA_MAX_COUNTERS 512

// #define PAPICUDA_KERNEL_REPLAY_MODE
// w to punctuate an embedded quoted question within a declarative sentence? [duplicate]

// Contains device list, pointer to device desciption, and the list of all available events.
typedef struct papicuda_context {
    int         deviceCount;
    struct papicuda_device_desc *deviceArray;
    uint32_t    availEventSize;
    CUpti_ActivityKind *availEventKind;
    int         *availEventDeviceNum;
    uint32_t    *availEventIDArray;
    uint32_t    *availEventIsBeingMeasuredInEventset;
    struct papicuda_name_desc *availEventDesc;
} papicuda_context_t;

/* Store the name and description for an event */
typedef struct papicuda_name_desc {
    char        name[PAPI_MAX_STR_LEN];
    char        description[PAPI_2MAX_STR_LEN];
    uint16_t        numMetricEvents;        // 0=event, if a metric, size of metricEvents array below.
    CUpti_EventID   *metricEvents;          // NULL for cuda events, an array of member events if a metric.
    CUpti_MetricValueKind MV_Kind;          // eg. % or counter or rate, etc. Needed to compute metric from individual events.
} papicuda_name_desc_t;

/* For a device, store device description */
typedef struct papicuda_device_desc {
    CUdevice    cuDev;
    int         deviceNum;
    char        deviceName[PAPI_MIN_STR_LEN];
    uint32_t    maxDomains;                 /* number of domains per device */
    CUpti_EventDomainID *domainIDArray;     /* Array[maxDomains] of domain IDs */
    uint32_t    *domainIDNumEvents;         /* Array[maxDomains] of num of events in that domain */
} papicuda_device_desc_t;

// For each active cuda context (one measuring something) we also track the
// cuda device number it is on. We track in separate arrays for each reading
// method.  cuda metrics and nvlink metrics require multiple events to be read,
// these are then arithmetically combined to produce the metric value. The
// allEvents array stores all the actual events; i.e. metrics are deconstructed
// to their individual events and stored there, as well as regular events, so
// we can perform an analysis of how to read with cuptiEventGroupSetsCreate(). 

typedef struct papicuda_active_cucontext_s {
    CUcontext cuCtx;
    int deviceNum;

    uint32_t      ctxActiveCount;                               // Count of entries in ctxActiveEvents.
    uint32_t      ctxActiveEvents    [PAPICUDA_MAX_COUNTERS];   // index into gctrl->activeEventXXXX arrays, so we can store values.

    uint32_t      allEventsCount;                               // entries in allEvents array.
    CUpti_EventID allEvents          [PAPICUDA_MAX_COUNTERS];   // allEvents, including sub-events of metrics. (no metric Ids in here).
    uint64_t      allEventValues     [PAPICUDA_MAX_COUNTERS];   // aggregated event values.

    CUpti_EventGroupSets *eventGroupSets;                       // Built during add, to save time not doing it at read.
} papicuda_active_cucontext_t;

// Control structure tracks array of active contexts and active events 
// in the order the user requested them; along with associated values 
// values and types (to save lookup time).
typedef struct papicuda_control {
    uint32_t    countOfActiveCUContexts;
    papicuda_active_cucontext_t *arrayOfActiveCUContexts[PAPICUDA_MAX_COUNTERS];
    uint32_t    activeEventCount;
    int         activeEventIndex            [PAPICUDA_MAX_COUNTERS];    // index into gctxt->availEventXXXXX arrays.
    long long   activeEventValues           [PAPICUDA_MAX_COUNTERS];    // values we will return.
    CUpti_MetricValueKind activeEventKind   [PAPICUDA_MAX_COUNTERS];    // For metrics: double, uint64, % or throughput. Needed to compute metric from individual events.
    uint64_t    cuptiStartTimestampNs;                                  // needed to compute duration for some metrics.
    uint64_t    cuptiReadTimestampNs;                                   // ..
} papicuda_control_t;

// file handles used to access cuda libraries with dlopen
static void *dl1 = NULL;
static void *dl2 = NULL;
static void *dl3 = NULL;

/* The PAPI side (external) variable as a global */
papi_vector_t _cuda_vector;

/* Global variable for hardware description, event and metric lists */
static papicuda_context_t *global_papicuda_context = NULL;

/* This global variable points to the head of the control state list */
static papicuda_control_t *global_papicuda_control = NULL;

/* Macros for error checking... each arg is only referenced/evaluated once */
#define CHECK_PRINT_EVAL( checkcond, str, evalthis )                        \
    do {                                                                    \
        int _cond = (checkcond);                                            \
        if (_cond) {                                                        \
            SUBDBG("error: condition %s failed: %s.\n", #checkcond, str);   \
            evalthis;                                                       \
        }                                                                   \
    } while (0)

#define CUDA_CALL( call, handleerror )                                              \
    do {                                                                            \
        cudaError_t _status = (call);                                               \
        if (_status != cudaSuccess) {                                               \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status);   \
            handleerror;                                                            \
        }                                                                           \
    } while (0)

#define CU_CALL( call, handleerror )                                                \
    do {                                                                            \
        CUresult _status = (call);                                                  \
        if (_status != CUDA_SUCCESS) {                                              \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status);   \
            /* fprintf(stderr,"Line %i CU_CALL error function %s failed with error %d.\n", __LINE__, #call, _status); */  \
            handleerror;                                                            \
        }                                                                           \
    } while (0)


#define CUPTI_CALL(call, handleerror)                                                                       \
    do {                                                                                                    \
        CUptiResult _status = (call);                                                                       \
        if (_status != CUPTI_SUCCESS) {                                                                     \
            const char *errstr;                                                                             \
            (*cuptiGetResultStringPtr)(_status, &errstr);                                                   \
            SUBDBG("error: function %s failed with error %s.\n", #call, errstr);                            \
            /* fprintf(stderr, "Line %i CUPTI_CALL macro '%s' failed with error '%s'.\n", __LINE__, #call, errstr); */ \
            handleerror;                                                                                    \
        }                                                                                                   \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

/* Function prototypes */
static int papicuda_cleanup_eventset(hwd_control_state_t * ctrl);

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
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;
DECLARECUFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARECUFUNC(cuCtxSetCurrent, (CUcontext));
DECLARECUFUNC(cuCtxDestroy, (CUcontext));
DECLARECUFUNC(cuCtxCreate, (CUcontext *pctx, unsigned int flags, CUdevice dev));
DECLARECUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUFUNC(cuDeviceGetCount, (int *));
DECLARECUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUFUNC(cuInit, (unsigned int));
DECLARECUFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARECUFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARECUFUNC(cuCtxSynchronize, ());

#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARECUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##Ptr ) funcsig;
DECLARECUDAFUNC(cudaGetDevice, (int *));
DECLARECUDAFUNC(cudaSetDevice, (int));
DECLARECUDAFUNC(cudaFree, (void *));

#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARECUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  CUptiResult( *funcname##Ptr ) funcsig;
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


/*****************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 *****************************************************************************/

/* 
 * Link the necessary CUDA libraries to use the cuda component.  If any of them can not be found, then
 * the CUDA component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the CUDA component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */
static int papicuda_linkCudaLibraries()
{
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror()!=NULL ) { strncpy( _cuda_vector.cmp_info.disabled_reason, "A CUDA required function was not found in dynamic libs", PAPI_MAX_STR_LEN ); return ( PAPI_ENOSUPP ); }

    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        strncpy(_cuda_vector.cmp_info.disabled_reason, "The CUDA component does not support statically linking to libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }
    /* Need to link in the cuda libraries, if not found disable the component */
    dl1 = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    CHECK_PRINT_EVAL(!dl1, "CUDA library libcuda.so not found.", return (PAPI_ENOSUPP));
    cuCtxGetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxGetCurrent");
    cuCtxSetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxSetCurrent");
    cuDeviceGetPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetName");
    cuInitPtr = DLSYM_AND_CHECK(dl1, "cuInit");
    cuCtxPopCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPushCurrent");
    cuCtxDestroyPtr = DLSYM_AND_CHECK(dl1, "cuCtxDestroy");
    cuCtxCreatePtr  = DLSYM_AND_CHECK(dl1, "cuCtxCreate");
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl1, "cuCtxSynchronize");

    dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    CHECK_PRINT_EVAL(!dl2, "CUDA runtime library libcudart.so not found.", return (PAPI_ENOSUPP));
    cudaGetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaGetDevice");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl2, "cudaFree");

    dl3 = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    CHECK_PRINT_EVAL(!dl3, "CUDA Profiling Tools Interface (CUPTI) library libcupti.so not found.", return (PAPI_ENOSUPP));
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
    return (PAPI_OK);
}


static int papicuda_add_native_events(papicuda_context_t * gctxt)
{
    SUBDBG("Entering\n");
    CUresult cuErr;
    int deviceNum;
    uint32_t domainNum, eventNum;
    papicuda_device_desc_t *mydevice;
    char tmpStr[PAPI_MIN_STR_LEN];
    tmpStr[PAPI_MIN_STR_LEN - 1] = '\0';
    size_t tmpSizeBytes;
    int ii;
    uint32_t maxEventSize;

    /* How many CUDA devices do we have? */
    cuErr = (*cuDeviceGetCountPtr) (&gctxt->deviceCount);
    if(cuErr == CUDA_ERROR_NOT_INITIALIZED) {
        /* If CUDA not initialized, initialize CUDA and retry the device list */
        /* This is required for some of the PAPI tools, that do not call the init functions */
        if(((*cuInitPtr) (0)) != CUDA_SUCCESS) {
            strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA cannot be found and initialized (cuInit failed).", PAPI_MAX_STR_LEN);
            return PAPI_ENOSUPP;
        }
        CU_CALL((*cuDeviceGetCountPtr) (&gctxt->deviceCount), return (PAPI_EMISC));
    }

    if(gctxt->deviceCount == 0) {
        strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA initialized but no CUDA devices found.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }
    SUBDBG("Found %d devices\n", gctxt->deviceCount);

    /* allocate memory for device information */
    gctxt->deviceArray = (papicuda_device_desc_t *) papi_calloc(gctxt->deviceCount, sizeof(papicuda_device_desc_t));
    CHECK_PRINT_EVAL(!gctxt->deviceArray, "ERROR CUDA: Could not allocate memory for CUDA device structure", return (PAPI_ENOMEM));

    /* For each device, get domains and domain-events counts */
    maxEventSize = 0;
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        mydevice = &gctxt->deviceArray[deviceNum];
        /* Get device id, name, numeventdomains for each device */
        CU_CALL((*cuDeviceGetPtr) (&mydevice->cuDev, deviceNum),                // get CUdevice.
            return (PAPI_EMISC));                                               // .. on failure.

        CU_CALL((*cuDeviceGetNamePtr) (mydevice->deviceName,                    // get device name,
            PAPI_MIN_STR_LEN - 1, mydevice->cuDev),                             // .. max length,
            return (PAPI_EMISC));                                               // .. on failure.

        mydevice->deviceName[PAPI_MIN_STR_LEN - 1] = '\0';                      // z-terminate it.

        CUPTI_CALL((*cuptiDeviceGetNumEventDomainsPtr)                          // get number of domains,
            (mydevice->cuDev, &mydevice->maxDomains), 
            return (PAPI_EMISC));                                               // .. on failure.

        /* Allocate space to hold domain IDs */
        mydevice->domainIDArray = (CUpti_EventDomainID *) papi_calloc(
            mydevice->maxDomains, sizeof(CUpti_EventDomainID));                 

        CHECK_PRINT_EVAL(!mydevice->domainIDArray, "ERROR CUDA: Could not allocate memory for CUDA device domains", return (PAPI_ENOMEM));

        /* Put domain ids into allocated space */
        size_t domainarraysize = mydevice->maxDomains * sizeof(CUpti_EventDomainID);
        CUPTI_CALL((*cuptiDeviceEnumEventDomainsPtr)                            // enumerate domain ids into space.
            (mydevice->cuDev, &domainarraysize, mydevice->domainIDArray), 
            return (PAPI_EMISC));                                               // .. on failure.

        /* Allocate space to hold domain event counts */
        mydevice->domainIDNumEvents = (uint32_t *) papi_calloc(mydevice->maxDomains, sizeof(uint32_t));
        CHECK_PRINT_EVAL(!mydevice->domainIDNumEvents, "ERROR CUDA: Could not allocate memory for domain event counts", return (PAPI_ENOMEM));

        /* For each domain, get event counts in domainNumEvents[] */
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {     // For each domain,
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];  // .. make a copy of the domain ID.
            /* Get num events in domain */
            CUPTI_CALL((*cuptiEventDomainGetNumEventsPtr)                       // Get number of events in this domain,
                (domainID, &mydevice->domainIDNumEvents[domainNum]),            // .. store in array.
                return (PAPI_EMISC));                                           // .. on failure.

            maxEventSize += mydevice->domainIDNumEvents[domainNum];             // keep track of overall number of events.
        } // end for each domain.
    } // end of for each device.

    // Increase maxEventSize for metrics on this device.
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {               // for each device,
        uint32_t maxMetrics = 0;
        CUptiResult cuptiRet;
        mydevice = &gctxt->deviceArray[deviceNum];                                  // Get papicuda_device_desc pointer.
        cuptiRet = (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics);   // Read the # metrics on this device.
        if (cuptiRet != CUPTI_SUCCESS || maxMetrics < 1) continue;                  // If no metrics, skip to next device.
        maxEventSize += maxMetrics;                                                 // make room for metrics we discover later.
    } // end for each device.

    /* Allocate space for all events and descriptors */
    gctxt->availEventKind = (CUpti_ActivityKind *) papi_calloc(maxEventSize, sizeof(CUpti_ActivityKind));
    CHECK_PRINT_EVAL(!gctxt->availEventKind, "ERROR CUDA: Could not allocate memory", return (PAPI_ENOMEM));
    gctxt->availEventDeviceNum = (int *) papi_calloc(maxEventSize, sizeof(int));
    CHECK_PRINT_EVAL(!gctxt->availEventDeviceNum, "ERROR CUDA: Could not allocate memory", return (PAPI_ENOMEM));
    gctxt->availEventIDArray = (CUpti_EventID *) papi_calloc(maxEventSize, sizeof(CUpti_EventID));
    CHECK_PRINT_EVAL(!gctxt->availEventIDArray, "ERROR CUDA: Could not allocate memory", return (PAPI_ENOMEM));
    gctxt->availEventIsBeingMeasuredInEventset = (uint32_t *) papi_calloc(maxEventSize, sizeof(uint32_t));
    CHECK_PRINT_EVAL(!gctxt->availEventIsBeingMeasuredInEventset, "ERROR CUDA: Could not allocate memory", return (PAPI_ENOMEM));
    gctxt->availEventDesc = (papicuda_name_desc_t *) papi_calloc(maxEventSize, sizeof(papicuda_name_desc_t));
    CHECK_PRINT_EVAL(!gctxt->availEventDesc, "ERROR CUDA: Could not allocate memory", return (PAPI_ENOMEM));

    // Record all events on each device, and their descriptions.
    uint32_t idxEventArray = 0;
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {           // loop through each device.
        mydevice = &gctxt->deviceArray[deviceNum];                              // get a pointer to the papicuda_device_desc struct.

        // For each domain, get and store event IDs, names, descriptions.
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {         // loop through the domains in this device.

            /* Get domain id */
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];      // get the domain id,
            uint32_t domainNumEvents = mydevice->domainIDNumEvents[domainNum];      // get the number of events in it.

            // SUBDBG( "For device %d domain %d domainID %d numEvents %d\n", mydevice->cuDev, domainNum, domainID, domainNumEvents );

            CUpti_EventID *domainEventIDArray =                                         // Make space for the events in this domain.
                (CUpti_EventID *) papi_calloc(domainNumEvents, sizeof(CUpti_EventID));  // .. 
            CHECK_PRINT_EVAL(!domainEventIDArray, "ERROR CUDA: Could not allocate memory for events", return (PAPI_ENOMEM));

            size_t domainEventArraySize = domainNumEvents * sizeof(CUpti_EventID);      // compute size of array we allocated.
            CUPTI_CALL((*cuptiEventDomainEnumEventsPtr)                                 // Enumerate the events in the domain,
                (domainID, &domainEventArraySize, domainEventIDArray),                  // .. 
                return (PAPI_EMISC));                                                   // .. on failure, exit.

            for(eventNum = 0; eventNum < domainNumEvents; eventNum++) {                 // Loop through the events in this domain.
                CUpti_EventID myeventCuptiEventId = domainEventIDArray[eventNum];       // .. get this event,
                gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_EVENT;       // .. record the kind,
                gctxt->availEventIDArray[idxEventArray] = myeventCuptiEventId;          // .. record the id,
                gctxt->availEventDeviceNum[idxEventArray] = deviceNum;                  // .. record the device number,

                tmpSizeBytes = PAPI_MIN_STR_LEN - 1 * sizeof(char);                     // .. compute size of name,
                CUPTI_CALL((*cuptiEventGetAttributePtr) (myeventCuptiEventId,           // .. Get the event name seen by cupti,
                    CUPTI_EVENT_ATTR_NAME, &tmpSizeBytes, tmpStr),                      // .. into tmpStr.
                    return (PAPI_EMISC));                                               // .. on failure, exit routine.

                snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN,   // record expaneded name for papi user.
                    "event:%s:device=%d", tmpStr, deviceNum);
                gctxt->availEventDesc[idxEventArray].name[PAPI_MIN_STR_LEN - 1] = '\0'; // ensure null termination.
                char *nameTmpPtr = gctxt->availEventDesc[idxEventArray].name;           // For looping, get pointer to name.
                for(ii = 0; ii < (int) strlen(nameTmpPtr); ii++) {                      // Replace spaces with underscores.
                    if(nameTmpPtr[ii] == ' ') nameTmpPtr[ii] = '_';                     // ..
                }

                /* Save description in the native event array */
                tmpSizeBytes = PAPI_2MAX_STR_LEN - 1 * sizeof(char);                    // Most space to use for description.
                CUPTI_CALL((*cuptiEventGetAttributePtr) (myeventCuptiEventId,           // Get it, 
                    CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &tmpSizeBytes,                  // .. Set limit (and recieve bytes written),
                    gctxt->availEventDesc[idxEventArray].description),                  // .. in the description.
                    return (PAPI_EMISC));                                               // .. on failure.
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
    SUBDBG("Checking for metrics\n");
    for (deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        uint32_t maxMetrics = 0, i, j;
        CUpti_MetricID *metricIdList = NULL;
        CUptiResult cuptiRet;
        mydevice = &gctxt->deviceArray[deviceNum];                                  // Get papicuda_device_desc pointer.
        cuptiRet = (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics);   // Read the # metrics on this device.
        if (cuptiRet != CUPTI_SUCCESS || maxMetrics < 1) continue;                  // If no metrics, skip to next device.

        SUBDBG("Device %d: Checking each of the (maxMetrics) %d metrics\n", deviceNum, maxMetrics);

        // Make a temporary list of the metric Ids to add to the available named collectables.
        size_t size = maxMetrics * sizeof(CUpti_EventID);                                   
        metricIdList = (CUpti_MetricID *) papi_calloc(maxMetrics, sizeof(CUpti_EventID));
        CHECK_PRINT_EVAL(metricIdList == NULL, "Out of memory", return (PAPI_ENOMEM));

        CUPTI_CALL((*cuptiDeviceEnumMetricsPtr)                                     // Enumerate the metric Ids for this device,
            (mydevice->cuDev, &size, metricIdList),                                 // .. into metricIdList.
            return (PAPI_EMISC));                                                   // .. On failure, but should work, we have metrics!

        // Elimination loop for metrics we cannot support.
        int saveDeviceNum = 0;
        CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));       // save caller's device num.

        for (i=0, j=0; i<maxMetrics; i++) {                                         // process each metric Id.
            size = PAPI_MIN_STR_LEN-1;                                              // Most bytes allowed to be written.
            CUPTI_CALL((*cuptiMetricGetAttributePtr) (metricIdList[i],              // Get the name.
                CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *) tmpStr), 
                return (PAPI_EMISC));
            
            // Note that 'size' also returned total bytes written.
            tmpStr[size] = '\0';

            if (strcmp("branch_efficiency", tmpStr) == 0) continue;                 // If it is branch efficiency, skip it.

            // We'd like to reject anything requiring more than 1
            // set, but there is a problem I cannot find; I have
            // been unable to create a CUcontext here so I can
            // execute the CreateEventGroups. I've tried both
            // ways, it returns an error saying no cuda devices
            // available.  There does not seem to be a way to get
            // the number of "sets" (passes) for a metric without
            // having a context.

            // CUpti_EventGroupSets *thisEventGroupSets = NULL;
            //CUPTI_CALL ((*cuptiMetricCreateEventGroupSetsPtr) (
            //    tempContext, 
            //    sizeof(CUpti_MetricID), 
            //    &metricIdList[i], 
            //    &thisEventGroupSets),
            //    return (PAPI_EMISC));
            //
            //int numSets = 0;                                                        // # of sets (passes) required.
            //if (thisEventGroupSets != NULL) {
            //    numSets=thisEventGroupSets->numSets;                                // Get sets if a grouping is necessary.
            //    CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr) (thisEventGroupSets),   // Done with this.
            //        return (PAPI_EMISC));
            //}
            //
            //if (numSets > 1) continue;                                              // skip this metric too many passes.

            metricIdList[j++] = metricIdList[i];                                    // we are compressing if we skipped any.
        } // end elimination loop.

        // Done with eliminations, the rest are valid. 
        maxMetrics = j;                                                             // Change the number to process.

        // Eliminations accomplished, now add the valid metric Ids to the list.
        for(i = 0; i < maxMetrics; i++) {                                           // for each id,
            gctxt->availEventIDArray[idxEventArray] = metricIdList[i];              // add to the list of collectables. 
            gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_METRIC;      // Indicate it is a metric.
            gctxt->availEventDeviceNum[idxEventArray] = deviceNum;                  // remember the device number.
            size = PAPI_MIN_STR_LEN;
            CUPTI_CALL((*cuptiMetricGetAttributePtr) (metricIdList[i],              // Get the name, fail if we cannot.
                CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *) tmpStr), 
                return (PAPI_EMISC));

            if (size >= PAPI_MIN_STR_LEN) {                                         // Truncate if we don't have room for the name.
                gctxt->availEventDesc[idxEventArray].name[PAPI_MIN_STR_LEN - 1] = '\0';
            }

            size_t MV_KindSize = sizeof(CUpti_MetricValueKind);
            CUPTI_CALL((*cuptiMetricGetAttributePtr)                                // Collect the metric kind. 
                (metricIdList[i], CUPTI_METRIC_ATTR_VALUE_KIND, &MV_KindSize,       // .. for this metric,
                &gctxt->availEventDesc[idxEventArray].MV_Kind),                     // .. store in the event description,
                return (PAPI_EMISC));                                               // .. on failure, but should always work.

            snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN,   // .. develop name for papi user in tmpStr.
                "metric:%s:device=%d", tmpStr, deviceNum);

            size = PAPI_2MAX_STR_LEN-1;                                             // Most bytes to return.
            CUPTI_CALL((*cuptiMetricGetAttributePtr)                                // Collect the long description.
                (metricIdList[i], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size,        // .. for this metric, no more than size.
                (uint8_t *) gctxt->availEventDesc[idxEventArray].description),      // .. and store in event description.
                return (PAPI_EMISC));                                               // .. on failure, but should always work.

            // Note that 'size' also returned total bytes written.             
            gctxt->availEventDesc[idxEventArray].description[size] = '\0';          // Always z-terminate.

            // Now we get all the sub-events of this metric.
            uint32_t numSubs;
            CUpti_MetricID itemId = metricIdList[i];                                //.. shortcut to metric id.
            CUPTI_CALL((*cuptiMetricGetNumEventsPtr) (itemId, &numSubs),            // .. Get number of sub-events in metric.
                return (PAPI_EINVAL));                                              // .. on failure of call.

            size_t sizeBytes = numSubs * sizeof(CUpti_EventID);                     // .. compute size of array we need.
            CUpti_EventID *subEventIds = papi_malloc(sizeBytes);                    // .. Make the space.
            CHECK_PRINT_EVAL(subEventIds == NULL, "Malloc failed",                  // .. If malloc fails,
                return (PAPI_ENOMEM));

            CUPTI_CALL((*cuptiMetricEnumEventsPtr)                                  // .. Enumrate events in the metric.
                (itemId, &sizeBytes, subEventIds),                                  // .. store in array.
                return (PAPI_EINVAL));                                              // .. If cupti call fails.

            gctxt->availEventDesc[idxEventArray].metricEvents = subEventIds;        // .. Copy the array pointer for IDs.
            gctxt->availEventDesc[idxEventArray].numMetricEvents = numSubs;         // .. Copy number of elements in it.

            idxEventArray++;                                                        // count another collectable found.
        } // end maxMetrics loop.

        papi_free(metricIdList);                                                    // Done with this enumeration of metrics.
        // Part of problem above, cannot create tempContext for unknown reason.
        // CU_CALL((*cuCtxDestroyPtr) (tempContext),     return (PAPI_EMISC));         // destroy the temporary context.
        CUDA_CALL((*cudaSetDevicePtr) (saveDeviceNum),  return (PAPI_EMISC));       // set the device pointer back to caller.
    } // end 'for each device'.

    gctxt->availEventSize = idxEventArray;

    /* return 0 if everything went OK */
    return 0;
} // end papicuda_add_native_events


/*
  This routine tries to convert all CUPTI values to long long values.
  If the CUPTI value is an integer type, it is cast to long long.  If
  the CUPTI value is a percent, it is multiplied by 100 to return the
  integer percentage.  If the CUPTI value is a double, the value
  is cast to long long... this can be a severe truncation.
 */
static int papicuda_convert_metric_value_to_long_long(CUpti_MetricValue metricValue, CUpti_MetricValueKind valueKind, long long int *papiValue)
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
static int papicuda_init_thread(hwd_context_t * ctx)
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
static int papicuda_init_component(int cidx)
{
    SUBDBG("Entering with component idx: %d\n", cidx);
    int rv;

    /* link in all the cuda libraries and resolve the symbols we need to use */
    if(papicuda_linkCudaLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of CUDA libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    /* Create the structure */
    if(!global_papicuda_context)
        global_papicuda_context = (papicuda_context_t *) papi_calloc(1, sizeof(papicuda_context_t));

    /* Get list of all native CUDA events supported */
    rv = papicuda_add_native_events(global_papicuda_context);
    if(rv != 0)
        return (rv);

    /* Export some information */
    _cuda_vector.cmp_info.CmpIdx = cidx;
    _cuda_vector.cmp_info.num_native_events = global_papicuda_context->availEventSize;
    _cuda_vector.cmp_info.num_cntrs = _cuda_vector.cmp_info.num_native_events;
    _cuda_vector.cmp_info.num_mpx_cntrs = _cuda_vector.cmp_info.num_native_events;

    return (PAPI_OK);
} // end init_component


/* Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int papicuda_init_control_state(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    papicuda_context_t *gctxt = global_papicuda_context;

    CHECK_PRINT_EVAL(!gctxt, "Error: The PAPI CUDA component needs to be initialized first", return (PAPI_ENOINIT));
    /* If no events were found during the initial component initialization, return error */
    if(global_papicuda_context->availEventSize <= 0) {
        strncpy(_cuda_vector.cmp_info.disabled_reason, "ERROR CUDA: No events exist", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }
    /* If it does not exist, create the global structure to hold CUDA contexts and active events */
    if(!global_papicuda_control) {
        global_papicuda_control = (papicuda_control_t *) papi_calloc(1, sizeof(papicuda_control_t));
        global_papicuda_control->countOfActiveCUContexts = 0;
        global_papicuda_control->activeEventCount = 0;
    }

    return PAPI_OK;
} // end papicuda_init_control_state

/* Triggered by eventset operations like add or remove.  For CUDA, needs to be
 * called multiple times from each seperate CUDA context with the events to be
 * measured from that context.  For each context, create eventgroups for the
 * events.
 */

/* Note: NativeInfo_t is defined in papi_internal.h */
static int papicuda_update_control_state(hwd_control_state_t * ctrl, 
    NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    SUBDBG("Entering with nativeCount %d\n", nativeCount);
    (void) ctx;
    papicuda_control_t *gctrl = global_papicuda_control;    // We don't use the passed-in parameter, we use a global.
    papicuda_context_t *gctxt = global_papicuda_context;    // We don't use the passed-in parameter, we use a global.
    int currDeviceNum;
    CUcontext currCuCtx;
    int eventContextIdx;
    CUcontext eventCuCtx;
    int index, ii, ee, cc;

    /* Return if no events */
    if(nativeCount == 0)
        return (PAPI_OK);

    /* Get deviceNum, initialize context if needed via free, get context */
    CUDA_CALL((*cudaGetDevicePtr) (&currDeviceNum), return (PAPI_EMISC)); 
    SUBDBG("currDeviceNum %d \n", currDeviceNum);

    CUDA_CALL((*cudaFreePtr) (NULL), return (PAPI_EMISC));
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
        if (gctxt->availEventIsBeingMeasuredInEventset[index] == 1) {       // If already being collected, skip it.
            SUBDBG("Skipping event %s which is already added\n", eventName);
            continue;
        } else {
            gctxt->availEventIsBeingMeasuredInEventset[index] = 1;          // If not being collected yet, flag it as being collected now.
        }

        /* Find context/control in papicuda, creating it if does not exist */
        for(cc = 0; cc < (int) gctrl->countOfActiveCUContexts; cc++) {              // Scan all active contexts.
            CHECK_PRINT_EVAL(cc >= PAPICUDA_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (PAPICUDA_MAX_COUNTERS)", return (PAPI_EMISC));

            if(gctrl->arrayOfActiveCUContexts[cc]->deviceNum == eventDeviceNum) {   // If this cuda context is for the device for this event,
                eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;             // Remember that context. 
                SUBDBG("Event %s device %d already has a cuCtx %p registered\n", eventName, eventDeviceNum, eventCuCtx);

                if(eventCuCtx != currCuCtx)                                         // If that is not our CURRENT context, push and make it so.
                    CU_CALL((*cuCtxPushCurrentPtr) (eventCuCtx),                    // .. Stack the current counter, replace with this one.
                        return (PAPI_EMISC));                                       // .. .. on failure.
                break;                                                              // .. exit the loop.
            } // end if found.
        } // end loop through active contexts. 

        if(cc == (int) gctrl->countOfActiveCUContexts) {                            // If we never found the context, create one.
            SUBDBG("Event %s device %d does not have a cuCtx registered yet...\n", eventName, eventDeviceNum);
            if(currDeviceNum != eventDeviceNum) {                           // .. If we need to switch to another device,
                CUDA_CALL((*cudaSetDevicePtr) (eventDeviceNum),             // .. .. set the device pointer to the event's device.
                    return (PAPI_EMISC));                                   // .. .. .. (on faiure).
                CUDA_CALL((*cudaFreePtr) (NULL), return (PAPI_EMISC));      // .. .. This is a no-op, but used to force init of a context.
                CU_CALL((*cuCtxGetCurrentPtr) (&eventCuCtx),                // .. .. So we can get a pointer to it.
                    return (PAPI_EMISC));                                   // .. .. .. On failure.
            } else {                                                        // .. If we are already on the right device,
                eventCuCtx = currCuCtx;                                     // .. .. just get the current context.
            }

            gctrl->arrayOfActiveCUContexts[cc] = papi_calloc(1, sizeof(papicuda_active_cucontext_t));   // allocate a structure.
            CHECK_PRINT_EVAL(gctrl->arrayOfActiveCUContexts[cc] == NULL, "Memory allocation for new active context failed", return (PAPI_ENOMEM));
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
        papicuda_active_cucontext_t *eventctrl = gctrl->arrayOfActiveCUContexts[eventContextIdx];   // get the context for this event.

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
                return(PAPI_EINVAL);
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
                    return(PAPI_EINVAL);
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
                    (eventctrl->eventGroupSets), return (PAPI_EMISC));              // .. If we can't, return error.
                eventctrl->eventGroupSets = NULL;                                   // .. Reset pointer.
            }

            size_t sizeBytes = (eventctrl->allEventsCount) * sizeof(CUpti_EventID); // compute bytes in the array.

            // SUBDBG("About to create eventGroupPasses for the context (sizeBytes %zu) \n", sizeBytes);
#ifdef PAPICUDA_KERNEL_REPLAY_MODE
            CUPTI_CALL((*cuptiEnableKernelReplayModePtr) (eventCuCtx), 
                return (PAPI_ECMP));
            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr) 
                (eventCuCtx, sizeBytes, eventctrl->allEvents, 
                &eventctrl->eventGroupSets), 
                return (PAPI_ECMP));

#else // Normal operation.
            CUPTI_CALL((*cuptiSetEventCollectionModePtr)
                (eventCuCtx,CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS),
                return(PAPI_ECMP));

// CUPTI provides two routines to create EventGroupSets, one is used
// here cuptiEventGroupSetsCreate(), the other is for metrics, it will
// automatically collect the events needed for a metric. It is called
// cuptiMetricCreateEventGroupSets(). We have checked and these two routines
// produce groups of the same size with the same event IDs, and work equally.

            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr) 
                (eventCuCtx, sizeBytes, eventctrl->allEvents, 
                &eventctrl->eventGroupSets), 
                return (PAPI_EMISC));

            if (eventctrl->eventGroupSets->numSets > 1) {                       // If more than one pass is required, 
                SUBDBG("Error occurred: The combined CUPTI events cannot be collected simultaneously ... try different events\n");
                papicuda_cleanup_eventset(ctrl);                                // Will do cuptiEventGroupSetsDestroy() to clean up memory.
                return(PAPI_ECOMBO);
            } else  {
                SUBDBG("Created eventGroupSets. nativeCount %d, allEventsCount %d. Sets (passes-required) = %d) \n", gctrl->activeEventCount, eventctrl->allEventsCount, eventctrl->eventGroupSets->numSets);
            }

#endif // #if/#else/#endif on PAPICUDA_KERNEL_REPLAY_MODE

        } // end if we had any events.
        
        if(eventCuCtx != currCuCtx)                                                 // restore original context for caller, if we changed it. 
            CU_CALL((*cuCtxPopCurrentPtr) (&eventCuCtx), return (PAPI_EMISC));

    }
    return (PAPI_OK);
} // end PAPI_update_control_state.


/* Triggered by PAPI_start().
 * For CUDA component, switch to each context and start all eventgroups.
*/
static int papicuda_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    // papicuda_context_t *gctxt = global_papicuda_context;
    uint32_t ii, gg, cc;
    int saveDeviceNum = -1;

    SUBDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)                             // These are the values we will return.
        gctrl->activeEventValues[ii] = 0;

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    CUPTI_CALL((*cuptiGetTimestampPtr) (&gctrl->cuptiStartTimestampNs), return (PAPI_EMISC));

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {                    // For each context, 
        int eventDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;     // .. get device number.
        CUcontext eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;       // .. get this context,
        SUBDBG("Set to device %d cuCtx %p \n", eventDeviceNum, eventCuCtx);
        if(eventDeviceNum != saveDeviceNum) {                                   // .. If we need to switch,
            CU_CALL((*cuCtxPushCurrentPtr) (eventCuCtx), return (PAPI_EMISC));  // .. .. push current on stack, use this one.
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
                    return (PAPI_EMISC));                                       // .. .. on failure of call.
            } // end for each group.

            CUPTI_CALL((*cuptiEventGroupSetEnablePtr) (groupset),               // .. Enable all groups in set (start collecting).
                return (PAPI_EMISC));                                           // .. on failure of call.

        if(eventDeviceNum != saveDeviceNum) {                                   // .. If we pushed a context,
            CU_CALL((*cuCtxPopCurrentPtr) (&eventCuCtx), return (PAPI_EMISC));  // .. Pop it.
        }
    } // end of loop on all contexts.

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

static int papicuda_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    (void) flags;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    uint32_t gg, i, j, cc;
    int saveDeviceNum;

    // Get read time stamp
    CUPTI_CALL((*cuptiGetTimestampPtr)                                          // Read current timestamp.
        (&gctrl->cuptiReadTimestampNs), 
        return (PAPI_EMISC));
    uint64_t durationNs = gctrl->cuptiReadTimestampNs - 
                          gctrl->cuptiStartTimestampNs;                         // compute duration from start.
    gctrl->cuptiStartTimestampNs = gctrl->cuptiReadTimestampNs;                 // Change start to value just read.

    SUBDBG("Save current context, then switch to each active device/context and enable context-specific eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));       // Save Caller's current device number on entry.

    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {                    // For each active context,
        papicuda_active_cucontext_t *activeCuCtxt = 
            gctrl->arrayOfActiveCUContexts[cc];                                 // A shortcut.
        int currDeviceNum = activeCuCtxt->deviceNum;                            // Get the device number.
        CUcontext currCuCtx = activeCuCtxt->cuCtx;                              // Get the actual CUcontext.

        SUBDBG("Set to device %d cuCtx %p \n", currDeviceNum, currCuCtx);
        if(currDeviceNum != saveDeviceNum) {                                    // If my current is not the same as callers,
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), return (PAPI_EMISC));   // .. Push the current, and replace with mine.
            // Note, cuCtxPushCurrent()  implicitly includes a cudaSetDevice().
        } else {                                                                // If my current IS the same as callers, 
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), return (PAPI_EMISC));    // .. No push. Just set the current.
        }

        CU_CALL((*cuCtxSynchronizePtr) (), return (PAPI_EMISC));                // Block until device finishes all prior tasks.
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
                return (PAPI_EMISC));

            // 'numTotalInstances' and 'numInstances are needed for scaling
            // the values retrieved. (Nvidia instructions and samples).
            CUPTI_CALL((*cuptiDeviceGetEventDomainAttributePtr)                 // Get 'numTotalInstances' for this domain.
                (cudevice, 
                groupDomainID, 
                CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, 
                &sizeofuint32num, 
                &numTotalInstances), 
                return (PAPI_EMISC));

            CUPTI_CALL((*cuptiEventGroupGetAttributePtr)                        // Get 'numInstances' for this domain.
                (group, 
                CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                &sizeofuint32num, 
                &numInstances), 
                return (PAPI_EMISC));

            CUPTI_CALL((*cuptiEventGroupGetAttributePtr)                        // Get 'numEvents' in this group.
                (group, 
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &sizeofuint32num, 
                &numEvents), 
                return (PAPI_EMISC));

            // Now we will read all events in this group; aggregate the values
            // and then distribute them.  We do not calculate metrics here;
            // wait until all groups are read and all values are available. 

            size_t resultArrayBytes        = sizeof(uint64_t) * numEvents * numTotalInstances;
            size_t eventIdArrayBytes       = sizeof(CUpti_EventID) * numEvents;
            size_t numCountersRead         = 2;

            CUpti_EventID *eventIdArray = (CUpti_EventID *) papi_malloc(eventIdArrayBytes);
            uint64_t *resultArray       = (uint64_t *)      papi_malloc(resultArrayBytes);
            uint64_t *aggrResultArray   = (uint64_t *)      papi_calloc(numEvents, sizeof(uint64_t));

            for (i=0; i<(resultArrayBytes/sizeof(uint64_t)); i++) resultArray[i]=0;

            if (eventIdArray == NULL || resultArray == NULL || aggrResultArray == NULL) {
                fprintf(stderr, "%s:%i failed to allocate memory.\n", __FILE__, __LINE__);
                return(PAPI_EMISC);
            }

            CUPTI_CALL( (*cuptiEventGroupReadAllEventsPtr)                      // Read all events.
                (group, CUPTI_EVENT_READ_FLAG_NONE,                             // This flag is the only allowed flag.
                &resultArrayBytes, resultArray, 
                &eventIdArrayBytes, eventIdArray, 
                &numCountersRead),
                return (PAPI_EMISC));

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
        // and allEventValues[] are populated. Now we compute metrics and move
        // event values. We do that by looping through the events assigned to
        // this context, and we must back track to the activeEventIdx[] and
        // activeEventValues[] array in gctrl. We have kept our indexes into
        // that array, in ctxActive[]. 

        uint32_t ctxActiveCount =  activeCuCtxt->ctxActiveCount;                // Number of (papi user) events in this context.
        uint32_t *ctxActive =  activeCuCtxt->ctxActiveEvents;                   // index of each event in gctrl->activeEventXXXX.

        for (j=0; j<ctxActiveCount; j++) {                                      // Search for matching active event.
            uint32_t activeIdx, availIdx;
                
            activeIdx=ctxActive[j];                                             // get index into activeEventIdx.
            availIdx = gctrl->activeEventIndex[activeIdx];                      // Get the availEventIdx.
            CUpti_EventID thisEventId = gctxt->availEventIDArray[availIdx];     // Get the event ID (or metric ID).
            struct papicuda_name_desc *myDesc=&(gctxt->availEventDesc[availIdx]);  // get pointer to the description.
            
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
                CUPTI_CALL( (*cuptiMetricGetValue)                              // Get the value,
                    (cudevice, thisEventId,                                     // device and metric Id,
                    AEIdx * sizeof(CUpti_EventID),                              // size of event list,
                    activeCuCtxt->allEvents,                                    // the event list.
                    AEIdx * sizeof(uint64_t),                                   // size of corresponding event values,
                    activeCuCtxt->allEventValues,                               // the event values.
                    durationNs, &myValue),                                      // duration (for rates), and where to return the value.
                    return(PAPI_EMISC));                                        // In case of error.

                papicuda_convert_metric_value_to_long_long(                     // convert the value computed to long long and store it.
                    myValue, myDesc->MV_Kind, 
                    &gctrl->activeEventValues[activeIdx]); 
            }
        } // end loop on active events in this context.

        if(currDeviceNum != saveDeviceNum) {                                    // If we had to change the context from user's,
            CUDA_CALL((*cudaSetDevicePtr) (saveDeviceNum),                      // set the device pointer to the user's original. 
                return (PAPI_EMISC));                                           // .. .. (on faiure).
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));   // .. pop the pushed context back to user's.
        }
    } // end of loop for each active context.

    *values = gctrl->activeEventValues;                                         // Return ptr to the list of computed values to user.
    return (PAPI_OK);
} // end of papicuda_read().

/* Triggered by PAPI_stop() */
static int papicuda_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    uint32_t cc, ss;
    int saveDeviceNum;

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        SUBDBG("Set to device %d cuCtx %p \n", currDeviceNum, currCuCtx);
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        else
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;
        for (ss=0; ss<currEventGroupSets->numSets; ss++) {                      // For each group in the set,
            CUpti_EventGroupSet groupset = currEventGroupSets->sets[ss];        // get the set,
            CUPTI_CALL((*cuptiEventGroupSetDisablePtr) (&groupset),             // disable the whole set.
                return (PAPI_EMISC));                                           // .. on failure.
        }
        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));

    }
    return (PAPI_OK);
} // end of papicuda_stop.


/* 
 * Disable and destroy the CUDA eventGroup
 */
static int papicuda_cleanup_eventset(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;                                                    // Don't need this parameter.
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    // papicuda_active_cucontext_t *currctrl;
    uint32_t cc;
    int saveDeviceNum;
    unsigned int ui;

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        else
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        //CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr) (currEventGroupPasses), return (PAPI_EMISC));
        (*cuptiEventGroupSetsDestroyPtr) (currEventGroupSets);
        gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets = NULL;
        papi_free( gctrl->arrayOfActiveCUContexts[cc] );
        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    }
    /* Record that there are no active contexts or events */
    for (ui=0; ui<gctrl->activeEventCount; ui++) {              // For each active event,
        int idx = gctrl->activeEventIndex[ui];                  // .. Get its index...
        gctxt->availEventIsBeingMeasuredInEventset[idx] = 0;    // .. No longer being measured.
    }
 
    gctrl->countOfActiveCUContexts = 0;
    gctrl->activeEventCount = 0;
    return (PAPI_OK);
} // end papicuda_cleanup_eventset


/* Called at thread shutdown. Does nothing in the CUDA component. */
int papicuda_shutdown_thread(hwd_context_t * ctx)
{
    SUBDBG("Entering\n");
    (void) ctx;

    return (PAPI_OK);
}

/* Triggered by PAPI_shutdown() and frees memory allocated in the CUDA component. */
static int papicuda_shutdown_component(void)
{
    SUBDBG("Entering\n");
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    int deviceNum;
    uint32_t i, cc;
    /* Free context */
    if(gctxt) {
        for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
            papicuda_device_desc_t *mydevice = &gctxt->deviceArray[deviceNum];
            papi_free(mydevice->domainIDArray);
            papi_free(mydevice->domainIDNumEvents);
        }

        for (i=0; i<gctxt->availEventSize; i++) {                               // For every event in this context,
            struct papicuda_name_desc *desc = &(gctxt->availEventDesc[i]);      // get a name description.
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
        global_papicuda_context = gctxt = NULL;
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
        global_papicuda_control = gctrl = NULL;
    }
    // close the dynamic libraries needed by this component (opened in the init substrate call)
    dlclose(dl1);
    dlclose(dl2);
    dlclose(dl3);
    return (PAPI_OK);
} // end papicuda_shutdown_component().


/* Triggered by PAPI_reset() but only if the EventSet is currently
 *  running. If the eventset is not currently running, then the saved
 *  value in the EventSet is set to zero without calling this
 *  routine.  */
static int papicuda_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    (void) ctx;
    (void) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    uint32_t gg, ii, cc, ss;
    int saveDeviceNum;

    SUBDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    SUBDBG("Save current context, then switch to each active device/context and reset\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        else
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        CUpti_EventGroupSets *currEventGroupSets = gctrl->arrayOfActiveCUContexts[cc]->eventGroupSets;
        for (ss=0; ss<currEventGroupSets->numSets; ss++) {
            CUpti_EventGroupSet groupset = currEventGroupSets->sets[ss]; 
            for(gg = 0; gg < groupset.numEventGroups; gg++) {
                CUpti_EventGroup group = groupset.eventGroups[gg];
                CUPTI_CALL((*cuptiEventGroupResetAllEventsPtr) (group), return (PAPI_EMISC));
            }
            CUPTI_CALL((*cuptiEventGroupSetEnablePtr) (&groupset), return (PAPI_EMISC));
        }
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    }
    return (PAPI_OK);
} // end papicuda_reset().


/* This function sets various options in the component - Does nothing in the CUDA component.
    @param[in] ctx -- hardware context
    @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
    @param[in] option -- options to be set
*/
static int papicuda_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
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
static int papicuda_set_domain(hwd_control_state_t * ctrl, int domain)
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
static int papicuda_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    // SUBDBG( "Entering (get next event after %u)\n", *EventCode );
    switch (modifier) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return (PAPI_OK);
        break;
    case PAPI_ENUM_EVENTS:
        if(*EventCode < global_papicuda_context->availEventSize - 1) {
            *EventCode = *EventCode + 1;
            return (PAPI_OK);
        } else
            return (PAPI_ENOEVNT);
        break;
    default:
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
}


/* Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int papicuda_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    // SUBDBG( "Entering EventCode %d\n", EventCode );
    unsigned int index = EventCode;
    papicuda_context_t *gctxt = global_papicuda_context;
    if(index < gctxt->availEventSize) {
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
static int papicuda_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
    // SUBDBG( "Entering\n" );
    unsigned int index = EventCode;
    papicuda_context_t *gctxt = global_papicuda_context;
    if(index < gctxt->availEventSize) {
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
                 }
    ,
    /* sizes of framework-opaque component-private structures... these are all unused in this component */
    .size = {
             .context = 1,      /* sizeof( papicuda_context_t ), */
             .control_state = 1,        /* sizeof( papicuda_control_t ), */
             .reg_value = 1,    /* sizeof( papicuda_register_t ), */
             .reg_alloc = 1,    /* sizeof( papicuda_reg_alloc_t ), */
             }
    ,
    /* function pointers in this component */
    .start = papicuda_start,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .stop = papicuda_stop,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .read = papicuda_read,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    .reset = papicuda_reset,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .cleanup_eventset = papicuda_cleanup_eventset,      /* ( hwd_control_state_t * ctrl ) */

    .init_component = papicuda_init_component,  /* ( int cidx ) */
    .init_thread = papicuda_init_thread,        /* ( hwd_context_t * ctx ) */
    .init_control_state = papicuda_init_control_state,  /* ( hwd_control_state_t * ctrl ) */
    .update_control_state = papicuda_update_control_state,      /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */

    .ctl = papicuda_ctrl,       /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    .set_domain = papicuda_set_domain,  /* ( hwd_control_state_t * cntrl, int domain ) */
    .ntv_enum_events = papicuda_ntv_enum_events,        /* ( unsigned int *EventCode, int modifier ) */
    .ntv_code_to_name = papicuda_ntv_code_to_name,      /* ( unsigned int EventCode, char *name, int len ) */
    .ntv_code_to_descr = papicuda_ntv_code_to_descr,    /* ( unsigned int EventCode, char *name, int len ) */
    .shutdown_thread = papicuda_shutdown_thread,        /* ( hwd_context_t * ctx ) */
    .shutdown_component = papicuda_shutdown_component,  /* ( void ) */
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

    CUPTI_CALL( (*cuptiEventGroupReadAllEvents) 
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
        CUPTI_CALL( (*cuptiMetricGetValue) 
                    (dev, metricId[i], arraySizeBytes, eventIdArray, 
                     aggrEventValueArraySize, aggrEventValueArray, 
                     timeDuration, &metricValue), 
                    return);

        papicuda_convert_metric_value_to_long_long(metricValue, myKinds[i], &values[i]); 
    }

    free(eventValueArray);
    free(eventIdArray);
} // end readMetricValue.


