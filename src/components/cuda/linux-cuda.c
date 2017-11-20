/**
 * @file    linux-cuda.c
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

#include <dlfcn.h>
#include <cupti.h>
#include <cuda_runtime_api.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

/* this number assumes that there will never be more events than indicated */
#define PAPICUDA_MAX_COUNTERS 512

// #define PAPICUDA_KERNEL_REPLAY_MODE

/* Contains device list, pointer to device desciption, and the list of available events */
typedef struct papicuda_context {
    int deviceCount;
    struct papicuda_device_desc *deviceArray;
    uint32_t availEventSize;
    CUpti_ActivityKind *availEventKind;
    int *availEventDeviceNum;
    uint32_t *availEventIDArray;
    uint32_t *availEventIsBeingMeasuredInEventset;
    struct papicuda_name_desc *availEventDesc;
} papicuda_context_t;

/* Store the name and description for an event */
typedef struct papicuda_name_desc {
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_2MAX_STR_LEN];
} papicuda_name_desc_t;

/* For a device, store device description */
typedef struct papicuda_device_desc {
    CUdevice cuDev;
    int deviceNum;
    char deviceName[PAPI_MIN_STR_LEN];
    uint32_t maxDomains;        /* number of domains per device */
    CUpti_EventDomainID *domainIDArray; /* Array[maxDomains] of domain IDs */
    uint32_t *domainIDNumEvents;        /* Array[maxDomains] of num of events in that domain */
} papicuda_device_desc_t;

/* Control structure tracks array of active contexts, records active events and their values */
typedef struct papicuda_control {
    uint32_t countOfActiveCUContexts;
    struct papicuda_active_cucontext_s *arrayOfActiveCUContexts[PAPICUDA_MAX_COUNTERS];
    uint32_t activeEventCount;
    int activeEventIndex[PAPICUDA_MAX_COUNTERS];
    long long activeEventValues[PAPICUDA_MAX_COUNTERS];
    uint64_t cuptiStartTimestampNs;
    uint64_t cuptiReadTimestampNs;
} papicuda_control_t;

/* For each active context, which CUDA events are being measured, context eventgroups containing events */
typedef struct papicuda_active_cucontext_s {
    CUcontext cuCtx;
    int deviceNum;
    uint32_t conMetricsCount;
    CUpti_EventID conMetrics[PAPICUDA_MAX_COUNTERS];
    CUpti_MetricValue conMetricValues[PAPICUDA_MAX_COUNTERS];
    uint32_t conEventsCount;
    CUpti_EventID conEvents[PAPICUDA_MAX_COUNTERS];
    uint64_t conEventValues[PAPICUDA_MAX_COUNTERS];
    CUpti_EventGroupSets *eventGroupPasses;
} papicuda_active_cucontext_t;

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
#define CHECK_PRINT_EVAL( checkcond, str, evalthis )                    \
    do {                                                                \
        int _cond = (checkcond);                                        \
        if (_cond) {                                                    \
            SUBDBG("error: condition %s failed: %s.\n", #checkcond, str); \
            evalthis;                                                   \
        }                                                               \
    } while (0)

#define CUDA_CALL( call, handleerror )                                \
    do {                                                                \
        cudaError_t _status = (call);                                   \
        if (_status != cudaSuccess) {                                   \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CU_CALL( call, handleerror )                                    \
    do {                                                                \
        CUresult _status = (call);                                      \
        if (_status != CUDA_SUCCESS) {                                  \
            SUBDBG("error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)


#define CUPTI_CALL(call, handleerror)                                 \
    do {                                                                \
        CUptiResult _status = (call);                                   \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            (*cuptiGetResultStringPtr)(_status, &errstr);               \
            SUBDBG("error: function %s failed with error %s.\n", #call, errstr); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                     \
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
DECLARECUPTIFUNC(cuptiEventGroupSetsDestroy, (CUpti_EventGroupSets * eventGroupSets));
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
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl1, "cuCtxSynchronize");

    dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    CHECK_PRINT_EVAL(!dl2, "CUDA runtime library libcudart.so not found.", return (PAPI_ENOSUPP));
    cudaGetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaGetDevice");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl2, "cudaFree");

    dl3 = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    CHECK_PRINT_EVAL(!dl3, "CUDA runtime library libcudart.so not found.", return (PAPI_ENOSUPP));
    /* The macro DLSYM_AND_CHECK results in the expansion example below */
    /* cuptiDeviceEnumEventDomainsPtr = dlsym( dl3, "cuptiDeviceEnumEventDomains" ); */
    /* if ( dlerror()!=NULL ) { strncpy( _cuda_vector.cmp_info.disabled_reason, "A CUDA required function was not found in dynamic libs", PAPI_MAX_STR_LEN ); return ( PAPI_ENOSUPP ); } */
    cuptiDeviceEnumMetricsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceEnumMetrics");
    cuptiDeviceGetEventDomainAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetEventDomainAttribute");
    cuptiDeviceGetNumMetricsPtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetNumMetrics");
    cuptiEventGroupGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupGetAttribute");
    cuptiEventGroupReadEventPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupReadEvent");
    cuptiEventGroupSetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetAttribute");
    cuptiEventGroupSetDisablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetDisable");
    cuptiEventGroupSetEnablePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetEnable");
    cuptiEventGroupSetsCreatePtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetsCreate");
    cuptiEventGroupSetsDestroyPtr = DLSYM_AND_CHECK(dl3, "cuptiEventGroupSetsDestroy");
    cuptiGetTimestampPtr = DLSYM_AND_CHECK(dl3, "cuptiGetTimestamp");
    cuptiMetricEnumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricEnumEvents");
    cuptiMetricGetAttributePtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetAttribute");
    cuptiMetricGetNumEventsPtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetNumEvents");
    cuptiMetricGetValuePtr = DLSYM_AND_CHECK(dl3, "cuptiMetricGetValue");
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
        /* If CUDA not initilaized, initialized CUDA and retry the device list */
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
        CU_CALL((*cuDeviceGetPtr) (&mydevice->cuDev, deviceNum), return (PAPI_EMISC));
        CU_CALL((*cuDeviceGetNamePtr) (mydevice->deviceName, PAPI_MIN_STR_LEN - 1, mydevice->cuDev), return (PAPI_EMISC));
        mydevice->deviceName[PAPI_MIN_STR_LEN - 1] = '\0';
        CUPTI_CALL((*cuptiDeviceGetNumEventDomainsPtr) (mydevice->cuDev, &mydevice->maxDomains), return (PAPI_EMISC));
        /* Allocate space to hold domain IDs */
        mydevice->domainIDArray = (CUpti_EventDomainID *) papi_calloc(mydevice->maxDomains, sizeof(CUpti_EventDomainID));
        CHECK_PRINT_EVAL(!mydevice->domainIDArray, "ERROR CUDA: Could not allocate memory for CUDA device domains", return (PAPI_ENOMEM));
        /* Put domain ids into allocated space */
        size_t domainarraysize = mydevice->maxDomains * sizeof(CUpti_EventDomainID);
        CUPTI_CALL((*cuptiDeviceEnumEventDomainsPtr) (mydevice->cuDev, &domainarraysize, mydevice->domainIDArray), return (PAPI_EMISC));
        /* Allocate space to hold domain event counts */
        mydevice->domainIDNumEvents = (uint32_t *) papi_calloc(mydevice->maxDomains, sizeof(uint32_t));
        CHECK_PRINT_EVAL(!mydevice->domainIDNumEvents, "ERROR CUDA: Could not allocate memory for domain event counts", return (PAPI_ENOMEM));
        /* For each domain, get event counts in domainNumEvents[] */
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];
            /* Get num events in domain */
            // SUBDBG( "Device %d:%d calling cuptiEventDomainGetNumEventsPtr with domainID %d \n", deviceNum, mydevice->cuDev, domainID );
            CUPTI_CALL((*cuptiEventDomainGetNumEventsPtr) (domainID, &mydevice->domainIDNumEvents[domainNum]), return (PAPI_EMISC));
            /* Keep track of overall number of events */
            maxEventSize += mydevice->domainIDNumEvents[domainNum];
        }
    }

    /* Create space for metrics */
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        uint32_t maxMetrics;
        mydevice = &gctxt->deviceArray[deviceNum];
        // CUPTI_CALL((*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics), return (PAPI_EMISC));
        if ( (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics) != CUPTI_SUCCESS )
            maxMetrics = 0;
        maxEventSize += maxMetrics;
    }

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

    /* Record the events and descriptions */
    uint32_t idxEventArray = 0;
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        mydevice = &gctxt->deviceArray[deviceNum];
        // SUBDBG( "For device %d %d maxdomains %d \n", deviceNum, mydevice->cuDev, mydevice->maxDomains );
        /* Get and store event IDs, names, descriptions into the large arrays allocated */
        for(domainNum = 0; domainNum < mydevice->maxDomains; domainNum++) {
            /* Get domain id */
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];
            uint32_t domainNumEvents = mydevice->domainIDNumEvents[domainNum];
            // SUBDBG( "For device %d domain %d domainID %d numEvents %d\n", mydevice->cuDev, domainNum, domainID, domainNumEvents );
            /* Allocate temp space for eventIDs for this domain */
            CUpti_EventID *domainEventIDArray = (CUpti_EventID *) papi_calloc(domainNumEvents, sizeof(CUpti_EventID));
            CHECK_PRINT_EVAL(!domainEventIDArray, "ERROR CUDA: Could not allocate memory for events", return (PAPI_ENOMEM));
            /* Load the domain eventIDs in temp space */
            size_t domainEventArraySize = domainNumEvents * sizeof(CUpti_EventID);
            CUPTI_CALL((*cuptiEventDomainEnumEventsPtr) (domainID, &domainEventArraySize, domainEventIDArray), return (PAPI_EMISC));
            /* For each event, get and store name and description */
            for(eventNum = 0; eventNum < domainNumEvents; eventNum++) {
                /* Record the event IDs in native event array */
                CUpti_EventID myeventCuptiEventId = domainEventIDArray[eventNum];
                gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_EVENT;
                gctxt->availEventIDArray[idxEventArray] = myeventCuptiEventId;
                gctxt->availEventDeviceNum[idxEventArray] = deviceNum;
                /* Get event name */
                tmpSizeBytes = PAPI_MIN_STR_LEN - 1 * sizeof(char);
                CUPTI_CALL((*cuptiEventGetAttributePtr) (myeventCuptiEventId, CUPTI_EVENT_ATTR_NAME, &tmpSizeBytes, tmpStr), return (PAPI_EMISC));
                /* Save a full path for the event, filling spaces with underscores */
                // snprintf( gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN, "%s:%d:%s", mydevice->deviceName, deviceNum, tmpStr );
                snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN, "event:%s:device=%d", tmpStr, deviceNum);
                gctxt->availEventDesc[idxEventArray].name[PAPI_MIN_STR_LEN - 1] = '\0';
                char *nameTmpPtr = gctxt->availEventDesc[idxEventArray].name;
                for(ii = 0; ii < (int) strlen(nameTmpPtr); ii++)
                    if(nameTmpPtr[ii] == ' ')
                        nameTmpPtr[ii] = '_';
                /* Save description in the native event array */
                tmpSizeBytes = PAPI_2MAX_STR_LEN - 1 * sizeof(char);
                CUPTI_CALL((*cuptiEventGetAttributePtr) (myeventCuptiEventId, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &tmpSizeBytes, gctxt->availEventDesc[idxEventArray].description), return (PAPI_EMISC));
                gctxt->availEventDesc[idxEventArray].description[PAPI_2MAX_STR_LEN - 1] = '\0';
                // SUBDBG( "Event ID:%d Name:%s Desc:%s\n", gctxt->availEventIDArray[idxEventArray], gctxt->availEventDesc[idxEventArray].name, gctxt->availEventDesc[idxEventArray].description );
                /* Increment index past events in this domain to start of next domain */
                idxEventArray++;
            }
            papi_free(domainEventIDArray);
        }
    }

    /* Retrieve and store metric information for each device */
    SUBDBG("Checking for metrics\n");
    for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
        uint32_t maxMetrics, i;
        CUpti_MetricID *metricIdList = NULL;
        mydevice = &gctxt->deviceArray[deviceNum];
        // CUPTI_CALL((*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics), return (PAPI_EMISC));
        if ( (*cuptiDeviceGetNumMetricsPtr) (mydevice->cuDev, &maxMetrics) != CUPTI_SUCCESS ) {
            maxMetrics = 0;
            continue;
        }
        SUBDBG("Device %d: Checking each of the (maxMetrics) %d metrics\n", deviceNum, maxMetrics);
        size_t size = maxMetrics * sizeof(CUpti_EventID);
        metricIdList = (CUpti_MetricID *) papi_calloc(maxMetrics, sizeof(CUpti_EventID));
        CHECK_PRINT_EVAL(metricIdList == NULL, "Out of memory", return (PAPI_ENOMEM));
        CUPTI_CALL((*cuptiDeviceEnumMetricsPtr) (mydevice->cuDev, &size, metricIdList), return (PAPI_EMISC));
        for(i = 0; i < maxMetrics; i++) {
            gctxt->availEventIDArray[idxEventArray] = metricIdList[i];
            gctxt->availEventKind[idxEventArray] = CUPTI_ACTIVITY_KIND_METRIC;
            gctxt->availEventDeviceNum[idxEventArray] = deviceNum;
            size = PAPI_MIN_STR_LEN;
            CUPTI_CALL((*cuptiMetricGetAttributePtr) (metricIdList[i], CUPTI_METRIC_ATTR_NAME, &size, (uint8_t *) tmpStr), return (PAPI_EMISC));
            // FIXME SOMEDAY: For this release the nvlink metrics are not functioning so skip them
            if(strstr(tmpStr, "nvlink")!=NULL)  continue;
            // FIXME SOMEDAY: For this release the nvlink metrics are not functioning so skip them
            if(size >= PAPI_MIN_STR_LEN)
                gctxt->availEventDesc[idxEventArray].name[PAPI_MIN_STR_LEN - 1] = '\0';
            snprintf(gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN, "metric:%s:device=%d", tmpStr, deviceNum);
            size = PAPI_2MAX_STR_LEN;
            CUPTI_CALL((*cuptiMetricGetAttributePtr) (metricIdList[i], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size, (uint8_t *) gctxt->availEventDesc[idxEventArray].description), return (PAPI_EMISC));
            if(size >= PAPI_2MAX_STR_LEN)
                gctxt->availEventDesc[idxEventArray].description[PAPI_2MAX_STR_LEN - 1] = '\0';
            // SUBDBG( "For device %d availEvent[%d] %s\n", mydevice->cuDev, idxEventArray, gctxt->availEventDesc[idxEventArray].name);
            idxEventArray++;
        }
        papi_free(metricIdList);
    }
    gctxt->availEventSize = idxEventArray;

    /* return 0 if everything went OK */
    return 0;
}


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
}


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
    SUBDBG("Entering with cidx: %d\n", cidx);
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
}


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
}

/* Triggered by eventset operations like add or remove.  For CUDA,
 * needs to be called multiple times from each seperate CUDA context
 * with the events to be measured from that context.  For each
 * context, create eventgroups for the events.
 */
/* Note: NativeInfo_t is defined in papi_internal.h */
static int papicuda_update_control_state(hwd_control_state_t * ctrl, NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    SUBDBG("Entering with nativeCount %d\n", nativeCount);
    (void) ctx;
    // (void) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    int currDeviceNum;
    CUcontext currCuCtx;
    int eventContextIdx;
    CUcontext eventCuCtx;
    int index, ii;
    uint32_t numEvents, ee, cc;

    /* Return if no events */
    if(nativeCount == 0)
        return (PAPI_OK);

    /* Get deviceNum, initialize context if needed via free, get context */
    // CU_CALL( (*cuCtxGetCurrentPtr)(&currCuCtx), return(PAPI_EMISC));
    CUDA_CALL((*cudaGetDevicePtr) (&currDeviceNum), return (PAPI_EMISC));
    SUBDBG("currDeviceNum %d \n", currDeviceNum);
    CUDA_CALL((*cudaFreePtr) (NULL), return (PAPI_EMISC));
    CU_CALL((*cuCtxGetCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    SUBDBG("currDeviceNum %d cuCtx %p \n", currDeviceNum, currCuCtx);

    /* Handle user request of events to be monitored */
    for(ii = 0; ii < nativeCount; ii++) {
        /* Get the PAPI event index from the user */
        index = nativeInfo[ii].ni_event;
#ifdef DEBUG
        char *eventName = gctxt->availEventDesc[index].name;
#endif
        int eventDeviceNum = gctxt->availEventDeviceNum[index];

        /* if this event is already added continue to next ii, if not, mark it as being added */
        if(gctxt->availEventIsBeingMeasuredInEventset[index] == 1) {
            SUBDBG("Skipping event %s which is already added\n", eventName);
            continue;
        } else
            gctxt->availEventIsBeingMeasuredInEventset[index] = 1;

        /* Find context/control in papicuda, creating it if does not exist */
        for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
            CHECK_PRINT_EVAL(cc >= PAPICUDA_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (PAPICUDA_MAX_COUNTERS)", return (PAPI_EMISC));
            if(gctrl->arrayOfActiveCUContexts[cc]->deviceNum == eventDeviceNum) {
                eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
                SUBDBG("Event %s device %d already has a cuCtx %p registered\n", eventName, eventDeviceNum, eventCuCtx);
                if(eventCuCtx != currCuCtx)
                    CU_CALL((*cuCtxPushCurrentPtr) (eventCuCtx), return (PAPI_EMISC));
                break;
            }
        }
        // Create context if it does not exit
        if(cc == gctrl->countOfActiveCUContexts) {
            SUBDBG("Event %s device %d does not have a cuCtx registered yet...\n", eventName, eventDeviceNum);
            if(currDeviceNum != eventDeviceNum) {
                CUDA_CALL((*cudaSetDevicePtr) (eventDeviceNum), return (PAPI_EMISC));
                CUDA_CALL((*cudaFreePtr) (NULL), return (PAPI_EMISC));
                CU_CALL((*cuCtxGetCurrentPtr) (&eventCuCtx), return (PAPI_EMISC));
            } else {
                eventCuCtx = currCuCtx;
            }
            gctrl->arrayOfActiveCUContexts[cc] = papi_calloc(1, sizeof(papicuda_active_cucontext_t));
            CHECK_PRINT_EVAL(gctrl->arrayOfActiveCUContexts[cc] == NULL, "Memory allocation for new active context failed", return (PAPI_ENOMEM));
            gctrl->arrayOfActiveCUContexts[cc]->deviceNum = eventDeviceNum;
            gctrl->arrayOfActiveCUContexts[cc]->cuCtx = eventCuCtx;
            gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses = NULL;
            gctrl->arrayOfActiveCUContexts[cc]->conMetricsCount = 0;
            gctrl->arrayOfActiveCUContexts[cc]->conEventsCount = 0;
            gctrl->countOfActiveCUContexts++;
            SUBDBG("Added a new context deviceNum %d cuCtx %p ... now countOfActiveCUContexts is %d\n", eventDeviceNum, eventCuCtx, gctrl->countOfActiveCUContexts);
        }
        eventContextIdx = cc;

        papicuda_active_cucontext_t *eventctrl = gctrl->arrayOfActiveCUContexts[eventContextIdx];
        switch (gctxt->availEventKind[index]) {
        case CUPTI_ACTIVITY_KIND_METRIC:
            SUBDBG("Need to add metric %d %s \n", index, eventName);
            /* For the metric, find list of events required */
            CUpti_MetricID metricId = gctxt->availEventIDArray[index];
            CUPTI_CALL((*cuptiMetricGetNumEventsPtr) (metricId, &numEvents), return (PAPI_EINVAL));
            size_t sizeBytes = numEvents * sizeof(CUpti_EventID);
            CUpti_EventID *eventIdArray = papi_malloc(sizeBytes);
            CHECK_PRINT_EVAL(eventIdArray == NULL, "Malloc failed", return (PAPI_ENOMEM));
            CUPTI_CALL((*cuptiMetricEnumEventsPtr) (metricId, &sizeBytes, eventIdArray), return (PAPI_EINVAL));
            SUBDBG("For metric %s, append the list of %d required events\n", eventName, numEvents);
            for(ee = 0; ee < numEvents; ee++) {
                eventctrl->conEvents[eventctrl->conEventsCount] = eventIdArray[ee];
                eventctrl->conEventsCount++;
                SUBDBG("For metric %s, appended event %d - %d %d to this context (conEventsCount %d)\n", eventName, ee, eventIdArray[ee], eventctrl->conEvents[eventctrl->conEventsCount], eventctrl->conEventsCount);
                if (eventctrl->conEventsCount >= PAPICUDA_MAX_COUNTERS) {
                    SUBDBG("Num events (generated by metric) exceeded PAPICUDA_MAX_COUNTERS\n");
                    return(PAPI_EINVAL);
                }
            }
            eventctrl->conMetrics[eventctrl->conMetricsCount] = metricId;
            eventctrl->conMetricsCount++;
            if (eventctrl->conMetricsCount >= PAPICUDA_MAX_COUNTERS) {
                SUBDBG("Num metrics exceeded PAPICUDA_MAX_COUNTERS\n");
                return(PAPI_EINVAL);
            }
            break;

        case CUPTI_ACTIVITY_KIND_EVENT:
            SUBDBG("Need to add event %d %s to the context\n", index, eventName);
            /* lookup cuptieventid for this event index */
            CUpti_EventID eventId = gctxt->availEventIDArray[index];
            eventctrl->conEvents[eventctrl->conEventsCount] = eventId;
            eventctrl->conEventsCount++;
            break;

        default:
            CHECK_PRINT_EVAL(1, "Unknown CUPTI measure", return (PAPI_EMISC));
            break;
        }
        
        if (eventctrl->conEventsCount >= PAPICUDA_MAX_COUNTERS) {
            SUBDBG("Num events exceeded PAPICUDA_MAX_COUNTERS\n");
            return(PAPI_EINVAL);
        }
        
        /* Record index of this active event back into the nativeInfo structure */
        nativeInfo[ii].ni_position = gctrl->activeEventCount;
        /* record added event at the higher level */
        CHECK_PRINT_EVAL(gctrl->activeEventCount == PAPICUDA_MAX_COUNTERS - 1, "Exceeded maximum num of events (PAPI_MAX_COUNTERS)", return (PAPI_EMISC));
        gctrl->activeEventIndex[gctrl->activeEventCount] = index;
        // gctrl->activeEventContextIdx[gctrl->activeEventCount] = eventContextIdx;
        gctrl->activeEventValues[gctrl->activeEventCount] = 0;
        gctrl->activeEventCount++;

        /* Create/recreate eventgrouppass structures for the added event and context */
        SUBDBG("Create eventGroupPasses for context (destroy pre-existing) (nativeCount %d, conEventsCount %d) \n", gctrl->activeEventCount, eventctrl->conEventsCount);
        if(eventctrl->conEventsCount > 0) {
            // SUBDBG("Destroy prevous eventGroupPasses for the context \n");
            if(eventctrl->eventGroupPasses != NULL)
                CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr) (eventctrl->eventGroupPasses), return (PAPI_EMISC));
            eventctrl->eventGroupPasses = NULL;
            size_t sizeBytes = (eventctrl->conEventsCount) * sizeof(CUpti_EventID);
            // SUBDBG("About to create eventGroupPasses for the context (sizeBytes %zu) \n", sizeBytes);
#ifdef PAPICUDA_KERNEL_REPLAY_MODE
            CUPTI_CALL((*cuptiEnableKernelReplayModePtr) (eventCuCtx), return (PAPI_ECMP));
            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr) (eventCuCtx, sizeBytes, eventctrl->conEvents, &eventctrl->eventGroupPasses), return (PAPI_ECMP));
#else
            CUPTI_CALL((*cuptiSetEventCollectionModePtr)(eventCuCtx,CUPTI_EVENT_COLLECTION_MODE_KERNEL), return(PAPI_ECMP));
            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr) (eventCuCtx, sizeBytes, eventctrl->conEvents, &eventctrl->eventGroupPasses), return (PAPI_EMISC));
            if (eventctrl->eventGroupPasses->numSets > 1) {
                SUBDBG("Error occured: The combined CUPTI events require more than 1 pass... try different events\n");
                papicuda_cleanup_eventset(ctrl);
                return(PAPI_ECOMBO);
            } else  {
                SUBDBG("Created eventGroupPasses for context total-events %d in-this-context %d passes-requied %d) \n", gctrl->activeEventCount, eventctrl->conEventsCount, eventctrl->eventGroupPasses->numSets);
            }

#endif
        }
        
        if(eventCuCtx != currCuCtx) 
            CU_CALL((*cuCtxPopCurrentPtr) (&eventCuCtx), return (PAPI_EMISC));

    }
    return (PAPI_OK);
}

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
    uint32_t ii, gg, cc, ss;
    int saveDeviceNum = -1;

    SUBDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    CUPTI_CALL((*cuptiGetTimestampPtr) (&gctrl->cuptiStartTimestampNs), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUcontext eventCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        SUBDBG("Set to device %d cuCtx %p \n", eventDeviceNum, eventCuCtx);
        // CUDA_CALL( (*cudaSetDevicePtr)(eventDeviceNum), return(PAPI_EMISC));
        if(eventDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPushCurrentPtr) (eventCuCtx), return (PAPI_EMISC));
        CUpti_EventGroupSets *eventEventGroupPasses = gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses;
        for (ss=0; ss<eventEventGroupPasses->numSets; ss++) {
            CUpti_EventGroupSet groupset = eventEventGroupPasses->sets[ss];
            for(gg = 0; gg < groupset.numEventGroups; gg++) {
                CUpti_EventGroup group = groupset.eventGroups[gg];
                uint32_t one = 1;
                CUPTI_CALL((*cuptiEventGroupSetAttributePtr) (group, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(uint32_t), &one), return (PAPI_EMISC));
            }
            CUPTI_CALL((*cuptiEventGroupSetEnablePtr) (&groupset), return (PAPI_EMISC));
        }
        if(eventDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&eventCuCtx), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* Triggered by PAPI_read().  For CUDA component, switch to each
 * context, read all the eventgroups, and put the values in the
 * correct places. */
static int papicuda_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    SUBDBG("Entering\n");
    (void) ctx;
    (void) ctrl;
    (void) flags;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    uint32_t gg, ii, jj, ee, instanceK, cc, rr, ss;
    int saveDeviceNum;
    size_t eventIdsSize = PAPICUDA_MAX_COUNTERS * sizeof(CUpti_EventID);
    uint64_t readEventValueBuffer[PAPICUDA_MAX_COUNTERS];
    CUpti_EventID readEventIDArray[PAPICUDA_MAX_COUNTERS];

    // Get read time stamp
    CUPTI_CALL((*cuptiGetTimestampPtr) (&gctrl->cuptiReadTimestampNs), return (PAPI_EMISC));
    uint64_t durationNs = gctrl->cuptiReadTimestampNs - gctrl->cuptiStartTimestampNs;
    gctrl->cuptiStartTimestampNs = gctrl->cuptiReadTimestampNs;
    
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

        size_t numEventIDsRead = 0;
        CU_CALL((*cuCtxSynchronizePtr) (), return (PAPI_EMISC));
        CUpti_EventGroupSets *currEventGroupPasses = gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses;
        uint32_t numEvents, numInstances, numTotalInstances;
        size_t sizeofuint32num = sizeof(uint32_t);
        CUpti_EventDomainID groupDomainID;
        size_t groupDomainIDSize = sizeof(groupDomainID);
        CUdevice cudevice = gctxt->deviceArray[currDeviceNum].cuDev;

        /* Since we accumulate the eventValues in a buffer, it needs to be cleared for each context */
        for(ee = 0; ee < PAPICUDA_MAX_COUNTERS; ee++)
            readEventValueBuffer[ee] = 0;

        for (ss=0; ss<currEventGroupPasses->numSets; ss++) {
            CUpti_EventGroupSet groupset = currEventGroupPasses->sets[ss]; 
            SUBDBG("Read events in this context\n");
            for(gg = 0; gg < groupset.numEventGroups; gg++) {
                CUpti_EventGroup group = groupset.eventGroups[gg];
                CUPTI_CALL((*cuptiEventGroupGetAttributePtr) (group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainIDSize, &groupDomainID), return (PAPI_EMISC));
                CUPTI_CALL((*cuptiDeviceGetEventDomainAttributePtr) (cudevice, groupDomainID, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &sizeofuint32num, &numTotalInstances), return (PAPI_EMISC));
                CUPTI_CALL((*cuptiEventGroupGetAttributePtr) (group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &sizeofuint32num, &numInstances), return (PAPI_EMISC));
                CUPTI_CALL((*cuptiEventGroupGetAttributePtr) (group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &sizeofuint32num, &numEvents), return (PAPI_EMISC));
                eventIdsSize = PAPICUDA_MAX_COUNTERS * sizeof(CUpti_EventID);
                CUpti_EventID eventIds[PAPICUDA_MAX_COUNTERS];
                CUPTI_CALL((*cuptiEventGroupGetAttributePtr) (group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds), return (PAPI_EMISC));
                SUBDBG("Context %d eventgroup %d domain numTotalInstaces %u numInstances %u numEvents %u\n", cc, gg, numTotalInstances, numInstances, numEvents);
                size_t valuesSize = sizeof(uint64_t) * numInstances;
                uint64_t *values = (uint64_t *) papi_malloc(valuesSize);
                CHECK_PRINT_EVAL(values == NULL, "Out of memory", return (PAPI_ENOMEM));
                /* For each event, read all values and normalize */
                for(ee = 0; ee < numEvents; ee++) {
                    CUPTI_CALL((*cuptiEventGroupReadEventPtr) (group, CUPTI_EVENT_READ_FLAG_NONE, eventIds[ee], &valuesSize, values), return (PAPI_EMISC));
                    // sum collect event values from all instances
                    uint64_t valuesum = 0;
                    for(instanceK = 0; instanceK < numInstances; instanceK++)
                        valuesum += values[instanceK];
                    // It seems that the same event can occur multiple times in eventIds, so we need to accumulate values in older valueBuffers if needed 
                    // Scan thru readEvents looking for a match, break if found, if not found, increment numEventIDsRead
                    for(rr = 0; rr < numEventIDsRead; rr++)
                        if(readEventIDArray[rr] == eventIds[ee])
                            break;
                    /* If the event was not found, increment the numEventIDsRead */
                    if(rr == numEventIDsRead)
                    numEventIDsRead++;
                    readEventIDArray[rr] = eventIds[ee];
                    readEventValueBuffer[rr] += valuesum;
                    size_t tmpStrSize = PAPI_MIN_STR_LEN - 1 * sizeof(char);
                    char tmpStr[PAPI_MIN_STR_LEN];
                    CUPTI_CALL((*cuptiEventGetAttributePtr) (eventIds[ee], CUPTI_EVENT_ATTR_NAME, &tmpStrSize, tmpStr), return (PAPI_EMISC));
                    SUBDBG("Read context %d eventgroup %d numEventIDsRead %lu device %d event %d/%d %d name %s value %lu (rr %d id %d val %lu) \n", cc, gg, numEventIDsRead, currDeviceNum, ee, numEvents, eventIds[ee], tmpStr, valuesum, rr,
                           eventIds[rr], readEventValueBuffer[rr]);
                }
                papi_free(values);
            }
        }

        // normalize the event values to represent the total number of domain instances on the device
        for(ii = 0; ii < numEventIDsRead; ii++) 
            readEventValueBuffer[numEventIDsRead] = (readEventValueBuffer[numEventIDsRead] * numTotalInstances) / numInstances;

        /* For this pushed device and context, figure out the event and metric values and record them into the arrays */
        SUBDBG("For this device and context, match read values against active events by scanning activeEvents array and matching associated availEventIDs\n");
        for(jj = 0; jj < gctrl->activeEventCount; jj++) {
            int index = gctrl->activeEventIndex[jj];
            /* If the device/context does not match the current context, move to next */
            if(gctxt->availEventDeviceNum[index] != currDeviceNum)
                continue;
            uint32_t eventId = gctxt->availEventIDArray[index];
            switch (gctxt->availEventKind[index]) {
            case CUPTI_ACTIVITY_KIND_EVENT:
                SUBDBG("Searching for activeEvent %s eventId %u\n", gctxt->availEventDesc[index].name, eventId);
                for(ii = 0; ii < numEventIDsRead; ii++) {
                    SUBDBG("Look at readEventIDArray[%u/%zu] with id %u\n", ii, numEventIDsRead, readEventIDArray[ii]);
                    if(readEventIDArray[ii] == eventId) {
                        gctrl->activeEventValues[jj] += (long long) readEventValueBuffer[ii];
                        SUBDBG("Matched read-eventID %d:%d eventName %s value %ld activeEvent %d value %lld \n", jj, (int) eventId, gctxt->availEventDesc[index].name, readEventValueBuffer[ii], index, gctrl->activeEventValues[jj]);
                        break;
                    }
                }
                break;

            case CUPTI_ACTIVITY_KIND_METRIC:
                SUBDBG("For the metric, find list of events required to calculate this metric value\n");
                CUpti_MetricID metricId = gctxt->availEventIDArray[index];
                int metricDeviceNum = gctxt->availEventDeviceNum[index];
                CUdevice cudevice = gctxt->deviceArray[metricDeviceNum].cuDev;
                uint32_t numEvents, ee;
                CUPTI_CALL((*cuptiMetricGetNumEventsPtr) (metricId, &numEvents), return (PAPI_EINVAL));
                SUBDBG("Metric %s needs %d events\n", gctxt->availEventDesc[index].name, numEvents);
                size_t eventIdArraySizeBytes = numEvents * sizeof(CUpti_EventID);
                CUpti_EventID *eventIdArray = papi_malloc(eventIdArraySizeBytes);
                CHECK_PRINT_EVAL(eventIdArray == NULL, "Malloc failed", return (PAPI_ENOMEM));
                size_t eventValueArraySizeBytes = numEvents * sizeof(uint64_t);
                uint64_t *eventValueArray = papi_malloc(eventValueArraySizeBytes);
                CHECK_PRINT_EVAL(eventValueArray == NULL, "Malloc failed", return (PAPI_ENOMEM));
                CUPTI_CALL((*cuptiMetricEnumEventsPtr) (metricId, &eventIdArraySizeBytes, eventIdArray), return (PAPI_EINVAL));
                // Match metrics for the users events
                for(ee = 0; ee < numEvents; ee++) {
                    for(ii = 0; ii < numEventIDsRead; ii++) {
                        if(eventIdArray[ee] == readEventIDArray[ii]) {
                            SUBDBG("Matched metric %s, found %d/%d events with eventId %d\n", gctxt->availEventDesc[index].name, ee, numEvents, readEventIDArray[ii]);
                            eventValueArray[ee] = readEventValueBuffer[ii];
                            break;
                        }
                    }
                    CHECK_PRINT_EVAL(ii == numEventIDsRead, "Could not find required event for metric", return (PAPI_EINVAL));
                }

                // Use CUPTI to calculate a metric.  Return all metric values mapped into long long values.
                CUpti_MetricValue metricValue;
                CUpti_MetricValueKind valueKind;
                size_t valueKindSize = sizeof(valueKind);
                CUPTI_CALL((*cuptiMetricGetAttributePtr) (metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind), return (PAPI_EMISC));
                CUPTI_CALL((*cuptiMetricGetValuePtr) (cudevice, metricId, eventIdArraySizeBytes, eventIdArray, eventValueArraySizeBytes, eventValueArray, durationNs, &metricValue), return (PAPI_EMISC));
                int retval = papicuda_convert_metric_value_to_long_long(metricValue, valueKind, &(gctrl->activeEventValues[jj]));
                if(retval != PAPI_OK)
                    return (retval);
                papi_free(eventIdArray);
                papi_free(eventValueArray);
                break;

            default:
                SUBDBG("Not handled");
                break;
            }
        }

        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    }
    *values = gctrl->activeEventValues;
    return (PAPI_OK);
}

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
        CUpti_EventGroupSets *currEventGroupPasses = gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses;
        for (ss=0; ss<currEventGroupPasses->numSets; ss++) {
            CUpti_EventGroupSet groupset = currEventGroupPasses->sets[ss]; 
            CUPTI_CALL((*cuptiEventGroupSetDisablePtr) (&groupset), return (PAPI_EMISC));
        }
        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));

    }
    return (PAPI_OK);
}


/* 
 * Disable and destroy the CUDA eventGroup
 */
static int papicuda_cleanup_eventset(hwd_control_state_t * ctrl)
{
    SUBDBG("Entering\n");
    (void) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    // papicuda_active_cucontext_t *currctrl;
    uint32_t cc;
    int saveDeviceNum;

    SUBDBG("Save current context, then switch to each active device/context and enable eventgroups\n");
    CUDA_CALL((*cudaGetDevicePtr) (&saveDeviceNum), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveCUContexts; cc++) {
        CUcontext currCuCtx = gctrl->arrayOfActiveCUContexts[cc]->cuCtx;
        int currDeviceNum = gctrl->arrayOfActiveCUContexts[cc]->deviceNum;
        CUpti_EventGroupSets *currEventGroupPasses = gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses;
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPushCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        else
            CU_CALL((*cuCtxSetCurrentPtr) (currCuCtx), return (PAPI_EMISC));
        //CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr) (currEventGroupPasses), return (PAPI_EMISC));
        (*cuptiEventGroupSetsDestroyPtr) (currEventGroupPasses);
        gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses = NULL;
        papi_free( gctrl->arrayOfActiveCUContexts[cc] );
        /* Pop the pushed context */
        if(currDeviceNum != saveDeviceNum)
            CU_CALL((*cuCtxPopCurrentPtr) (&currCuCtx), return (PAPI_EMISC));
    }
    /* Record that there are no active contexts or events */
    gctrl->countOfActiveCUContexts = 0;
    gctrl->activeEventCount = 0;
    return (PAPI_OK);
}


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
    uint32_t cc;
    /* Free context */
    if(gctxt) {
        for(deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++) {
            papicuda_device_desc_t *mydevice = &gctxt->deviceArray[deviceNum];
            papi_free(mydevice->domainIDArray);
            papi_free(mydevice->domainIDNumEvents);
        }
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
}


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
        CUpti_EventGroupSets *currEventGroupPasses = gctrl->arrayOfActiveCUContexts[cc]->eventGroupPasses;
        for (ss=0; ss<currEventGroupPasses->numSets; ss++) {
            CUpti_EventGroupSet groupset = currEventGroupPasses->sets[ss]; 
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
}


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
