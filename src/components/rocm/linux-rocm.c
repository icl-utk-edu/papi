/**
 * @file    linux-rocm.c
 *
 * @ingroup rocm_components
 *
 * @brief This implements a PAPI component that enables PAPI-C to
 *  access hardware monitoring counters for AMD ROCM GPU devices
 *  through the ROC-profiler library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <dlfcn.h>
#include <hsa.h>
#include <rocprofiler.h>
#include <string.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

/* this number assumes that there will never be more events than indicated */
#define PAPIROCM_MAX_COUNTERS 512

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                     \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#if 0
#define ROCMDBG(format, args...) do { fprintf(stdout, format, ## args); fflush(stdout); } while(0)
#else
//#define ROCMDBG(format, args...) do {} while(0)
#define ROCMDBG SUBDBG
#endif

/* Macros for error checking... each arg is only referenced/evaluated once */
#define CHECK_PRINT_EVAL(checkcond, str, evalthis)                      \
    do {                                                                \
        int _cond = (checkcond);                                        \
        if (_cond) {                                                    \
            fprintf(stderr, "error: condition %s failed: %s.\n", #checkcond, str); \
            evalthis;                                                   \
        }                                                               \
    } while (0)

#define ROCM_CALL_CK(call, args, handleerror)                           \
    do {                                                                \
        hsa_status_t _status = (*call##Ptr)args;                        \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {                                   \
            fprintf(stderr, "error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define DLSYM_AND_CHECK(dllib, name)                                    \
    do {                                                                \
        name##Ptr = dlsym(dllib, #name);                                            \
        if (dlerror()!=NULL) {                                          \
              strncpy(_rocm_vector.cmp_info.disabled_reason,            \
                  "A ROCM required function was not found in dynamic libs", \
                  PAPI_MAX_STR_LEN);                                        \
          return ( PAPI_ENOSUPP );                                      \
        }                                                               \
    } while (0)

typedef rocprofiler_t* Context;
typedef rocprofiler_feature_t EventID;

/* Contains device list, pointer to device desciption, and the list of available events */
typedef struct papirocm_context {
    uint32_t availAgentSize;
    hsa_agent_t* availAgentArray;
    uint32_t availEventSize;
    int *availEventDeviceNum;
    EventID *availEventIDArray;
    uint32_t *availEventIsBeingMeasuredInEventset;
    struct ev_name_desc *availEventDesc;
} papirocm_context_t;

/* Store the name and description for an event */
typedef struct ev_name_desc {
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_2MAX_STR_LEN];
} ev_name_desc_t;

/* Control structure tracks array of active contexts, records active events and their values */
typedef struct papirocm_control {
    uint32_t countOfActiveContexts;
    struct papirocm_active_context_s *arrayOfActiveContexts[PAPIROCM_MAX_COUNTERS];
    uint32_t activeEventCount;
    int activeEventIndex[PAPIROCM_MAX_COUNTERS];
    long long activeEventValues[PAPIROCM_MAX_COUNTERS];
    uint64_t startTimestampNs;
    uint64_t readTimestampNs;
} papirocm_control_t;

/* For each active context, which ROCM events are being measured, context eventgroups containing events */
typedef struct papirocm_active_context_s {
    Context ctx;
    int deviceNum;
    uint32_t conEventsCount;
    EventID conEvents[PAPIROCM_MAX_COUNTERS];
    int conEventIndex[PAPIROCM_MAX_COUNTERS];
} papirocm_active_context_t;

/* Function prototypes */
static int papirocm_cleanup_eventset(hwd_control_state_t * ctrl);

/* ******  CHANGE PROTOTYPES TO DECLARE ROCM LIBRARY SYMBOLS AS WEAK  **********
 *  This is done so that a version of PAPI built with the rocm component can   *
 *  be installed on a system which does not have the rocm libraries installed. *
 *                                                                             *
 *  If this is done without these prototypes, then all papi services on the    *
 *  system without the rocm libraries installed will fail.  The PAPI libraries *
 *  contain references to the rocm libraries which are not installed.  The     *
 *  load of PAPI commands fails because the rocm library references can not be *
 *  resolved.                                                                  *
 *                                                                             *
 *  This also defines pointers to the rocm library functions that we call.     *
 *  These function pointers will be resolved with dlopen/dlsym calls at        *
 *  component initialization time.  The component then calls the rocm library  *
 *  functions through these function pointers.                                 *
 *******************************************************************************/
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

#define DECLAREROCMFUNC(funcname, funcsig) \
    hsa_status_t __attribute__((weak)) funcname funcsig; \
    hsa_status_t(*funcname##Ptr) funcsig;

// ROCR API declaration
DECLAREROCMFUNC(hsa_init, ());
DECLAREROCMFUNC(hsa_shut_down, ());
DECLAREROCMFUNC(hsa_iterate_agents, (hsa_status_t (*)(hsa_agent_t, void*),
                                     void*));
DECLAREROCMFUNC(hsa_system_get_info, (hsa_system_info_t, void*));
DECLAREROCMFUNC(hsa_agent_get_info, (hsa_agent_t agent, hsa_agent_info_t attribute, void* value));

// ROC-profiler API declaration
DECLAREROCMFUNC(rocprofiler_get_info, (const hsa_agent_t*, rocprofiler_info_kind_t, void *));
DECLAREROCMFUNC(rocprofiler_iterate_info, (const hsa_agent_t*,
                                          rocprofiler_info_kind_t,
                                          hsa_status_t (*)(const rocprofiler_info_data_t, void *), void *));
DECLAREROCMFUNC(rocprofiler_open, (hsa_agent_t agent,                       // GPU handle
                                   rocprofiler_feature_t* features,         // [in] profiling features array
                                   uint32_t feature_count,                  // profiling info count
                                   rocprofiler_t** context,                 // [out] context object
                                   uint32_t mode,                           // profiling mode mask
                                   rocprofiler_properties_t* properties));  // profiling properties
DECLAREROCMFUNC(rocprofiler_close, (rocprofiler_t*));
DECLAREROCMFUNC(rocprofiler_group_count, (const rocprofiler_t*, uint32_t*));
DECLAREROCMFUNC(rocprofiler_start, (rocprofiler_t*, uint32_t));
DECLAREROCMFUNC(rocprofiler_read, (rocprofiler_t*, uint32_t));
DECLAREROCMFUNC(rocprofiler_stop, (rocprofiler_t*, uint32_t));
DECLAREROCMFUNC(rocprofiler_get_data, (rocprofiler_t*, uint32_t));
DECLAREROCMFUNC(rocprofiler_get_metrics, (const rocprofiler_t*));
DECLAREROCMFUNC(rocprofiler_reset, (rocprofiler_t*, uint32_t));

// file handles used to access rocm libraries with dlopen
static void *dl1 = NULL;
static void *dl2 = NULL;

/* The PAPI side (external) variable as a global */
papi_vector_t _rocm_vector;

/* Global variable for hardware description, event and metric lists */
static papirocm_context_t *global_papirocm_context = NULL;

/* This global variable points to the head of the control state list */
static papirocm_control_t *global_papirocm_control = NULL;


/*****************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/

/* 
 * Link the necessary ROCM libraries to use the rocm component.  If any of them can not be found, then
 * the ROCM component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the ROCM component can be installed and used on systems which have the ROCM libraries installed
 * and on systems where these libraries are not installed.
 */
static int papirocm_linkRocmLibraries()
{
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        strncpy(_rocm_vector.cmp_info.disabled_reason, "The ROCM component does not support statically linking to libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }
    /* Need to link in the ROCm libraries, if not found disable the component */
    dl1 = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    CHECK_PRINT_EVAL(!dl1, "ROC runtime library libhsa-runtime64.so not found.", return (PAPI_ENOSUPP));
    DLSYM_AND_CHECK(dl1, hsa_init);
    DLSYM_AND_CHECK(dl1, hsa_iterate_agents);
    DLSYM_AND_CHECK(dl1, hsa_system_get_info);
    DLSYM_AND_CHECK(dl1, hsa_agent_get_info);
    dl2 = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    CHECK_PRINT_EVAL(!dl2, "ROCm profiling library librocprofiler64.so not found.", return (PAPI_ENOSUPP));
    DLSYM_AND_CHECK(dl2, rocprofiler_get_info);
    DLSYM_AND_CHECK(dl2, rocprofiler_iterate_info);
    DLSYM_AND_CHECK(dl2, rocprofiler_open);
    DLSYM_AND_CHECK(dl2, rocprofiler_close);
    DLSYM_AND_CHECK(dl2, rocprofiler_group_count);
    DLSYM_AND_CHECK(dl2, rocprofiler_start);
    DLSYM_AND_CHECK(dl2, rocprofiler_read);
    DLSYM_AND_CHECK(dl2, rocprofiler_stop);
    DLSYM_AND_CHECK(dl2, rocprofiler_get_data);
    DLSYM_AND_CHECK(dl2, rocprofiler_get_metrics);
    DLSYM_AND_CHECK(dl2, rocprofiler_reset);

    return (PAPI_OK);
}

// Callback function to get the number of agents
static hsa_status_t papirocm_get_gpu_handle(hsa_agent_t agent, void* arg) {
  papirocm_context_t * gctxt = (papirocm_context_t*) arg;

  hsa_device_type_t type;
  ROCM_CALL_CK(hsa_agent_get_info,(agent, HSA_AGENT_INFO_DEVICE, &type), return (PAPI_EMISC));

  // Device is a GPU agent
  if (type == HSA_DEVICE_TYPE_GPU) {
    gctxt->availAgentSize += 1;
    gctxt->availAgentArray = (hsa_agent_t*) papi_realloc(gctxt->availAgentArray, gctxt->availAgentSize);
    gctxt->availAgentArray[gctxt->availAgentSize - 1] = agent;
  }

  return HSA_STATUS_SUCCESS;
}

typedef struct {
    int device_num;
    papirocm_context_t * ctx;
} events_callback_arg_t;

// Callback function to get the number of metrics
static hsa_status_t papirocm_add_native_events_callback(const rocprofiler_info_data_t info, void * arg) {
    events_callback_arg_t * callback_arg = (events_callback_arg_t*) arg;
    papirocm_context_t * ctx = callback_arg->ctx;
    const int eventDeviceNum = callback_arg->device_num;
    const uint32_t index = ctx->availEventSize;

    snprintf(ctx->availEventDesc[index].name, PAPI_MAX_STR_LEN, "device:%d:%s", eventDeviceNum, info.metric.name);
    ctx->availEventDesc[index].name[PAPI_MAX_STR_LEN - 1] = '\0';
    strncpy(ctx->availEventDesc[index].description, info.metric.description, PAPI_2MAX_STR_LEN);
    ctx->availEventDesc[index].description[PAPI_2MAX_STR_LEN - 1] = '\0';

    EventID eventId;            // Removed ={} initialization.
    eventId.parameters=NULL;    // Init.
    eventId.parameter_count=0;
	
    eventId.kind = ROCPROFILER_FEATURE_KIND_METRIC;
    eventId.name = strdup(info.metric.name);

    ctx->availEventDeviceNum[index] = eventDeviceNum;
    ctx->availEventIDArray[index] = eventId;
    ctx->availEventSize = index + 1;

    return HSA_STATUS_SUCCESS;
}

static int papirocm_add_native_events(papirocm_context_t * ctx)
{
    uint32_t i, maxEventSize = 0;

    for (i = 0; i < ctx->availAgentSize; ++i) {
      uint32_t size = 0;
      ROCM_CALL_CK(rocprofiler_get_info, (&(ctx->availAgentArray[i]), ROCPROFILER_INFO_KIND_METRIC_COUNT, &size), return (PAPI_EMISC));
      maxEventSize += size;
    } 
  
    /* Allocate space for all events and descriptors */
    ctx->availEventDeviceNum = (int *) papi_calloc(maxEventSize, sizeof(int));
    CHECK_PRINT_EVAL(!ctx->availEventDeviceNum, "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventIDArray = (EventID *) papi_calloc(maxEventSize, sizeof(EventID));
    CHECK_PRINT_EVAL(!ctx->availEventIDArray, "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventIsBeingMeasuredInEventset = (uint32_t *) papi_calloc(maxEventSize, sizeof(uint32_t));
    CHECK_PRINT_EVAL(!ctx->availEventIsBeingMeasuredInEventset, "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventDesc = (ev_name_desc_t *) papi_calloc(maxEventSize, sizeof(ev_name_desc_t));
    CHECK_PRINT_EVAL(!ctx->availEventDesc, "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));

    for (i = 0; i < ctx->availAgentSize; ++i) {
      events_callback_arg_t arg;
      arg.device_num = i;
      arg.ctx = ctx;
      ROCM_CALL_CK(rocprofiler_iterate_info, (&(ctx->availAgentArray[i]), ROCPROFILER_INFO_KIND_METRIC, papirocm_add_native_events_callback, (void*)(&arg)), return (PAPI_EMISC));
    }

    /* return 0 if everything went OK */
    return 0;
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/* 
 * This is called whenever a thread is initialized.
 */
static int papirocm_init_thread(hwd_context_t * ctx)
{
    ROCMDBG("Entering papirocm_init_thread\n");

    (void) ctx;
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int papirocm_init_component(int cidx)
{
    ROCMDBG("Entering papirocm_init_component\n");

    /* link in all the rocm libraries and resolve the symbols we need to use */
    if(papirocm_linkRocmLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of ROCM libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    ROCM_CALL_CK(hsa_init, (), return (PAPI_EMISC));

    /* Create the structure */
    if(!global_papirocm_context)
        global_papirocm_context = (papirocm_context_t *) papi_calloc(1, sizeof(papirocm_context_t));

    /* Get GPU agent */
    ROCM_CALL_CK(hsa_iterate_agents, (papirocm_get_gpu_handle, global_papirocm_context), return (PAPI_EMISC));

    int rv;

    /* Get list of all native ROCM events supported */
    rv = papirocm_add_native_events(global_papirocm_context);
    if(rv != 0)
        return (rv);

    /* Export some information */
    _rocm_vector.cmp_info.CmpIdx = cidx;
    _rocm_vector.cmp_info.num_native_events = global_papirocm_context->availEventSize;
    _rocm_vector.cmp_info.num_cntrs = _rocm_vector.cmp_info.num_native_events;
    _rocm_vector.cmp_info.num_mpx_cntrs = _rocm_vector.cmp_info.num_native_events;

    return (PAPI_OK);
}


/* Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int papirocm_init_control_state(hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering papirocm_init_control_state\n");

    (void) ctrl;
    papirocm_context_t *gctxt = global_papirocm_context;

    CHECK_PRINT_EVAL(!gctxt, "Error: The PAPI ROCM component needs to be initialized first", return (PAPI_ENOINIT));
    /* If no events were found during the initial component initialization, return error */
    if(global_papirocm_context->availEventSize <= 0) {
        strncpy(_rocm_vector.cmp_info.disabled_reason, "ERROR ROCM: No events exist", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }
    /* If it does not exist, create the global structure to hold ROCM contexts and active events */
    if(!global_papirocm_control) {
        global_papirocm_control = (papirocm_control_t *) papi_calloc(1, sizeof(papirocm_control_t));
        global_papirocm_control->countOfActiveContexts = 0;
        global_papirocm_control->activeEventCount = 0;
    }
    return PAPI_OK;
}


/* Triggered by eventset operations like add or remove.  For ROCM,
 * needs to be called multiple times from each seperate ROCM context
 * with the events to be measured from that context.  For each
 * context, create eventgroups for the events.
 */
/* Note: NativeInfo_t is defined in papi_internal.h */
static int papirocm_update_control_state(hwd_control_state_t * ctrl, NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    ROCMDBG("Entering papirocm_update_control_state with nativeCount %d\n", nativeCount);

    (void) ctx;
    papirocm_control_t *gctrl = global_papirocm_control;
    papirocm_context_t *gctxt = global_papirocm_context;
    int eventContextIdx = 0;
    int index, ii;
    uint32_t cc;
    uint32_t numPasses = 1;

    /* Return if no events */
    if(nativeCount == 0)
        return (PAPI_OK);

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
            ROCMDBG("Skipping event %s which is already added\n", eventName);
            continue;
        } else
            gctxt->availEventIsBeingMeasuredInEventset[index] = 1;

        /* Find context/control in papirocm, creating it if does not exist */
        for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
            CHECK_PRINT_EVAL(cc >= PAPIROCM_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (PAPIROCM_MAX_COUNTERS)", return (PAPI_EMISC));
            if(gctrl->arrayOfActiveContexts[cc]->deviceNum == eventDeviceNum) {
                break;
            }
        }
        // Create context if it does not exit
        if(cc == gctrl->countOfActiveContexts) {
            ROCMDBG("Event %s device %d does not have a ctx registered yet...\n", eventName, eventDeviceNum);
            gctrl->arrayOfActiveContexts[cc] = papi_calloc(1, sizeof(papirocm_active_context_t));
            CHECK_PRINT_EVAL(gctrl->arrayOfActiveContexts[cc] == NULL, "Memory allocation for new active context failed", return (PAPI_ENOMEM));
            gctrl->arrayOfActiveContexts[cc]->deviceNum = eventDeviceNum;
            gctrl->arrayOfActiveContexts[cc]->ctx = NULL;
            gctrl->arrayOfActiveContexts[cc]->conEventsCount = 0;
            gctrl->countOfActiveContexts++;
            ROCMDBG("Added a new context deviceNum %d ... now countOfActiveContexts is %d\n", eventDeviceNum, gctrl->countOfActiveContexts);
        }
        eventContextIdx = cc;

        papirocm_active_context_t *eventctrl = gctrl->arrayOfActiveContexts[eventContextIdx];
        ROCMDBG("Need to add event %d %s to the context\n", index, eventName);
        /* lookup eventid for this event index */
        EventID eventId = gctxt->availEventIDArray[index];
        eventctrl->conEvents[eventctrl->conEventsCount] = eventId;
        eventctrl->conEventIndex[eventctrl->conEventsCount] = index;
        eventctrl->conEventsCount++;

        if (eventctrl->conEventsCount >= PAPIROCM_MAX_COUNTERS) {
            ROCMDBG("Num events exceeded PAPIROCM_MAX_COUNTERS\n");
            return(PAPI_EINVAL);
        }
        
        /* Record index of this active event back into the nativeInfo structure */
        nativeInfo[ii].ni_position = gctrl->activeEventCount;
        /* record added event at the higher level */
        CHECK_PRINT_EVAL(gctrl->activeEventCount == PAPIROCM_MAX_COUNTERS - 1, "Exceeded maximum num of events (PAPI_MAX_COUNTERS)", return (PAPI_EMISC));
        gctrl->activeEventIndex[gctrl->activeEventCount] = index;
        gctrl->activeEventValues[gctrl->activeEventCount] = 0;
        gctrl->activeEventCount++;

        /* Create/recreate eventgrouppass structures for the added event and context */
        ROCMDBG("Create eventGroupPasses for context (destroy pre-existing) (nativeCount %d, conEventsCount %d) \n", gctrl->activeEventCount, eventctrl->conEventsCount);
        if(eventctrl->conEventsCount > 0) {
            if (eventctrl->ctx != NULL) {
                ROCM_CALL_CK(rocprofiler_close, (eventctrl->ctx), return (PAPI_EMISC));
            }
            rocprofiler_properties_t properties;
            memset(&properties, 0, sizeof(properties));
            properties.queue_depth = 128;
            ROCM_CALL_CK(rocprofiler_open, (gctxt->availAgentArray[eventDeviceNum], eventctrl->conEvents, eventctrl->conEventsCount, &(eventctrl->ctx),
                                          ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE, &properties), return (PAPI_EBUG));
            ROCM_CALL_CK(rocprofiler_group_count, (eventctrl->ctx, &numPasses), return (PAPI_EMISC));

            if (numPasses > 1) {
                ROCMDBG("Error occured: The combined ROCM events require more than 1 pass... try different events\n");
                papirocm_cleanup_eventset(ctrl);
                return(PAPI_ECOMBO);
            } else  {
                ROCMDBG("Created eventGroupPasses for context total-events %d in-this-context %d passes-requied %d) \n", gctrl->activeEventCount, eventctrl->conEventsCount, numPasses);
            }
        }
    }
    return (PAPI_OK);
}


/* Triggered by PAPI_start().
 * For ROCM component, switch to each context and start all eventgroups.
 */
static int papirocm_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering papirocm_start\n");

    (void) ctx;
    (void) ctrl;
    papirocm_control_t *gctrl = global_papirocm_control;
    uint32_t ii, cc;

    ROCMDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->startTimestampNs), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Start device %d ctx %p ts %lu\n", eventDeviceNum, eventCtx, gctrl->startTimestampNs);
        if (eventCtx == NULL) abort();
        ROCM_CALL_CK(rocprofiler_start, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* Triggered by PAPI_read().  For ROCM component, switch to each
 * context, read all the eventgroups, and put the values in the
 * correct places. */
static int papirocm_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    ROCMDBG("Entering papirocm_read\n");

    (void) ctx;
    (void) ctrl;
    (void) flags;
    papirocm_control_t *gctrl = global_papirocm_control;
    papirocm_context_t *gctxt = global_papirocm_context;
    uint32_t cc, jj, ee;

    // Get read time stamp
    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->readTimestampNs), return (PAPI_EMISC));
    uint64_t durationNs = gctrl->readTimestampNs - gctrl->startTimestampNs;
    gctrl->startTimestampNs = gctrl->readTimestampNs;


    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Read device %d ctx %p(%u) ts %lu\n", eventDeviceNum, eventCtx, cc, gctrl->readTimestampNs);
        ROCM_CALL_CK(rocprofiler_read, (eventCtx, 0), return (PAPI_EMISC));
        ROCMDBG("waiting for data\n");
        ROCM_CALL_CK(rocprofiler_get_data, (eventCtx, 0), return (PAPI_EMISC));
        ROCM_CALL_CK(rocprofiler_get_metrics, (eventCtx), return (PAPI_EMISC));
        ROCMDBG("done\n");

        for(jj = 0; jj < gctrl->activeEventCount; jj++) {
            int index = gctrl->activeEventIndex[jj];
            EventID eventId = gctxt->availEventIDArray[index];

            /* If the device/context does not match the current context, move to next */
            if(gctxt->availEventDeviceNum[index] != eventDeviceNum)
                continue;

            for(ee = 0; ee < gctrl->arrayOfActiveContexts[cc]->conEventsCount; ee++) {
                ROCMDBG("Reading for activeEvent %s(%u) eventId %d duration %lu\n", eventId.name, ee, index, durationNs);
                if (gctrl->arrayOfActiveContexts[cc]->conEventIndex[ee] == index) {
                  gctrl->activeEventValues[jj] = gctrl->arrayOfActiveContexts[cc]->conEvents[ee].data.result_int64;
                  ROCMDBG("Matching event %d:%d eventName %s value %lld\n", jj, index, eventId.name, gctrl->activeEventValues[jj]);
                  break;
                }
            }
        }
    }

    *values = gctrl->activeEventValues;
    return (PAPI_OK);
}


/* Triggered by PAPI_stop() */
static int papirocm_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering papirocm_stop\n");

    (void) ctx;
    (void) ctrl;
    papirocm_control_t *gctrl = global_papirocm_control;
    uint32_t cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Stop device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCM_CALL_CK(rocprofiler_stop, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* 
 * Disable and destroy the ROCM eventGroup
 */
static int papirocm_cleanup_eventset(hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering papirocm_cleanup_eventset\n");

    (void) ctrl;
    papirocm_control_t *gctrl = global_papirocm_control;
    uint32_t cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Destroy device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCM_CALL_CK(rocprofiler_close, (eventCtx), return (PAPI_EMISC));
        papi_free( gctrl->arrayOfActiveContexts[cc] );
    }
    /* Record that there are no active contexts or events */
    gctrl->countOfActiveContexts = 0;
    gctrl->activeEventCount = 0;
    return (PAPI_OK);
}


/* Called at thread shutdown. Does nothing in the ROCM component. */
static int papirocm_shutdown_thread(hwd_context_t * ctx)
{
    ROCMDBG("Entering papirocm_shutdown_thread\n");

    (void) ctx;

    return (PAPI_OK);
}


/* Triggered by PAPI_shutdown() and frees memory allocated in the ROCM component. */
static int papirocm_shutdown_component(void)
{
    ROCMDBG("Entering papirocm_shutdown_component\n");

    papirocm_control_t *gctrl = global_papirocm_control;
    papirocm_context_t *gctxt = global_papirocm_context;
    uint32_t cc;

    /* Free context */
    if(gctxt) {
        papi_free(gctxt->availEventIDArray);
        papi_free(gctxt->availEventDeviceNum);
        papi_free(gctxt->availEventIsBeingMeasuredInEventset);
        papi_free(gctxt->availEventDesc);
        papi_free(gctxt);
        global_papirocm_context = gctxt = NULL;
    }

    /* Free control */
    if(gctrl) {
        for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
            if(gctrl->arrayOfActiveContexts[cc] != NULL)
                papi_free(gctrl->arrayOfActiveContexts[cc]);
        }
        papi_free(gctrl);
        global_papirocm_control = gctrl = NULL;
    }

    // Shutdown ROC runtime
    ROCM_CALL_CK(hsa_shut_down, (), return (PAPI_EMISC));

    // close the dynamic libraries needed by this component (opened in the init substrate call)
    dlclose(dl1);
    dlclose(dl2);
    return (PAPI_OK);
}


/* Triggered by PAPI_reset() but only if the EventSet is currently
 *  running. If the eventset is not currently running, then the saved
 *  value in the EventSet is set to zero without calling this
 *  routine.  */
static int papirocm_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering papirocm_reset\n");

    (void) ctx;
    (void) ctrl;
    papirocm_control_t *gctrl = global_papirocm_control;
    uint32_t ii, cc;

    ROCMDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Reset device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCM_CALL_CK(rocprofiler_reset, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* This function sets various options in the component - Does nothing in the ROCM component.
    @param[in] ctx -- hardware context
    @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
    @param[in] option -- options to be set
*/
static int papirocm_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    ROCMDBG("Entering papirocm_ctrl\n");

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
static int papirocm_set_domain(hwd_control_state_t * ctrl, int domain)
{
    ROCMDBG("Entering papirocm_set_domain\n");

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
static int papirocm_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    //ROCMDBG("Entering (get next event after %u)\n", *EventCode );

    switch (modifier) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return (PAPI_OK);
        break;
    case PAPI_ENUM_EVENTS:
        if(*EventCode < global_papirocm_context->availEventSize - 1) {
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
static int papirocm_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    //ROCMDBG("Entering EventCode %d\n", EventCode );

    unsigned int index = EventCode;
    papirocm_context_t *gctxt = global_papirocm_context;
    if(index < gctxt->availEventSize) {
        strncpy(name, gctxt->availEventDesc[index].name, len);
    } else {
        return (PAPI_EINVAL);
    }
    //ROCMDBG( "Exit: EventCode %d: Name %s\n", EventCode, name );
    return (PAPI_OK);
}


/* Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int papirocm_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
    //ROCMDBG("Entering papirocm_ntv_code_to_descr\n");

    unsigned int index = EventCode;
    papirocm_context_t *gctxt = global_papirocm_context;
    if(index < gctxt->availEventSize) {
        strncpy(name, gctxt->availEventDesc[index].description, len);
    } else {
        return (PAPI_EINVAL);
    }
    return (PAPI_OK);
}


/* Vector that points to entry points for the component */
papi_vector_t _rocm_vector = {
    .cmp_info = {
                 /* default component information (unspecified values are initialized to 0) */
                 .name = "rocm",
                 .short_name = "rocm",
                 .version = "1.0",
                 .description = "GPU events and metrics via AMD ROCm-PL API",
                 .num_mpx_cntrs = PAPIROCM_MAX_COUNTERS,
                 .num_cntrs = PAPIROCM_MAX_COUNTERS,
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
             .context = 1,      /* sizeof( papirocm_context_t ), */
             .control_state = 1,        /* sizeof( papirocm_control_t ), */
             .reg_value = 1,    /* sizeof( papirocm_register_t ), */
             .reg_alloc = 1,    /* sizeof( papirocm_reg_alloc_t ), */
             }
    ,
    /* function pointers in this component */
    .start = papirocm_start,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .stop = papirocm_stop,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .read = papirocm_read,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    .reset = papirocm_reset,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .cleanup_eventset = papirocm_cleanup_eventset,      /* ( hwd_control_state_t * ctrl ) */

    .init_component = papirocm_init_component,  /* ( int cidx ) */
    .init_thread = papirocm_init_thread,        /* ( hwd_context_t * ctx ) */
    .init_control_state = papirocm_init_control_state,  /* ( hwd_control_state_t * ctrl ) */
    .update_control_state = papirocm_update_control_state,      /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */

    .ctl = papirocm_ctrl,       /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    .set_domain = papirocm_set_domain,  /* ( hwd_control_state_t * cntrl, int domain ) */
    .ntv_enum_events = papirocm_ntv_enum_events,        /* ( unsigned int *EventCode, int modifier ) */
    .ntv_code_to_name = papirocm_ntv_code_to_name,      /* ( unsigned int EventCode, char *name, int len ) */
    .ntv_code_to_descr = papirocm_ntv_code_to_descr,    /* ( unsigned int EventCode, char *name, int len ) */
    .shutdown_thread = papirocm_shutdown_thread,        /* ( hwd_context_t * ctx ) */
    .shutdown_component = papirocm_shutdown_component,  /* ( void ) */
};
