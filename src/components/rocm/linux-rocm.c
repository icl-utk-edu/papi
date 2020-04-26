/*/
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
#define ROCMDBG(format, args...) fprintf(stderr, format, ## args)
#else
//#define ROCMDBG(format, args...) do {} while(0)
#define ROCMDBG SUBDBG
#endif

/* Macros for error checking... each arg is only referenced/evaluated once */
#define CHECK_PRINT_EVAL(checkcond, str, evalthis)                      \
    do {                                                                \
        int _cond = (checkcond);                                        \
        if (_cond) {                                                    \
            fprintf(stderr, "%s:%i error: condition %s failed: %s.\n", __FILE__, __LINE__, #checkcond, str); \
            evalthis;                                                   \
        }                                                               \
    } while (0)

#define ROCM_CALL_CK(call, args, handleerror)                           \
    do {                                                                \
        hsa_status_t _status = (*call##Ptr)args;                        \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {    \
            fprintf(stderr, "%s:%i error: function %s failed with error %d.\n",     \
            __FILE__, __LINE__, #call, _status);                                    \
            handleerror;                                                \
        }                                                               \
    } while (0)

// Roc Profiler call.
#define ROCP_CALL_CK(call, args, handleerror)                           \
    do {                                                                \
        hsa_status_t _status = (*call##Ptr)args;                        \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {     \
            const char *profErr;                                                     \
            (*rocprofiler_error_stringPtr)(&profErr);                              \
            fprintf(stderr, "%s:%i error: function %s failed with error %d [%s].\n", \
            __FILE__, __LINE__, #call, _status, profErr);               \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define DLSYM_AND_CHECK(dllib, name)                                    \
    do {                                                                \
        name##Ptr = dlsym(dllib, #name);                                \
        if (dlerror()!=NULL) {                                          \
            snprintf(_rocm_vector.cmp_info.disabled_reason,             \
                PAPI_MAX_STR_LEN,                                                   \
                "The ROCM required function '%s' was not found in dynamic libs",    \
                #name);                                                             \
            fprintf(stderr, "%s:%i ROCM component disabled: %s\n",                  \
                __FILE__, __LINE__, _rocm_vector.cmp_info.disabled_reason);         \
          return ( PAPI_ENOSUPP );                                      \
        }                                                               \
    } while (0)

typedef rocprofiler_t* Context;
typedef rocprofiler_feature_t EventID;

// Contains device list, pointer to device description, and the list of available events.
// Note that "indexed variables" in ROCM are read with eventname[%d], where %d is
// 0 to #instances. This is what we store in the EventID.name element. But the PAPI name
// doesn't use brackets; so in the ev_name_desc.name we store the user-visible name,
// something like "eventname:device=%d:instance=%d".
typedef struct _rocm_context {
    uint32_t availAgentSize;
    hsa_agent_t* availAgentArray;
    uint32_t availEventSize;
    int *availEventDeviceNum;
    EventID *availEventIDArray;                         // Note: The EventID struct has its own .name element for ROCM internal operation.
    uint32_t *availEventIsBeingMeasuredInEventset;
    struct ev_name_desc *availEventDesc;                // Note: This is where the PAPI name is stored; for user consumption.
} _rocm_context_t;

/* Store the name and description for an event */
typedef struct ev_name_desc {
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_2MAX_STR_LEN];
} ev_name_desc_t;

/* Control structure tracks array of active contexts, records active events and their values */
typedef struct _rocm_control {
    uint32_t countOfActiveContexts;
    struct _rocm_active_context_s *arrayOfActiveContexts[PAPIROCM_MAX_COUNTERS];
    uint32_t activeEventCount;
    int activeEventIndex[PAPIROCM_MAX_COUNTERS];
    long long activeEventValues[PAPIROCM_MAX_COUNTERS];
    uint64_t startTimestampNs;
    uint64_t readTimestampNs;
} _rocm_control_t;

/* For each active context, which ROCM events are being measured, context eventgroups containing events */
typedef struct _rocm_active_context_s {
    Context ctx;
    int deviceNum;
    uint32_t conEventsCount;
    EventID conEvents[PAPIROCM_MAX_COUNTERS];
    int conEventIndex[PAPIROCM_MAX_COUNTERS];
} _rocm_active_context_t;

/* Function prototypes */
static int _rocm_cleanup_eventset(hwd_control_state_t * ctrl);

// GLOBALS
static void     *dl1 = NULL;
static void     *dl2 = NULL;
static char     rocm_hsa[]=PAPI_ROCM_HSA;
static char     rocm_prof[]=PAPI_ROCM_PROF;

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
DECLAREROCMFUNC(hsa_queue_destroy, (hsa_queue_t* queue));

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
DECLAREROCMFUNC(rocprofiler_error_string, (const char**));

/* The PAPI side (external) variable as a global */
papi_vector_t _rocm_vector;

/* Global variable for hardware description, event and metric lists */
static _rocm_context_t *global__rocm_context = NULL;
static uint32_t maxEventSize=0;                 // We accumulate all agent counts into this.
static rocprofiler_properties_t global__ctx_properties = {
  NULL, // queue
  128, // queue depth
  NULL, // handler on completion
  NULL // handler_arg
};

/* This global variable points to the head of the control state list */
static _rocm_control_t *global__rocm_control = NULL;


/*****************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/

/*
 * Link the necessary ROCM libraries to use the rocm component.  If any of them can not be found, then
 * the ROCM component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the ROCM component can be installed and used on systems which have the ROCM libraries installed
 * and on systems where these libraries are not installed.
 */
static int _rocm_linkRocmLibraries(void)
{
    ROCMDBG("Entering _rocm_linkRocmLibraries\n");

    char path_name[1024];
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        strncpy(_rocm_vector.cmp_info.disabled_reason, "The ROCM component does not support statically linking to libc.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }

    // collect any defined environment variables, or "NULL" if not present.
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    dl1 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(rocm_hsa) > 0) {                             // If override given, it has to work.
        dl1 = dlopen(rocm_hsa, RTLD_NOW | RTLD_GLOBAL);     // Try to open that path.
        if (dl1 == NULL) {
            snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_ROCM_HSA override '%s' given in Rules.rocm not found.", rocm_hsa);
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl1 == NULL) {                                                  // No override,
        dl1 = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);    // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl1 == NULL && rocm_root != NULL) {                          // if root given, try it.
        snprintf(path_name, 1024, "%s/lib/libhsa-runtime64.so", rocm_root);  // PAPI Root check.
        dl1 = dlopen(path_name, RTLD_NOW | RTLD_GLOBAL);             // Try to open that path.
    }

    // Check for failure.
    if (dl1 == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libhsa-runtime64.so not found.");
        return(PAPI_ENOSUPP);
    }

    // We have a dl1. (libhsa-runtime64.so).

    DLSYM_AND_CHECK(dl1, hsa_init);
    DLSYM_AND_CHECK(dl1, hsa_iterate_agents);
    DLSYM_AND_CHECK(dl1, hsa_system_get_info);
    DLSYM_AND_CHECK(dl1, hsa_agent_get_info);
    DLSYM_AND_CHECK(dl1, hsa_shut_down);
    DLSYM_AND_CHECK(dl1, hsa_queue_destroy);

    //-------------------------------------------------------------------------

    dl2 = NULL;                                                 // Ensure reset to NULL.

    // Step 1: Process override if given.
    if (strlen(rocm_prof) > 0) {                             // If override given, it has to work.
        dl2 = dlopen(rocm_prof, RTLD_NOW | RTLD_GLOBAL);     // Try to open that path.
        if (dl1 == NULL) {
            snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_ROCM_PROF override '%s' given in Rules.rocm not found.", rocm_prof);
            return(PAPI_ENOSUPP);   // Override given but not found.
        }
    }

    // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
    if (dl2 == NULL) {                                                  // No override,
        dl2 = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL);    // Try system paths.
    }

    // Step 3: Try the explicit install default.
    if (dl2 == NULL && rocm_root != NULL) {                          // if root given, try it.
        snprintf(path_name, 1024, "%s/lib/librocprofiler64.so", rocm_root);  // PAPI Root check.
        dl2 = dlopen(path_name, RTLD_NOW | RTLD_GLOBAL);             // Try to open that path.
    }

    // Check for failure.
    if (dl2 == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "librocprofiler64.so not found.");
        return(PAPI_ENOSUPP);
    }

    // We have a dl2. (librocprofiler64.so).

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
    DLSYM_AND_CHECK(dl2, rocprofiler_error_string);

    // Disable if ROCPROFILER env vars not present.
    if (getenv("ROCP_METRICS") == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Env. Var. ROCP_METRICS not set; rocprofiler is not configured.");
        return(PAPI_ENOSUPP);   // Wouldn't have any events.
    }

    if (getenv("ROCPROFILER_LOG") == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Env. Var. ROCPROFILER_LOG not set; rocprofiler is not configured.");
        return(PAPI_ENOSUPP);   // Wouldn't have any events.
    }

    if (getenv("HSA_VEN_AMD_AQLPROFILE_LOG") == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Env. Var. HSA_VEN_AMD_AQLPROFILE_LOG not set; rocprofiler is not configured.");
        return(PAPI_ENOSUPP);   // Wouldn't have any events.
    }

    if (getenv("AQLPROFILE_READ_API") == NULL) {
        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Env. Var.AQLPROFILE_READ_API not set; rocprofiler is not configured.");
        return(PAPI_ENOSUPP);   // Wouldn't have any events.
    }

    return (PAPI_OK);
}


// ----------------------------------------------------------------------------
// Callback function to get the number of agents
static hsa_status_t _rocm_get_gpu_handle(hsa_agent_t agent, void* arg)
{
  _rocm_context_t * gctxt = (_rocm_context_t*) arg;

  hsa_device_type_t type;
  ROCM_CALL_CK(hsa_agent_get_info,(agent, HSA_AGENT_INFO_DEVICE, &type), return (PAPI_EMISC));

  // Device is a GPU agent
  if (type == HSA_DEVICE_TYPE_GPU) {
    gctxt->availAgentSize += 1;
    gctxt->availAgentArray = (hsa_agent_t*) papi_realloc(gctxt->availAgentArray, (gctxt->availAgentSize*sizeof(hsa_agent_t)));
    gctxt->availAgentArray[gctxt->availAgentSize - 1] = agent;
  }

  return HSA_STATUS_SUCCESS;
}

typedef struct {
    int device_num;
    int count;
    _rocm_context_t * ctx;
} events_callback_arg_t;

// ----------------------------------------------------------------------------
// Callback function to get the number of events we will see;
// Each element of instanced metrics must be created as a separate event
static hsa_status_t _rocm_count_native_events_callback(const rocprofiler_info_data_t info, void * arg)
{
    const uint32_t instances = info.metric.instances;
    uint32_t* count = (uint32_t*) arg;
    (*count) += instances;
    return HSA_STATUS_SUCCESS;
} // END CALLBACK.


// ----------------------------------------------------------------------------
// Callback function that adds individual events.
static hsa_status_t _rocm_add_native_events_callback(const rocprofiler_info_data_t info, void * arg)
{
    uint32_t ui;
    events_callback_arg_t * callback_arg = (events_callback_arg_t*) arg;
    _rocm_context_t * ctx = callback_arg->ctx;
    const uint32_t eventDeviceNum = callback_arg->device_num;
    const uint32_t count = callback_arg->count;
          uint32_t index = ctx->availEventSize;
    const uint32_t instances = info.metric.instances;   // short cut to instances.


//  information about AMD Event.
//   fprintf(stderr, "%s:%i name=%s block_name=%s, instances=%i block_counters=%i.\n",
//      __FILE__, __LINE__, info.metric.name, info.metric.block_name, info.metric.instances,
//      info.metric.block_counters);
    if (index + instances > count) return HSA_STATUS_ERROR; // Should have enough space.

    for (ui=0; ui<instances; ui++) {
      char ROCMname[PAPI_MAX_STR_LEN];

        if (instances > 1) {
            snprintf(ctx->availEventDesc[index].name,
                PAPI_MAX_STR_LEN, "%s:device=%d:instance=%d",       // What PAPI user sees.
                info.metric.name, eventDeviceNum, ui);
            snprintf(ROCMname, PAPI_MAX_STR_LEN, "%s[%d]",
                info.metric.name, ui);                               // use indexed version.
        } else {
            snprintf(ctx->availEventDesc[index].name,
                PAPI_MAX_STR_LEN, "%s:device=%d",                   // What PAPI user sees.
                info.metric.name, eventDeviceNum);
            snprintf(ROCMname, PAPI_MAX_STR_LEN, "%s",
                info.metric.name);                                  // use non-indexed version.
        }

        ROCMname[PAPI_MAX_STR_LEN - 1] = '\0';                      // ensure z-terminated.
        strncpy(ctx->availEventDesc[index].description, info.metric.description, PAPI_2MAX_STR_LEN);
        ctx->availEventDesc[index].description[PAPI_2MAX_STR_LEN - 1] = '\0';   // ensure z-terminated.

        EventID eventId;                                            // Removed declaration init.
        eventId.kind = ROCPROFILER_FEATURE_KIND_METRIC;
        eventId.name = strdup(ROCMname);                            // what ROCM needs to see.
        eventId.parameters = NULL;                                  // Not currently used, but init for safety.
        eventId.parameter_count=0;                                  // Not currently used, but init for safety.

        ctx->availEventDeviceNum[index] = eventDeviceNum;
        ctx->availEventIDArray[index] = eventId;
        index++;                                                    // increment index.
        ctx->availEventSize = index;                                // Always set availEventSize.
    } // end for each instance.

    return HSA_STATUS_SUCCESS;
} // end CALLBACK, _rocm_add_native_events_callback

// ----------------------------------------------------------------------------
// function called during initialization.
static int _rocm_add_native_events(_rocm_context_t * ctx)
{
    ROCMDBG("Entering _rocm_add_native_events\n");

    uint32_t i;

    // Count all events in all agents; Each element of 'indexed' metrics is considered a separate event.
    // NOTE: The environment variable ROCP_METRICS should point at a path and file like metrics.xml.
    //       If that file doesn't exist, this iterate info fails with a general error (0x1000).
    // NOTE: We are *accumulating* into maxEventSize.
    for (i = 0; i < ctx->availAgentSize; i++) {
        ROCP_CALL_CK(rocprofiler_iterate_info, (&(ctx->availAgentArray[i]), ROCPROFILER_INFO_KIND_METRIC,
            _rocm_count_native_events_callback, (void*)(&maxEventSize)), return (PAPI_EMISC));
    }

    /* Allocate space for all events and descriptors, includes space for instances. */
    ctx->availEventDeviceNum = (int *) papi_calloc(maxEventSize, sizeof(int));
    CHECK_PRINT_EVAL((ctx->availEventDeviceNum == NULL), "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventIDArray = (EventID *) papi_calloc(maxEventSize, sizeof(EventID));
    CHECK_PRINT_EVAL((ctx->availEventIDArray == NULL), "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventIsBeingMeasuredInEventset = (uint32_t *) papi_calloc(maxEventSize, sizeof(uint32_t));
    CHECK_PRINT_EVAL((ctx->availEventIsBeingMeasuredInEventset == NULL), "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));
    ctx->availEventDesc = (ev_name_desc_t *) papi_calloc(maxEventSize, sizeof(ev_name_desc_t));
    CHECK_PRINT_EVAL((ctx->availEventDesc == NULL), "ERROR ROCM: Could not allocate memory", return (PAPI_ENOMEM));

    for (i = 0; i < ctx->availAgentSize; ++i) {
        events_callback_arg_t arg;
        arg.device_num = i;
        arg.count = maxEventSize;
        arg.ctx = ctx;
        ROCP_CALL_CK(rocprofiler_iterate_info, (&(ctx->availAgentArray[i]), ROCPROFILER_INFO_KIND_METRIC,
            _rocm_add_native_events_callback, (void*)(&arg)), return (PAPI_EMISC));
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
static int _rocm_init_thread(hwd_context_t * ctx)
{
    ROCMDBG("Entering _rocm_init_thread\n");

    (void) ctx;
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int _rocm_init_component(int cidx)
{
    ROCMDBG("Entering _rocm_init_component\n");

    /* link in all the rocm libraries and resolve the symbols we need to use */
    if(_rocm_linkRocmLibraries() != PAPI_OK) {
        SUBDBG("Dynamic link of ROCM libraries failed, component will be disabled.\n");
        SUBDBG("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    ROCM_CALL_CK(hsa_init, (), return (PAPI_EMISC));

    /* Create the structure */
    if(global__rocm_context == NULL)
        global__rocm_context = (_rocm_context_t *) papi_calloc(1, sizeof(_rocm_context_t));

    /* Get GPU agent */
    ROCM_CALL_CK(hsa_iterate_agents, (_rocm_get_gpu_handle, global__rocm_context), return (PAPI_EMISC));

    int rv;

    /* Get list of all native ROCM events supported */
    rv = _rocm_add_native_events(global__rocm_context);
    if(rv != 0)
        return (rv);

    /* Export some information */
    _rocm_vector.cmp_info.CmpIdx = cidx;
    _rocm_vector.cmp_info.num_native_events = global__rocm_context->availEventSize;
    _rocm_vector.cmp_info.num_cntrs = _rocm_vector.cmp_info.num_native_events;
    _rocm_vector.cmp_info.num_mpx_cntrs = _rocm_vector.cmp_info.num_native_events;

    ROCMDBG("Exiting _rocm_init_component cidx %d num_native_events %d num_cntrs %d num_mpx_cntrs %d\n",
        cidx,
        _rocm_vector.cmp_info.num_native_events,
        _rocm_vector.cmp_info.num_cntrs,
        _rocm_vector.cmp_info.num_mpx_cntrs);

    if (_rocm_vector.cmp_info.num_native_events == 0) {
        char *metrics = getenv("ROCP_METRICS");
        if (metrics == NULL) {
            strncpy(_rocm_vector.cmp_info.disabled_reason, "Environment Variable ROCP_METRICS is not defined, should point to a valid metrics.xml.", PAPI_MAX_STR_LEN);
            return (PAPI_EMISC);
        }

        snprintf(_rocm_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "No events.  Ensure ROCP_METRICS=%s is correct.", metrics);
        return (PAPI_EMISC);
    }

    return (PAPI_OK);
}


/* Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int _rocm_init_control_state(hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering _rocm_init_control_state\n");

    (void) ctrl;
    _rocm_context_t *gctxt = global__rocm_context;

    CHECK_PRINT_EVAL((gctxt == NULL), "Error: The PAPI ROCM component needs to be initialized first", return (PAPI_ENOINIT));
    /* If no events were found during the initial component initialization, return error */
    if(global__rocm_context->availEventSize <= 0) {
        strncpy(_rocm_vector.cmp_info.disabled_reason, "ERROR ROCM: No events exist", PAPI_MAX_STR_LEN);
        return (PAPI_EMISC);
    }
    /* If it does not exist, create the global structure to hold ROCM contexts and active events */
    if(global__rocm_control == NULL) {
        global__rocm_control = (_rocm_control_t *) papi_calloc(1, sizeof(_rocm_control_t));
        global__rocm_control->countOfActiveContexts = 0;
        global__rocm_control->activeEventCount = 0;
    }
    return PAPI_OK;
}


/* Triggered by eventset operations like add or remove.  For ROCM,
 * needs to be called multiple times from each seperate ROCM context
 * with the events to be measured from that context.  For each
 * context, create eventgroups for the events.
 */
/* Note: NativeInfo_t is defined in papi_internal.h */
static int _rocm_update_control_state(hwd_control_state_t * ctrl, NativeInfo_t * nativeInfo, int nativeCount, hwd_context_t * ctx)
{
    ROCMDBG("Entering _rocm_update_control_state with nativeCount %d\n", nativeCount);

    (void) ctx;
    _rocm_control_t *gctrl = global__rocm_control;
    _rocm_context_t *gctxt = global__rocm_context;
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
        char *eventName = gctxt->availEventDesc[index].name;
        (void) eventName;
        int eventDeviceNum = gctxt->availEventDeviceNum[index];

        /* if this event is already added continue to next ii, if not, mark it as being added */
        if(gctxt->availEventIsBeingMeasuredInEventset[index] == 1) {
            ROCMDBG("Skipping event %s (%i of %i) which is already added\n", eventName, ii, nativeCount);
            continue;
        } else {
            gctxt->availEventIsBeingMeasuredInEventset[index] = 1;
        }

        /* Find context/control in papirocm, creating it if does not exist */
        for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
            CHECK_PRINT_EVAL(cc >= PAPIROCM_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (PAPIROCM_MAX_COUNTERS)", return (PAPI_EMISC));
            if(gctrl->arrayOfActiveContexts[cc]->deviceNum == eventDeviceNum) {
                break;
            }
        }
        // Create context if it does not exist
        if(cc == gctrl->countOfActiveContexts) {
            ROCMDBG("Event %s device %d does not have a ctx registered yet...\n", eventName, eventDeviceNum);
            gctrl->arrayOfActiveContexts[cc] = papi_calloc(1, sizeof(_rocm_active_context_t));
            CHECK_PRINT_EVAL(gctrl->arrayOfActiveContexts[cc] == NULL, "Memory allocation for new active context failed", return (PAPI_ENOMEM));
            gctrl->arrayOfActiveContexts[cc]->deviceNum = eventDeviceNum;
            gctrl->arrayOfActiveContexts[cc]->ctx = NULL;
            gctrl->arrayOfActiveContexts[cc]->conEventsCount = 0;
            gctrl->countOfActiveContexts++;
            ROCMDBG("Added a new context deviceNum %d ... now countOfActiveContexts is %d\n", eventDeviceNum, gctrl->countOfActiveContexts);
        }
        eventContextIdx = cc;

        _rocm_active_context_t *eventctrl = gctrl->arrayOfActiveContexts[eventContextIdx];
        ROCMDBG("Need to add event %d %s to the context\n", index, eventName);
        // Now we have eventctrl, we can check on max event count.
        if (eventctrl->conEventsCount >= PAPIROCM_MAX_COUNTERS) {
            ROCMDBG("Num events exceeded PAPIROCM_MAX_COUNTERS\n");
            return(PAPI_EINVAL);
        }

        /* lookup eventid for this event index */
        EventID eventId = gctxt->availEventIDArray[index];
        eventctrl->conEvents[eventctrl->conEventsCount] = eventId;
        eventctrl->conEventIndex[eventctrl->conEventsCount] = index;
        eventctrl->conEventsCount++;
//      fprintf(stderr, "%s:%d Added eventId.name='%s' as conEventsCount=%i with index=%i.\n", __FILE__, __LINE__, eventId.name, eventctrl->conEventsCount-1, index); // test indexed events.

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
                ROCP_CALL_CK(rocprofiler_close, (eventctrl->ctx), return (PAPI_EMISC));
            }
            int openFailed=0;
//          fprintf(stderr,"%s:%i calling rocprofiler_open, ii=%i device=%i numEvents=%i name='%s'.\n", __FILE__, __LINE__, ii, eventDeviceNum, eventctrl->conEventsCount, eventId.name);
            const uint32_t mode = (global__ctx_properties.queue != NULL) ? ROCPROFILER_MODE_STANDALONE : ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE;
            ROCP_CALL_CK(rocprofiler_open, (gctxt->availAgentArray[eventDeviceNum], eventctrl->conEvents, eventctrl->conEventsCount, &(eventctrl->ctx),
                         mode, &global__ctx_properties), openFailed=1);
            if (openFailed) {                       // If the open failed,
                ROCMDBG("Error occurred: The ROCM event was not accepted by the ROCPROFILER.\n");
//              fprintf(stderr, "Error occurred: The ROCM event '%s' was not accepted by the ROCPROFILER.\n", eventId.name);
                _rocm_cleanup_eventset(ctrl);       // Try to cleanup,
//              fprintf(stderr, "%s:%i Returning PAPI_ECOMBO.\n", __FILE__, __LINE__);
                return(PAPI_ECOMBO);                // Say its a bad combo.
            }

            ROCP_CALL_CK(rocprofiler_group_count, (eventctrl->ctx, &numPasses), return (PAPI_EMISC));

            if (numPasses > 1) {
                ROCMDBG("Error occurred: The combined ROCM events require more than 1 pass... try different events\n");
                _rocm_cleanup_eventset(ctrl);
                return(PAPI_ECOMBO);
            } else  {
                ROCMDBG("Created eventGroupPasses for context total-events %d in-this-context %d passes-required %d) \n", gctrl->activeEventCount, eventctrl->conEventsCount, numPasses);
            }
        }
    }
    return (PAPI_OK);
}


/* Triggered by PAPI_start().
 * For ROCM component, switch to each context and start all eventgroups.
 */
static int _rocm_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering _rocm_start\n");

    (void) ctx;
    (void) ctrl;
    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t ii, cc;

    ROCMDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->startTimestampNs), return (PAPI_EMISC));
    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // suppress "not used" error when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Start device %d ctx %p ts %lu\n", eventDeviceNum, eventCtx, gctrl->startTimestampNs);
        if (eventCtx == NULL) abort();
        ROCP_CALL_CK(rocprofiler_start, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* Triggered by PAPI_read().  For ROCM component, switch to each
 * context, read all the eventgroups, and put the values in the
 * correct places. */
static int _rocm_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **values, int flags)
{
    ROCMDBG("Entering _rocm_read\n");

    (void) ctx;
    (void) ctrl;
    (void) flags;
    _rocm_control_t *gctrl = global__rocm_control;
    _rocm_context_t *gctxt = global__rocm_context;
    uint32_t cc, jj, ee;

    // Get read time stamp
    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->readTimestampNs), return (PAPI_EMISC));
    uint64_t durationNs = gctrl->readTimestampNs - gctrl->startTimestampNs;
    (void) durationNs;                                                  // Suppress 'not used' warning when not debug.
    gctrl->startTimestampNs = gctrl->readTimestampNs;


    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Read device %d ctx %p(%u) ts %lu\n", eventDeviceNum, eventCtx, cc, gctrl->readTimestampNs);
        ROCP_CALL_CK(rocprofiler_read, (eventCtx, 0), return (PAPI_EMISC));
        ROCMDBG("waiting for data\n");
        ROCP_CALL_CK(rocprofiler_get_data, (eventCtx, 0), return (PAPI_EMISC));
        ROCP_CALL_CK(rocprofiler_get_metrics, (eventCtx), return (PAPI_EMISC));
        ROCMDBG("done\n");

        for(jj = 0; jj < gctrl->activeEventCount; jj++) {
            int index = gctrl->activeEventIndex[jj];
            EventID eventId = gctxt->availEventIDArray[index];
            ROCMDBG("jj=%i of %i, index=%i, device#=%i.\n", jj, gctrl->activeEventCount, index, gctxt->availEventDeviceNum[index]);
            (void) eventId;                                             // Suppress 'not used' warning when not debug.

            /* If the device/context does not match the current context, move to next */
            if(gctxt->availEventDeviceNum[index] != eventDeviceNum)
                continue;

            for(ee = 0; ee < gctrl->arrayOfActiveContexts[cc]->conEventsCount; ee++) {
                ROCMDBG("Searching for activeEvent %s in Activecontext %u eventIndex %d duration %lu\n", eventId.name, ee, index, durationNs);
                if (gctrl->arrayOfActiveContexts[cc]->conEventIndex[ee] == index) {
                  gctrl->activeEventValues[jj] = gctrl->arrayOfActiveContexts[cc]->conEvents[ee].data.result_int64;
                  ROCMDBG("Matched event %d:%d eventName %s value %lld\n", jj, index, eventId.name, gctrl->activeEventValues[jj]);
                  break;
                }
            }
        }
    }

    *values = gctrl->activeEventValues;
    return (PAPI_OK);
}


/* Triggered by PAPI_stop() */
static int _rocm_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering _rocm_stop\n");

    (void) ctx;
    (void) ctrl;
    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // Suppress 'not used' warning when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Stop device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCP_CALL_CK(rocprofiler_stop, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
} // END ROUTINE.

/*
 * Disable and destroy the ROCM eventGroup
 */
static int _rocm_cleanup_eventset(hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering _rocm_cleanup_eventset\n");
//  fprintf(stderr, "%s:%i _rocm_cleanup_eventset called.\n", __FILE__, __LINE__);

    (void) ctrl;
    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t i, cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // Suppress 'not used' warning when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Destroy device %d ctx %p \n", eventDeviceNum, eventCtx);
//      fprintf(stderr, "%s:%i About to call rocprofiler_close.\n", __FILE__, __LINE__);
        ROCP_CALL_CK(rocprofiler_close, (eventCtx), return (PAPI_EMISC));
//      fprintf(stderr, "%s:%i Returned from call to rocprofiler_close, papi_free ptr=%p.\n", __FILE__, __LINE__, gctrl->arrayOfActiveContexts[cc] );
        papi_free( gctrl->arrayOfActiveContexts[cc] );
//      fprintf(stderr, "%s:%i Returned from call to papi_free.\n", __FILE__, __LINE__);
    }
    if (global__ctx_properties.queue != NULL) {
      ROCM_CALL_CK(hsa_queue_destroy, (global__ctx_properties.queue), return (PAPI_EMISC));
      global__ctx_properties.queue = NULL;
    }
    /* Record that there are no active contexts or events */
//  fprintf(stderr, "%s:%i Checkpoint, maxEventSize=%i.\n", __FILE__, __LINE__, maxEventSize);
    gctrl->countOfActiveContexts = 0;
    gctrl->activeEventCount = 0;

    /* Clear all indicators of event being measured. */
    _rocm_context_t *gctxt = global__rocm_context;
    for (i=0; i<maxEventSize; i++) {
            gctxt->availEventIsBeingMeasuredInEventset[i] = 0;
    }

//  fprintf(stderr, "%s:%i Returning from _rocm_cleanup_eventset.\n", __FILE__, __LINE__);
    return (PAPI_OK);
}


/* Called at thread shutdown. Does nothing in the ROCM component. */
static int _rocm_shutdown_thread(hwd_context_t * ctx)
{
    ROCMDBG("Entering _rocm_shutdown_thread\n");

    (void) ctx;
    return (PAPI_OK);
}


/* Triggered by PAPI_shutdown() and frees memory allocated in the ROCM component. */
static int _rocm_shutdown_component(void)
{
    ROCMDBG("Entering _rocm_shutdown_component\n");

    _rocm_control_t *gctrl = global__rocm_control;
    _rocm_context_t *gctxt = global__rocm_context;
    uint32_t cc;

    /* Free context */
    if(gctxt != NULL) {
        papi_free(gctxt->availEventIDArray);
        papi_free(gctxt->availEventDeviceNum);
        papi_free(gctxt->availEventIsBeingMeasuredInEventset);
        papi_free(gctxt->availEventDesc);
        papi_free(gctxt);
        global__rocm_context = gctxt = NULL;
    }

    /* Free control */
    if(gctrl != NULL) {
        for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
            if(gctrl->arrayOfActiveContexts[cc] != NULL) {
                papi_free(gctrl->arrayOfActiveContexts[cc]);
            }
        }

        papi_free(gctrl);
        global__rocm_control = gctrl = NULL;
    }

    // Shutdown ROC runtime
    // DEBUG: This causes a segfault.
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
static int _rocm_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
    ROCMDBG("Entering _rocm_reset\n");

    (void) ctx;
    (void) ctrl;
    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t ii, cc;

    ROCMDBG("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // Suppress 'not used' error when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        ROCMDBG("Reset device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCP_CALL_CK(rocprofiler_reset, (eventCtx, 0), return (PAPI_EMISC));
    }

    return (PAPI_OK);
}


/* This function sets various options in the component - Does nothing in the ROCM component.
    @param[in] ctx -- hardware context
    @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
    @param[in] option -- options to be set
*/
static int _rocm_ctrl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
    ROCMDBG("Entering _rocm_ctrl\n");

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
static int _rocm_set_domain(hwd_control_state_t * ctrl, int domain)
{
    ROCMDBG("Entering _rocm_set_domain\n");

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
static int _rocm_ntv_enum_events(unsigned int *EventCode, int modifier)
{
    //ROCMDBG("Entering (get next event after %u)\n", *EventCode );

    switch (modifier) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return (PAPI_OK);
        break;
    case PAPI_ENUM_EVENTS:
        if(global__rocm_context == NULL) {
            return (PAPI_ENOEVNT);
        } else if(*EventCode < global__rocm_context->availEventSize - 1) {
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


//----------------------------------------------------------------------------
// Takes a native event code and passes back the name, but the PAPI version
// of the name in availEventDesc[], not the ROCM internal name (in
// availEventIDArray[].name).
// @param EventCode is the native event code
// @param name is a pointer for the name to be copied to
// @param len is the size of the name string
//----------------------------------------------------------------------------
static int _rocm_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
    //ROCMDBG("Entering EventCode %d\n", EventCode );

    unsigned int index = EventCode;
    _rocm_context_t *gctxt = global__rocm_context;
    if(gctxt != NULL && index < gctxt->availEventSize) {
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
static int _rocm_ntv_code_to_descr(unsigned int EventCode, char *name, int len)
{
    //ROCMDBG("Entering _rocm_ntv_code_to_descr\n");

    unsigned int index = EventCode;
    _rocm_context_t *gctxt = global__rocm_context;
    if(gctxt != NULL && index < gctxt->availEventSize) {
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
             .context = 1,      /* sizeof( _rocm_context_t ), */
             .control_state = 1,        /* sizeof( _rocm_control_t ), */
             .reg_value = 1,    /* sizeof( _rocm_register_t ), */
             .reg_alloc = 1,    /* sizeof( _rocm_reg_alloc_t ), */
             }
    ,
    /* function pointers in this component */
    .start = _rocm_start,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .stop = _rocm_stop,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .read = _rocm_read,      /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    .reset = _rocm_reset,    /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .cleanup_eventset = _rocm_cleanup_eventset,      /* ( hwd_control_state_t * ctrl ) */

    .init_component = _rocm_init_component,  /* ( int cidx ) */
    .init_thread = _rocm_init_thread,        /* ( hwd_context_t * ctx ) */
    .init_control_state = _rocm_init_control_state,  /* ( hwd_control_state_t * ctrl ) */
    .update_control_state = _rocm_update_control_state,      /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */

    .ctl = _rocm_ctrl,       /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    .set_domain = _rocm_set_domain,  /* ( hwd_control_state_t * cntrl, int domain ) */
    .ntv_enum_events = _rocm_ntv_enum_events,        /* ( unsigned int *EventCode, int modifier ) */
    .ntv_code_to_name = _rocm_ntv_code_to_name,      /* ( unsigned int EventCode, char *name, int len ) */
    .ntv_code_to_descr = _rocm_ntv_code_to_descr,    /* ( unsigned int EventCode, char *name, int len ) */
    .shutdown_thread = _rocm_shutdown_thread,        /* ( hwd_context_t * ctx ) */
    .shutdown_component = _rocm_shutdown_component,  /* ( void ) */
};

