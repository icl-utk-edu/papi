//-----------------------------------------------------------------------------
// This code will uses the API implemented in linux-rocm.c, but without any
// dependence on the rest of PAPI. The main() function will read command 
// line events.
// This is compiled with ROCM_SA_Makefile using hipcc; so a HIP kernel 
// can be included and tested.  
//
// requires linking with libhsa-runtime64.so.
// requires linking wtih librocprofiler64.so.
//-----------------------------------------------------------------------------

#include <dlfcn.h>
#include <hsa.h>
#include <rocprofiler.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

// this number assumes that there will never be more events than indicated
#define ROCM_MAX_COUNTERS 512

// Defines for some things usually defined by PAPI.
#define ROCM_MAX_STR_LEN (1024)
#define ROCM_SA_ECOMBO  (-4)
#define ROCM_SA_INVALID (-3)
#define ROCM_SA_ENOMEM (-2)
#define ROCM_SA_FAIL (-1)
#define ROCM_SA_OK   (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                     \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define printf_v1(format, args...) {if (Verbose>0) {printf(format, ## args);}}
#define printf_v2(format, args...) {if (Verbose>1) {printf(format, ## args);}}
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

/* Macros for error checking... each arg is only referenced/evaluated once */
#define CHECK_PRINT_EVAL(checkcond, str, evalthis)                      \
    do {                                                                \
        int _cond = (checkcond);                                        \
        if (_cond) {                                                    \
            fprintf(stderr, "%s:%i error: condition %s failed: %s.\n", __FILE__, __LINE__, #checkcond, str); \
            evalthis;                                                   \
        }                                                               \
    } while (0)

// To allow direct linking of the shared library, we do not append
// Ptr to the routine name, and don't use the arg as a pointer. 
#define ROCM_CALL_CK(call, args, handleerror)                           \
    do {                                                                \
        /* hsa_status_t _status = (*call##Ptr)args;     */              \
        hsa_status_t _status = call args;                               \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {    \
            fprintf(stderr, "%s:%i error: function %s failed with error %d.\n",     \
            __FILE__, __LINE__, #call, _status);                                    \
            handleerror;                                                \
        }                                                               \
    } while (0)

// Roc Profiler call.
// To allow direct linking of the shared library, we do not append
// Ptr to the routine name, and don't use the arg as a pointer. 
#define ROCP_CALL_CK(call, args, handleerror)                           \
    do {                                                                \
        /* hsa_status_t _status = (*call##Ptr)args; */                  \
        hsa_status_t _status = (call)args;                              \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {     \
            const char *profErr;                                                     \
            /* (*rocprofiler_error_stringPtr)(&profErr); */                          \
            rocprofiler_error_string(&profErr);                                      \
            fprintf(stderr, "%s:%i error: function %s failed with error %d [%s].\n", \
            __FILE__, __LINE__, #call, _status, profErr);               \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define HIPCHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}


void do_work(void);

typedef rocprofiler_t* Context;
typedef rocprofiler_feature_t EventID;

// Contains device list, pointer to device description, and the list of available events.
// Note that "indexed variables" in ROCM are read with eventname[%d], where %d is
// 0 to #instances-1. This is what we store in the EventID.name element. But the PAPI name
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
    char name[ROCM_MAX_STR_LEN];
    char description[2*ROCM_MAX_STR_LEN];
} ev_name_desc_t;

/* Control structure tracks array of active contexts, records active events and their values */
typedef struct _rocm_control {
    uint32_t countOfActiveContexts;
    struct _rocm_active_context_s *arrayOfActiveContexts[ROCM_MAX_COUNTERS];
    uint32_t activeEventCount;
    int activeEventIndex[ROCM_MAX_COUNTERS];
    long long activeEventValues[ROCM_MAX_COUNTERS];
    uint64_t startTimestampNs;
    uint64_t readTimestampNs;
} _rocm_control_t;

/* For each active context, which ROCM events are being measured, context eventgroups containing events */
typedef struct _rocm_active_context_s {
    Context ctx;
    int deviceNum;
    uint32_t conEventsCount;
    EventID conEvents[ROCM_MAX_COUNTERS];
    int conEventIndex[ROCM_MAX_COUNTERS];
} _rocm_active_context_t;

// unlike PAPI counterpart, no argument.
static int _rocm_cleanup_eventset(void);

// GLOBALS
#define         EVENT_TABLE_SIZE 64
static int      global_num_native_events = 0;
static int      Number_Events_Requested=0;
static char     *MyEventTable[EVENT_TABLE_SIZE];
static int      RunKernel=1;     
static int      Verbose   = 0; 

static unsigned long buffer_size = 16*1024;
static unsigned long tpb = 64;

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

// ----------------------------------------------------------------------------
// Callback function to get the number of agents
static hsa_status_t _rocm_get_gpu_handle(hsa_agent_t agent, void* arg)
{
  _rocm_context_t * gctxt = (_rocm_context_t*) arg;

  hsa_device_type_t type;
        
  ROCM_CALL_CK(hsa_agent_get_info,(agent, HSA_AGENT_INFO_DEVICE, &type), {return(_status);});

  // Device is a GPU agent
  if (type == HSA_DEVICE_TYPE_GPU) {
    gctxt->availAgentSize += 1;
    gctxt->availAgentArray = (hsa_agent_t*) realloc(gctxt->availAgentArray, (gctxt->availAgentSize*sizeof(hsa_agent_t)));
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
      char ROCMname[ROCM_MAX_STR_LEN];

        if (instances > 1) {
            snprintf(ctx->availEventDesc[index].name,
                ROCM_MAX_STR_LEN, "%s:device=%d:instance=%d",       // What PAPI user sees.
                info.metric.name, eventDeviceNum, ui);
            snprintf(ROCMname, ROCM_MAX_STR_LEN, "%s[%d]",
                info.metric.name, ui);                               // use indexed version.
        } else {
            snprintf(ctx->availEventDesc[index].name,
                ROCM_MAX_STR_LEN, "%s:device=%d",                   // What PAPI user sees.
                info.metric.name, eventDeviceNum);
            snprintf(ROCMname, ROCM_MAX_STR_LEN, "%s",
                info.metric.name);                                  // use non-indexed version.
        }

        ROCMname[ROCM_MAX_STR_LEN - 1] = '\0';                      // ensure z-terminated.
        strncpy(ctx->availEventDesc[index].description, info.metric.description, 2*ROCM_MAX_STR_LEN);
        ctx->availEventDesc[index].description[(2*ROCM_MAX_STR_LEN) - 1] = '\0';   // ensure z-terminated.

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

//-----------------------------------------------------------------------------
// The routines below are modified versions of the PAPI Component routines in
// linux-rocm.c, with PAPI interfaces removed or replaced. If those are
// changed, these should be changed to match. The easiest way to do that is
// copying the group of routines to different files and doing a DIFF.
//-----------------------------------------------------------------------------

// Note: The full NativeInfo_t is defined in papi_internal.h, this is just the
// fields we need (and all that needs to be filled out).
typedef struct _NativeInfo {
   int ni_event;                // index into our event array. 
   int ni_position;             // index into the results array.
} NativeInfo_t;

//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_start
// We don't inline this code, to make it easier to use in the do_work area.
// For ROCM component, switch to each context and start all eventgroups.
//-----------------------------------------------------------------------------
static int _rocm_start(void)
{
    printf_v2("Entering _rocm_start\n");

    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t ii, cc;

    printf_v2("Reset all active event values\n");
    for(ii = 0; ii < gctrl->activeEventCount; ii++)
        gctrl->activeEventValues[ii] = 0;

    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->startTimestampNs), return (ROCM_SA_FAIL));
    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // suppress "not used" error when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        printf_v2("Start device %d ctx %p ts %lu\n", eventDeviceNum, eventCtx, gctrl->startTimestampNs);
        if (eventCtx == NULL) abort();
        ROCP_CALL_CK(rocprofiler_start, (eventCtx, 0), return (ROCM_SA_FAIL));
    }

    return (ROCM_SA_OK);
}


//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_read 
// We don't inline this code, to make it easier to use in the do_work area.
// For ROCM component, switch to each context, read all the eventgroups, and 
// put the values in the correct places.
//-----------------------------------------------------------------------------
static int _rocm_read(long long **values)
{
    printf_v2("Entering _rocm_read\n");

    _rocm_control_t *gctrl = global__rocm_control;
    _rocm_context_t *gctxt = global__rocm_context;
    uint32_t cc, jj, ee;

    // Get read time stamp
    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &gctrl->readTimestampNs), return (ROCM_SA_FAIL));
    uint64_t durationNs = gctrl->readTimestampNs - gctrl->startTimestampNs;
    (void) durationNs;                                                  // Suppress 'not used' warning when not debug.
    gctrl->startTimestampNs = gctrl->readTimestampNs;


    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        printf_v2("Read device %d ctx %p(%u) ts %lu\n", eventDeviceNum, eventCtx, cc, gctrl->readTimestampNs);
        ROCP_CALL_CK(rocprofiler_read, (eventCtx, 0), return (ROCM_SA_FAIL));
        printf_v2("waiting for data\n");
        ROCP_CALL_CK(rocprofiler_get_data, (eventCtx, 0), return (ROCM_SA_FAIL));
        ROCP_CALL_CK(rocprofiler_get_metrics, (eventCtx), return (ROCM_SA_FAIL));
        printf_v2("done\n");

        for(jj = 0; jj < gctrl->activeEventCount; jj++) {
            int index = gctrl->activeEventIndex[jj];
            EventID eventId = gctxt->availEventIDArray[index];
            printf_v2("jj=%i of %i, index=%i, device#=%i.\n", jj, gctrl->activeEventCount, index, gctxt->availEventDeviceNum[index]);
            (void) eventId;                                             // Suppress 'not used' warning when not debug.

            /* If the device/context does not match the current context, move to next */
            if(gctxt->availEventDeviceNum[index] != eventDeviceNum)
                continue;

            for(ee = 0; ee < gctrl->arrayOfActiveContexts[cc]->conEventsCount; ee++) {
                printf_v2("Searching for activeEvent %s in Activecontext %u eventIndex %d duration %lu\n", eventId.name, ee, index, durationNs);
                if (gctrl->arrayOfActiveContexts[cc]->conEventIndex[ee] == index) {
                  gctrl->activeEventValues[jj] = gctrl->arrayOfActiveContexts[cc]->conEvents[ee].data.result_int64;
                  printf_v2("Matched event %d:%d eventName %s value %lld\n", jj, index, eventId.name, gctrl->activeEventValues[jj]);
                  break;
                }
            }
        }
    }

    *values = gctrl->activeEventValues;
    return (ROCM_SA_OK);
}


//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_stop
// We don't inline this code, to make it easier to use in the do_work area.
//-----------------------------------------------------------------------------
static int _rocm_stop(void)
{
    printf_v2("Entering _rocm_stop\n");

    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                                          // Suppress 'not used' warning when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        printf_v2("Stop device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCP_CALL_CK(rocprofiler_stop, (eventCtx, 0), return (ROCM_SA_FAIL));
    }

    return (ROCM_SA_OK);
} // END ROUTINE.

//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_cleanup_eventset
// We don't inline this code, to make it easier to use in the do_work area.
// Disable and destroy the ROCM eventGroup.
//-----------------------------------------------------------------------------
static int _rocm_cleanup_eventset(void)
{
    printf_v2("Entering _rocm_cleanup_eventset\n");
//  fprintf(stderr, "%s:%i _rocm_cleanup_eventset called.\n", __FILE__, __LINE__);

    _rocm_control_t *gctrl = global__rocm_control;
    uint32_t i, cc;

    for(cc = 0; cc < gctrl->countOfActiveContexts; cc++) {
        int eventDeviceNum = gctrl->arrayOfActiveContexts[cc]->deviceNum;
        (void) eventDeviceNum;                      // Suppress 'not used' warning when not debug.
        Context eventCtx = gctrl->arrayOfActiveContexts[cc]->ctx;
        printf_v2("Destroy device %d ctx %p \n", eventDeviceNum, eventCtx);
        ROCP_CALL_CK(rocprofiler_close, (eventCtx), return (ROCM_SA_FAIL));
        free( gctrl->arrayOfActiveContexts[cc] );
    }
    if (global__ctx_properties.queue != NULL) {
      ROCM_CALL_CK(hsa_queue_destroy, (global__ctx_properties.queue), return (ROCM_SA_FAIL));
      global__ctx_properties.queue = NULL;
    }
    // Record that there are no active contexts or events
    gctrl->countOfActiveContexts = 0;
    gctrl->activeEventCount = 0;

    // Clear all indicators of event being measured.
    _rocm_context_t *gctxt = global__rocm_context;
    for (i=0; i<maxEventSize; i++) {
            gctxt->availEventIsBeingMeasuredInEventset[i] = 0;
    }

    return (ROCM_SA_OK);
}


void helpText(void) {
    fprintf(stderr, "This utility reads events from an AMD GPU using the same API as the ROCM  \n");
    fprintf(stderr, "component, but without the use of the PAPI infrastructure. The routines   \n");
    fprintf(stderr, "used are modified versions of the linux-rocm.c component code; but without\n");
    fprintf(stderr, "eventSets or other PAPI control structures.                               \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "Events should be the same names as used in PAPI, **except** leave off the \n");
    fprintf(stderr, "leading 'rocm:::'. Thus, not 'rocm:::SQ_INSTS_LDS:device=0', just         \n");
    fprintf(stderr, "'SQ_INSTS_LDS:device=0'. These names can be seen in the output of the     \n");
    fprintf(stderr, "utility papi/src/utils/papi_native_avail. Each event to be read should be \n");
    fprintf(stderr, "a command line argument. At least 1 must be specified, the maximum is %d. \n", EVENT_TABLE_SIZE);
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "In addition, any argument beginning with -- will not be used as an event, \n");
    fprintf(stderr, "but interpreted as a control, below the controls are detailed. An argument\n");
    fprintf(stderr, "of 'help' or starting with '-h' will be considered a plea for help and    \n");
    fprintf(stderr, "also ignored as an event.                                                 \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "An example of AMD Kernel code is included, executed by the function       \n");
    fprintf(stderr, "do_work(). These can be replaced by user code to test their specific      \n");
    fprintf(stderr, "application.                                                              \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "No controls are required, the following are available. Default in [].     \n");
    fprintf(stderr, "--verbose=#      [0]   0=normal output only, 1=Milestones, 2=Every Step.  \n");
    fprintf(stderr, "help | --help | -h*    This text. Any argument beginning '-h'.            \n");
    fprintf(stderr, "--runKernel=#    [1]   Times to run 'do_work() function; >=0.             \n");
    fprintf(stderr, "--bufferSize=# [16384] Example of user data if modified. In this case, the\n");
    fprintf(stderr, "                       provided kernel does pointer chasing to find cache \n");
    fprintf(stderr, "                       boundaries, this is the # of pointers in the chain.\n");
    fprintf(stderr, "                       bash shell compute example: $((16*1024*1024))      \n");
}; 

void parseArgs(int argc, char **argv) {
    int i, n;
    if (argc < 1) {
        fprintf(stderr, "ERROR: Cannot run without command line arguments.\n");
        helpText();
        exit(-1);
    }

    for (i=1; i<argc; i++) {

        // USER COMMAND LINE DATA EXAMPLE: Collect an INT for --bufferSize=#
        if (strncmp("--bufferSize=", argv[i], 13) == 0) {
            n = atoi(argv[i]+13);
            if (n < 0) {
                fprintf(stderr, "--bufferSize cannot be negative.\n");
                exit(-1);
            }
            
            buffer_size = n;
            continue;
        }


        if (strncmp("--verbose=", argv[i], 10) == 0) {
            n = atoi(argv[i]+10);
            if (n < 0 || n>2) {
                fprintf(stderr, "--verbose must be 0, 1 or 2.\n");
                exit(-1);
            }
            
            Verbose = n;
            continue;
        }

       if (strncmp("--runKernel=", argv[i], 12) == 0) {
            n = atoi(argv[i]+12);
            if (n < 0) {
                fprintf(stderr, "--runKernel must be >=0.\n");
                exit(-1);
            }
            
            RunKernel = n;
            continue;
        }

       if (strncmp("--help", argv[i], 6) == 0 ||
           strncmp("-h", argv[i], 2) == 0     ||
           strncmp("help", argv[i], 4)   == 0) {
            helpText();
            exit(-1);
        }

        // if it wasn't above, it must be an event to request.
        if (Number_Events_Requested >= EVENT_TABLE_SIZE) {
            fprintf(stderr, "ERROR: Maximum number of events is %d, Event '%s' exceeds that.\n",
                EVENT_TABLE_SIZE, argv[i]);
            exit(-1);
        }

        MyEventTable[Number_Events_Requested++] = argv[i];
    } // end loop.

} // end parseArgs.

//-----------------------------------------------------------------------------
// Main function. 
//-----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int i,j;
    uint32_t ui;

    parseArgs(argc, argv);
    printf_v1("Number Events Requested=%d, RunKernel=%i, Verbose=%i\n", 
        Number_Events_Requested, RunKernel, Verbose);
    if (Number_Events_Requested < 1) {
        fprintf(stderr, "At least one event must be specified.\n");
        exit(-1);
    }

    printf_v1("Requested Events:\n");
    for (i=0; i<Number_Events_Requested; i++) {
        printf_v1("%d: '%s'\n", i+1, MyEventTable[i]);
    }
 
    //-------------------------------------------------------------------------
    // Similar component code: linux-rocm.c:_rocm_linkRocmLibraries().
    // Ensure ROCPROFILER environment variables are set.
    //-------------------------------------------------------------------------
    if (getenv("ROCP_METRICS") == NULL) {
        fprintf(stderr, "%s:%s:%i Env. Var. ROCP_METRICS not set; rocprofiler is not configured.",
        __FILE__, __func__, __LINE__);
        exit(-1);                       // Wouldn't have any events.
    }

    if (getenv("ROCPROFILER_LOG") == NULL) {
        fprintf(stderr, "%s:%s:%i Env. Var. ROCPROFILER_LOG not set; rocprofiler is not configured.",
        __FILE__, __func__, __LINE__);
        exit(-1);                       // Wouldn't have any events.
    }

    if (getenv("HSA_VEN_AMD_AQLPROFILE_LOG") == NULL) {
        fprintf(stderr, "%s:%s:%i Env. Var. HSA_VEN_AMD_AQLPROFILE_LOG not set; rocprofiler is not configured.",
        __FILE__, __func__, __LINE__);
        exit(-1);                       // Wouldn't have any events.
    }

    if (getenv("AQLPROFILE_READ_API") == NULL) {
        fprintf(stderr, "%s:%s:%i Env. Var.AQLPROFILE_READ_API not set; rocprofiler is not configured.",
        __FILE__, __func__, __LINE__);
        exit(-1);                       // Wouldn't have any events.
    }

    //-------------------------------------------------------------------------
    // Similar component code: linux-rocm.c:_rocm_init_component()
    //-------------------------------------------------------------------------
    // Initialize hardware counters, setup the function vector table and get
    // hardware information.
    //-------------------------------------------------------------------------
    hsa_status_t status;
    status = (hsa_init)();
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        fprintf(stderr, "%s:%s:%i ROCM hsa_init() failed with error %d.", 
        __FILE__, __func__, __LINE__, status);
        exit(-2);
    }
        
    /* Create the structure */
    global__rocm_context = (_rocm_context_t *) calloc(1, sizeof(_rocm_context_t));

    /* Get GPU agent */
    status = (hsa_iterate_agents)(_rocm_get_gpu_handle, global__rocm_context);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        fprintf(stderr, "%s:%s:%i ROCM hsa_iterate_agents() failed with error %d.",
        __FILE__, __func__, __LINE__, status);
        exit(-2);
    }

    //-------------------------------------------------------------------------
    // Similar component code: linux-rocm.c:_rocm_add_native_events()
    // Builds a list of all native ROCM events supported. 
    //-------------------------------------------------------------------------

    // Count all events in all agents; Each element of 'indexed' metrics is considered a separate event.
    // NOTE: The environment variable ROCP_METRICS should point at a path and file like metrics.xml.
    //       If that file doesn't exist, this iterate info fails with a general error (0x1000).
    // NOTE: We are *accumulating* into maxEventSize.
    for (ui = 0; ui < global__rocm_context->availAgentSize; ui++) {
        ROCP_CALL_CK(rocprofiler_iterate_info, (&(global__rocm_context->availAgentArray[ui]), ROCPROFILER_INFO_KIND_METRIC,
            _rocm_count_native_events_callback, (void*)(&maxEventSize)), exit(-3));
    }

    // Allocate space for all events and descriptors, includes space for instances.
    global__rocm_context->availEventDeviceNum = (int *) calloc(maxEventSize, sizeof(int));
    CHECK_PRINT_EVAL((global__rocm_context->availEventDeviceNum == NULL), "ERROR ROCM: Could not allocate memory", exit(-3));
    global__rocm_context->availEventIDArray = (EventID *) calloc(maxEventSize, sizeof(EventID));
    CHECK_PRINT_EVAL((global__rocm_context->availEventIDArray == NULL), "ERROR ROCM: Could not allocate memory", exit(-3));
    global__rocm_context->availEventIsBeingMeasuredInEventset = (uint32_t *) calloc(maxEventSize, sizeof(uint32_t));
    CHECK_PRINT_EVAL((global__rocm_context->availEventIsBeingMeasuredInEventset == NULL), "ERROR ROCM: Could not allocate memory", exit(-3));
    global__rocm_context->availEventDesc = (ev_name_desc_t *) calloc(maxEventSize, sizeof(ev_name_desc_t));
    CHECK_PRINT_EVAL((global__rocm_context->availEventDesc == NULL), "ERROR ROCM: Could not allocate memory", exit(-3));

    for (ui = 0; ui < global__rocm_context->availAgentSize; ++ui) {
        events_callback_arg_t arg;
        arg.device_num = ui;
        arg.count = maxEventSize;
        arg.ctx = global__rocm_context;
        ROCP_CALL_CK(rocprofiler_iterate_info, (&(global__rocm_context->availAgentArray[ui]), ROCPROFILER_INFO_KIND_METRIC,
            _rocm_add_native_events_callback, (void*)(&arg)), exit(-3));
    }

    // Set a global shortcut for number of native events.
    global_num_native_events = global__rocm_context->availEventSize;

    if (global_num_native_events == 0) {
        fprintf(stderr, "%s:%s:%i No events.  Ensure ROCP_METRICS=%s points at a valid metrics.xml file.",
        __FILE__, __func__, __LINE__, getenv("ROCP_METRICS"));
        exit(-1);
    }

    printf_v2("ROCM init component succeeded. Native Events Found: %d.\n",
        global_num_native_events); 

    //-------------------------------------------------------------------------
    // Similar component code: linux-rocm.c:_rocm_init_control_state()
    // Allocates and initializes a structure to keep track of events.
    //-------------------------------------------------------------------------
    global__rocm_control = (_rocm_control_t *) calloc(1, sizeof(_rocm_control_t));
    global__rocm_control->countOfActiveContexts = 0;
    global__rocm_control->activeEventCount = 0;
    
    printf_v2("ROCM init control state succeeded.\n");

    //-------------------------------------------------------------------------
    // Find the events in the table, construct a NativeInfo array.
    //-------------------------------------------------------------------------
    NativeInfo_t* MyNI = (NativeInfo_t*) calloc(Number_Events_Requested, sizeof(NativeInfo_t));
    _rocm_context_t *gctx = global__rocm_context;
    int eventsFound = 0;
    
    for (i=0; i<Number_Events_Requested; i++) {
        for (j=0; j<global_num_native_events; j++) {
//          fprintf(stderr, "Comparing to '%s'.\n", gctx->availEventDesc[j].name);
            if (strcmp(MyEventTable[i], gctx->availEventDesc[j].name) == 0) break;
        }
        
        if (j == global_num_native_events) {
            fprintf(stderr, "Specified event #%d ('%s') is not a recognized event.\n",
                i+1, MyEventTable[i]);
        } else {
            MyNI[i].ni_event = j;
            MyNI[i].ni_position = i;
            eventsFound++;
        }
    }

    if (eventsFound == Number_Events_Requested) {
        printf_v2("All requested Events were recognized.\n");
    } else {
        fprintf(stderr, "%d of %d requested Events were not recognized; aborting.\n",
            (Number_Events_Requested-eventsFound), Number_Events_Requested);
        goto Shutdown;
    }       

    //-------------------------------------------------------------------------
    // Similar code in linux-rocm.c:_rocm_update_control_state.
    //-------------------------------------------------------------------------
    for(i = 0; i < Number_Events_Requested; i++) {
        int NI_idx, eventContextIdx;
        uint32_t numPasses;
        NI_idx = MyNI[i].ni_event;
        char *eventName = global__rocm_context->availEventDesc[NI_idx].name;
        (void) eventName;           // Suppress warning if eventName is not used (e.g. may only be in debug code.)
        int eventDeviceNum = global__rocm_context->availEventDeviceNum[NI_idx];

        /* if this event is already added continue to next i, if not, mark it as being added */
        if(global__rocm_context->availEventIsBeingMeasuredInEventset[NI_idx] == 1) {
            printf_v2("Skipping event %s (%i of %i) which is already added\n", eventName, i, Number_Events_Requested);
            continue;
        } else {
            global__rocm_context->availEventIsBeingMeasuredInEventset[NI_idx] = 1;
        }

        // Find context/control in papirocm, creating it if does not exist.
        for(ui = 0; ui < global__rocm_control->countOfActiveContexts; ui++) {
            CHECK_PRINT_EVAL(ui >= ROCM_MAX_COUNTERS, "Exceeded hardcoded maximum number of contexts (ROCM_MAX_COUNTERS)", goto Shutdown);
            if(global__rocm_control->arrayOfActiveContexts[ui]->deviceNum == eventDeviceNum) {
                break;
            }
        }
        // Create context if it does not exist
        if(ui == global__rocm_control->countOfActiveContexts) {
            printf_v2("Event %s device %d does not have a ctx registered yet...\n", eventName, eventDeviceNum);
            global__rocm_control->arrayOfActiveContexts[ui] = (struct _rocm_active_context_s*) calloc(1, sizeof(_rocm_active_context_t));
            CHECK_PRINT_EVAL(global__rocm_control->arrayOfActiveContexts[ui] == NULL, "Memory allocation for new active context failed", goto Shutdown);
            global__rocm_control->arrayOfActiveContexts[ui]->deviceNum = eventDeviceNum;
            global__rocm_control->arrayOfActiveContexts[ui]->ctx = NULL;
            global__rocm_control->arrayOfActiveContexts[ui]->conEventsCount = 0;
            global__rocm_control->countOfActiveContexts++;
            printf_v2("Added a new context deviceNum %d ... now countOfActiveContexts is %d\n", eventDeviceNum, global__rocm_control->countOfActiveContexts);
        }
        eventContextIdx = ui;

        _rocm_active_context_t *eventctrl = global__rocm_control->arrayOfActiveContexts[eventContextIdx];
        printf_v2("Need to add event %d %s to the context\n", NI_idx, eventName);
        // Now we have eventctrl, we can check on max event count.
        if (eventctrl->conEventsCount >= ROCM_MAX_COUNTERS) {
            fprintf(stderr, "Error: Num events exceeded ROCM_MAX_COUNTERS\n");
            goto Shutdown; 
        }

        /* lookup eventid for this event NI_idx */
        EventID eventId = global__rocm_context->availEventIDArray[NI_idx];
        printf_v2("eventId.name='%s', stored in eventctrl->conEvents[%i], conEventsIndex[%i]=%i.\n", 
            eventId.name, eventctrl->conEventsCount, eventctrl->conEventsCount, NI_idx);
        eventctrl->conEvents[eventctrl->conEventsCount] = eventId;
        eventctrl->conEventIndex[eventctrl->conEventsCount] = NI_idx;
        eventctrl->conEventsCount++;
//      fprintf(stderr, "%s:%d Added eventId.name='%s' as conEventsCount=%i with NI_idx=%i.\n", __FILE__, __LINE__, eventId.name, eventctrl->conEventsCount-1, NI_idx); // test indexed events.

        /* Record index of this active event back into the MyNI structure */
        MyNI[i].ni_position = global__rocm_control->activeEventCount;
        /* record added event at the higher level */
        CHECK_PRINT_EVAL(global__rocm_control->activeEventCount == ROCM_MAX_COUNTERS - 1, "Exceeded maximum num of events (ROCM_MAX_COUNTERS)", goto Shutdown);

        global__rocm_control->activeEventIndex[global__rocm_control->activeEventCount] = NI_idx;
        global__rocm_control->activeEventValues[global__rocm_control->activeEventCount] = 0;
        global__rocm_control->activeEventCount++;

        /* Create/recreate eventgrouppass structures for the added event and context */
        printf_v2("Create eventGroupPasses for context (destroy pre-existing) (Number_Events_Requested %d, conEventsCount %d) \n", global__rocm_control->activeEventCount, eventctrl->conEventsCount);
        if(eventctrl->conEventsCount > 0) {
            if (eventctrl->ctx != NULL) {
                ROCP_CALL_CK(rocprofiler_close, (eventctrl->ctx), goto Shutdown);
            }
            int openFailed=0;
            printf_v2("%s:%i calling rocprofiler_open, i=%i device=%i numEvents=%i name='%s'.\n",
                 __FILE__, __LINE__, i, eventDeviceNum, eventctrl->conEventsCount, eventId.name);
            const uint32_t mode = (global__ctx_properties.queue != NULL) ? 
                ROCPROFILER_MODE_STANDALONE : ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE;
            ROCP_CALL_CK(rocprofiler_open, (global__rocm_context->availAgentArray[eventDeviceNum], 
                eventctrl->conEvents, eventctrl->conEventsCount, &(eventctrl->ctx),
                mode, &global__ctx_properties), openFailed=1);
            if (openFailed) {
                fprintf(stderr, "Error: The ROCM event '%s' was not accepted by the ROCPROFILER.\n", eventId.name);
                goto Shutdown;
            }

            ROCP_CALL_CK(rocprofiler_group_count, (eventctrl->ctx, &numPasses), goto Shutdown);

            if (numPasses > 1) {
                fprintf(stderr, "Error: The combined ROCM events require more than 1 pass.\n");
                goto Shutdown;
            } else  {
                printf_v2("Created eventGroupPasses for context total-events %d in-this-context %d passes-required %d) \n", global__rocm_control->activeEventCount, eventctrl->conEventsCount, numPasses);
            }
        }
    }

    //-------------------------------------------------------------------------
    // Run the work kernel as often as user requested.
    //-------------------------------------------------------------------------
    for (i=0; i<RunKernel; i++) {
        do_work();
    }

    //-------------------------------------------------------------------------
    // Cleanup and Exit.
    //-------------------------------------------------------------------------
Shutdown: 
    i = _rocm_cleanup_eventset();
    if (i < ROCM_SA_OK) {
        fprintf(stderr, "_rocm_cleanup_eventset failed.\n");
    }
    
    if (MyNI != NULL) free(MyNI);

    //-------------------------------------------------------------------------
    // Similar code: linux-rocm.c:_rocm_shutdown_component
    //-------------------------------------------------------------------------

    // Free context
    if(global__rocm_context != NULL) {
        free(global__rocm_context->availEventIDArray);
        free(global__rocm_context->availEventDeviceNum);
        free(global__rocm_context->availEventIsBeingMeasuredInEventset);
        free(global__rocm_context->availEventDesc);
        free(global__rocm_context);
        global__rocm_context = NULL;
    }

    /* Free control */
    if(global__rocm_control != NULL) {
        for(ui = 0; ui < global__rocm_control->countOfActiveContexts; ui++) {
            if(global__rocm_control->arrayOfActiveContexts[ui] != NULL) {
                free(global__rocm_control->arrayOfActiveContexts[ui]);
            }
        }

        free(global__rocm_control);
        global__rocm_control = NULL;
    }

    // Shutdown ROC runtime. If this fails, macro prints the error.
    ROCM_CALL_CK(hsa_shut_down, (), status=_status);
    if (status == HSA_STATUS_SUCCESS) {
        printf_v2("ROCM shutdown component succeeded.\n");
    } 

    return(0);
} // END MAIN.




//-----------------------------------------------------------------------------
// Below is an example kernel, to execute pointer chasing.
//-----------------------------------------------------------------------------

__global__
void chase(unsigned long* dst, unsigned long *src, unsigned long N, int print_chain) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tpb = blockDim.x;
    unsigned long idx, start_time, end_time, max_repeat;

    idx = src[tid];

    if( print_chain ){
        dst[idx] = src[idx];
        for(unsigned long j=0; j<(N/tpb); j++){
            idx = src[idx];
            dst[idx] = src[idx];
        }
        return;
    }

    max_repeat = 1024*1024;

    __asm__ volatile ("s_memrealtime %0\n s_waitcnt lgkmcnt(0)" : "=s" (start_time) );


    for(unsigned long j=0; j<max_repeat; j+=16){
        // Have a side-effect so the compiler does not throw our code away.
        if( !(j%(31*16)) )
            dst[idx] = src[idx];

        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];

        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
    }

    __asm__ volatile ("s_memrealtime %0\n s_waitcnt lgkmcnt(0)" : "=s" (end_time) );

    dst[0] = max_repeat*tpb;
    dst[1+tid] = start_time;
    dst[1+tpb+tid] = end_time;
}


//-----------------------------------------------------------------------------
// Below is 'do_work' which sets up data on the device and on the host, then
// starts the event counters, calls the kernel, and reads and reports the 
// event counters.
//-----------------------------------------------------------------------------

void do_work(void){
    unsigned long stride;
    unsigned long *avail, *h_array, *d_res, *d_inp;
    int i;


    stride = tpb;

    // Allocate a few more elements to send meta-data out of the GPU
    auto status = hipMalloc((void**)&d_res, (buffer_size+32)*sizeof(unsigned long));
    if(status != hipSuccess){
        fprintf(stderr,"Can't allocate DEVICE memory\n");
        exit(-1);
    }

    status = hipMalloc((void**)&d_inp, buffer_size*sizeof(unsigned long));
    if(status != hipSuccess){
        fprintf(stderr,"Can't allocate DEVICE memory\n");
        exit(-1);
    }

    // Allocate a few more elements to send meta-data out of the GPU
    status = hipHostMalloc((void**)&h_array, (buffer_size+32)*sizeof(unsigned long), 0);
    if(status != hipSuccess){
        fprintf(stderr,"Can't allocate HOST memory\n");
        exit(-1);
    }

    avail = (unsigned long *)malloc( buffer_size/stride*sizeof(unsigned long));

// Initialize matrix
    for(unsigned long i=0; i<buffer_size/stride; i++){
        avail[i] = i;
    }

    unsigned long remainingElemCnt = buffer_size/stride;
    unsigned long currIndex=0;

    for(unsigned long elemCnt=0; elemCnt<buffer_size/stride-1; ++elemCnt){
        // We add 1 (and subtract 1 from the modulo divisor) because the first
        //element (0) is the currIndex in the first iteration, so it can't be in
        // the list of available elements.
        unsigned long index = 1+random() % (remainingElemCnt-1);
        unsigned long uniqIndex = stride*avail[index];
        // replace the chosen number with the last element.
        avail[index] = avail[remainingElemCnt-1];
        // shrink the effective array size so the last element "drops off".
        remainingElemCnt--;

        for(unsigned long j=0; j<tpb; j++)
            h_array[currIndex+j] = uniqIndex+j;

        currIndex = uniqIndex;
    }

    // close the circle by making the last element(s) point to the zero-th element(s)
    for(unsigned long j=0; j<tpb; j++)
        h_array[currIndex+j] = 0+j;

    // We have allocate a few more elements to send meta-data out of the GPU,
    // but they don't contain any values, so we don't need to copy them.
    status = hipMemcpyHtoD(d_inp, h_array, buffer_size*sizeof(unsigned long));
    if(status != hipSuccess){
        fprintf(stderr,"Can't copy memory to device\n");
        exit(-1);
    }

    //-------------------------------------------------------------------------
    // Start the event set we have built.
    //-------------------------------------------------------------------------
    i = _rocm_start();
    if (i < ROCM_SA_OK) {
        fprintf(stderr, "_rocm_start failed; aborting.\n");
        return;
    }
   
    printf_v2("rocm_start was successful.\n"); 
  

//////////////////////////////////////////////////////////////////////////////// 
/// KERNEL START
    int num_blocks = 1;

    hipLaunchKernelGGL(chase, dim3(num_blocks), dim3(tpb), 0, 0, d_res, d_inp, buffer_size, 0);

    hipDeviceSynchronize();

/// KERNEL END
//////////////////////////////////////////////////////////////////////////////// 

    //-------------------------------------------------------------------------
    // Read the counters.
    //-------------------------------------------------------------------------
    long long *dataread = NULL;
    i = _rocm_read(&dataread);
    if (i < ROCM_SA_OK) {
        fprintf(stderr, "_rocm_start failed; aborting.\n");
        return;
    }
       
    printf_v2("rocm_read was successful, dataread ptr = %p.\n", dataread);
    for (i=0; i<Number_Events_Requested; i++) {
        printf("Result %6lli for Event #%d '%s'.\n", dataread[i], i, MyEventTable[i]);
    }

    //-------------------------------------------------------------------------
    // Stop the counters. 
    //-------------------------------------------------------------------------
    i = _rocm_stop();
    if (i < ROCM_SA_OK) {
        fprintf(stderr, "_rocm_stop failed; aborting.\n");
        return;
    }

    printf_v2("rocm_stop was successful.\n"); 

    status = hipMemcpy(h_array, d_res, (buffer_size+32)*sizeof(unsigned long), hipMemcpyDeviceToHost);
    if(status != hipSuccess){
        fprintf(stderr,"Can't copy memory to HOST\n");
        exit(-1);
    }

    unsigned long minT, maxT, dt;

    minT = 0;
    for(unsigned long i=0; i<tpb; i++){
        auto t = h_array[1+i];
        if( 0 == minT || t < minT )
            minT = t;
    }

    maxT = 0;
    for(unsigned long i=0; i<tpb; i++){
        auto t = h_array[1+tpb+i];
        if( t > maxT )
            maxT = t;
    }

    dt = maxT-minT;

    printf("%8lu %9lu %lf",buffer_size, dt, dt/((double)h_array[0]));

    printf("\n");

    // Cleanup, so we can run it again.
    HIPCHECK(hipFree(d_res));
    HIPCHECK(hipFree(d_inp));
    HIPCHECK(hipHostFree(h_array)); 
} // end do_work

