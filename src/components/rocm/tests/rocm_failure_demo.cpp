//-----------------------------------------------------------------------------
// This code will uses the API implemented in linux-rocm.c, but without any
// dependence on the rest of PAPI. The main() function will read command 
// line events.
// Further this is compiled with ROCM_SA_Makefile using hipcc; so a HIP kernel 
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

//*****************************************************************************
// The one and only event we will add, in AMD ROCPROFILER format; for example,
// if the event is indexed use square brackets: TA_TA_BUSY[0], TCC_HIT[0].
// The device can be changed as well.
static char     MainEvent[]="Wavefronts";
static uint32_t MainDevice = 0;
static int      MainFound = 0;      // Set to 1 if valid event.
//*****************************************************************************
static int      Verbose   = 2; // 0=Just necessary output, 1=Milestones, 2=fine step details.

// this number assumes that there will never be more events than indicated
#define ROCM_MAX_COUNTERS 512

// Defines for some things usually defined by PAPI.
#define ROCM_MAX_STR_LEN (1024)
#define ROCM_SA_ECOMBO  (-4)
#define ROCM_SA_INVALID (-3)
#define ROCM_SA_ENOMEM (-2)
#define ROCM_SA_FAIL (-1)
#define ROCM_SA_OK   (0)

#define BUF_SIZE (64 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                     \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define printf_v1(format, args...) {if (Verbose>0) {printf(format, ## args); fflush(stdout);}}
#define printf_v2(format, args...) {if (Verbose>1) {printf(format, ## args); fflush(stdout);}}
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

// GLOBALS
static EventID  MainEventId; 

static unsigned long buffer_size = BUF_SIZE;
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
// Callback function to ensure given event is valid.
static hsa_status_t _rocm_search_native_events_callback(const rocprofiler_info_data_t info, void * arg)
{
    events_callback_arg_t * callback_arg = (events_callback_arg_t*) arg;
    const uint32_t eventDeviceNum = callback_arg->device_num;

//  information about AMD Event.
    #if 0
    printf_v2("%s:%i name=%s block_name=%s, instances=%i block_counters=%i.\n",
       __FILE__, __LINE__, info.metric.name, info.metric.block_name, info.metric.instances,
       info.metric.block_counters);
    #endif

    if (strcmp(MainEvent, info.metric.name) == 0 && 
        eventDeviceNum == MainDevice) {
        MainFound=1;
        MainEventId.kind = ROCPROFILER_FEATURE_KIND_METRIC;
        MainEventId.name = MainEvent;                                   // what ROCM needs to see.
        MainEventId.parameters = NULL;                                  // Not currently used, but init for safety.
        MainEventId.parameter_count=0;                                  // Not currently used, but init for safety.
    }        

    return HSA_STATUS_SUCCESS;
} // end CALLBACK, _rocm_add_native_events_callback

//-----------------------------------------------------------------------------
// The routines below are modified versions of the PAPI Component routines in
// linux-rocm.c, with PAPI interfaces removed or replaced. If those are
// changed, these should be changed to match. The easiest way to do that is
// copying the group of routines to different files and doing a DIFF.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_start
// We don't inline this code, to make it easier to use in the do_work area.
// For ROCM component, switch to each context and start all eventgroups.
//-----------------------------------------------------------------------------
static int _rocm_start(void)
{
    printf_v2("Entering _rocm_start\n");
    hsa_status_t status=HSA_STATUS_SUCCESS;

    printf_v2("_rocm_start: Reset all active event values, activeEventCount=%d.\n", global__rocm_control->activeEventCount);
    // There is only one event.
    global__rocm_control->activeEventValues[0] = 0;

    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &global__rocm_control->startTimestampNs), status=_status);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed on hsa_system_get_info.\n");
        exit(-1);
    }

    // there is only one context (and one event).
    int eventDeviceNum = global__rocm_control->arrayOfActiveContexts[0]->deviceNum;
    printf_v2("_rocm_start: eventDeviceNum=%d.\n", eventDeviceNum);
    Context eventCtx = global__rocm_control->arrayOfActiveContexts[0]->ctx;
    printf_v2("_rocm_start: Start device %d ctx %p ts %lu\n", eventDeviceNum, eventCtx, global__rocm_control->startTimestampNs);
    if (eventCtx == NULL) {
        fprintf(stderr,"_rocm_start: eventCtx is NULL.\n");
        exit(-1);
    }

    ROCP_CALL_CK(rocprofiler_start, (eventCtx, 0), status=_status);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "_rocm_start: Failed on rocprofiler_start.\n");
        exit(-1);
    }

    printf_v2("_rocm_start successful.\n");
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
    char *data_kind[] = {"UNINIT","UINT32","UINT64","FLOAT","DOUBLE","BYTES"};

    // Get read time stamp
    ROCM_CALL_CK(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &global__rocm_control->readTimestampNs), return (ROCM_SA_FAIL));
    uint64_t durationNs = global__rocm_control->readTimestampNs - global__rocm_control->startTimestampNs;
    (void) durationNs;                                                  // Suppress 'not used' warning.
    global__rocm_control->startTimestampNs = global__rocm_control->readTimestampNs;

    // There is only one context.
    Context eventCtx = global__rocm_control->arrayOfActiveContexts[0]->ctx;
    printf_v2("_rocm_read: Read device=%d &ctx=%p timestamp (ns)=%lu\n", MainDevice, eventCtx, global__rocm_control->readTimestampNs);
    ROCP_CALL_CK(rocprofiler_read, (eventCtx, 0), return (ROCM_SA_FAIL));
    printf_v2("_rocm_read: waiting for data\n");
    ROCP_CALL_CK(rocprofiler_get_data, (eventCtx, 0), return (ROCM_SA_FAIL));
    ROCP_CALL_CK(rocprofiler_get_metrics, (eventCtx), return (ROCM_SA_FAIL));
    // There is only one event.
    global__rocm_control->activeEventValues[0] = global__rocm_control->arrayOfActiveContexts[0]->conEvents[0].data.result_int64;
    printf_v2("_rocm_read; wait done, data_kind='%s' value=%llu\n",
        data_kind[global__rocm_control->arrayOfActiveContexts[0]->conEvents[0].data.kind],
        global__rocm_control->arrayOfActiveContexts[0]->conEvents[0].data.result_int64);

    *values = global__rocm_control->activeEventValues;
    return (ROCM_SA_OK);
}


//-----------------------------------------------------------------------------
// Similar code:linux-rocm.c:_rocm_stop
// We don't inline this code, to make it easier to use in the do_work area.
//-----------------------------------------------------------------------------
static int _rocm_stop(void)
{
    printf_v2("Entering _rocm_stop\n");

    int eventDeviceNum = global__rocm_control->arrayOfActiveContexts[0]->deviceNum;
    Context eventCtx = global__rocm_control->arrayOfActiveContexts[0]->ctx;
    printf_v2("_rocm_stop: Stopping device %d ctx %p \n", eventDeviceNum, eventCtx);
    ROCP_CALL_CK(rocprofiler_stop, (eventCtx, 0), return (ROCM_SA_FAIL));

    return (ROCM_SA_OK);
} // END ROUTINE.

//-----------------------------------------------------------------------------
// Main function. 
//-----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    (void) argc;
    (void) argv;

    uint32_t ui;
    uint32_t numPasses;
    _rocm_active_context_t *eventctrl = NULL;

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
    for (ui = 0; ui < global__rocm_context->availAgentSize; ++ui) {
        events_callback_arg_t arg;
        arg.device_num = ui;
        arg.count = maxEventSize;
        arg.ctx = global__rocm_context;
        ROCP_CALL_CK(rocprofiler_iterate_info, (&(global__rocm_context->availAgentArray[ui]), ROCPROFILER_INFO_KIND_METRIC,
            _rocm_search_native_events_callback, (void*)(&arg)), exit(-3));
    }

    if (MainFound == 0) {
        fprintf(stderr, "%s:%s:%i Failed to find event '%s' in file ROCP_METRICS='%s'.",
        __FILE__, __func__, __LINE__, MainEvent, getenv("ROCP_METRICS"));
        exit(-1);
    }

    printf_v2("main: ROCM init component succeeded. Found Event '%s'.\n", MainEvent);

    //-------------------------------------------------------------------------
    // Similar component code: linux-rocm.c:_rocm_init_control_state()
    // Allocates and initializes a structure to keep track of events.
    //-------------------------------------------------------------------------
    global__rocm_control = (_rocm_control_t *) calloc(1, sizeof(_rocm_control_t));
    global__rocm_control->countOfActiveContexts = 0;
    global__rocm_control->activeEventCount = 0;
    
    printf_v2("main: ROCM init control state succeeded.\n");


    if (MainFound == 0) {
        fprintf(stderr, "The MainEvent '%s' was not recognized; aborting.\n", MainEvent);
        goto Shutdown;
    }       

    //-------------------------------------------------------------------------
    // Similar code in linux-rocm.c:_rocm_update_control_state.
    //-------------------------------------------------------------------------

    printf_v2("main: Creating context for device %d.\n", MainDevice);
    global__rocm_control->arrayOfActiveContexts[0] = (struct _rocm_active_context_s*) calloc(1, sizeof(_rocm_active_context_t));
    CHECK_PRINT_EVAL(global__rocm_control->arrayOfActiveContexts[0] == NULL, "Memory allocation for new active context failed", goto Shutdown);
    global__rocm_control->arrayOfActiveContexts[0]->deviceNum = MainDevice;
    global__rocm_control->arrayOfActiveContexts[0]->ctx = NULL;
    global__rocm_control->arrayOfActiveContexts[0]->conEventsCount = 0;
    global__rocm_control->countOfActiveContexts++;

    eventctrl = global__rocm_control->arrayOfActiveContexts[0];

    printf_v2("main: Adding Event %s to the context\n", MainEvent);

    eventctrl->conEvents[0] = MainEventId;
    eventctrl->conEventIndex[0] = 0;
    eventctrl->conEventsCount++;

    global__rocm_control->activeEventValues[0] = 0;
    global__rocm_control->activeEventCount++;

    /* Create/recreate eventgrouppass structures for the added event and context */
    printf_v2("main: Create eventGroupPasses for context (destroy pre-existing) (Number_Events_Requested %d, conEventsCount %d) \n", global__rocm_control->activeEventCount, eventctrl->conEventsCount);
    if(eventctrl->conEventsCount > 0) {
        if (eventctrl->ctx != NULL) {
            ROCP_CALL_CK(rocprofiler_close, (eventctrl->ctx), goto Shutdown);
        }
        int openFailed=0;
        printf_v2("main: Calling rocprofiler_open, device=%i numEvents=%i name='%s'.\n",
             MainDevice, eventctrl->conEventsCount, MainEvent);
        const uint32_t mode = (global__ctx_properties.queue != NULL) ? 
            ROCPROFILER_MODE_STANDALONE : ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE;
        ROCP_CALL_CK(rocprofiler_open, (global__rocm_context->availAgentArray[MainDevice], 
            eventctrl->conEvents, eventctrl->conEventsCount, &(eventctrl->ctx),
            mode, &global__ctx_properties), openFailed=1);
        if (openFailed) {
            fprintf(stderr, "Error: The ROCM event '%s' was not accepted by the ROCPROFILER.\n", MainEvent);
            goto Shutdown;
        }

        ROCP_CALL_CK(rocprofiler_group_count, (eventctrl->ctx, &numPasses), goto Shutdown);

        if (numPasses > 1) {
            fprintf(stderr, "Error: The combined ROCM events require more than 1 pass.\n");
            goto Shutdown;
        } else  {
            printf_v2("main: Created eventGroupPasses for context total-events %d in-this-context %d passes-required %d) \n", global__rocm_control->activeEventCount, eventctrl->conEventsCount, numPasses);
        }
    }

    //-------------------------------------------------------------------------
    // Run the work kernel as often as user requested.
    //-------------------------------------------------------------------------
    do_work();

    //-------------------------------------------------------------------------
    // Cleanup and Exit.
    //-------------------------------------------------------------------------
Shutdown: 
    printf_v2("main: Shutting down component.\n");

    //-------------------------------------------------------------------------
    // Similar code:linux-rocm.c:_rocm_cleanup_eventset
    // We don't inline this code, to make it easier to use in the do_work area.
    // Disable and destroy the ROCM eventGroup.
    //-------------------------------------------------------------------------
    // There is only one context.
    int eventDeviceNum = global__rocm_control->arrayOfActiveContexts[0]->deviceNum;
    Context eventCtx = global__rocm_control->arrayOfActiveContexts[0]->ctx;
    printf_v2("main: Closing Rocprofiler context on device %d ctx %p \n", eventDeviceNum, eventCtx);
    ROCP_CALL_CK(rocprofiler_close, (eventCtx), status=_status);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "main: Failed on rocprofiler_close.\n");
    }

    free( global__rocm_control->arrayOfActiveContexts[0] );
    global__rocm_control->arrayOfActiveContexts[0] = NULL;

    if (global__ctx_properties.queue != NULL) {
        printf_v2("main: Destroying HSA Queue, on device %d\n", eventDeviceNum);
        ROCM_CALL_CK(hsa_queue_destroy, (global__ctx_properties.queue), status=_status);
        if (status != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "main: Failed on hsa_queue_destroy.\n");
        }

        global__ctx_properties.queue = NULL;
    }

    // Record that there are no active contexts or events
    global__rocm_control->countOfActiveContexts = 0;
    global__rocm_control->activeEventCount = 0;
    
    //-------------------------------------------------------------------------
    // Similar code: linux-rocm.c:_rocm_shutdown_component
    //-------------------------------------------------------------------------

    if(global__rocm_context != NULL) {
        free(global__rocm_context);
        global__rocm_context = NULL;
    }

    /* Free control */
    if(global__rocm_control != NULL) {
        if(global__rocm_control->arrayOfActiveContexts[0] != NULL) {
            free(global__rocm_control->arrayOfActiveContexts[0]);
        }

        free(global__rocm_control);
        global__rocm_control = NULL;
    }

    // Shutdown ROC runtime. If this fails, macro prints the error.
    ROCM_CALL_CK(hsa_shut_down, (), status=_status);
    if (status == HSA_STATUS_SUCCESS) {
        printf_v2("main: ROCM shutdown component succeeded.\n");
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
   
    printf_v2("do_work: rocm_start was successful.\n"); 
  

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
       
    printf_v2("do_work: rocm_read was successful, dataread ptr = %p.\n", dataread);
    printf("\ndo_work: EVENT '%s' RETURNED %lli.\n\n", MainEvent, dataread[0]);

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

    printf("do_work: buffer_size=%8lu time=%9lu per-cycle=%lf\n",buffer_size, dt, dt/((double)h_array[0]));

    // Cleanup, so we can run it again.
    HIPCHECK(hipFree(d_res));
    HIPCHECK(hipFree(d_inp));
    HIPCHECK(hipHostFree(h_array)); 
} // end do_work

