#include <dlfcn.h>

#include "papi.h"
#include "papi_debug.h"
#include "cupti_event_and_metric.h"
#include "papi_cupti_common.h"
#include "htable.h"

#include "cupti_events.h"
#include "cupti_metrics.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * Event identifier encoding format:
 * +---------------------------------+-------+----+------------+
 * |         unused                  |  dev  | ql |   nameid   |
 * +---------------------------------+-------+----+------------+
 *
 * unused    : 34 bits 
 * device    : 7  bits ([0 - 127] devices)
 * qlmask    : 2  bits (qualifier mask)
 * nameid    : 21: bits (roughly > 2 million event names)
 */
#define EVENTS_WIDTH (sizeof(uint64_t) * 8)
#define DEVICE_WIDTH ( 7)
#define QLMASK_WIDTH ( 2) 
#define NAMEID_WIDTH (21)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - DEVICE_WIDTH)
#define QLMASK_SHIFT (DEVICE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define DEVICE_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define DEVICE_FLAG  (0x1)

/* Functions needed by CUPTI Events API */
/* ... */

//TODO: Move event_info_t to utils file?
typedef struct {
    int device;
    int flags;
    int nameid;
} event_info_t;

typedef struct cuptie_gpu_state_s {
    int                    dev_id;
    cuptiu_event_and_metric_table_t  *added_events;
} cuptie_gpu_state_t;

struct cuptie_control_s {
    cuptie_gpu_state_t *gpu_ctl;
    long long           *counters;
    int                 read_count;
    int                 running;
    cuptic_info_t       info;
};

static int numDevicesOnMachine;

/* main event table to store metrics */
static cuptiu_event_and_metric_table_t *cuptiu_table_p; // TODO: Possibly make this event take accessible from cupti_utils.h

static int get_ntv_events(cuptiu_event_and_metric_table_t *evt_table, const char *evt_name, const char *evt_desc, int gpu_id, int api);


static int evt_id_to_info(uint32_t event_id, event_info_t *info);
static int evt_id_create(event_info_t *info, uint32_t *event_id);

static void init_main_htable(void);
static int init_events_and_metrics_table(void);

/* Events API Function Pointers */
// Functions for initializing the event table/utilities
CUptiResult (*cuptiDeviceEnumEventDomainsPtr) (CUdevice device, size_t *arraySizeBytes, CUpti_EventDomainID *domainArray);
CUptiResult (*cuptiDeviceGetNumEventDomainsPtr) (CUdevice device, uint32_t *numDomains);
CUptiResult (*cuptiEventDomainEnumEventsPtr) (CUpti_EventDomainID eventDomain, size_t *sarraySizeBytes, CUpti_EventID *eventArray);
CUptiResult (*cuptiEventGetAttributePtr) (CUpti_EventID event, CUpti_EventAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiEventDomainGetNumEventsPtr) (CUpti_EventDomainID eventDomain, uint32_t *numEvents);
CUptiResult (*cuptiEventGroupSetsCreatePtr) (CUcontext context, size_t eventIdArraySizeBytes, CUpti_EventID *eventIdArray, CUpti_EventGroupSets **eventGroupPasses);

// Functions for Event API Profiling
CUptiResult (*cuptiEventGroupCreatePtr) (CUcontext context, CUpti_EventGroup *eventGroup, uint32_t flags);
CUptiResult (*cuptiEventGetIdFromNamePtr) (CUdevice device, const char *eventName, CUpti_EventID *event);
CUptiResult (*cuptiEventGroupAddEventPtr) (CUpti_EventGroup eventGroup, CUpti_EventID event);
CUptiResult (*cuptiEventGroupEnablePtr) (CUpti_EventGroup eventGroup);
CUptiResult (*cuptiEventGroupReadAllEventsPtr) (CUpti_EventGroup eventGroup, CUpti_ReadEventFlags flags, size_t *eventValueBufferSizeBytes, uint64_t *eventValueBuffer, size_t *eventIdArraySizeBytes, CUpti_EventID *eventIdArray, size_t *numEventIdsRead);
CUptiResult (*cuptiEventGroupGetAttributePtr) (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiDeviceGetEventDomainAttributePtr) (CUdevice device, CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiEventGroupDisablePtr) (CUpti_EventGroup eventGroup);

CUptiResult ( *cuptiProfilerInitializeMetricsPtr ) (CUpti_Profiler_Initialize_Params* params);

/* Metrics API Function Pointers */
CUptiResult (*cuptiDeviceGetNumMetricsPtr) (CUdevice device, uint32_t *numMetrics);
CUptiResult (*cuptiDeviceEnumMetricsPtr) (CUdevice device, size_t *arraySizeBytes, CUpti_MetricID *metricArray);
CUptiResult (*cuptiMetricGetAttributePtr) (CUpti_MetricID metric, CUpti_MetricAttribute attrib, size_t *valueSize, void *value);
CUptiResult (*cuptiMetricCreateEventGroupSetsPtr) (CUcontext context, size_t metricIdArraySizeBytes, CUpti_MetricID *metricIdArray, CUpti_EventGroupSets **eventGroupPasses);
CUptiResult (*cuptiMetricGetIdFromNamePtr) (CUdevice device, const char *metricName, CUpti_MetricID *metric);
CUptiResult (*cuptiMetricGetNumEventsPtr) (CUpti_MetricID metric, uint32_t *numEvents);
CUptiResult (*cuptiMetricEnumEventsPtr) (CUpti_MetricID metric, size_t *eventIdArraySizeBytes, CUpti_EventID *eventIdArray);
CUptiResult (*cuptiMetricGetValuePtr) (CUdevice device, CUpti_MetricID metric, size_t eventIdArraySizeBytes, CUpti_EventID *eventIdArray, size_t eventValueArraySizeBytes, uint64_t *eventValueArray, uint64_t timeDuration, CUpti_MetricValue *metricValue);

//CUptiResult ( *cuptiDeviceGetChipNamePtr ) (CUpti_Device_GetChipName_Params* params);

// Helper functions
int determine_dev_cc(int dev_id);
static int evt_name_to_device(const char *name, int *device);
static int evt_name_to_basename(const char *name, char *base, int len);
static int verify_user_added_event_or_metric(uint32_t *events_id, int num_events, cuptie_control_t state);
static int find_same_chipname(int gpu_id);
static int init_all_metrics(void);
static int initialize_cupti_profiler_api(void);

static int enumerate_events_for_event_api(cuptiu_event_and_metric_table_t *table, CUcontext ctx, CUdevice device, int dev_id);
static int enumerate_metrics_for_metric_api(cuptiu_event_and_metric_table_t *table, CUcontext ctx, CUdevice device, int dev_id);
static int check_if_event_or_metric_requires_mutiple_passes(const char *addedEventName, int cuptiApi, int deviceIdx, uint32_t *addedNativeEventID);


static gpu_record_event_and_metric_t *avail_gpu_info;

//CUpti_EventID *eventIDs = NULL;
//CUpti_MetricID *metricIDs = NULL;
//CUpti_EventGroup eventGroup;

static int load_events_sym(void)
{
    //TODO: Do I want to just make these loaded when I need them?
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


    // Events API
    cuptiDeviceEnumEventDomainsPtr   = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceEnumEventDomains");
    cuptiDeviceGetNumEventDomainsPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetNumEventDomains");
    cuptiEventDomainEnumEventsPtr    = DLSYM_AND_CHECK(dl_cupti, "cuptiEventDomainEnumEvents");
    cuptiEventGetAttributePtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGetAttribute");
    cuptiEventDomainGetNumEventsPtr  = DLSYM_AND_CHECK(dl_cupti, "cuptiEventDomainGetNumEvents");

    cuptiEventGroupSetsCreatePtr     = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupSetsCreate");
    cuptiEventGroupCreatePtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupCreate");
    cuptiEventGetIdFromNamePtr       = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGetIdFromName"); 
    cuptiEventGroupAddEventPtr       = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupAddEvent");
    cuptiEventGroupGetAttributePtr      = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupGetAttribute");
    cuptiDeviceGetEventDomainAttributePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetEventDomainAttribute");

    cuptiEventGroupEnablePtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupEnable");
    cuptiEventGroupReadAllEventsPtr     = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupReadAllEvents");
    cuptiEventGroupDisablePtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiEventGroupDisable");

    //cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    cuptiProfilerInitializeMetricsPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");

    // Metrics API
    cuptiDeviceGetNumMetricsPtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetNumMetrics");
    cuptiDeviceEnumMetricsPtr          = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceEnumMetrics");
    cuptiMetricGetAttributePtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetAttribute");
    cuptiMetricCreateEventGroupSetsPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricCreateEventGroupSets");
    cuptiMetricGetIdFromNamePtr        = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetIdFromName");
    cuptiMetricGetNumEventsPtr         = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetNumEvents");
    cuptiMetricEnumEventsPtr           = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricEnumEvents");
    cuptiMetricGetValuePtr            = DLSYM_AND_CHECK(dl_cupti, "cuptiMetricGetValue");

    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiGetVersion");

    Dl_info info;
    dladdr(cuptiGetVersionPtr, &info);
    LOGDBG("CUPTI library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}


// TODO: Move this to a universal file as events and perfworks use this? Yes do this.
static int initialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    profilerInitializeParams.pPriv = NULL;
    cuptiCheckErrors( cuptiProfilerInitializeMetricsPtr(&profilerInitializeParams), return PAPI_EMISC );

    return PAPI_OK;
}


void init_main_htable(void) 
{
    int i, val = 1, base = 2;  
 
    /* allocate (2 ^ NAMEID_WIDTH) metric names, this matches the 
       number of bits for the event encoding format */
    for (i = 0; i < NAMEID_WIDTH; i++) {
        val *= base;
    }     
   
    /* initialize struct */ 
    cuptiu_table_p = papi_malloc(sizeof(cuptiu_event_and_metric_table_t));
    cuptiu_table_p->capacity = val; 
    cuptiu_table_p->count = 0;  
    cuptiu_table_p->events = papi_calloc(val, sizeof(cuptiu_event_and_metric_t));


    cuptiu_table_p->avail_gpu_info = (gpu_record_event_and_metric_t *) papi_calloc(numDevicesOnMachine, sizeof(gpu_record_event_and_metric_t));
    if (cuptiu_table_p->avail_gpu_info == NULL) {
        printf("Memory allocation failed in init_main_htable.\n");
    }
 
   
    /* initialize the main hash table for metric collection */ 
    htable_init(&cuptiu_table_p->htable);
}

int init_all_metrics(void)
{
    int i, len, papi_errno;
    char chipName[PAPI_MIN_STR_LEN];


    for (i = 0; i < numDevicesOnMachine; i++) {
        papi_errno = get_chip_name(i, chipName);
        if (papi_errno != PAPI_OK) {
            printf("failed to get chip name: %d\n", papi_errno);
            return papi_errno;
        }

        printf("Chipname is: %s\n", chipName);
        len = snprintf(cuptiu_table_p->avail_gpu_info[i].chipName, PAPI_MIN_STR_LEN, "%s", chipName);
        if (len < 0 || len >= PAPI_MIN_STR_LEN) {
             printf("Failed to write chipname.\n");
             return PAPI_ENOMEM;
        }
 
    }
    return PAPI_OK;
}


int cuptie_init(void)
{
    SUBDBG("ENTERING: Initializing the Cuda component for CUPTI Event and Metric API support.\n");
    int papi_errno;

    papi_errno = load_events_sym();
    if (papi_errno != PAPI_OK) {
        printf("Failed to load events api.\n");
    }

    // Get the number of GPUs on the machine
    papi_errno = cuptic_device_get_count(&numDevicesOnMachine);
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }

    papi_errno = initialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        printf("initialize profiler: %d\n", papi_errno);
        return PAPI_EMISC;
    }


    printf("numDevicesOnMachine: %d\n", numDevicesOnMachine); 

    // Init the main htable
    init_main_htable();

    init_all_metrics();

    // Init the event table
    init_events_and_metrics_table();

    CUresult cuError = cuInitPtr(0);
    if (cuError != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }

    SUBDBG("EXITING: Initialization for the CUPTI Event and Metric API completed.\n");
    return PAPI_OK;
}


int init_events_and_metrics_table(void)
{
    SUBDBG("ENTERING: Adding events and metrics from the Event and Metric APIs to the hash table.\n");
    // Loop through all of the available devices on the machine
    int dev_id, table_idx = 0;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        // Skip devices that will require the Perfworks API to be profiled
        // TODO: Part this out to a separate file since it is used in the perfworks AP as well
        if (determine_dev_cc(dev_id) == 0) {
            continue;
        }
        int found = find_same_chipname(dev_id);
        // Unique device found, collect metadata
        if (found == -1) {
            // Increment table index
            if (dev_id > 0)
                table_idx++;

            // Get the handle for the current compute device 
            CUdevice device;
            cudaCheckErrors( cuDeviceGetPtr(&device, dev_id), return PAPI_EMISC );
            // Store the handle to be used later with 
            cuptiu_table_p->avail_gpu_info[table_idx].deviceHandle = device;
        }
        /* Metadata already collect for device */
        else {
            /* set table_idx to */
            table_idx = found;
        }

        // For both the Events and Metrics API we must have a Cuda context created to get either the Events or Metrics
        // This context will be destroyed after each loop
        CUcontext ctx;
        cudaCheckErrors( cuCtxCreatePtr(&ctx, 0, dev_id), return PAPI_EMISC ); 

        // For the Events API get the events and the events descriptions with the number of passes
        int papi_errno = enumerate_events_for_event_api(cuptiu_table_p, ctx, cuptiu_table_p->avail_gpu_info[table_idx].deviceHandle, dev_id);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        // For the Metrics API get the metrics and the metrics descriptions with the number of passes
        papi_errno = enumerate_metrics_for_metric_api(cuptiu_table_p, ctx, cuptiu_table_p->avail_gpu_info[table_idx].deviceHandle, dev_id);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        cudaCheckErrors( cuCtxDestroyPtr(ctx), return PAPI_EMISC ); 
    }

    SUBDBG("EXITING: Successfully added events and metrics to the hash table.\n");
    return PAPI_OK;
}


/** @class enumerate_events_for_event_api
  * @brief For the Events API enumerate through the available events
  *        collecting the description and number of passes.
  * @param *table
  *   Structure containing member variables such as name, evt_code, evt_pos,
  *   and htable.
  * @param ctx
  *   The created Cuda context for a device.
  * @param device
  *   A created Cuda device handle.
  * @param dev_id
  *   Device index.
*/
int enumerate_events_for_event_api(cuptiu_event_and_metric_table_t *table, CUcontext ctx, CUdevice device, int dev_id)
{
    SUBDBG("ENTERING: Enumerating events for the CUPTI Event API.\n");
    // Get the total number of Event Domains for a device
    uint32_t numDomains;
    cuptiCheckErrors( cuptiDeviceGetNumEventDomainsPtr(device, &numDomains),  return PAPI_EMISC );

    // Get the event domains for a device
    size_t size = numDomains * sizeof(CUpti_EventDomainID);
    CUpti_EventDomainID *domainArray = (CUpti_EventDomainID *) calloc(numDomains, sizeof(CUpti_EventDomainID));
    if (domainArray == NULL) {
        SUBDBG("Failed to allocate memory for domainArray.\n");
        return PAPI_ENOMEM;
    }
    cuptiCheckErrors( cuptiDeviceEnumEventDomainsPtr(device, &size, domainArray), return PAPI_EMISC );

    // Go over the total number of domains found for the device
    int domainIdx;
    for (domainIdx = 0; domainIdx < numDomains; domainIdx++) {
        // For each domain, get the total number of events 
        uint32_t numEvents;
        cuptiCheckErrors( cuptiEventDomainGetNumEventsPtr(domainArray[domainIdx], &numEvents), return PAPI_EMISC );

        // Allocate memory
        CUpti_EventID *eventArray = (CUpti_EventID *) calloc(numEvents, sizeof(CUpti_EventID));
        if (eventArray == NULL) {
            SUBDBG("Failed to allocate memory for eventArray.\n");
            return PAPI_ENOMEM;
        }
        size = numEvents * sizeof(CUpti_EventID);

        // For the domain, get the actual events
        cuptiCheckErrors( cuptiEventDomainEnumEventsPtr(domainArray[domainIdx], &size, eventArray), return PAPI_EMISC );

        // Go over the total number of events found in the domain for the device
        CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
        if (eventGroupPasses == NULL) {
            SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
            return PAPI_ENOMEM;
        }
        int eventIdx;
        for (eventIdx = 0; eventIdx < numEvents; eventIdx++) {
            // Name attribute
            size = PAPI_2MAX_STR_LEN * sizeof(char);
            char eventName[PAPI_2MAX_STR_LEN];
            cuptiCheckErrors( cuptiEventGetAttributePtr(eventArray[eventIdx], CUPTI_EVENT_ATTR_NAME, &size, eventName), return PAPI_EMISC );

            // Long description attribute
            size = PAPI_HUGE_STR_LEN * sizeof(char);
            char eventDesc[PAPI_HUGE_STR_LEN];
            cuptiCheckErrors( cuptiEventGetAttributePtr(eventArray[eventIdx], CUPTI_EVENT_ATTR_LONG_DESCRIPTION, &size, eventDesc), return PAPI_EMISC );

            // For each event, get the number of passes
            cuptiCheckErrors( cuptiEventGroupSetsCreatePtr(ctx, sizeof(CUpti_EventID), &eventArray[eventIdx], &eventGroupPasses), return PAPI_EMISC);

            const char *format;
            if (eventDesc[size - 1] != '.')
                format = eventGroupPasses->numSets > 1 ? "%s. Numpass=%d (multi-pass not supported)" : "%s. Numpass=%d";
            else
                format = eventGroupPasses->numSets > 1 ? "%s Numpass=%d (multi-pass not supported)" : "%s Numpass=%d";

            char fullEventDesc[PAPI_HUGE_STR_LEN];
            int strLen = snprintf(fullEventDesc, PAPI_HUGE_STR_LEN, format, eventDesc, eventGroupPasses->numSets);
            if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
                SUBDBG("Failed to fully write description for event.\n");
                return PAPI_EBUF;
            }

            int papi_errno = get_ntv_events(table, eventName, fullEventDesc, dev_id, event_api);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }
        }
        free(eventGroupPasses);
        free(eventArray);
    }

    SUBDBG("EXITING: Enumeration completed for the CUPTI Event API.\n");
    return PAPI_OK; 
}

/** @class enumerate_metrics_for_metric_api
  * @brief For the Metric API enumerate through the available metrics
  *        collecting the description and number of passes.
  * @param *table
  *   Structure containing member variables such as name, evt_code, evt_pos,
  *   and htable.
  * @param ctx
  *   The created Cuda context for a device.
  * @param device
  *   A created Cuda device handle.
  * @param dev_id
  *   Device index.
*/
int enumerate_metrics_for_metric_api(cuptiu_event_and_metric_table_t *table, CUcontext ctx, CUdevice device, int dev_id) 
{
    SUBDBG("ENTERING: Enumerating metrics for the CUPTI Metric API.\n");
    // Get the total number of metrics for a device
    int numMetrics;
    cuptiCheckErrors( cuptiDeviceGetNumMetricsPtr(device, &numMetrics), return PAPI_EMISC );

    // Get the metrics for the device
    size_t size = numMetrics * sizeof(CUpti_MetricID);
    CUpti_MetricID *metricIdList = (CUpti_MetricID *) calloc(numMetrics, sizeof(CUpti_MetricID));
    if (metricIdList == NULL) {
        SUBDBG("Failed to allocate memory for metricIdList.\n");
        return PAPI_ENOMEM;
    }
    cuptiCheckErrors( cuptiDeviceEnumMetricsPtr(device, &size, metricIdList), return PAPI_EMISC );

    // For each metric get the name and description attribute
    CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
    if (eventGroupPasses == NULL) {
        SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
        return PAPI_ENOMEM;
    }
    int metricIdx;
    for (metricIdx = 0; metricIdx < numMetrics; metricIdx++) {
        // Name attribute
        size = PAPI_2MAX_STR_LEN * sizeof(char);
        char metricName[PAPI_2MAX_STR_LEN];
        cuptiCheckErrors( cuptiMetricGetAttributePtr(metricIdList[metricIdx], CUPTI_METRIC_ATTR_NAME, &size, metricName), return PAPI_EMISC );

        // Long description attribute
        size = PAPI_HUGE_STR_LEN * sizeof(char);
        char metricDesc[PAPI_HUGE_STR_LEN];
        cuptiCheckErrors( cuptiMetricGetAttributePtr(metricIdList[metricIdx], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size, metricDesc), return PAPI_EMISC );

        // For each metric, get the number of passes
        cuptiCheckErrors( cuptiMetricCreateEventGroupSetsPtr(ctx, sizeof(CUpti_MetricID), &metricIdList[metricIdx], &eventGroupPasses), return PAPI_EMISC );

        const char *format;
        if (metricDesc[size - 1] == '.')
            format = eventGroupPasses->numSets > 1 ? "%s Numpass=%d (multi-pass not supported)" : "%s Numpass=%d";
        else
            format = eventGroupPasses->numSets > 1 ? "%s. Numpass=%d (multi-pass not supported)" : "%s. Numpass=%d";

        char fullMetricDesc[PAPI_HUGE_STR_LEN];
        int strLen = snprintf(fullMetricDesc, PAPI_HUGE_STR_LEN, format, metricDesc, eventGroupPasses->numSets);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write description for metric.\n");
            return PAPI_EBUF;
        }

        int papi_errno = get_ntv_events(table, metricName, fullMetricDesc, dev_id, metric_api);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }    
    }
    free(eventGroupPasses);

    SUBDBG("EXITING: Enumeration completed for the CUPTI Metric API.\n");
    return PAPI_OK;
}



int get_ntv_events(cuptiu_event_and_metric_table_t *evt_table, const char *evt_name, const char *evt_desc, int gpu_id, int api)
{
    int *count = &evt_table->count;
    cuptiu_event_and_metric_t *events = evt_table->events;

    if (evt_name == NULL) {
        return PAPI_EINVAL;
    }

    if (evt_table->count >= evt_table->capacity) {
        printf("Table count is larger than allocated capacity.");
        return PAPI_EBUG;
    }

    cuptiu_event_and_metric_t *event;
    if (htable_find(evt_table->htable, evt_name, (void **) &event) != HTABLE_SUCCESS) {
        event = &events[*count];
        (*count)++;

        int strLen = snprintf(event->name, PAPI_2MAX_STR_LEN, "%s", evt_name);
        if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
            SUBDBG("Failed to fully write evt_name to event->name.\n");
            return PAPI_EBUF;
        }

        strLen = snprintf(event->desc, PAPI_HUGE_STR_LEN, "%s", evt_desc);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write evt_desc to event->desc.\n");
            return PAPI_EBUF;
        }

        event->api = api;

        if (htable_insert(evt_table->htable, evt_name, event) != HTABLE_SUCCESS) {
            return PAPI_ESYS;
        } 
    }
    cuptiu_dev_set(&event->device_map, gpu_id);

    return PAPI_OK;
}

/** @class cuptip_evt_code_to_info
  * @brief Takes a Cuda native event code and collects info such as Cuda native 
  *        event name, Cuda native event description, and number of devices. 
  * @param event_code
  *   Cuda native event code. 
  * @param *info
  *   Structure for member variables such as symbol, short description, and 
  *   long desctiption. 
*/
int cuptie_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info)
{

    int papi_errno, i, gpu_id;
    char description[PAPI_HUGE_STR_LEN];

    /* get the events nameid and flags */
    event_info_t inf;
    papi_errno = evt_id_to_info(event_code, &inf);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch (inf.flags) {
        case (0):
            /* store details for the Cuda event */
            snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].name );
            snprintf( info->short_descr, PAPI_MIN_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s", cuptiu_table_p->events[inf.nameid].desc );
            break;
        case DEVICE_FLAG:
        {
            int init_metric_dev_id;
            char devices[PAPI_MAX_STR_LEN] = { 0 };
            for (i = 0; i < numDevicesOnMachine; ++i) {
                if (cuptiu_dev_check(cuptiu_table_p->events[inf.nameid].device_map, i)) {
                    /* for an event, store the first device found to use with :device=#, 
                       as on a heterogenous system events may not appear on each device */
                    if (devices[0] == '\0') {
                        init_metric_dev_id = i;
                    }

                    sprintf(devices + strlen(devices), "%i,", i);
                }
            }
            *(devices + strlen(devices) - 1) = 0;

            /* store details for the Cuda event */
            snprintf( info->symbol, PAPI_HUGE_STR_LEN, "%s:device=%i", cuptiu_table_p->events[inf.nameid].name, init_metric_dev_id );
            snprintf( info->short_descr, PAPI_MIN_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                     cuptiu_table_p->events[inf.nameid].desc, devices );
            snprintf( info->long_descr, PAPI_HUGE_STR_LEN, "%s masks:Mandatory device qualifier [%s]",
                      cuptiu_table_p->events[inf.nameid].desc, devices );
            break;
        }
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int check_if_event_or_metric_requires_mutiple_passes(const char *addedEventName, int cuptiApi, int deviceIdx, uint32_t *addedNativeEventID) 
{
    SUBDBG("ENTERING: Checking if the user added event does not require multiple passes.\n");
    // Check if a Cuda context is on the calling CPU thread
    CUcontext pctx;
    cudaCheckErrors( cuCtxGetCurrentPtr(&pctx), return PAPI_EMISC );
    // A Cuda context was found on the calling CPU thread
    if (pctx != NULL) {
        CUdevice device;
        CUresult cudaErr = cuCtxGetDevicePtr(&device);
        if (cudaErr != CUDA_SUCCESS) {
            printf("Failed to get device");
            return PAPI_EMISC;
        }
        printf("Device found is: %d\n", device);
    }
    // A Cuda context was not found on the calling CPU thread
    else {
        printf("We create a Cuda context.\n");
        int flags = 0;
        //cudaCheckErrors( cuCtxCreatePtr(&pctx, flags, deviceIdx), return PAPI_EMISC);
        CUresult result = cuCtxCreatePtr(&pctx, flags, deviceIdx);
        if (result != CUDA_SUCCESS) {
            printf("Error for cuCtxCreate: %d\n", result);
            return PAPI_EMISC;
        }
    }

    CUdevice device;
    cudaCheckErrors( cuDeviceGetPtr(&device, deviceIdx), return PAPI_EMISC);

    CUpti_EventGroupSets *eventGroupPasses = (CUpti_EventGroupSets *) calloc(1, sizeof(CUpti_EventGroupSets));
    if (eventGroupPasses == NULL) {
        SUBDBG("Failed to allocate memory for eventGroupPasses.\n");
        return PAPI_ENOMEM;
    } 

    CUpti_EventID eventId;
    CUpti_MetricID metricId;
    int papi_errno = PAPI_OK;
    switch(cuptiApi) 
    {    
        case event_api:
            cuptiCheckErrors( cuptiEventGetIdFromNamePtr(device, addedEventName, &eventId), return PAPI_EMISC );
            cuptiCheckErrors( cuptiEventGroupSetsCreatePtr(pctx, sizeof(CUpti_EventID), &eventId, &eventGroupPasses), return PAPI_EMISC );
            break;
        case metric_api:
            cuptiCheckErrors( cuptiMetricGetIdFromNamePtr(device, addedEventName, &metricId), return PAPI_EMISC );
            cuptiCheckErrors( cuptiMetricCreateEventGroupSetsPtr(pctx, sizeof(CUpti_MetricID), &metricId, &eventGroupPasses), return PAPI_EMISC );
            break;
        default:
            SUBDBG("API option is not supported. Internal error.\n");
            papi_errno = PAPI_EBUG;
            break;
    }

    if (eventGroupPasses->numSets > 1) {
        SUBDBG("Multiple pass event/metric detected. PAPI does not support events/metrics that require multiple passes.\n");
        papi_errno = PAPI_EMULPASS;
    }

    // Store the found Event or Metric ID 
    *addedNativeEventID = (cuptiApi == event_api) ? eventId : metricId;

    // Cleanup
    cudaCheckErrors( cuCtxDestroyPtr(pctx), return PAPI_EMISC );
    free(eventGroupPasses);

    SUBDBG("EXITING: Check for multiple passes completed.\n");
    return papi_errno;
}

void cuptiu_event_and_metric_table_destroy(cuptiu_event_and_metric_table_t **pevt_table)
{
    cuptiu_event_and_metric_table_t *evt_table = *pevt_table;
    if (evt_table == NULL)
        return;

    if (evt_table->htable) {
        htable_shutdown(evt_table->htable);
        evt_table->htable = NULL;
    }    

    free(evt_table);
    *pevt_table = NULL;
}

// TODO: Do I want to move these somewhere?
int cuptiu_event_and_metric_table_create_init_capacity(int capacity, int sizeof_rec, cuptiu_event_and_metric_table_t **pevt_table)
{
    cuptiu_event_and_metric_table_t *evt_table = (cuptiu_event_and_metric_table_t *) malloc(sizeof(cuptiu_event_and_metric_table_t));
    if (evt_table == NULL) {
        goto fn_fail;
    }   

    evt_table->capacity = capacity;
    evt_table->countOfMetricIDs = 0;
    evt_table->countOfEventIDs = 0;
    
    if (htable_init(&(evt_table->htable)) != HTABLE_SUCCESS) {
        cuptiu_event_and_metric_table_destroy(&evt_table);
        goto fn_fail;
    }   
    
    *pevt_table = evt_table;
    return 0;
fn_fail:
    *pevt_table = NULL;
    return PAPI_ENOMEM;
}

/** @class verify_user_added_event_or_metric
  * @brief For user added events or metrics, verify they exist and do not require
  *        multiple passes. If both are true, store metadata.
  * @param *events_id
  *   Cuda native event id's.
  * @param num_events
  *   Number of Cuda native events a user is wanting to count.
  * @param state
  *   Struct that holds read count, running, cuptip_info_t, and 
  *   cuptip_gpu_state_t. 
*/
int verify_user_added_event_or_metric(uint32_t *events_id, int num_events, cuptie_control_t state)
{
    SUBDBG("ENTERING: Verifying user added events exist and do not require multiple passes.\n");
    int i, papi_errno = PAPI_OK;
    for (i = 0; i < numDevicesOnMachine; i++) {
        papi_errno = cuptiu_event_and_metric_table_create_init_capacity(
                         num_events,
                         sizeof(cuptiu_event_and_metric_t), &(state->gpu_ctl[i].added_events)

                     );
       if (papi_errno != PAPI_OK) {
           return papi_errno;
       }
    }

    for (i = 0; i < num_events; i++) {
        event_info_t native_event_info;
        papi_errno = evt_id_to_info(events_id[i], &native_event_info);
        if (papi_errno != PAPI_OK) {
            printf("Failed evt_id_to_info: %d\n", papi_errno);
            return papi_errno;
        }

        // Verify the user added event exists
        void *p;
        if (htable_find(cuptiu_table_p->htable, cuptiu_table_p->events[native_event_info.nameid].name, (void **) &p) != HTABLE_SUCCESS) {
            return PAPI_ENOEVNT;
        }

        uint32_t addedNativeEventID;
        // Verify that the user added event does not require multiple passes
        int papi_errno = check_if_event_or_metric_requires_mutiple_passes(cuptiu_table_p->events[native_event_info.nameid].name,
                                                                          cuptiu_table_p->events[native_event_info.nameid].api,
                                                                          native_event_info.device, &addedNativeEventID);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        // If everything checks out, store the user added event for profiling
        if (cuptiu_table_p->events[native_event_info.nameid].api == event_api) {
            state->gpu_ctl[native_event_info.device].added_events->eventIDs[state->gpu_ctl[native_event_info.device].added_events->countOfEventIDs] = addedNativeEventID;
            state->gpu_ctl[native_event_info.device].added_events->countOfEventIDs++;
        }
        else {
            state->gpu_ctl[native_event_info.device].added_events->metricIDs[state->gpu_ctl[native_event_info.device].added_events->countOfMetricIDs] = addedNativeEventID;
            state->gpu_ctl[native_event_info.device].added_events->countOfMetricIDs++;
        }
    }

    SUBDBG("EXITING: Checking user added a valid event completed.\n");
    return PAPI_OK; 
}


int cuptie_ctx_create(cuptic_info_t thr_info, cuptie_control_t *pstate, uint32_t *events_id, int num_events)
{
    SUBDBG("ENTERING: Creating a profiling context for the request cuda native events.\n");
    cuptie_control_t state = (cuptie_control_t) calloc(1, sizeof(struct cuptie_control_s));
    if (state == NULL) {
        SUBDBG("Failed to allocate memory for state.\n");
        return PAPI_ENOMEM; 
    }

    state->gpu_ctl = (cuptie_gpu_state_t *) calloc(numDevicesOnMachine, sizeof(cuptie_gpu_state_t));
    if (state->gpu_ctl == NULL) {
        SUBDBG("Failed to allocate memory for state->gpu_ctl.\n");
        return PAPI_ENOMEM;
    }

    // TODO: Is this correct? As in how does it work?
    long long *counters = (long long *) malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        SUBDBG("Failed to allocate memory for counters.\n");
        return PAPI_ENOMEM;
    }

    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        state->gpu_ctl[dev_id].dev_id = dev_id;
    }

    event_info_t info;
    int papi_errno = evt_id_to_info(events_id[num_events - 1], &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = verify_user_added_event_or_metric(events_id, num_events, state);
    if (papi_errno != PAPI_OK) {
        printf("We failed: %d\n", papi_errno);
        return papi_errno;    
    }

    //TODO: This causes issues with having to create a context for a user with CUPTI Event and Metric API.
    // Store a user created cuda context or create one
    // Why not move this to start? To Check at that point? I do not see a point for this now. 
//    papi_errno = cuptic_ctxarr_update_current(thr_info, info.device);
//    if (papi_errno != PAPI_OK) {
//        return papi_errno;
//    }  
printf("7.\n");
    state->info = thr_info;
    state->counters = counters;
    *pstate = state;

   //printf("cuptiu_table_p->events[info.nameid].name: %s\n", cuptiu_table_p->events[info.nameid].name);

   SUBDBG("EXITING: Creation of a profiling context completed.\n");
   return PAPI_OK;
}

int cuptie_ctx_start(cuptie_control_t state)
{
    SUBDBG("ENTERING: Setting up profiling for the Event and Metric APIs.\n");
    CUcontext pctx;
    cudaCheckErrors( cuCtxGetCurrentPtr(&pctx), return PAPI_EMISC );
    if (pctx == NULL) {
        cudaCheckErrors( cuCtxCreatePtr(&pctx, 0, 0), return PAPI_EMISC );
    }

    // This setup is only required if a user adds native events from the Events API
    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->countOfEventIDs == 0) {
            continue;
        }

        CUpti_EventGroup eventGroup;
        int flags = 0; // From documentation flags are reserved for future use and should be set to zero
        CUptiResult cuptiErr = cuptiEventGroupCreatePtr(pctx, &eventGroup, flags);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failes cuptiEventGroupSetsCreate: %d\n", cuptiErr);
            return PAPI_EMISC;
        } 

        int eventIdx;
        for (eventIdx = 0; eventIdx < gpu_ctl->added_events->countOfEventIDs; eventIdx++) {
            cuptiErr = cuptiEventGroupAddEventPtr(eventGroup, gpu_ctl->added_events->eventIDs[eventIdx]);
            if (cuptiErr != CUPTI_SUCCESS) {
                printf("Failed to add event: %d\n", cuptiErr);
                return PAPI_EMISC;
            }   
        }   

        // Enable the event group as late as possible as you cannot add more events once enabled
        cuptiErr = cuptiEventGroupEnablePtr(eventGroup);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failed to enable event group: %d\n", cuptiErr);
            return PAPI_EMISC;
        }

        gpu_ctl->added_events->eventGroup = eventGroup;
    }
 
    SUBDBG("EXITING: Profiling setup completed.\n");
    return PAPI_OK;
}

int cuptie_ctx_read(cuptie_control_t state, long long **values)
{
    SUBDBG("ENTERING: Reading values for the Event and Metric APIs.\n");

    CUdevice device;
    int deviceIndex = 0;
    cuDeviceGetPtr(&device, deviceIndex);

    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->countOfEventIDs == 0 && gpu_ctl->added_events->countOfMetricIDs == 0) {
            continue;
        }
 
        CUpti_EventGroup eventGroup = gpu_ctl->added_events->eventGroup; 

        int numGroupEvents = 0;
        size_t numGroupEventsSize = sizeof(numGroupEvents);
        CUptiResult cuptiErr = cuptiEventGroupGetAttributePtr(eventGroup, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numGroupEventsSize, &numGroupEvents);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failed to get num events attribute: %d\n", cuptiErr);
            exit(1);
        } 

        CUpti_EventDomainID groupDomain;
        size_t groupDomainSize = sizeof(groupDomain);
        cuptiErr = cuptiEventGroupGetAttributePtr(eventGroup, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failed to get attribture event domain id: %d\n", cuptiErr);
            exit(1);
        } 

        uint32_t numTotalInstances;
        size_t numTotalInstancesSize = sizeof(numTotalInstances);
        cuptiErr = cuptiDeviceGetEventDomainAttributePtr(device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize, &numTotalInstances);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failed to get domain attribute: %d\n", cuptiErr);
            exit(1);
        } 

        size_t sizeOfEventValueBufferInBytes = sizeof(uint64_t) * numGroupEvents * numTotalInstances;
        uint64_t *eventValueBuffer = (uint64_t *) malloc(sizeOfEventValueBufferInBytes);

        size_t sizeOfEventIdArrayInBytes = numGroupEvents * sizeof(CUpti_EventID);
        CUpti_EventID *eventIdArray = (CUpti_EventID *) malloc(sizeOfEventIdArrayInBytes);

        size_t numEventIdsRead;
        cuptiErr = cuptiEventGroupReadAllEventsPtr(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, &sizeOfEventValueBufferInBytes, eventValueBuffer, &sizeOfEventIdArrayInBytes, eventIdArray, &numEventIdsRead);
        if (cuptiErr != CUPTI_SUCCESS) {
            printf("Failed to call cuptiEventGroupReadAllEvents: %d\n", cuptiErr);
            exit(1);         
        } 

        printf("eventValueBuffer: %d\n", eventValueBuffer[0]);
    }

// This workflow for the Metric API is good.
/*
    int dev_id;
    for (dev_id = 0; dev_id < numDevicesOnMachine; dev_id++) {
        cuptie_gpu_state_t *gpu_ctl = &(state->gpu_ctl[dev_id]);
        if (gpu_ctl->added_events->countOfMetricIDs == 0) {
            continue;
        }

        int metricIdx;
        for (metricIdx = 0; metricIdx < gpu_ctl->added_events->countOfMetricIDs; metricIdx++) {
            uint32_t numEvents;
            cuptiCheckErrors( cuptiMetricGetNumEventsPtr(gpu_ctl->added_events->metricIDs[metricIdx], &numEvents), return PAPI_EMISC);

     
            // Event IDs required to calculate the metric 
            CUpti_EventID *eventIdsArray = (CUpti_EventID *) malloc(numEvents * sizeof(CUpti_EventID));
            if (eventIdsArray == NULL) {
                printf("Failed to allocate memory for eventIdArray.\n");
                exit(1);
            }
            // Size of eventIdsArray 
            size_t sizeOfEventIdsArrayInBytes = numEvents * sizeof(CUpti_EventID);

            cuptiCheckErrors( cuptiMetricEnumEventsPtr(gpu_ctl->added_events->metricIDs[metricIdx], &sizeOfEventIdsArrayInBytes, eventIdsArray), return PAPI_EMISC);

            // Allocate memory for normalized event values required to calculate the metric
            uint64_t *eventValuesArray = (uint64_t *) malloc(numEvents * sizeof(uint64_t));
            if (eventValuesArray == NULL) {
                printf("Failed to allocate memory for eventValueArray\n");
                exit(1);
            }
            // Size of eventValueArray
            size_t sizeOfEventValuesArrayInBytes = numEvents * sizeof(uint64_t);
       
            uint64_t duration = 10;
            CUpti_MetricValue metricValue;
            cuptiCheckErrors( cuptiMetricGetValuePtr(device, gpu_ctl->added_events->metricIDs[metricIdx], sizeOfEventIdsArrayInBytes, eventIdsArray, sizeOfEventValuesArrayInBytes, eventValuesArray, duration, &metricValue), return PAPI_EMISC);
            printf("metricValue is: %f\n", metricValue.metricValueDouble);
        }
    }
*/
    SUBDBG("EXITING: Reading values completed.\n");
    return PAPI_ENOIMPL;
}

int cuptie_ctx_stop(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_ctx_reset(cuptie_control_t ctl)
{
    // Probably will need cuptiEventGroupResetAllEvents
    return PAPI_ENOIMPL;
}

int cuptie_ctx_destroy(cuptie_control_t *pctl)
{
    return PAPI_ENOIMPL;
}

/** @class evt_id_create
  * @brief Create event ID. Function is needed for cuptip_event_enum.
  *
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
  * @param *event_id
  *   Created event id.
*/

//TODO: Move both evt_id_create and evt_id_to_info out to possibly cupti_utils.h
int evt_id_create(event_info_t *info, uint32_t *event_id)
{

    *event_id  = (uint64_t)(info->device   << DEVICE_SHIFT);
    *event_id |= (uint64_t)(info->flags    << QLMASK_SHIFT);
    *event_id |= (uint64_t)(info->nameid   << NAMEID_SHIFT);

    return PAPI_OK;
}

/** @class evt_id_to_info
  * @brief Convert event id to info. Function is needed for cuptip_event_enum.
  *
  * @param event_id
  *   An event id.
  * @param *info
  *   Structure which contains member variables of device, flags, and nameid.
*/
int evt_id_to_info(uint32_t event_id, event_info_t *info)
{
    info->device   = (int)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->flags    = (int)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid   = (int)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->device >= numDevicesOnMachine) {
        return PAPI_ENOEVNT;
    }    

    if (0 == (info->flags & DEVICE_FLAG) && info->device > 0) {
        return PAPI_ENOEVNT;
    }    

    if (info->nameid >= cuptiu_table_p->count) {
        return PAPI_ENOEVNT;
    }

    return PAPI_OK;
}

int cuptie_evt_enum(uint32_t *event_code, int modifier)
{

    int papi_errno = PAPI_OK;
    event_info_t info;
    SUBDBG("ENTER: event_code: %lu, modifier: %d\n", *event_code, modifier);

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if(cuptiu_table_p->count == 0) { 
                papi_errno = PAPI_ENOEVNT;
                break;
            }    
            info.device = 0; 
            info.flags = 0; 
            info.nameid = 0; 
            papi_errno = evt_id_create(&info, event_code);
            break;
        case PAPI_ENUM_EVENTS:
            papi_errno = evt_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }    
            if (cuptiu_table_p->count > info.nameid + 1) { 
                info.device = 0; 
                info.flags = 0; 
                info.nameid++;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }    
            papi_errno = PAPI_END;
            break;
        case PAPI_NTV_ENUM_UMASKS:
            papi_errno = evt_id_to_info(*event_code, &info);
            if (papi_errno != PAPI_OK) {
                break;
            }    
            if (info.flags == 0){
                info.device = 0; 
                info.flags = DEVICE_FLAG;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }    
            papi_errno = PAPI_END;
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }    
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
}

int cuptie_evt_code_to_descr(uint32_t event_code, char *descr, int len) 
{
    return PAPI_ENOIMPL;
}

int cuptie_evt_name_to_code(const char *name, uint32_t *event_code)
{
    int htable_errno, device, flags, nameid, papi_errno = PAPI_OK;
    cuptiu_event_and_metric_t *event;
    char base[PAPI_MAX_STR_LEN] = { 0 };
    SUBDBG("ENTER: name: %s, event_code: %p\n", name, event_code);

    papi_errno = evt_name_to_device(name, &device);
    if (papi_errno != PAPI_OK) {
        printf("Name to device: %d\n", papi_errno);
        goto fn_exit;
    }

    printf("Device is: %d\n", device);

    papi_errno = evt_name_to_basename(name, base, PAPI_MAX_STR_LEN);
    if (papi_errno != PAPI_OK) {
        printf("name to basename: %d\n", papi_errno);
        goto fn_exit;
    }
    printf("Base is: %s\n", base);

    htable_errno = htable_find(cuptiu_table_p->htable, base, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ? PAPI_ENOEVNT : PAPI_ECMP;
        printf("papi_errno after htable_find: %d\n", papi_errno);
        goto fn_exit;
    }

    /* flags = DEVICE_FLAG will need to be updated if more qualifiers are added,
       see implemtation in rocm (roc_profiler.c) */
    flags = (device >= 0) ? DEVICE_FLAG:0;
    if (flags == 0){
        papi_errno = PAPI_EINVAL;
        goto fn_exit;
    }

    nameid = (int) (event - cuptiu_table_p->events);
    event_info_t info = { device, flags, nameid };
    papi_errno = evt_id_create(&info, event_code);
    if (papi_errno != PAPI_OK) {
        printf("evt_id_create: %d\n", papi_errno);
        goto fn_exit;
    }

    papi_errno = evt_id_to_info(*event_code, &info);

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
}

int cuptie_evt_code_to_name(uint32_t event_code, char *name, int len)
{
    int papi_errno, str_len;
    event_info_t info;
    
    papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch (info.flags) {
        case (DEVICE_FLAG):
            str_len = snprintf(name, len, "%s:device=%i", cuptiu_table_p->events[info.nameid].name, info.device);
            if (str_len < 0 || str_len > len) {
                ERRDBG("String has not been completely written.\n");
                return PAPI_ESYS;
            }
            break;
        default:
            str_len = snprintf(name, len, "%s", cuptiu_table_p->events[info.nameid].name, info.device);
            if (str_len < 0 || str_len > len) {
                ERRDBG("String has not been completely written.\n");
                return PAPI_ESYS;
            }   
            break; 
   }
   return papi_errno;
}

int cuptie_shutdown(void)
{
    return PAPI_ENOIMPL;
}

int determine_dev_cc(int dev_id) 
{
    int cc_major;

    cudaDeviceGetAttributePtr(&cc_major, cudaDevAttrComputeCapabilityMajor, dev_id);

    if (cc_major > 7)
        return 0;
    else if (cc_major == 7)
        return 1;
    else 
        return 2;
}

/** @class evt_name_to_device
  * @brief Return the device number for a user provided Cuda native event.
  *        This can be done with a device qualifier present (:device=#) or
  *        we internally find the first device the native event exists for.
  * @param *name
  *   Cuda native event name with a device qualifier appended.
  * @param *device
  *   Device number.
*/
int evt_name_to_device(const char *name, int *device)
{
    char *p = strstr(name, ":device=");
    // User did provide :device=# qualifier
    if (p) {
        *device = (int) strtol(p + strlen(":device="), NULL, 10);
    }
    // User did not provide :device=# qualifier
    else {
        int i, htable_errno;
        cuptiu_event_and_metric_t *event;

        htable_errno = htable_find(cuptiu_table_p->htable, name, (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_EINVAL;
        }

        // Search for the first device the event exists for.
        for (i = 0; i < numDevicesOnMachine; ++i) {
            if (cuptiu_dev_check(event->device_map, i)) {
                *device = i;
                break;
            }
        }
    }
    return PAPI_OK;
}

/** @class evt_name_to_basename
  * @brief Convert a Cuda native event name with a device qualifer appended to 
  *        it, back to the base Cuda native event name provided by NVIDIA.
  * @param *name
  *   Cuda native event name with a device qualifier appended.
  * @param *base
  *   Base Cuda native event name (excludes device qualifier).
  * @param len
  *   Maximum alloted characters for base Cuda native event name. 
*/
static int evt_name_to_basename(const char *name, char *base, int len)
{
    char *p = strstr(name, ":");
    if (p) {
        if (len < (int)(p - name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t)(p - name));
    } else {
        if (len < (int) strlen(name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t) len);
    }
    return PAPI_OK;
}

/** @class find_same_chipname
  * @brief Check to see if chipnames are identical.
  * 
  * @param gpu_id
  *   A gpu id number, e.g 0, 1, 2, etc.
*/
int find_same_chipname(int gpu_id)
{
    int i;
    for (i = 0; i < gpu_id; i++) {
        if (!strcmp(cuptiu_table_p->avail_gpu_info[gpu_id].chipName, cuptiu_table_p->avail_gpu_info[i].chipName)) {
            return i;
        }    
    }    
    return -1;
}
