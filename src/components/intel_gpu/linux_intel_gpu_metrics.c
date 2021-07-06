/*
 * linux_intel_gpu_metrics.c:  Intel® Graphics Processing Unit (GPU) Component for PAPI.
 *
 * Copyright (c) 2020 Intel Corp. All rights reserved
 * Contributed by Peinan Zhang  <peinan.zhang@intel.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>

#include "inc/GPUMetricInterface.h"
#include "linux_intel_gpu_metrics.h"


#define DEFAULT_LOOPS     0                // collection infinitely until receive stop command
#define DEFAULT_MODE      METRIC_SUMMARY   // summary report

#define GPUDEBUG     SUBDBG

static DEVICE_HANDLE handle;
static int           num_device = 0;
static MetricInfo    metricInfoList;

static int           total_metrics        = 0;          // each for a metric set
static int           global_metrics_type  = TIME_BASED; // default type

papi_vector_t _intel_gpu_vector;

/************************* PAPI Functions **********************************/

static int
intel_gpu_init_thread(hwd_context_t *ctx)
{
    GPUDEBUG("Entering intel_gpu_init_thread\n");
    MetricContext *mContext = (MetricContext *)ctx;
    mContext->num_metrics = 0;
    for (int i=0; i< GPU_MAX_METRICS; i++) {
        mContext->metric_idx[i] = 0;
        mContext->metric_values[i] = 0;
    }
    return PAPI_OK;
}


static int 
intel_gpu_init_component(int cidx)
{
    GPUDEBUG("Entering intel_init_component\n");
    if (cidx < 0) {
        return  PAPI_EINVAL;
    }
    char *errStr = NULL;
    memset(_intel_gpu_vector.cmp_info.disabled_reason, 0, PAPI_MAX_STR_LEN);
    if ( _dl_non_dynamic_init != NULL ) {
        errStr = "The intel_gpu component does not support statically linking of libc.";
        strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        errStr, strlen(errStr));
        return PAPI_ENOSUPP;
    }

    if (GPUDetectDevice(&handle, &num_device) || (num_device==0)) {
        errStr = "The intel_gpu component does not detect metrics device.";
        strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        errStr, strlen(errStr));
        return PAPI_ENOSUPP;
    }
    char *envStr = NULL;
    envStr =  getenv(ENABLE_API_TRACING);
    if (!envStr) {
        envStr =  getenv(ENABLE_API_TRACING0);
    }
    if (envStr != NULL) {
       if (atoi(envStr) == 1) {
           global_metrics_type = EVENT_BASED;
       }
    }

    metricInfoList.numEntries = 0;
    if (GPUGetMetricList(handle, "", global_metrics_type, &metricInfoList)) {
        errStr = "The intel_gpu component failed on get all available metrics.";
        strncpy_se(_intel_gpu_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        errStr, strlen(errStr));
        GPUFreeDevice(handle);
        return PAPI_ENOSUPP;
    }; 
    total_metrics = metricInfoList.numEntries;
    GPUDEBUG("total metrics %d\n", total_metrics);

    _intel_gpu_vector.cmp_info.num_native_events = metricInfoList.numEntries;
    _intel_gpu_vector.cmp_info.num_cntrs         = GPU_MAX_COUNTERS;
    _intel_gpu_vector.cmp_info.num_mpx_cntrs     = GPU_MAX_COUNTERS;

    /* Export the component id */
    _intel_gpu_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}

/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int
intel_gpu_init_control_state( hwd_control_state_t * ctl )
{
    GPUDEBUG("Entering intel_gpu_control_state\n");

    if (!ctl) {
        return PAPI_EINVAL;
    }

    MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
    mCtlSt->metrics_type  = global_metrics_type;

    char *envStr = NULL;
    mCtlSt->interval = DEFAULT_SAMPLING_PERIOD;
    envStr =  getenv(METRICS_SAMPLING_PERIOD);
    if (envStr) {
        mCtlSt->interval = atoi(envStr);
        if (mCtlSt->interval < MINIMUM_SAMPLING_PERIOD) {
            mCtlSt->interval = DEFAULT_SAMPLING_PERIOD;
        }
    }
    mCtlSt->mode     = DEFAULT_MODE;   // default mode  (get aggragated value or average value overtime).
    mCtlSt->loops    = DEFAULT_LOOPS;

    return PAPI_OK;
}

static int
intel_gpu_update_control_state( hwd_control_state_t *ctl,
                         NativeInfo_t *native,
                         int count,
                         hwd_context_t *ctx )
{
    GPUDEBUG("Entering intel_gpu_control_state\n");
    (void)ctl;

    // use local maintained context,
    if (!count ||!native)  {
        return PAPI_OK;
    }

    MetricContext *mContext = (MetricContext *)ctx;

#if defined(_DEBUG)
    for (int i=0; i<count; i++) {
         GPUDEBUG("\t i=%d, ni_event %d,  ni_papi_code 0x%x, ni_position %d, ni_owners %d   \n", i, native[i].ni_event, native[i].ni_papi_code, native[i].ni_position, native[i].ni_owners);
    }
#endif

    uint32_t nmetrics = mContext->num_metrics;
    uint32_t midx = 0;
    int ni = 0;
    for (ni = 0; ni < count; ni++) {
       int index = native[ni].ni_event;
       // check whether this metric is in the list
       for (midx=0; midx<nmetrics; midx++) {
           if (mContext->metric_idx[midx] == index) {
               GPUDEBUG("metric code %d: already in the list, ignore\n", index); 
               break;
           }
       }
       if (midx == nmetrics) {
           // add this metric
           mContext->metric_idx[nmetrics] = index;
           mContext->metric_values[nmetrics] = 0;
           native[ni].ni_position = nmetrics;
           nmetrics++;
       }
    }
    // add this metric
    mContext->num_metrics = nmetrics;
    return PAPI_OK;
}


static int 
intel_gpu_start( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
    GPUDEBUG("Entering intel_gpu_start\n");

    MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
    MetricContext *mContext = (MetricContext *)ctx;

    int             ret    = 0;
    if (mContext->num_metrics == 0) {
        GPUDEBUG("intel_gpu_start : No metric selected, abort.\n");
        return PAPI_EINVAL;
    }

    char **metrics = calloc(mContext->num_metrics, sizeof(char *));
    if (!metrics) {
        GPUDEBUG("intel_gpu_start : insufficient memory, abort.\n");
        return PAPI_ENOMEM;
    }

    for (unsigned i=0; i<mContext->num_metrics; i++) {
        int idx = mContext->metric_idx[i];
        if ((idx < 0) || (idx >= metricInfoList.numEntries)) {
            GPUDEBUG("intel_gpu_start : failed on add metric with idx: %d\n", idx);
            free(metrics);
            return PAPI_ENOSUPP;
        }
        metrics[i] = metricInfoList.infoEntries[idx].name; 
    }
    mContext->num_reports = 0;
    
    ret = GPUEnableMetrics(handle, metrics, mContext->num_metrics, 
                   mCtlSt->metrics_type, mCtlSt->interval, mCtlSt->loops);
    free(metrics);
    if (ret) {
        GPUDEBUG("intel_gpu_start on EnableMetrics failed, return 0x%x \n", PAPI_ENOSUPP);
        return PAPI_ENOSUPP;
    }
    return PAPI_OK;
}

static int 
intel_gpu_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    GPUDEBUG("Entering intel_gpu_stop\n");

    (void)ctx;
    MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
    int             ret    = PAPI_OK;

    ret = GPUDisableMetricGroup(handle, "", mCtlSt->metrics_type);
    if (ret) { 
        GPUDEBUG("intel_gpu_stop : failed with ret %d\n", ret);
        return PAPI_EINVAL;
    }
    return PAPI_OK;
}

static int 
intel_gpu_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags )
{
    GPUDEBUG("Entering intel_gpu_read\n");

    (void)flags;

    MetricCtlState *mCtlSt      = (MetricCtlState *)ctl;
    MetricContext  *mContext    = (MetricContext *)ctx;
    MetricData     *data        = NULL;
    MetricData     *reports     = NULL;
    uint32_t       numReports   = 0;
    int            ret          = PAPI_OK;

    if (!events) {
        return PAPI_EINVAL;
    }

    if (mContext->num_metrics == 0) {
        GPUDEBUG("intel_gpu_read: no metric is selected\n");
        return PAPI_OK;
    }
    // read data
    reports = GPUReadMetricData(handle, mCtlSt->mode, &numReports);
    if (!reports) {
        GPUDEBUG("intel_gpu_read failed\n");
        return PAPI_EINVAL;
    }
    if (!numReports) {
        freeMetricData(reports, numReports); 
        GPUDEBUG("intel_gpu_read: no data available\n");
        return PAPI_EINVAL;
    }

    // take the most recent one, in summary, it expects numReports is 1
    data = &reports[numReports-1];

    uint32_t i=0;
    for (i=0; i<data->numEntries; i++) {
        if (!data->dataEntries[i].type) {
            mContext->metric_values[i] = (long long)data->dataEntries[i].value.ival;
        } else {
            mContext->metric_values[i] = (long long)data->dataEntries[i].value.fpval;
        }
    }

    mContext->num_reports = numReports;
    *events = mContext->metric_values;
    freeMetricData(reports, numReports);
    return ret;
}

static int
intel_gpu_shutdown_thread( hwd_context_t *ctx )
{
    (void) ctx;
    GPUDEBUG("Entering intel_gpu_shutdown_thread\n" );
    return PAPI_OK;
}

static int
intel_gpu_shutdown_component(void)
{
    GPUDEBUG("Entering intel_gpu_shutdown_component\n");
    GPUFreeDevice(handle);
    return PAPI_OK;
}

/* 
 * reset function will reset the global accumualted metrics values
 */
static int 
intel_gpu_reset( hwd_context_t *ctx, hwd_control_state_t *ctl)
{

    GPUDEBUG("Entering intel_gpu_reset\n");

    (void)ctx;
    (void)ctl;

    GPUSetMetricControl(handle, METRIC_RESET);
    return PAPI_OK;
}

/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
                        PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
*/
static int
intel_gpu_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    GPUDEBUG("Entering intel_gpu_ctl\n");

    (void) ctx;
    (void) code;
    (void) option;
    return PAPI_OK;
}



static int
intel_gpu_set_domain( hwd_control_state_t * ctl, int domain )
{
    GPUDEBUG("Entering intel_gpu_set_domain\n");

    if (!ctl) {
        return PAPI_EINVAL;
    }
    MetricCtlState *mCtlSt = (MetricCtlState *)ctl;
    mCtlSt->domain = domain;
    return PAPI_OK;

}


static int 
intel_gpu_ntv_enum_events(uint32_t *EventCode, int modifier )
{
    GPUDEBUG("Entering intel_gpu_ntv_enum_events\n");
    int index = 0;

    if (!EventCode)  {
        return PAPI_EINVAL;
    }
    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            *EventCode = 0;
            break;
        case PAPI_ENUM_EVENTS:
            index = *EventCode;
            if ( (index < 0 ) || (index >= (total_metrics-1)) ) {
                return PAPI_ENOEVNT;
            }
            *EventCode = *EventCode + 1;
            break;
        default:
            return PAPI_EINVAL;
    } 
    return PAPI_OK;
}



static int 
intel_gpu_ntv_code_to_name( uint32_t EventCode, char *name, int len )
{
    GPUDEBUG("Entering intel_gpu_ntv_code_to_name\n");
    int index = EventCode;

    if( ( index < 0 ) || ( index >= total_metrics ) || !name || !len ) {
        return PAPI_EINVAL;
    }

    memset(name, 0, len);
    strncpy_se(name,  len, metricInfoList.infoEntries[index].name, strlen(metricInfoList.infoEntries[index].name));
    return PAPI_OK;
}

static int 
intel_gpu_ntv_code_to_descr( uint32_t EventCode, char *desc, int len )
{
    GPUDEBUG("Entering intel_gpu_ntv_code_to_descr\n");
    int index = EventCode;

    if( ( index < 0 ) || ( index >= total_metrics ) || !desc || !len) {
        return PAPI_EINVAL;
    }

    memset(desc, 0, len);
    strncpy_se(desc, len, metricInfoList.infoEntries[index].desc, MAX_STR_LEN-1);
    return PAPI_OK;
}


static int
intel_gpu_ntv_name_to_code( const char *name, uint32_t *event_code)
{
    GPUDEBUG("Entering intel_gpu_ntv_name_to_code\n");
    if( !name || !event_code) {
        return PAPI_EINVAL;
    }

    for (int i=0; i<total_metrics; i++) {
        if (strncmp(metricInfoList.infoEntries[i].name, name, MAX_STR_LEN) == 0) {
           *event_code = i;
           return PAPI_OK;
        }
    }
    return PAPI_ENOEVNT;
}

static int
intel_gpu_ntv_code_to_info(uint32_t EventCode, PAPI_event_info_t *info)
{
    GPUDEBUG("Entering intel_gpu_ntv_code_to_info\n");
    int  index = EventCode;
    if( ( index < 0 ) || ( index >= total_metrics ) || !info) {
        return PAPI_EINVAL;
    }

    info->event_code = EventCode;
    strncpy_se(info->symbol, PAPI_HUGE_STR_LEN,
            metricInfoList.infoEntries[index].name, MAX_STR_LEN-1);
    // short description could be truncated due to the longer string size of metric name
    strncpy_se(info->short_descr, PAPI_MIN_STR_LEN,
           metricInfoList.infoEntries[index].name, 
           strlen(metricInfoList.infoEntries[index].name));
    strncpy_se(info->long_descr, PAPI_HUGE_STR_LEN,
           metricInfoList.infoEntries[index].desc,  MAX_STR_LEN-1);
    info->component_index = _intel_gpu_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

/* Our component vector */

papi_vector_t _intel_gpu_vector = {
  .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "intel_gpu",
      .short_name = "intel_gpu",
      .version = "1.0",
      .description = "Intel GPU performance metrics",
      .default_domain = PAPI_DOM_ALL,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_SYS,
      .available_granularities = PAPI_GRN_SYS,

      .num_mpx_cntrs = GPU_MAX_COUNTERS,

      /* component specific cmp_info initializations */
      .fast_virtual_timer = 0,
      .attach = 0,
      .attach_must_ptrace = 0,
      .cpu = 0,
      .inherit = 0,
      .cntr_umasks = 0,

  },

  /* sizes of framework-opaque component-private structures */
  .size = {
      .context = sizeof (MetricContext),
      .control_state = sizeof (MetricCtlState),
      .reg_value = sizeof ( int ),
      .reg_alloc = sizeof ( int ),
  },

  /* function pointers in this component */
  .init_thread           = intel_gpu_init_thread,
  .init_component        = intel_gpu_init_component,
  .init_control_state    = intel_gpu_init_control_state,
  .update_control_state  =  intel_gpu_update_control_state,
  .start =                 intel_gpu_start,
  .stop =                  intel_gpu_stop,
  .read =                  intel_gpu_read,
  .shutdown_thread       = intel_gpu_shutdown_thread,
  .shutdown_component    = intel_gpu_shutdown_component,
  .ctl =                   intel_gpu_ctl,

  /* these are dummy implementation */
  .set_domain =            intel_gpu_set_domain,
  .reset =                 intel_gpu_reset,

  /* from counter name mapper */
  .ntv_enum_events =   intel_gpu_ntv_enum_events,
  .ntv_code_to_name =  intel_gpu_ntv_code_to_name,
  .ntv_name_to_code =  intel_gpu_ntv_name_to_code,
  .ntv_code_to_descr = intel_gpu_ntv_code_to_descr,
  .ntv_code_to_info =  intel_gpu_ntv_code_to_info,
};


