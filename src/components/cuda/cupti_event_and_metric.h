/**
 * @file    cupti_events.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

#include "cupti_utils.h"
#include "papi_cupti_common.h"

#include <stdint.h>

typedef struct cuptie_control_s     *cuptie_control_t;

/* init and shutdown interfaces */
int cuptie_init(void);
int cuptie_shutdown(void);

/* native event interfaces */
int cuptie_evt_enum(uint32_t *event_code, int modifier); 
int cuptie_evt_code_to_descr(uint32_t event_code, char *descr, int len);
int cuptie_evt_name_to_code(const char *name, uint32_t *event_code);
int cuptie_evt_code_to_name(uint32_t event_code, char *name, int len);
int cuptie_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptie_ctx_create(cuptic_info_t thr_info, cuptie_control_t *pstate, uint32_t *events_id, int num_events);
int cuptie_ctx_start(cuptie_control_t ctl);
int cuptie_ctx_read(cuptie_control_t ctl, long long **counters);
int cuptie_ctx_reset(cuptie_control_t ctl);
int cuptie_ctx_stop(cuptie_control_t ctl);
int cuptie_ctx_destroy(cuptie_control_t *pctl);

typedef enum 
{
   event_api = 0,  
   metric_api, 
} cupti_profiling_api;

typedef struct gpu_record_event_and_metric_s {
    char chipName[PAPI_MIN_STR_LEN];
//    int totalMetricCount;
//    char **metricNames;
    CUdevice deviceHandle;
} gpu_record_event_and_metric_t;

typedef struct event_and_metric_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_HUGE_STR_LEN];
    cupti_profiling_api api;
    cuptiu_bitmap_t device_map;
} cuptiu_event_and_metric_t;

typedef struct event_and_metric_table_s {
    unsigned int count;
    unsigned int capacity;
    CUpti_EventID eventIDs[30];
    int countOfEventIDs;
    CUpti_MetricID metricIDs[30];
    int countOfMetricIDs;
    int cuda_devs[30];
    int evt_pos[30];
    gpu_record_event_and_metric_t *avail_gpu_info;
    cuptiu_event_and_metric_t *events;
    void *htable;
} cuptiu_event_and_metric_table_t;

#endif  /* __CUPTI_EVENTS_H__ */
