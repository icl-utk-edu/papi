/**
 * @file    cupti_profiler.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_PROFILER_H__
#define __CUPTI_PROFILER_H__

#include "cupti_utils.h"

typedef struct cuptip_control_s     *cuptip_control_t;

int cuptip_init(void);
int cuptip_control_create(cuptiu_event_table_t *event_names, cuptic_info_t thr_info, cuptip_control_t *pstate);
int cuptip_control_destroy(cuptip_control_t *pstate);
int cuptip_control_start(cuptip_control_t state);
int cuptip_control_stop(cuptip_control_t state);
int cuptip_control_read(cuptip_control_t state, long long *values);
int cuptip_control_reset(cuptip_control_t state);
int cuptip_event_enum(cuptiu_event_table_t *all_evt_names);
int cuptip_event_name_to_descr(const char *evt_name, char *description);
int cuptip_evt_name_to_info(const char *evt_name, PAPI_event_info_t *info);
int cuptiu_format_evt_name(const char *evt_name, char *qualifier_name, char *qualifier, int len);
int cuptiu_get_num_devices(int number_of_gpus, char *formatted_devices);
int cuptiu_collect_qualifiers(int num_of_events,  const char * const*all_event_names, cuptiu_event_info_t *head, int len);
int cuptip_shutdown(void);
#endif  /* __CUPTI_PROFILER_H__ */
