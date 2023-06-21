/**
 * @file    cupti_events.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

#include "lcuda_common.h"

int cuptie_init(const char **pdisabled_reason);
int cuptie_control_create(ntv_event_table_t *event_names, void *thr_info, void **pctl);
int cuptie_control_destroy(void **pctl);
int cuptie_control_start(void *ctl, void *thr_info);
int cuptie_control_stop(void *ctl, void *thr_info);
int cuptie_control_read(void *ctl, long long *values);
int cuptie_control_reset(void *ctl);
int cuptie_event_enum(ntv_event_table_t *all_evt_names);
int cuptie_event_name_to_descr(const char *evt_name, char *description);
void cuptie_shutdown(void);

#endif  // __CUPTI_EVENTS_H__
