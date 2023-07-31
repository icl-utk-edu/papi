/**
 * @file    cupti_events.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

#include "cupti_utils.h"

typedef void *cuptie_control_t;

int cuptie_init(void);
int cuptie_control_create(cuptiu_event_table_t *event_names, void *thr_info, cuptie_control_t *pctl);
int cuptie_control_destroy(cuptie_control_t *pctl);
int cuptie_control_start(cuptie_control_t ctl);
int cuptie_control_stop(cuptie_control_t ctl);
int cuptie_control_read(cuptie_control_t ctl, long long *values);
int cuptie_control_reset(cuptie_control_t ctl);
int cuptie_event_enum(cuptiu_event_table_t *all_evt_names);
int cuptie_event_name_to_descr(const char *evt_name, char *description);
int cuptie_shutdown(void);

#endif  /* __CUPTI_EVENTS_H__ */
