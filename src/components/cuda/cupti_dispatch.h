/**
 * @file    cupti_dispatch.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_DISPATCH_H__
#define __CUPTI_DISPATCH_H__

#include "cupti_utils.h"

typedef void *cuptid_ctl_t;
typedef void *cuptid_thread_info_t;

void cuptid_shutdown(void);
void cuptid_disabled_reason_get(const char **msg);
int cuptid_init(void);
int cuptid_thread_info_create(cuptid_thread_info_t *info);
int cuptid_thread_info_destroy(cuptid_thread_info_t *info);
int cuptid_control_create(cuptiu_event_table_t *event_names, void *thread_info, cuptid_ctl_t *pcupti_ctl);
int cuptid_control_destroy(cuptid_ctl_t *ctl);
int cuptid_control_start(cuptid_ctl_t ctl, void *thread_info);
int cuptid_control_stop(cuptid_ctl_t ctl, void *thread_info);
int cuptid_control_read(cuptid_ctl_t ctl, long long *values);
int cuptid_control_reset(cuptid_ctl_t ctl);
int cuptid_event_enum(cuptiu_event_table_t *all_evt_names);
int cuptid_event_name_to_descr(char *evt_name, char *descr);
#endif /* __CUPTI_DISPATCH_H__ */
