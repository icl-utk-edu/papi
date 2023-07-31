/**
 * @file    cupti_dispatch.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_DISPATCH_H__
#define __CUPTI_DISPATCH_H__

#include "cupti_utils.h"

extern unsigned int _cuda_lock;

typedef void *cuptid_ctl_t;
typedef void *cuptid_info_t;
typedef cuptiu_event_table_t *ntv_event_table_t;
typedef cuptiu_event_t *ntv_event_t;

int cuptid_shutdown(void);
void cuptid_disabled_reason_get(const char **msg);
int cuptid_init(void);
int cuptid_thread_info_create(cuptid_info_t *info);
int cuptid_thread_info_destroy(cuptid_info_t *info);
int cuptid_control_create(ntv_event_table_t event_names, cuptid_info_t thread_info, cuptid_ctl_t *pcupti_ctl);
int cuptid_control_destroy(cuptid_ctl_t *ctl);
int cuptid_control_start(cuptid_ctl_t ctl);
int cuptid_control_stop(cuptid_ctl_t ctl);
int cuptid_control_read(cuptid_ctl_t ctl, long long *values);
int cuptid_control_reset(cuptid_ctl_t ctl);
int cuptid_event_enum(ntv_event_table_t all_evt_names);
int cuptid_event_name_to_descr(char *evt_name, char *descr);

void cuptid_event_table_destroy(ntv_event_table_t *evt_table);
int cuptid_event_table_create(ntv_event_table_t *evt_table);
int cuptid_event_table_select_by_idx(ntv_event_table_t src, int count, int *idcs, ntv_event_table_t *pevt_names);
int cuptid_event_table_find_name(ntv_event_table_t evt_table, const char *evt_name, ntv_event_t *found_rec);
int cuptid_event_table_insert_record(ntv_event_table_t evt_table, const char *evt_name, unsigned int evt_code, int evt_pos);
int cuptid_event_table_get_item(ntv_event_table_t evt_table, unsigned int evt_idx, ntv_event_t *record);

#endif /* __CUPTI_DISPATCH_H__ */
