/**
 * @file    cupti_dispatch.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_DISPATCH_H__
#define __CUPTI_DISPATCH_H__

#include "cupti_utils.h"
#include "cupti_config.h"

extern unsigned int _cuda_lock;

typedef struct cuptip_control_s *cuptip_control_t;
//typedef void *cuptid_ctl_t;
typedef void *cuptid_info_t;
typedef cuptiu_event_table_t *ntv_event_table_t;
typedef cuptiu_event_t *ntv_event_t;

/* init and shutdown interfaces */
int cuptid_init(void);
int cuptid_shutdown(void);

/* native event interfaces */
int cuptid_evt_enum(uint64_t *event_code, int modifier);
int cuptid_evt_code_to_descr(uint64_t event_code, char *descr, int len);
int cuptid_evt_name_to_code(const char *name, uint64_t *event_code);
int cuptid_evt_code_to_name(uint64_t event_code, char *name, int len);
int cuptid_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptid_ctx_create(cuptid_info_t thread_info, cuptip_control_t *pcupti_ctl, uint64_t *events_id, int num_events);
int cuptid_ctx_start(cuptip_control_t ctl);
int cuptid_ctx_read(cuptip_control_t ctl, long long **counters);
int cuptid_ctx_reset(cuptip_control_t state);
int cuptid_ctx_stop(cuptip_control_t ctl);
int cuptid_ctx_destroy(cuptip_control_t *ctl);

/* thread interfaces */
int cuptid_thread_info_create(cuptid_info_t *info);
int cuptid_thread_info_destroy(cuptid_info_t *info);

/* cuda event table interfaces*/
void cuptid_event_table_destroy(ntv_event_table_t *evt_table);
int cuptid_event_table_create(ntv_event_table_t *evt_table);
int cuptid_event_table_select_by_idx(ntv_event_table_t src, int count, int *idcs, ntv_event_table_t *pevt_names);
int cuptid_event_table_find_name(ntv_event_table_t evt_table, const char *evt_name, ntv_event_t *found_rec);
int cuptid_event_table_insert_record(ntv_event_table_t evt_table, const char *evt_name, unsigned int evt_code, int evt_pos);
int cuptid_event_table_get_item(ntv_event_table_t evt_table, unsigned int evt_idx, ntv_event_t *record);

/* misc. */
void cuptid_disabled_reason_get(const char **msg);

#endif /* __CUPTI_DISPATCH_H__ */
