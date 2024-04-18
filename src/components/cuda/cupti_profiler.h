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
int cuptip_event_name_to_descr(const char *evt_name, char *description);
int cuptip_evt_enum(uint64_t *event_code, int modifier);
int cuptip_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info);
int cuptip_evt_code_to_descr(uint64_t event_code, char *descr, int len);
int cuptip_evt_code_to_name(uint64_t event_code, char *name, int len);
int cuptip_evt_name_to_code(const char *name, uint64_t *event_code);
int init_event_table(void);
int cuptip_shutdown(void);
#endif  /* __CUPTI_PROFILER_H__ */
