/**
 * @file    rocp.h
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 */

#ifndef __ROCP_H__
#define __ROCP_H__

#include "common.h"

#define ROCM_PROFILE_SAMPLING_MODE (0)

typedef struct rocp_ctx *rocp_ctx_t;

int rocp_init_environment(const char **err_string);
int rocp_init(ntv_event_table_t *ntv_table, const char **err_string);
int rocp_evt_enum(unsigned int *event_code, int modifier);
int rocp_evt_get_descr(unsigned int event_code, char *descr, int len);
int rocp_evt_name_to_code(const char *name, unsigned int *event_code);
int rocp_evt_code_to_name(unsigned int event_code, char *name, int len);
int rocp_ctx_open(ntv_event_table_t *ntv_table, unsigned int *events_id,
                  int num_events, rocp_ctx_t *ctx);
int rocp_ctx_open_v2(unsigned int *events_id, int num_events, rocp_ctx_t *ctx);
int rocp_err_get_last(const char **err_string);
int rocp_ctx_close(rocp_ctx_t ctx);
int rocp_ctx_start(rocp_ctx_t ctx);
int rocp_ctx_stop(rocp_ctx_t ctx);
int rocp_ctx_read(rocp_ctx_t ctx, long long **counts);
int rocp_ctx_reset(rocp_ctx_t ctx);
int rocp_shutdown(ntv_event_table_t *ntv_table);

#endif /* End of __ROCP_H__ */
