#ifndef __ROC_PROFILER_V2_H__
#define __ROC_PROFILER_V2_H__

typedef struct rocd_ctx *rocp_ctx_t;

/* init and shutdown interfaces */
int rocp2_init_environment(void);
int rocp2_init(void);
int rocp2_shutdown(void);

/* native event interfaces */
int rocp2_evt_enum(unsigned int *event_code, int modifier);
int rocp2_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int rocp2_evt_name_to_code(const char *name, unsigned int *event_code);
int rocp2_evt_code_to_name(unsigned int event_code, char *name, int len);

/* profiler context handling interfaces */
int rocp2_ctx_open(unsigned int *events_id, int num_events, rocp_ctx_t *ctx);
int rocp2_ctx_close(rocp_ctx_t ctx);
int rocp2_ctx_start(rocp_ctx_t ctx);
int rocp2_ctx_stop(rocp_ctx_t ctx);
int rocp2_ctx_read(rocp_ctx_t ctx, long long **counters);
int rocp2_ctx_reset(rocp_ctx_t ctx);

#endif /* End of __ROC_PROFILER_V2_H__ */
