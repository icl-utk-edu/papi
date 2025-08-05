/**
 * @file    cupti_profiler.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.) 
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __CUPTI_PROFILER_H__
#define __CUPTI_PROFILER_H__

typedef struct cuptip_control_s     *cuptip_control_t;

/* used to determine collection method in cupti_profiler.c, see cuptip_ctx_read */
#define CUDA_AVG 0x1
#define CUDA_MAX 0x2
#define CUDA_MIN 0x3
#define CUDA_SUM 0x4
#define CUDA_DEFAULT 0x5

/* init and shutdown interfaces */
int cuptip_init(void);
int cuptip_shutdown(void);

/* native event interfaces */
int cuptip_evt_enum(uint32_t *event_code, int modifier);
int cuptip_evt_code_to_descr(uint32_t event_code, char *descr, int len);
int cuptip_evt_name_to_code(const char *name, uint32_t *event_code);
int cuptip_evt_code_to_name(uint32_t event_code, char *name, int len);
int cuptip_evt_code_to_info(uint32_t event_code, PAPI_event_info_t *info);

/* profiling context handling interfaces */
int cuptip_ctx_create(cuptic_info_t thr_info, cuptip_control_t *pstate,  uint32_t *events_id, int num_events);
int cuptip_ctx_destroy(cuptip_control_t *pstate);
int cuptip_ctx_start(cuptip_control_t state);
int cuptip_ctx_stop(cuptip_control_t state);
int cuptip_ctx_read(cuptip_control_t state, long long **counters);
int cuptip_ctx_reset(cuptip_control_t state);

/*
void init_vector(StringVector *vec);
int push_back(StringVector *vec, const char *str);
void free_vector(StringVector *vec);

typedef struct gpu_record_s {
    char chipName[PAPI_MIN_STR_LEN];
    int totalMetricCount;
    char **metricNames;
} gpu_record_t;

typedef struct {
    char **arrayMetricStatistics ;   
    size_t size;   
    size_t capacity;
} StringVector;

typedef struct event_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char basenameWithStatReplaced[PAPI_2MAX_STR_LEN];
    char desc[PAPI_HUGE_STR_LEN];
    StringVector * stat;
    cuptiu_bitmap_t device_map;
} cuptiu_event_t;

typedef struct event_table_s {
    unsigned int count;
    unsigned int event_stats_count;
    unsigned int capacity;
    char cuda_evts[30][PAPI_2MAX_STR_LEN];
    int cuda_devs[30];
    int evt_pos[30];
    gpu_record_t *avail_gpu_info;
    cuptiu_event_t *events;
    StringVector   *event_stats;
    void *htable;
} cuptiu_event_table_t;
*/

#endif  /* __CUPTI_PROFILER_H__ */
