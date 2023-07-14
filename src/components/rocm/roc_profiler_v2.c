#include <rocprofiler.h>
#include "roc_profiler_v2.h"
#include "roc_common.h"
#include "htable.h"

#define EVENT_TYPE_MULTI_INSTANCE (0x80000000)
#define EVENT_INSTANCE_VALUE_MASK (0x0000FFFF)
#define EVENT_INSTANCE_TYPE_MASK  EVENT_TYPE_MULTI_INSTANCE
#define GET_INSTANCE_VALUE(inst)  (inst & EVENT_INSTANCE_VALUE_MASK)
#define GET_INSTANCE_TYPE(inst)   (inst & EVENT_INSTANCE_TYPE_MASK)

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char desc[PAPI_2MAX_STR_LEN];
    unsigned int event_id;
    unsigned int gpu_id;
    unsigned int instance;
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int num_events;
} ntv_event_table_t;

struct rocd_ctx {
    union {
        struct {
            int state;
            rocc_bitmap_t gpu_map;
        } intercept;
        struct {
            int state;
            char **events_name;
            unsigned int *events_id;
            long long *counters;
            int num_events;
            rocprofiler_session_id_t *sessions_id;
            rocc_bitmap_t gpu_map;
        } sampling;
    } u;
};

unsigned int rocm_prof_mode;
unsigned int _rocm_lock;

/* Init interface */
static rocprofiler_status_t (*rocp_init)(void);
static rocprofiler_status_t (*rocp_fini)(void);

/* Counters interface */
static rocprofiler_status_t (*rocp_iter_counters)(rocprofiler_counters_info_callback_t callback);

/* Device profiling interface */
static rocprofiler_status_t (*rocp_dev_prof_sess_create)(const char **metrics, uint64_t count, rocprofiler_session_id_t *id, int cpu_idx, int gpu_idx);
static rocprofiler_status_t (*rocp_dev_prof_sess_start)(rocprofiler_session_id_t id);
static rocprofiler_status_t (*rocp_dev_prof_sess_poll)(rocprofiler_session_id_t id, rocprofiler_device_profile_metric_t *data);
static rocprofiler_status_t (*rocp_dev_prof_sess_stop)(rocprofiler_session_id_t id);
static rocprofiler_status_t (*rocp_dev_prof_sess_destroy)(rocprofiler_session_id_t id);

static int load_rocp_sym(void);
static int unload_rocp_sym(void);
static int init_event_table(void);
static int finalize_event_table(void);
static int sampling_ctx_open(unsigned int *, int, rocp_ctx_t *);
static int sampling_ctx_close(rocp_ctx_t);
static int sampling_ctx_start(rocp_ctx_t);
static int sampling_ctx_stop(rocp_ctx_t);
static int sampling_ctx_read(rocp_ctx_t, long long **);
static int sampling_ctx_reset(rocp_ctx_t);
static int intercept_ctx_open(unsigned int *, int, rocp_ctx_t *);
static int intercept_ctx_close(rocp_ctx_t);
static int intercept_ctx_start(rocp_ctx_t);
static int intercept_ctx_stop(rocp_ctx_t);
static int intercept_ctx_read(rocp_ctx_t, long long **);
static int intercept_ctx_reset(rocp_ctx_t);

static void *rocp_dlp;
static void *htable;
static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p;

int
rocp2_init_environment(void)
{
    rocm_prof_mode = ROCM_PROFILE_SAMPLING_MODE;
    return PAPI_OK;
}

int
rocp2_init(void)
{
    int papi_errno;

    papi_errno = load_rocp_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocprofiler_status_t rocp_errno = rocp_init();
    if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    htable_init(&htable);

    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_table_p = &ntv_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    rocp_fini();
    unload_rocp_sym();
    goto fn_exit;
}

int
rocp2_shutdown(void)
{
    finalize_event_table();
    ntv_table_p = NULL;
    htable_shutdown(htable);
    rocp_fini();
    unload_rocp_sym();
    return PAPI_OK;
}

int
rocp2_evt_enum(unsigned int *event_code, int modifier)
{
    int papi_errno = PAPI_OK;

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if (ntv_table_p->num_events == 0) {
                papi_errno = PAPI_ENOEVNT;
            }
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code + 1 < (unsigned int) ntv_table_p->num_events) {
                ++(*event_code);
            } else {
                papi_errno = PAPI_END;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int
rocp2_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->num_events) {
        return PAPI_ENOEVNT;
    }
    snprintf(descr, len, "%s", ntv_table_p->events[event_code].desc);
    return PAPI_OK;
}

int
rocp2_evt_name_to_code(const char *name, unsigned int *event_code)
{
    ntv_event_t *event;
    if (htable_find(htable, name, (void **) &event) != HTABLE_SUCCESS) {
        return PAPI_ENOEVNT;
    }
    *event_code = event->event_id;
    return PAPI_OK;
}

int
rocp2_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->num_events) {
        return PAPI_ENOEVNT;
    }

    ntv_event_t *event = &ntv_table_p->events[event_code];
    if (GET_INSTANCE_TYPE(event->instance) != EVENT_TYPE_MULTI_INSTANCE) {
        snprintf(name, len, "%s:device=%u", event->name, event->gpu_id);
    } else {
        snprintf(name, len, "%s:device=%u:instance=%u", event->name, event->gpu_id, GET_INSTANCE_VALUE(event->instance));
    }

    return PAPI_OK;
}

int
rocp2_ctx_open(unsigned int *events_id, int num_events, rocp_ctx_t *ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_open(events_id, num_events, ctx);
    }
    return intercept_ctx_open(events_id, num_events, ctx);
}

int
rocp2_ctx_close(rocp_ctx_t ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_close(ctx);
    }
    return intercept_ctx_close(ctx);
}

int
rocp2_ctx_start(rocp_ctx_t ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_start(ctx);
    }
    return intercept_ctx_start(ctx);
}

int
rocp2_ctx_stop(rocp_ctx_t ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_stop(ctx);
    }
    return intercept_ctx_stop(ctx);
}

int
rocp2_ctx_read(rocp_ctx_t ctx, long long **counters)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_read(ctx, counters);
    }
    return intercept_ctx_read(ctx, counters);
}

int
rocp2_ctx_reset(rocp_ctx_t ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_reset(ctx);
    }
    return intercept_ctx_reset(ctx);
}

int
load_rocp_sym(void)
{
    int papi_errno = PAPI_OK;
    char pathname[PATH_MAX] = "librocprofiler64.so";

    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root != NULL) {
        sprintf(pathname, "%s/lib/librocprofiler64.so", rocm_root);
    }

    char *rocp_path = getenv("HSA_TOOLS_LIB");
    if (rocp_path != NULL) {
        sprintf(pathname, "%s", rocp_path);
    }

    rocp_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rocp_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    rocp_init = dlsym(rocp_dlp, "rocprofiler_initialize");
    rocp_fini = dlsym(rocp_dlp, "rocprofiler_finalize");
    rocp_iter_counters = dlsym(rocp_dlp, "rocprofiler_iterate_counters");
    rocp_dev_prof_sess_create = dlsym(rocp_dlp, "rocprofiler_device_profiling_session_create");
    rocp_dev_prof_sess_start = dlsym(rocp_dlp, "rocprofiler_device_profiling_session_start");
    rocp_dev_prof_sess_poll = dlsym(rocp_dlp, "rocprofiler_device_profiling_session_poll");
    rocp_dev_prof_sess_stop = dlsym(rocp_dlp, "rocprofiler_device_profiling_session_stop");
    rocp_dev_prof_sess_destroy = dlsym(rocp_dlp, "rocprofiler_device_profiling_session_destroy");

    int rocp_not_initialized = (
        !rocp_init ||
        !rocp_fini ||
        !rocp_iter_counters ||
        !rocp_dev_prof_sess_create ||
        !rocp_dev_prof_sess_start ||
        !rocp_dev_prof_sess_poll ||
        !rocp_dev_prof_sess_stop ||
        !rocp_dev_prof_sess_destroy
    );

    papi_errno = (rocp_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Error while loading rocprofiler symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

int
unload_rocp_sym(void)
{
    if (rocp_dlp == NULL) {
        return PAPI_OK;
    }

    rocp_init = NULL;
    rocp_fini = NULL;
    rocp_iter_counters = NULL;
    rocp_dev_prof_sess_create = NULL;
    rocp_dev_prof_sess_start = NULL;
    rocp_dev_prof_sess_poll = NULL;
    rocp_dev_prof_sess_stop = NULL;
    rocp_dev_prof_sess_destroy = NULL;

    dlclose(rocp_dlp);
    return PAPI_OK;
}

static int get_events_count(void);
static int get_events(void);

int
init_event_table(void)
{
    int papi_errno;

    papi_errno = get_events_count();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_table.events = papi_calloc(ntv_table.num_events, sizeof(ntv_event_t));
    if (ntv_table.events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = get_events();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    if (ntv_table.events) {
        papi_free(ntv_table.events);
    }
    ntv_table.num_events = 0;
    snprintf(error_string, PAPI_MAX_STR_LEN, "Error while intializing the event table");
    goto fn_exit;
}

static int get_events_count_callback(rocprofiler_counter_info_t counter, const char *gpu_name, uint32_t gpu_idx);

int
get_events_count(void)
{
    int papi_errno = PAPI_OK;
    rocprofiler_status_t rocp_errno;

    rocp_errno = rocp_iter_counters(get_events_count_callback);
    if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int
get_events_count_callback(rocprofiler_counter_info_t counter, const char *gpu_name __attribute__((unused)), uint32_t gpu_idx __attribute__((unused)))
{
    ntv_table.num_events += counter.instances_count;
    return ROCPROFILER_STATUS_SUCCESS;
}

static int get_events_callback(rocprofiler_counter_info_t counter, const char *gpu_name, uint32_t gpu_idx);

int
get_events(void)
{
    int papi_errno = PAPI_OK;
    rocprofiler_status_t rocp_errno;

    rocp_errno = rocp_iter_counters(get_events_callback);
    if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

int
get_events_callback(rocprofiler_counter_info_t counter, const char *gpu_name __attribute__((unused)), uint32_t gpu_idx)
{
    int i;
    static int count;

    for (i = 0; i < (int) counter.instances_count; ++i) {
        snprintf(ntv_table.events[count + i].name, PAPI_MAX_STR_LEN, "%s", counter.name);
        snprintf(ntv_table.events[count + i].desc, PAPI_MAX_STR_LEN, "%s", counter.description);
        ntv_table.events[count + i].event_id = count + i;
        ntv_table.events[count + i].gpu_id = gpu_idx;

        char event_name[PAPI_MAX_STR_LEN] = { 0 };
        if (counter.instances_count == 1) {
            ntv_table.events[count + i].instance = 0;
            snprintf(event_name, PAPI_MAX_STR_LEN, "%s:device=%u", counter.name, gpu_idx);
        } else {
            ntv_table.events[count + i].instance = (EVENT_TYPE_MULTI_INSTANCE | i);
            snprintf(event_name, PAPI_MAX_STR_LEN, "%s:device=%u:instance=%i", counter.name, gpu_idx, i);
        }
        htable_insert(htable, event_name, &ntv_table.events[count + i]);
    }

    count += i;
    return ROCPROFILER_STATUS_SUCCESS;
}

int
finalize_event_table(void)
{
    papi_free(ntv_table_p->events);
    ntv_table_p->num_events = 0;
    return PAPI_OK;
}

static int
event_id_to_dev_id(unsigned int event_id, unsigned int *dev_id)
{
    *dev_id = ntv_table_p->events[event_id].gpu_id;
    return PAPI_OK;
}

static int
init_sampling_ctx(unsigned int *events_id, int num_events, rocp_ctx_t *ctx)
{
    int papi_errno;
    char **events_name = NULL;
    long long *counters = NULL;
    rocprofiler_session_id_t *sessions_id = NULL;
    *ctx = NULL;

    rocc_bitmap_t bitmap;
    papi_errno = rocc_dev_get_map(event_id_to_dev_id, events_id, num_events, &bitmap);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    int num_gpus;
    papi_errno = rocc_dev_get_count(bitmap, &num_gpus);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    events_name = papi_calloc(num_events, sizeof(char *));
    if (events_name == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    counters = papi_calloc(num_events, sizeof(long long));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    sessions_id = papi_calloc(num_gpus, sizeof(*sessions_id));
    if (sessions_id == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    *ctx = papi_calloc(1, sizeof(rocp_ctx_t));
    if (*ctx == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int i;
    for (i = 0; i < num_events; ++i) {
        events_name[i] = ntv_table_p->events[events_id[i]].name;
    }

    (*ctx)->u.sampling.state = 0;
    (*ctx)->u.sampling.events_id = events_id;
    (*ctx)->u.sampling.events_name = events_name;
    (*ctx)->u.sampling.num_events = num_events;
    (*ctx)->u.sampling.counters = counters;
    (*ctx)->u.sampling.sessions_id = sessions_id;
    (*ctx)->u.sampling.gpu_map = bitmap;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (events_name) {
        papi_free(events_name);
    }
    if (counters) {
        papi_free(counters);
    }
    if (sessions_id) {
        papi_free(sessions_id);
    }
    if (*ctx) {
        papi_free(*ctx);
    }
    goto fn_exit;
}

static int
get_gpu_events_count(unsigned int *events_id, int num_events, unsigned int gpu_id, int *num_gpu_events)
{
    int i = 0, j = 0;

    while (i < num_events && ntv_table_p->events[events_id[i]].gpu_id != gpu_id) {
        ++i;
    }

    while (j < num_events && ntv_table_p->events[events_id[j]].gpu_id == gpu_id) {
        ++j;
    }

    *num_gpu_events = j - i + 1;
    return PAPI_OK;
}

static int
open_sampling_ctx(rocp_ctx_t ctx)
{
    int papi_errno = PAPI_OK;

    int num_gpus;
    rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

    int i, j;
    for (i = 0, j = 0; i < num_gpus; ++i) {
        int gpu_id, num_events;
        rocc_dev_get_id(ctx->u.sampling.gpu_map, i, &gpu_id);
        const char **events_name = (const char **) &ctx->u.sampling.events_name[j];
        get_gpu_events_count(ctx->u.sampling.events_id, ctx->u.sampling.num_events, gpu_id, &num_events);
        rocprofiler_session_id_t *sid = &ctx->u.sampling.sessions_id[j];

        rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_create(events_name, num_events, sid, 0, gpu_id);
        if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        j += num_events;
    }

    papi_errno = rocc_dev_acquire(ctx->u.sampling.gpu_map);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    for (j = 0; j < i; ++j) {
        rocp_dev_prof_sess_destroy(ctx->u.sampling.sessions_id[j]);
    }
    goto fn_exit;
}

static int
finalize_sampling_ctx(rocp_ctx_t *ctx)
{
    papi_free((*ctx)->u.sampling.events_name);
    papi_free((*ctx)->u.sampling.counters);
    papi_free((*ctx)->u.sampling.sessions_id);
    papi_free(*ctx);
    return PAPI_OK;
}

int
sampling_ctx_open(unsigned int *events_id, int num_events, rocp_ctx_t *ctx)
{
    int papi_errno = PAPI_OK;

    if (num_events < 1) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_rocm_lock);

    papi_errno = init_sampling_ctx(events_id, num_events, ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = open_sampling_ctx(*ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    (*ctx)->u.sampling.state |= ROCM_EVENTS_OPENED;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    finalize_sampling_ctx(ctx);
    goto fn_exit;
}

static int
close_sampling_ctx(rocp_ctx_t ctx)
{
    int papi_errno = PAPI_OK;

    int num_gpus;
    rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

    int i;
    for (i = 0; i < num_gpus; ++i) {
        rocprofiler_session_id_t *sid = &ctx->u.sampling.sessions_id[i];
        rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_destroy(sid[i]);
        if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    papi_errno = rocc_dev_release(ctx->u.sampling.gpu_map);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
sampling_ctx_close(rocp_ctx_t ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    papi_errno = close_sampling_ctx(ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    finalize_sampling_ctx(&ctx);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
sampling_ctx_start(rocp_ctx_t ctx)
{
    if (!(ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        return PAPI_EINVAL;
    }

    if (ctx->u.sampling.state & ROCM_EVENTS_RUNNING) {
        return PAPI_EINVAL;
    }

    int num_gpus;
    rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

    int i;
    for (i = 0; i < num_gpus; ++i) {
        rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_start(ctx->u.sampling.sessions_id[i]);
        if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    ctx->u.sampling.state |= ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_stop(rocp_ctx_t ctx)
{
    if (!(ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        return PAPI_EINVAL;
    }

    if (!(ctx->u.sampling.state & ROCM_EVENTS_RUNNING)) {
        return PAPI_EINVAL;
    }

    int num_gpus;
    rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

    int i;
    for (i = 0; i < num_gpus; ++i) {
        rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_stop(ctx->u.sampling.sessions_id[i]);
        if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    ctx->u.sampling.state &= ~ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_read(rocp_ctx_t ctx, long long **counters)
{
    int num_gpus;
    rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

    int i, j, k = 0;
    for (i = 0; i < num_gpus; ++i) {
        int gpu_id, num_gpu_events;
        rocc_dev_get_id(ctx->u.sampling.gpu_map, i, &gpu_id);
        get_gpu_events_count(ctx->u.sampling.events_id, ctx->u.sampling.num_events, gpu_id, &num_gpu_events);

        /* FIXME: replace static array with dynamic array */
        rocprofiler_device_profile_metric_t data[32] = { 0 };
        rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_poll(ctx->u.sampling.sessions_id[i], data);
        if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }

        for (j = 0; j < num_gpu_events; ++j) {
            /* FIXME: assuming counters in data have same order as ctx->u.sampling.events_id */
            ctx->u.sampling.counters[k++] = (long long) data[j].value.value;
        }
    }

    *counters = ctx->u.sampling.counters;
    return PAPI_OK;
}

int
sampling_ctx_reset(rocp_ctx_t ctx)
{
     int num_gpus;
     rocc_dev_get_count(ctx->u.sampling.gpu_map, &num_gpus);

     int i;
     for (i = 0; i < num_gpus; ++i) {
         rocprofiler_status_t rocp_errno = rocp_dev_prof_sess_stop(ctx->u.sampling.sessions_id[i]);
         if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
             break;
         }
         rocp_errno = rocp_dev_prof_sess_start(ctx->u.sampling.sessions_id[i]);
         if (rocp_errno != ROCPROFILER_STATUS_SUCCESS) {
             break;
         }
         ctx->u.sampling.counters[i] = 0;
     }

     return (i == num_gpus) ? PAPI_OK : PAPI_EMISC;
}

int
intercept_ctx_open(unsigned int *events_id, int num_events, rocp_ctx_t *ctx)
{
}

int
intercept_ctx_close(rocp_ctx_t ctx)
{
}

int
intercept_ctx_start(rocp_ctx_t ctx)
{
}

int
intercept_ctx_stop(rocp_ctx_t ctx)
{
}

int
intercept_ctx_read(rocp_ctx_t ctx, long long **counters)
{
}

int
intercept_ctx_reset(rocp_ctx_t ctx)
{
}
