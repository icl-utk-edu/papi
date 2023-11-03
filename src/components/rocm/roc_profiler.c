/**
 * @file    roc_profiler.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 */

#include <rocprofiler.h>
#include "roc_profiler.h"
#include "roc_common.h"
#include "htable.h"

/**
 * Event identifier encoding format:
 * +---------------------------------+-------+-------+--+------------+
 * |         unused                  |  dev  | inst  |  |   nameid   |
 * +---------------------------------+-------+-------+--+------------+
 *
 * unused    : 36 bits
 * device    : 7  bits ([0 - 127] devices)
 * instance  : 7  bits ([0 - 127] instances)
 * qlmask    : 2  bits (qualifier mask)
 * nameid    : 12 bits ([0 - 4095] event names)
 */
#define EVENTS_WIDTH (sizeof(uint64_t) * 8)
#define DEVICE_WIDTH ( 7)
#define INSTAN_WIDTH ( 7)
#define QLMASK_WIDTH ( 2)
#define NAMEID_WIDTH (12)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - INSTAN_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - DEVICE_WIDTH)
#define INSTAN_SHIFT (DEVICE_SHIFT - INSTAN_WIDTH)
#define QLMASK_SHIFT (INSTAN_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define DEVICE_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define INSTAN_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - INSTAN_WIDTH)) << INSTAN_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFFFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define DEVICE_FLAG  (0x2)
#define INSTAN_FLAG  (0x1)

typedef struct {
    char *name;
    char *descr;
    int instances;
    rocc_bitmap_t device_map;
    char *feature;
    int ntv_dev;
    uint64_t ntv_id;
    int instance;
} ntv_event_t;

typedef struct ntv_event_table {
    ntv_event_t *events;
    int count;
} ntv_event_table_t;

struct rocd_ctx {
    union {
        struct {
            int state;
            uint64_t *events_id;
            long long *counters;
            int dispatch_count;
            rocc_bitmap_t device_map;
            int feature_count;
        } intercept;
        struct {
            int state;
            uint64_t *events_id;
            long long *counters;
            rocprofiler_feature_t *features;
            int feature_count;
            rocprofiler_t **contexts;
            rocc_bitmap_t device_map;
            rocprofiler_properties_t *ctx_prop;
        } sampling;
    } u;
};

typedef struct {
    int device;
    int instance;
    int flags;
    int nameid;
} event_info_t;

unsigned int rocm_prof_mode;
unsigned int _rocm_lock;

/* rocprofiler function pointers */
static hsa_status_t (*rocp_get_info_p)(const hsa_agent_t *, rocprofiler_info_kind_t, void *);
static hsa_status_t (*rocp_iterate_info_p)(const hsa_agent_t *, rocprofiler_info_kind_t, hsa_status_t (*)(const rocprofiler_info_data_t, void *), void *);
static hsa_status_t (*rocp_error_string_p)(const char **);

/* for sampling mode */
static hsa_status_t (*rocp_open_p)(hsa_agent_t, rocprofiler_feature_t *, uint32_t, rocprofiler_t **, uint32_t, rocprofiler_properties_t *);
static hsa_status_t (*rocp_close_p)(rocprofiler_t *);
static hsa_status_t (*rocp_group_count_p)(const rocprofiler_t *, uint32_t *);
static hsa_status_t (*rocp_start_p)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_read_p)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_stop_p)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_get_group_p)(rocprofiler_t *, uint32_t, rocprofiler_group_t *);
static hsa_status_t (*rocp_get_data_p)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_group_get_data_p)(rocprofiler_group_t *);
static hsa_status_t (*rocp_get_metrics_p)(const rocprofiler_t *);
static hsa_status_t (*rocp_reset_p)(rocprofiler_t *, uint32_t);

/* for intercept mode */
static hsa_status_t (*rocp_pool_open_p)(hsa_agent_t, rocprofiler_feature_t *, uint32_t, rocprofiler_pool_t **, uint32_t, rocprofiler_pool_properties_t *);
static hsa_status_t (*rocp_pool_close_p)(rocprofiler_pool_t *);
static hsa_status_t (*rocp_pool_fetch_p)(rocprofiler_pool_t *, rocprofiler_pool_entry_t *);
static hsa_status_t (*rocp_pool_flush_p)(rocprofiler_pool_t *);
static hsa_status_t (*rocp_set_queue_cbs_p)(rocprofiler_queue_callbacks_t, void *);
static hsa_status_t (*rocp_start_queue_cbs_p)(void);
static hsa_status_t (*rocp_stop_queue_cbs_p)(void);
static hsa_status_t (*rocp_remove_queue_cbs_p)(void);

/**
 * rocp_{init,shutdown} and rocp_ctx_{open,close,start,stop,read,reset} functions
 *
 */
static int load_rocp_sym(void);
static int init_rocp_env(void);
static int init_event_table(void);
static int unload_rocp_sym(void);
static int sampling_ctx_open(uint64_t *, int, rocp_ctx_t *);
static int intercept_ctx_open(uint64_t *, int, rocp_ctx_t *);
static int sampling_ctx_close(rocp_ctx_t);
static int intercept_ctx_close(rocp_ctx_t);
static int sampling_ctx_start(rocp_ctx_t);
static int intercept_ctx_start(rocp_ctx_t);
static int sampling_ctx_stop(rocp_ctx_t);
static int intercept_ctx_stop(rocp_ctx_t);
static int sampling_ctx_read(rocp_ctx_t, long long **);
static int intercept_ctx_read(rocp_ctx_t, long long **);
static int sampling_ctx_reset(rocp_ctx_t);
static int intercept_ctx_reset(rocp_ctx_t);
static int sampling_shutdown(void);
static int intercept_shutdown(void);
static int evt_code_to_name(uint64_t event_code, char *name, int len);
static int evt_id_create(event_info_t *info, uint64_t *event_id);
static int evt_id_to_info(uint64_t event_id, event_info_t *info);

static void *rocp_dlp = NULL;
static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p;
static void *htable;
static void *htable_intercept;

/* rocp_init_environment - initialize ROCm environment variables */
int
rocp_init_environment(void)
{
    return init_rocp_env();
}

/* rocp_init - load runtime and profiling symbols, init runtime and profiling */
int
rocp_init(void)
{
    int papi_errno = PAPI_OK;
    SUBDBG("ENTER\n");

    papi_errno = load_rocp_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    htable_init(&htable);

    if (rocm_prof_mode != ROCM_PROFILE_SAMPLING_MODE) {
        htable_init(&htable_intercept);
    }

    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        (*hsa_shut_down_p)();
        goto fn_fail;
    }

    ntv_table_p = &ntv_table;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    unload_rocp_sym();
    goto fn_exit;
}

/* rocp_evt_enum - enumerate native events */
int
rocp_evt_enum(uint64_t *event_code, int modifier)
{
    int papi_errno = PAPI_OK;
    SUBDBG("ENTER: event_code: %lu, modifier: %d\n", *event_code, modifier);

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if (ntv_table_p->count == 0) {
                papi_errno = PAPI_ENOEVNT;
            }
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code + 1 < (uint64_t) ntv_table_p->count) {
                ++(*event_code);
            } else {
                papi_errno = PAPI_END;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
}

/* rocp_evt_code_to_descr - return descriptor string for event_code */
int
rocp_evt_code_to_descr(uint64_t event_code, char *descr, int len)
{
    if (event_code >= (uint64_t) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    snprintf(descr, (size_t) len, "%s", ntv_table_p->events[event_code].descr);
    return PAPI_OK;
}

/* rocp_evt_name_to_code - convert native event name to code */
int
rocp_evt_name_to_code(const char *name, uint64_t *event_code)
{
    int papi_errno = PAPI_OK;
    int htable_errno;
    SUBDBG("ENTER: name: %s, event_code: %p\n", name, event_code);

    ntv_event_t *event;
    htable_errno = htable_find(htable, name, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ?
            PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }
    *event_code = event->ntv_id;

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
}

/* rocp_evt_code_to_name - convert native event code to name */
int
rocp_evt_code_to_name(uint64_t event_code, char *name, int len)
{
    if (event_code >= (uint64_t) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    return evt_code_to_name(event_code, name, len);
}

/* rocp_ctx_open - open a profiling context for the requested events */
int
rocp_ctx_open(uint64_t *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_open(events_id, num_events, rocp_ctx);
    }

    return intercept_ctx_open(events_id, num_events, rocp_ctx);
}

/* rocp_ctx_close - close profiling context */
int
rocp_ctx_close(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_close(rocp_ctx);
    }

    return intercept_ctx_close(rocp_ctx);
}

/* rocp_ctx_start - start monitoring events associated to profiling context */
int
rocp_ctx_start(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_start(rocp_ctx);
    }

    return intercept_ctx_start(rocp_ctx);
}

/* rocp_ctx_stop - stop monitoring events associated to profiling context */
int
rocp_ctx_stop(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_stop(rocp_ctx);
    }

    return intercept_ctx_stop(rocp_ctx);
}

/* rocp_ctx_read - read counters for events associated to profiling context */
int
rocp_ctx_read(rocp_ctx_t rocp_ctx, long long **counts)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_read(rocp_ctx, counts);
    }

    return intercept_ctx_read(rocp_ctx, counts);
}

/* rocp_ctx_reset - reset counters for events associated to profiling context */
int
rocp_ctx_reset(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_reset(rocp_ctx);
    }

    return intercept_ctx_reset(rocp_ctx);
}

/* rocp_shutdown - shutdown runtime and profiling, unload runtime and profiling symbols */
int
rocp_shutdown(void)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_shutdown();
    }

    return intercept_shutdown();
}

/**
 * rocp_init utility functions
 *
 */
int
load_rocp_sym(void)
{
    int papi_errno = PAPI_OK;

    char *pathname = getenv("HSA_TOOLS_LIB");
    if (pathname == NULL) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Can't load librocprofiler64.so, neither PAPI_ROCM_ROOT nor HSA_TOOLS_LIB are set.");
        goto fn_fail;
    }

    rocp_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rocp_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    rocp_get_info_p        = dlsym(rocp_dlp, "rocprofiler_get_info");
    rocp_iterate_info_p    = dlsym(rocp_dlp, "rocprofiler_iterate_info");
    rocp_error_string_p    = dlsym(rocp_dlp, "rocprofiler_error_string");
    rocp_open_p            = dlsym(rocp_dlp, "rocprofiler_open");
    rocp_close_p           = dlsym(rocp_dlp, "rocprofiler_close");
    rocp_group_count_p     = dlsym(rocp_dlp, "rocprofiler_group_count");
    rocp_start_p           = dlsym(rocp_dlp, "rocprofiler_start");
    rocp_read_p            = dlsym(rocp_dlp, "rocprofiler_read");
    rocp_stop_p            = dlsym(rocp_dlp, "rocprofiler_stop");
    rocp_get_group_p       = dlsym(rocp_dlp, "rocprofiler_get_group");
    rocp_get_data_p        = dlsym(rocp_dlp, "rocprofiler_get_data");
    rocp_group_get_data_p  = dlsym(rocp_dlp, "rocprofiler_group_get_data");
    rocp_get_metrics_p     = dlsym(rocp_dlp, "rocprofiler_get_metrics");
    rocp_reset_p           = dlsym(rocp_dlp, "rocprofiler_reset");
    rocp_pool_open_p       = dlsym(rocp_dlp, "rocprofiler_pool_open");
    rocp_pool_close_p      = dlsym(rocp_dlp, "rocprofiler_pool_close");
    rocp_pool_fetch_p      = dlsym(rocp_dlp, "rocprofiler_pool_fetch");
    rocp_pool_flush_p      = dlsym(rocp_dlp, "rocprofiler_pool_flush");
    rocp_set_queue_cbs_p   = dlsym(rocp_dlp, "rocprofiler_set_queue_callbacks");
    rocp_start_queue_cbs_p = dlsym(rocp_dlp, "rocprofiler_start_queue_callbacks");
    rocp_stop_queue_cbs_p  = dlsym(rocp_dlp, "rocprofiler_stop_queue_callbacks");
    rocp_remove_queue_cbs_p= dlsym(rocp_dlp, "rocprofiler_remove_queue_callbacks");

    int rocp_not_initialized = (!rocp_get_info_p       ||
                                !rocp_iterate_info_p   ||
                                !rocp_error_string_p   ||
                                !rocp_open_p           ||
                                !rocp_close_p          ||
                                !rocp_group_count_p    ||
                                !rocp_start_p          ||
                                !rocp_read_p           ||
                                !rocp_stop_p           ||
                                !rocp_get_group_p      ||
                                !rocp_get_data_p       ||
                                !rocp_group_get_data_p ||
                                !rocp_get_metrics_p    ||
                                !rocp_reset_p          ||
                                !rocp_pool_open_p      ||
                                !rocp_pool_close_p     ||
                                !rocp_pool_fetch_p     ||
                                !rocp_pool_flush_p     ||
                                !rocp_set_queue_cbs_p  ||
                                !rocp_start_queue_cbs_p||
                                !rocp_stop_queue_cbs_p ||
                                !rocp_remove_queue_cbs_p);

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

    rocp_get_info_p        = NULL;
    rocp_iterate_info_p    = NULL;
    rocp_error_string_p    = NULL;
    rocp_open_p            = NULL;
    rocp_close_p           = NULL;
    rocp_group_count_p     = NULL;
    rocp_start_p           = NULL;
    rocp_read_p            = NULL;
    rocp_stop_p            = NULL;
    rocp_get_group_p       = NULL;
    rocp_get_data_p        = NULL;
    rocp_group_get_data_p  = NULL;
    rocp_get_metrics_p     = NULL;
    rocp_reset_p           = NULL;
    rocp_pool_open_p       = NULL;
    rocp_pool_close_p      = NULL;
    rocp_pool_fetch_p      = NULL;
    rocp_pool_flush_p      = NULL;
    rocp_set_queue_cbs_p   = NULL;
    rocp_start_queue_cbs_p = NULL;
    rocp_stop_queue_cbs_p  = NULL;
    rocp_remove_queue_cbs_p= NULL;

    dlclose(rocp_dlp);

    return PAPI_OK;
}

int
init_rocp_env(void)
{
    static int rocp_env_initialized;

    if (rocp_env_initialized) {
        return PAPI_OK;
    }

    const char *rocp_mode = getenv("ROCP_HSA_INTERCEPT");
    rocm_prof_mode = (rocp_mode != NULL) ?
        atoi(rocp_mode) : ROCM_PROFILE_SAMPLING_MODE;

    char pathname[PATH_MAX];
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root == NULL) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Can't set HSA_TOOLS_LIB. PAPI_ROCM_ROOT not set.");
        return PAPI_EMISC;
    }

    int err;
    int override_hsa_tools_lib = 1;
    struct stat stat_info;
    char *hsa_tools_lib = getenv("HSA_TOOLS_LIB");
    if (hsa_tools_lib) {
        err = stat(hsa_tools_lib, &stat_info);
        if (err == 0 && S_ISREG(stat_info.st_mode)) {
            override_hsa_tools_lib = 0;
        }
    }

    if (override_hsa_tools_lib) {
        /* Account for change of librocprofiler64.so file location in rocm-5.2.0
         * directory structure */

        /* prefer .so.1 as .so might not be available in 5.7 anymore, in 5.6 it
         * was a linker script. */
        const char *candidates[] = {
            "lib/librocprofiler64.so.1",
            "lib/librocprofiler64.so",
            "rocprofiler/lib/libprofiler64.so.1",
            "rocprofiler/lib/libprofiler64.so",
            NULL
        };
        const char **candidate = candidates;
        while (*candidate) {
            sprintf(pathname, "%s/%s", rocm_root, *candidate);

            err = stat(pathname, &stat_info);
            if (err == 0) {
                break;
            }
            ++candidate;
        }
        if (!*candidate) {
            snprintf(error_string, PAPI_MAX_STR_LEN, "Rocprofiler librocprofiler64.so file not found.");
            return PAPI_EMISC;
        }

        setenv("HSA_TOOLS_LIB", pathname, 1);
    }

    int override_rocp_metrics = 1;
    char *rocp_metrics = getenv("ROCP_METRICS");
    if (rocp_metrics) {
        err = stat(rocp_metrics, &stat_info);
        if (err == 0 && S_ISREG(stat_info.st_mode)) {
            override_rocp_metrics = 0;
        }
    }

    if (override_rocp_metrics) {
        /* Account for change of metrics file location in rocm-5.2.0
         * directory structure */
        sprintf(pathname, "%s/lib/rocprofiler/metrics.xml", rocm_root);

        err = stat(pathname, &stat_info);
        if (err < 0) {
            sprintf(pathname, "%s/rocprofiler/lib/metrics.xml", rocm_root);

            err = stat(pathname, &stat_info);
            if (err < 0) {
                snprintf(error_string, PAPI_MAX_STR_LEN, "Rocprofiler metrics.xml file not found.");
                return PAPI_EMISC;
            }
        }

        setenv("ROCP_METRICS", pathname, 1);
    }

    rocp_env_initialized = 1;
    return PAPI_OK;
}

static hsa_status_t count_ntv_events_cb(const rocprofiler_info_data_t, void *);
static hsa_status_t get_ntv_events_cb(const rocprofiler_info_data_t, void *);

struct ntv_arg {
    int count;                    /* number of devices counted so far */
    int dev_id;                   /* id of device */
};

int
init_event_table(void)
{
    int papi_errno = PAPI_OK;
    int i;

    for (i = 0; i < device_table_p->count; ++i) {
        hsa_status_t rocp_errno = rocp_iterate_info_p(&device_table_p->devices[i],
                                                      ROCPROFILER_INFO_KIND_METRIC,
                                                      &count_ntv_events_cb,
                                                      &ntv_table.count);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            const char *error_string_p;
            hsa_status_string_p(rocp_errno, &error_string_p);
            snprintf(error_string, PAPI_MAX_STR_LEN, "%s", error_string_p);
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    ntv_table.events = papi_calloc(ntv_table.count, sizeof(ntv_event_t));
    if (ntv_table.events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    struct ntv_arg arg;
    arg.count = 0;

    for (i = 0; i < device_table_p->count; ++i) {
        arg.dev_id = i;
        hsa_status_t rocp_errno = rocp_iterate_info_p(&device_table_p->devices[i],
                                                      ROCPROFILER_INFO_KIND_METRIC,
                                                      &get_ntv_events_cb,
                                                      &arg);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            const char *error_string_p;
            hsa_status_string_p(rocp_errno, &error_string_p);
            snprintf(error_string, PAPI_MAX_STR_LEN, "%s", error_string_p);
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
evt_code_to_name(uint64_t event_code, char *name, int len)
{
    if (ntv_table.events[event_code].instance >= 0) {
        snprintf(name, (size_t) len, "%s:device=%i:instance=%i",
                 ntv_table.events[event_code].name,
                 ntv_table.events[event_code].ntv_dev,
                 ntv_table.events[event_code].instance);
    } else {
        snprintf(name, (size_t) len, "%s:device=%i",
                 ntv_table.events[event_code].name,
                 ntv_table.events[event_code].ntv_dev);
    }
    return PAPI_OK;
}

int
evt_id_create(event_info_t *info, uint64_t *event_id)
{
    *event_id  = (uint64_t)(info->device   << DEVICE_SHIFT);
    *event_id |= (uint64_t)(info->instance << INSTAN_SHIFT);
    *event_id |= (uint64_t)(info->flags    << QLMASK_SHIFT);
    *event_id |= (uint64_t)(info->nameid   << NAMEID_SHIFT);
    return PAPI_OK;
}

int
evt_id_to_info(uint64_t event_id, event_info_t *info)
{
    info->device   = (int)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->instance = (int)((event_id & INSTAN_MASK) >> INSTAN_SHIFT);
    info->flags    = (int)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid   = (int)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->device >= device_table_p->count) {
        return PAPI_ENOEVNT;
    }

    if (0 == (info->flags & DEVICE_FLAG) && info->device > 0) {
        return PAPI_ENOEVNT;
    }

    if (rocc_dev_check(ntv_table_p->events[info->nameid].device_map, info->device) == 0) {
        return PAPI_ENOEVNT;
    }

    if (info->nameid >= ntv_table_p->count) {
        return PAPI_ENOEVNT;
    }

    if (ntv_table_p->events[info->nameid].instances > 1 && 0 == (info->flags & INSTAN_FLAG) && info->instance > 0) {
        return PAPI_ENOEVNT;
    }

    if (info->instance >= ntv_table_p->events[info->nameid].instances) {
        return PAPI_ENOEVNT;
    }

    return PAPI_OK;
}

/**
 * init_event_table utility functions
 *
 */
hsa_status_t
count_ntv_events_cb(const rocprofiler_info_data_t info, void *count)
{
    (*(int *) count) += info.metric.instances;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_ntv_events_cb(const rocprofiler_info_data_t info, void *ntv_arg)
{
    struct ntv_arg *arg = (struct ntv_arg *) ntv_arg;
    const int instances = info.metric.instances;
    int capacity = ntv_table.count;
    int *count = &arg->count;
    ntv_event_t *events = ntv_table.events;
    int instance;

    if (*count + instances > capacity) {
        snprintf(error_string, PAPI_MAX_STR_LEN, "Number of events exceeds detected count.");
        return HSA_STATUS_ERROR;
    }

    for (instance = 0; instance < instances; ++instance) {
        char feature[PAPI_MAX_STR_LEN] = { 0 };
        if (instances > 1) {
            sprintf(feature, "%s[%d]", info.metric.name, instance);
        } else {
            sprintf(feature, "%s", info.metric.name);
        }
        events[*count].name = strdup(info.metric.name);
        events[*count].descr = strdup(info.metric.description);
        events[*count].feature = strdup(feature);
        events[*count].ntv_dev = arg->dev_id;
        events[*count].ntv_id = *count;
        events[*count].instance = (instances > 1) ? (int) instance : -1;
        char key[PAPI_MAX_STR_LEN + 1];
        evt_code_to_name((uint64_t) *count, key, PAPI_MAX_STR_LEN);
        htable_insert(htable, key, &events[*count]);
        ++(*count);
    }

    return HSA_STATUS_SUCCESS;
}

/**
 * rocp_ctx_{open,close,start,stop,read,reset} sampling mode utility functions
 *
 */
static int init_features(uint64_t *, int, rocprofiler_feature_t *);
static int sampling_ctx_init(uint64_t *, int, rocp_ctx_t *);
static int sampling_ctx_finalize(rocp_ctx_t *);
static int ctx_open(rocp_ctx_t);
static int ctx_close(rocp_ctx_t);
static int ctx_init(uint64_t *, int, rocp_ctx_t *);
static int ctx_finalize(rocp_ctx_t *);
static int ctx_get_dev_feature_count(rocp_ctx_t, int);

int
sampling_ctx_open(uint64_t *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;

    if (num_events <= 0) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_rocm_lock);

    papi_errno = ctx_init(events_id, num_events, rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = ctx_open(*rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    (*rocp_ctx)->u.sampling.state |= ROCM_EVENTS_OPENED;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    ctx_finalize(rocp_ctx);
    goto fn_exit;
}

int
sampling_ctx_close(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    papi_errno = ctx_close(rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ctx_finalize(&rocp_ctx);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
sampling_ctx_start(rocp_ctx_t rocp_ctx)
{
    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP sampling mode] Cannot start eventset, not opened.");
        return PAPI_EINVAL;
    }

    if (rocp_ctx->u.sampling.state & ROCM_EVENTS_RUNNING) {
        SUBDBG("[ROCP sampling mode] Cannot start eventset, already running.");
        return PAPI_EINVAL;
    }

    int devs_count;
    rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &devs_count);

    int i;
    for (i = 0; i < devs_count; ++i) {
        hsa_status_t rocp_errno = rocp_start_p(rocp_ctx->u.sampling.contexts[i], 0);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    rocp_ctx->u.sampling.state |= ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_stop(rocp_ctx_t rocp_ctx)
{
    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP sampling mode] Cannot stop eventset, not opened.");
        return PAPI_EINVAL;
    }

    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_RUNNING)) {
        SUBDBG("[ROCP sampling mode] Cannot stop eventset, not running.");
        return PAPI_EINVAL;
    }

    int devs_count;
    rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &devs_count);

    int i;
    for (i = 0; i < devs_count; ++i) {
        hsa_status_t rocp_errno = rocp_stop_p(rocp_ctx->u.sampling.contexts[i], 0);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    rocp_ctx->u.sampling.state &= ~ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_read(rocp_ctx_t rocp_ctx, long long **counts)
{
    int i, j, k = 0;
    int dev_feature_offset = 0;
    int dev_id, dev_count;
    rocprofiler_feature_t *features = rocp_ctx->u.sampling.features;

    rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &dev_count);

    for (i = 0; i < dev_count; ++i) {
        hsa_status_t rocp_errno = rocp_read_p(rocp_ctx->u.sampling.contexts[i], 0);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }

        rocp_errno = rocp_get_data_p(rocp_ctx->u.sampling.contexts[i], 0);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }

        rocp_errno = rocp_get_metrics_p(rocp_ctx->u.sampling.contexts[i]);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }

        int papi_errno = rocc_dev_get_id(rocp_ctx->u.sampling.device_map, i, &dev_id);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        int dev_feature_count = ctx_get_dev_feature_count(rocp_ctx, dev_id);
        rocprofiler_feature_t *dev_features = features + dev_feature_offset;
        long long *counters = rocp_ctx->u.sampling.counters;

        for (j = 0; j < dev_feature_count; ++j) {
            switch(dev_features[j].data.kind) {
                case ROCPROFILER_DATA_KIND_INT32:
                    counters[k++] = (long long) dev_features[j].data.result_int32;
                    break;
                case ROCPROFILER_DATA_KIND_INT64:
                    counters[k++] = dev_features[j].data.result_int64;
                    break;
                case ROCPROFILER_DATA_KIND_FLOAT:
                    counters[k++] = (long long) dev_features[j].data.result_float;
                    break;
                case ROCPROFILER_DATA_KIND_DOUBLE:
                    counters[k++] = (long long) dev_features[j].data.result_double;
                    break;
                default:
                    return PAPI_EINVAL;
            }
        }
        dev_feature_offset += dev_feature_count;
    }
    *counts = rocp_ctx->u.sampling.counters;

    return PAPI_OK;
}

int
sampling_ctx_reset(rocp_ctx_t rocp_ctx)
{
    int i, devs_count;
    rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &devs_count);

    for (i = 0; i < devs_count; ++i) {
        hsa_status_t rocp_errno = rocp_reset_p(rocp_ctx->u.sampling.contexts[i], 0);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            return PAPI_EMISC;
        }
    }
    for (i = 0; i < rocp_ctx->u.sampling.feature_count; ++i) {
        rocp_ctx->u.sampling.counters[i] = 0;
    }
    return PAPI_OK;
}

/**
 * rocp_shutdown sampling mode utility functions
 *
 */
static int shutdown_event_table(void);

int
sampling_shutdown(void)
{
    shutdown_event_table();
    htable_shutdown(htable);

    unload_rocp_sym();

    return PAPI_OK;
}

int
shutdown_event_table(void)
{
    int i;

    for (i = 0; i < ntv_table_p->count; ++i) {
        papi_free(ntv_table_p->events[i].name);
        papi_free(ntv_table_p->events[i].descr);
        papi_free(ntv_table_p->events[i].feature);
    }

    ntv_table_p->count = 0;

    papi_free(ntv_table_p->events);

    return PAPI_OK;
}

/**
 * sampling_ctx_open utility functions
 *
 */
static int
event_id_to_dev_id_cb(uint64_t event_id, int *dev_id)
{
    *dev_id = ntv_table_p->events[event_id].ntv_dev;
    return PAPI_OK;
}

int
sampling_ctx_init(uint64_t *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    int num_devs;
    rocprofiler_feature_t *features = NULL;
    rocprofiler_t **contexts = NULL;
    rocprofiler_properties_t *ctx_prop = NULL;
    long long *counters = NULL;
    *rocp_ctx = NULL;

    rocc_bitmap_t bitmap;
    papi_errno = rocc_dev_get_map(event_id_to_dev_id_cb, events_id, num_events, &bitmap);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = rocc_dev_get_count(bitmap, &num_devs);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    contexts = papi_calloc(num_devs, sizeof(*contexts));
    if (contexts == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    ctx_prop = papi_calloc(num_devs, sizeof(*ctx_prop));
    if (ctx_prop == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    features = papi_calloc(num_events, sizeof(*features));
    if (features == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    counters = papi_malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = init_features(events_id, num_events, features);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    *rocp_ctx = papi_calloc(1, sizeof(**rocp_ctx));
    if (*rocp_ctx == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    (*rocp_ctx)->u.sampling.events_id = events_id;
    (*rocp_ctx)->u.sampling.features = features;
    (*rocp_ctx)->u.sampling.feature_count = num_events;
    (*rocp_ctx)->u.sampling.contexts = contexts;
    (*rocp_ctx)->u.sampling.counters = counters;
    (*rocp_ctx)->u.sampling.device_map = bitmap;
    (*rocp_ctx)->u.sampling.ctx_prop = ctx_prop;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (contexts) {
        papi_free(contexts);
    }
    if (features) {
        papi_free(features);
    }
    if (counters) {
        papi_free(counters);
    }
    if (*rocp_ctx) {
        papi_free(*rocp_ctx);
    }
    *rocp_ctx = NULL;
    goto fn_exit;
}

int
sampling_ctx_finalize(rocp_ctx_t *rocp_ctx)
{
    if (*rocp_ctx == NULL) {
        return PAPI_OK;
    }

    if ((*rocp_ctx)->u.sampling.features) {
        papi_free((*rocp_ctx)->u.sampling.features);
    }

    if ((*rocp_ctx)->u.sampling.contexts) {
        papi_free((*rocp_ctx)->u.sampling.contexts);
    }

    if ((*rocp_ctx)->u.sampling.ctx_prop) {
        papi_free((*rocp_ctx)->u.sampling.ctx_prop);
    }

    if ((*rocp_ctx)->u.sampling.counters) {
        papi_free((*rocp_ctx)->u.sampling.counters);
    }

    papi_free(*rocp_ctx);
    *rocp_ctx = NULL;

    return PAPI_OK;
}

int
ctx_open(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;
    int i, j;
    rocprofiler_feature_t *features = rocp_ctx->u.sampling.features;
    int dev_feature_offset = 0;
    int dev_count;
    rocprofiler_t **contexts = rocp_ctx->u.sampling.contexts;
    rocprofiler_properties_t *ctx_prop = rocp_ctx->u.sampling.ctx_prop;

    papi_errno = rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &dev_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    for (i = 0; i < dev_count; ++i) {
        int dev_id;
        papi_errno = rocc_dev_get_id(rocp_ctx->u.sampling.device_map, i, &dev_id);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        int dev_feature_count = ctx_get_dev_feature_count(rocp_ctx, dev_id);
        rocprofiler_feature_t *dev_features = features + dev_feature_offset;

        const uint32_t mode =
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

        ctx_prop[i].queue_depth = 128;
        hsa_status_t rocp_errno = rocp_open_p(device_table_p->devices[dev_id], dev_features,
                                              dev_feature_count, &contexts[i], mode,
                                              &ctx_prop[i]);
        if (rocp_errno != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        dev_feature_offset += dev_feature_count;
    }

    papi_errno = rocc_dev_acquire(rocp_ctx->u.sampling.device_map);
    if (papi_errno != PAPI_OK) {
       goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    for (j = 0; j < i; ++j) {
        rocp_close_p(contexts[j]);
        hsa_queue_destroy_p(ctx_prop[j].queue);
    }
    goto fn_exit;
}

int
ctx_close(rocp_ctx_t rocp_ctx)
{
    int papi_errno;
    int i, devs_count;
    rocc_dev_get_count(rocp_ctx->u.sampling.device_map, &devs_count);

    for (i = 0; i < devs_count; ++i) {
        if (rocp_close_p(rocp_ctx->u.sampling.contexts[i]) != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
        }

        if (hsa_queue_destroy_p(rocp_ctx->u.sampling.ctx_prop[i].queue) != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
        }
    }

    papi_errno = rocc_dev_release(rocp_ctx->u.sampling.device_map);

    return papi_errno;
}

int
init_features(uint64_t *events_id, int num_events, rocprofiler_feature_t *features)
{
    int papi_errno = PAPI_OK;

    int i;
    for (i = 0; i < num_events; ++i) {
        features[i].kind =
            (rocprofiler_feature_kind_t) ROCPROFILER_INFO_KIND_METRIC;
        features[i].name = (const char *) ntv_table_p->events[events_id[i]].feature;
    }

    return papi_errno;
}

static int sampling_ctx_get_dev_feature_count(rocp_ctx_t, int);
static int intercept_ctx_get_dev_feature_count(rocp_ctx_t, int);

int
ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, int i)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_get_dev_feature_count(rocp_ctx, i);
    }

    return intercept_ctx_get_dev_feature_count(rocp_ctx, i);
}

int
sampling_ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, int i)
{
    int start, stop, j = 0;
    int num_events = rocp_ctx->u.sampling.feature_count;
    uint64_t *events_id = rocp_ctx->u.sampling.events_id;
    ntv_event_t *ntv_events = ntv_table_p->events;

    while (j < num_events && ntv_events[events_id[j]].ntv_dev != i) {
        ++j;
    }

    start = j;

    while (j < num_events && ntv_events[events_id[j]].ntv_dev == i) {
        ++j;
    }

    stop = j;

    return stop - start;
}

int
intercept_ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, int i)
{
    int start, stop, j = 0;
    int num_events = rocp_ctx->u.intercept.feature_count;
    uint64_t *events_id = rocp_ctx->u.intercept.events_id;
    ntv_event_t *ntv_events = ntv_table_p->events;

    while (j < num_events && ntv_events[events_id[j]].ntv_dev != i) {
        ++j;
    }

    start = j;

    while (j < num_events && ntv_events[events_id[j]].ntv_dev == i) {
        ++j;
    }

    stop = j;

    return stop - start;
}

/**
 * rocp_ctx_{open,close,start,stop,read,reset} intercept mode utility functions
 *
 */
typedef struct cb_context_node {
    unsigned long tid;
    long long *counters;
    struct cb_context_node *next;
} cb_context_node_t;

static struct {
    uint64_t *events_id;
    int events_count;
    rocprofiler_feature_t *features;
    int feature_count;
    int active_thread_count;
    int kernel_count;
} intercept_global_state;

static int verify_events(uint64_t *, int);
static int init_callbacks(rocprofiler_feature_t *, int);
static int register_dispatch_counter(unsigned long, int *);
static int increment_and_fetch_dispatch_counter(unsigned long);
static int decrement_and_fetch_dispatch_counter(unsigned long);
static int unregister_dispatch_counter(unsigned long);
static int fetch_dispatch_counter(unsigned long);
static cb_context_node_t *alloc_context_node(int);
static void free_context_node(cb_context_node_t *);
static int get_context_node(int, cb_context_node_t **);
static int get_context_counters(uint64_t *, int, cb_context_node_t *, rocp_ctx_t);
static void put_context_counters(rocprofiler_feature_t *, int, cb_context_node_t *);
static void put_context_node(int, cb_context_node_t *);
static int intercept_ctx_init(uint64_t *, int, rocp_ctx_t *);
static int intercept_ctx_finalize(rocp_ctx_t *);

int
intercept_ctx_open(uint64_t *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;

    if (num_events <= 0) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_rocm_lock);

    papi_errno = verify_events(events_id, num_events);
    if (papi_errno != PAPI_OK) {
        SUBDBG("[ROCP intercept mode] Can only monitor one set of events "
               "per application run.");
        goto fn_fail;
    }

    papi_errno = ctx_init(events_id, num_events, rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    unsigned long tid;
    rocc_thread_get_id(&tid);
    papi_errno = register_dispatch_counter(tid, &(*rocp_ctx)->u.intercept.dispatch_count);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    (*rocp_ctx)->u.intercept.state |= ROCM_EVENTS_OPENED;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    ctx_finalize(rocp_ctx);
    goto fn_exit;
}

int
intercept_ctx_close(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    if (intercept_global_state.active_thread_count == 0) {
        goto fn_exit;
    }

    unsigned long tid;
    rocc_thread_get_id(&tid);
    papi_errno = unregister_dispatch_counter(tid);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    ctx_finalize(&rocp_ctx);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
}

int
intercept_ctx_start(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP intercept mode] Cannot start eventset, not opened.");
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }

    if (rocp_ctx->u.intercept.state & ROCM_EVENTS_RUNNING) {
        SUBDBG("[ROCP intercept mode] Cannot start eventset, already running.");
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }

    if (intercept_global_state.kernel_count++ == 0) {
        if (rocp_start_queue_cbs_p() != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    rocp_ctx->u.intercept.state |= ROCM_EVENTS_RUNNING;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
intercept_ctx_stop(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP intercept mode] Cannot stop eventset, not opened.");
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }

    if (!(rocp_ctx->u.intercept.state & ROCM_EVENTS_RUNNING)) {
        SUBDBG("[ROCP intercept mode] Cannot stop eventset, not running.");
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }

    if (--intercept_global_state.kernel_count == 0) {
        if (rocp_stop_queue_cbs_p() != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    rocp_ctx->u.intercept.state &= ~ROCM_EVENTS_RUNNING;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
intercept_ctx_read(rocp_ctx_t rocp_ctx, long long **counts)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    unsigned long tid;
    rocc_thread_get_id(&tid);
    int dispatch_count = fetch_dispatch_counter(tid);
    if (dispatch_count == 0) {
        *counts = rocp_ctx->u.intercept.counters;
        goto fn_exit;
    }

    cb_context_node_t *n = NULL;

    int devs_count;
    papi_errno = rocc_dev_get_count(rocp_ctx->u.intercept.device_map, &devs_count);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    int i;
    for (i = 0; i < devs_count; ++i) {
        int dev_id;
        papi_errno = rocc_dev_get_id(rocp_ctx->u.intercept.device_map, i, &dev_id);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }

        while (dispatch_count > 0) {
            get_context_node(dev_id, &n);
            if (n == NULL) {
                break;
            }

            get_context_counters(dev_id, n, rocp_ctx);
            dispatch_count = decrement_and_fetch_dispatch_counter(tid);
            free_context_node(n);
        }
    }

    if (dispatch_count > 0) {
        SUBDBG("[ROCP intercept mode] User monitoring GPU i but running on j.");
    }

    *counts = rocp_ctx->u.intercept.counters;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
}

int
intercept_ctx_reset(rocp_ctx_t rocp_ctx)
{
    int i;

    for (i = 0; i < rocp_ctx->u.intercept.feature_count; ++i) {
        rocp_ctx->u.intercept.counters[i] = 0;
    }

    return PAPI_OK;
}

/**
 * rocp_shutdown intercept mode utility functions
 *
 */

/**
 * Dispatch callback arguments. These are prepared by
 * the init_callbacks function
 */
typedef struct {
    rocprofiler_pool_t *pools[PAPI_ROCM_MAX_DEV_COUNT];
} cb_dispatch_arg_t;

static cb_dispatch_arg_t cb_dispatch_arg;

int
intercept_shutdown(void)
{
    /* calling rocprofiler_pool_close() here would cause
     * a double free runtime error. */

    shutdown_event_table();
    htable_shutdown(htable);
    htable_shutdown(htable_intercept);

    (*hsa_shut_down_p)();

    unload_rocp_sym();

    if (intercept_global_state.features) {
        papi_free(intercept_global_state.features);
    }

    if (intercept_global_state.events_id) {
        papi_free(intercept_global_state.events_id);
    }

    return PAPI_OK;
}

/**
 * intercept_ctx_{open,close} utility functions
 *
 */
int
verify_events(uint64_t *events_id, int num_events)
{
    int i;

    if (intercept_global_state.events_id == NULL) {
        return PAPI_OK;
    }

    if (intercept_global_state.events_count != num_events) {
        return PAPI_ECNFLCT;
    }

    for (i = 0; i < num_events; ++i) {
        void *out;
        if (htable_find(htable_intercept, ntv_table_p->events[events_id[i]].feature, &out)) {
            return PAPI_ECNFLCT;
        }
    }

    return PAPI_OK;
}

int
intercept_ctx_init(uint64_t *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    long long *counters = NULL;
    int num_devs;
    *rocp_ctx = NULL;

    rocc_bitmap_t bitmap;
    papi_errno = rocc_dev_get_map(event_id_to_dev_id_cb, events_id, num_events, &bitmap);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = rocc_dev_get_count(bitmap, &num_devs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (intercept_global_state.events_id == NULL) {
        intercept_global_state.events_id = papi_calloc(num_events, sizeof(int));
        if (intercept_global_state.events_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        memcpy(intercept_global_state.events_id, events_id, num_events * sizeof(uint64_t));

        /* FIXME: assuming the same number of events per device might not be an
         *        always valid assumption */
        int num_events_per_dev = num_events / num_devs;
        intercept_global_state.features = papi_calloc(num_events_per_dev, sizeof(*intercept_global_state.features));
        if (intercept_global_state.features == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        papi_errno = init_features(intercept_global_state.events_id, num_events_per_dev, intercept_global_state.features);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        intercept_global_state.events_count = num_events;
        intercept_global_state.feature_count = num_events_per_dev;

        int i;
        for (i = 0; i < num_events; ++i) {
            htable_insert(htable_intercept, ntv_table_p->events[events_id[i]].feature, NULL);
        }

        papi_errno = init_callbacks(intercept_global_state.features, intercept_global_state.feature_count);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }
    }

    counters = papi_calloc(num_events, sizeof(*counters));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    *rocp_ctx = papi_calloc(1, sizeof(**rocp_ctx));
    if (*rocp_ctx == NULL) {
        return PAPI_ENOMEM;
    }

    (*rocp_ctx)->u.intercept.events_id = events_id;
    (*rocp_ctx)->u.intercept.counters = counters;
    (*rocp_ctx)->u.intercept.dispatch_count = 0;
    (*rocp_ctx)->u.intercept.device_map = bitmap;
    (*rocp_ctx)->u.intercept.feature_count = num_events;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (counters) {
        papi_free(counters);
    }
    if (*rocp_ctx) {
        papi_free(*rocp_ctx);
    }
    *rocp_ctx = NULL;
    goto fn_exit;
}

int
intercept_ctx_finalize(rocp_ctx_t *rocp_ctx)
{
    if (*rocp_ctx == NULL) {
        return PAPI_OK;
    }

    if ((*rocp_ctx)->u.intercept.counters) {
        papi_free((*rocp_ctx)->u.intercept.counters);
    }

    papi_free(*rocp_ctx);
    *rocp_ctx = NULL;

    return PAPI_OK;
}

/**
 * Context init and finalize
 *
 */
int
ctx_init(uint64_t *events_id, int num_events,
         rocp_ctx_t *rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_init(events_id, num_events, rocp_ctx);
    }

    return intercept_ctx_init(events_id, num_events, rocp_ctx);
}

int
ctx_finalize(rocp_ctx_t *rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_finalize(rocp_ctx);
    }

    return intercept_ctx_finalize(rocp_ctx);
}

/**
 * Context handler arguments. These are prepared
 * by the dispatch callback and added to the
 * rocprofiler context
 */
typedef struct {
    rocprofiler_feature_t *features;
    int feature_count;
} cb_context_arg_t;

/**
 * The payload is prepared by the dispatch_callback
 * for the context handler
 */
typedef struct {
    int valid;
    unsigned long tid;
    hsa_agent_t agent;
    rocprofiler_group_t group;
    rocprofiler_callback_data_t data;
} cb_context_payload_t;

static bool context_handler_cb(const rocprofiler_pool_entry_t *, void *);
static hsa_status_t dispatch_cb(const rocprofiler_callback_data_t *, void *, rocprofiler_group_t *);

int
init_callbacks(rocprofiler_feature_t *features, int feature_count)
{
    int papi_errno = PAPI_OK;

    cb_context_arg_t *context_arg = papi_calloc(1, sizeof(cb_context_arg_t));
    if (context_arg == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    context_arg->features = features;
    context_arg->feature_count = feature_count;

    rocprofiler_pool_properties_t properties;
    properties.num_entries = 128;
    properties.payload_bytes = sizeof(cb_context_payload_t);
    properties.handler = context_handler_cb;
    properties.handler_arg = context_arg;

    /* FIXME: the intercept code initializes callbacks for every device
     *        regardless what the user asked for. Moreover, every device
     *        is initialized with the same callback events (features).
     *        The intercept code should eventually be changed to allow
     *        user to initialize different callbacks on different devices
     *        and also to reinitialize already initialized callbacks on
     *        any given device. Rocm 5.3.0 still does not support this
     *        callback initialization mechanism.
     */
    int i;
    for (i = 0; i < device_table_p->count; ++i) {
        hsa_agent_t agent = device_table_p->devices[i];

        rocprofiler_pool_t *pool = NULL;
        if (rocp_pool_open_p(agent, features, feature_count, &pool, 0, &properties) != HSA_STATUS_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        cb_dispatch_arg.pools[i] = pool;
    }

    rocprofiler_queue_callbacks_t dispatch_ptrs = { 0 };
    dispatch_ptrs.dispatch = dispatch_cb;

    if (rocp_set_queue_cbs_p(dispatch_ptrs, &cb_dispatch_arg) != HSA_STATUS_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    if (context_arg) {
        papi_free(context_arg);
    }
    goto fn_exit;
}

int
register_dispatch_counter(unsigned long tid, int *counter)
{
    int papi_errno = PAPI_OK;
    int htable_errno = HTABLE_SUCCESS;
    char key[PAPI_MIN_STR_LEN] = { 0 };

    /* FIXME: probably better using a different hash table for this */
    sprintf(key, "%lu", tid);
    int *counter_p;
    htable_errno = htable_find(htable, key, (void **) &counter_p);
    if (htable_errno == HTABLE_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_exit;
    }

    htable_insert(htable, (const char *) key, counter);
    ++intercept_global_state.active_thread_count;

  fn_exit:
    return papi_errno;
}

int
unregister_dispatch_counter(unsigned long tid)
{
    int papi_errno = PAPI_OK;
    int htable_errno = HTABLE_SUCCESS;
    char key[PAPI_MIN_STR_LEN] = { 0 };

    sprintf(key, "%lu", tid);
    int *counter_p;
    htable_errno = htable_find(htable, (const char *) key, (void **) &counter_p);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = PAPI_ECMP;
        goto fn_exit;
    }

    htable_delete(htable, (const char *) key);
    --intercept_global_state.active_thread_count;

  fn_exit:
    return papi_errno;
}

/**
 * intercept mode counter read infrastructure
 *
 */
static void process_context_entry(cb_context_payload_t *, rocprofiler_feature_t *, int);

/**
 * The context handler prepares a node for every
 * processes entry. The node is associated with
 * the thread that generated the monitoring
 * request and contains the value of the counters
 * read by rocprofiler. Each node is then added
 * to the corresponding device queue and is
 * eventually read by intercept_ctx_read
 */
static cb_context_node_t *cb_ctx_list_heads[PAPI_ROCM_MAX_DEV_COUNT];

hsa_status_t
dispatch_cb(const rocprofiler_callback_data_t *callback_data, void *arg, rocprofiler_group_t *group)
{
    hsa_agent_t agent = callback_data->agent;
    hsa_status_t status = HSA_STATUS_SUCCESS;

    int dev_id;
    rocc_dev_get_agent_id(agent, &dev_id);

    cb_dispatch_arg_t *dispatch_arg = (cb_dispatch_arg_t *) arg;
    rocprofiler_pool_t *pool = dispatch_arg->pools[dev_id];
    rocprofiler_pool_entry_t pool_entry;
    hsa_status_t rocp_errno = rocp_pool_fetch_p(pool, &pool_entry);
    if (rocp_errno != HSA_STATUS_SUCCESS) {
        status = rocp_errno;
        goto fn_exit;
    }

    rocprofiler_t *context = pool_entry.context;
    cb_context_payload_t *payload = (cb_context_payload_t *) pool_entry.payload;

    rocp_errno = rocp_get_group_p(context, 0, group);
    if (rocp_errno != HSA_STATUS_SUCCESS) {
        status = rocp_errno;
        goto fn_exit;
    }

    unsigned long tid;
    rocc_thread_get_id(&tid);
    payload->tid = tid;
    payload->agent = agent;
    payload->group = *group;
    payload->data = *callback_data;

    _papi_hwi_lock(_rocm_lock);
    payload->valid = true;
    _papi_hwi_unlock(_rocm_lock);

  fn_exit:
    return status;
}

bool
context_handler_cb(const rocprofiler_pool_entry_t *entry, void *arg)
{
    cb_context_payload_t *payload = (cb_context_payload_t *) entry->payload;
    cb_context_arg_t *context_arg = (cb_context_arg_t *) arg;

    process_context_entry(payload, context_arg->features, context_arg->feature_count);

    return false;
}

void
process_context_entry(cb_context_payload_t *payload, rocprofiler_feature_t *features, int feature_count)
{
  fn_check_again:
    _papi_hwi_lock(_rocm_lock);
    if (payload->valid == false) {
        _papi_hwi_unlock(_rocm_lock);
        goto fn_check_again;
    }

    if (feature_count < 1) {
        goto fn_exit;
    }

    if (rocp_group_get_data_p(&payload->group) != HSA_STATUS_SUCCESS) {
        goto fn_exit;
    }

    if (rocp_get_metrics_p(payload->group.context)) {
        goto fn_exit;
    }

    if (increment_and_fetch_dispatch_counter(payload->tid) < 0) {
        /* thread not registered, ignore counters */
        goto fn_exit;
    }

    cb_context_node_t *n = alloc_context_node(feature_count);
    if (n == NULL) {
        decrement_and_fetch_dispatch_counter(payload->tid);
        goto fn_exit;
    }

    int dev_id;
    rocc_dev_get_agent_id(payload->agent, &dev_id);

    n->tid = payload->tid;
    put_context_counters(features, feature_count, n);
    put_context_node(dev_id, n);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
}

cb_context_node_t *
alloc_context_node(int num_events)
{
    cb_context_node_t *n = papi_malloc(sizeof(*n));
    if (n == NULL) {
        return NULL;
    }

    n->counters = papi_malloc(num_events * sizeof(long long));
    if (n->counters == NULL) {
        papi_free(n);
        return NULL;
    }

    return n;
}

void
put_context_counters(rocprofiler_feature_t *features, int feature_count, cb_context_node_t *n)
{
    int i;
    for (i = 0; i < feature_count; ++i) {
        const rocprofiler_feature_t *f = &features[i];
        switch(f->data.kind) {
            case ROCPROFILER_DATA_KIND_INT32:
                n->counters[i] = (long long) f->data.result_int32;
                break;
            case ROCPROFILER_DATA_KIND_INT64:
                n->counters[i] = f->data.result_int64;
                break;
            case ROCPROFILER_DATA_KIND_FLOAT:
                n->counters[i] = (long long) f->data.result_float;
                break;
            case ROCPROFILER_DATA_KIND_DOUBLE:
                n->counters[i] = (long long) f->data.result_double;
                break;
            default:
                SUBDBG("Unsupported data kind from rocprofiler");
        }
    }
}

void
put_context_node(int dev_id, cb_context_node_t *n)
{
    n->next = NULL;

    if (cb_ctx_list_heads[dev_id] != NULL) {
        n->next = cb_ctx_list_heads[dev_id];
    }

    cb_ctx_list_heads[dev_id] = n;
}

int
increment_and_fetch_dispatch_counter(unsigned long tid)
{
    int htable_errno = HTABLE_SUCCESS;
    char key[PAPI_MIN_STR_LEN] = { 0 };

    sprintf(key, "%lu", tid);
    int *counter_p;
    htable_errno = htable_find(htable, (const char *) key, (void **) &counter_p);
    if (htable_errno != HTABLE_SUCCESS) {
        return 0;
    }

    return ++(*counter_p);
}

int
fetch_dispatch_counter(unsigned long tid)
{
    int htable_errno = HTABLE_SUCCESS;
    char key[PAPI_MIN_STR_LEN] = { 0 };

    sprintf(key, "%lu", tid);
    int *counter_p;
    htable_errno = htable_find(htable, (const char *) key, (void **) &counter_p);
    if (htable_errno != HTABLE_SUCCESS) {
        return 0;
    }

    return (*counter_p);
}

int
get_context_node(int dev_id, cb_context_node_t **n)
{
    cb_context_node_t *curr = cb_ctx_list_heads[dev_id];
    cb_context_node_t *flag = NULL;
    cb_context_node_t *prev = curr;
    cb_context_node_t *flag_prev;

    while (curr) {
        unsigned long tid;
        rocc_thread_get_id(&tid);
        if (curr->tid == tid) {
            flag_prev = prev;
            flag = curr;
        }
        prev = curr;
        curr = curr->next;
    }

    if (flag != NULL) {
        flag_prev->next = flag->next;
        if (cb_ctx_list_heads[dev_id] == flag) {
            cb_ctx_list_heads[dev_id] = NULL;
        }
    }

    *n = flag;
    return PAPI_OK;
}

int
decrement_and_fetch_dispatch_counter(unsigned long tid)
{
    int htable_errno = HTABLE_SUCCESS;
    char key[PAPI_MIN_STR_LEN] = { 0 };

    sprintf(key, "%lu", tid);
    int *counter_p;
    htable_errno = htable_find(htable, (const char *) key, (void **) &counter_p);
    if (htable_errno != HTABLE_SUCCESS) {
        return 0;
    }

    return --(*counter_p);
}

int
get_context_counters(uint64_t *events_id, int dev_id, cb_context_node_t *n, rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;
    uint64_t *events_id = rocp_ctx->u.intercept.events_id;

    /* Here we get events_id ordered according to user's viewpoint and we want
     * to map these to events_id ordered according to callbacks' viewpoint. We
     * compare events from the user and the callbacks using a brute force
     * approach as the number of events is typically small. */
    int i, j;
    for (i = 0; i < intercept_global_state.feature_count; ++i) {
        const char *cb_name = intercept_global_state.features[i].name;

        for (j = 0; j < rocp_ctx->u.intercept.feature_count; ++j) {
            const char *usr_name = ntv_table_p->events[events_id[j]].name;
            int usr_ntv_dev = ntv_table_p->events[events_id[j]].ntv_dev;

            if (dev_id == usr_ntv_dev && strcmp(usr_name, cb_name) == 0) {
                break;
            }
        }
        if (j >= rocp_ctx->u.intercept.feature_count) {
            return PAPI_ECMP;
        }
        rocp_ctx->u.intercept.counters[j] += n->counters[i];
    }

    return papi_errno;
}

void
free_context_node(cb_context_node_t *n)
{
    papi_free(n->counters);
    papi_free(n);
}

void __attribute__((visibility("default")))
OnLoadToolProp(rocprofiler_settings_t *settings __attribute__((unused)))
{
    init_rocp_env();
}

void __attribute__((visibility("default")))
OnUnloadTool(void)
{
    return;
}
