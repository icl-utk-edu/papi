/**
 * @file    rocp.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 */

#ifdef ROCM_PROF_ROCPROFILER
#include <sys/stat.h>
#include <dlfcn.h>
#include <hsa.h>
#include <rocprofiler.h>
#include <unistd.h>
#include <stdlib.h>

#include "rocp.h"
#include "htable.h"
#include "common.h"

typedef struct {
    char *name;
    char *descr;
    unsigned int ntv_dev;
    unsigned int ntv_id;
    int instance;
} ntv_event_t;

typedef struct ntv_event_table {
    ntv_event_t *events;
    int count;
} ntv_event_table_t;

struct rocd_ctx {
    union {
        struct {
            int state;                       /* state of kernel interception */
            unsigned int *events_id;
            long long *counters;             /* thread's private counters */
            int dispatch_count;              /* how many kernel dispatches this thread has done */
            unsigned int *devs_id;           /* list of monitored devices */
            int devs_count;                  /* number of monitored devices */
            int feature_count;               /* number of features being monitored */
        } intercept;
        struct {
            int state;                       /* state of sampling */
            unsigned int *events_id;
            long long *counters;             /* thread's private counters */
            rocprofiler_feature_t *features; /* rocprofiler features */
            int feature_count;               /* number of features being monitored */
            rocprofiler_t **contexts;        /* rocprofiler context array for multiple device monitoring */
            unsigned int *devs_id;           /* list of monitored device ids */
            int devs_count;                  /* number of monitored devices */
        } sampling;
    } u;
};

#ifndef PAPI_ROCM_MAX_DEV_COUNT
#define PAPI_ROCM_MAX_DEV_COUNT (32)
#endif

typedef struct {
    hsa_agent_t agents[PAPI_ROCM_MAX_DEV_COUNT]; /* array of hsa agents */
    int count;                                   /* number of hsa agents in agent array */
} hsa_agent_arr_t;

unsigned int rocm_prof_mode;
unsigned int _rocm_lock;                         /* internal rocm component lock (allocated at configure time) */

/* hsa function pointers */
static hsa_status_t (*hsa_initPtr)(void);
static hsa_status_t (*hsa_shut_downPtr)(void);
static hsa_status_t (*hsa_iterate_agentsPtr)(hsa_status_t (*)(hsa_agent_t,
                                                              void *),
                                             void *);
static hsa_status_t (*hsa_system_get_infoPtr)(hsa_system_info_t, void *);
static hsa_status_t (*hsa_agent_get_infoPtr)(hsa_agent_t, hsa_agent_info_t,
                                             void *);
static hsa_status_t (*hsa_queue_destroyPtr)(hsa_queue_t *);
static hsa_status_t (*hsa_status_stringPtr)(hsa_status_t,
                                            const char **);

/* rocprofiler function pointers */
static hsa_status_t (*rocp_get_infoPtr)(const hsa_agent_t *,
                                        rocprofiler_info_kind_t, void *);
static hsa_status_t (*rocp_iterate_infoPtr)(const hsa_agent_t *,
                                            rocprofiler_info_kind_t,
                                            hsa_status_t (*)
                                                (const rocprofiler_info_data_t,
                                                 void *),
                                            void *);
static hsa_status_t (*rocp_error_stringPtr)(const char **);

/* for sampling mode */
static hsa_status_t (*rocp_openPtr)(hsa_agent_t, rocprofiler_feature_t *,
                                    uint32_t,
                                    rocprofiler_t **, uint32_t,
                                    rocprofiler_properties_t *);
static hsa_status_t (*rocp_closePtr)(rocprofiler_t *);
static hsa_status_t (*rocp_group_countPtr)(const rocprofiler_t *, uint32_t *);
static hsa_status_t (*rocp_startPtr)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_readPtr)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_stopPtr)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_get_groupPtr)(rocprofiler_t *, uint32_t,
                                         rocprofiler_group_t *);
static hsa_status_t (*rocp_get_dataPtr)(rocprofiler_t *, uint32_t);
static hsa_status_t (*rocp_group_get_dataPtr)(rocprofiler_group_t *);
static hsa_status_t (*rocp_get_metricsPtr)(const rocprofiler_t *);
static hsa_status_t (*rocp_resetPtr)(rocprofiler_t *, uint32_t);

/* for intercept mode */
static hsa_status_t (*rocp_pool_openPtr)(hsa_agent_t, rocprofiler_feature_t *,
                                         uint32_t, rocprofiler_pool_t **,
                                         uint32_t,
                                         rocprofiler_pool_properties_t *);
static hsa_status_t (*rocp_pool_closePtr)(rocprofiler_pool_t *);
static hsa_status_t (*rocp_pool_fetchPtr)(rocprofiler_pool_t *,
                                          rocprofiler_pool_entry_t *);
static hsa_status_t (*rocp_pool_flushPtr)(rocprofiler_pool_t *);
static hsa_status_t (*rocp_set_queue_cbsPtr)(rocprofiler_queue_callbacks_t,
                                             void *);
static hsa_status_t (*rocp_start_queue_cbsPtr)(void);
static hsa_status_t (*rocp_stop_queue_cbsPtr)(void);
static hsa_status_t (*rocp_remove_queue_cbsPtr)(void);

#define ROCM_CALL(call, err_handle) do {                        \
    hsa_status_t _status = (call);                              \
    if (_status == HSA_STATUS_SUCCESS ||                        \
        _status == HSA_STATUS_INFO_BREAK) {                     \
        break;                                                  \
    }                                                           \
    err_handle;                                                 \
} while(0)

#define ROCM_GET_ERR_STR(status) do {                           \
    (*hsa_status_stringPtr)(status, &init_err_str_ptr);         \
    int _exp = snprintf(init_err_str, PAPI_MAX_STR_LEN, "%s",   \
                        init_err_str_ptr);                      \
    if (_exp > PAPI_MAX_STR_LEN) {                              \
        SUBDBG("Error string truncated");                       \
    }                                                           \
    init_err_str_ptr = init_err_str;                            \
} while(0)

#define ROCM_PUT_ERR_STR(err_str) do {                          \
    err_str = init_err_str_ptr;                                 \
} while(0)

#define ROCP_CALL ROCM_CALL

#define ROCP_GET_ERR_STR() do {                                 \
    (*rocp_error_stringPtr)(&init_err_str_ptr);                 \
    int _exp = snprintf(init_err_str, PAPI_MAX_STR_LEN, "%s",   \
                        init_err_str_ptr);                      \
    if (_exp > PAPI_MAX_STR_LEN) {                              \
        SUBDBG("Error string truncated");                       \
    }                                                           \
    init_err_str_ptr = init_err_str;                            \
} while(0)

#define ROCP_PUT_ERR_STR ROCM_PUT_ERR_STR

#define ROCP_REC_ERR_STR(string) do {                           \
    int _exp = snprintf(init_err_str, PAPI_MAX_STR_LEN, "%s",   \
                        string);                                \
    if (_exp > PAPI_MAX_STR_LEN) {                              \
        SUBDBG("Error string truncated");                       \
    }                                                           \
    init_err_str_ptr = init_err_str;                            \
} while(0)

/**
 * rocp_{init,shutdown} and rocp_ctx_{open,close,start,stop,read,reset} functions
 *
 */
static int load_hsa_sym(void);
static int load_rocp_sym(void);
static int init_rocp_env(void);
static int init_event_table(void);
static int unload_hsa_sym(void);
static int unload_rocp_sym(void);
static int init_agent_array(void);
static int sampling_ctx_open(unsigned int *, int, rocp_ctx_t *);
static int intercept_ctx_open(unsigned int *, int, rocp_ctx_t *);
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
static void init_thread_id_fn(void);
static int evt_code_to_name(unsigned int event_code, char *name, int len);

static void *hsa_dlp = NULL;
static void *rocp_dlp = NULL;
static const char *init_err_str_ptr;
static char init_err_str[PAPI_MAX_STR_LEN];
static hsa_agent_arr_t agent_arr;
static unsigned long (*thread_id_fn)(void);
static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p;
static void *htable;

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

    papi_errno = load_hsa_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = load_rocp_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    /* NOTE: hsa_init() initializes hsa runtime, which further
     *       initializes rocprofiler whenever HSA_TOOLS_LIB is
     *       set (as done by init_rocp_env()). */
    hsa_status_t status = (*hsa_initPtr)();
    if (status != HSA_STATUS_SUCCESS) {
        ROCM_GET_ERR_STR(status);
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = init_agent_array();
    if (papi_errno != PAPI_OK) {
        (*hsa_shut_downPtr)();
        goto fn_fail;
    }

    htable_init(&htable);

    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        (*hsa_shut_downPtr)();
        goto fn_fail;
    }

    init_thread_id_fn();
    ntv_table_p = &ntv_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_rocp_sym();
    unload_hsa_sym();
    goto fn_exit;
}

/* rocp_evt_enum - enumerate native events */
int
rocp_evt_enum(unsigned int *event_code, int modifier)
{
    int papi_errno = PAPI_OK;

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if (ntv_table_p->count == 0) {
                papi_errno = PAPI_ENOEVNT;
            }
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code + 1 < (unsigned int) ntv_table_p->count) {
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

/* rocp_evt_get_descr - return descriptor string for event_code */
int
rocp_evt_get_descr(unsigned int event_code, char *descr, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    /* FIXME: make sure descr is not longer than len */
    strncpy(descr, ntv_table_p->events[event_code].descr, len);
    return PAPI_OK;
}

/* rocp_evt_name_to_code - convert native event name to code */
int
rocp_evt_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = PAPI_OK;
    int htable_errno;

    ntv_event_t *event;
    htable_errno = htable_find(htable, name, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ?
            PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }
    *event_code = event->ntv_id;

  fn_exit:
    return papi_errno;
}

/* rocp_evt_code_to_name - convert native event code to name */
int
rocp_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    if (event_code >= (unsigned int) ntv_table_p->count) {
        return PAPI_EINVAL;
    }
    return evt_code_to_name(event_code, name, len);
}

/* rocp_err_get_last - get error string for last occured error */
int
rocp_err_get_last(const char **err_string)
{
    ROCM_PUT_ERR_STR(*err_string);
    return PAPI_OK;
}

/* rocp_ctx_open - open a profiling context for the requested events */
int
rocp_ctx_open(unsigned int *events_id, int num_events, rocp_ctx_t *rocp_ctx)
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
load_hsa_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = { 0 };
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root == NULL) {
        ROCP_REC_ERR_STR("Can't load libhsa-runtime64.so, PAPI_ROCM_ROOT not set.");
        goto fn_fail;
    }

    sprintf(pathname, "%s/lib/libhsa-runtime64.so", rocm_root);

    hsa_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (hsa_dlp == NULL) {
        ROCP_REC_ERR_STR(dlerror());
        goto fn_fail;
    }

    hsa_initPtr            = dlsym(hsa_dlp, "hsa_init");
    hsa_shut_downPtr       = dlsym(hsa_dlp, "hsa_shut_down");
    hsa_iterate_agentsPtr  = dlsym(hsa_dlp, "hsa_iterate_agents");
    hsa_system_get_infoPtr = dlsym(hsa_dlp, "hsa_system_get_info");
    hsa_agent_get_infoPtr  = dlsym(hsa_dlp, "hsa_agent_get_info");
    hsa_queue_destroyPtr   = dlsym(hsa_dlp, "hsa_queue_destroy");
    hsa_status_stringPtr   = dlsym(hsa_dlp, "hsa_status_string");

    int hsa_not_initialized = (!hsa_initPtr            ||
                               !hsa_shut_downPtr       ||
                               !hsa_iterate_agentsPtr  ||
                               !hsa_system_get_infoPtr ||
                               !hsa_agent_get_infoPtr  ||
                               !hsa_queue_destroyPtr   ||
                               !hsa_status_stringPtr);

    papi_errno = (hsa_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        ROCP_REC_ERR_STR("Error while loading hsa symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

int
unload_hsa_sym(void)
{
    if (hsa_dlp == NULL) {
        return PAPI_OK;
    }

    hsa_initPtr            = NULL;
    hsa_shut_downPtr       = NULL;
    hsa_iterate_agentsPtr  = NULL;
    hsa_system_get_infoPtr = NULL;
    hsa_agent_get_infoPtr  = NULL;
    hsa_queue_destroyPtr   = NULL;
    hsa_status_stringPtr   = NULL;

    dlclose(hsa_dlp);

    return PAPI_OK;
}

int
load_rocp_sym(void)
{
    int papi_errno = PAPI_OK;

    char *pathname = getenv("HSA_TOOLS_LIB");
    if (pathname == NULL) {
        ROCP_REC_ERR_STR("Can't load librocprofiler64.so, neither PAPI_ROCM_ROOT "
                         " nor HSA_TOOLS_LIB are set.");
        goto fn_fail;
    }

    rocp_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rocp_dlp == NULL) {
        ROCP_REC_ERR_STR(dlerror());
        goto fn_fail;
    }

    rocp_get_infoPtr        = dlsym(rocp_dlp, "rocprofiler_get_info");
    rocp_iterate_infoPtr    = dlsym(rocp_dlp, "rocprofiler_iterate_info");
    rocp_error_stringPtr    = dlsym(rocp_dlp, "rocprofiler_error_string");
    rocp_openPtr            = dlsym(rocp_dlp, "rocprofiler_open");
    rocp_closePtr           = dlsym(rocp_dlp, "rocprofiler_close");
    rocp_group_countPtr     = dlsym(rocp_dlp, "rocprofiler_group_count");
    rocp_startPtr           = dlsym(rocp_dlp, "rocprofiler_start");
    rocp_readPtr            = dlsym(rocp_dlp, "rocprofiler_read");
    rocp_stopPtr            = dlsym(rocp_dlp, "rocprofiler_stop");
    rocp_get_groupPtr       = dlsym(rocp_dlp, "rocprofiler_get_group");
    rocp_get_dataPtr        = dlsym(rocp_dlp, "rocprofiler_get_data");
    rocp_group_get_dataPtr  = dlsym(rocp_dlp, "rocprofiler_group_get_data");
    rocp_get_metricsPtr     = dlsym(rocp_dlp, "rocprofiler_get_metrics");
    rocp_resetPtr           = dlsym(rocp_dlp, "rocprofiler_reset");
    rocp_pool_openPtr       = dlsym(rocp_dlp, "rocprofiler_pool_open");
    rocp_pool_closePtr      = dlsym(rocp_dlp, "rocprofiler_pool_close");
    rocp_pool_fetchPtr      = dlsym(rocp_dlp, "rocprofiler_pool_fetch");
    rocp_pool_flushPtr      = dlsym(rocp_dlp, "rocprofiler_pool_flush");
    rocp_set_queue_cbsPtr   = dlsym(rocp_dlp, "rocprofiler_set_queue_callbacks");
    rocp_start_queue_cbsPtr = dlsym(rocp_dlp, "rocprofiler_start_queue_callbacks");
    rocp_stop_queue_cbsPtr  = dlsym(rocp_dlp, "rocprofiler_stop_queue_callbacks");
    rocp_remove_queue_cbsPtr= dlsym(rocp_dlp, "rocprofiler_remove_queue_callbacks");

    int rocp_not_initialized = (!rocp_get_infoPtr       ||
                                !rocp_iterate_infoPtr   ||
                                !rocp_error_stringPtr   ||
                                !rocp_openPtr           ||
                                !rocp_closePtr          ||
                                !rocp_group_countPtr    ||
                                !rocp_startPtr          ||
                                !rocp_readPtr           ||
                                !rocp_stopPtr           ||
                                !rocp_get_groupPtr      ||
                                !rocp_get_dataPtr       ||
                                !rocp_group_get_dataPtr ||
                                !rocp_get_metricsPtr    ||
                                !rocp_resetPtr          ||
                                !rocp_pool_openPtr      ||
                                !rocp_pool_closePtr     ||
                                !rocp_pool_fetchPtr     ||
                                !rocp_pool_flushPtr     ||
                                !rocp_set_queue_cbsPtr  ||
                                !rocp_start_queue_cbsPtr||
                                !rocp_stop_queue_cbsPtr ||
                                !rocp_remove_queue_cbsPtr);

    papi_errno = (rocp_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        ROCP_REC_ERR_STR("Error while loading rocprofiler symbols.");
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

    rocp_get_infoPtr        = NULL;
    rocp_iterate_infoPtr    = NULL;
    rocp_error_stringPtr    = NULL;
    rocp_openPtr            = NULL;
    rocp_closePtr           = NULL;
    rocp_group_countPtr     = NULL;
    rocp_startPtr           = NULL;
    rocp_readPtr            = NULL;
    rocp_stopPtr            = NULL;
    rocp_get_groupPtr       = NULL;
    rocp_get_dataPtr        = NULL;
    rocp_group_get_dataPtr  = NULL;
    rocp_get_metricsPtr     = NULL;
    rocp_resetPtr           = NULL;
    rocp_pool_openPtr       = NULL;
    rocp_pool_closePtr      = NULL;
    rocp_pool_fetchPtr      = NULL;
    rocp_pool_flushPtr      = NULL;
    rocp_set_queue_cbsPtr   = NULL;
    rocp_start_queue_cbsPtr = NULL;
    rocp_stop_queue_cbsPtr  = NULL;
    rocp_remove_queue_cbsPtr= NULL;

    dlclose(rocp_dlp);

    return PAPI_OK;
}

static hsa_status_t get_agent_handle_cb(hsa_agent_t, void *);

int
init_agent_array(void)
{
    int papi_errno = PAPI_OK;

    ROCM_CALL((*hsa_iterate_agentsPtr)(get_agent_handle_cb, &agent_arr),
              { ROCM_GET_ERR_STR(_status); goto fn_fail; });

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC;
    agent_arr.count = 0;
    goto fn_exit;
}

hsa_status_t
get_agent_handle_cb(hsa_agent_t agent, void *agent_arr)
{
    hsa_device_type_t type;
    hsa_agent_arr_t *agent_arr_ = (hsa_agent_arr_t *) agent_arr;

    ROCM_CALL((*hsa_agent_get_infoPtr)(agent, HSA_AGENT_INFO_DEVICE, &type),
              return _status);

    if (type == HSA_DEVICE_TYPE_GPU) {
        assert(agent_arr_->count < PAPI_ROCM_MAX_DEV_COUNT);
        agent_arr_->agents[agent_arr_->count] = agent;
        ++agent_arr_->count;
    }

    return HSA_STATUS_SUCCESS;
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
        ROCP_REC_ERR_STR("Can't set HSA_TOOLS_LIB. PAPI_ROCM_ROOT not set.");
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
        sprintf(pathname, "%s/lib/librocprofiler64.so", rocm_root);

        err = stat(pathname, &stat_info);
        if (err < 0) {
            sprintf(pathname, "%s/rocprofiler/lib/libprofiler64.so", rocm_root);

            err = stat(pathname, &stat_info);
            if (err < 0) {
                ROCP_REC_ERR_STR("Rocprofiler librocprofiler64.so file not "
                                 "found.");
                return PAPI_EMISC;
            }
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
                ROCP_REC_ERR_STR("Rocprofiler metrics.xml file not found.");
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
    unsigned int dev_id;          /* id of device */
};

int
init_event_table(void)
{
    int papi_errno = PAPI_OK;
    int i;

    for (i = 0; i < agent_arr.count; ++i) {
        ROCP_CALL((*rocp_iterate_infoPtr)(&agent_arr.agents[i],
                                          ROCPROFILER_INFO_KIND_METRIC,
                                          &count_ntv_events_cb,
                                          &ntv_table.count),
                  { ROCP_GET_ERR_STR(); goto fn_fail; });
    }

    ntv_table.events = papi_calloc(ntv_table.count, sizeof(ntv_event_t));
    assert(ntv_table.events);

    struct ntv_arg arg;
    arg.count = 0;

    for (i = 0; i < agent_arr.count; ++i) {
        arg.dev_id = i;
        ROCP_CALL((*rocp_iterate_infoPtr)(&agent_arr.agents[i],
                                          ROCPROFILER_INFO_KIND_METRIC,
                                          &get_ntv_events_cb,
                                          &arg),
                  { ROCP_GET_ERR_STR(); goto fn_fail; });
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_EMISC;
    goto fn_exit;
}

void
init_thread_id_fn(void)
{
    if (thread_id_fn) {
        return;
    }

    thread_id_fn = (_papi_hwi_thread_id_fn) ?
        _papi_hwi_thread_id_fn : _papi_getpid;
}

int
evt_code_to_name(unsigned int event_code, char *name, int len)
{
    /* FIXME: make sure the copied string is not longer than len */
    if (ntv_table.events[event_code].instance >= 0) {
        snprintf(name, (size_t) len, "%s:device=%u:instance=%i",
                 ntv_table.events[event_code].name,
                 ntv_table.events[event_code].ntv_dev,
                 ntv_table.events[event_code].instance);
    } else {
        snprintf(name, (size_t) len, "%s:device=%u",
                 ntv_table.events[event_code].name,
                 ntv_table.events[event_code].ntv_dev);
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
        ROCP_REC_ERR_STR("Number of events exceeds detected count.");
        return HSA_STATUS_ERROR;
    }

    for (instance = 0; instance < instances; ++instance) {
        events[*count].name = strdup(info.metric.name);
        events[*count].descr = strdup(info.metric.description);
        events[*count].ntv_dev = arg->dev_id;
        events[*count].ntv_id = *count;
        events[*count].instance = (instances > 1) ? (int) instance : -1;
        char key[PAPI_MAX_STR_LEN + 1];
        evt_code_to_name((unsigned int) *count, key, PAPI_MAX_STR_LEN);
        htable_insert(htable, key, &events[*count]);
        ++(*count);
    }

    return HSA_STATUS_SUCCESS;
}

/**
 * rocp_ctx_{open,close,start,stop,read,reset} sampling mode utility functions
 *
 */
static struct {
    int device_state[PAPI_ROCM_MAX_DEV_COUNT];
    int queue_ref_count;
    rocprofiler_properties_t ctx_prop;
} sampling_global_state = {{ 0 }, 0, { NULL, 128, NULL, NULL }};

#define SAMPLING_ACQUIRE_DEVICE(i) do { sampling_global_state.device_state[i] = 1; } while(0)
#define SAMPLING_RELEASE_DEVICE(i) do { sampling_global_state.device_state[i] = 0; } while(0)
#define SAMPLING_DEVICE_AVAIL(i)                     (sampling_global_state.device_state[i] == 0)
#define SAMPLING_CONTEXT_PROP                        (sampling_global_state.ctx_prop)
#define SAMPLING_CONTEXT_PROP_QUEUE                  (SAMPLING_CONTEXT_PROP.queue)
#define SAMPLING_FETCH_AND_INCREMENT_QUEUE_COUNTER() (sampling_global_state.queue_ref_count++)
#define SAMPLING_DECREMENT_AND_FETCH_QUEUE_COUNTER() (--sampling_global_state.queue_ref_count)

static int get_target_devs_id(unsigned int *, int, unsigned int **, int *);
static int target_devs_avail(unsigned int *, int);
static int init_features(unsigned int *, int, rocprofiler_feature_t *);
static int sampling_ctx_init(unsigned int *, int, rocp_ctx_t *);
static int sampling_ctx_finalize(rocp_ctx_t *);
static int ctx_open(rocp_ctx_t);
static int ctx_close(rocp_ctx_t);
static int ctx_init(unsigned int *, int, rocp_ctx_t *);
static int ctx_finalize(rocp_ctx_t *);
static int ctx_get_dev_feature_count(rocp_ctx_t, unsigned int);

int
sampling_ctx_open(unsigned int *events_id, int num_events,
                  rocp_ctx_t *rocp_ctx)
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
        return PAPI_ECMP;
    }

    if (rocp_ctx->u.sampling.state & ROCM_EVENTS_RUNNING) {
        SUBDBG("[ROCP sampling mode] Cannot start eventset, already running.");
        return PAPI_ECMP;
    }

    int i;
    for (i = 0; i < rocp_ctx->u.sampling.devs_count; ++i) {
        ROCP_CALL((*rocp_startPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
    }

    rocp_ctx->u.sampling.state |= ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_stop(rocp_ctx_t rocp_ctx)
{
    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP sampling mode] Cannot stop eventset, not opened.");
        return PAPI_ECMP;
    }

    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_RUNNING)) {
        SUBDBG("[ROCP sampling mode] Cannot stop eventset, not running.");
        return PAPI_ECMP;
    }

    int i;
    for (i = 0; i < rocp_ctx->u.sampling.devs_count; ++i) {
        ROCP_CALL((*rocp_stopPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
    }

    rocp_ctx->u.sampling.state &= ~ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_read(rocp_ctx_t rocp_ctx, long long **counts)
{
    int i, j, k = 0;
    int dev_feature_offset = 0;
    unsigned int *devs_id = rocp_ctx->u.sampling.devs_id;
    int dev_count = rocp_ctx->u.sampling.devs_count;
    rocprofiler_feature_t *features = rocp_ctx->u.sampling.features;

    for (i = 0; i < dev_count; ++i) {
        ROCP_CALL((*rocp_readPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
        ROCP_CALL((*rocp_get_dataPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
        ROCP_CALL((*rocp_get_metricsPtr)(rocp_ctx->u.sampling.contexts[i]),
                  return PAPI_EMISC);

        int dev_feature_count = ctx_get_dev_feature_count(rocp_ctx, devs_id[i]);
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
                    return PAPI_EMISC;
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
    int i;
    for (i = 0; i < rocp_ctx->u.sampling.devs_count; ++i) {
        ROCP_CALL((*rocp_resetPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
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

    (*hsa_shut_downPtr)();

    unload_hsa_sym();
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
    }

    ntv_table_p->count = 0;

    papi_free(ntv_table_p->events);

    return PAPI_OK;
}

/**
 * sampling_ctx_open utility functions
 *
 */
int
sampling_ctx_init(unsigned int *events_id, int num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    int num_devs;
    unsigned int *devs_id = NULL;
    rocprofiler_feature_t *features = NULL;
    rocprofiler_t **contexts = NULL;
    long long *counters = NULL;
    *rocp_ctx = NULL;

    papi_errno = get_target_devs_id(events_id, num_events, &devs_id, &num_devs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = target_devs_avail(devs_id, num_devs);
    if (papi_errno != PAPI_OK) {
        SUBDBG("[ROCP sampling mode] Selected GPU devices currently used "
               "by another eventset.");
        goto fn_fail;
    }

    contexts = papi_calloc(num_devs, sizeof(*contexts));
    if (contexts == NULL) {
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
    (*rocp_ctx)->u.sampling.devs_id = devs_id;
    (*rocp_ctx)->u.sampling.devs_count = num_devs;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (devs_id) {
        papi_free(devs_id);
    }
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

    if ((*rocp_ctx)->u.sampling.counters) {
        papi_free((*rocp_ctx)->u.sampling.counters);
    }

    if ((*rocp_ctx)->u.sampling.devs_id) {
        papi_free((*rocp_ctx)->u.sampling.devs_id);
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
    unsigned int *devs_id = rocp_ctx->u.sampling.devs_id;
    int dev_count = rocp_ctx->u.sampling.devs_count;
    rocprofiler_t **contexts = rocp_ctx->u.sampling.contexts;

    for (i = 0; i < dev_count; ++i) {
        int dev_feature_count = ctx_get_dev_feature_count(rocp_ctx, devs_id[i]);
        rocprofiler_feature_t *dev_features = features + dev_feature_offset;

        const uint32_t mode =
            (SAMPLING_FETCH_AND_INCREMENT_QUEUE_COUNTER() == 0) ?
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE |
            ROCPROFILER_MODE_SINGLEGROUP :
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP;

        ROCP_CALL((*rocp_openPtr)(agent_arr.agents[devs_id[i]], dev_features,
                                  dev_feature_count, &contexts[i], mode,
                                  &SAMPLING_CONTEXT_PROP),
                  { papi_errno = PAPI_ECOMBO; goto fn_fail; });

        SAMPLING_ACQUIRE_DEVICE(devs_id[i]);
        dev_feature_offset += dev_feature_count;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    for (j = 0; j < i; ++j) {
        ROCP_CALL((*rocp_closePtr)(contexts[i]),);
        SAMPLING_RELEASE_DEVICE(devs_id[j]);
        SAMPLING_DECREMENT_AND_FETCH_QUEUE_COUNTER();
    }
    goto fn_exit;
}

int
ctx_close(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;
    int i;

    for (i = 0; i < rocp_ctx->u.sampling.devs_count; ++i) {
        ROCP_CALL((*rocp_closePtr)(rocp_ctx->u.sampling.contexts[i]), );
        SAMPLING_RELEASE_DEVICE(rocp_ctx->u.sampling.devs_id[i]);

        if (SAMPLING_DECREMENT_AND_FETCH_QUEUE_COUNTER() == 0) {
            ROCM_CALL((*hsa_queue_destroyPtr)(SAMPLING_CONTEXT_PROP_QUEUE),
                      papi_errno = PAPI_EMISC);
            SAMPLING_CONTEXT_PROP_QUEUE = NULL;
        }
    }

    return papi_errno;
}

int
get_target_devs_id(unsigned int *events_id, int num_events,
                   unsigned int **devs_id, int *num_devs)
{
    int papi_errno = PAPI_OK;

    int devices[PAPI_ROCM_MAX_DEV_COUNT] = { 0 };
    *num_devs = 0;

    int i;
    for (i = 0; i < num_events; ++i) {
        int dev = ntv_table_p->events[events_id[i]].ntv_dev;
        if (devices[dev] == 0) {
            devices[dev] = 1;
            ++(*num_devs);
        }
    }

    *devs_id = papi_calloc(*num_devs, sizeof(*devs_id));
    if (*devs_id == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int j = 0;
    for (i = 0; i < PAPI_ROCM_MAX_DEV_COUNT; ++i) {
        if (devices[i] != 0) {
            (*devs_id)[j++] = i;
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int target_devs_avail(unsigned int *devs_id, int num_devs)
{
    int i;
    for (i = 0; i < num_devs; ++i) {
        if (!SAMPLING_DEVICE_AVAIL(devs_id[i])) {
            return PAPI_ECNFLCT;
        }
    }
    return PAPI_OK;
}

int
init_features(unsigned int *events_id, int num_events,
              rocprofiler_feature_t *features)
{
    int papi_errno = PAPI_OK;

    int i;
    for (i = 0; i < num_events; ++i) {
        char *name = ntv_table_p->events[events_id[i]].name;
        features[i].kind =
            (rocprofiler_feature_kind_t) ROCPROFILER_INFO_KIND_METRIC;
        features[i].name = (const char *) name;
    }

    return papi_errno;
}

static int sampling_ctx_get_dev_feature_count(rocp_ctx_t, unsigned int);
static int intercept_ctx_get_dev_feature_count(rocp_ctx_t, unsigned int);

int
ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, unsigned int i)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_get_dev_feature_count(rocp_ctx, i);
    }

    return intercept_ctx_get_dev_feature_count(rocp_ctx, i);
}

int
sampling_ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, unsigned int i)
{
    int start, stop, j = 0;
    int num_events = rocp_ctx->u.sampling.feature_count;
    unsigned int *events_id = rocp_ctx->u.sampling.events_id;
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
intercept_ctx_get_dev_feature_count(rocp_ctx_t rocp_ctx, unsigned int i)
{
    int start, stop, j = 0;
    int num_events = rocp_ctx->u.intercept.feature_count;
    unsigned int *events_id = rocp_ctx->u.intercept.events_id;
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
    unsigned int *events_id;                      /* array containing ids of events
                                                     monitored in intercept mode */
    int events_count;                             /* number of event ids monitored
                                                     in intercept mode */
    rocprofiler_feature_t *features;              /* array containing rocprofiler
                                                     features monitored in intercept
                                                     mode */
    int feature_count;                            /* number of rocm features monitored
                                                     in intercept mode */
    int active_thread_count;                      /* # threads that launched kernel
                                                     evices in intercept mode */
    int kernel_count;                             /* # number of kernels currently
                                                     running */
} intercept_global_state;

#define INTERCEPT_EVENTS_ID          (intercept_global_state.events_id)
#define INTERCEPT_EVENTS_COUNT       (intercept_global_state.events_count)
#define INTERCEPT_ROCP_FEATURES      (intercept_global_state.features)
#define INTERCEPT_ROCP_FEATURE_COUNT (intercept_global_state.feature_count)
#define INTERCEPT_ACTIVE_THR_COUNT   (intercept_global_state.active_thread_count)
#define INTERCEPT_KERNEL_COUNT       (intercept_global_state.kernel_count)

static int verify_events(unsigned int *, int);
static int init_callbacks(rocprofiler_feature_t *, int);
static int register_dispatch_counter(unsigned long, int *);
static int increment_and_fetch_dispatch_counter(unsigned long);
static int decrement_and_fetch_dispatch_counter(unsigned long);
static int unregister_dispatch_counter(unsigned long);
static int fetch_dispatch_counter(unsigned long);
static cb_context_node_t *alloc_context_node(int);
static void free_context_node(cb_context_node_t *);
static int get_context_node(int, cb_context_node_t **);
static int get_context_counters(unsigned int *, unsigned int,
                                cb_context_node_t *, rocp_ctx_t);
static void put_context_counters(rocprofiler_feature_t *, int,
                                 cb_context_node_t *);
static void put_context_node(unsigned int, cb_context_node_t *);
static int intercept_ctx_init(unsigned int *, int, rocp_ctx_t *);
static int intercept_ctx_finalize(rocp_ctx_t *);

int
intercept_ctx_open(unsigned int *events_id, int num_events,
                   rocp_ctx_t *rocp_ctx)
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

    unsigned long tid = (*thread_id_fn)();
    papi_errno =
        register_dispatch_counter(tid,
                                  &(*rocp_ctx)->u.intercept.dispatch_count);
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

    if (INTERCEPT_ACTIVE_THR_COUNT == 0) {
        goto fn_exit;
    }

    unsigned long tid = (*thread_id_fn)();
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
        goto fn_fail;
    }

    if (rocp_ctx->u.intercept.state & ROCM_EVENTS_RUNNING) {
        SUBDBG("[ROCP intercept mode] Cannot start eventset, already running.");
        goto fn_fail;
    }

    if (INTERCEPT_KERNEL_COUNT++ == 0) {
        ROCP_CALL((*rocp_start_queue_cbsPtr)(), goto fn_fail);
    }

    rocp_ctx->u.intercept.state |= ROCM_EVENTS_RUNNING;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ECMP;
    goto fn_exit;
}

int
intercept_ctx_stop(rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    if (!(rocp_ctx->u.sampling.state & ROCM_EVENTS_OPENED)) {
        SUBDBG("[ROCP intercept mode] Cannot stop eventset, not opened.");
        goto fn_fail;
    }

    if (!(rocp_ctx->u.intercept.state & ROCM_EVENTS_RUNNING)) {
        SUBDBG("[ROCP intercept mode] Cannot stop eventset, not running.");
        goto fn_fail;
    }

    if (--INTERCEPT_KERNEL_COUNT == 0) {
        ROCP_CALL((*rocp_stop_queue_cbsPtr)(), goto fn_fail);
    }

    rocp_ctx->u.intercept.state &= ~ROCM_EVENTS_RUNNING;

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ECMP;
    goto fn_exit;
}

int
intercept_ctx_read(rocp_ctx_t rocp_ctx, long long **counts)
{
    int papi_errno = PAPI_OK;
    unsigned int *events_id = rocp_ctx->u.intercept.events_id;

    _papi_hwi_lock(_rocm_lock);

    unsigned long tid = (*thread_id_fn)();
    int dispatch_count = fetch_dispatch_counter(tid);
    if (dispatch_count == 0) {
        *counts = rocp_ctx->u.intercept.counters;
        goto fn_exit;
    }

    cb_context_node_t *n = NULL;

    int i;
    for (i = 0; i < rocp_ctx->u.intercept.devs_count; ++i) {
        while (dispatch_count > 0) {
            unsigned int dev_id = rocp_ctx->u.intercept.devs_id[i];
            get_context_node(dev_id, &n);
            if (n == NULL) {
                break;
            }

            get_context_counters(events_id, dev_id, n, rocp_ctx);
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

    for (i = 0; i < INTERCEPT_ROCP_FEATURE_COUNT; ++i) {
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

    (*hsa_shut_downPtr)();

    unload_hsa_sym();
    unload_rocp_sym();

    if (INTERCEPT_ROCP_FEATURES) {
        papi_free(INTERCEPT_ROCP_FEATURES);
    }

    if (INTERCEPT_EVENTS_ID) {
        papi_free(INTERCEPT_EVENTS_ID);
    }

    return PAPI_OK;
}

/**
 * intercept_ctx_{open,close} utility functions
 *
 */
int
verify_events(unsigned int *events_id, int num_events)
{
    int i;

    if (INTERCEPT_EVENTS_ID == NULL) {
        return PAPI_OK;
    }

    if (INTERCEPT_EVENTS_COUNT != num_events) {
        return PAPI_ECNFLCT;
    }

    for (i = 0; i < num_events; ++i) {
        void *out;
        if (htable_find(htable, ntv_table_p->events[events_id[i]].name, &out)) {
            return PAPI_ECNFLCT;
        }
    }

    return PAPI_OK;
}

int
intercept_ctx_init(unsigned int *events_id, int num_events,
                   rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    long long *counters = NULL;
    int num_devs;
    unsigned int *devs_id = NULL;
    *rocp_ctx = NULL;

    papi_errno = get_target_devs_id(events_id, num_events, &devs_id,
                                    &num_devs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (INTERCEPT_EVENTS_ID == NULL) {
        INTERCEPT_EVENTS_ID = papi_calloc(num_events, sizeof(int));
        if (INTERCEPT_EVENTS_ID == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        memcpy(INTERCEPT_EVENTS_ID, events_id, num_events * sizeof(unsigned int));

        /* FIXME: assuming the same number of events per device might not be an
         *        always valid assumption */
        int num_events_per_dev = num_events / num_devs;
        INTERCEPT_ROCP_FEATURES = papi_calloc(num_events_per_dev,
                                              sizeof(*INTERCEPT_ROCP_FEATURES));
        if (INTERCEPT_ROCP_FEATURES == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        papi_errno = init_features(INTERCEPT_EVENTS_ID, num_events_per_dev,
                                   INTERCEPT_ROCP_FEATURES);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        INTERCEPT_EVENTS_COUNT = num_events;
        INTERCEPT_ROCP_FEATURE_COUNT = num_events_per_dev;

        int i;
        for (i = 0; i < num_events; ++i) {
            htable_insert(htable, ntv_table_p->events[events_id[i]].name, NULL);
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
    (*rocp_ctx)->u.intercept.devs_id = devs_id;
    (*rocp_ctx)->u.intercept.devs_count = num_devs;
    (*rocp_ctx)->u.intercept.feature_count = num_events;

    papi_errno = init_callbacks(INTERCEPT_ROCP_FEATURES,
                                INTERCEPT_ROCP_FEATURE_COUNT);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    if (devs_id) {
        papi_free(devs_id);
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
intercept_ctx_finalize(rocp_ctx_t *rocp_ctx)
{
    if (*rocp_ctx == NULL) {
        return PAPI_OK;
    }

    if ((*rocp_ctx)->u.intercept.devs_id) {
        papi_free((*rocp_ctx)->u.intercept.devs_id);
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
ctx_init(unsigned int *events_id, int num_events,
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
static hsa_status_t dispatch_cb(const rocprofiler_callback_data_t *, void *,
                                rocprofiler_group_t *);

int
init_callbacks(rocprofiler_feature_t *features, int feature_count)
{
    int papi_errno = PAPI_OK;

    static int callbacks_initialized;

    if (callbacks_initialized) {
        return PAPI_OK;
    }

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
    for (i = 0; i < agent_arr.count; ++i) {
        hsa_agent_t agent = agent_arr.agents[i];

        rocprofiler_pool_t *pool = NULL;
        ROCP_CALL((*rocp_pool_openPtr)(agent, features, feature_count, &pool,
                                       0, &properties),
                  { papi_errno = PAPI_ECMP; goto fn_fail; });

        cb_dispatch_arg.pools[i] = pool;
    }

    rocprofiler_queue_callbacks_t dispatch_ptrs = { 0 };
    dispatch_ptrs.dispatch = dispatch_cb;

    ROCP_CALL((*rocp_set_queue_cbsPtr)(dispatch_ptrs, &cb_dispatch_arg),
              { papi_errno = PAPI_ECMP; goto fn_fail; });

    callbacks_initialized = 1;

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
    ++INTERCEPT_ACTIVE_THR_COUNT;

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
        papi_errno = PAPI_EMISC;
        goto fn_exit;
    }

    htable_delete(htable, (const char *) key);
    --INTERCEPT_ACTIVE_THR_COUNT;

  fn_exit:
    return papi_errno;
}

/**
 * intercept mode counter read infrastructure
 *
 */
static void process_context_entry(cb_context_payload_t *,
                                  rocprofiler_feature_t *, int);
static unsigned int get_dev_id(hsa_agent_t);

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
dispatch_cb(const rocprofiler_callback_data_t *callback_data, void *arg,
            rocprofiler_group_t *group)
{
    hsa_agent_t agent = callback_data->agent;
    hsa_status_t status = HSA_STATUS_SUCCESS;

    unsigned int dev_id = get_dev_id(agent);

    cb_dispatch_arg_t *dispatch_arg = (cb_dispatch_arg_t *) arg;
    rocprofiler_pool_t *pool = dispatch_arg->pools[dev_id];
    rocprofiler_pool_entry_t pool_entry;
    ROCP_CALL((*rocp_pool_fetchPtr)(pool, &pool_entry),
              { status = _status; goto fn_exit; });

    rocprofiler_t *context = pool_entry.context;
    cb_context_payload_t *payload = (cb_context_payload_t *) pool_entry.payload;

    ROCP_CALL((*rocp_get_groupPtr)(context, 0, group),
              { status = _status; goto fn_exit; });

    unsigned long tid = (*thread_id_fn)();
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

    process_context_entry(payload, context_arg->features,
                          context_arg->feature_count);

    return false;
}

void
process_context_entry(cb_context_payload_t *payload,
                      rocprofiler_feature_t *features, int feature_count)
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

    ROCP_CALL((*rocp_group_get_dataPtr)(&payload->group), goto fn_exit);
    ROCP_CALL((*rocp_get_metricsPtr)(payload->group.context), goto fn_exit);

    if (increment_and_fetch_dispatch_counter(payload->tid) < 0) {
        /* thread not registered, ignore counters */
        goto fn_exit;
    }

    cb_context_node_t *n = alloc_context_node(feature_count);
    if (n == NULL) {
        decrement_and_fetch_dispatch_counter(payload->tid);
        goto fn_exit;
    }

    n->tid = payload->tid;
    put_context_counters(features, feature_count, n);
    put_context_node(get_dev_id(payload->agent), n);

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
put_context_counters(rocprofiler_feature_t *features, int feature_count,
                     cb_context_node_t *n)
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
put_context_node(unsigned int dev_id, cb_context_node_t *n)
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
    htable_errno = htable_find(htable, (const char *) key,
                               (void **) &counter_p);
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
    htable_errno = htable_find(htable, (const char *) key,
                               (void **) &counter_p);
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
        unsigned long tid = (*thread_id_fn)();
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
    htable_errno = htable_find(htable, (const char *) key,
                               (void **) &counter_p);
    if (htable_errno != HTABLE_SUCCESS) {
        return 0;
    }

    return --(*counter_p);
}

int
get_context_counters(unsigned int *events_id, unsigned int dev_id,
                     cb_context_node_t *n, rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    /* Here we get events_id ordered according to user's viewpoint and we want
     * to map these to events_id ordered according to callbacks' viewpoint. We
     * compare events from the user and the callbacks using a brute force
     * approach as the number of events is typically small. */
    int i, j;
    for (i = 0; i < INTERCEPT_ROCP_FEATURE_COUNT; ++i) {
        const char *cb_name = INTERCEPT_ROCP_FEATURES[i].name;

        for (j = 0; j < rocp_ctx->u.intercept.feature_count; ++j) {
            const char *usr_name =
                ntv_table_p->events[events_id[j]].name;
            unsigned int usr_ntv_dev =
                ntv_table_p->events[events_id[j]].ntv_dev;

            if (dev_id == usr_ntv_dev && strcmp(usr_name, cb_name) == 0) {
                break;
            }
        }
        assert(j < rocp_ctx->u.intercept.feature_count);
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

unsigned int
get_dev_id(hsa_agent_t agent)
{
    unsigned int dev_id;
    for (dev_id = 0; dev_id < (unsigned int) agent_arr.count; ++dev_id) {
        if (memcmp(&agent_arr.agents[dev_id], &agent, sizeof(agent)) == 0) {
            return dev_id;
        }
    }
    return -1;
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
#endif /* End of ROCM_PROF_ROCPROFILER */
