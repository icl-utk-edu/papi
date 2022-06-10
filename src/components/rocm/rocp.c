/**
 * @file    rocp.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 */

#include <sys/stat.h>
#include <dlfcn.h>
#include <hsa.h>
#include <rocprofiler.h>
#include <unistd.h>

#include "rocp.h"

struct rocp_ctx {
    union {
        struct {
            int state;                       /* state of kernel interception */
            ntv_event_table_t *ntv_table;    /* table containing all component events */
            long long *counters;             /* thread's private counters */
            int dispatch_count;              /* how many kernel dispatches this thread has done */
            unsigned *devs_id;               /* list of monitored devices */
            unsigned devs_count;             /* number of monitored devices */
            unsigned feature_count;          /* number of features being monitored */
        } intercept;
        struct {
            int state;                       /* state of sampling */
            ntv_event_table_t *ntv_table;    /* table containing all component events */
            long long *counters;             /* thread's private counters */
            rocprofiler_feature_t *features; /* rocprofiler features */
            unsigned feature_count;          /* number of features being monitored */
            rocprofiler_t **contexts;        /* rocprofiler context array for multiple device monitoring */
            unsigned *devs_id;               /* list of monitored device ids */
            unsigned devs_count;             /* number of monitored devices */
            int *sorted_events_id;           /* list of event ids sorted by device */
        } sampling;
    } u;
};

#ifndef PAPI_ROCM_MAX_DEV_COUNT
#define PAPI_ROCM_MAX_DEV_COUNT (32)
#endif

typedef struct {
    hsa_agent_t agents[PAPI_ROCM_MAX_DEV_COUNT]; /* array of hsa agents */
    unsigned count;                              /* number of hsa agents in agent array */
} hsa_agent_arr_t;

unsigned rocm_prof_mode;
unsigned _rocm_lock;                         /* internal rocm component lock (allocated at configure time) */

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
static int init_event_table(ntv_event_table_t *ntv_table);
static int unload_hsa_sym(void);
static int unload_rocp_sym(void);
static int init_agent_array(void);
static int sampling_ctx_open(ntv_event_table_t *, int *, unsigned, rocp_ctx_t *);
static int intercept_ctx_open(ntv_event_table_t *, int *, unsigned,
                              rocp_ctx_t *);
static int sampling_ctx_close(rocp_ctx_t);
static int intercept_ctx_close(rocp_ctx_t);
static int sampling_ctx_start(rocp_ctx_t);
static int intercept_ctx_start(rocp_ctx_t);
static int sampling_ctx_stop(rocp_ctx_t);
static int intercept_ctx_stop(rocp_ctx_t);
static int sampling_ctx_read(rocp_ctx_t, int *, long long **);
static int intercept_ctx_read(rocp_ctx_t, int *, long long **);
static int sampling_ctx_reset(rocp_ctx_t);
static int intercept_ctx_reset(rocp_ctx_t);
static int sampling_rocp_shutdown(ntv_event_table_t *);
static int intercept_rocp_shutdown(ntv_event_table_t *);
static void init_thread_id_fn(void);

static void *hsa_dlp = NULL;
static void *rocp_dlp = NULL;
static const char *init_err_str_ptr;
static char init_err_str[PAPI_MAX_STR_LEN];
static hsa_agent_arr_t agent_arr;
static unsigned long (*thread_id_fn)(void);

int
rocp_init_environment(const char **err_string)
{
    int papi_errno = init_rocp_env();
    if (papi_errno != PAPI_OK) {
        ROCP_PUT_ERR_STR(*err_string);
    }
    return papi_errno;
}

int
rocp_init(ntv_event_table_t *ntv_table, const char **err_string)
{
    int papi_errno = PAPI_OK;

    papi_errno = load_hsa_sym();
    if (papi_errno != PAPI_OK) {
        ROCM_PUT_ERR_STR(*err_string);
        goto fn_fail;
    }

    papi_errno = load_rocp_sym();
    if (papi_errno != PAPI_OK) {
        ROCM_PUT_ERR_STR(*err_string);
        goto fn_fail;
    }

    /* NOTE: hsa_init() initializes hsa runtime, which further
     *       initializes rocprofiler whenever HSA_TOOLS_LIB is
     *       set (as done by init_rocp_env()). */
    hsa_status_t status = (*hsa_initPtr)();
    if (status != HSA_STATUS_SUCCESS) {
        ROCM_GET_ERR_STR(status);
        ROCM_PUT_ERR_STR(*err_string);
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = init_agent_array();
    if (papi_errno != PAPI_OK) {
        ROCM_PUT_ERR_STR(*err_string);
        (*hsa_shut_downPtr)();
        goto fn_fail;
    }

    papi_errno = init_event_table(ntv_table);
    if (papi_errno != PAPI_OK) {
        ROCM_PUT_ERR_STR(*err_string);
        (*hsa_shut_downPtr)();
        goto fn_fail;
    }

    init_thread_id_fn();

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_rocp_sym();
    unload_hsa_sym();
    goto fn_exit;
}

int
rocp_ctx_open(ntv_event_table_t *ntv_table, int *events_id, unsigned num_events,
              rocp_ctx_t *rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_open(ntv_table, events_id, num_events, rocp_ctx);
    }

    return intercept_ctx_open(ntv_table, events_id, num_events, rocp_ctx);
}

int
rocp_ctx_close(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_close(rocp_ctx);
    }

    return intercept_ctx_close(rocp_ctx);
}

int
rocp_ctx_start(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_start(rocp_ctx);
    }

    return intercept_ctx_start(rocp_ctx);
}

int
rocp_ctx_stop(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_stop(rocp_ctx);
    }

    return intercept_ctx_stop(rocp_ctx);
}

int
rocp_ctx_read(rocp_ctx_t rocp_ctx, int *events_id, long long **counts)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_read(rocp_ctx, events_id, counts);
    }

    return intercept_ctx_read(rocp_ctx, events_id, counts);
}

int
rocp_ctx_reset(rocp_ctx_t rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_reset(rocp_ctx);
    }

    return intercept_ctx_reset(rocp_ctx);
}

int
rocp_shutdown(ntv_event_table_t *ntv_table)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_rocp_shutdown(ntv_table);
    }

    return intercept_rocp_shutdown(ntv_table);
}

/**
 * rocp_init utility functions
 *
 */
int
load_hsa_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PAPI_MAX_STR_LEN] = { 0 };
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root == NULL) {
        ROCP_REC_ERR_STR("Can't load libhsa-runtime64.so, PAPI_ROCM_ROOT not set.");
        goto fn_fail;
    }

    int expect = snprintf(pathname, PAPI_MAX_STR_LEN,
                          "%s/lib/libhsa-runtime64.so", rocm_root);
    if (expect > PAPI_MAX_STR_LEN) {
        SUBDBG("Error string truncated");
    }

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

    char pathname[PAPI_MAX_STR_LEN];
    char *rocm_root = getenv("PAPI_ROCM_ROOT");
    if (rocm_root == NULL) {
        ROCP_REC_ERR_STR("Can't set HSA_TOOLS_LIB. PAPI_ROCM_ROOT not set.");
        return PAPI_EMISC;
    }

    int err;
    int override_hsa_tools_lib = 1;
    int expect;
    struct stat stat_info;
    char *hsa_tools_lib = getenv("HSA_TOOLS_LIB");
    if (hsa_tools_lib) {
        err = stat(hsa_tools_lib, &stat_info);
        if (err == 0) {
            override_hsa_tools_lib = 0;
        }
    }

    if (override_hsa_tools_lib) {
        expect = snprintf(pathname, PAPI_MAX_STR_LEN,
                          "%s/rocprofiler/lib/librocprofiler64.so",
                          rocm_root);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("Error string truncated");
        }

        setenv("HSA_TOOLS_LIB", pathname, 1);
    }

    int override_rocp_metrics = 1;
    char *rocp_metrics = getenv("ROCP_METRICS");
    if (rocp_metrics) {
        err = stat(rocp_metrics, &stat_info);
        if (err == 0) {
            override_rocp_metrics = 0;
        }
    }

    if (override_rocp_metrics) {
        expect = snprintf(pathname, PAPI_MAX_STR_LEN,
                          "%s/rocprofiler/lib/metrics.xml",
                          rocm_root);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("Error string truncated");
        }

        err = stat(pathname, &stat_info);
        if (err < 0) {
            ROCP_REC_ERR_STR("Rocprofiler metrics.xml file not found.");
            return PAPI_EMISC;
        }

        setenv("ROCP_METRICS", pathname, 1);
    }

    //setenv("AQLPROFILE_READ_API", "1", 0);
    //setenv("ROCPROFILER_LOG", "1", 0);
    //setenv("HSA_VEN_AMD_AQLPROFILE_LOG", "1", 0);

    rocp_env_initialized = 1;
    return PAPI_OK;
}

static hsa_status_t count_ntv_events_cb(const rocprofiler_info_data_t, void *);
static hsa_status_t get_ntv_events_cb(const rocprofiler_info_data_t, void *);

struct ntv_arg {
    ntv_event_table_t *ntv_table; /* pointer to component's native events table */
    unsigned count;               /* number of devices counted so far */
    unsigned dev_id;              /* id of device */
};

int
init_event_table(ntv_event_table_t *ntv_table)
{
    int papi_errno = PAPI_OK;
    unsigned i;

    for (i = 0; i < agent_arr.count; ++i) {
        ROCP_CALL((*rocp_iterate_infoPtr)(&agent_arr.agents[i],
                                          ROCPROFILER_INFO_KIND_METRIC,
                                          &count_ntv_events_cb,
                                          &ntv_table->count),
                  { ROCP_GET_ERR_STR(); goto fn_fail; });
    }

    ntv_table->events = papi_calloc(ntv_table->count, sizeof(ntv_event_t));
    assert(ntv_table->events);

    struct ntv_arg arg;
    arg.ntv_table = ntv_table;
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
        _papi_hwi_thread_id_fn :
        (unsigned long (*)(void)) getpid;
}

/**
 * init_event_table utility functions
 *
 */
hsa_status_t
count_ntv_events_cb(const rocprofiler_info_data_t info, void *count)
{
    (*(unsigned *) count) += info.metric.instances;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_ntv_events_cb(const rocprofiler_info_data_t info, void *ntv_arg)
{
    struct ntv_arg *arg = (struct ntv_arg *) ntv_arg;
    const unsigned instances = info.metric.instances;
    ntv_event_table_t *ntv_table_ = arg->ntv_table;
    unsigned capacity = ntv_table_->count;
    unsigned *count = &arg->count;
    ntv_event_t *events = ntv_table_->events;
    unsigned instance;

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

static int get_target_devs_id(ntv_event_table_t *, int *, unsigned, unsigned **,
                              unsigned *);
static int target_devs_avail(unsigned *, unsigned);
static int sort_events_by_device(ntv_event_table_t *, int *, unsigned,
                                 unsigned *, unsigned, int *);
static int init_features(ntv_event_table_t *, int *, unsigned,
                         rocprofiler_feature_t *);
static int sampling_ctx_init(ntv_event_table_t *, int *, unsigned,
                             rocp_ctx_t *);
static int sampling_ctx_finalize(rocp_ctx_t *);
static int ctx_open(rocp_ctx_t);
static int ctx_close(rocp_ctx_t);
static unsigned get_user_counter_id(rocp_ctx_t, int *, unsigned);
static int ctx_init(ntv_event_table_t *, int *, unsigned, rocp_ctx_t *);
static int ctx_finalize(rocp_ctx_t *);

int
sampling_ctx_open(ntv_event_table_t *ntv_table, int *events_id,
                  unsigned num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;

    if (num_events <= 0) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_rocm_lock);

    papi_errno = ctx_init(ntv_table, events_id, num_events, rocp_ctx);
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

    unsigned i;
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

    unsigned i;
    for (i = 0; i < rocp_ctx->u.sampling.devs_count; ++i) {
        ROCP_CALL((*rocp_stopPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
    }

    rocp_ctx->u.sampling.state &= ~ROCM_EVENTS_RUNNING;
    return PAPI_OK;
}

int
sampling_ctx_read(rocp_ctx_t rocp_ctx, int *events_id, long long **counts)
{
    unsigned i, j, k;
    unsigned dev_count = rocp_ctx->u.sampling.devs_count;

    for (i = 0; i < dev_count; ++i) {
        ROCP_CALL((*rocp_readPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
        ROCP_CALL((*rocp_get_dataPtr)(rocp_ctx->u.sampling.contexts[i], 0),
                  return PAPI_EMISC);
        ROCP_CALL((*rocp_get_metricsPtr)(rocp_ctx->u.sampling.contexts[i]),
                  return PAPI_EMISC);

        unsigned dev_feature_count =
            rocp_ctx->u.sampling.feature_count / dev_count;
        rocprofiler_feature_t *dev_features =
            rocp_ctx->u.sampling.features + (i * dev_feature_count);
        long long *counters = rocp_ctx->u.sampling.counters;

        for (j = 0; j < dev_feature_count; ++j) {
            unsigned sorted_event_id = (i * dev_feature_count) + j;
            k = get_user_counter_id(rocp_ctx, events_id, sorted_event_id);
            switch(dev_features[j].data.kind) {
                case ROCPROFILER_DATA_KIND_INT32:
                    counters[k] = (long long) dev_features[j].data.result_int32;
                    break;
                case ROCPROFILER_DATA_KIND_INT64:
                    counters[k] = dev_features[j].data.result_int64;
                    break;
                case ROCPROFILER_DATA_KIND_FLOAT:
                    counters[k] = (long long) dev_features[j].data.result_float;
                    break;
                case ROCPROFILER_DATA_KIND_DOUBLE:
                    counters[k] = (long long) dev_features[j].data.result_double;
                    break;
                default:
                    return PAPI_EMISC;
            }
        }
    }
    *counts = rocp_ctx->u.sampling.counters;

    return PAPI_OK;
}

int
sampling_ctx_reset(rocp_ctx_t rocp_ctx)
{
    unsigned i;
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
static int shutdown_event_table(ntv_event_table_t *);

int
sampling_rocp_shutdown(ntv_event_table_t *ntv_table)
{
    shutdown_event_table(ntv_table);

    (*hsa_shut_downPtr)();

    unload_hsa_sym();
    unload_rocp_sym();

    return PAPI_OK;
}

int
shutdown_event_table(ntv_event_table_t *ntv_table)
{
    unsigned i;

    for (i = 0; i < ntv_table->count; ++i) {
        papi_free(ntv_table->events[i].name);
        papi_free(ntv_table->events[i].descr);
    }

    ntv_table->count = 0;

    papi_free(ntv_table->events);

    return PAPI_OK;
}

/**
 * sampling_ctx_open utility functions
 *
 */
int
sampling_ctx_init(ntv_event_table_t *ntv_table, int *events_id,
                  unsigned num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    unsigned num_devs;
    unsigned *devs_id = NULL;
    int *sorted_events_id = NULL;
    rocprofiler_feature_t *features = NULL;
    rocprofiler_t **contexts = NULL;
    long long *counters = NULL;
    *rocp_ctx = NULL;

    papi_errno = get_target_devs_id(ntv_table, events_id, num_events,
                                    &devs_id, &num_devs);
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

    /* Events can be added to eventsets in any order. When contexts are opened
     * the feature array has to contain events ordered by device. For this
     * reason we need to remap events from the user order to device order. */
    sorted_events_id = papi_malloc(num_events * sizeof(int));
    if (sorted_events_id == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    counters = papi_malloc(num_events * sizeof(*counters));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = sort_events_by_device(ntv_table, events_id, num_events,
                                       devs_id, num_devs, sorted_events_id);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = init_features(ntv_table, sorted_events_id, num_events,
                               features);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    *rocp_ctx = papi_calloc(1, sizeof(struct rocp_ctx));
    if (*rocp_ctx == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    (*rocp_ctx)->u.sampling.ntv_table = ntv_table;
    (*rocp_ctx)->u.sampling.features = features;
    (*rocp_ctx)->u.sampling.feature_count = num_events;
    (*rocp_ctx)->u.sampling.contexts = contexts;
    (*rocp_ctx)->u.sampling.sorted_events_id = sorted_events_id;
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
    if (sorted_events_id) {
        papi_free(sorted_events_id);
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

    if ((*rocp_ctx)->u.sampling.sorted_events_id) {
        papi_free((*rocp_ctx)->u.sampling.sorted_events_id);
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
    unsigned i, j;
    rocprofiler_feature_t *features = rocp_ctx->u.sampling.features;
    unsigned feature_count = rocp_ctx->u.sampling.feature_count;
    unsigned *devs_id = rocp_ctx->u.sampling.devs_id;
    unsigned dev_count = rocp_ctx->u.sampling.devs_count;
    rocprofiler_t **contexts = rocp_ctx->u.sampling.contexts;

    for (i = 0; i < dev_count; ++i) {
        unsigned dev_feature_count = feature_count / dev_count;
        rocprofiler_feature_t *dev_features = features + (i * dev_feature_count);

        const uint32_t mode =
            (SAMPLING_FETCH_AND_INCREMENT_QUEUE_COUNTER() == 0) ?
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE |
            ROCPROFILER_MODE_SINGLEGROUP :
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP;

        ROCP_CALL((*rocp_openPtr)(agent_arr.agents[i], dev_features,
                                  dev_feature_count, &contexts[i], mode,
                                  &SAMPLING_CONTEXT_PROP),
                  { papi_errno = PAPI_ECOMBO; goto fn_fail; });

        SAMPLING_ACQUIRE_DEVICE(devs_id[i]);
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
    unsigned i;

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
get_target_devs_id(ntv_event_table_t *ntv_table, int *events_id,
                   unsigned num_events, unsigned **devs_id, unsigned *num_devs)
{
    int papi_errno = PAPI_OK;

    int devices[PAPI_ROCM_MAX_DEV_COUNT] = { 0 };
    *num_devs = 0;

    unsigned i;
    for (i = 0; i < num_events; ++i) {
        int dev = ntv_table->events[events_id[i]].ntv_dev;
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

    unsigned j = 0;
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

int target_devs_avail(unsigned *devs_id, unsigned num_devs)
{
    unsigned i;
    for (i = 0; i < num_devs; ++i) {
        if (!SAMPLING_DEVICE_AVAIL(devs_id[i])) {
            return PAPI_ECNFLCT;
        }
    }
    return PAPI_OK;
}

int
sort_events_by_device(ntv_event_table_t *ntv_table, int *events_id,
                      unsigned num_events, unsigned *devs_id,
                      unsigned dev_count, int *sorted_events_id)
{
    unsigned i, j, k = 0;
    for (i = 0; i < dev_count; ++i) {
        for (j = 0; j < num_events; ++j) {
            if (ntv_table->events[events_id[j]].ntv_dev == devs_id[i]) {
                sorted_events_id[k++] = events_id[j];
            }
        }
    }
    return PAPI_OK;
}

int
init_features(ntv_event_table_t *ntv_table, int *events_id, unsigned num_events,
              rocprofiler_feature_t *features)
{
    int papi_errno = PAPI_OK;

    unsigned i;
    for (i = 0; i < num_events; ++i) {
        char *name = ntv_table->events[events_id[i]].name;
        features[i].kind = ROCPROFILER_INFO_KIND_METRIC;
        features[i].name = (const char *) name;
    }

    return papi_errno;
}

/**
 * sampling_ctx_read utility functions
 *
 */
unsigned
get_user_counter_id(rocp_ctx_t rocp_ctx, int *events_id, unsigned j)
{
    unsigned i, curr_event_id = rocp_ctx->u.sampling.sorted_events_id[j];
    for (i = 0; i < rocp_ctx->u.sampling.feature_count; ++i) {
        unsigned counter_event_id = events_id[i];
        if (counter_event_id == curr_event_id) {
            break;
        }
    }

    return i;
}

/**
 * rocp_ctx_{open,close,start,stop,read,reset} intercept mode utility functions
 *
 */
typedef struct cb_dispatch_counter {
    unsigned long tid;                               /* id of owning thread */
    int *count;                                      /* pointer to rocp_ctx dispatch counter */
} cb_dispatch_counter_t;

typedef struct cb_context_node {
    unsigned long tid;
    long long *counters;
    struct cb_context_node *next;
} cb_context_node_t;

static struct {
    int *events_id;                               /* array containing ids of events
                                                     monitored in intercept mode */
    unsigned events_count;                        /* number of event ids monitored
                                                     in intercept mode */
    rocprofiler_feature_t *features;              /* array containing rocprofiler
                                                     features monitored in intercept
                                                     mode */
    unsigned feature_count;                       /* number of rocm features monitored
                                                     in intercept mode */
    cb_dispatch_counter_t *dispatch_count_arr;    /* array containing, for each
                                                     active thread, the number
                                                     of kernel dispatches done */
    unsigned active_thread_count;                 /* # threads that launched kernel
                                                     evices in intercept mode */
    int kernel_count;                             /* # number of kernels currently
                                                     running */
} intercept_global_state;

#define INTERCEPT_EVENTS_ID          (intercept_global_state.events_id)
#define INTERCEPT_EVENTS_COUNT       (intercept_global_state.events_count)
#define INTERCEPT_ROCP_FEATURES      (intercept_global_state.features)
#define INTERCEPT_ROCP_FEATURE_COUNT (intercept_global_state.feature_count)
#define INTERCEPT_DISPATCH_COUNT_ARR (intercept_global_state.dispatch_count_arr)
#define INTERCEPT_ACTIVE_THR_COUNT   (intercept_global_state.active_thread_count)
#define INTERCEPT_KERNEL_COUNT       (intercept_global_state.kernel_count)

static int compare_events(ntv_event_table_t *, int *, int *, unsigned);
static int init_callbacks(rocprofiler_feature_t *, unsigned);
static int register_dispatch_counter(unsigned long, int *);
static int increment_and_fetch_dispatch_counter(unsigned long);
static int decrement_and_fetch_dispatch_counter(unsigned long);
static int unregister_dispatch_counter(unsigned long);
static int fetch_dispatch_counter(unsigned long);
static cb_context_node_t *alloc_context_node(unsigned);
static void free_context_node(cb_context_node_t *);
static int get_context_node(int, cb_context_node_t **);
static int get_context_counters(int *, unsigned, cb_context_node_t *,
                                rocp_ctx_t);
static void put_context_counters(rocprofiler_feature_t *, unsigned,
                                 cb_context_node_t *);
static void put_context_node(int, cb_context_node_t *);
static int intercept_ctx_init(ntv_event_table_t *, int *, unsigned,
                              rocp_ctx_t *);
static int intercept_ctx_finalize(rocp_ctx_t *);

int
intercept_ctx_open(ntv_event_table_t *ntv_table, int *events_id,
                   unsigned num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;

    if (num_events <= 0) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_rocm_lock);

    if (INTERCEPT_EVENTS_ID != NULL) {
        int res = compare_events(ntv_table, events_id, INTERCEPT_EVENTS_ID,
                                 num_events);
        if (res != 0) {
            SUBDBG("[ROCP intercept mode] Can only monitor one set of events "
                   "per application run.");
            papi_errno = PAPI_ECNFLCT;
            goto fn_fail;
        }
    }

    papi_errno = ctx_init(ntv_table, events_id, num_events, rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = init_callbacks(INTERCEPT_ROCP_FEATURES,
                                INTERCEPT_ROCP_FEATURE_COUNT);
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
intercept_ctx_read(rocp_ctx_t rocp_ctx, int *events_id, long long **counts)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_rocm_lock);

    unsigned long tid = (*thread_id_fn)();
    int dispatch_count = fetch_dispatch_counter(tid);
    if (dispatch_count == 0) {
        *counts = rocp_ctx->u.intercept.counters;
        goto fn_exit;
    }

    cb_context_node_t *n = NULL;

    unsigned i;
    for (i = 0; i < rocp_ctx->u.intercept.devs_count; ++i) {
        while (dispatch_count > 0) {
            unsigned dev_id = rocp_ctx->u.intercept.devs_id[i];
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
    unsigned i;

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
intercept_rocp_shutdown(ntv_event_table_t *ntv_table)
{
    /* Uncommenting this causes a double free error. */
    //unsigned i;
    //for (i = 0; i < agent_arr.count; ++i) {
    //    ROCP_CALL((*rocp_pool_closePtr)(cb_dispatch_arg.pools[i]), );
    //}

    shutdown_event_table(ntv_table);

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
compare_events(ntv_event_table_t *ntv_table, int *events_id, int *cb_events_id,
               unsigned num_events)
{
    int res;
    unsigned i, j;

    if (INTERCEPT_EVENTS_COUNT != num_events) {
        return INTERCEPT_ROCP_FEATURE_COUNT - num_events;
    }

    /* brute force search is fine as an eventset will never contain more
     * than a few tens of events */
    for (i = 0; i < num_events; ++i) {
        char *event_name = ntv_table->events[events_id[i]].name;
        for (j = 0; j < num_events; ++j) {
            char *cb_event_name = ntv_table->events[cb_events_id[j]].name;
            res = strcmp(event_name, cb_event_name);
            if (res == 0) {
                break;
            }
        }
        if (res != 0) {
            return res;
        }
    }

    return 0;
}

int
intercept_ctx_init(ntv_event_table_t *ntv_table, int *events_id,
                   unsigned num_events, rocp_ctx_t *rocp_ctx)
{
    int papi_errno = PAPI_OK;
    long long *counters = NULL;
    unsigned num_devs;
    unsigned *devs_id = NULL;
    *rocp_ctx = NULL;

    papi_errno = get_target_devs_id(ntv_table, events_id, num_events,
                                    &devs_id, &num_devs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (INTERCEPT_EVENTS_ID == NULL) {
        INTERCEPT_EVENTS_ID = papi_calloc(num_events, sizeof(int));
        if (INTERCEPT_EVENTS_ID == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        papi_errno = sort_events_by_device(ntv_table, events_id, num_events,
                                           devs_id, num_devs,
                                           INTERCEPT_EVENTS_ID);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        int num_events_per_dev = num_events / num_devs;
        INTERCEPT_ROCP_FEATURES = papi_calloc(num_events_per_dev,
                                              sizeof(*INTERCEPT_ROCP_FEATURES));
        if (INTERCEPT_ROCP_FEATURES == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        papi_errno = init_features(ntv_table, INTERCEPT_EVENTS_ID,
                                   num_events_per_dev,
                                   INTERCEPT_ROCP_FEATURES);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        INTERCEPT_EVENTS_COUNT = num_events;
        INTERCEPT_ROCP_FEATURE_COUNT = num_events_per_dev;
    }

    counters = papi_calloc(num_events, sizeof(*counters));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    *rocp_ctx = papi_calloc(1, sizeof(struct rocp_ctx));
    if (*rocp_ctx == NULL) {
        return PAPI_ENOMEM;
    }

    (*rocp_ctx)->u.intercept.ntv_table = ntv_table;
    (*rocp_ctx)->u.intercept.counters = counters;
    (*rocp_ctx)->u.intercept.dispatch_count = 0;
    (*rocp_ctx)->u.intercept.devs_id = devs_id;
    (*rocp_ctx)->u.intercept.devs_count = num_devs;
    (*rocp_ctx)->u.intercept.feature_count = num_events;

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
ctx_init(ntv_event_table_t *ntv_table, int *events_id, unsigned num_events,
         rocp_ctx_t *rocp_ctx)
{
    if (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE) {
        return sampling_ctx_init(ntv_table, events_id, num_events, rocp_ctx);
    }

    return intercept_ctx_init(ntv_table, events_id, num_events, rocp_ctx);
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
    unsigned feature_count;
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
init_callbacks(rocprofiler_feature_t *features, unsigned feature_count)
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

    unsigned i;
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

    if (INTERCEPT_DISPATCH_COUNT_ARR) {
        unsigned i;
        for (i = 0; i < INTERCEPT_ACTIVE_THR_COUNT; ++i) {
            if (INTERCEPT_DISPATCH_COUNT_ARR[i].tid == tid) {
                SUBDBG("Trying to PAPI_start an eventset that has not been "
                       "PAPI_stop'ed");
                papi_errno = PAPI_ECNFLCT;
                goto fn_exit;
            }
        }

        INTERCEPT_DISPATCH_COUNT_ARR =
            papi_realloc(INTERCEPT_DISPATCH_COUNT_ARR,
                         ++INTERCEPT_ACTIVE_THR_COUNT *
                         sizeof(*INTERCEPT_DISPATCH_COUNT_ARR));
        if (INTERCEPT_DISPATCH_COUNT_ARR == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
        INTERCEPT_DISPATCH_COUNT_ARR[INTERCEPT_ACTIVE_THR_COUNT - 1].tid = tid;
        INTERCEPT_DISPATCH_COUNT_ARR[INTERCEPT_ACTIVE_THR_COUNT - 1].count = counter;
        goto fn_exit;
    }

    INTERCEPT_DISPATCH_COUNT_ARR =
        papi_calloc(1, sizeof(*INTERCEPT_DISPATCH_COUNT_ARR));
    if (INTERCEPT_DISPATCH_COUNT_ARR == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_exit;
    }

    INTERCEPT_DISPATCH_COUNT_ARR[INTERCEPT_ACTIVE_THR_COUNT].tid = tid;
    INTERCEPT_DISPATCH_COUNT_ARR[INTERCEPT_ACTIVE_THR_COUNT].count = counter;
    ++INTERCEPT_ACTIVE_THR_COUNT;

  fn_exit:
    return papi_errno;
}

int
unregister_dispatch_counter(unsigned long tid)
{
    int papi_errno = PAPI_OK;

    if (INTERCEPT_ACTIVE_THR_COUNT == 1) {
        assert(INTERCEPT_DISPATCH_COUNT_ARR[0].tid == tid);
        papi_free(INTERCEPT_DISPATCH_COUNT_ARR);
        INTERCEPT_DISPATCH_COUNT_ARR = NULL;
        INTERCEPT_ACTIVE_THR_COUNT = 0;
        return papi_errno;
    }

    cb_dispatch_counter_t *tmp =
        papi_calloc(INTERCEPT_ACTIVE_THR_COUNT - 1,
                    sizeof(*tmp));
    if (tmp == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_exit;
    }

    unsigned i, j = 0;
    for (i = 0; i < INTERCEPT_ACTIVE_THR_COUNT; ++i) {
        if (INTERCEPT_DISPATCH_COUNT_ARR[i].tid != tid) {
            tmp[j].tid = INTERCEPT_DISPATCH_COUNT_ARR[i].tid;
            tmp[j].count = INTERCEPT_DISPATCH_COUNT_ARR[i].count;
            ++j;
        }
    }

    papi_free(INTERCEPT_DISPATCH_COUNT_ARR);
    INTERCEPT_DISPATCH_COUNT_ARR = tmp;

    --INTERCEPT_ACTIVE_THR_COUNT;

  fn_exit:
    return papi_errno;
}

/**
 * intercept mode counter read infrastructure
 *
 */
static void process_context_entry(cb_context_payload_t *,
                                  rocprofiler_feature_t *, unsigned);
static int get_dev_id(hsa_agent_t);

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

    unsigned dev_id = get_dev_id(agent);

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
                      rocprofiler_feature_t *features, unsigned feature_count)
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
alloc_context_node(unsigned num_events)
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
put_context_counters(rocprofiler_feature_t *features, unsigned feature_count,
                     cb_context_node_t *n)
{
    unsigned i;
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
    unsigned i;

    for (i = 0; i < INTERCEPT_ACTIVE_THR_COUNT; ++i) {
        if (INTERCEPT_DISPATCH_COUNT_ARR[i].tid == tid) {
            ++(*INTERCEPT_DISPATCH_COUNT_ARR[i].count);
            break;
        }
    }

    return (i == INTERCEPT_ACTIVE_THR_COUNT) ?
        -1 : *INTERCEPT_DISPATCH_COUNT_ARR[i].count;
}

int
fetch_dispatch_counter(unsigned long tid)
{
    unsigned i;
    for (i = 0; i < INTERCEPT_ACTIVE_THR_COUNT; ++i) {
        if (INTERCEPT_DISPATCH_COUNT_ARR[i].tid == tid) {
            break;
        }
    }
    assert(i < INTERCEPT_ACTIVE_THR_COUNT);
    return *INTERCEPT_DISPATCH_COUNT_ARR[i].count;
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
    unsigned i;
    for (i = 0; i < INTERCEPT_ACTIVE_THR_COUNT; ++i) {
        if (INTERCEPT_DISPATCH_COUNT_ARR[i].tid == tid) {
            --(*INTERCEPT_DISPATCH_COUNT_ARR[i].count);
            break;
        }
    }

    return (i == INTERCEPT_ACTIVE_THR_COUNT) ?
        -1 : *INTERCEPT_DISPATCH_COUNT_ARR[i].count;
}

int
get_context_counters(int *events_id, unsigned dev_id, cb_context_node_t *n,
                     rocp_ctx_t rocp_ctx)
{
    int papi_errno = PAPI_OK;

    /* Here we get events_id ordered according to user's viewpoint and we want
     * to map these to events_id ordered according to callbacks' viewpoint. We
     * compare events from the user and the callbacks using a brute force
     * approach as the number of events is typically small. */
    unsigned i, j;
    for (i = 0; i < INTERCEPT_ROCP_FEATURE_COUNT; ++i) {
        const char *cb_name = INTERCEPT_ROCP_FEATURES[i].name;

        for (j = 0; j < rocp_ctx->u.intercept.feature_count; ++j) {
            const char *usr_name =
                rocp_ctx->u.intercept.ntv_table->events[events_id[j]].name;
            unsigned usr_ntv_dev =
                rocp_ctx->u.intercept.ntv_table->events[events_id[j]].ntv_dev;

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

int
get_dev_id(hsa_agent_t agent)
{
    unsigned dev_id;
    for (dev_id = 0; dev_id < agent_arr.count; ++dev_id) {
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
