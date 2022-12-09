/**
 * @file    rocm.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief This implements a PAPI component that enables PAPI-C to
 *  access hardware monitoring counters for AMD ROCM GPU devices
 *  through the ROC-profiler library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include "common.h"
#include "papi_vector.h"
#include "extras.h"
#include "rocp.h"
#include "htable.h"

/* Init and finalize */
static int rocm_init_component(int cid);
static int rocm_init_thread(hwd_context_t *ctx);
static int rocm_init_control_state(hwd_control_state_t *ctl);
static int rocm_init_private(void);
static int rocm_shutdown_component(void);
static int rocm_shutdown_thread(hwd_context_t *ctx);
static int rocm_cleanup_eventset(hwd_control_state_t *ctl);
static void rocm_dispatch_timer(int n, hwd_siginfo_t *info, void *uc);

/* set and update component state */
static int rocm_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, hwd_context_t *ctx);
static int rocm_set_domain(hwd_control_state_t *ctl, int domain);
static int rocm_ctrl(hwd_context_t *ctx, int code, _papi_int_option_t *option);

/* start and stop monitoring of hardware counters */
static int rocm_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocm_read(hwd_context_t *ctx, hwd_control_state_t *ctl,
                     long long **val, int flags);
static int rocm_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int rocm_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);

/* event conversion utilities */
static int rocm_ntv_enum_events(unsigned int *event_code, int modifier);
static int rocm_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int rocm_ntv_name_to_code(const char *name, unsigned int *event_code);
static int rocm_ntv_code_to_descr(unsigned int event_code, char *descr,
                                  int len);

static int insert_ntv_events_to_htable(void);

extern unsigned rocm_prof_mode;

/* table containing all ROCm events */
ntv_event_table_t ntv_table;
void *htable;


typedef struct {
    int initialized;
    int state;
    int component_id;
    ntv_event_table_t *ntv_table;
} rocm_context_t;

typedef struct {
    int num_events;
    unsigned int domain;
    unsigned int granularity;
    unsigned int overflow;
    unsigned int overflow_signal;
    unsigned int attached;
    int component_id;
    unsigned int *events_id;
    rocp_ctx_t rocp_ctx;
} rocm_control_t;

papi_vector_t _rocm_vector = {
    .cmp_info = {
        .name = "rocm",
        .short_name = "rocm",
        .version = "2.0",
        .description = "GPU events and metrics via AMD ROCm-PL API",
        .num_mpx_cntrs = PAPI_ROCM_MAX_COUNTERS,
        .num_cntrs = PAPI_ROCM_MAX_COUNTERS,
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .initialized = 0,
    },

    .size = {
        .context = sizeof(rocm_context_t),
        .control_state = sizeof(rocm_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = rocm_init_component,
    .init_thread = rocm_init_thread,
    .init_control_state = rocm_init_control_state,
    .shutdown_component = rocm_shutdown_component,
    .shutdown_thread = rocm_shutdown_thread,
    .cleanup_eventset = rocm_cleanup_eventset,
    .dispatch_timer = rocm_dispatch_timer,

    .update_control_state = rocm_update_control_state,
    .set_domain = rocm_set_domain,
    .ctl = rocm_ctrl,

    .start = rocm_start,
    .stop = rocm_stop,
    .read = rocm_read,
    .reset = rocm_reset,

    .ntv_enum_events = rocm_ntv_enum_events,
    .ntv_code_to_name = rocm_ntv_code_to_name,
    .ntv_name_to_code = rocm_ntv_name_to_code,
    .ntv_code_to_descr = rocm_ntv_code_to_descr,
};

static void check_n_initialize(void);

int
rocm_init_component(int cid)
{
    _rocm_vector.cmp_info.CmpIdx = cid;
    _rocm_vector.cmp_info.num_native_events = -1;
    _rocm_vector.cmp_info.num_cntrs = -1;
    _rocm_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cid;

    const char *err_string;
    int papi_errno = rocp_init_environment(&err_string);
    if (papi_errno != PAPI_OK) {
        _rocm_vector.cmp_info.initialized = 1;
        _rocm_vector.cmp_info.disabled = papi_errno;
        rocp_err_get_last(&err_string);
        int expect = snprintf(_rocm_vector.cmp_info.disabled_reason,
                              PAPI_MAX_STR_LEN, "%s", err_string);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("disabled_reason truncated");
        }
        return papi_errno;
    }

    htable_init(&htable);

    sprintf(_rocm_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    _rocm_vector.cmp_info.disabled = PAPI_EDELAY_INIT;

    return PAPI_EDELAY_INIT;
}

int
rocm_init_thread(hwd_context_t *ctx)
{
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    memset(rocm_ctx, 0, sizeof(*rocm_ctx));
    rocm_ctx->initialized = 1;
    rocm_ctx->component_id = _rocm_vector.cmp_info.CmpIdx;
    rocm_ctx->ntv_table = &ntv_table;
    return PAPI_OK;
}

int
rocm_init_control_state(hwd_control_state_t *ctl __attribute__((unused)))
{
    check_n_initialize();
    return PAPI_OK;
}

int
rocm_init_private(void)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(COMPONENT_LOCK);

    if (_rocm_vector.cmp_info.initialized) {
        papi_errno = _rocm_vector.cmp_info.disabled;
        goto fn_exit;
    }

    const char *err_string;
    papi_errno = rocp_init(&ntv_table, &err_string);
    if (papi_errno != PAPI_OK) {
        _rocm_vector.cmp_info.disabled = papi_errno;
        rocp_err_get_last(&err_string);
        int expect = snprintf(_rocm_vector.cmp_info.disabled_reason,
                              PAPI_MAX_STR_LEN, "%s", err_string);
        if (expect > PAPI_MAX_STR_LEN) {
            SUBDBG("disabled_reason truncated");
        }

        goto fn_fail;
    }

    insert_ntv_events_to_htable();

    _rocm_vector.cmp_info.num_native_events = ntv_table.count;
    _rocm_vector.cmp_info.num_cntrs = ntv_table.count;

  fn_exit:
    _rocm_vector.cmp_info.initialized = 1;
    _rocm_vector.cmp_info.disabled = papi_errno;
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
rocm_shutdown_component(void)
{
    if (!_rocm_vector.cmp_info.initialized) {
        return PAPI_OK;
    }

    if (_rocm_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_OK;
    }

    int papi_errno = rocp_shutdown(&ntv_table);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    htable_shutdown(htable);

  fn_exit:
    _rocm_vector.cmp_info.initialized = 0;
    return papi_errno;
}

int
rocm_shutdown_thread(hwd_context_t *ctx)
{
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_ctx->initialized = 0;
    rocm_ctx->state = 0;
    return PAPI_OK;
}

int
rocm_cleanup_eventset(hwd_control_state_t *ctl)
{
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocp_ctx != NULL) {
        SUBDBG("Cannot cleanup an eventset that is running.");
        return PAPI_ECMP;
    }

    papi_free(rocm_ctl->events_id);
    rocm_ctl->events_id = NULL;
    rocm_ctl->num_events = 0;

    return PAPI_OK;
}

static int
counter_sampling_compatible_with_prof_mode(void)
{
    return (rocm_prof_mode == ROCM_PROFILE_SAMPLING_MODE);
}

void
rocm_dispatch_timer(int n __attribute__((unused)), hwd_siginfo_t *info, void *uc)
{
    _papi_hwi_context_t hw_context;
    vptr_t address;
    EventSetInfo_t *ESI;
    ThreadInfo_t *thread;
    int cidx;

    if (!counter_sampling_compatible_with_prof_mode()) {
        SUBDBG("Counter sampling is not compatible with intercept mode");
        return;
    }

    _papi_hwi_lock(_rocm_lock);

    cidx = _rocm_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread(0);

    if (thread == NULL) {
        SUBDBG("thread == NULL in user_signal_handler!");
        goto fn_exit;
    }

    ESI = thread->running_eventset[cidx];
    rocm_control_t *rocm_ctl = (rocm_control_t *) ESI->ctl_state;

    if (rocm_ctl->rocp_ctx == NULL) {
        SUBDBG("ESI == NULL in user_signal_handler!");
        goto fn_exit;
    }

    hw_context.si = info;
    hw_context.ucontext = (hwd_ucontext_t *) uc;

    address = GET_OVERFLOW_ADDRESS(hw_context);

    _papi_hwi_dispatch_overflow_signal((void *)&hw_context, address, NULL, 0, 0,
                                       &thread, cidx);

  fn_exit:
    _papi_hwi_unlock(_rocm_lock);
    return;
}

static int update_native_events(rocm_control_t *, NativeInfo_t *, int,
                                rocm_context_t *);
static int try_open_events(rocm_control_t *);

int
rocm_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info,
                          int ntv_count, hwd_context_t *ctx)
{
    check_n_initialize();

    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;

    if (rocm_ctl->rocp_ctx != NULL) {
        SUBDBG("Cannot update events in an eventset that has been already "
               "started.");
        return PAPI_ECMP;
    }

    int papi_errno = update_native_events(rocm_ctl, ntv_info, ntv_count,
                                          rocm_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return try_open_events(rocm_ctl);
}

int
update_native_events(rocm_control_t *ctl, NativeInfo_t *ntv_info,
                     int ntv_count, rocm_context_t *ctx)
{
    int papi_errno = PAPI_OK;
    int i, j;
    unsigned int *events_id = NULL;
    int num_events = 0;

    for (i = 0; i < ntv_count; ++i) {
        unsigned int ntv_id = (unsigned) ntv_info[i].ni_event;
        const char *ntv_name = ctx->ntv_table->events[ntv_id].name;
        unsigned ntv_dev = ctx->ntv_table->events[ntv_id].ntv_dev;

        for (j = 0; j < num_events; ++j) {
            unsigned int ctl_ntv_id = events_id[j];
            char *ctl_ntv_name = ctx->ntv_table->events[ctl_ntv_id].name;
            unsigned int ctl_ntv_dev =
                ctx->ntv_table->events[ctl_ntv_id].ntv_dev;

            if (strcmp(ctl_ntv_name, ntv_name) == 0 &&
                ctl_ntv_dev == ntv_dev) {
                /* Do not add events that have the same name and
                 * device as they refer to the same native rocm
                 * event. */
                SUBDBG("[ROCP] Event already in eventset.");
                break;
            }
        }
        if (j == num_events) {
            ntv_info[i].ni_position = j;

            events_id = papi_realloc(events_id, ++num_events * sizeof(*events_id));
            if (events_id == NULL) {
                SUBDBG("Cannot allocate memory for control events.");
                goto fn_fail;
            }

            events_id[j] = ntv_id;
        }
    }

    if (ctl->events_id) {
        papi_free(ctl->events_id);
    }
    ctl->events_id = events_id;
    ctl->num_events = num_events;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (events_id) {
        papi_free(events_id);
    }
    ctl->num_events = 0;
    papi_errno = PAPI_ENOMEM;
    goto fn_exit;
}

int
try_open_events(rocm_control_t *rocm_ctl)
{
    int papi_errno = PAPI_OK;
    rocp_ctx_t rocp_ctx;

    if (rocm_prof_mode != ROCM_PROFILE_SAMPLING_MODE) {
        /* Do not try open for intercept mode */
        return papi_errno;
    }

    if (rocm_ctl->num_events <= 0) {
        return PAPI_OK;
    }

    papi_errno = rocp_ctx_open_v2(rocm_ctl->events_id, rocm_ctl->num_events,
                                  &rocp_ctx);
    if (papi_errno != PAPI_OK) {
        rocm_cleanup_eventset(rocm_ctl);
        return papi_errno;
    }

    return rocp_ctx_close(rocp_ctx);
}

int
rocm_set_domain(hwd_control_state_t *ctl __attribute__((unused)),
                int domain __attribute__((unused)))
{
    return PAPI_OK;
}

int
rocm_ctrl(hwd_context_t *ctx __attribute__((unused)),
          int code __attribute__((unused)),
          _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

int
rocm_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctx->state & ROCM_EVENTS_OPENED) {
        SUBDBG("Error! Cannot PAPI_start more than one eventset at a time for every component.");
        return PAPI_ECNFLCT;
    }

    papi_errno = rocp_ctx_open_v2(rocm_ctl->events_id, rocm_ctl->num_events,
                                  &rocm_ctl->rocp_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocm_ctx->state = ROCM_EVENTS_OPENED;

    papi_errno = rocp_ctx_start(rocm_ctl->rocp_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    rocm_ctx->state |= ROCM_EVENTS_RUNNING;

  fn_exit:
    return papi_errno;
  fn_fail:
    rocp_ctx_close(rocm_ctl->rocp_ctx);
    rocm_ctx->state = 0;
    goto fn_exit;
}

int
rocm_read(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl,
          long long **val, int flags __attribute__((unused)))
{
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocp_ctx == NULL) {
        SUBDBG("Error! Cannot PAPI_read counters for an eventset that has not been PAPI_start'ed.");
        return PAPI_EMISC;
    }

    return rocp_ctx_read(rocm_ctl->rocp_ctx, val);
}

int
rocm_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;
    rocm_context_t *rocm_ctx = (rocm_context_t *) ctx;
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (!(rocm_ctx->state & ROCM_EVENTS_OPENED)) {
        SUBDBG("Error! Cannot PAPI_stop counters for an eventset that has not been PAPI_start'ed.");
        return PAPI_EMISC;
    }

    papi_errno = rocp_ctx_stop(rocm_ctl->rocp_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    rocm_ctx->state &= ~ROCM_EVENTS_RUNNING;

    papi_errno = rocp_ctx_close(rocm_ctl->rocp_ctx);

    rocm_ctx->state = 0;
    rocm_ctl->rocp_ctx = NULL;

    return papi_errno;
}

int
rocm_reset(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl)
{
    rocm_control_t *rocm_ctl = (rocm_control_t *) ctl;

    if (rocm_ctl->rocp_ctx == NULL) {
        SUBDBG("Cannot reset counters for an eventset that has not been started.");
        return PAPI_EMISC;
    }

    return rocp_ctx_reset(rocm_ctl->rocp_ctx);
}

int
rocm_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int papi_errno = PAPI_OK;

    check_n_initialize();

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code < (unsigned int) ntv_table.count - 1) {
                ++(*event_code);
            } else {
                papi_errno = PAPI_ENOEVNT;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

static void get_ntv_event_name(unsigned int event_code, char *name, int len);

int
rocm_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = PAPI_OK;

    check_n_initialize();

    if (event_code >= (unsigned int) ntv_table.count) {
        return PAPI_EINVAL;
    }

    get_ntv_event_name(event_code, name, len);

    return papi_errno;
}

int
rocm_ntv_name_to_code(const char *name, unsigned int *code)
{
    int papi_errno = PAPI_OK;
    int htable_errno;

    check_n_initialize();

    ntv_event_t *event;
    htable_errno = htable_find(htable, name, (void **) &event);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = (htable_errno == HTABLE_ENOVAL) ?
            PAPI_ENOEVNT : PAPI_ECMP;
        goto fn_exit;
    }
    *code = event->ntv_id;

  fn_exit:
    return papi_errno;
}

int
rocm_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno = PAPI_OK;

    check_n_initialize();

    if (event_code >= (unsigned int) ntv_table.count) {
        return PAPI_EINVAL;
    }

    strncpy(descr, ntv_table.events[event_code].descr, len);

    return papi_errno;
}

void
check_n_initialize(void)
{
    if (!_rocm_vector.cmp_info.initialized) {
        rocm_init_private();
    }
}

int
insert_ntv_events_to_htable(void)
{
    int papi_errno = PAPI_OK;
    int htable_errno;
    unsigned int event_code;

    for (event_code = 0; event_code < (unsigned int) ntv_table.count;
         ++event_code) {
        char key[PAPI_2MAX_STR_LEN] = { 0 };

        get_ntv_event_name(event_code, key, PAPI_2MAX_STR_LEN);

        htable_errno = htable_insert(htable, key, &ntv_table.events[event_code]);
        if (htable_errno != HTABLE_SUCCESS) {
            papi_errno = PAPI_ECMP;
            break;
        }
    }

    return papi_errno;
}

void
get_ntv_event_name(unsigned int event_code, char *name, int len)
{
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
}
