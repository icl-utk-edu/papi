/**
 * @file    linux-cuda.c
 *
 * @author  Anustuv Pal   anustuv@icl.utk.edu (updated in 2023, redesigned with multi-threading support.)
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 08/2019, to make counters accumulate.)
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 2018, to use batch reads and support nvlink metrics.
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This file implements a PAPI component that enables PAPI-C to access
 *  hardware monitoring counters for NVIDIA GPU devices through the CuPTI library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <papi.h>
#include <papi_internal.h>
#include <papi_vector.h>

#include <string.h>

#include "cupti_dispatch.h"
#include "lcuda_debug.h"

papi_vector_t _cuda_vector;

ntv_event_table_t global_event_names;

static int cuda_init_component(int cidx);
static int cuda_shutdown_component(void);
static int cuda_init_thread(hwd_context_t *ctx);
static int cuda_shutdown_thread(hwd_context_t *ctx);

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len);

static int cuda_init_control_state(hwd_control_state_t *ctl);
static int cuda_set_domain(hwd_control_state_t * ctrl, int domain);
static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, hwd_context_t *ctx);

static int cuda_cleanup_eventset(hwd_control_state_t *ctl);
static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int cuda_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_init_private(void);

#define PAPI_CUDA_MPX_COUNTERS 512
#define PAPI_CUDA_MAX_COUNTERS  30

typedef struct cuda_ctl {
    int           events_count;
    int           events_id[PAPI_CUDA_MAX_COUNTERS];
    long long     values[PAPI_CUDA_MAX_COUNTERS];
    cuptid_info_t info;
    cuptid_ctl_t  cupti_ctl;
} cuda_ctl_t;

papi_vector_t _cuda_vector = {
    .cmp_info = {
        .name = "cuda",
        .short_name = "cuda",
        .version = "0.1",
        .description = "CUDA profiling via NVIDIA CuPTI interfaces",
        .num_mpx_cntrs = PAPI_CUDA_MPX_COUNTERS,
        .num_cntrs = PAPI_CUDA_MAX_COUNTERS,
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        /* component specific cmp_info initializations */
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .initialized = 0,
    },
    .size = {
        .context = 0,
        .control_state = sizeof(cuda_ctl_t),
    },
    .init_component = cuda_init_component,
    .shutdown_component = cuda_shutdown_component,

    .init_thread = cuda_init_thread,
    .shutdown_thread = cuda_shutdown_thread,

    .ntv_enum_events = cuda_ntv_enum_events,
    .ntv_code_to_name = cuda_ntv_code_to_name,
    .ntv_name_to_code = cuda_ntv_name_to_code,
    .ntv_code_to_descr = cuda_ntv_code_to_descr,

    .init_control_state = cuda_init_control_state,
    .set_domain = cuda_set_domain,
    .update_control_state = cuda_update_control_state,
    .cleanup_eventset = cuda_cleanup_eventset,

    .start = cuda_start,
    .stop = cuda_stop,
    .read = cuda_read,
    .reset = cuda_reset,
};

static int cuda_init_component(int cidx)
{
    COMPDBG("Entering with component idx: %d\n", cidx);

    _cuda_vector.cmp_info.CmpIdx = cidx;
    _cuda_vector.cmp_info.num_native_events = -1;
    _cuda_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    _cuda_vector.cmp_info.initialized = 1;
    _cuda_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    sprintf(_cuda_vector.cmp_info.disabled_reason,
        "Not initialized. Access component events to initialize it.");
    return PAPI_EDELAY_INIT;
}

static int cuda_shutdown_component(void)
{
    COMPDBG("Entering.\n");
    cuptid_event_table_destroy(&global_event_names);

    if (!_cuda_vector.cmp_info.initialized ||
    _cuda_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_OK;
    }

    _cuda_vector.cmp_info.initialized = 0;

    return cuptid_shutdown();
}

static int cuda_init_private(void)
{
    int papi_errno = PAPI_OK;
    const char *disabled_reason;
    COMPDBG("Entering.\n");

    papi_errno = cuptid_event_table_create(&global_event_names);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = cuptid_init();
    if (papi_errno != PAPI_OK) {
        cuptid_disabled_reason_get(&disabled_reason);
        sprintf(_cuda_vector.cmp_info.disabled_reason, disabled_reason);
        _cuda_vector.cmp_info.disabled = papi_errno;
        goto fn_exit;
    }

    _cuda_vector.cmp_info.disabled = PAPI_OK;
    strcpy(_cuda_vector.cmp_info.disabled_reason, "");

fn_exit:
    return papi_errno;
}

static int check_n_initialize(void)
{
    _papi_hwi_lock(COMPONENT_LOCK);
    int papi_errno = PAPI_OK;
    if (_cuda_vector.cmp_info.initialized
        && _cuda_vector.cmp_info.disabled == PAPI_EDELAY_INIT
    ) {
        papi_errno = cuda_init_private();
    }

    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
}

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    _papi_hwi_lock(COMPONENT_LOCK);
    LOCKDBG("Locked COMPONENT_LOCK to enumerate all events.\n");
    papi_errno = cuptid_event_enum(global_event_names);
    _papi_hwi_unlock(COMPONENT_LOCK);
    LOCKDBG("Unlocked COMPONENT_LOCK.\n");
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    _cuda_vector.cmp_info.num_native_events = global_event_names->count;
    switch (modifier) {
        case PAPI_ENUM_FIRST:
            *event_code = 0;
            papi_errno = PAPI_OK;
            break;
        case PAPI_ENUM_EVENTS:
            if (global_event_names->count == 0) {
                papi_errno = PAPI_ENOEVNT;
            } else if (*event_code < global_event_names->count - 1) {
                *event_code = *event_code + 1;
                papi_errno = PAPI_OK;
            } else {
                papi_errno = PAPI_ENOEVNT;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }
fn_exit:
    return papi_errno;
}

static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    ntv_event_t evt_rec;
    papi_errno = cuptid_event_table_find_name(global_event_names, name, &evt_rec);
    if (papi_errno == PAPI_OK) {
        *event_code = evt_rec->evt_code;
    }
    else {
        _papi_hwi_lock(COMPONENT_LOCK);
        *event_code = global_event_names->count;
        papi_errno = cuptid_event_table_insert_record(global_event_names, name, global_event_names->count, 0);
        _papi_hwi_unlock(COMPONENT_LOCK);
    }
fn_exit:
    return papi_errno;
}

static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    ntv_event_t evt_rec;
    papi_errno = cuptid_event_table_get_item(global_event_names, event_code, &evt_rec);
    if (papi_errno != PAPI_OK) {
        return PAPI_ENOEVNT;
    }
    strncpy(name, evt_rec->name, len);
    return PAPI_OK;
}

static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int __attribute__((unused)) len)
{
    char evt_name[PAPI_2MAX_STR_LEN];
    int papi_errno;
    papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    _papi_hwi_lock(COMPONENT_LOCK);
    papi_errno = cuptid_event_enum(global_event_names);
    _papi_hwi_unlock(COMPONENT_LOCK);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = cuda_ntv_code_to_name(event_code, evt_name, PAPI_2MAX_STR_LEN);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = cuptid_event_name_to_descr(evt_name, descr);
fn_exit:
    return papi_errno;
}

static int cuda_init_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int cuda_shutdown_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int cuda_init_control_state(hwd_control_state_t __attribute__((unused)) *ctl)
{
    COMPDBG("Entering.\n");
    return PAPI_OK;
}

static int cuda_set_domain(hwd_control_state_t __attribute__((unused)) *ctrl, int domain)
{
    COMPDBG("Entering\n");
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain))
        return (PAPI_OK);
    else
        return (PAPI_EINVAL);
}

static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, __attribute__((unused)) hwd_context_t *ctx
) {
    COMPDBG("Entering with events_count %d.\n", ntv_count);
    int i, papi_errno;
    papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if (ntv_count == 0) {
        return PAPI_OK;
    }

    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    if (control->info == NULL) {
        papi_errno = cuptid_thread_info_create(&(control->info));
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
    control->events_count = ntv_count;

    if (ntv_count > PAPI_CUDA_MAX_COUNTERS) {
        ERRDBG("Too many events added.\n");
        papi_errno = PAPI_ECMP;
        goto fn_exit;
    }
    for (i=0; i<ntv_count; i++) {
        control->events_id[i] = ntv_info[i].ni_event;
        ntv_info[i].ni_position = i;
    }

    void *tmp_context = NULL;
    ntv_event_table_t select_names;
    papi_errno = cuptid_event_table_select_by_idx(global_event_names, control->events_count, control->events_id, &select_names);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = cuptid_control_create(select_names, control->info, &tmp_context);
    if (papi_errno != PAPI_OK) {
        cuptid_control_destroy(&tmp_context);
        goto fn_exit;
    }
    papi_errno = cuptid_control_destroy(&tmp_context);

fn_exit:
    cuptid_event_table_destroy(&select_names);
    return papi_errno;
}

static int cuda_cleanup_eventset(hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    int papi_errno = PAPI_OK;
    if (control->cupti_ctl) {
        papi_errno += cuptid_control_destroy(&(control->cupti_ctl));
    }
    if (control->info) {
        papi_errno += cuptid_thread_info_destroy(&(control->info));
    }
    if (papi_errno != PAPI_OK) {
        return PAPI_ECMP;
    }
    return PAPI_OK;
}

static int cuda_start(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno, i;

    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    for (i = 0; i < control->events_count; i++) {
        control->values[i] = 0;
    }
    ntv_event_table_t select_names;
    papi_errno = cuptid_event_table_select_by_idx(global_event_names, control->events_count, control->events_id, &select_names);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = cuptid_control_create(select_names, control->info, &(control->cupti_ctl));
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    papi_errno = cuptid_control_start( control->cupti_ctl );

fn_exit:
    cuptid_event_table_destroy(&select_names);
    return papi_errno;
}

int cuda_stop(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    int papi_errno;
    papi_errno = cuptid_control_stop( control->cupti_ctl );
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    papi_errno = cuptid_control_destroy( &(control->cupti_ctl) );
fn_exit:
    return papi_errno;
}

static int cuda_read(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl, long long **val, int __attribute__((unused)) flags)
{
    COMPDBG("Entering.\n");
    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    int papi_errno;

    papi_errno = cuptid_control_read( control->cupti_ctl, (long long *) &(control->values) );
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    *val = control->values;

fn_exit:
    return papi_errno;
}

static int cuda_reset(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    int i;
    for (i = 0; i < control->events_count; i++) {
        control->values[i] = 0;
    }
    return cuptid_control_reset( control->cupti_ctl );
}
