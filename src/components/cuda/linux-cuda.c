/**
 * @file    linux-cuda.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in Spring of 2024, redesigned to add qualifier support.)
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
#include <stdint.h>

#include "papi_memory.h"
#include "cupti_dispatch.h"
#include "lcuda_debug.h"

papi_vector_t _cuda_vector;

extern cuptiu_event_table_t *global_event_names;

static int cuda_init_component(int cidx);
static int cuda_shutdown_component(void);
static int cuda_init_thread(hwd_context_t *ctx);
static int cuda_shutdown_thread(hwd_context_t *ctx);

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len);
static int cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

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
static int cuda_get_evt_count(int *count);

#define PAPI_CUDA_MPX_COUNTERS 512
#define PAPI_CUDA_MAX_COUNTERS  30

typedef struct {
    int initialized;
    int state;
    int component_id;
} cuda_context_t;

typedef struct cuda_ctl {
    int           events_count; //num_events
    uint64_t      *events_id;
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
    .ntv_code_to_info = cuda_ntv_code_to_info,

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
    //cuptid_event_table_destroy(&global_event_names);

    if (!_cuda_vector.cmp_info.initialized ||
    _cuda_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_OK;
    }

    _cuda_vector.cmp_info.initialized = 0;

    return cuptid_shutdown();
}

static int cuda_init_private(void)
{
    int papi_errno = PAPI_OK, count = 0;
    const char *disabled_reason;
    COMPDBG("Entering.\n");

    //papi_errno = cuptid_event_table_create(&global_event_names);
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

    papi_errno = cuda_get_evt_count(&count);
    _cuda_vector.cmp_info.num_native_events = count;

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
    SUBDBG("ENTER: event_code: %u, modifier: %d\n", *event_code, modifier);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    
    uint64_t code = *(uint64_t *) event_code;
    papi_errno = cuptid_evt_enum(&code, modifier);
    *event_code = (unsigned int) code;
    
fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
    
    uint64_t code;
    papi_errno = cuptid_evt_name_to_code(name, &code);
    *event_code = (unsigned int) code;

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
    fn_fail:
        goto fn_exit;
}

static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = cuptid_evt_code_to_name((uint64_t) event_code, name, len);

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
    fn_fail:
        goto fn_exit;
}

static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    SUBDBG("ENTER: event_code: %u, descr: %p, len: %d\n", event_code, descr, len);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptid_evt_code_to_descr((uint64_t) event_code, descr, len);

fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    SUBDBG("ENTER: event_code: %u, info: %p\n", event_code, info);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptid_evt_code_to_info((uint64_t) event_code, info); 

fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
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

static int update_native_events(cuda_ctl_t *, NativeInfo_t *, int);


int cuda_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info,
                              int ntv_count,
                              hwd_context_t *ctx __attribute__((unused)))
{    SUBDBG("ENTER: ctl: %p, ntv_info: %p, ntv_count: %d, ctx: %p\n", ctl, ntv_info, ntv_count, ctx);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cuda_ctl_t *cuda_ctl = (cuda_ctl_t *) ctl;

    papi_errno = update_native_events(cuda_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

/*
static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, __attribute__((unused)) hwd_context_t *ctx
) {
    COMPDBG("Entering with events_count %d.\n", ntv_count);
    int i, papi_errno;
    papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        printf("initialized failed: %d\n", papi_errno);
        return papi_errno;
    }
    if (ntv_count == 0) {
        return PAPI_OK;
    }

    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    papi_errno = cuptid_thread_info_create(&(control->info));
    if (papi_errno != PAPI_OK) {
        printf("cuptid_thread_info_create: %d\n", papi_errno);
        goto fn_exit;
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
        printf("cuptid_event_table_select: %d\n", papi_errno);
        goto fn_exit;
    }
    papi_errno = cuptid_control_create(select_names, control->info, &tmp_context);
    if (papi_errno != PAPI_OK) {
        cuptid_control_destroy(&tmp_context);
         printf("cuptid_control_create: %d\n", papi_errno);
        goto fn_exit;
    }
    papi_errno = cuptid_control_destroy(&tmp_context);

fn_exit:
    cuptid_event_table_destroy(&select_names);
    return papi_errno;
}
*/

struct event_map_item {
    int event_id;
    int frontend_idx;
};

static int
compare(const void *a, const void *b)
{
    struct event_map_item *A = (struct event_map_item *) a;
    struct event_map_item *B = (struct event_map_item *) b;
    return  A->event_id - B->event_id;
}

int
update_native_events(cuda_ctl_t *ctl, NativeInfo_t *ntv_info,
                     int ntv_count)
{
    int papi_errno = PAPI_OK;
    struct event_map_item sorted_events[PAPI_CUDA_MAX_COUNTERS];

    if (ntv_count != ctl->events_count) {
        ctl->events_id = papi_realloc(ctl->events_id,
                                      ntv_count * sizeof(*ctl->events_id));
        if (ctl->events_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        ctl->events_count = ntv_count;
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        sorted_events[i].event_id = ntv_info[i].ni_event;
        sorted_events[i].frontend_idx = i;
    }

    qsort(sorted_events, ntv_count, sizeof(struct event_map_item), compare);

    for (i = 0; i < ntv_count; ++i) {
        ctl->events_id[i] = sorted_events[i].event_id;
        ntv_info[sorted_events[i].frontend_idx].ni_position = i;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->events_count = 0;
    goto fn_exit;
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

static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno, i;

    //cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    //cuda_ctl_t *cuda_ctl = (cuda_ctl_t *) ctl;
    cuda_ctl_t *control = (cuda_ctl_t *) ctl;
    for (i = 0; i < control->events_count; i++) {
        control->values[i] = 0;
    }
    //ntv_event_table_t select_names;
    //papi_errno = cuptid_event_table_select_by_idx(global_event_names, control->events_count, control->events_id, &select_names);
    //if (papi_errno != PAPI_OK) {
    //    goto fn_exit;
    //}

    //papi_errno = cupti_control_create(cuda_ctl->events_id, cuda_ctl->events_count, &cuda_ctl->cuda_ctx)

    //papi_errno = cuptid_control_create(select_names, control->info, &(control->cupti_ctl));
    //if (papi_errno != PAPI_OK) {
    //    printf("failed to create\n");
    //    goto fn_exit;
    //}
    
    //papi_errno = cuptid_control_start( control->cupti_ctl );
    //printf("papi_errno is: %d\n", papi_errno);

    //cuda_ctx->state |= 0x02;

fn_exit:
    //cuptid_event_table_destroy(&select_names);
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

static int cuda_get_evt_count(int *count)
{
    uint64_t event_code = 0;
    int papi_errno;

    if (cuptid_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        papi_errno = cuptid_get_num_qualified_evts(count, event_code);
        if (papi_errno != PAPI_OK)
            return papi_errno;
         
    }
    while (cuptid_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        papi_errno = cuptid_get_num_qualified_evts(count, event_code);
        if (papi_errno != PAPI_OK)
            return papi_errno;
    }

    return PAPI_OK;
}
