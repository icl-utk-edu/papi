//-----------------------------------------------------------------------------
// @file    linux-amd-smi.c
//
// @brief PAPI component for AMD SMI (System Management Interface) on Linux.
//        Bridges PAPI to AMD GPU monitoring via libamd_smi.so.
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "amd_smi/amdsmi.h"
#include "amds.h"

// PAPI component context (per thread)
typedef struct {
    int initialized;
    int state;
    int component_id;
} amdsmi_context_t;

// PAPI component control state (per EventSet)
typedef struct {
    unsigned int *events_id;    // array of native event indices
    int num_events;             // number of events in the array
    int component_id;
    amdsmi_ctx_t amdsmi_ctx;    // handle to internal AMD SMI context for the eventset
} amdsmi_control_t;

// Lock for thread safety
extern unsigned int _amd_smi_lock;
papi_vector_t _amd_smi_vector;  // exported to PAPI core

// Forward declarations of component callbacks
static int _amd_smi_init_private(void);
static int _amd_smi_init_component(int cidx);
static int _amd_smi_init_thread(hwd_context_t *ctx);
static int _amd_smi_shutdown_component(void);
static int _amd_smi_shutdown_thread(hwd_context_t *ctx);
static int _amd_smi_init_control_state(hwd_control_state_t *ctrl);
static int _amd_smi_update_control_state(hwd_control_state_t *ctrl, NativeInfo_t *nativeInfo,
                                         int nativeCount, hwd_context_t *ctx);
static int _amd_smi_start(hwd_context_t *ctx, hwd_control_state_t *ctrl);
static int _amd_smi_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl);
static int _amd_smi_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **values, int flags);
static int _amd_smi_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *values);
static int _amd_smi_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl);
static int _amd_smi_cleanup_eventset(hwd_control_state_t *ctrl);
static int _amd_smi_ctrl(hwd_context_t *ctx, int code, _papi_int_option_t *option);
static int _amd_smi_set_domain(hwd_control_state_t *ctrl, int domain);
static int _amd_smi_ntv_enum_events(unsigned int *EventCode, int modifier);
static int _amd_smi_ntv_code_to_name(unsigned int EventCode, char *name, int len);
static int _amd_smi_ntv_code_to_descr(unsigned int EventCode, char *descr, int len);
static int _amd_smi_ntv_name_to_code(const char *name, unsigned int *EventCode);

// Helper to ensure component is initialized (calls _amd_smi_init_private on first use)
static int _amd_smi_check_n_initialize(void) {
    if (!_amd_smi_vector.cmp_info.initialized)
        return _amd_smi_init_private();
    return _amd_smi_vector.cmp_info.disabled;
}

// Initialize per-thread context (called when a thread registers this component)
static int _amd_smi_init_thread(hwd_context_t *ctx) {
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    memset(amd_ctx, 0, sizeof(*amd_ctx));
    amd_ctx->initialized = 1;
    amd_ctx->component_id = _amd_smi_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

// Initialize component (called when PAPI_library_init() or component is first used)
static int _amd_smi_init_component(int cidx) {
    // Set component index and mark not initialized until first use
    _amd_smi_vector.cmp_info.CmpIdx = cidx;
    _amd_smi_vector.cmp_info.num_native_events = -1;
    _amd_smi_vector.cmp_info.num_cntrs = -1;
    _amd_smi_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    sprintf(_amd_smi_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    _amd_smi_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    return PAPI_EDELAY_INIT;
}

// Internal one-time initialization: load AMD SMI library and build event table
static int _amd_smi_init_private(void) {
    int papi_errno = PAPI_OK;
    PAPI_lock(COMPONENT_LOCK);
    if (_amd_smi_vector.cmp_info.initialized) {
        papi_errno = _amd_smi_vector.cmp_info.disabled;
        goto fn_exit;
    }
    // Initialize AMD SMI component (load library, enumerate events)
    papi_errno = amdsmi_init();   // calls dynamic loading and event table setup
    if (papi_errno != PAPI_OK) {
        // On failure, disable component with error code
        _amd_smi_vector.cmp_info.disabled = papi_errno;
        const char *error_str;
        amdsmi_err_get_last(&error_str);  // get last error string from amdsmi
        snprintf(_amd_smi_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", error_str);
        goto fn_exit;
    }
    // Set number of native events and counters from amdsmi
    _amd_smi_vector.cmp_info.num_native_events = amdsmi_get_event_count();
    _amd_smi_vector.cmp_info.num_cntrs = _amd_smi_vector.cmp_info.num_native_events;
    _amd_smi_vector.cmp_info.available_domains = PAPI_DOM_USER;
    _amd_smi_vector.cmp_info.initialized = 1;
    _amd_smi_vector.cmp_info.disabled = PAPI_OK;
fn_exit:
    PAPI_unlock(COMPONENT_LOCK);
    return papi_errno;
}

// Initialize a new control state (EventSet)
static int _amd_smi_init_control_state(hwd_control_state_t *ctrl) {
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    memset(ctl, 0, sizeof(*ctl));
    ctl->component_id = _amd_smi_vector.cmp_info.CmpIdx;
    return PAPI_OK;
}

// Update control state with a new list of events
static int _amd_smi_update_control_state(hwd_control_state_t *ctrl, NativeInfo_t *nativeInfo,
                                         int nativeCount, hwd_context_t *ctx) {
    int papi_errno = _amd_smi_check_n_initialize();
    if (papi_errno != PAPI_OK) return papi_errno;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    amdsmi_context_t *tctx = (amdsmi_context_t *) ctx;
    if (tctx->state & AMDSMI_EVENTS_RUNNING) {
        // cannot update while running
        return PAPI_EISRUN;
    }
    // Allocate and store native event codes
    unsigned int *events = (unsigned int *) papi_calloc(nativeCount, sizeof(unsigned int));
    if (!events && nativeCount > 0) return PAPI_ENOMEM;
    for (int i = 0; i < nativeCount; ++i) {
        events[i] = nativeInfo[i].ni_event;
        nativeInfo[i].ni_position = i;
    }
    // Free old event list and assign new
    papi_free(ctl->events_id);
    ctl->events_id = events;
    ctl->num_events = nativeCount;
    return PAPI_OK;
}

// Start counting events in an EventSet
static int _amd_smi_start(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
    int papi_errno = PAPI_OK;
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    if (amd_ctx->state & AMDSMI_EVENTS_OPENED) {
        // events already opened, cannot start again without stop
        return PAPI_EMISC;
    }
    // Open and initialize the AMD SMI event context
    papi_errno = amdsmi_ctx_open(ctl->events_id, ctl->num_events, &ctl->amdsmi_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    amd_ctx->state = AMDSMI_EVENTS_OPENED;
    // Start the counters
    papi_errno = amdsmi_ctx_start(ctl->amdsmi_ctx);
    if (papi_errno != PAPI_OK) {
        // failure: cleanup
        amdsmi_ctx_close(ctl->amdsmi_ctx);
        amd_ctx->state = 0;
        return papi_errno;
    }
    amd_ctx->state |= AMDSMI_EVENTS_RUNNING;
    return PAPI_OK;
}

// Stop counting events and release resources for an EventSet
static int _amd_smi_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
    int papi_errno = PAPI_OK;
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    if (!(amd_ctx->state & AMDSMI_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    // Stop counters
    papi_errno = amdsmi_ctx_stop(ctl->amdsmi_ctx);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    // Mark not running
    amd_ctx->state &= ~AMDSMI_EVENTS_RUNNING;
    // Close and free context
    papi_errno = amdsmi_ctx_close(ctl->amdsmi_ctx);
    amd_ctx->state = 0;
    ctl->amdsmi_ctx = NULL;
    return papi_errno;
}

// Read current values from events in EventSet
static int _amd_smi_read(hwd_context_t *ctx, hwd_control_state_t *ctrl,
                         long long **values, int flags) {
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    if (!(amd_ctx->state & AMDSMI_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    return amdsmi_ctx_read(ctl->amdsmi_ctx, values);
}

// Write values to controllable events (for events that support write)
static int _amd_smi_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *values) {
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    if (!(amd_ctx->state & AMDSMI_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    return amdsmi_ctx_write(ctl->amdsmi_ctx, values);
}

// Reset events in an EventSet to initial state (zero counters, etc.)
static int _amd_smi_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl) {
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    if (!(amd_ctx->state & AMDSMI_EVENTS_RUNNING)) {
        return PAPI_EMISC;
    }
    return amdsmi_ctx_reset(ctl->amdsmi_ctx);
}

// Cleanup EventSet (after stop, free allocated event list)
static int _amd_smi_cleanup_eventset(hwd_control_state_t *ctrl) {
    amdsmi_control_t *ctl = (amdsmi_control_t *) ctrl;
    // Context should have been closed in stop. Ensure it's NULL.
    if (ctl->amdsmi_ctx != NULL) {
        return PAPI_EMISC;
    }
    papi_free(ctl->events_id);
    ctl->events_id = NULL;
    ctl->num_events = 0;
    return PAPI_OK;
}

// Shutdown thread context
static int _amd_smi_shutdown_thread(hwd_context_t *ctx) {
    amdsmi_context_t *amd_ctx = (amdsmi_context_t *) ctx;
    amd_ctx->state = 0;
    amd_ctx->initialized = 0;
    return PAPI_OK;
}

// Shutdown entire component (cleanup global resources)
static int _amd_smi_shutdown_component(void) {
    if (!_amd_smi_vector.cmp_info.initialized) {
        return PAPI_EMISC;
    }
    if (_amd_smi_vector.cmp_info.disabled != PAPI_OK) {
        // not successfully initialized
        return PAPI_EMISC;
    }
    int papi_errno = amdsmi_shutdown();
    _amd_smi_vector.cmp_info.initialized = 0;
    return papi_errno;
}

// Ioctl/ctl interface (not used, no custom options)
static int _amd_smi_ctrl(hwd_context_t *ctx, int code, _papi_int_option_t *option) {
    (void)ctx; (void)code; (void)option;
    return PAPI_OK;
}

// Set domain (User/Kernel, etc. - AMD SMI metrics are all user-space domain)
static int _amd_smi_set_domain(hwd_control_state_t *ctrl, int domain) {
    (void)ctrl;
    if (domain != PAPI_DOM_USER) {
        return PAPI_EINVAL;
    }
    return PAPI_OK;
}

// Enumerate native events
static int _amd_smi_ntv_enum_events(unsigned int *EventCode, int modifier) {
    return amdsmi_evt_enum(EventCode, modifier);
}

// Convert event code to name string
static int _amd_smi_ntv_code_to_name(unsigned int EventCode, char *name, int len) {
    return amdsmi_evt_code_to_name(EventCode, name, len);
}

// Convert event code to description string
static int _amd_smi_ntv_code_to_descr(unsigned int EventCode, char *descr, int len) {
    return amdsmi_evt_code_to_descr(EventCode, descr, len);
}

// Lookup event code by name
static int _amd_smi_ntv_name_to_code(const char *name, unsigned int *EventCode) {
    return amdsmi_evt_name_to_code(name, EventCode);
}

// Define the component's function vector table
papi_vector_t _amd_smi_vector = {
    .cmp_info = {
        .name = "amd_smi",
        .short_name = "amd_smi",
        .version = "1.0",
        .description = "AMD GPU System Management Interface via libamd_smi",
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER,
        .initialized = 0,
    },
    .size = {
        .context = sizeof(amdsmi_context_t),
        .control_state = sizeof(amdsmi_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },
    .start = _amd_smi_start,
    .stop  = _amd_smi_stop,
    .read  = _amd_smi_read,
    .write = _amd_smi_write,
    .reset = _amd_smi_reset,
    .cleanup_eventset = _amd_smi_cleanup_eventset,
    .init_component = _amd_smi_init_component,
    .init_thread = _amd_smi_init_thread,
    .init_control_state = _amd_smi_init_control_state,
    .update_control_state = _amd_smi_update_control_state,
    .ctl = _amd_smi_ctrl,
    .set_domain = _amd_smi_set_domain,
    .ntv_enum_events = _amd_smi_ntv_enum_events,
    .ntv_code_to_name = _amd_smi_ntv_code_to_name,
    .ntv_code_to_descr = _amd_smi_ntv_code_to_descr,
    .ntv_name_to_code = _amd_smi_ntv_name_to_code,
    .shutdown_thread = _amd_smi_shutdown_thread,
    .shutdown_component = _amd_smi_shutdown_component,
};
