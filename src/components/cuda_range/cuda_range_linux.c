// C STL headers.
#include <stdint.h>
#include <string.h>

// Internal headers.
#include "cuda_range_profiling.h"
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#define CUDA_RANGE_EVENTS_STOPPED (0x1)
#define CUDA_RANGE_EVENTS_RUNNING (0x2)

unsigned int _cuda_range_lock;

// Initialization and finalize.
static int cuda_range_init_component(int cid);
static int cuda_range_init_thread(hwd_context_t *ctx);
static int cuda_range_init_control_state(hwd_control_state_t *ctl);
static int cuda_range_init_private(void);
static int cuda_range_shutdown_component(void);
static int cuda_range_shutdown_thread(hwd_context_t *ctx);
static int cuda_range_cleanup_eventset(hwd_control_state_t *ctl);

//  Set and update component state.
static int cuda_range_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx);
static int cuda_range_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_range_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int cuda_range_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_range_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);

// Event conversion.
static int cuda_range_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_range_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_range_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_range_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);
static int cuda_range_ntv_code_to_descr(unsigned int event_code, char *descr, int len);

static int cuda_range_set_domain(hwd_control_state_t *ctl, int domain);
static int cuda_range_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option);

typedef struct {
    int initialized;
    int state;
    int component_id;
} cuda_range_context_t;

typedef struct {
    uint32_t *event_codes;
    int num_events;
    cuda_range_ctx_t cuda_range_ctx;
} cuda_range_control_t;

papi_vector_t _cuda_range_vector = {
    .cmp_info = {
        .name = "cuda_range",
        .short_name = "cuda_range",
        .version = "1.0",
        .description = "Profiling of NVIDIA GPU's via CUPTI Profiler Host and Cupti Range Profiling API's.",
        .initialized = 0,
    },

    .size = {
        .context = sizeof(cuda_range_context_t),
        .control_state = sizeof(cuda_range_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },

    .init_component = cuda_range_init_component,
    .init_thread = cuda_range_init_thread,
    .init_control_state = cuda_range_init_control_state,
    .shutdown_component = cuda_range_shutdown_component,
    .shutdown_thread = cuda_range_shutdown_thread,
    .cleanup_eventset = cuda_range_cleanup_eventset,

    .update_control_state = cuda_range_update_control_state,
    .start = cuda_range_start,
    .stop = cuda_range_stop,
    .read = cuda_range_read,
    .reset = cuda_range_reset,

    .ntv_enum_events = cuda_range_ntv_enum_events,
    .ntv_code_to_name = cuda_range_ntv_code_to_name,
    .ntv_name_to_code = cuda_range_ntv_name_to_code,
    .ntv_code_to_descr = cuda_range_ntv_code_to_descr,
    .ntv_code_to_info = cuda_range_ntv_code_to_info,

    .set_domain = cuda_range_set_domain,
    .ctl = cuda_range_ctl,
};

static int check_n_initialize(void);

int cuda_range_init_component(int cid)
{
    _cuda_range_vector.cmp_info.CmpIdx = cid;
    _cuda_range_vector.cmp_info.num_native_events = -1;
    _cuda_range_vector.cmp_info.num_cntrs = -1;
    _cuda_range_vector.cmp_info.num_mpx_cntrs = -1;
    _cuda_range_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cid;

    _cuda_range_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    int strLen = snprintf(_cuda_range_vector.cmp_info.disabled_reason, sizeof(_cuda_range_vector.cmp_info.disabled_reason),
                          "%s", "Not initialized. Access component events to initialize it.");
    if (strLen < 0 || (size_t) strLen >= sizeof(_cuda_range_vector.cmp_info.disabled_reason)) {
        SUBDBG("Failed to fully write disabled reason into buffer. Proceeding.\n");
    }  

    return PAPI_EDELAY_INIT;
}

int cuda_range_init_thread(hwd_context_t *ctx)
{
    cuda_range_context_t *cuda_range_ctx = (cuda_range_context_t *) ctx;
    memset(cuda_range_ctx, 0, sizeof(*cuda_range_ctx));
    cuda_range_ctx->initialized = 1;
    cuda_range_ctx->component_id = _cuda_range_vector.cmp_info.CmpIdx;

    return PAPI_OK;
}

int cuda_range_init_control_state(hwd_control_state_t *ctl __attribute__((unused)))
{
    return check_n_initialize();
}

static int evt_get_count(int *count)
{
    unsigned int event_code = 0;

    if (cuda_range_event_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (cuda_range_event_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

int cuda_range_init_private(void)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(COMPONENT_LOCK);

    if (_cuda_range_vector.cmp_info.initialized) {
        papi_errno = _cuda_range_vector.cmp_info.disabled;
        goto fn_exit;
    }

    int strLen;
    papi_errno = initialize_cuda_range_component();
    if (papi_errno != PAPI_OK) {
        _cuda_range_vector.cmp_info.disabled = papi_errno;
        const char *err_string = cuda_range_get_last_err_msg();
        strLen = snprintf(_cuda_range_vector.cmp_info.disabled_reason, sizeof(_cuda_range_vector.cmp_info.disabled_reason),
                              "%s", err_string);
        if (strLen < 0 || (size_t) strLen >= sizeof(_cuda_range_vector.cmp_info.disabled_reason)){
            SUBDBG("Failed to fully write the cuda_range disabled reason. Proceeding.\n");
        }

        goto fn_fail;
    }

    int number_of_native_events = 0;
    papi_errno = evt_get_count(&number_of_native_events);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    _cuda_range_vector.cmp_info.num_native_events = number_of_native_events;

    int number_of_counters = 0;
    papi_errno = get_the_maximum_number_of_hardware_metrics_per_device(&number_of_counters);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    _cuda_range_vector.cmp_info.num_cntrs = number_of_counters;
    _cuda_range_vector.cmp_info.num_mpx_cntrs = number_of_counters;

    _cuda_range_vector.cmp_info.initialized = 1;
    strLen = snprintf(_cuda_range_vector.cmp_info.disabled_reason, sizeof(_cuda_range_vector.cmp_info.disabled_reason),
                          "%s", "");
    if (strLen < 0 || (size_t) strLen >= sizeof(_cuda_range_vector.cmp_info.disabled_reason)) {
        SUBDBG("Failed to fully write the empty cuda_range disabled reason. Proceeding.\n");
    }
    
  fn_exit:
    _cuda_range_vector.cmp_info.disabled = papi_errno;
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int cuda_range_shutdown_component(void)
{
    if (_cuda_range_vector.cmp_info.initialized == 0) {
        SUBDBG("PAPI_shutdown has been called, but either the cuda_range component was never"
               " initialized or PAPI_shutdown was already previously called.\n");
        return PAPI_OK;
    }

    if (_cuda_range_vector.cmp_info.disabled != PAPI_OK) {
        SUBDBG("PAPI_shutdown has been called, but the cuda_range component is disabled.\n");
        return PAPI_OK;
    }

    _cuda_range_vector.cmp_info.initialized = 0;

    return cuda_range_unload_function_pointers_and_shutdown();
}

int cuda_range_shutdown_thread(hwd_context_t *ctx)
{
    cuda_range_context_t *cuda_range_ctx = (cuda_range_context_t *) ctx;
    cuda_range_ctx->initialized = 0;
    cuda_range_ctx->state = 0;

    return PAPI_OK;
}

int cuda_range_cleanup_eventset(hwd_control_state_t *ctl)
{
    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;
    papi_free(cuda_range_ctl->event_codes);
    cuda_range_ctl->event_codes = NULL;
    cuda_range_ctl->num_events = 0;

    return PAPI_OK;
}

static int update_native_events(cuda_range_control_t *, NativeInfo_t *, int);


int initialize_cuda_range_ctx(cuda_range_ctx_t *cuda_range_ctx, uint32_t *event_codes, int native_event_count)
{
    if ((*cuda_range_ctx) == NULL) {
        (*cuda_range_ctx) = (cuda_range_ctx_t) calloc(1, sizeof(struct cuda_range_ctx));
        if ((*cuda_range_ctx) == NULL) {
            SUBDBG("Failed to allocate memory for cuda_range_ctl->cuda_range_ctx.\n");
            return PAPI_ENOMEM;
        }   
    }   

    (*cuda_range_ctx)->event_codes = event_codes;
    (*cuda_range_ctx)->num_events = native_event_count;
    (*cuda_range_ctx)->counters = (long long *) realloc((*cuda_range_ctx)->counters, (*cuda_range_ctx)->num_events * sizeof(long long));
    if ((*cuda_range_ctx)->counters == NULL) {
        SUBDBG("Failed to allocate memory for cuda_range_ctl->cuda_range_ctx->counters.\n");
        return PAPI_ENOMEM;
    } 

    return PAPI_OK;
}


int cuda_range_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx __attribute__((unused)))
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // This check is required such that PAPI_EMULPASS error codes are caught.
    if (ntv_count == 0) {
        return PAPI_OK;
    }

    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;

    papi_errno = update_native_events(cuda_range_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = initialize_cuda_range_ctx(&cuda_range_ctl->cuda_range_ctx, cuda_range_ctl->event_codes, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = cuda_range_store_added_native_events(cuda_range_ctl->event_codes, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}

int update_native_events(cuda_range_control_t *ctl, NativeInfo_t *ntv_info, int ntv_count)
{
    int papi_errno = PAPI_OK;

    if (ntv_count != ctl->num_events) {
        ctl->num_events = ntv_count;
        if (ntv_count == 0) {
            papi_free(ctl->event_codes);
            ctl->event_codes = NULL;
            goto fn_exit;
        }
        else {
            ctl->event_codes = papi_realloc(ctl->event_codes, ntv_count * sizeof(*ctl->event_codes));
            if (ctl->event_codes == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }
        }
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        ctl->event_codes[i] = ntv_info[i].ni_event;
        ntv_info[i].ni_position = i;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->num_events = 0;
    goto fn_exit;
}

int cuda_range_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    cuda_range_context_t *cuda_range_ctx = (cuda_range_context_t *) ctx;
    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;

    if (cuda_range_ctl->num_events == 0) {
        SUBDBG("Error! Cannot call PAPI_start on an empty eventset.\n");
        return PAPI_ENOSUPP;
    }

    if (cuda_range_ctx->state == CUDA_RANGE_EVENTS_RUNNING) {
        SUBDBG("Error! Cannot PAPI_start more than one eventset at a time for every component.\n");
        return PAPI_EISRUN;
    }

    int papi_errno = cuda_range_start_profiling();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuda_range_ctx->state = CUDA_RANGE_EVENTS_RUNNING;

    return PAPI_OK;
}

int cuda_range_read(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl, long long **val, int flags __attribute__((unused)))
{
    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;

    int papi_errno = cuda_range_decode_and_evaluate_counter_data(cuda_range_ctl->cuda_range_ctx, val);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return PAPI_OK;
}

int cuda_range_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    cuda_range_context_t *cuda_range_ctx = (cuda_range_context_t *) ctx;
    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;

    if (cuda_range_ctx->state == CUDA_RANGE_EVENTS_STOPPED) {
        SUBDBG("Error! Cannot PAPI_stop an eventset that has yet to have PAPI_start called on it.\n");
        return PAPI_ENOTRUN;
    }

    int papi_errno = cuda_range_stop_profiling();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    } 

    cuda_range_ctx->state = CUDA_RANGE_EVENTS_STOPPED;
    cuda_range_ctl->cuda_range_ctx = NULL;

    return papi_errno;
}

int cuda_range_reset(hwd_context_t *ctx __attribute__((unused)), hwd_control_state_t *ctl)
{
    cuda_range_control_t *cuda_range_ctl = (cuda_range_control_t *) ctl;

    return cuda_range_reset_counters(cuda_range_ctl->cuda_range_ctx);
}

int cuda_range_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return cuda_range_event_enum(event_code, modifier);
}

int cuda_range_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return cuda_range_native_event_code_to_native_event_name(event_code, name, len);
}

int cuda_range_ntv_name_to_code(const char *name, unsigned int *code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    uint32_t event_code;
    papi_errno = cuda_range_native_event_name_to_native_event_code(name, &event_code);
    if(papi_errno != PAPI_OK) {
        return papi_errno;
    }
    *code = (unsigned int) event_code;

    return PAPI_OK;
}

int cuda_range_ntv_code_to_descr(unsigned int event_code __attribute__((unused)), char *descr __attribute__((unused)), int len __attribute__((unused)))
{
    SUBDBG("cuda_range_ntv_code_to_info is implemented; therefore, implementation of"
           " cuda_range_ntv_code_to_descr should not be necessary.\n");
    return PAPI_ENOIMPL;
}

int cuda_range_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return cuda_range_event_code_to_info(event_code, info);
}

int cuda_range_set_domain(hwd_control_state_t *ctl __attribute__((unused)), int domain __attribute__((unused)))
{
    return PAPI_OK;
}

int cuda_range_ctl(hwd_context_t *ctx __attribute__((unused)), int code __attribute__((unused)), _papi_int_option_t *option __attribute__((unused)))
{
    return PAPI_OK;
}

int check_n_initialize(void)
{
    if (!_cuda_range_vector.cmp_info.initialized) {
        return cuda_range_init_private();
    }

    return _cuda_range_vector.cmp_info.disabled;
}
