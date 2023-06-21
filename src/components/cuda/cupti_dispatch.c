/**
 * @file    cupti_dispatch.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include "lcuda_config.h"
#include "cupti_common.h"
#include "cupti_dispatch.h"
#include "lcuda_debug.h"

#if defined(API_PERFWORKS)
#   include "cupti_profiler.h"
#endif

#if defined(API_EVENTS)
#   include "cupti_events.h"
#endif

void cuptid_shutdown(void)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        cuptip_shutdown();
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        cuptie_shutdown();
#endif

    }

    cuptic_shutdown();
}

int cuptid_init(const char **pdisabled_reason)
{
    int papi_errno;
    papi_errno = cuptic_init(pdisabled_reason);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        papi_errno = cuptip_init(pdisabled_reason);
#else
        *pdisabled_reason = "PAPI not built with NVIDIA profiler API support.";
        papi_errno = PAPI_ECMP;
        goto fn_exit;
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        papi_errno = cuptie_init(pdisabled_reason);
#else
        *pdisabled_reason = "Unknown events API problem.";
        papi_errno = PAPI_ECMP;
#endif

    } else {
        *pdisabled_reason = "CUDA configuration not supported.";
        papi_errno = PAPI_ECMP;
    }
fn_exit:
    return papi_errno;
}

int cuptid_thread_info_create(void **pthread_info)
{
    return cuptic_ctxarr_create(pthread_info);
}

int cuptid_thread_info_destroy(void **pthread_info)
{
    return cuptic_ctxarr_destroy(pthread_info);
}

int cuptid_control_create(ntv_event_table_t *event_names, void *thread_info, void **pcupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_create(event_names, thread_info, pcupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined (API_EVENTS)
        return cuptie_control_create(event_names, thread_info, pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_destroy(void **pcupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_destroy(pcupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_destroy(pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_start(void *cupti_ctl, void *thread_info)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_start(cupti_ctl, thread_info);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_start(cupti_ctl, thread_info);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_stop(void *cupti_ctl, void *thread_info)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_stop(cupti_ctl, thread_info);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_stop(cupti_ctl, thread_info);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_read(void *cupti_ctl, long long *values)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_read(cupti_ctl, values);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_read(cupti_ctl, values);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_reset(void *cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_reset(cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_reset(cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_event_enum(ntv_event_table_t *all_evt_names)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_event_enum(all_evt_names);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_event_enum(all_evt_names);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_event_name_to_descr(char *evt_name, char *descr)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_event_name_to_descr(evt_name, descr);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_event_name_to_descr(evt_name, descr);
#endif

    }
    return PAPI_ECMP;
}
