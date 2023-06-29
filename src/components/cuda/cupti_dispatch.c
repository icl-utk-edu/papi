/**
 * @file    cupti_dispatch.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include "cupti_config.h"
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

void cuptid_disabled_reason_get(const char **msg)
{
    cuptic_disabled_reason_get(msg);
}

int cuptid_init(void)
{
    int papi_errno;
    papi_errno = cuptic_init();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        papi_errno = cuptip_init();
#else
        *pdisabled_reason = "PAPI not built with NVIDIA profiler API support.";
        papi_errno = PAPI_ECMP;
        goto fn_exit;
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        papi_errno = cuptie_init();
#else
        cuptic_disabled_reason_set("Unknown events API problem.");
        papi_errno = PAPI_ECMP;
#endif

    } else {
        cuptic_disabled_reason_set("CUDA configuration not supported.");
        papi_errno = PAPI_ECMP;
    }
fn_exit:
    return papi_errno;
}

int cuptid_thread_info_create(cuptid_thread_info_t *info)
{
    return cuptic_ctxarr_create(info);
}

int cuptid_thread_info_destroy(cuptid_thread_info_t *info)
{
    return cuptic_ctxarr_destroy(info);
}

int cuptid_control_create(cuptiu_event_table_t *event_names, void *thread_info, cuptid_ctl_t *pcupti_ctl)
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

int cuptid_control_destroy(cuptid_ctl_t *pcupti_ctl)
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

int cuptid_control_start(cuptid_ctl_t cupti_ctl, void *thread_info)
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

int cuptid_control_stop(cuptid_ctl_t cupti_ctl, void *thread_info)
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

int cuptid_control_read(cuptid_ctl_t cupti_ctl, long long *values)
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

int cuptid_control_reset(cuptid_ctl_t cupti_ctl)
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

int cuptid_event_enum(cuptiu_event_table_t *all_evt_names)
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

void cuptid_event_table_destroy(ntv_event_table_t **evt_table)
{
    cuptiu_event_table_destroy(evt_table);
}

int cuptid_event_table_create(ntv_event_table_t **evt_table)
{
    return cuptiu_event_table_create(sizeof(cuptiu_event_t), evt_table);
}

int cuptid_event_table_select_by_idx(ntv_event_table_t *evt_table, int count, int *idcs, ntv_event_table_t **pevt_names)
{
    return cuptiu_event_table_select_by_idx(evt_table, count, idcs, pevt_names);
}

int cuptid_event_table_find_name(ntv_event_table_t *evt_table, const char *evt_name, void **found_rec)
{
    return cuptiu_event_table_find_name(evt_table, evt_name, found_rec);
}

int cuptid_event_table_insert_record(ntv_event_table_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    return cuptiu_event_table_insert_record(evt_table, evt_name, evt_code, evt_pos);
}

int cuptid_event_table_get_item(ntv_event_table_t *evt_table, unsigned int evt_idx, void **record)
{
    return cuptiu_event_table_get_item(evt_table, evt_idx, record);
}
