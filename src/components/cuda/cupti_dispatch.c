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

int cuptid_shutdown(void)
{
    int papi_errno;
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        papi_errno = cuptip_shutdown();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        papi_errno = cuptie_shutdown();
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
#endif

    }

    return cuptic_shutdown();
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
        cuptic_disabled_reason_set("PAPI not built with NVIDIA profiler API support.");
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

int cuptid_thread_info_create(cuptid_info_t *info)
{
    return cuptic_ctxarr_create((cuptic_info_t *) info);
}

int cuptid_thread_info_destroy(cuptid_info_t *info)
{
    return cuptic_ctxarr_destroy((cuptic_info_t *) info);
}

int cuptid_control_create(ntv_event_table_t event_names, cuptid_info_t info, cuptid_ctl_t *pcupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_create(event_names, (cuptic_info_t) info, (cuptip_control_t *) pcupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined (API_EVENTS)
        return cuptie_control_create(event_names, (cuptic_info_t) info, (cuptie_control_t *) pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_destroy(cuptid_ctl_t *pcupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_destroy((cuptip_control_t *) pcupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_destroy((cuptie_control_t *) pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_start(cuptid_ctl_t cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_start((cuptip_control_t) cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_start((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_stop(cuptid_ctl_t cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_stop((cuptip_control_t) cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_stop((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_read(cuptid_ctl_t cupti_ctl, long long *values)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_read((cuptip_control_t) cupti_ctl, values);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_read((cuptie_control_t) cupti_ctl, values);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_control_reset(cuptid_ctl_t cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_control_reset((cuptip_control_t) cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_control_reset((cuptie_control_t) cupti_ctl);
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

void cuptid_event_table_destroy(ntv_event_table_t *evt_table)
{
    cuptiu_event_table_destroy(evt_table);
}

int cuptid_event_table_create(ntv_event_table_t *evt_table)
{
    return cuptiu_event_table_create(sizeof(cuptiu_event_t), evt_table);
}

int cuptid_event_table_select_by_idx(ntv_event_table_t evt_table, int count, int *idcs, ntv_event_table_t *pevt_names)
{
    return cuptiu_event_table_select_by_idx(evt_table, count, idcs, pevt_names);
}

int cuptid_event_table_find_name(ntv_event_table_t evt_table, const char *evt_name, ntv_event_t *found_rec)
{
    return cuptiu_event_table_find_name(evt_table, evt_name, found_rec);
}

int cuptid_event_table_insert_record(ntv_event_table_t evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    return cuptiu_event_table_insert_record(evt_table, evt_name, evt_code, evt_pos);
}

int cuptid_event_table_get_item(ntv_event_table_t evt_table, unsigned int evt_idx, ntv_event_t *record)
{
    return cuptiu_event_table_get_item(evt_table, evt_idx, record);
}
