/**
 * @file    cupti_dispatch.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include "cupti_config.h"
#include "papi_cupti_common.h"
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

int cuptid_ctx_create(cuptid_info_t info, cuptid_ctl_t *pcupti_ctl, uint64_t *events_id, int num_events)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_create((cuptic_info_t) info, (cuptip_control_t *) pcupti_ctl, events_id, num_events);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined (API_EVENTS)
        return cuptie_ctx_create((cuptic_info_t) info, (cuptie_control_t *) pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_destroy(cuptid_ctl_t *pcupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_destroy((cuptip_control_t *) pcupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_ctx_destroy((cuptie_control_t *) pcupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_start(cuptid_ctl_t cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_start((cuptip_control_t) cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_ctx_start((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_stop(cuptid_ctl_t cupti_ctl)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_stop((cuptip_control_t) cupti_ctl);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_ctx_stop((cuptie_control_t) cupti_ctl);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_ctx_read(cuptid_ctl_t cupti_ctl, long long **counters)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_ctx_read((cuptip_control_t) cupti_ctl, counters);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_ctx_read((cuptie_control_t) cupti_ctl, counters);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_enum(uint64_t *event_code, int modifier)
{
    if (cuptic_is_runtime_perfworks_api()) {

#if defined(API_PERFWORKS)
        return cuptip_evt_enum(event_code, modifier);
#endif

    } else if (cuptic_is_runtime_events_api()) {

#if defined(API_EVENTS)
        return cuptie_evt_enum(event_code, modifier);
#endif

    }
    return PAPI_ECMP;
}

int cuptid_evt_code_to_descr(uint64_t event_code, char *descr, int len)
{
    if (cuptic_is_runtime_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cuptip_evt_code_to_descr(event_code, descr, len);
#endif
    } 
    else if (cuptic_is_runtime_events_api()) {
#if defined(API_EVENTS)
        return cuptie_event_code_to_descr(event_code, descr, len);
#endif
    }
    return PAPI_ECMP;
}

int cuptid_ctx_reset(long long *values) 
{
    return cuptip_ctx_reset(values);
}

int cuptid_evt_code_to_info(uint64_t event_code, PAPI_event_info_t *info)
{
    return cuptip_evt_code_to_info(event_code, info);
}

int cuptid_evt_code_to_name(uint64_t event_code, char *name, int len)
{
    return cuptip_evt_code_to_name(event_code, name, len);
}

int cuptid_evt_name_to_code(const char *name, uint64_t *event_code)
{
    return cuptip_evt_name_to_code(name, event_code);
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
