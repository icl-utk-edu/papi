/**
 * @file    cupti_events.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <papi.h>
#include "cupti_events.h"
#include "cupti_common.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
/* Functions needed by CUPTI Events API */
/* ... */

/* CUPTI Events component API functions */

int cuptie_init(void)
{
    cuptic_disabled_reason_set("CUDA events API not implemented.");
    return PAPI_ENOIMPL;
}

int cuptie_control_create(cuptiu_event_table_t *event_names, void *thr_info, cuptie_control_t *pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_destroy(cuptie_control_t *pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_start(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_stop(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_read(cuptie_control_t ctl, long long *values)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_reset(cuptie_control_t ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_event_enum(cuptiu_event_table_t *all_evt_names)
{
    return PAPI_ENOIMPL;
}

int cuptie_event_name_to_descr(const char *evt_name, char *description)
{
    return PAPI_ENOIMPL;
}

int cuptie_shutdown(void)
{
    return PAPI_ENOIMPL;
}
