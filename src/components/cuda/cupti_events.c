/**
 * @file    cupti_events.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <papi.h>
#include "lcuda_common.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
// Functions needed by CUPTI Events API
// ...

// CUPTI Events component API functions

int cuptie_init(char **pdisabled_reason)
{
    *pdisabled_reason = "CUDA events API not implemented.";
    return PAPI_ENOIMPL;
}

int cuptie_control_create(ntv_event_table_t *event_names, void *thr_info, void **pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_destroy(void **pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_start(void *ctl, void *thr_info)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_stop(void *ctl, void *thr_info)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_read(void *ctl, long long *values)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_reset(void *ctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_event_enum(ntv_event_table_t *all_evt_names)
{
    return PAPI_ENOIMPL;
}

int cuptie_event_name_to_descr(const char *evt_name, char *description)
{
    return PAPI_ENOIMPL;
}

void cuptie_shutdown(void)
{
    return;
}
