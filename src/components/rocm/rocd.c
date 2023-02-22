/**
 * @file    rocd.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @brief rocm component dispatch layer. Dispatches profiling
 * to the appropriate backend interface (e.g. rocprofiler).
 */

#include "rocd.h"

#ifdef ROCM_PROF_ROCPROFILER
#include "rocp.h"
#endif

int
rocd_init_environment(void)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_init_environment();
#endif
    return PAPI_ENOSUPP;
}

int
rocd_init(void)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_init();
#endif
    return PAPI_ENOSUPP;
}

int
rocd_shutdown(void)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_shutdown();
#endif
    return PAPI_ENOSUPP;
}

int
rocd_evt_enum(unsigned int *event_code, int modifier)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_evt_enum(event_code, modifier);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_evt_get_descr(unsigned int event_code, char *descr, int len)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_evt_get_descr(event_code, descr, len);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_evt_name_to_code(const char *name, unsigned int *event_code)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_evt_name_to_code(name, event_code);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_evt_code_to_name(unsigned int event_code, char *name, int len)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_evt_code_to_name(event_code, name, len);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_err_get_last(const char **error_str)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_err_get_last(error_str);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_open(unsigned int *events_id, int num_events, rocd_ctx_t *ctx)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_open(events_id, num_events, ctx);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_close(rocd_ctx_t ctx)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_close(ctx);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_start(rocd_ctx_t ctx)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_start(ctx);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_stop(rocd_ctx_t ctx)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_stop(ctx);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_read(rocd_ctx_t ctx, long long **counters)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_read(ctx, counters);
#endif
    return PAPI_ENOSUPP;
}

int
rocd_ctx_reset(rocd_ctx_t ctx)
{
#ifdef ROCM_PROF_ROCPROFILER
    return rocp_ctx_reset(ctx);
#endif
    return PAPI_ENOSUPP;
}
