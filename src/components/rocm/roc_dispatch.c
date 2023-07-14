/**
 * @file    roc_dispatch.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * @brief rocm component dispatch layer. Dispatches profiling
 * to the appropriate backend interface (e.g. rocprofiler).
 */

#include "roc_dispatch.h"
#include "roc_common.h"

#ifdef ROCPROFILER_V1
#include "roc_profiler.h"
#else
#include "roc_profiler_v2.h"
#endif

int
rocd_init_environment(void)
{
#ifdef ROCPROFILER_V1
    return rocp_init_environment();
#else
    return rocp2_init_environment();
#endif
}

int
rocd_init(void)
{
    int papi_errno = rocc_init();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

#ifdef ROCPROFILER_V1
    papi_errno = rocp_init();
#else
    papi_errno = rocp2_init();
#endif
    return papi_errno;
}

int
rocd_shutdown(void)
{
#ifdef ROCPROFILER_V1
    int papi_errno = rocp_shutdown();
#else
    int papi_errno = rocp2_shutdown();
#endif
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    papi_errno = rocc_shutdown();
    return papi_errno;
}

int
rocd_evt_enum(unsigned int *event_code, int modifier)
{
#ifdef ROCPROFILER_V1
    return rocp_evt_enum(event_code, modifier);
#else
    return rocp2_evt_enum(event_code, modifier);
#endif
}

int
rocd_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
#ifdef ROCPROFILER_V1
    return rocp_evt_code_to_descr(event_code, descr, len);
#else
    return rocp2_evt_code_to_descr(event_code, descr, len);
#endif
}

int
rocd_evt_name_to_code(const char *name, unsigned int *event_code)
{
#ifdef ROCPROFILER_V1
    return rocp_evt_name_to_code(name, event_code);
#else
    return rocp2_evt_name_to_code(name, event_code);
#endif
}

int
rocd_evt_code_to_name(unsigned int event_code, char *name, int len)
{
#ifdef ROCPROFILER_V1
    return rocp_evt_code_to_name(event_code, name, len);
#else
    return rocp2_evt_code_to_name(event_code, name, len);
#endif
}

int
rocd_err_get_last(const char **error_str)
{
    return rocc_err_get_last(error_str);
}

int
rocd_ctx_open(unsigned int *events_id, int num_events, rocd_ctx_t *ctx)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_open(events_id, num_events, ctx);
#else
    return rocp2_ctx_open(events_id, num_events, ctx);
#endif
}

int
rocd_ctx_close(rocd_ctx_t ctx)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_close(ctx);
#else
    return rocp2_ctx_close(ctx);
#endif
}

int
rocd_ctx_start(rocd_ctx_t ctx)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_start(ctx);
#else
    return rocp2_ctx_start(ctx);
#endif
}

int
rocd_ctx_stop(rocd_ctx_t ctx)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_stop(ctx);
#else
    return rocp2_ctx_stop(ctx);
#endif
}

int
rocd_ctx_read(rocd_ctx_t ctx, long long **counters)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_read(ctx, counters);
#else
    return rocp2_ctx_read(ctx, counters);
#endif
}

int
rocd_ctx_reset(rocd_ctx_t ctx)
{
#ifdef ROCPROFILER_V1
    return rocp_ctx_reset(ctx);
#else
    return rocp2_ctx_reset(ctx);
#endif
}
