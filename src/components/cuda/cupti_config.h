/**
 * @file    cupti_config.h
 */

#ifndef __LCUDA_CONFIG_H__
#define __LCUDA_CONFIG_H__

#include <cupti.h>

/* used to assign the EventSet state  */
#define CUDA_EVENTS_STOPPED (0x0)
#define CUDA_EVENTS_RUNNING (0x2)

/*
#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)
#if (CUPTI_API_VERSION >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION)
#   define API_PERFWORKS 1
#endif
*/

// NOTE: Cuda Toolkit 13 now formats the CUPTI API version as
// xxyyzz where:
// xx: Major version of the Cuda Toolkit
// yy: Minor version of the Cuda Toolkit
// zz: CUPTI-specific update or patch version
// which is why we have 13000
/*
#define CUPTI_EVENT_AND_METRIC_MAX_SUPPORTED_VERSION (13000)
#if (CUPTI_API_VERSION < CUPTI_EVENT_AND_METRIC_MAX_SUPPORTED_VERSION)
#   define API_EVENTS 2
#endif
*/

#define API_PERFWORKS 1
#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)

#define API_LEGACY 2
#define CUPTI_EVENT_AND_METRIC_MAX_SUPPORTED_VERSION (13000)


#endif  /* __LCUDA_CONFIG_H__ */
