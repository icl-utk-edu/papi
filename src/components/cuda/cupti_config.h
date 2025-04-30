/**
 * @file    cupti_config.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __LCUDA_CONFIG_H__
#define __LCUDA_CONFIG_H__

#include <cupti.h>

/* used to assign the EventSet state  */
#define CUDA_EVENTS_STOPPED (0x0)
#define CUDA_EVENTS_RUNNING (0x2)

#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)

#if (CUPTI_API_VERSION >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION)
#   define API_PERFWORKS 1
#endif

// The Events API has been deprecated in Cuda Toolkit 12.8 and will be removed in a future
// CUDA release (https://docs.nvidia.com/cupti/api/group__CUPTI__EVENT__API.html).
// TODO: When the Events API has been removed #define CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION
// and set it to the last version that is supported. Use this macro as a runtime check in
// `cuptic_determine_runtime_api`.
#define API_EVENTS 2

#endif  /* __LCUDA_CONFIG_H__ */
