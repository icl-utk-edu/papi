#ifndef __ROCP_CONFIG_H__
#define __ROCP_CONFIG_H__

#define PAPI_ROCM_MAX_COUNTERS (512)

#define ROCM_PROFILE_SAMPLING_MODE (0x0)
#define ROCM_EVENTS_OPENED         (0x1)
#define ROCM_EVENTS_RUNNING        (0x2)

extern unsigned _rocm_lock;

#endif /* End of __ROCP_CONFIG_H__ */
