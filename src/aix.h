#ifndef _PAPI_AIX_H               /* _PAPI_AIX */
#define _PAPI_AIX_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <time.h>
#if defined( _AIXVERSION_510) || defined(_AIXVERSION_520)
#include <sys/procfs.h>
#include <sys/cred.h>
#endif
#include <procinfo.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <sys/atomic_op.h>
#include <sys/utsname.h>


#include "pmapi.h"

#define inline_static static __inline

#define ANY_THREAD_GETS_SIGNAL
#define POWER_MAX_COUNTERS MAX_COUNTERS
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define INVALID_EVENT -2
#define POWER_MAX_COUNTERS_MAPPING 8

extern _text;
extern _etext;
extern _edata;
extern _end;
extern _data;

/* globals */
#ifdef PM_INITIALIZE
#ifdef _AIXVERSION_510
#define PMINFO_T pm_info2_t
#define PMEVENTS_T pm_events2_t
#else
#define PMINFO_T pm_info_t
#define PMEVENTS_T pm_events_t
#endif
PMINFO_T pminfo;
#else
#define PMINFO_T pm_info_t
#define PMEVENTS_T pm_events_t
/*pm_info_t pminfo;*/
#endif

/* Locks */
extern atomic_p lock[];

#define _papi_hwd_lock(lck)                       \
{                                                 \
  while(_check_lock(lock[lck],0,1) == TRUE) { ; } \
}

#define _papi_hwd_unlock(lck)                   \
{                                               \
  _clear_lock(lock[lck], 0);                    \
}

/* overflow */
/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef hwd_siginfo_t
#undef hwd_ucontext_t
typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void *)(((hwd_ucontext_t *)(ctx->ucontext))->sc_jmpbuf.jmp_context.iar)

#define MY_VECTOR _aix_vector

#endif                          /* _PAPI_AIX */
