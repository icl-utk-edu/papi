#ifndef _PAPI_AIX               /* _PAPI_AIX */
#define _PAPI_AIX

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <procinfo.h>
#include <sys/atomic_op.h>
#if( ( defined( _AIXVERSION_510) || defined(_AIXVERSION_520)))
#include <sys/procfs.h>
#endif
#include <sys/utsname.h>

#define ANY_THREAD_GETS_SIGNAL
#include <dlfcn.h>

#include "pmapi.h"
#define POWER_MAX_COUNTERS MAX_COUNTERS
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define INVALID_EVENT -2
#define POWER_MAX_COUNTERS_MAPPING 8

#include "papi.h"
#include "papi_preset.h"

extern _text;
extern _etext;
extern _edata;
extern _end;
extern _data;

/* globals */
pm_info_t pminfo;

/* Locks */
extern atomic_p lock[];

#define _papi_hwd_lock(lck)                     \
while(_check_lock(lock[lck],0,1) == TRUE)       \
{                                               \
   usleep(1000);                                \
}

#define _papi_hwd_unlock(lck)                   \
{                                               \
  _clear_lock(lock[lck], 0);                   \
}

/* overflow */
typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void *)(((hwd_ucontext_t *)(ctx->ucontext))->sc_jmpbuf.jmp_context.iar)

/* prototypes */

#endif                          /* _PAPI_AIX */
