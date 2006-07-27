#ifndef _PAPI_PERFMON_H
#define _PAPI_PERFMON_H
/* 
* File:    linux-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <fcntl.h>
#include <ctype.h>
#include <inttypes.h>
#include <libgen.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>

#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"

#define inline_static inline static

typedef int hwd_register_t;
typedef int hwd_register_map_t;
typedef int hwd_reg_alloc_t;

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMCS
#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

typedef struct {
   /* Preset code */
   int preset;
   /* Derived code */
   int derived;
   /* Strings to look for, more than 1 means derived */
   char *(findme[MAX_COUNTERS]);
   /* Operations between entities */
   char operation[MAX_COUNTERS];
} pfm_preset_search_entry_t;

typedef struct {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  hwd_register_map_t bits;
  /* Buffer to pass to library to control the counters */
  pfmlib_input_param_t in;
  /* Buffer to pass from the library to control the counters */
  pfmlib_output_param_t out;
  /* Arguments to the kernel */
  pfarg_pmc_t pc[PFMLIB_MAX_PMCS];
  /* Arguments to the kernel */
  pfarg_pmd_t pd[PFMLIB_MAX_PMDS];
  /* Buffer to gather counters */
  long_long counts[PFMLIB_MAX_PMDS];
} hwd_control_state_t;

typedef struct {
  int fd;  /* file descriptor */
  pfarg_ctx_t ctx;
  pfarg_load_t load;
#if defined(HAS_PER_PROCESS_TIMES)
   int stat_fd;
#endif
} hwd_context_t;

typedef struct hwd_native_register {
  pfmlib_regmask_t selector;
  int pfmlib_event_index;
} hwd_native_register_t;

typedef struct hwd_native_event_entry {
   /* If it exists, then this is the name of this event */
   char name[PAPI_MAX_STR_LEN];
   /* If it exists, then this is the description of this event */
   char description[PAPI_HUGE_STR_LEN];
  /* description of the resources required by this native event */
  hwd_native_register_t resources;
} hwd_native_event_entry_t;

/* Lock macros. */
/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

extern volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0

#if defined(__ia64__)
#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck) { while(_InterlockedCompareExchange_acq(&lock[lck],MUTEX_CLOSED,MUTEX_OPEN) != MUTEX_OPEN) { ; } } 
#define _papi_hwd_unlock(lck) { _InterlockedExchange((volatile int *)&lock[lck], MUTEX_OPEN); }
#else                           /* GCC */
#define _papi_hwd_lock(lck)			 			      \
   { uint64_t res = 0;							      \
    do {								      \
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));            \
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&lock[lck]), "r"(MUTEX_CLOSED) : "memory");				      \
    } while (res != (uint64_t)MUTEX_OPEN); }

#define _papi_hwd_unlock(lck)			 			      \
    { uint64_t res = 0;							      \
    __asm__ __volatile__ ("xchg4 %0=[%1],%2" : "=r"(res) : "r"(&lock[lck]), "r"(MUTEX_OPEN) : "memory"); }
#endif
#elif defined(__i386__)||defined(__x86_64__)
#define  _papi_hwd_lock(lck)                    \
do                                              \
{                                               \
   unsigned int res = 0;                        \
   do {                                         \
      __asm__ __volatile__ ("lock ; " "cmpxchg %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");  \
   } while(res != (unsigned int)MUTEX_OPEN);   \
} while(0)
#define  _papi_hwd_unlock(lck)                  \
do                                              \
{                                               \
   unsigned int res = 0;                       \
   __asm__ __volatile__ ("xchg %0,%1" : "=r"(res) : "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");                                \
} while(0)
#elif defined(mips)
#else
#error "_papi_hwd_lock/unlock undefined!"
#endif

#if defined(__ia64__)
typedef struct sigcontext hwd_ucontext_t;
#define GET_OVERFLOW_ADDRESS(ctx)  (caddr_t)(ctx->ucontext->sc_ip)
#elif defined(__i386__)
typedef ucontext_t hwd_ucontext_t;
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#elif defined(__x86_64__)
typedef ucontext_t hwd_ucontext_t;
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
#elif defined(mips)
#define GET_OVERFLOW_ADDRESS(ctx)  (caddr_t)(ctx->ucontext->sc_pc)
#else
#error "GET_OVERFLOW_ADDRESS() undefined!"
#endif

typedef struct siginfo hwd_siginfo_t;

#endif
