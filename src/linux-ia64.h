#ifndef _PAPI_LINUX_IA64_H
#define _PAPI_LINUX_IA64_H
/* 
* File:    linux-ia64.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*
*          Kevin London
*	   london@cs.utk.edu
*
* Mods:    Per Eckman
*          pek@pdc.kth.se
*/

#include <stdio.h>
#include <stdlib.h>
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
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
#ifdef PFM30
#include "perfmon/perfmon_default_smpl.h"
#endif
#ifdef ITANIUM2
#include "perfmon/pfmlib_itanium2.h"
#else
#include "perfmon/pfmlib_itanium.h"
#endif

#define inline_static inline static

#define MAX_COUNTER_TERMS 4
#ifdef ITANIUM2
#define MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
#else                           /* itanium */
#define MAX_COUNTERS PMU_ITA_NUM_COUNTERS
#endif

typedef int hwd_register_t;
typedef int hwd_register_map_t;
typedef int hwd_reg_alloc_t;

#ifdef PFM30
   #define NUM_PMCS PFMLIB_MAX_PMCS
   #define NUM_PMDS PFMLIB_MAX_PMDS
   typedef struct param_t {
      pfarg_reg_t pc[NUM_PMCS];
      pfmlib_input_param_t inp;
      pfmlib_output_param_t outp;
   } pfmw_param_t;
   typedef int pfmw_ita_param_t;
   #define PMU_FIRST_COUNTER  4
   #ifdef ITANIUM2
      #define MAX_NATIVE_EVENT  497 /*the number comes from itanium2_events.h*/
   #else
      #define MAX_NATIVE_EVENT  230 /*the number comes from itanium_events.h */
   #endif
#else
 #ifdef ITANIUM2
      typedef pfmlib_ita2_param_t pfmw_ita_param_t;
      #define MAX_NATIVE_EVENT  475 /*the number comes from itanium2_events.h*/
 #else
      #define MAX_NATIVE_EVENT  230 /*the number comes from itanium_events.h */
      typedef pfmlib_ita_param_t pfmw_ita_param_t;
 #endif
   #define NUM_PMCS PMU_MAX_PMCS
   #define NUM_PMDS PMU_MAX_PMDS
   typedef pfmlib_param_t pfmw_param_t;
#endif

typedef struct hwd_control_state {
   /* Arg to perfmonctl */
   pid_t pid;
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   hwd_register_map_t bits;

   pfmw_ita_param_t ita_lib_param;

   /* Buffer to pass to kernel to control the counters */
   pfmw_param_t evt;

   long_long counters[MAX_COUNTERS];
   pfarg_reg_t pd[NUM_PMDS];

/* sampling buffer address */
   void *smpl_vaddr;
   /* Buffer to pass to library to control the counters */
} hwd_control_state_t;


typedef struct itanium_preset_search {
   /* Preset code */
   int preset;
   /* Derived code */
   int derived;
   /* Strings to look for */
   char *(findme[MAX_COUNTERS]);
   char operation[MAX_COUNTERS*5];
} itanium_preset_search_t;

typedef int hwd_context_t;

/* for _papi_hwi_context_t */
#ifdef PFM30
   typedef struct siginfo  hwd_siginfo_t;
#else
   typedef pfm_siginfo_t hwd_siginfo_t;
#endif
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void*)ctx->ucontext->sc_ip

#define PAPI_MAX_NATIVE_EVENTS  MAX_NATIVE_EVENT

#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))
#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))


extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;

#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0
extern volatile uint32_t lock[PAPI_MAX_LOCK];

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck)			 			      \
    while(_InterlockedCompareExchange_acq(&lock[lck],MUTEX_CLOSED,MUTEX_OPEN) \
        != (uint64_t)MUTEX_OPEN);

#define _papi_hwd_unlock(lck)						\
    _InterlockedExchange(&lock[lck], (unsigned __int64)MUTEX_OPEN);

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

#endif
