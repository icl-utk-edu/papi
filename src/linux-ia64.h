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
* Mods:    Per Ekman
*          pek@pdc.kth.se
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
#include <sys/types.h>
#include <sys/ipc.h>
#ifdef USE_SEMAPHORES
#include <sys/sem.h>
#endif

#if defined(HAVE_MMTIMER)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/mmtimer.h>
#ifndef MMTIMER_FULLNAME
#define MMTIMER_FULLNAME "/dev/mmtimer"
#endif
#endif

#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
#endif

#include "config.h"
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
#include "perfmon/perfmon_default_smpl.h"
#include "perfmon/pfmlib_montecito.h"
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_itanium.h"

#define inline_static inline static

typedef int ia64_register_t;
typedef int ia64_register_map_t;
typedef int ia64_reg_alloc_t;


   #define NUM_PMCS PFMLIB_MAX_PMCS
   #define NUM_PMDS PFMLIB_MAX_PMDS
   
   /* Native events consist of a flag field, an event field, and a unit mask field.
    * The next 4 macros define the characteristics of the event and unit mask fields.
    * Unit Masks are only supported on Montecito and above.
    */
   #define PAPI_NATIVE_EVENT_AND_MASK 0x00000fff	/* 12 bits == 4096 max events */
   #define PAPI_NATIVE_EVENT_SHIFT 0
   #define PAPI_NATIVE_UMASK_AND_MASK 0x0ffff000	/* 16 bits for unit masks */
   #define PAPI_NATIVE_UMASK_MAX 16				/* 16 possible unit masks */
   #define PAPI_NATIVE_UMASK_SHIFT 12

   typedef struct param_t {
      pfarg_reg_t pd[NUM_PMDS];
      pfarg_reg_t pc[NUM_PMCS];
      pfmlib_input_param_t inp;
      pfmlib_output_param_t outp;
      void *mod_inp;	/* model specific input parameters to libpfm    */
      void *mod_outp;	/* model specific output parameters from libpfm */
   } pfmw_param_t;
//   #ifdef ITANIUM3
   typedef struct mont_param_t {
      pfmlib_mont_input_param_t mont_input_param;
      pfmlib_mont_output_param_t  mont_output_param;
   } pfmw_mont_param_t;
//   typedef pfmw_mont_param_t pfmw_ita_param_t;
//   #elif defined(ITANIUM2)
   typedef struct ita2_param_t {
      pfmlib_ita2_input_param_t ita2_input_param;
      pfmlib_ita2_output_param_t ita2_output_param;
   } pfmw_ita2_param_t;
//   typedef pfmw_ita2_param_t pfmw_ita_param_t;
//   #else
   typedef int pfmw_ita1_param_t;
//   #endif

   #define PMU_FIRST_COUNTER  4

   typedef union {
     pfmw_ita1_param_t ita_param;
     pfmw_ita2_param_t ita2_param;
     pfmw_mont_param_t mont_param;
   } pfmw_ita_param_t;


#define MAX_COUNTERS 12
#define MAX_COUNTER_TERMS MAX_COUNTERS

typedef struct ia64_control_state {
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   ia64_register_map_t bits;

   pfmw_ita_param_t ita_lib_param;

   /* Buffer to pass to kernel to control the counters */
   pfmw_param_t evt;

   long long counters[MAX_COUNTERS];
   pfarg_reg_t pd[NUM_PMDS];

/* sampling buffer address */
   void *smpl_vaddr;
   /* Buffer to pass to library to control the counters */
} ia64_control_state_t;


typedef struct itanium_preset_search {
   /* Preset code */
   int preset;
   /* Derived code */
   int derived;
   /* Strings to look for */
   char *(findme[MAX_COUNTERS]);
   char operation[MAX_COUNTERS*5];
} itanium_preset_search_t;

typedef struct Itanium_context {
   int fd;  /* file descriptor */
   pid_t tid;  /* thread id */
#if defined(USE_PROC_PTTIMER)
   int stat_fd;
#endif
} ia64_context_t;

//typedef Itanium_context_t hwd_context_t;

/* for _papi_hwi_context_t */
#undef hwd_siginfo_t
typedef struct siginfo  hwd_siginfo_t;

#undef  hwd_ucontext_t
typedef struct sigcontext hwd_ucontext_t;

/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef  hwd_reg_alloc_t
typedef ia64_reg_alloc_t hwd_reg_alloc_t;
#undef  hwd_register_t
typedef ia64_register_t hwd_register_t;
#undef  hwd_control_state_t
typedef ia64_control_state_t hwd_control_state_t;
#undef  hwd_context_t
typedef ia64_context_t hwd_context_t;

#define GET_OVERFLOW_ADDRESS(ctx)  ((caddr_t)((ctx->ucontext)->sc_ip))

#define SMPL_BUF_NENTRIES 64
#define M_PMD(x)        (1UL<<(x))

#define MONT_DEAR_REGS_MASK	    (M_PMD(32)|M_PMD(33)|M_PMD(36))
#define MONT_ETB_REGS_MASK		(M_PMD(38)| M_PMD(39)| \
		                 M_PMD(48)|M_PMD(49)|M_PMD(50)|M_PMD(51)|M_PMD(52)|M_PMD(53)|M_PMD(54)|M_PMD(55)|\
				 M_PMD(56)|M_PMD(57)|M_PMD(58)|M_PMD(59)|M_PMD(60)|M_PMD(61)|M_PMD(62)|M_PMD(63))

#define DEAR_REGS_MASK      (M_PMD(2)|M_PMD(3)|M_PMD(17))
#define BTB_REGS_MASK       (M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))


#define MY_VECTOR _ia64_vector

#ifdef USE_SEMAPHORES
extern int sem_set;

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

#define  _papi_hwd_lock(lck)                    \
{                                               \
struct sembuf sem_lock = { lck, -1, 0 }; \
if (semop(sem_set, &sem_lock, 1) == -1 ) {      \
abort(); } }
// PAPIERROR("semop errno %d",errno); abort(); } }

#define  _papi_hwd_unlock(lck)                   \
{                                                \
struct sembuf sem_unlock = { lck, 1, 0 }; \
if (semop(sem_set, &sem_unlock, 1) == -1 ) {     \
abort(); } }
// PAPIERROR("semop errno %d",errno); abort(); } }

#else
extern volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#define MUTEX_OPEN 0
#define MUTEX_CLOSED 1

#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck) { while(_InterlockedCompareExchange_acq(&_papi_hwd_lock_data[lck],MUTEX_CLOSED,MUTEX_OPEN) != MUTEX_OPEN) { ; } } 

#define _papi_hwd_unlock(lck) { _InterlockedExchange((volatile int *)&_papi_hwd_lock_data[lck], MUTEX_OPEN); }
#else                           /* GCC */
#define _papi_hwd_lock(lck)			 			      \
   { int res = 0;							      \
    do {								      \
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));            \
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_CLOSED) : "memory");				      \
    } while (res != MUTEX_OPEN); }

#define _papi_hwd_unlock(lck) {  __asm__ __volatile__ ("st4.rel [%0]=%1" : : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_OPEN) : "memory"); }
#endif /* __INTEL_COMPILER */
#endif /* USE_SEMAPHORES */
#endif /* _PAPI_LINUX_IA64_H */
