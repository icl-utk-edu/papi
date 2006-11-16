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
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>
#include <sys/ptrace.h>
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
#include "perfmon/perfmon_dfl_smpl.h"

#ifdef __ia64__
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_montecito.h"
#endif

#if defined(DEBUG)
#define DEBUGCALL(a,b) { if (ISLEVEL(a)) { b; } }
#else
#define DEBUGCALL(a,b)
#endif

#define inline_static inline static

typedef pfmlib_event_t hwd_register_t;
typedef int hwd_register_map_t;
typedef int hwd_reg_alloc_t;

#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS
#define PERFMON_EVENT_FILE "perfmon_events.csv"

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
  /* Context structure to kernel, different for attached */
  pfarg_ctx_t *ctx;
  /* Load structure to kernel, different for attached */
  pfarg_load_t *load;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  hwd_register_map_t bits;
  /* Buffer to pass to library to control the counters */
  pfmlib_input_param_t in;
  /* Buffer to pass from the library to control the counters */
  pfmlib_output_param_t out;
  /* Is this eventset multiplexed? */
  int multiplexed;
  /* Arguments to kernel for multiplexing, first number of sets */
  int num_sets;
  /* Arguments to kernel to set up the sets */
  pfarg_setdesc_t set[PFMLIB_MAX_PMDS];
  /* Buffer to get information out of the sets when reading */
  pfarg_setinfo_t setinfo[PFMLIB_MAX_PMDS];
  /* Arguments to the kernel */
  pfarg_pmc_t pc[PFMLIB_MAX_PMCS];
  /* Arguments to the kernel */
  pfarg_pmd_t pd[PFMLIB_MAX_PMDS];
  /* Buffer to gather counters */
  long_long counts[PFMLIB_MAX_PMDS];
} hwd_control_state_t;

typedef struct {
#if defined(USE_PROC_PTTIMER)
   int stat_fd;
#endif
  /* Main context structure to kernel */
  pfarg_ctx_t ctx;
  /* Main load structure to kernel */
  pfarg_load_t load;
  /* Structure to inform the kernel about sampling */
  pfm_dfl_smpl_arg_t smpl;
  /* Address of mmap()'ed sample buffer */
  void *smpl_buf;
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

/* Locking functions */

#if defined(__ia64__)
#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck) { while(_InterlockedCompareExchange_acq(&_papi_hwd_lock_data[lck],MUTEX_CLOSED,MUTEX_OPEN) != MUTEX_OPEN) { ; } } 
#define _papi_hwd_unlock(lck) { _InterlockedExchange((volatile int *)&_papi_hwd_lock_data[lck], MUTEX_OPEN); }
#else                           /* GCC */
#define _papi_hwd_lock(lck)			 			      \
   { uint64_t res = 0;							      \
    do {								      \
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));            \
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_CLOSED) : "memory");				      \
    } while (res != (uint64_t)MUTEX_OPEN); }

#define _papi_hwd_unlock(lck)			 			      \
    { uint64_t res = 0;							      \
    __asm__ __volatile__ ("xchg4 %0=[%1],%2" : "=r"(res) : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_OPEN) : "memory"); }
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
static inline unsigned int papi_cmpxchg_u32(volatile unsigned int * m, unsigned int old,
	unsigned int new)
{
	unsigned int retval;
		__asm__ __volatile__(
		"	.set	push					\n"
		"	.set	noat					\n"
		"	.set	mips3					\n"
		"1:	ll	%0, %2			# __cmpxchg_u32	\n"
		"	bne	%0, %z3, 2f				\n"
		"	.set	mips0					\n"
		"	move	$1, %z4					\n"
		"	.set	mips3					\n"
		"	sc	$1, %1					\n"
		"	beqz	$1, 1b					\n"
		"	sync						\n"
		"2:							\n"
		"	.set	pop					\n"
		: "=&r" (retval), "=R" (*m)
		: "R" (*m), "Jr" (old), "Jr" (new)
		: "memory");
	return retval;
}
static inline unsigned int papi_xchg_u32(volatile unsigned int * m, unsigned int val)
{
	unsigned int retval;
	unsigned int dummy;
	
	__asm__ __volatile__(
		"	.set	mips3					\n"
		"1:	ll	%0, %3			# xchg_u32	\n"
		"	.set	mips0					\n"
		"	move	%2, %z4					\n"
		"	.set	mips3					\n"
		"	sc	%2, %1					\n"
		"	beqz	%2, 1b					\n"
		"	sync						\n"
		"	.set	mips0					\n"
		: "=&r" (retval), "=m" (*m), "=&r" (dummy)
		: "R" (*m), "Jr" (val)
		: "memory");
	return retval;
}

#define  _papi_hwd_lock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  do {                                                  \
  retval = papi_cmpxchg_u32(&_papi_hwd_lock_data[lck],MUTEX_CLOSED,MUTEX_OPEN);  \
  } while(retval != (unsigned int)MUTEX_OPEN);	        \
} while(0)
#define  _papi_hwd_unlock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  retval = papi_xchg_u32(&_papi_hwd_lock_data[lck],MUTEX_OPEN); \
} while(0)
#else
#error "_papi_hwd_lock/unlock undefined!"
#endif

/* Signal handling functions */

typedef struct siginfo hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#if defined(__ia64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.sc_ip
#elif defined(__i386__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_EIP]
#elif defined(__x86_64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_RIP]
#elif defined(mips)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.pc
#else
#error "OVERFLOW_ADDRESS() undefined!"
#endif

#define GET_OVERFLOW_ADDRESS(ctx) (OVERFLOW_ADDRESS((*ctx)))

#endif
