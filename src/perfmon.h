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

typedef pfmlib_event_t pfm_register_t;
typedef int pfm_register_map_t;
typedef int pfm_reg_alloc_t;

/* Native events consist of a flag field, an event field, and a unit mask field.
 * The next 4 macros define the characteristics of the event and unit mask fields.
 */
#define PAPI_NATIVE_EVENT_AND_MASK 0x000003ff	/* 12 bits == 4096 max events */
#define PAPI_NATIVE_EVENT_SHIFT 0
#define PAPI_NATIVE_UMASK_AND_MASK 0x03fffc00	/* 16 bits for unit masks */
#define PAPI_NATIVE_UMASK_MAX 16				/* 16 possible unit masks */
#define PAPI_NATIVE_UMASK_SHIFT 10

#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

typedef struct {
  /* Context structure to kernel, different for attached */
  int ctx_fd;
  pfarg_ctx_t *ctx;
  /* Load structure to kernel, different for attached */
  pfarg_load_t *load;
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  pfm_register_map_t bits;
  /* Buffer to pass to library to control the counters */
  pfmlib_input_param_t in;
  /* Buffer to pass from the library to control the counters */
  pfmlib_output_param_t out;
  /* Is this eventset multiplexed? Actually it holds the microseconds of the switching interval, 0 if not mpx. */
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
  long long counts[PFMLIB_MAX_PMDS];
} pfm_control_state_t;

typedef struct {
#if defined(USE_PROC_PTTIMER)
   int stat_fd;
#endif
  /* Main context structure to kernel */
  int ctx_fd;
  pfarg_ctx_t ctx;
  /* Main load structure to kernel */
  pfarg_load_t load;
  /* Structure to inform the kernel about sampling */
  pfm_dfl_smpl_arg_t smpl;
  /* Address of mmap()'ed sample buffer */
  void *smpl_buf;
} pfm_context_t;

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef pfm_reg_alloc_t cmp_reg_alloc_t;
typedef pfm_register_t cmp_register_t;
typedef pfm_control_state_t cmp_control_state_t;
typedef pfm_context_t cmp_context_t;

#define MY_VECTOR _papi_pfm_vector

extern volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#define MUTEX_OPEN 0
#define MUTEX_CLOSED 1

/* Locking functions */

#if defined(__ia64__)
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
static inline void __raw_spin_lock(volatile unsigned int *lock)
{
  unsigned int tmp;
  extern int _perfmon2_pfm_pmu_type;
  if (_perfmon2_pfm_pmu_type == PFMLIB_MIPS_R10000_PMU)
    {
		__asm__ __volatile__(
		"	.set	noreorder	# __raw_spin_lock	\n"
		"1:	ll	%1, %2					\n"
		"	bnez	%1, 1b					\n"
		"	 li	%1, 1					\n"
		"	sc	%1, %0					\n"
		"	beqzl	%1, 1b					\n"
		"	 nop						\n"
		"	sync						\n"
		"	.set	reorder					\n"
		: "=m" (*lock), "=&r" (tmp)
		: "m" (*lock)
		: "memory");
    } 
  else if (_perfmon2_pfm_pmu_type == PFMLIB_MIPS_ICE9A_PMU) 
    {
		__asm__ __volatile__(
		"	.set	noreorder	# __raw_spin_lock	\n"
		"1:	ll	%1, %2					\n"
		"  	ll	%1, %2					\n"
		"	bnez	%1, 1b					\n"
		"	 li	%1, 1					\n"
		"	sc	%1, %0					\n"
		"	beqz	%1, 1b					\n"
		"	 sync						\n"
		"	.set	reorder					\n"
		: "=m" (*lock), "=&r" (tmp)
		: "m" (*lock)
		: "memory");
    } 
  else 
    {
		__asm__ __volatile__(
		"	.set	noreorder	# __raw_spin_lock	\n"
		"1:	ll	%1, %2					\n"
		"	bnez	%1, 1b					\n"
		"	 li	%1, 1					\n"
		"	sc	%1, %0					\n"
		"	beqz	%1, 1b					\n"
		"	 sync						\n"
		"	.set	reorder					\n"
		: "=m" (*lock), "=&r" (tmp)
		: "m" (*lock)
		: "memory");
    }
}

static inline void __raw_spin_unlock(volatile unsigned int *lock)
{
	__asm__ __volatile__(
	"	.set	noreorder	# __raw_spin_unlock	\n"
	"	sync						\n"
	"	sw	$0, %0					\n"
	"	.set\treorder					\n"
	: "=m" (*lock)
	: "m" (*lock)
	: "memory");
}
#define  _papi_hwd_lock(lck) __raw_spin_lock(&_papi_hwd_lock_data[lck]);
#define  _papi_hwd_unlock(lck) __raw_spin_unlock(&_papi_hwd_lock_data[lck])
#elif defined(__powerpc__)

/*
 * These functions are slight modifications of the functions in
 * /usr/include/asm-ppc/system.h.
 *
 *  We can't use the ones in system.h directly because they are defined
 *  only when __KERNEL__ is defined.
 */

static __inline__ unsigned long
papi_xchg_u32(volatile void *p, unsigned long val)
{
        unsigned long prev;

        __asm__ __volatile__ ("\n\
        sync \n\
1:      lwarx   %0,0,%2 \n\
        stwcx.  %3,0,%2 \n\
        bne-    1b \n\
        isync"
        : "=&r" (prev), "=m" (*(volatile unsigned long *)p)
        : "r" (p), "r" (val), "m" (*(volatile unsigned long *)p)
        : "cc", "memory");

        return prev;
}

/*
 * The two defines below are taken directly from the MIPS implementation.
 */
#define  _papi_hwd_lock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  do {                                                  \
  retval = papi_xchg_u32(&_papi_hwd_lock_data[lck],MUTEX_CLOSED);  \
  } while(retval != (unsigned int)MUTEX_OPEN);	        \
} while(0)
#define  _papi_hwd_unlock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  retval = papi_xchg_u32(&_papi_hwd_lock_data[lck],MUTEX_OPEN); \
} while(0)

#elif defined(__crayx2)					/* CRAY X2 */
#include <pthread.h>
static pthread_spinlock_t crayx2_mutex[PAPI_MAX_LOCK];
inline static void _papi_hwd_lock_init (void)
{
	int i;
	for (i=0; i<PAPI_MAX_LOCK; i++) {
		pthread_spin_init (&crayx2_mutex[i], PTHREAD_PROCESS_PRIVATE);
	}
}
inline static void _papi_hwd_lock (int lck)
{
	pthread_spin_lock (&crayx2_mutex[lck]);
	_papi_hwd_lock_data[lck] = MUTEX_CLOSED;
}
inline static void _papi_hwd_unlock (int lck)
{
	pthread_spin_unlock (&crayx2_mutex[lck]);
	_papi_hwd_lock_data[lck] = MUTEX_OPEN;
}
#elif defined(__sparc__)
static inline void __raw_spin_lock(volatile unsigned int *lock)
{
	__asm__ __volatile__(
	"\n1:\n\t"
	"ldstub	[%0], %%g2\n\t"
	"orcc	%%g2, 0x0, %%g0\n\t"
	"bne,a	2f\n\t"
	" ldub	[%0], %%g2\n\t"
	".subsection	2\n"
	"2:\n\t"
	"orcc	%%g2, 0x0, %%g0\n\t"
	"bne,a	2b\n\t"
	" ldub	[%0], %%g2\n\t"
	"b,a	1b\n\t"
	".previous\n"
	: /* no outputs */
	: "r" (lock)
	: "g2", "memory", "cc");
}
static inline void __raw_spin_unlock(volatile unsigned int *lock)
{
	__asm__ __volatile__("stb %%g0, [%0]" : : "r" (lock) : "memory");
}
#define  _papi_hwd_lock(lck) __raw_spin_lock(&_papi_hwd_lock_data[lck]);
#define  _papi_hwd_unlock(lck) __raw_spin_unlock(&_papi_hwd_lock_data[lck])
#else
#error "_papi_hwd_lock/unlock undefined!"
#endif

/* Signal handling functions */
#undef hwd_siginfo_t
typedef struct siginfo hwd_siginfo_t;
#undef hwd_ucontext_t
typedef ucontext_t hwd_ucontext_t;

#if defined(__ia64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.sc_ip
#elif defined(__i386__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_EIP]
#elif defined(__x86_64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.gregs[REG_RIP]
#elif defined(mips)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.pc
#elif defined(__powerpc__) && !defined(__powerpc64__)
/*
 * The index of the Next IP (REG_NIP) was obtained by looking at kernel
 * source code.  It wasn't documented anywhere else that I could find.
 */
#define REG_NIP 32
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.uc_regs->gregs[REG_NIP]
#elif defined(__powerpc64__)
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.regs->nip
#elif defined(__crayx2)					/* CRAY X2 */
#define OVERFLOW_ADDRESS(ctx) ctx.ucontext->uc_mcontext.regs.pc
#elif defined(__sparc__)
#define OVERFLOW_ADDRESS(ctx) ((struct sigcontext *)ctx.ucontext)->si_regs.pc
#else
#error "OVERFLOW_ADDRESS() undefined!"
#endif

#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(OVERFLOW_ADDRESS(ctx))

#endif
