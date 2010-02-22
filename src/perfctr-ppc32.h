#ifndef _PAPI_PERFCTR_PPC32_H
#define _PAPI_PERFCTR_PPC32_H

#ifndef __USE_GNU
#define __USE_GNU
#endif
#ifndef __USE_UNIX98
#define __USE_UNIX98
#endif
#ifndef __USE_XOPEN_EXTENDED
#define __USE_XOPEN_EXTENDED
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#include <sys/ipc.h>
#include <sys/sem.h>

#define inline_static inline static
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <sys/ucontext.h>
#include <sys/times.h>
#include <sys/time.h>
#include <linux/unistd.h>

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#include <inttypes.h>
#include "libperfctr.h"
#endif

#define PERF_MAX_COUNTERS 6
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define ppc32_MAX_REGS_PER_EVENT 2

#include "papi.h"
#include "papi_preset.h"

//guanglei: added defination  of union semun

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#if defined(__GNU_LIBRARY__) && !defined(_SEM_SEMUN_UNDEFINED)
/* union semun is defined by including <sys/sem.h> */
#else
/* according to X/OPEN we have to define it ourselves */
union semun
{
	int val;						   /* value for SETVAL */
	struct semid_ds *buf;			   /* buffer for IPC_STAT, IPC_SET */
	unsigned short *array;			   /* array for GETALL, SETALL */
	/* Linux specific part: */
	struct seminfo *__buf;			   /* buffer for IPC_INFO */
};
#endif

extern int sem_set;

#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */
//guanglei
#if 0
#define  _papi_hwd_lock(lck)                    \
do                                              \
{                                               \
   unsigned int res = 0;                        \
   do {                                         \
      __asm__ __volatile__ ("lock ; " "cmpxchg %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(lock[lck]), "0"(MUTEX_OPEN) : "memory");  \
   } while(res != (unsigned int)MUTEX_OPEN);   \
} while(0)

#define  _papi_hwd_unlock(lck)                  \
do                                              \
{                                               \
   unsigned int res = 0;                       \
   __asm__ __volatile__ ("xchg %0,%1" : "=r"(res) : "m"(lock[lck]), "0"(MUTEX_OPEN) : "memory");                                \
} while(0)
#else
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

#endif // #if 0

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

/* Overflow macros */
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)ctx->ucontext->uc_mcontext.regs->nip

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

typedef struct ppc32_register
{
	unsigned int selector;			   /* Mask for which counters in use */
	int counter_cmd;				   /* The event code */
} ppc32_register_t;

typedef struct ppc32_reg_alloc
{
	ppc32_register_t ra_bits;		   /* Info about this native event mapping */
	unsigned ra_selector;			   /* Bit mask showing which counters can carry this metric */
	unsigned ra_rank;				   /* How many counters can carry this metric */
} ppc32_reg_alloc_t;

/* Per eventset data structure for thread level counters */

typedef struct hwd_native
{
	/* index in the native table, required */
	int index;
	/* Which counters can be used?  */
	unsigned int selector;
	/* Rank determines how many counters carry each metric */
	unsigned char rank;
	/* which counter this native event stays */
	int position;
	int mod;
	int link;
} hwd_native_t;

typedef struct native_event_entry
{
	/* If it exists, then this is the name of this event */
	char name[PAPI_MAX_STR_LEN];
	/* If it exists, then this is the description of this event */
	char *description;
	/* description of the resources required by this native event */
	ppc32_register_t resources;
} native_event_entry_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef ppc32_reg_alloc_t hwd_reg_alloc_t;
typedef ppc32_register_t hwd_register_t;

typedef struct ppc32_perfctr_control
{
	hwd_native_t native[MAX_COUNTERS];
	int native_idx;
	unsigned char master_selector;
	ppc32_register_t allocated_registers;
	struct vperfctr_control control;
	struct perfctr_sum_ctrs state;
	/* Allow attach to be per-eventset. */
	struct rvperfctr *rvperfctr;
} ppc32_perfctr_control_t;

typedef struct ppc32_perfctr_context
{
	struct vperfctr *perfctr;
/*  ppc32_perfctr_control_t start; */
} ppc32_perfctr_context_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef ppc32_perfctr_control_t hwd_control_state_t;
typedef ppc32_perfctr_context_t hwd_context_t;
#define hwd_pmc_control vperfctr_control

/* Used in determining on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTR5 0x10
#define CNTR6 0x20
#define CNTRS12 (CNTR1|CNTR2)
#define CNTRS23 (CNTR2|CNTR3)
#define CNTRS56 (CNTR5|CNTR6)
#define CNTRS1234 (CNTR1|CNTR2|CNTR3|CNTR4)
#define ALLCNTRS_PPC750 (CNTR1|CNTR2|CNTR3|CNTR4)
#define ALLCNTRS_PPC7450 (CNTR1|CNTR2|CNTR3|CNTR4|CNTR5|CNTR6)

#define PMC_OVFL	       0x80000000
#define PERF_USR_ONLY          (1<<(31-1))
#define PERF_OS_ONLY           (1<<(31-2))
#define PERF_MODE_MASK         ~(PERF_USR_ONLY|PERF_OS_ONLY)
#define PERF_INT_ENABLE        (1<<(31-5))
#define PERF_INT_PMC1EN        (1<<(31-16))
#define PERF_INT_PMCxEN        (1<<(31-17))

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL, please run perfex -i to verify your perfctr installation"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define RCNTRL_ERROR "rvperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a Pentium I,II,III, Athlon or Opteron"

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;

/* PPC750 MMCR0
   bit 1 disable supervisor counting
   bit 2 disable user counting
   bit 5 enable interrupts
   bit 16 enable interrupt PMC1
   bit 17 enable interrupt PMC2-4 */

/* PPC7450 same as PPC750 */
#if 0
AO_test_and_set_full( volatile AO_TS_t * addr )
{
	int oldval;
	int temp = 1;					   /* locked value */

	__asm__ __volatile__( "1:\tlwarx %0,0,%3\n"	/* load and reserve               */
						  "\tcmpwi %0, 0\n"	/* if load is                     */
						  "\tbne 2f\n" /*   non-zero, return already set */
						  "\tstwcx. %2,0,%1\n"	/* else store conditional         */
						  "\tbne- 1b\n"	/* retry if lost reservation      */
						  "2:\t\n"	   /* oldval is zero if we set       */
						  :"=&r"( oldval ), "=p"( addr )
						  :"r"( temp ), "1"( addr )
						  :"memory" );

	return oldval;
}
#endif
#endif /* _PAPI_PERFCTR_PPC32_H */
