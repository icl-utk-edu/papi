#ifndef _PAPI_PERFCTR_PPC32_H
#define _PAPI_PERFCTR_PPC32_H

#define HAVE_FFSLL
#define _GNU_SOURCE
#define __USE_GNU
#define __USE_UNIX98
#define __USE_XOPEN_EXTENDED

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
#define P3_MAX_REGS_PER_EVENT 2

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
union semun {
 int val;		       /* value for SETVAL */
 struct semid_ds *buf;     /* buffer for IPC_STAT, IPC_SET */
 unsigned short *array;    /* array for GETALL, SETALL */
			   /* Linux specific part: */
 struct seminfo *__buf;    /* buffer for IPC_INFO */
};
#endif

/* Lock macros. */
extern volatile unsigned int lock[PAPI_MAX_LOCK];
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

#endif  // #if 0

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

/* Overflow macros */

#define GET_OVERFLOW_ADDRESS(ctx) ((struct ucontext *)ctx)->uc_mcontext.regs->nip
#define GET_OVERFLOW_CTR_BITS(ctx) ((_papi_hwi_context_t *)ctx)->overflow_vector
/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

typedef struct P3_register {
   unsigned int selector;       /* Mask for which counters in use */
   int counter_cmd;             /* The event code */
   //guanglei
   //int counter_cmd[MAX_COUNTERS];  //definitions from power3
} P3_register_t;

typedef struct P3_reg_alloc {
   P3_register_t ra_bits;       /* Info about this native event mapping */
   unsigned ra_selector;        /* Bit mask showing which counters can carry this metric */
   unsigned ra_rank;            /* How many counters can carry this metric */
} P3_reg_alloc_t;

/* Per eventset data structure for thread level counters */

typedef struct hwd_native {
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

typedef struct native_event_entry {
   /* If it exists, then this is the name of this event */
   char name[PAPI_MAX_STR_LEN];
   /* If it exists, then this is the description of this event */
   char *description;
   /* description of the resources required by this native event */
   P3_register_t resources;
} native_event_entry_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef P3_reg_alloc_t hwd_reg_alloc_t;
typedef P3_register_t hwd_register_t;

typedef struct P3_perfctr_control {
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   unsigned char master_selector;
   P3_register_t allocated_registers;
   struct vperfctr_control control;
   struct perfctr_sum_ctrs state;
} P3_perfctr_control_t;

typedef struct P3_perfctr_context {
   struct vperfctr *perfctr;
/*  P3_perfctr_control_t start; */
} P3_perfctr_context_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef P3_perfctr_control_t hwd_control_state_t;
typedef P3_perfctr_context_t hwd_context_t;
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

/* Masks to craft an eventcode to perfctr's liking */
#define PERF_CTR_MASK          0xFF000000
#define PERF_INV_CTR_MASK      0x00800000
#define PERF_ENABLE            0x00400000
#define PERF_INT_ENABLE        0x00100000
#define PERF_PIN_CONTROL       0x00080000
#define PERF_EDGE_DETECT       0x00040000
#define PERF_OS                0x00020000
#define PERF_USR               0x00010000
#define PERF_UNIT_MASK         0x0000FF00
#define PERF_EVNT_MASK         0x000000FF

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a Pentium I,II,III, Athlon or Opteron"

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;

#endif /* _PAPI_PERFCTR_PPC32_H */
