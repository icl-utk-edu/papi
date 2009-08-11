#ifndef _PAPI_PENTIUM3_H
#define _PAPI_PENTIUM3_H

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

#ifndef __BSD__ /* #include <malloc.h> */
#include <malloc.h>
#endif

#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#ifdef XML
#include <expat.h>
#endif

#define inline_static inline static
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>

#ifdef __BSD__
#include <ucontext.h>
#else
#include <sys/ucontext.h>
#endif

#include <sys/times.h>
#include <sys/time.h>

#ifndef __BSD__ /* #include <linux/unistd.h> */
  #ifndef __CATAMOUNT__
    #include <linux/unistd.h>	
  #endif
#endif

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#endif
#include <inttypes.h>
#include "libperfctr.h"

/* Native events consist of a flag field, an event field, and a unit mask field.
 * The next 4 macros define the characteristics of the event and unit mask fields.
 */
#define PAPI_NATIVE_EVENT_AND_MASK 0x000003ff	/* 10 bits == 1024 max events */
#define PAPI_NATIVE_EVENT_SHIFT 0
#define PAPI_NATIVE_UMASK_AND_MASK 0x03fffc00	/* 16 bits for unit masks */
#define PAPI_NATIVE_UMASK_MAX 16				/* 16 possible unit masks */
#define PAPI_NATIVE_UMASK_SHIFT 10

#define PERF_MAX_COUNTERS 5
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define P3_MAX_REGS_PER_EVENT 2

#include "papi.h"
#include "papi_preset.h"


/* Lock macros. */
extern volatile unsigned int lock[PAPI_MAX_LOCK];
#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

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

#undef hwd_siginfo_t
typedef siginfo_t hwd_siginfo_t;
#undef hwd_ucontext_t
typedef ucontext_t hwd_ucontext_t;

/* Overflow macros */
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->sc_rip)
  #else
    #define GET_OVERFLOW_ADDRESS(ctx) ctx->ucontext->uc_mcontext.gregs[REG_RIP]
  #endif
#else
  #define GET_OVERFLOW_ADDRESS(ctx) ctx->ucontext->uc_mcontext.gregs[REG_EIP]
#endif

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

typedef struct P3_register {
   unsigned int selector;       /* Mask for which counters in use */
   int counter_cmd;             /* The event code */
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
#undef hwd_reg_alloc_t
typedef P3_reg_alloc_t hwd_reg_alloc_t;
#undef hwd_register_t
typedef P3_register_t hwd_register_t;

typedef struct P3_perfctr_control {
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   unsigned char master_selector;
   P3_register_t allocated_registers;
   struct vperfctr_control control;
   struct perfctr_sum_ctrs state;
   /* Allow attach to be per-eventset. */
   struct rvperfctr * rvperfctr;
} P3_perfctr_control_t;

typedef struct P3_perfctr_context {
   struct vperfctr *perfctr;
/*  P3_perfctr_control_t start; */
} P3_perfctr_context_t;

/* Override void* definitions from PAPI framework layer */
/* typedefs to conform to hardware independent PAPI code. */
#undef hwd_control_state_t
typedef P3_perfctr_control_t hwd_control_state_t;
#undef hwd_context_t
typedef P3_perfctr_context_t hwd_context_t;
#define hwd_pmc_control vperfctr_control

/* Used in resources.selector to determine on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTR5 0x10
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4|CNTR5)

#define HAS_MESI	  0x0100 /* indicates this event supports MESI modifiers */ 
#define HAS_MOESI	  0x0200 /* indicates this event supports MOESI modifiers */
#define HAS_UMASK	  0x0400 /* indicates this event has defined unit mask bits */
#define MOESI_M		  0x1000 /* Modified bit */
#define MOESI_O		  0x0800 /* Owner bit */
#define MOESI_E		  0x0400 /* Exclusive bit */
#define MOESI_S		  0x0200 /* Shared bit */
#define MOESI_I		  0x0100 /* Invalid bit */
#define MOESI_M_INTEL	  MOESI_O /* Modified bit on Intel processors */
#define MOESI_ALL	  0x1F00 /* mask for MOESI bits in event code or counter_cmd */
#define UNIT_MASK_ALL	  0xFF00 /* mask for unit mask bits in event code or counter_cmd */

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
#define VOPEN_ERROR "vperfctr_open() returned NULL, please run perfex -i to verify your perfctr installation"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define RCNTRL_ERROR "rvperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a supported cpu."

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;

#define MY_VECTOR _p3_vector


#if __CATAMOUNT__
  extern void _start ( );
  extern caddr_t _etext[ ], _edata[ ];
  extern caddr_t __stop___libc_freeres_ptrs[ ];
#else
  extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;
#endif

#endif /* _PAPI_PENTIUM3 */
