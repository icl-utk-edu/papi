#ifndef _PAPI_PENTIUM3
#define _PAPI_PENTIUM3

/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>
#ifdef _WIN32
#include <errno.h>
#include "cpuinfo.h"
#include "pmclib.h"
#else
#include <sys/ucontext.h>       /* sys/ucontext.h  apparently broken */
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/times.h>
#include <sys/time.h>
#ifndef __x86_64__
#include <asm/system.h>
#include <asm/param.h>
#include <asm/bitops.h>
#endif
#include <linux/unistd.h>

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#endif
#include "asm/atomic.h"
#include <inttypes.h>
#include "libperfctr.h"
#endif

#define PERF_MAX_COUNTERS 4
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define P3_MAX_REGS_PER_EVENT 2

#include "papi.h"
#include "papi_preset.h"


#ifdef _WIN32
#define inline_static static __inline

/* cpu_type values:: lifted from perfctr.h */
#define PERFCTR_X86_GENERIC	0	/* any x86 with rdtsc */
#define PERFCTR_X86_INTEL_P5	1	/* no rdpmc */
#define PERFCTR_X86_INTEL_P5MMX	2
#define PERFCTR_X86_INTEL_P6	3
#define PERFCTR_X86_INTEL_PII	4
#define PERFCTR_X86_INTEL_PIII	5
#define PERFCTR_X86_CYRIX_MII	6
#define PERFCTR_X86_WINCHIP_C6	7	/* no rdtsc */
#define PERFCTR_X86_WINCHIP_2	8	/* no rdtsc */
#define PERFCTR_X86_AMD_K7	9
#define PERFCTR_X86_VIA_C3	10	/* no pmc0 */
#define PERFCTR_X86_INTEL_P4	11	/* model 0 and 1 */
#define PERFCTR_X86_INTEL_P4M2	12	/* model 2 and above */

/* Lock macros. */
extern CRITICAL_SECTION lock[PAPI_MAX_LOCK];

#define  _papi_hwd_lock(lck) EnterCriticalSection(&lock[lck])
#define  _papi_hwd_unlock(lck) LeaveCriticalSection(&lock[lck])

//typedef siginfo_t hwd_siginfo_t;
typedef int hwd_siginfo_t;
//typedef ucontext_t hwd_ucontext_t;
typedef CONTEXT hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) ((caddr_t)(ctx->ucontext->Eip))

#else

#define inline_static inline static

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

/* Overflow-related defines and declarations */
typedef struct {
   siginfo_t *si;
   struct sigcontext *ucontext;
} _papi_hwd_context_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

/* Overflow macros */
#ifdef __x86_64__
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
#else
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif
#define GET_OVERFLOW_CTR_BITS(ctx) ((_papi_hwi_context_t *)ctx)->overflow_vector
#define HASH_OVERFLOW_CTR_BITS_TO_PAPI_INDEX(bit) _papi_hwi_event_index_map[bit]

#define inline_static inline static

#endif /* _WIN32 */

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
typedef P3_reg_alloc_t hwd_reg_alloc_t;
typedef P3_register_t hwd_register_t;

#ifdef _WIN32
/* Per eventset data structure for thread level counters */

typedef struct P3_WinPMC_control {
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   unsigned char master_selector;
   P3_register_t allocated_registers;
   /* Buffer to pass to the kernel to control the counters */
   struct vpmc_control control;
   struct pmc_state state;
} P3_WinPMC_control_t;

/* Per thread data structure for thread level counters */

typedef struct P3_WinPMC_context {
   /* Handle to the open kernel driver */
   HANDLE self;
/*   P3_WinPMC_control_t start; */
} P3_WinPMC_context_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef P3_WinPMC_control_t hwd_control_state_t;
typedef P3_WinPMC_context_t hwd_context_t;
#define hwd_pmc_control vpmc_control

#else

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

#endif

/* Used in determining on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4)

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
#define MODEL_ERROR "This is not a Pentium 3"

#define PAPI_VENDOR_UNKNOWN -1
#define PAPI_VENDOR_INTEL   1
#define PAPI_VENDOR_AMD     2
#define PAPI_VENDOR_CYRIX   3

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;
extern unsigned int p3_size, p2_size, ath_size, opt_size, NATIVE_TABLE_SIZE;
extern char *basename(char *);
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;

#endif /* _PAPI_PENTIUM3 */
