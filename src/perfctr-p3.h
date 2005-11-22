#ifndef _PAPI_PENTIUM3_H
#define _PAPI_PENTIUM3_H

#ifdef _WIN32
#define NEED_FFSLL
#define inline_static static __inline
#include <errno.h>
#include "cpuinfo.h"
#include "pmclib.h"
#endif

#define PERF_MAX_COUNTERS 4
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define P3_MAX_REGS_PER_EVENT 2

#include "papi.h"
#include "papi_preset.h"


#ifdef _WIN32

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
#define PERFCTR_X86_AMD_K8	13
#define PERFCTR_X86_INTEL_PENTM	14	/* Pentium M */
#define PERFCTR_X86_AMD_K8C	15	/* Revision C */
#define PERFCTR_X86_INTEL_P4M3	16	/* model 3 and above */

#define PERFCTR26 /* make it look like a recent Perfctr */

/* Lock macros. */
extern CRITICAL_SECTION lock[PAPI_MAX_LOCK];

#define  _papi_hwd_lock(lck) EnterCriticalSection(&lock[lck])
#define  _papi_hwd_unlock(lck) LeaveCriticalSection(&lock[lck])

typedef int hwd_siginfo_t;
typedef CONTEXT hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) ((caddr_t)(((ucontext_t *)ctx)->Eip))

/* Windows DOES NOT support hardware overflow */
#define HW_OVERFLOW 0

#else

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;


/* Overflow macros */
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->sc_rip)
  #else
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
  #endif
#else
  #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1


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

/* Used in resources.selector to determine on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4)

#define HAS_MESI  0x100 /* indicates this event supports MESI modifiers */ 
#define HAS_MOESI 0x200 /* indicates this event supports MOESI modifiers */
#define MOESI_M   0x1000 /* Modified bit */
#define MOESI_O   0x0800 /* Owner bit */
#define MOESI_E   0x0400 /* Exclusive bit */
#define MOESI_S   0x0200 /* Shared bit */
#define MOESI_I   0x0100 /* Invalid bit */
#define MOESI_M_INTEL   MOESI_O /* Modified bit on Intel processors */
#define MOESI_ALL 0x1F00 /* mask for MOESI bits in event code or counter_cmd */


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

#define MODEL_ERROR "This is not a Pentium I,II,III, Athlon or Opteron"

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;

/* Overflow macros */
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext))->sc_rip)
  #else    
     #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&((hwd_ucontext_t *)ctx.ucontext)->uc_mcontext))->rip)
  #endif
#else
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&((hwd_ucontext_t *)ctx.ucontext)->uc_mcontext))->eip)
#endif

#define HW_OVERFLOW     1

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME        perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
#endif

int check_p4(int);
int mdi_init();
int linux_vector_table_setup(papi_vectors_t *vtable);
void lock_init();
#endif /* _PAPI_PENTIUM3 */
