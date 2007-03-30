#ifndef _PAPI_PENTIUM3_H
#define _PAPI_PENTIUM3_H

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#endif
#include <inttypes.h>
#include "libperfctr.h"

#define MAX_COUNTERS 4

#include "papi.h"
#include "papi_preset.h"

/* Overflow macros */
/* old (PAPI <= 3.5) style overflow address:
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->sc_rip)
  #else
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
  #endif
#else
  #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif
*/

/* new (PAPI => 3.9.0) style overflow address: */
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define OVERFLOW_PC sc_rip
  #else
    #define OVERFLOW_PC rip
  #endif
#else
    #define OVERFLOW_PC eip
#endif
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&((hwd_ucontext_t *)ctx.ucontext)->uc_mcontext))->OVERFLOW_PC)

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

typedef struct _p3_register {
   unsigned int selector;       /* Mask for which counters in use */
   int counter_cmd;             /* The event code */
} _p3_register_t;

typedef struct _p3_reg_alloc {
   _p3_register_t ra_bits;       /* Info about this native event mapping */
   unsigned ra_selector;        /* Bit mask showing which counters can carry this metric */
   unsigned ra_rank;            /* How many counters can carry this metric */
} _p3_reg_alloc_t;

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
   _p3_register_t resources;
} native_event_entry_t;

typedef struct _p3_perfctr_control {
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   unsigned char master_selector;
   _p3_register_t allocated_registers;
   struct vperfctr_control control;
   struct perfctr_sum_ctrs state;
   /* Allow attach to be per-eventset. */
   struct rvperfctr * rvperfctr;
} _p3_perfctr_control_t;

typedef struct _p3_perfctr_context {
   struct vperfctr *perfctr;
/*  _p3_perfctr_control_t start; */
} _p3_perfctr_context_t;

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef _p3_reg_alloc_t cmp_reg_alloc_t;
typedef _p3_register_t cmp_register_t;
typedef _p3_perfctr_control_t cmp_control_state_t;
typedef _p3_perfctr_context_t cmp_context_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#define hwd_pmc_control vperfctr_control


/* Used in resources.selector to determine on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4)

#define HAS_MESI  0x0010 /* indicates this event supports MESI modifiers */ 
#define HAS_MOESI 0x0020 /* indicates this event supports MOESI modifiers */
#define HAS_UMASK 0x0040 /* indicates this event supports general UMASK modifiers */
#define MOESI_M   0x1000 /* Modified bit */
#define MOESI_O   0x0800 /* Owner bit */
#define MOESI_E   0x0400 /* Exclusive bit */
#define MOESI_S   0x0200 /* Shared bit */
#define MOESI_I   0x0100 /* Invalid bit */
#define MOESI_M_INTEL   MOESI_O /* Modified bit on Intel processors */
#define MOESI_ALL 0x1F00 /* mask for MOESI bits in event code or counter_cmd */
#define UNIT_MASK_ALL 0xFF00 /* indicates this event supports general UMASK modifiers */


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
#define MODEL_ERROR "This is not a Pentium I,II,III, Athlon or Opteron"

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;

//extern papi_vector_t _p3_vector;
#define MY_VECTOR _p3_vector


#if __CATAMOUNT__
  extern void _start ( );
  extern caddr_t _etext[ ], _edata[ ];
  extern caddr_t __stop___libc_freeres_ptrs[ ];
#else
  extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;
#endif

#endif /* _PAPI_PENTIUM3 */
