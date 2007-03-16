#ifndef _PAPI_PENTIUM4_H
#define _PAPI_PENTIUM4_H

#include "p4_events.h"

#ifdef _WIN32
#include "cpuinfo.h"
#include "pmclib.h"
#else
#include "libperfctr.h"
#define inline_static inline static
#endif

/* Per event data structure for each event */

typedef struct _p4_perfctr_event {
   unsigned pmc_map;
   unsigned evntsel;
   unsigned evntsel_aux;
   unsigned pebs_enable;
   unsigned pebs_matrix_vert;
   unsigned ireset;
} _p4_perfctr_event_t;


#define MAX_COUNTERS	       18
#define MAX_COUNTER_TERMS	8

/*
The name and description fields should be self-explanatory.
The counter and escr fields are what must be used for register allocation.
There are 18 possible counters. There are 45 possible escrs.
Each native event requires and consumes one counter and one escr.
Multiple counters (2 or 3) are valid with each escr, 
and typically 2 escrs are valid with each event.
Thus, you must track and optimize two different selectors: one for the 18 counters,
and a second for the 45 escrs. 
The bits set in counter[0] correspond to the counters available for use with escr[0].
The value of escr[0] is the bit position in a long_long selector determining which escrs
have been allocated.
Example:
Native event PNE_branch_retired_all can live on 6 counters
counter[0] = MSR_IQ_COUNTER014 = COUNTER(12) | COUNTER(13) | COUNTER(16)
counter[1] = MSR_IQ_COUNTER235 = COUNTER(14) | COUNTER(15) | COUNTER(17)
This event also MUST live on one of two escrs:
escr[0] = MSR_CRU_ESCR2 = 41
escr[1] = MSR_CRU_ESCR3 = 42
If one of counter[0] is chosen, then escr[0] must be chosen.
Likewise for counter[1] and escr[1].
So if this event is assigned to counter 12, bit 12 of counter_selector must be set, 
and bit 41 of escr_selector must be set. These resources are then not available for
any other native event.
*/

typedef struct _p4_perfctr_codes {
   _p4_perfctr_event_t data[MAX_COUNTER_TERMS];
} _p4_perfctr_preset_t;

typedef struct _p4_register {
   unsigned counter[2];         // bitmap of valid counters for each escr
   unsigned escr[2];            // bit offset for each of 2 valid escrs
   unsigned cccr;               // value to be loaded into cccr register
   unsigned event;              // value defining event to be loaded into escr register
   unsigned pebs_enable;        // flag for PEBS counting
   unsigned pebs_matrix_vert;   // flag for PEBS_MATRIX_VERT, whatever that is 
   unsigned ireset;             // I don't really know what this does
} _p4_register_t;

/* defines the fields needed by _papi_hwd_allocate_registers
   to map the counter set */
typedef struct _p4_reg_alloc {
   _p4_register_t ra_bits;       /* Info about this native event mapping */
   unsigned ra_selector;        /* Bit mask showing which counters can carry this metric */
   unsigned ra_rank;            /* How many counters can carry this metric */
   unsigned ra_escr[2];         /* Bit field array showing which (of 45) esc registers can carry this metric */
} _p4_reg_alloc_t;

typedef struct hwd_p4_native_map {
   char *name;                  // ASCII name of the native event
   char *description;           // ASCII description of the native event
   _p4_register_t bits;          // description of resources needed by this event
   int mask;                    // contains all valid mask bits for this event group
   int synonym;                 // index of next synonym if event can be multiply encoded 
} hwd_p4_native_map_t;

typedef struct hwd_p4_mask {
   int bit_pos;                 // bit position of mask bit
   char *name;                  // ASCII name of the native event
   char *description;           // ASCII description of the native event
} hwd_p4_mask_t;

typedef struct _p4_perfctr_control {
   struct vperfctr_control control;
   struct perfctr_sum_ctrs state;
   /* Allow attach to be per-eventset. */
   struct rvperfctr * rvperfctr;
} _p4_perfctr_control_t;

/* Per thread data structure for thread level counters */

typedef struct _p4_perfctr_context {
   struct vperfctr *perfctr;
/*  _p4_perfctr_control_t start; */
} _p4_perfctr_context_t;

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef _p4_reg_alloc_t cmp_reg_alloc_t;
typedef _p4_perfctr_control_t cmp_control_state_t;
typedef _p4_register_t cmp_register_t;
typedef _p4_perfctr_context_t cmp_context_t;

//typedef P4_perfctr_event_t cmp_event_t;

#define hwd_pmc_control vperfctr_control

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL, please run perfex -i to verify your perfctr installation"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define RCNTRL_ERROR "rvperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a Pentium 4"

/* old (PAPI <= 3.5) style overflow address:
#ifdef __x86_64__
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
#else
#define GET_OVERFLOW_ADDRESS(ctx)  (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif
*/

/* new (PAPI => 3.9.0) style overflow address: */
#ifdef __x86_64__
    #define OVERFLOW_PC rip
#else
    #define OVERFLOW_PC eip
#endif
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&((hwd_ucontext_t *)ctx.ucontext)->uc_mcontext))->OVERFLOW_PC)

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

#include <inttypes.h>

extern int sighold(int);
extern int sigrelse(int);

extern papi_vector_t _p4_vector;
#define MY_VECTOR _p4_vector

/* Undefined identifiers in executable */

extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;
extern int _papi_hwd_get_system_info(void);

#endif
