#ifndef _PERFCTR_X86_H
#define _PERFCTR_X86_H

#include "perfmon/pfmlib.h"
#include "libperfctr.h"
#include "papi_lock.h"

#define MAX_COUNTERS 18
#define MAX_COUNTER_TERMS 8
#define HW_OVERFLOW 1
#define hwd_pmc_control vperfctr_control

#include "linux-context.h"

/* bit fields unique to P4 */
#define ESCR_T0_OS (1 << 3)
#define ESCR_T0_USR (1 << 2)
#define CCCR_OVF_PMI_T0 (1 << 26)
#define FAST_RDPMC (1 << 31)

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels */
#define CONFIG_SMP
#endif


/* Used in resources.selector to determine on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTR5 0x10
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4|CNTR5)

#define HAS_MESI	  0x0100 // indicates this event supports MESI modifiers
#define HAS_MOESI	  0x0200 // indicates this event supports MOESI modifiers
#define HAS_UMASK	  0x0400 // indicates this event has defined unit mask bits
#define MOESI_M		  0x1000 // modified bit
#define MOESI_O		  0x0800 // owner bit
#define MOESI_E		  0x0400 // exclusive bit
#define MOESI_S		  0x0200 // shared bit
#define MOESI_I		  0x0100 // invalid bit
#define MOESI_M_INTEL MOESI_O	// modified bit on Intel processors
#define MOESI_ALL	  0x1F00 // mask for MOESI bits in event code or counter_cmd
#define UNIT_MASK_ALL 0xFF00 // mask for unit mask bits in event code or counter_cmd

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

#define AI_ERROR        "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR     "vperfctr_open() returned NULL, please run perfex -i to verify your perfctr installation"
#define GOPEN_ERROR     "gperfctr_open() returned NULL"
#define VINFO_ERROR     "vperfctr_info() returned < 0"
#define VCNTRL_ERROR    "vperfctr_control() returned < 0"
#define RCNTRL_ERROR    "rvperfctr_control() returned < 0"
#define GCNTRL_ERROR    "gperfctr_control() returned < 0"
#define FOPEN_ERROR     "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR     "This is not a supported cpu."

typedef struct X86_register
{
	unsigned int selector;			   // mask for which counters in use 
	int counter_cmd;				   // event code 
  /******************   P4 elements   *******************/
	unsigned counter[2];			   // bitmap of valid counters for each escr
	unsigned escr[2];				   // bit offset for each of 2 valid escrs
	unsigned cccr;					   // value to be loaded into cccr register
	unsigned event;					   // value defining event to be loaded into escr register
	unsigned pebs_enable;			   // flag for PEBS counting
	unsigned pebs_matrix_vert;		   // flag for PEBS_MATRIX_VERT 
	unsigned ireset;
} X86_register_t;

typedef struct X86_reg_alloc
{
	X86_register_t ra_bits;			   // info about this native event mapping 
	unsigned ra_selector;			   // bit mask showing which counters can carry this metric 
	unsigned ra_rank;				   // how many counters can carry this metric
  /***************  P4 specific element ****************/
	unsigned ra_escr[2];			   // bit field array showing which esc registers can carry this metric
} X86_reg_alloc_t;

typedef struct hwd_native
{
	int index;						   // index in the native table, required    
	unsigned int selector;			   // which counters     
	unsigned char rank;				   // rank determines how many counters carry each metric 
	int position;					   // which counter this native event stays 
	int mod;
	int link;
} hwd_native_t;

typedef struct X86_perfctr_control
{
	hwd_native_t native[MAX_COUNTERS];
	int native_idx;
	unsigned char master_selector;
	X86_register_t allocated_registers;
	struct vperfctr_control control;
	struct perfctr_sum_ctrs state;
	struct rvperfctr *rvperfctr;	   // Allow attach to be per-eventset
} X86_perfctr_control_t;

typedef struct X86_perfctr_context
{
	struct vperfctr *perfctr;
        int stat_fd;
} X86_perfctr_context_t;

/* Override void* definitions from PAPI framework layer 
   with typedefs to conform to PAPI component layer code. */
#undef  hwd_reg_alloc_t
typedef X86_reg_alloc_t hwd_reg_alloc_t;
#undef  hwd_register_t
typedef X86_register_t hwd_register_t;
#undef  hwd_control_state_t
typedef X86_perfctr_control_t hwd_control_state_t;
#undef  hwd_context_t
typedef X86_perfctr_context_t hwd_context_t;

typedef struct native_event_entry
{
	char name[PAPI_MAX_STR_LEN];	   // name of this event
	char *description;				   // description of this event     
	X86_register_t resources;		   // resources required by this native event 
} native_event_entry_t;

typedef pfmlib_event_t pfm_register_t;

#endif
