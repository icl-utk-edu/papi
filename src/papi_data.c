/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_data.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Haihang You
*	   you@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    <your name here>
*          <your email address>
*/

#ifndef NO_LIBPAPI

#include "papi.h"
#include "papi_internal.h"
#include <string.h>

int init_retval = DEADBEEF;
int init_level = PAPI_NOT_INITED;

/********************/
/*  BEGIN GLOBALS   */
/********************/

/* NEVER EVER STATICALLY ASSIGN VARIABLES THAT MAY BE CHANGED AT RUNTIME
   THIS BREAKS FORK AND LIBRARY PRELOADING, EVERY NON-CONST ITEM HERE
   SHOULD BE INITIALIZED INSIDE OF papi_internal.c */

#ifdef DEBUG
int _papi_hwi_debug;
#endif

/* Machine dependent info structure */
papi_mdi_t _papi_hwi_system_info;

/* Various preset items, why they are separate members no-one knows */
hwi_presets_t _papi_hwi_presets;

/* table matching derived types to derived strings.
   used by get_info, encode_event, xml translator
*/
const hwi_describe_t _papi_hwi_derived[] = {
	{NOT_DERIVED, "NOT_DERIVED", "Do nothing"},
	{DERIVED_ADD, "DERIVED_ADD", "Add counters"},
	{DERIVED_PS, "DERIVED_PS",
	 "Divide by the cycle counter and convert to seconds"},
	{DERIVED_ADD_PS, "DERIVED_ADD_PS",
	 "Add 2 counters then divide by the cycle counter and xl8 to secs."},
	{DERIVED_CMPD, "DERIVED_CMPD",
	 "Event lives in first counter but takes 2 or more codes"},
	{DERIVED_SUB, "DERIVED_SUB", "Sub all counters from first counter"},
	{DERIVED_POSTFIX, "DERIVED_POSTFIX",
	 "Process counters based on specified postfix string"},
	{-1, NULL, NULL}
};

#endif /* NO_LIBPAPI */

const hwi_preset_info_t _papi_hwi_preset_info[PAPI_MAX_PRESET_EVENTS] = {
	/*  0 */ {"PAPI_L1_DCM", "L1D cache misses", "Level 1 data cache misses"},
	/*  1 */ {"PAPI_L1_ICM", "L1I cache misses",
			  "Level 1 instruction cache misses"},
	/*  2 */ {"PAPI_L2_DCM", "L2D cache misses", "Level 2 data cache misses"},
	/*  3 */ {"PAPI_L2_ICM", "L2I cache misses",
			  "Level 2 instruction cache misses"},
	/*  4 */ {"PAPI_L3_DCM", "L3D cache misses", "Level 3 data cache misses"},
	/*  5 */ {"PAPI_L3_ICM", "L3I cache misses",
			  "Level 3 instruction cache misses"},
	/*  6 */ {"PAPI_L1_TCM", "L1 cache misses", "Level 1 cache misses"},
	/*  7 */ {"PAPI_L2_TCM", "L2 cache misses", "Level 2 cache misses"},
	/*  8 */ {"PAPI_L3_TCM", "L3 cache misses", "Level 3 cache misses"},
	/*  9 */ {"PAPI_CA_SNP", "Snoop Requests", "Requests for a snoop"},
	/* 10 */ {"PAPI_CA_SHR", "Ex Acces shared CL",
			  "Requests for exclusive access to shared cache line"},
	/* 11 */ {"PAPI_CA_CLN", "Ex Access clean CL",
			  "Requests for exclusive access to clean cache line"},
	/* 12 */ {"PAPI_CA_INV", "Cache ln invalid",
			  "Requests for cache line invalidation"},
	/* 13 */ {"PAPI_CA_ITV", "Cache ln intervene",
			  "Requests for cache line intervention"},
	/* 14 */ {"PAPI_L3_LDM", "L3 load misses", "Level 3 load misses"},
	/* 15 */ {"PAPI_L3_STM", "L3 store misses", "Level 3 store misses"},
	/* 16 */ {"PAPI_BRU_IDL", "Branch idle cycles",
			  "Cycles branch units are idle"},
	/* 17 */ {"PAPI_FXU_IDL", "IU idle cycles",
			  "Cycles integer units are idle"},
	/* 18 */ {"PAPI_FPU_IDL", "FPU idle cycles",
			  "Cycles floating point units are idle"},
	/* 19 */ {"PAPI_LSU_IDL", "L/SU idle cycles",
			  "Cycles load/store units are idle"},
	/* 20 */ {"PAPI_TLB_DM", "Data TLB misses",
			  "Data translation lookaside buffer misses"},
	/* 21 */ {"PAPI_TLB_IM", "Instr TLB misses",
			  "Instruction translation lookaside buffer misses"},
	/* 22 */ {"PAPI_TLB_TL", "Total TLB misses",
			  "Total translation lookaside buffer misses"},
	/* 23 */ {"PAPI_L1_LDM", "L1 load misses", "Level 1 load misses"},
	/* 24 */ {"PAPI_L1_STM", "L1 store misses", "Level 1 store misses"},
	/* 25 */ {"PAPI_L2_LDM", "L2 load misses", "Level 2 load misses"},
	/* 26 */ {"PAPI_L2_STM", "L2 store misses", "Level 2 store misses"},
	/* 27 */ {"PAPI_BTAC_M", "Br targt addr miss",
			  "Branch target address cache misses"},
	/* 28 */ {"PAPI_PRF_DM", "Data prefetch miss",
			  "Data prefetch cache misses"},
	/* 29 */ {"PAPI_L3_DCH", "L3D cache hits", "Level 3 data cache hits"},
	/* 30 */ {"PAPI_TLB_SD", "TLB shootdowns",
			  "Translation lookaside buffer shootdowns"},
	/* 31 */ {"PAPI_CSR_FAL", "Failed store cond",
			  "Failed store conditional instructions"},
	/* 32 */ {"PAPI_CSR_SUC", "Good store cond",
			  "Successful store conditional instructions"},
	/* 33 */ {"PAPI_CSR_TOT", "Total store cond",
			  "Total store conditional instructions"},
	/* 34 */ {"PAPI_MEM_SCY", "Stalled mem cycles",
			  "Cycles Stalled Waiting for memory accesses"},
	/* 35 */ {"PAPI_MEM_RCY", "Stalled rd cycles",
			  "Cycles Stalled Waiting for memory Reads"},
	/* 36 */ {"PAPI_MEM_WCY", "Stalled wr cycles",
			  "Cycles Stalled Waiting for memory writes"},
	/* 37 */ {"PAPI_STL_ICY", "No instr issue",
			  "Cycles with no instruction issue"},
	/* 38 */ {"PAPI_FUL_ICY", "Max instr issue",
			  "Cycles with maximum instruction issue"},
	/* 39 */ {"PAPI_STL_CCY", "No instr done",
			  "Cycles with no instructions completed"},
	/* 40 */ {"PAPI_FUL_CCY", "Max instr done",
			  "Cycles with maximum instructions completed"},
	/* 41 */ {"PAPI_HW_INT", "Hdw interrupts", "Hardware interrupts"},
	/* 42 */ {"PAPI_BR_UCN", "Uncond branch",
			  "Unconditional branch instructions"},
	/* 43 */ {"PAPI_BR_CN", "Cond branch", "Conditional branch instructions"},
	/* 44 */ {"PAPI_BR_TKN", "Cond branch taken",
			  "Conditional branch instructions taken"},
	/* 45 */ {"PAPI_BR_NTK", "Cond br not taken",
			  "Conditional branch instructions not taken"},
	/* 46 */ {"PAPI_BR_MSP", "Cond br mspredictd",
			  "Conditional branch instructions mispredicted"},
	/* 47 */ {"PAPI_BR_PRC", "Cond br predicted",
			  "Conditional branch instructions correctly predicted"},
	/* 48 */ {"PAPI_FMA_INS", "FMAs completed", "FMA instructions completed"},
	/* 49 */ {"PAPI_TOT_IIS", "Instr issued", "Instructions issued"},
	/* 50 */ {"PAPI_TOT_INS", "Instr completed", "Instructions completed"},
	/* 51 */ {"PAPI_INT_INS", "Int instructions", "Integer instructions"},
	/* 52 */ {"PAPI_FP_INS", "FP instructions", "Floating point instructions"},
	/* 53 */ {"PAPI_LD_INS", "Loads", "Load instructions"},
	/* 54 */ {"PAPI_SR_INS", "Stores", "Store instructions"},
	/* 55 */ {"PAPI_BR_INS", "Branches", "Branch instructions"},
	/* 56 */ {"PAPI_VEC_INS", "Vector/SIMD instr",
			  "Vector/SIMD instructions (could include integer)"},
	/* 57 */ {"PAPI_RES_STL", "Stalled res cycles",
			  "Cycles stalled on any resource"},
	/* 58 */ {"PAPI_FP_STAL", "Stalled FPU cycles",
			  "Cycles the FP unit(s) are stalled"},
	/* 59 */ {"PAPI_TOT_CYC", "Total cycles", "Total cycles"},
	/* 60 */ {"PAPI_LST_INS", "L/S completed",
			  "Load/store instructions completed"},
	/* 61 */ {"PAPI_SYC_INS", "Syncs completed",
			  "Synchronization instructions completed"},
	/* 62 */ {"PAPI_L1_DCH", "L1D cache hits", "Level 1 data cache hits"},
	/* 63 */ {"PAPI_L2_DCH", "L2D cache hits", "Level 2 data cache hits"},
	/* 64 */ {"PAPI_L1_DCA", "L1D cache accesses",
			  "Level 1 data cache accesses"},
	/* 65 */ {"PAPI_L2_DCA", "L2D cache accesses",
			  "Level 2 data cache accesses"},
	/* 66 */ {"PAPI_L3_DCA", "L3D cache accesses",
			  "Level 3 data cache accesses"},
	/* 67 */ {"PAPI_L1_DCR", "L1D cache reads", "Level 1 data cache reads"},
	/* 68 */ {"PAPI_L2_DCR", "L2D cache reads", "Level 2 data cache reads"},
	/* 69 */ {"PAPI_L3_DCR", "L3D cache reads", "Level 3 data cache reads"},
	/* 70 */ {"PAPI_L1_DCW", "L1D cache writes", "Level 1 data cache writes"},
	/* 71 */ {"PAPI_L2_DCW", "L2D cache writes", "Level 2 data cache writes"},
	/* 72 */ {"PAPI_L3_DCW", "L3D cache writes", "Level 3 data cache writes"},
	/* 73 */ {"PAPI_L1_ICH", "L1I cache hits",
			  "Level 1 instruction cache hits"},
	/* 74 */ {"PAPI_L2_ICH", "L2I cache hits",
			  "Level 2 instruction cache hits"},
	/* 75 */ {"PAPI_L3_ICH", "L3I cache hits",
			  "Level 3 instruction cache hits"},
	/* 76 */ {"PAPI_L1_ICA", "L1I cache accesses",
			  "Level 1 instruction cache accesses"},
	/* 77 */ {"PAPI_L2_ICA", "L2I cache accesses",
			  "Level 2 instruction cache accesses"},
	/* 78 */ {"PAPI_L3_ICA", "L3I cache accesses",
			  "Level 3 instruction cache accesses"},
	/* 79 */ {"PAPI_L1_ICR", "L1I cache reads",
			  "Level 1 instruction cache reads"},
	/* 80 */ {"PAPI_L2_ICR", "L2I cache reads",
			  "Level 2 instruction cache reads"},
	/* 81 */ {"PAPI_L3_ICR", "L3I cache reads",
			  "Level 3 instruction cache reads"},
	/* 82 */ {"PAPI_L1_ICW", "L1I cache writes",
			  "Level 1 instruction cache writes"},
	/* 83 */ {"PAPI_L2_ICW", "L2I cache writes",
			  "Level 2 instruction cache writes"},
	/* 84 */ {"PAPI_L3_ICW", "L3I cache writes",
			  "Level 3 instruction cache writes"},
	/* 85 */ {"PAPI_L1_TCH", "L1 cache hits", "Level 1 total cache hits"},
	/* 86 */ {"PAPI_L2_TCH", "L2 cache hits", "Level 2 total cache hits"},
	/* 87 */ {"PAPI_L3_TCH", "L3 cache hits", "Level 3 total cache hits"},
	/* 88 */ {"PAPI_L1_TCA", "L1 cache accesses",
			  "Level 1 total cache accesses"},
	/* 89 */ {"PAPI_L2_TCA", "L2 cache accesses",
			  "Level 2 total cache accesses"},
	/* 90 */ {"PAPI_L3_TCA", "L3 cache accesses",
			  "Level 3 total cache accesses"},
	/* 91 */ {"PAPI_L1_TCR", "L1 cache reads", "Level 1 total cache reads"},
	/* 92 */ {"PAPI_L2_TCR", "L2 cache reads", "Level 2 total cache reads"},
	/* 93 */ {"PAPI_L3_TCR", "L3 cache reads", "Level 3 total cache reads"},
	/* 94 */ {"PAPI_L1_TCW", "L1 cache writes", "Level 1 total cache writes"},
	/* 95 */ {"PAPI_L2_TCW", "L2 cache writes", "Level 2 total cache writes"},
	/* 96 */ {"PAPI_L3_TCW", "L3 cache writes", "Level 3 total cache writes"},
	/* 97 */ {"PAPI_FML_INS", "FPU multiply",
			  "Floating point multiply instructions"},
	/* 98 */ {"PAPI_FAD_INS", "FPU add", "Floating point add instructions"},
	/* 99 */ {"PAPI_FDV_INS", "FPU divide",
			  "Floating point divide instructions"},
	/*100 */ {"PAPI_FSQ_INS", "FPU square root",
			  "Floating point square root instructions"},
	/*101 */ {"PAPI_FNV_INS", "FPU inverse",
			  "Floating point inverse instructions"},
	/*102 */ {"PAPI_FP_OPS", "FP operations", "Floating point operations"},
	/*103 */ {"PAPI_SP_OPS", "SP operations",
			  "Floating point operations; optimized to count scaled single precision vector operations"},
	/*104 */ {"PAPI_DP_OPS", "DP operations",
			  "Floating point operations; optimized to count scaled double precision vector operations"},
	/*105 */ {"PAPI_VEC_SP", "SP Vector/SIMD instr",
			  "Single precision vector/SIMD instructions"},
	/*106 */ {"PAPI_VEC_DP", "DP Vector/SIMD instr",
			  "Double precision vector/SIMD instructions"},
	/* empty entries are now null pointers instead of pointers to empty strings */
#if defined (_BGL)
	/*107 */ {"PAPI_BGL_OED", "Oedipus operations",
			  "BGL special event: Oedipus operations"},
	/*108 */ {"PAPI_BGL_TS_32B", "Torus 32B chunks sent",
			  "BGL special event: Torus 32B chunks sent"},
	/*109 */ {"PAPI_BGL_TS_FULL", "Torus no token UPC cycles",
			  "BGL special event: Torus no token UPC cycles"},
	/*110 */ {"PAPI_BGL_TR_DPKT", "Tree 256 byte packets",
			  "BGL special event: Tree 256 byte packets"},
	/*111 */ {"PAPI_BGL_TR_FULL", "UPC cycles (CLOCKx2) tree rcv is full",
			  "BGL special event: UPC cycles (CLOCKx2) tree rcv is full"},
#elif defined (__bgp__)
	/*107 */ {"PAPI_BGL_TS_32B", "Torus 32B chunks sent",
			  "BGL special event: Torus 32B chunks sent"},
	/*108 */ {"PAPI_BGL_TR_DPKT", "Tree 256 byte packets",
			  "BGL special event: Tree 256 byte packets"},
	/*109 */ {NULL, NULL, NULL},
	/*110 */ {NULL, NULL, NULL},
	/*111 */ {NULL, NULL, NULL},
#else
	/*107 */ {NULL, NULL, NULL},
	/*108 */ {NULL, NULL, NULL},
	/*109 */ {NULL, NULL, NULL},
	/*110 */ {NULL, NULL, NULL},
	/*111 */ {NULL, NULL, NULL},
#endif
	/*112 */ {NULL, NULL, NULL},
	/*113 */ {NULL, NULL, NULL},
	/*114 */ {NULL, NULL, NULL},
	/*115 */ {NULL, NULL, NULL},
	/*116 */ {NULL, NULL, NULL},
	/*117 */ {NULL, NULL, NULL},
	/*118 */ {NULL, NULL, NULL},
	/*119 */ {NULL, NULL, NULL},
	/*120 */ {NULL, NULL, NULL},
	/*121 */ {NULL, NULL, NULL},
	/*122 */ {NULL, NULL, NULL},
	/*123 */ {NULL, NULL, NULL},
	/*124 */ {NULL, NULL, NULL},
	/*125 */ {NULL, NULL, NULL},
	/*126 */ {NULL, NULL, NULL},
	/*127 */ {NULL, NULL, NULL}
};

const unsigned int _papi_hwi_preset_type[] = {
	/*  0: PAPI_L1_DCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/*  1: PAPI_L1_ICM */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
	/*  2: PAPI_L2_DCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/*  3: PAPI_L2_ICM */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
	/*  4: PAPI_L3_DCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/*  5: PAPI_L3_ICM */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
	/*  6: PAPI_L1_TCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/*  7: PAPI_L2_TCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/*  8: PAPI_L3_TCM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/*  9: PAPI_CA_SNP */ PAPI_PRESET_BIT_CACH,
	/* 10: PAPI_CA_SHR */ PAPI_PRESET_BIT_CACH,
	/* 11: PAPI_CA_CLN */ PAPI_PRESET_BIT_CACH,
	/* 12: PAPI_CA_INV */ PAPI_PRESET_BIT_CACH,
	/* 13: PAPI_CA_ITV */ PAPI_PRESET_BIT_CACH,
	/* 14: PAPI_L3_LDM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 15: PAPI_L3_STM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 16: PAPI_BRU_IDL */ PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_BR,
	/* 17: PAPI_FXU_IDL */ PAPI_PRESET_BIT_IDL,
	/* 18: PAPI_FPU_IDL */ PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_FP,
	/* 19: PAPI_LSU_IDL */ PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_MEM,
	/* 20: PAPI_TLB_DM */ PAPI_PRESET_BIT_TLB,
	/* 21: PAPI_TLB_IM */ PAPI_PRESET_BIT_TLB + PAPI_PRESET_BIT_INS,
	/* 22: PAPI_TLB_TL */ PAPI_PRESET_BIT_TLB,
	/* 23: PAPI_L1_LDM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 24: PAPI_L1_STM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 25: PAPI_L2_LDM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 26: PAPI_L2_STM */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 27: PAPI_BTAC_M */ PAPI_PRESET_BIT_BR,
	/* 28: PAPI_PRF_DM */ PAPI_PRESET_BIT_CACH,
	/* 29: PAPI_L3_DCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 30: PAPI_TLB_SD */ PAPI_PRESET_BIT_TLB,
	/* 31: PAPI_CSR_FAL */ PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
	/* 32: PAPI_CSR_SUC */ PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
	/* 33: PAPI_CSR_TOT */ PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
	/* 34: PAPI_MEM_SCY */ PAPI_PRESET_BIT_MEM,
	/* 35: PAPI_MEM_RCY */ PAPI_PRESET_BIT_MEM,
	/* 36: PAPI_MEM_WCY */ PAPI_PRESET_BIT_MEM,
	/* 37: PAPI_STL_ICY */ PAPI_PRESET_BIT_INS,
	/* 38: PAPI_FUL_ICY */ PAPI_PRESET_BIT_INS,
	/* 39: PAPI_STL_CCY */ PAPI_PRESET_BIT_INS,
	/* 40: PAPI_FUL_CCY */ PAPI_PRESET_BIT_INS,
	/* 41: PAPI_HW_INT */ PAPI_PRESET_BIT_MSC,
	/* 42: PAPI_BR_UCN */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 43: PAPI_BR_CN */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 44: PAPI_BR_TKN */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 45: PAPI_BR_NTK */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 46: PAPI_BR_MSP */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 47: PAPI_BR_PRC */ PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
	/* 48: PAPI_FMA_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/* 49: PAPI_TOT_IIS */ PAPI_PRESET_BIT_INS,
	/* 50: PAPI_TOT_INS */ PAPI_PRESET_BIT_INS,
	/* 51: PAPI_INT_INS */ PAPI_PRESET_BIT_INS,
	/* 52: PAPI_FP_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/* 53: PAPI_LD_INS */ PAPI_PRESET_BIT_MEM,
	/* 54: PAPI_SR_INS */ PAPI_PRESET_BIT_MEM,
	/* 55: PAPI_BR_INS */ PAPI_PRESET_BIT_BR,
	/* 56: PAPI_VEC_INS */ PAPI_PRESET_BIT_MSC,
	/* 57: PAPI_RES_STL */ PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_MSC,
	/* 58: PAPI_FP_STAL */ PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_FP,
	/* 59: PAPI_TOT_CYC */ PAPI_PRESET_BIT_MSC,
	/* 60: PAPI_LST_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_MEM,
	/* 61: PAPI_SYC_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_MSC,
	/* 62: PAPI_L1_DCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 63: PAPI_L2_DCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 64: PAPI_L1_DCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 65: PAPI_L2_DCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 66: PAPI_L3_DCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 67: PAPI_L1_DCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 68: PAPI_L2_DCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 69: PAPI_L3_DCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 70: PAPI_L1_DCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 71: PAPI_L2_DCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 72: PAPI_L3_DCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 73: PAPI_L1_ICH */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
	/* 74: PAPI_L2_ICH */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
	/* 75: PAPI_L3_ICH */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
	/* 76: PAPI_L1_ICA */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
	/* 77: PAPI_L2_ICA */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
	/* 78: PAPI_L3_ICA */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
	/* 79: PAPI_L1_ICR */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
	/* 80: PAPI_L2_ICR */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
	/* 81: PAPI_L3_ICR */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
	/* 82: PAPI_L1_ICW */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
	/* 83: PAPI_L2_ICW */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
	/* 84: PAPI_L3_ICW */
		PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
	/* 85: PAPI_L1_TCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 86: PAPI_L2_TCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 87: PAPI_L3_TCH */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 88: PAPI_L1_TCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 89: PAPI_L2_TCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 90: PAPI_L3_TCA */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 91: PAPI_L1_TCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 92: PAPI_L2_TCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 93: PAPI_L3_TCR */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 94: PAPI_L1_TCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
	/* 95: PAPI_L2_TCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
	/* 96: PAPI_L3_TCW */ PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
	/* 97: PAPI_FML_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/* 98: PAPI_FAD_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/* 99: PAPI_FDV_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*100: PAPI_FSQ_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*101: PAPI_FNV_INS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*102: PAPI_FP_OPS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*103: PAPI_SP_OPS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*104: PAPI_DP_OPS */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*105: PAPI_VEC_SP */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/*106: PAPI_VEC_DP */ PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
	/* empty entries are now null pointers instead of pointers to empty strings */
#if defined (_BGL)
	/*107: PAPI_BGL_OED */ PAPI_PRESET_BIT_MSC,
	/*108: PAPI_BGL_TS_32B */ PAPI_PRESET_BIT_MSC,
	/*109: PAPI_BGL_TS_FULL */ PAPI_PRESET_BIT_MSC,
	/*110: PAPI_BGL_TR_DPKT */ PAPI_PRESET_BIT_MSC,
	/*111: PAPI_BGL_TR_FULL */ PAPI_PRESET_BIT_MSC,
#elif defined (__bgp__)
	/*107: PAPI_BGL_TS_32B */ PAPI_PRESET_BIT_MSC,
	/*108: PAPI_BGL_TR_DPKT */ PAPI_PRESET_BIT_MSC,
	/*109 */ 0,
	/*110 */ 0,
	/*111 */ 0,
#else
	/*107 */ 0,
	/*108 */ 0,
	/*109 */ 0,
	/*110 */ 0,
	/*111 */ 0,
#endif
	/*112 */ 0,
	/*113 */ 0,
	/*114 */ 0,
	/*115 */ 0,
	/*116 */ 0,
	/*117 */ 0,
	/*118 */ 0,
	/*119 */ 0,
	/*120 */ 0,
	/*121 */ 0,
	/*122 */ 0,
	/*123 */ 0,
	/*124 */ 0,
	/*125 */ 0,
	/*126 */ 0,
	/*127 */ 0
};

const hwi_describe_t _papi_hwi_err[PAPI_NUM_ERRORS] = {
	/* 0 */ {PAPI_OK, "PAPI_OK", "No error"},
	/* 1 */ {PAPI_EINVAL, "PAPI_EINVAL", "Invalid argument"},
	/* 2 */ {PAPI_ENOMEM, "PAPI_ENOMEM", "Insufficient memory"},
	/* 3 */ {PAPI_ESYS, "PAPI_ESYS", "A System/C library call failed"},
	/* 4 */ {PAPI_ESBSTR, "PAPI_ESBSTR", "Not supported by substrate"},
	/* 5 */ {PAPI_ECLOST, "PAPI_ECLOST",
			 "Access to the counters was lost or interrupted"},
	/* 6 */ {PAPI_EBUG, "PAPI_EBUG",
			 "Internal error, please send mail to the developers"},
	/* 7 */ {PAPI_ENOEVNT, "PAPI_ENOEVNT", "Event does not exist"},
	/* 8 */ {PAPI_ECNFLCT, "PAPI_ECNFLCT",
			 "Event exists, but cannot be counted due to hardware resource limits"},
	/* 9 */ {PAPI_ENOTRUN, "PAPI_ENOTRUN", "EventSet is currently not running"},
	/*10 */ {PAPI_EISRUN, "PAPI_EISRUN", "EventSet is currently counting"},
	/*11 */ {PAPI_ENOEVST, "PAPI_ENOEVST", "No such EventSet available"},
	/*12 */ {PAPI_ENOTPRESET, "PAPI_ENOTPRESET",
			 "Event in argument is not a valid preset"},
	/*13 */ {PAPI_ENOCNTR, "PAPI_ENOCNTR",
			 "Hardware does not support performance counters"},
	/*14 */ {PAPI_EMISC, "PAPI_EMISC", "Unknown error code"},
	/*15 */ {PAPI_EPERM, "PAPI_EPERM",
			 "Permission level does not permit operation"},
	/*16 */ {PAPI_ENOINIT, "PAPI_ENOINIT", "PAPI hasn't been initialized yet"},
	/*17 */ {PAPI_ENOCMP, "PAPI_ENOCMP", "Component Index isn't set"},
    /*18 */ {PAPI_ENOSUPP, "PAPI_ENOSUPP", "Not supported"},
    /*19 */ {PAPI_ENOIMPL, "PAPI_ENOIMPL", "Not implemented"},
    /*20 */ {PAPI_EBUF, "PAPI_EBUF", "Buffer size exceeded"},
    /*21 */ {PAPI_EINVAL_DOM, "PAPI_EINVAL_DOM", "EventSet domain is not supported for the operation"}
};

#ifndef NO_LIBPAPI

/* _papi_hwi_derived_type:
   Helper routine to extract a derived type from a derived string
   returns type value if found, otherwise returns -1
*/
int
_papi_hwi_derived_type( char *tmp, int *code )
{
	int i = 0;
	while ( _papi_hwi_derived[i].name != NULL ) {
		if ( strcasecmp( tmp, _papi_hwi_derived[i].name ) == 0 ) {
			*code = _papi_hwi_derived[i].value;
			return ( PAPI_OK );
		}
		i++;
	}
	INTDBG( "Invalid derived string %s\n", tmp );
	return ( PAPI_EINVAL );
}

/* _papi_hwi_derived_string:
   Helper routine to extract a derived string from a derived type
   copies derived type string into derived if found,
   otherwise returns PAPI_EINVAL
*/
int
_papi_hwi_derived_string( int type, char *derived, int len )
{
	int j;

	for ( j = 0; _papi_hwi_derived[j].value != -1; j++ ) {
		if ( _papi_hwi_derived[j].value == type ) {
			strncpy( derived, _papi_hwi_derived[j].name, ( size_t ) len );
			return ( PAPI_OK );
		}
	}
	INTDBG( "Invalid derived type %d\n", type );
	return ( PAPI_EINVAL );
}

#endif /* NO_LIBPAPI */


/********************/
/*    END GLOBALS   */
/********************/
