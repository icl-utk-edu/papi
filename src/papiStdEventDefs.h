
/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful
in tuning application performance. These events have identical
assignments in the header files on different platforms however they
may differ in their actual semantics. In addition, all of these events
are not guaranteed to be present on all platforms.  Please check your
platform's documentation carefully.

*/
#ifndef _PAPISTDEVENTDEFS
#define _PAPISTDEVENTDEFS

/*
   Masks to indicate the event is a preset- the presets will have 
   the high bit set to one, as the vendors probably won't use the 
   higher numbers for the native events 
   This causes a problem for signed ints on 64 bit systems, since the
   'high bit' is no longer the high bit. An alternative is to AND
   with PRESET_AND_MASK instead of XOR with PRESET_MASK to isolate
   the event bits.
   Native events for a specific platform can be defined by setting
   the next-highest bit. This gives PAPI a standardized way of 
   differentiating native events from preset events for query
   functions, etc.
*/

#define PRESET_MASK 0x80000000
#define NATIVE_MASK 0x40000000
#define PRESET_AND_MASK 0x7FFFFFFF
#define NATIVE_AND_MASK 0x3FFFFFFF

#define PAPI_MAX_PRESET_EVENTS 128      /*The maxmimum number of preset events */

/*
   NOTE: The table below defines each entry in terms of a mask and an integer.
   The integers MUST be in consecutive order with no gaps.
   If an event is removed or added, all following events MUST be renumbered.
   One way to fix this would be to recast each #define in terms of the preceeding
   one instead of an absolute number. e.g.:
     #define PAPI_L1_ICM  (PAPI_L1_DCM + 1)
   That way inserting or deleting events would only affect the definition of one
   other event.
*/

enum {
   PAPI_L1_DCM = PRESET_MASK, /*Level 1 data cache misses */
   PAPI_L1_ICM,  /*Level 1 instruction cache misses */
   PAPI_L2_DCM,  /*Level 2 data cache misses */
   PAPI_L2_ICM,  /*Level 2 instruction cache misses */
   PAPI_L3_DCM,  /*Level 3 data cache misses */
   PAPI_L3_ICM,  /*Level 3 instruction cache misses */
   PAPI_L1_TCM,  /*Level 1 total cache misses */
   PAPI_L2_TCM,  /*Level 2 total cache misses */
   PAPI_L3_TCM,  /*Level 3 total cache misses */
   PAPI_CA_SNP,  /*Snoops */
   PAPI_CA_SHR,  /*Request for shared cache line (SMP) */
   PAPI_CA_CLN,  /*Request for clean cache line (SMP) */
   PAPI_CA_INV,  /*Request for cache line Invalidation (SMP) */
   PAPI_CA_ITV,  /*Request for cache line Intervention (SMP) */
   PAPI_L3_LDM,  /*Level 3 load misses */
   PAPI_L3_STM,  /*Level 3 store misses */
   PAPI_BRU_IDL, /*Cycles branch units are idle */
   PAPI_FXU_IDL, /*Cycles integer units are idle */
   PAPI_FPU_IDL, /*Cycles floating point units are idle */
   PAPI_LSU_IDL, /*Cycles load/store units are idle */
   PAPI_TLB_DM,  /*Data translation lookaside buffer misses */
   PAPI_TLB_IM,  /*Instr translation lookaside buffer misses */
   PAPI_TLB_TL,  /*Total translation lookaside buffer misses */
   PAPI_L1_LDM,  /*Level 1 load misses */
   PAPI_L1_STM,  /*Level 1 store misses */
   PAPI_L2_LDM,  /*Level 2 load misses */
   PAPI_L2_STM,  /*Level 2 store misses */
   PAPI_BTAC_M,  /*BTAC miss */
   PAPI_PRF_DM,  /*Prefetch data instruction caused a miss */
   PAPI_L3_DCH,  /*Level 3 Data Cache Hit */
   PAPI_TLB_SD,  /*Xlation lookaside buffer shootdowns (SMP) */
   PAPI_CSR_FAL, /*Failed store conditional instructions */
   PAPI_CSR_SUC, /*Successful store conditional instructions */
   PAPI_CSR_TOT, /*Total store conditional instructions */
   PAPI_MEM_SCY, /*Cycles Stalled Waiting for Memory Access */
   PAPI_MEM_RCY, /*Cycles Stalled Waiting for Memory Read */
   PAPI_MEM_WCY, /*Cycles Stalled Waiting for Memory Write */
   PAPI_STL_ICY, /*Cycles with No Instruction Issue */
   PAPI_FUL_ICY, /*Cycles with Maximum Instruction Issue */
   PAPI_STL_CCY, /*Cycles with No Instruction Completion */
   PAPI_FUL_CCY, /*Cycles with Maximum Instruction Completion */
   PAPI_HW_INT,  /*Hardware interrupts */
   PAPI_BR_UCN,  /*Unconditional branch instructions executed */
   PAPI_BR_CN,   /*Conditional branch instructions executed */
   PAPI_BR_TKN,  /*Conditional branch instructions taken */
   PAPI_BR_NTK,  /*Conditional branch instructions not taken */
   PAPI_BR_MSP,  /*Conditional branch instructions mispred */
   PAPI_BR_PRC,  /*Conditional branch instructions corr. pred */
   PAPI_FMA_INS, /*FMA instructions completed */
   PAPI_TOT_IIS, /*Total instructions issued */
   PAPI_TOT_INS, /*Total instructions executed */
   PAPI_INT_INS, /*Integer instructions executed */
   PAPI_FP_INS,  /*Floating point instructions executed */
   PAPI_LD_INS,  /*Load instructions executed */
   PAPI_SR_INS,  /*Store instructions executed */
   PAPI_BR_INS,  /*Total branch instructions executed */
   PAPI_VEC_INS, /*Vector/SIMD instructions executed */
   PAPI_RES_STL, /*Cycles processor is stalled on resource */
   PAPI_FP_STAL, /*Cycles any FP units are stalled */
   PAPI_TOT_CYC, /*Total cycles */
   PAPI_LST_INS, /*Total load/store inst. executed */
   PAPI_SYC_INS, /*Sync. inst. executed */
   PAPI_L1_DCH,  /*L1 D Cache Hit */
   PAPI_L2_DCH,  /*L2 D Cache Hit */
   PAPI_L1_DCA,  /*L1 D Cache Access */
   PAPI_L2_DCA,  /*L2 D Cache Access */
   PAPI_L3_DCA,  /*L3 D Cache Access */
   PAPI_L1_DCR,  /*L1 D Cache Read */
   PAPI_L2_DCR,  /*L2 D Cache Read */
   PAPI_L3_DCR,  /*L3 D Cache Read */
   PAPI_L1_DCW,  /*L1 D Cache Write */
   PAPI_L2_DCW,  /*L2 D Cache Write */
   PAPI_L3_DCW,  /*L3 D Cache Write */
   PAPI_L1_ICH,  /*L1 instruction cache hits */
   PAPI_L2_ICH,  /*L2 instruction cache hits */
   PAPI_L3_ICH,  /*L3 instruction cache hits */
   PAPI_L1_ICA,  /*L1 instruction cache accesses */
   PAPI_L2_ICA,  /*L2 instruction cache accesses */
   PAPI_L3_ICA,  /*L3 instruction cache accesses */
   PAPI_L1_ICR,  /*L1 instruction cache reads */
   PAPI_L2_ICR,  /*L2 instruction cache reads */
   PAPI_L3_ICR,  /*L3 instruction cache reads */
   PAPI_L1_ICW,  /*L1 instruction cache writes */
   PAPI_L2_ICW,  /*L2 instruction cache writes */
   PAPI_L3_ICW,  /*L3 instruction cache writes */
   PAPI_L1_TCH,  /*L1 total cache hits */
   PAPI_L2_TCH,  /*L2 total cache hits */
   PAPI_L3_TCH,  /*L3 total cache hits */
   PAPI_L1_TCA,  /*L1 total cache accesses */
   PAPI_L2_TCA,  /*L2 total cache accesses */
   PAPI_L3_TCA,  /*L3 total cache accesses */
   PAPI_L1_TCR,  /*L1 total cache reads */
   PAPI_L2_TCR,  /*L2 total cache reads */
   PAPI_L3_TCR,  /*L3 total cache reads */
   PAPI_L1_TCW,  /*L1 total cache writes */
   PAPI_L2_TCW,  /*L2 total cache writes */
   PAPI_L3_TCW,  /*L3 total cache writes */
   PAPI_FML_INS, /*FM ins */
   PAPI_FAD_INS, /*FA ins */
   PAPI_FDV_INS, /*FD ins */
   PAPI_FSQ_INS, /*FSq ins */
   PAPI_FNV_INS, /*Finv ins */
   PAPI_FP_OPS,  /*Floating point operations executed */
};

#endif

