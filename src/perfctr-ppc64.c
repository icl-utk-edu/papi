/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    perfctr-ppc64.c
* CVS:     $Id$
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* PAPI stuff */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include SUBSTRATE

#ifdef PERFCTR26
#define PERFCTR_CPU_NAME   perfctr_info_cpu_name

#define PERFCTR_CPU_NRCTRS perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_cpu_nrctrs
#endif

static hwi_search_t preset_name_map_PPC64[PAPI_MAX_PRESET_EVENTS] = {
#if defined(_POWER5) || defined(_POWER5p) || defined(_POWER6)
   {PAPI_L1_DCM, {DERIVED_ADD, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 data cache misses */
   {PAPI_L1_DCA, {DERIVED_ADD, {PNE_PM_LD_REF_L1, PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Level 1 data cache access */
   /* can't count level 1 data cache hits due to hardware limitations. */
   {PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L1,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Level 1 load misses */
   {PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Level 1 store misses */
   {PAPI_L1_DCW, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 D cache write */
   {PAPI_L1_DCR, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 D cache read */
   /* can't count level 2 data cache reads due to hardware limitations. */
   /* can't count level 2 data cache hits due to hardware limitations. */
   {PAPI_L2_DCM, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 2 data cache misses */
   {PAPI_L2_LDM, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 2 cache read misses */
   {PAPI_L3_DCR, {0, {PNE_PM_DATA_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 3 data cache reads */
   /* can't count level 3 data cache hits due to hardware limitations. */
   {PAPI_L3_DCM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache misses (reads & writes) */
   {PAPI_L3_LDM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache read misses */
   /* can't count level 1 instruction cache accesses due to hardware limitations. */
   {PAPI_L1_ICH, {0, {PNE_PM_INST_FROM_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 1 inst cache hits */
#if defined(_POWER6)
   {PAPI_L1_ICM, {0, {PNE_PM_L1_ICACHE_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /* Level 1 instruction cache misses */
#else
   /* can't count level 1 instruction cache misses due to hardware limitations. */
#endif
   /* can't count level 2 instruction cache accesses due to hardware limitations. */
   {PAPI_L2_ICM, {0, {PNE_PM_INST_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 2 inst cache misses */
   {PAPI_L2_ICH, {0, {PNE_PM_INST_FROM_L2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 2 inst cache hits */
   {PAPI_L3_ICA, {0, {PNE_PM_INST_FROM_L2MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 3 inst cache accesses */
   {PAPI_L3_ICH, {0, {PNE_PM_INST_FROM_L3, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 2 inst cache hits */
#if defined(_POWER6)
   {PAPI_L3_ICM, {0, {PNE_PM_INST_FROM_L3MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},  /* Level 2 inst cache misses */
#else
   {PAPI_L3_ICM, {DERIVED_ADD, {PNE_PM_DATA_FROM_LMEM, PNE_PM_DATA_FROM_RMEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 instruction cache misses (reads & writes) */
#endif
   {PAPI_FMA_INS, {0, {PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},       /*FMA instructions completed */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Total instructions executed */
#if defined (_POWER6)
   {PAPI_INT_INS, {DERIVED_ADD, {PNE_PM_FXU0_FIN, PNE_PM_FXU1_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},       /*Integer instructions executed */
#else
   {PAPI_INT_INS, {0, {PNE_PM_FXU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},       /*Integer instructions executed */
#endif
   {PAPI_FP_OPS, {DERIVED_ADD, {PNE_PM_FPU_1FLOP, PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL},{0}}}, /*Floating point instructions executed */
   {PAPI_FP_INS, {0, {PNE_PM_FPU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Floating point instructions executed */
#if defined (_POWER6)
   /* Can't count FDIV and FSQRT instructions individually on POWER6 */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /*Processor cycles */
#else
   {PAPI_TOT_CYC, {0, {PNE_PM_RUN_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /*Processor cycles gated by the run latch */
   {PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*FD ins */
   {PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*FSq ins */
#endif
#if defined (_POWER6)
   /* POWER6 does not have a TLB */
#else
   {PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Data translation lookaside buffer misses */
   {PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Instr translation lookaside buffer misses */
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Total translation lookaside buffer misses */
#endif
   {PAPI_HW_INT, {0, {PNE_PM_EXT_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Hardware interrupts */
   {PAPI_STL_ICY, {0, {PNE_PM_0INST_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /*Cycles with No Instruction Issue */
   {PAPI_LD_INS, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Load instructions*/
   {PAPI_SR_INS, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Store instructions*/
   {PAPI_LST_INS, {DERIVED_ADD, {PNE_PM_ST_REF_L1, PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Load and Store instructions*/
#if defined (_POWER6)
   {PAPI_BR_INS, {0, {PNE_PM_BRU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branch instructions*/
   {PAPI_BR_MSP, {0, {PNE_PM_BR_MPRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branch mispredictions */
   {PAPI_BR_PRC, {0, {PNE_PM_BR_PRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branches correctly predicted */
#else
   {PAPI_BR_INS, {0, {PNE_PM_BR_ISSUED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branch instructions*/
   {PAPI_BR_MSP, {DERIVED_ADD, {PNE_PM_BR_MPRED_CR, PNE_PM_BR_MPRED_TA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branch mispredictions */
   {PAPI_BR_PRC, {0, {PNE_PM_BR_PRED_CR_TA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branches correctly predicted */
#endif
   {PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Cycles integer units are idle */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}        /* end of list */
#else
#ifdef _PPC970
   {PAPI_L2_DCM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 2 data cache misses */
   {PAPI_L2_DCR, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache read attempts */
   {PAPI_L2_DCH, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache hits */
   {PAPI_L2_LDM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache read misses */
   /* no PAPI_L1_ICA since PM_INST_FROM_L1 and PM_INST_FROM_L2 cannot be counted simultaneously. */
   {PAPI_L1_ICM, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 1 inst cache misses */
   {PAPI_L2_ICA, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache accesses */
   {PAPI_L2_ICH, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_SHR, PNE_PM_INST_FROM_L25_MOD, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache hits */
   {PAPI_L2_ICM, {0, {PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache misses */
#else
#ifdef _POWER4
   {PAPI_L2_DCR, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PNE_PM_DATA_FROM_L275_MOD, PNE_PM_DATA_FROM_L275_SHR, PNE_PM_DATA_FROM_L3, PNE_PM_DATA_FROM_L35, PNE_PM_DATA_FROM_MEM}, {0}}}, /* Level 2 data cache read attemptss */
   {PAPI_L2_DCH, {DERIVED_ADD, {PNE_PM_DATA_FROM_L2, PNE_PM_DATA_FROM_L25_MOD, PNE_PM_DATA_FROM_L25_SHR, PNE_PM_DATA_FROM_L275_MOD, PNE_PM_DATA_FROM_L275_SHR, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache hits */
   {PAPI_L2_DCM, {DERIVED_ADD, {PNE_PM_DATA_FROM_L3, PNE_PM_DATA_FROM_L35, PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache misses (reads & writes) */
   {PAPI_L2_LDM, {DERIVED_ADD, {PNE_PM_DATA_FROM_L3, PNE_PM_DATA_FROM_L35, PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 data cache read misses */
   /* no PAPI_L3_STM, PAPI_L3_DCW nor PAPI_L3_DCA since stores/writes to L3 aren't countable */
   {PAPI_L3_DCR, {DERIVED_ADD, {PNE_PM_DATA_FROM_L3, PNE_PM_DATA_FROM_L35, PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache reads */
   {PAPI_L3_DCH, {DERIVED_ADD, {PNE_PM_DATA_FROM_L3, PNE_PM_DATA_FROM_L35, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache hits */
   {PAPI_L1_ICA, {DERIVED_ADD, {PNE_PM_INST_FROM_L1, PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_L275, PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 1 inst cache accesses */
   {PAPI_L1_ICM, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_L275, PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 1 inst cache misses */
   {PAPI_L2_ICA, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_L275, PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache accesses */
   {PAPI_L2_ICH, {DERIVED_ADD, {PNE_PM_INST_FROM_L2, PNE_PM_INST_FROM_L25_L275, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache hits */
   {PAPI_L2_ICM, {DERIVED_ADD, {PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 2 inst cache misses */
   {PAPI_L3_ICA, {DERIVED_ADD, {PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 inst cache accesses */
   {PAPI_L3_ICH, {DERIVED_ADD, {PNE_PM_INST_FROM_L3, PNE_PM_INST_FROM_L35, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 inst cache hits */
#endif
#endif
/* Common preset events for Power4 and PPC970 */
   {PAPI_L1_DCM, {DERIVED_ADD, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 data cache misses */
   {PAPI_L1_DCA, {DERIVED_ADD, {PNE_PM_LD_REF_L1, PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Level 1 data cache access */
   {PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Cycles integer units are idle */
   {PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L1,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Level 1 load misses */
   {PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Level 1 store misses */
   {PAPI_L1_DCW, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 D cache write */
   {PAPI_L1_DCR, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Level 1 D cache read */
   {PAPI_FMA_INS, {0, {PNE_PM_FPU_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},       /*FMA instructions completed */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*Total instructions executed */
   {PAPI_INT_INS, {0, {PNE_PM_FXU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},       /*Integer instructions executed */
   {PAPI_FP_OPS, {DERIVED_POSTFIX, {PNE_PM_FPU0_FIN, PNE_PM_FPU1_FIN, PNE_PM_FPU_FMA, PNE_PM_FPU_STF, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|+|N2|+|N3|-|"}},      /*Floating point instructions executed */
   {PAPI_FP_INS, {0, {PNE_PM_FPU_FIN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Floating point instructions executed */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /*Total cycles */
   {PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*FD ins */
   {PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},     /*FSq ins */
   {PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Data translation lookaside buffer misses */
   {PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Instr translation lookaside buffer misses */
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS,PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Total translation lookaside buffer misses */
   {PAPI_HW_INT, {0, {PNE_PM_EXT_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},        /*Hardware interrupts */
   {PAPI_STL_ICY, {0, {PNE_PM_0INST_FETCH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /*Cycles with No Instruction Issue */
   {PAPI_LD_INS, {0, {PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Load instructions*/
   {PAPI_SR_INS, {0, {PNE_PM_ST_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Store instructions*/
   {PAPI_LST_INS, {DERIVED_ADD, {PNE_PM_ST_REF_L1, PNE_PM_LD_REF_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},      /*Load and Store instructions*/
   {PAPI_BR_INS, {0, {PNE_PM_BR_ISSUED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}},   /* Branch instructions*/
   {PAPI_BR_MSP, {DERIVED_ADD, {PNE_PM_BR_MPRED_CR, PNE_PM_BR_MPRED_TA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Branch mispredictions */
   {PAPI_L1_DCH, {DERIVED_POSTFIX, {PNE_PM_LD_REF_L1, PNE_PM_LD_MISS_L1, PNE_PM_ST_REF_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, "N0|N1|-|N2|+|N3|-|"}}, /* Level 1 data cache hits */
   /* no PAPI_L2_STM, PAPI_L2_DCW nor PAPI_L2_DCA since stores/writes to L2 aren't countable */
   {PAPI_L3_DCM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache misses (reads & writes) */
   {PAPI_L3_LDM, {0, {PNE_PM_DATA_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 data cache read misses */
   {PAPI_L1_ICH, {0, {PNE_PM_INST_FROM_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 1 inst cache hits */
   {PAPI_L3_ICM, {0, {PNE_PM_INST_FROM_MEM, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}, /* Level 3 inst cache misses */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0}}}        /* end of list */
#endif
};
hwi_search_t *preset_search_map;

#if defined(_POWER5) || defined(_POWER5p) || defined(_POWER6)
unsigned long long pmc_sel_mask[NUM_COUNTER_MASKS] = {
	PMC1_SEL_MASK,
	PMC2_SEL_MASK,
	PMC3_SEL_MASK,
	PMC4_SEL_MASK
};
#else
unsigned long long pmc_sel_mask[NUM_COUNTER_MASKS] = {
	PMC1_SEL_MASK,
	PMC2_SEL_MASK,
	PMC3_SEL_MASK,
	PMC4_SEL_MASK,
	PMC5_SEL_MASK,
	PMC6_SEL_MASK,
	PMC7_SEL_MASK,
	PMC8_SEL_MASK,
	PMC8a_SEL_MASK
};
#endif


/* pmc_dom_vector exists because some counters (POWER6, counters 5 & 6) do
   not support being gated by the privilege level.  This array of bit
   vectors is a general purpose mechanism that allows us to state whether a
   given set of privilege levels are supported by a particular counter.  The
   array is indexed by the counter number and then within the unsigned word
   bit vector by the domain vector.  For example, to check counter 4 for
   kernel domain and user domain compatibility, we address pmc_dom_vector[4
   - 1] and then take the requested domain vector, in this case
   PAPI_DOM_KERNEL (0x2) + PAPI_DOM_USER (0x1) = 0x3, and shift a 1 left by
   that amount, giving a value of 0x0008.  This value is then AND'd with
   pmc_dom_vector[4 - 1].  If this result is non-zero, the requested domain
   is supported by that counter.  Note that the least significant bit is
   not set in any of these vectors because there is no domain corresponding
   to all zeros in the domain vector.
  
   PMC_INT_VECTOR defines which counters support an exception on overflow.
   On POWER6, only counters 1-4 support generating an exception on
   overflow.  */

#ifdef _POWER6
static unsigned pmc_dom_vector[MAX_COUNTERS] = {
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   /* Counters 5 & 6 support only PAPI_DOM_USER+PAPI_DOM_KERNEL+PAPI_DOM_SUPERVISOR */
   0x0800,
   0x0800
};
#define PMC_INT_VECTOR 0x0f
#elif defined(_POWER5) || defined(POWER5p)
static unsigned pmc_dom_vector[MAX_COUNTERS] = {
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e
};
#define PMC_INT_VECTOR 0x3f
#else
static unsigned pmc_dom_vector[MAX_COUNTERS] = {
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e,
   0x0f0e
};
#define PMC_INT_VECTOR 0xff
#endif

/* Defined in papi_data.c */
extern hwi_presets_t _papi_hwi_presets;

inline_static int pmc_domain_is_supported(int reg, int dom)
{
	return (pmc_dom_vector[reg] & (1 << (dom))) != 0; 
}


inline_static int pmc_interrupt_is_supported(reg)
{
	return (PMC_INT_VECTOR & (1 << (reg))) != 0;
}

static void clear_unused_pmcsel_bits(hwd_control_state_t * cntrl) {
	struct perfctr_cpu_control * cpu_ctl = &cntrl->control.cpu_control;
	int i;
	int num_used_counters = cpu_ctl->nractrs + cpu_ctl->nrictrs;
	unsigned int used_counters = 0x0;
	for (i = 0; i < num_used_counters; i++ ) {
		used_counters |= 1 << cpu_ctl->pmc_map[i];
	}
#if defined(_POWER5) || defined(_POWER5p)
        int freeze_pmc5_pmc6 = 0; /* for Power5 use only */
#endif

	for (i = 0; i < MAX_COUNTERS; i++) {
		unsigned int active_counter = ((1 << i) & used_counters);
		if (!active_counter) {
#if defined(_POWER5) || defined(_POWER5p)
			if (i > 3) 
				freeze_pmc5_pmc6++;
			else
				cpu_ctl->ppc64.mmcr1 &= pmc_sel_mask[i];			
#elif defined(_POWER6)
			if (i <= 3)
				cpu_ctl->ppc64.mmcr1 &= pmc_sel_mask[i];
#else		
			if (i < 2) {
				cpu_ctl->ppc64.mmcr0 &= pmc_sel_mask[i];
			} else {
				cpu_ctl->ppc64.mmcr1 &= pmc_sel_mask[i];
				if (i == (MAX_COUNTERS -1))
					cpu_ctl->ppc64.mmcra &= pmc_sel_mask[NUM_COUNTER_MASKS -1];
			}
#endif				
		}
	}
#if defined(_POWER5) || defined(_POWER5p)
	if (freeze_pmc5_pmc6 == 2)
		cpu_ctl->ppc64.mmcr0 |= PMC5_PMC6_FREEZE;
#endif	
}

static int internal_allocate_registers(EventSetInfo_t *ESI, int domain);

static int set_domain(EventSetInfo_t * ESI, unsigned int domain) 
{
   struct perfctr_cpu_control * cpu_ctl = &ESI->machdep.control.cpu_control;
   int num_used_counters = cpu_ctl->nractrs + cpu_ctl->nrictrs;
   int i, did = 0;

   /* Check for compatibility of the requested domain with the active
      counters */
   for (i = 0; i < num_used_counters; i++) {
      if (pmc_domain_is_supported(cpu_ctl->pmc_map[i], domain))
         continue;
      else {
         /* Make an attempt at remapping to counters that are capable of
            the requested domain. */
         if (internal_allocate_registers(ESI, domain)) {
            /* successful.  We don't need to continue the check or recheck
               because allocate_registers should not return success unless
               the domain is compatible */
            SUBDBG("Registers were reallocated successfully to account for new domain - %x\n", domain);
            break;
         } else {
            SUBDBG("Counter %d doesn't support the requested domain mask - %x\n", cpu_ctl->pmc_map[i] + 1, domain);
            return(PAPI_ECNFLCT);
         }
      }
   }

	/* A bit setting of '0' indicates "count this context".
	 * Start off by turning off counting for all contexts; 
	 * then, selectively re-enable.
	 */
	cpu_ctl->ppc64.mmcr0 |= PERF_USER | PERF_KERNEL | PERF_HYPERVISOR;
   if(domain & PAPI_DOM_USER) {
   	cpu_ctl->ppc64.mmcr0 &= ~PERF_USER;
      did = 1;
   }
   if(domain & PAPI_DOM_KERNEL) {
   	cpu_ctl->ppc64.mmcr0 &= ~PERF_KERNEL;
      did = 1;
   }
   if(domain & PAPI_DOM_SUPERVISOR) {
   	cpu_ctl->ppc64.mmcr0 &= ~PERF_HYPERVISOR;
      did = 1;
   }
   
   if(did) {
      return(PAPI_OK);
   } else {
      return(PAPI_EINVAL);
   }
	
}


//extern native_event_entry_t *native_table;
//extern hwi_search_t _papi_hwd_preset_map[];
extern papi_mdi_t _papi_hwi_system_info;

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control) {
  unsigned int i;

   SUBDBG("Control used:\n");
   SUBDBG("tsc_on\t\t\t%u\n", control->tsc_on);
   SUBDBG("nractrs\t\t\t%u\n", control->nractrs);
   SUBDBG("nrictrs\t\t\t%u\n", control->nrictrs);
   SUBDBG("mmcr0\t\t\t0x%X\n", control->ppc64.mmcr0);
   SUBDBG("mmcr1\t\t\t0x%llX\n", (unsigned long long) control->ppc64.mmcr1);
   SUBDBG("mmcra\t\t\t0x%X\n", control->ppc64.mmcra);
   
   for (i = 0; i < (control->nractrs + control->nrictrs); ++i) {
   	 SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
   	 if (control->ireset[i])
   	   SUBDBG("ireset[%d]\t%X\n", i, control->ireset[i]);
   }
   	
}
#endif


/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
int setup_ppc64_presets(int cputype) {	
   preset_search_map = preset_name_map_PPC64;
   return (_papi_hwi_setup_all_presets(preset_search_map, NULL));
}

/*called when an EventSet is allocated */
int _papi_hwd_init_control_state(hwd_control_state_t * ptr) {
   int retval;

   /* The set_domain function for ppc64 needs an EventSetInfo_t struct, not
      just a machdep struct.  In order to avoid modifying the
      hardware-independant caller (papi_internal.c), we do a bodge here to
      create an empty EventSetInfo_t, initialize it, then copy the machdep
      component to where ptr points.  */
   EventSetInfo_t ESI_temp;

   memset(&ESI_temp, 0x00, sizeof(EventSetInfo_t));

   int i = 0;
   for(i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      ESI_temp.machdep.control.cpu_control.pmc_map[i] = i;
   }
   ESI_temp.machdep.control.cpu_control.tsc_on = 1;
   if ((retval = set_domain(&ESI_temp, _papi_hwi_system_info.sub_info.default_domain)) != PAPI_OK) {
      SUBDBG("set_domain returned error - %d\n", retval);
		return retval;
   }
   memcpy(ptr, &ESI_temp.machdep, sizeof(hwd_control_state_t));

   return(PAPI_OK);
}

/* No longer needed if not implemented
int _papi_hwd_add_prog_event(hwd_control_state_t * state, unsigned int code, void *tmp, EventInfo_t *tmp2) {
   return (PAPI_ESBSTR);
} */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */


/* Called once per process. */
/* No longer needed if not implemented
int _papi_hwd_shutdown_global(void) {
   return (PAPI_OK);
} */


/* the following bpt functions are empty functions in POWER4 */
/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
/*
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
	return PAPI_OK;
}
*/
/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
/*
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
}
*/
/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
/*
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
	return PAPI_OK;
}
*/
/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
/*
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
	return PAPI_OK;
}
*/
/* this function recursively does Modified Bipartite Graph counter allocation 
     success  return 1
	 fail     return 0
*/
static int do_counter_allocation(ppc64_reg_alloc_t * event_list, int size, int domain, EventSetOverflowInfo_t *overflow)
{
   int i, j, k, m, group, retval;
   unsigned int map[GROUP_INTS], first_bit;
   hwd_register_t ovf_event_reg[MAX_COUNTERS];
   unsigned native_code;

   for (m = 0; m < overflow->event_counter; m++) {
      if (overflow->EventCode[m] & PAPI_PRESET_MASK) {
         /* Only non-derived events support overflow, so only the first native event is used */
         native_code = _papi_hwi_presets.data[overflow->EventCode[m] & PAPI_PRESET_AND_MASK]->native[0];
      } else {
         /* must be a native mask */
         native_code = overflow->EventCode[m] & PAPI_NATIVE_AND_MASK;
      }
      
      if ((retval = _papi_hwd_ntv_code_to_bits(native_code, &ovf_event_reg[m]))) {
         SUBDBG("_papi_hwd_ntv_code_to_bits returned error");
         return PAPI_EINVAL;
      }
   }

   for (i = 0; i < GROUP_INTS; i++) {
      map[i] = event_list[0].ra_group[i];
   }

   for (i = 1; i < size; i++) {
      for (j = 0; j < GROUP_INTS; j++) 
         map[j] &= event_list[i].ra_group[j];
   }

   k = 0;

	try:

   group = -1;
   for (; k < GROUP_INTS; k++) {
      if (map[k]) {
         first_bit = ffs(map[k]) - 1; /* first_bit [0,31] */
         group = first_bit + k * 32;
         break;
      }
   }

   if (group < 0)
      return group;             /* allocation fail */
   else {
      for (i = 0; i < size; i++) {
         event_list[i].ra_position = -1;
         for (j = 0; j < MAX_COUNTERS; j++) {
            if (event_list[i].ra_counter_cmd[j] >= 0
                && event_list[i].ra_counter_cmd[j] == group_map[group].counter_cmd[j]) {
               /* check that the ra_position can handle this event
                  set's domain */
               if (! pmc_domain_is_supported(j, domain))
                  /* Domain is not supported this counter.  Continue looking
                     for a counter that does. */
                  goto continue_searching;
               for (m = 0; m < overflow->event_counter; m++) {
                  /* does counter j support event code overflow->EventCode[m]? */
                  if ((ovf_event_reg[m].selector & (1 << j)) &&
                      (ovf_event_reg[m].counter_cmd[j] == group_map[group].counter_cmd[j])) {
                     /* it does.  Check to see if it supports generating an interrupt. */
                     if (! pmc_interrupt_is_supported(j)) 
                        /* Interrupt on overflow is not supported on this
                           counter.  Continue looking for a counter that
                           does. */
                        goto continue_searching;
                     else
                        /* we found a counter that supports interrupts */
                        break;
                  }
               }
               /* we found a match that passes the tests.  Go onto the next event. */
               event_list[i].ra_position = j;
               break;
            }
            continue_searching:
            /* null statement required after a label that is at the end of a block */
            ;
         }
         if (event_list[i].ra_position == -1) {
            /* No counter was found that supported the interrupt and domain
               requirements.  Go try another group */
            map[k] &= ~(1 << first_bit);
            goto try;
         }
      }
      return group;
   }
}



/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
/*
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
}
*/

/* Register allocation */
static int internal_allocate_registers(EventSetInfo_t *ESI, int domain) {
   hwd_control_state_t *this_state = &ESI->machdep;
   int i, j, natNum, index;
   ppc64_reg_alloc_t event_list[MAX_COUNTERS];
   int group;

   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for (i = 0; i < natNum; i++) {
      event_list[i].ra_position = -1;
      for (j = 0; j < MAX_COUNTERS; j++) {
         if ((index =
              native_name_map[ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK].index) <
             0)
            return 0;
         event_list[i].ra_counter_cmd[j] = native_table[index].resources.counter_cmd[j];
      }
      for (j = 0; j < GROUP_INTS; j++) {
         if ((index =
              native_name_map[ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK].index) <
             0)
            return 0;
         event_list[i].ra_group[j] = native_table[index].resources.group[j];
      }
   }
   if ((group = do_counter_allocation(event_list, natNum, domain, &ESI->overflow)) >= 0) {      /* successfully mapped */
      /* copy counter allocations info back into NativeInfoArray */
      this_state->group_id = group;
      for (i = 0; i < natNum; i++) {
//         ESI->NativeInfoArray[i].ni_position = event_list[i].ra_position;
        this_state->control.cpu_control.pmc_map[i] = event_list[i].ra_position;
         ESI->NativeInfoArray[i].ni_position = i;
      }
      /* update the control structure based on the NativeInfoArray */\
      SUBDBG("Group ID: %d\n", group);

      return 1;
   } else {
      return 0;
   }
}

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
	return internal_allocate_registers(ESI, ESI->domain.domain);
}


/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t *context) {

      
   this_state->control.cpu_control.nractrs = count - this_state->control.cpu_control.nrictrs;
   // save control state
   unsigned int save_mmcr0_ctlbits = PERF_CONTROL_MASK & this_state->control.cpu_control.ppc64.mmcr0;
 
   this_state->control.cpu_control.ppc64.mmcr0 = 
   	group_map[this_state->group_id].mmcr0 | save_mmcr0_ctlbits;

   unsigned long long mmcr1 = ((unsigned long long)group_map[this_state->group_id].mmcr1U) <<32;
   mmcr1 += group_map[this_state->group_id].mmcr1L;
   this_state->control.cpu_control.ppc64.mmcr1 = mmcr1;
   	
   this_state->control.cpu_control.ppc64.mmcra = 
   	group_map[this_state->group_id].mmcra;

   clear_unused_pmcsel_bits(this_state);   
   return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
/*   clear_unused_pmcsel_bits(this_state);   moved to update_control_state */
#ifdef DEBUG
   print_control(&state->control.cpu_control);
#endif
   if (state->rvperfctr != NULL) 
     {
       if((error = rvperfctr_control(state->rvperfctr, &state->control)) < 0) 
	 {
	   SUBDBG("rvperfctr_control returns: %d\n", error);
	   PAPIERROR(RCNTRL_ERROR); 
	   return(PAPI_ESYS); 
	 }
       return (PAPI_OK);
     }
   if((error = vperfctr_control(ctx->perfctr, &state->control)) < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      PAPIERROR(VCNTRL_ERROR); 
      return(PAPI_ESYS);      
   }
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if( state->rvperfctr != NULL ) {
     if(rvperfctr_stop((struct rvperfctr*)ctx->perfctr) < 0)
       { PAPIERROR( RCNTRL_ERROR); return(PAPI_ESYS); }
     return (PAPI_OK);
   }
   if(vperfctr_stop(ctx->perfctr) < 0)
     {PAPIERROR(VCNTRL_ERROR); return(PAPI_ESYS);}      
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long ** dp, int flags) {
   if (flags & PAPI_PAUSED) {
     vperfctr_read_state(ctx->perfctr, &spc->state, NULL);
   } else {
      SUBDBG("vperfctr_read_ctrs\n");
      if( spc->rvperfctr != NULL ) {
        rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
      } else {
        vperfctr_read_ctrs(ctx->perfctr, &spc->state);
        }
   }
       
   *dp = (long_long *) spc->state.pmc;
#ifdef DEBUG
   {
     if (ISLEVEL(DEBUG_SUBSTRATE)) {
         int i;
         for(i = 0; i < spc->control.cpu_control.nractrs + spc->control.cpu_control.nrictrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long_long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}


int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

/*
int _papi_hwd_setmaxmem() {
   return (PAPI_OK);
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from) {
   return(PAPI_ESBSTR);
}
*/

/* Perfctr requires that interrupting counters appear at the end of the pmc list
   In the case a user wants to interrupt on a counter in an evntset that is not
   among the last events, we need to move the perfctr virtual events around to
   make it last. This function swaps two perfctr events, and then adjust the
   position entries in both the NativeInfoArray and the EventInfoArray to keep
   everything consistent.
*/
static void swap_events(EventSetInfo_t * ESI, struct hwd_pmc_control *contr, int cntr1, int cntr2) {
   unsigned int ui;
   int si, i, j;

   for(i = 0; i < ESI->NativeCount; i++) {
      if(ESI->NativeInfoArray[i].ni_position == cntr1)
         ESI->NativeInfoArray[i].ni_position = cntr2;
      else if(ESI->NativeInfoArray[i].ni_position == cntr2)
         ESI->NativeInfoArray[i].ni_position = cntr1;
   }
   for(i = 0; i < ESI->NumberOfEvents; i++) {
      for(j = 0; ESI->EventInfoArray[i].pos[j] >= 0; j++) {
         if(ESI->EventInfoArray[i].pos[j] == cntr1)
            ESI->EventInfoArray[i].pos[j] = cntr2;
         else if(ESI->EventInfoArray[i].pos[j] == cntr2)
            ESI->EventInfoArray[i].pos[j] = cntr1;
      }
   }
   ui = contr->cpu_control.pmc_map[cntr1];
   contr->cpu_control.pmc_map[cntr1] = contr->cpu_control.pmc_map[cntr2];
   contr->cpu_control.pmc_map[cntr2] = ui;

   si = contr->cpu_control.ireset[cntr1];
   contr->cpu_control.ireset[cntr1] = contr->cpu_control.ireset[cntr2];
   contr->cpu_control.ireset[cntr2] = si;
}


int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d, threshold = %d\n", EventIndex, threshold);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_system_info.sub_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
      OVFDBG("Selector id (%d) larger than ncntrs (%d)\n", i, ncntrs);
      return PAPI_EINVAL;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
      if (ESI->EventInfoArray[EventIndex].derived) {
         OVFDBG("Can't overflow on a derived event.\n");
         return PAPI_EINVAL;
      }

      if (! pmc_interrupt_is_supported(contr->cpu_control.pmc_map[EventIndex])) {
         /* Make an attempt to reallocate to PMC registers that do support
            interrupt on overflow */
         retval = _papi_hwd_allocate_registers(ESI);
         if (!retval) {
            OVFDBG("Unable to allocate registers with requested event overflow\n");
            return PAPI_ECNFLCT;
         }
      }

      if ((retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig,NEED_CONTEXT)) != PAPI_OK)
	      return(retval);

      contr->cpu_control.ireset[i] = PMC_OVFL - threshold;
      nricntrs = ++contr->cpu_control.nrictrs;
      nracntrs = --contr->cpu_control.nractrs;
      contr->si_signo = _papi_hwi_system_info.sub_info.hardware_intr_sig;
      contr->cpu_control.ppc64.mmcr0 |= PERF_INT_ENABLE;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);

      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.ppc64.mmcr0 & PERF_INT_ENABLE) {
         contr->cpu_control.ireset[i] = 0;
         nricntrs = --contr->cpu_control.nrictrs;
         nracntrs = ++contr->cpu_control.nractrs;
         if (!nricntrs)
	         contr->cpu_control.ppc64.mmcr0 &= (~PERF_INT_ENABLE);
      }
      /* move this event to the top part of the list if needed */
      if (i >= nracntrs)
         swap_events(ESI, contr, i, nracntrs - 1);
      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

      retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
   }
#ifdef DEBUG   
   print_control(&contr->cpu_control);
#endif
   OVFDBG("%s:%d: Hardware overflow is still experimental.\n", __FILE__, __LINE__);
   OVFDBG("End of call. Exit code: %d\n", retval);

   return (retval);
}



int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   /* This function is not used and shouldn't be called. */
   return (PAPI_ESBSTR);
}


int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

int _papi_hwd_set_domain(EventSetInfo_t * ESI, int domain) {
	return set_domain(ESI, domain);
}

/* Routines to support an opaque native event table */
int _papi_hwd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
   if ((EventCode & PAPI_NATIVE_AND_MASK) >= _papi_hwi_system_info.sub_info.num_native_events)
       return (PAPI_ENOEVNT);
   strncpy(ntv_name, native_name_map[EventCode & PAPI_NATIVE_AND_MASK].name, len);
   if (strlen(native_name_map[EventCode & PAPI_NATIVE_AND_MASK].name) > len-1) return (PAPI_EBUF);
   return (PAPI_OK);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
  if ((EventCode & PAPI_NATIVE_AND_MASK) >= _papi_hwi_system_info.sub_info.num_native_events) {
     return (PAPI_ENOEVNT);
  }

  memcpy(bits,&native_table[native_name_map[EventCode & PAPI_NATIVE_AND_MASK].index].resources,sizeof(hwd_register_t)); 
  return (PAPI_OK);
}

static void copy_value(unsigned int val, char *nam, char *names, unsigned int *values, int len)
{
   *values = val;
   strncpy(names, nam, len);
   names[len-1] = 0;
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
   int i = 0;
   copy_value(bits->selector, "Available counters", &names[i*name_len], &values[i], name_len);
   if (++i == count) return(i);
   int j;
   int event_found = 0;
   for (j = 0; j < 5; j++) {
     if (bits->counter_cmd[j] >= 0) {
       event_found = 1;
       break;
     }
   }
   if (event_found) {
     copy_value(bits->counter_cmd[j], "Event on first counter", &names[i*name_len], &values[i], name_len);
   }
   if (++i == count) return(i);

   int group_sets = 0;
   int k;
   for (k = 0; k < GROUP_INTS; k++) {
     if (bits->group[k]) 
       group_sets++;
   }
   char * msg_base = "Available group";
   char * set_id_msg = ", set ";
   char * msg = (char *) malloc(30);
   int current_group_set = 0;
   for (k = 0; k < GROUP_INTS; k++) {
     if (bits->group[k]) {
       if (group_sets > 1) {
         sprintf(msg, "%s%s%d", msg_base, set_id_msg, ++current_group_set);
         copy_value(bits->group[k], msg, &names[i*name_len], &values[i], name_len);
       } else {
           copy_value(bits->group[k], msg_base, &names[i*name_len], &values[i], name_len);
       }
     }
   }
   
   return(++i);
}


int _papi_hwd_ntv_code_to_descr(unsigned int EventCode, char *ntv_descr, int len)
{
   if ((EventCode & PAPI_NATIVE_AND_MASK) >= _papi_hwi_system_info.sub_info.num_native_events)
       return (PAPI_EINVAL);
   strncpy(ntv_descr, native_table[native_name_map[EventCode & PAPI_NATIVE_AND_MASK].index].description, len);
   if (strlen(native_table[native_name_map[EventCode & PAPI_NATIVE_AND_MASK].index].description) > len-1) return (PAPI_EBUF);
   return (PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (modifier == PAPI_ENUM_FIRST) {
         *EventCode = PAPI_NATIVE_MASK;
         return (PAPI_OK);
   }

   if (modifier == PAPI_ENUM_EVENTS) {
      int index = *EventCode & PAPI_NATIVE_AND_MASK;
	if (index+1 == MAX_NATNAME_MAP_INDEX) {
	    return (PAPI_ENOEVNT);
	} else {
	    *EventCode = *EventCode + 1;
	    return (PAPI_OK);
	}
   } else if (modifier == PAPI_NTV_ENUM_GROUPS) {
/* Use this modifier for all supported PPC64 processors. */
      unsigned int group = (*EventCode & PAPI_NTV_GROUP_AND_MASK) >> PAPI_NTV_GROUP_SHIFT;
      int index = *EventCode & 0x000001FF;
      int i;
      unsigned int tmpg;

      *EventCode = *EventCode & (~PAPI_NTV_GROUP_SHIFT);
      for (i = 0; i < GROUP_INTS; i++) {
         tmpg = native_table[index].resources.group[i];
         if (group != 0) {
            while ((ffs(tmpg) + i * 32) <= group && tmpg != 0)
               tmpg = tmpg ^ (1 << (ffs(tmpg) - 1));
         }
         if (tmpg != 0) {
            group = ffs(tmpg) + i * 32;
            *EventCode = *EventCode | (group << PAPI_NTV_GROUP_SHIFT);
            return (PAPI_OK);
         }
      }
	  return (PAPI_ENOEVNT);
   }
   else
      return (PAPI_EINVAL);
}

papi_svector_t _ppc64_vector_table[] = {
 { (void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 { (void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
 { (void (*)())_papi_hwd_update_control_state, VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 { (void (*)())_papi_hwd_start, VEC_PAPI_HWD_START},
 { (void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP},
 { (void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ},
 { (void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 { (void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN},
 { (void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 { (void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 { (void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 { (void (*)())_papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
 { (void (*)())*_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 { (void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 { (void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 { (void (*)())*_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 { (void (*)())*_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 { NULL, VEC_PAPI_END}
};
                                                                                
int ppc64_setup_vector_table(papi_vectors_t *vtable){
  int retval=PAPI_OK;
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _ppc64_vector_table);
#endif
  return(retval);
}

