/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

extern hwi_preset_data_t _papi_hwd_preset_map[];

/* These defines smooth out the differences between versions of pmtoolkit */
#if defined(PMTOOLKIT_1_2_1)
#define PMTOOLKIT_1_2
#endif

/* Put any modified metrics in the appropriate spot here */
#ifdef PMTOOLKIT_1_2
#ifdef PMTOOLKIT_1_2_1
    /*#undef  PM_SNOOP */
#define PNE_PM_SNOOP       PNE_PM_SNOOP_RECV    /* The name in pre pmtoolkit-1.2.2 */
#define PNE_PM_LSU_EXEC	   PNE_PM_LS_EXEC
#define PNE_PM_ST_MISS_L1  PNE_PM_ST_MISS
#define PNE_PM_MPRED_BR	   PNE_PM_MPRED_BR_CAUSED_GC
#else
#define PNE_PM_LSU_EXEC	   PNE_PM_LS_EXEC
#define PNE_PM_ST_MISS_L1  PNE_PM_ST_MISS
#define PNE_PM_MPRED_BR	   PNE_PM_MPRED_BR_CAUSED_GC
#endif                          /*PMTOOLKIT_1_2_1 */
#else                           /* pmtoolkit 1.3 and later */
#ifdef _AIXVERSION_510          /* AIX Version 5 */
#define PNE_PM_LSU_EXEC   PNE_PM_LSU_CMPL
#define PNE_PM_RESRV_CMPL PNE_PM_STCX_SUCCESS
#define PNE_PM_RESRV_RQ	  PNE_PM_LARX
#define PNE_PM_MPRED_BR	  PNE_PM_BR_MPRED_GC
#define PNE_PM_EXEC_FMA	  PNE_PM_FPU_FMA
#define PNE_PM_BR_FINISH  PNE_PM_BRU_FIN
#else                           /* AIX Version 4 */
#define PNE_PM_ST_MISS_L1 PNE_PM_ST_L1MISS
#define PNE_PM_MPRED_BR	  PNE_PM_MPRED_BR_CAUSED_GC
#endif   /*_AIXVERSION_510*/
#endif                          /*PMTOOLKIT_1_2 */

extern hwi_preset_data_t _papi_hwd_preset_map[];

static hwi_search_t preset_name_map_604[PAPI_MAX_PRESET_EVENTS] = {
   {PAPI_L1_DCM, {0, {PNE_PM_DC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 data cache misses */
   {PAPI_L1_ICM, {0, {PNE_PM_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 instruction cache misses */
   {PAPI_L1_TCM, {DERIVED_ADD, {PNE_PM_DC_MISS, PNE_PM_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Level 1 total cache misses */
   {PAPI_CA_SNP, {0, {PNE_PM_SNOOP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Snoops */
   {PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Data translation lookaside buffer misses */
   {PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Instr translation lookaside buffer misses */
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total translation lookaside buffer misses */
   {PAPI_L2_LDM, {0, {PNE_PM_LD_MISS_EXCEED_L2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Level 2 load misses */
   {PAPI_L2_STM, {0, {PNE_PM_ST_MISS_EXCEED_L2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Level 2 store misses */
   {PAPI_CSR_SUC,{ 0, {PNE_PM_RESRV_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Successful store conditional instructions */
   {PAPI_CSR_FAL, {DERIVED_SUB, {PNE_PM_RESRV_RQ, PNE_PM_RESRV_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Failed store conditional instructions */
   {PAPI_CSR_TOT, {0, {PNE_PM_RESRV_RQ, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total store conditional instructions */
   {PAPI_MEM_RCY, {0, {PNE_PM_LD_MISS_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Cycles Stalled Waiting for Memory Read */
   {PAPI_BR_CN, {0, {PNE_PM_BR_FINISH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Conditional branch instructions executed */
   {PAPI_BR_MSP, {0, {PNE_PM_BR_MPRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Conditional branch instructions mispred */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions executed */
   {PAPI_INT_INS, {0, {PNE_PM_FXU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Integer instructions executed */
   {PAPI_FP_INS, {0, {PNE_PM_FPU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Floating point instructions executed */
   {PAPI_FP_OPS, {0, {PNE_PM_FPU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Floating point instructions executed */
   {PAPI_LD_INS, {0, {PNE_PM_LD_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Load instructions executed */
   {PAPI_BR_INS, {0, {PNE_PM_BR_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total branch instructions executed */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Total cycles */
   {PAPI_LST_INS, {0, {PNE_PM_LSU_EXEC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total load/store inst. executed */
   {PAPI_SYC_INS, {0, {PNE_PM_SYNC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Sync. inst. executed */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}     /* end of list */
};

static hwi_search_t preset_name_map_604e[PAPI_MAX_PRESET_EVENTS] = {
   {PAPI_L1_DCM, {0, {PNE_PM_DC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 data cache misses */
   {PAPI_L1_ICM, {0, {PNE_PM_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 instruction cache misses */
   {PAPI_L1_TCM, {DERIVED_ADD, {PNE_PM_DC_MISS, PNE_PM_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Level 1 total cache misses */
   {PAPI_CA_SNP, {0, {PNE_PM_SNOOP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Snoops */
   {PAPI_CA_SHR, {0, {PNE_PM_LD_MISS_DC_SHR, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Request for shared cache line (SMP) */
   {PAPI_CA_INV, {0, {PNE_PM_WR_HIT_SHR_KILL_BRC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Request for cache line Invalidation (SMP) */
   {PAPI_CA_ITV, {0, {PNE_PM_WR_HIT_SHR_KILL_BRC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Request for cache line Intervention (SMP) */
   {PAPI_BRU_IDL, {0, {PNE_PM_BRU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles branch units are idle */
   {PAPI_FXU_IDL, {0, {PNE_PM_MCI_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles integer units are idle */
   {PAPI_FPU_IDL, {0, {PNE_PM_FPU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles floating point units are idle */
   {PAPI_LSU_IDL, {0, {PNE_PM_LSU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles load/store units are idle */
   {PAPI_TLB_DM, {0, {PNE_PM_DTLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Data translation lookaside buffer misses */
   {PAPI_TLB_IM, {0, {PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Instr translation lookaside buffer misses */
   {PAPI_TLB_TL, {DERIVED_ADD, {PNE_PM_DTLB_MISS, PNE_PM_ITLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total translation lookaside buffer misses */
   {PAPI_L2_LDM, {0, {PNE_PM_LD_MISS_EXCEED_L2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Level 2 load misses */
   {PAPI_L2_STM, {0, {PNE_PM_ST_MISS_EXCEED_L2, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Level 2 store misses */
   {PAPI_CSR_SUC, {0, {PNE_PM_RESRV_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Successful store conditional instructions */
   {PAPI_CSR_FAL, {DERIVED_SUB, {PNE_PM_RESRV_RQ, PNE_PM_RESRV_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Failed store conditional instructions */
   {PAPI_CSR_TOT, {0, {PNE_PM_RESRV_RQ, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total store conditional instructions */
   {PAPI_MEM_SCY, {DERIVED_ADD, {PNE_PM_CMPLU_WT_LD, PNE_PM_CMPLU_WT_ST, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Cycles Stalled Waiting for Memory Access */
   {PAPI_MEM_RCY, {0, {PNE_PM_CMPLU_WT_LD, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Cycles Stalled Waiting for Memory Read */
   {PAPI_MEM_WCY, {0, {PNE_PM_CMPLU_WT_ST, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Cycles Stalled Waiting for Memory Write */
   {PAPI_STL_ICY, {0, {PNE_PM_DPU_WT_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Cycles with No Instruction Issue */
   {PAPI_FUL_ICY, {0, {PNE_PM_4INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Cycles with Maximum Instruction Issue */
   {PAPI_STL_CCY, {0, {PNE_PM_CMPLU_WT_UNF_INST, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Cycles with No Instruction Completion */
   {PAPI_FUL_CCY, {0, {PNE_PM_4INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Cycles with Maximum Instruction Completion */
   {PAPI_BR_CN, {0, {PNE_PM_BR_FINISH, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Conditional branch instructions executed */
   {PAPI_BR_MSP, {0, {PNE_PM_BR_MPRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Conditional branch instructions mispred */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions executed */
   {PAPI_INT_INS, {0, {PNE_PM_FXU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Integer instructions executed */
   {PAPI_FP_INS, {0, {PNE_PM_FPU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Floating point instructions executed */
   {PAPI_FP_OPS, {0, {PNE_PM_FPU_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Floating point instructions executed */
   {PAPI_LD_INS, {0, {PNE_PM_LD_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Load instructions executed */
   {PAPI_BR_INS, {0, {PNE_PM_BR_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total branch instructions executed */
   {PAPI_FP_STAL, {0, {PNE_PM_FPU_WT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Cycles any FP units are stalled */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Total cycles */
   {PAPI_LST_INS, {0, {PNE_PM_LSU_EXEC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total load/store inst. executed */
   {PAPI_SYC_INS, {0, {PNE_PM_SYNC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Sync. inst. executed */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}     /* end of list */
};

static hwi_search_t preset_name_map_630[PAPI_MAX_PRESET_EVENTS] = {
   {PAPI_L1_DCM, {DERIVED_ADD, {PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Level 1 data cache misses */
   {PAPI_L1_ICM, {0, {PNE_PM_IC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Level 1 instruction cache misses */
   {PAPI_L1_TCM, {DERIVED_ADD, {PNE_PM_IC_MISS, PNE_PM_LD_MISS_L1, PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Level 1 total cache misses */
   {PAPI_CA_SNP, {0, {PNE_PM_SNOOP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Snoops */
   {PAPI_CA_SHR, {0, {PNE_PM_SNOOP_E_TO_S, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Request for shared cache line (SMP) */
   {PAPI_CA_ITV, {0, {PNE_PM_SNOOP_PUSH_INT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},      /*Request for cache line Intervention (SMP) */
   {PAPI_BRU_IDL, {0, {PNE_PM_BRU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles branch units are idle */
   {PAPI_FXU_IDL, {0, {PNE_PM_FXU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles integer units are idle */
   {PAPI_FPU_IDL, {0, {PNE_PM_FPU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles floating point units are idle */
   {PAPI_LSU_IDL, {0, {PNE_PM_LSU_IDLE, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Cycles load/store units are idle */
   {PAPI_TLB_TL, {0, {PNE_PM_TLB_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Total translation lookaside buffer misses */
   {PAPI_L1_LDM, {0, {PNE_PM_LD_MISS_L2HIT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Level 1 load misses */
   {PAPI_L1_STM, {0, {PNE_PM_ST_MISS_L1, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Level 1 store misses */
   {PAPI_L2_LDM, {0, {PNE_PM_BIU_LD_NORTRY, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Level 2 load misses */
   {PAPI_BTAC_M, {0, {PNE_PM_BTAC_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*BTAC miss */
   {PAPI_PRF_DM, {0, {PNE_PM_PREF_MATCH_DEM_MISS, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Prefetch data instruction caused a miss */
   {PAPI_TLB_SD, {0, {PNE_PM_TLBSYNC_RERUN, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Xlation lookaside buffer shootdowns (SMP) */
   {PAPI_CSR_SUC, {0, {PNE_PM_RESRV_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Successful store conditional instructions */
   {PAPI_CSR_FAL, {0, {PNE_PM_ST_COND_FAIL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Failed store conditional instructions */
   {PAPI_CSR_TOT, {0, {PNE_PM_RESRV_RQ, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*Total store conditional instructions */
   {PAPI_MEM_SCY, {DERIVED_ADD, {PNE_PM_CMPLU_WT_LD, PNE_PM_CMPLU_WT_ST, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Cycles Stalled Waiting for Memory Access */
   {PAPI_MEM_RCY, {0, {PNE_PM_CMPLU_WT_LD, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Cycles Stalled Waiting for Memory Read */
   {PAPI_MEM_WCY, {0, {PNE_PM_CMPLU_WT_ST, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Cycles Stalled Waiting for Memory Write */
   {PAPI_STL_ICY, {0, {PNE_PM_0INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Cycles with No Instruction Issue */
   {PAPI_STL_CCY, {0, {PNE_PM_0INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}, /*Cycles with No Instruction Completion */
   {PAPI_BR_CN, {0, {PNE_PM_CBR_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Conditional branch instructions executed */
   {PAPI_BR_MSP, {0, {PNE_PM_MPRED_BR, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},    /*Conditional branch instructions mispred */
   {PAPI_BR_PRC, {0, {PNE_PM_BR_PRED, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Conditional branch instructions corr. pred */
   {PAPI_FMA_INS, {0, {PNE_PM_EXEC_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*FMA instructions completed */
   {PAPI_TOT_IIS, {0, {PNE_PM_INST_DISP, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions issued */
   {PAPI_TOT_INS, {0, {PNE_PM_INST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Total instructions executed */
   {PAPI_INT_INS, {DERIVED_ADD, {PNE_PM_FXU0_PROD_RESULT, PNE_PM_FXU1_PROD_RESULT, PNE_PM_FXU2_PROD_RESULT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Integer instructions executed */
   {PAPI_FP_INS, {DERIVED_ADD, {PNE_PM_FPU0_CMPL, PNE_PM_FPU1_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Floating point instructions executed */
   {PAPI_FP_OPS, {DERIVED_ADD, {PNE_PM_FPU0_CMPL, PNE_PM_FPU1_CMPL, PNE_PM_EXEC_FMA, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*Floating point instructions executed */
   {PAPI_LD_INS, {0, {PNE_PM_LD_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Load instructions executed */
   {PAPI_SR_INS, {0, {PNE_PM_ST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Store instructions executed */
   {PAPI_BR_INS, {0, {PNE_PM_BR_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total branch instructions executed */
   {PAPI_TOT_CYC, {0, {PNE_PM_CYC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},        /*Total cycles */
   {PAPI_LST_INS, {DERIVED_ADD, {PNE_PM_LD_CMPL, PNE_PM_ST_CMPL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},     /*Total load/store inst. executed */
   {PAPI_SYC_INS, {0, {PNE_PM_SYNC, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},       /*Sync. inst. executed */
   {PAPI_FDV_INS, {0, {PNE_PM_FPU_FDIV, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},   /*FD ins */
   {PAPI_FSQ_INS, {0, {PNE_PM_FPU_FSQRT, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}},  /*FSq ins */
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, 0}}     /* end of list */
};

hwi_search_t *preset_search_map;

/*#define DEBUG_SETUP*/


/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
   dst->ra_selector = (1 << ctr);
   dst->ra_rank = 1;
}

/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   return (dst->ra_selector & src->ra_selector);
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   dst->ra_selector ^= src->ra_selector;
   dst->ra_rank -= src->ra_rank;
}

/* This function updates the selection status of 
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   dst->ra_selector = src->ra_selector;
}

/* initialize preset_search_map table by type of CPU */
int _papi_hwd_init_preset_search_map(pm_info_t * info)
{
   info = &pminfo;

   if (__power_630()) {
      preset_search_map = preset_name_map_630;
   } else if (__power_604()) {
      if (strstr(info->proc_name, "604e")) {
         preset_search_map = preset_name_map_604e;
      } else {
         preset_search_map = preset_name_map_604;
      }
   } else {
      return 0;
   }
   return 1;
}

/* this function will be called when there are counters available 
     success  return 1
	 fail     return 0
*/
int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   unsigned char selector;
   int i, j, natNum, index;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];
   int position;

   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for (i = 0; i < natNum; i++) {
      /* CAUTION: Since this is in the hardware layer, it's ok 
         to access the native table directly, but in general this is a bad idea */
      if ((index =
           native_name_map[ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK].index) < 0)
         return 0;
      event_list[i].ra_selector = native_table[index].resources.selector;
      /* calculate native event rank, which is number of counters it can live on, this is power3 specific */
      event_list[i].ra_rank = 0;
      for (j = 0; j < MAX_COUNTERS; j++) {
         if (event_list[i].ra_selector & (1 << j))
            event_list[i].ra_rank++;
      }
      /*event_list[i].ra_mod = -1; */
   }

   if (_papi_hwi_bipartite_alloc(event_list, natNum)) { /* successfully mapped */
      /* copy counter allocations info back into NativeInfoArray */
      for (i = 0; i < natNum; i++)
         ESI->NativeInfoArray[i].ni_position = ffs(event_list[i].ra_selector) - 1;
      /* update the control structure based on the NativeInfoArray */
      /*_papi_hwd_update_control_state(this_state, ESI->NativeInfoArray, natNum);*/
      return 1;
   } else {
      return 0;
   }
}


/* This used to be init_config, static to the substrate.
   Now its exposed to the hwi layer and called when an EventSet is allocated.
*/
void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   int i;

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      ptr->counter_cmd.events[i] = COUNT_NOTHING;
   }
   set_domain(ptr, _papi_hwi_system_info.default_domain);
   set_granularity(ptr, _papi_hwi_system_info.default_granularity);
}


/* This function updates the control structure with whatever resources are allocated
    for all the native events in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count)
{
   int i, index;

   /* empty all the counters */
   for (i = 0; i < MAX_COUNTERS; i++) {
      this_state->counter_cmd.events[i] = COUNT_NOTHING;
   }

   /* refill the counters we're using */
   for (i = 0; i < count; i++) {
      /* CAUTION: Since this is in the hardware layer, it's ok 
         to access the native table directly, but in general this is a bad idea */
      if ((index = native_name_map[native[i].ni_event & PAPI_NATIVE_AND_MASK].index) < 0)
         return PAPI_ENOEVNT;
      this_state->counter_cmd.events[native[i].ni_position] =
          native_table[index].resources.counter_cmd[native[i].ni_position];
   }

   return PAPI_OK;
}
