/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = {
  {PAPI_L1_DCM,0,{X1_P_EV_DCACHE_MISS,0,0,0,0,0,0,0},"X1_P_EV_DCACHE_MISS"},  /* Level 1 data cache misses */
  {PAPI_L1_ICM,0,{X1_P_EV_ICACHE_MISS,0,0,0,0,0,0,0},"X1_P_EV_DCACHE_MISS"},  /* Level 1 instruction cache misses */
  {PAPI_TLB_IM,0,{X1_P_EV_ITLB_MISS,0,0,0,0,0,0,0},"X1_P_EV_ITLB_MISS"},  /* Instruction translation lookaside buffer misses */
  {PAPI_TLB_TL,0,{X1_P_EV_TLB_MISS,0,0,0,0,0,0,0},"X1_P_EV_TLB_MISS"},  /* Total translation lookaside buffer misses */
  {PAPI_BR_CN,0,{X1_P_EV_BRANCH_TAKEN,0,0,0,0,0,0,0},"X1_P_EV_BRANCH_TAKEN"},  /* Conditional branch instructions taken */ 
  {PAPI_BR_MSP,0,{X1_P_EV_BRANCH_MISPREDICT,0,0,0,0,0,0,0},"X1_P_EV_BRANCH_MISPREDICT"},  /* Conditional branch instructions mispredicted */
  {PAPI_BR_PRC,0,{X1_P_EV_BHT_CORRECT,0,0,0,0,0,0,0},"X1_P_EV_BHT_CORRECT"},  /* Conditional branch instructions correctly predicted */
  {PAPI_TOT_IIS,0,{X1_P_EV_INST_DISPATCH,0,0,0,0,0,0,0},"X1_P_EV_INST_DISPATCH"},  /* Instructions issued */
  {PAPI_TOT_INS,0,{X1_P_EV_INST_GRAD,0,0,0,0,0,0,0},"X1_P_EV_INST_GRAD"},  /* Instructions completed */
  {PAPI_INT_INS,0,{X1_P_EV_INST_S_INT,0,0,0,0,0,0,0},"X1_P_EV_INST_S_INT"},  /* Integer instructions */
  {PAPI_FP_INS,0,{X1_P_EV_INST_S_FP,0,0,0,0,0,0,0},"X1_P_EV_INST_S_FP"},  /* Floating point instructions */
  {PAPI_LD_INS,0,{X1_P_EV_INST_LOAD,0,0,0,0,0,0,0},"X1_P_EV_INST_LOAD"},  /* Load instructions */
  {PAPI_BR_INS,0,{X1_P_EV_INST_BRANCH,0,0,0,0,0,0,0},"X1_P_EV_INST_BRANCH"},  /* Branch instructions */
  {PAPI_TOT_CYC,0,{X1_P_EV_CYCLES,0,0,0,0,0,0,0},"X1_P_EV_CYCLES"},  /* Total cycles */
  {PAPI_SYC_INS,0,{X1_P_EV_INST_SYNCS,0,0,0,0,0,0,0},"X1_P_EV_INST_SYNCS"},  /* Synchronization instructions completed */
  {PAPI_L1_ICH,0,{X1_P_EV_ICACHE_HITS,0,0,0,0,0,0,0},"X1_P_EV_ICACHE_HITS"},  /* Level 1 instruction cache hits */
  {0,0,{0,0,0,0,0,0,0,0},""} /* end of list */
};

