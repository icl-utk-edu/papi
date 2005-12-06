/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

extern hwi_preset_data_t _papi_hwd_preset_map[];

static hwi_search_t preset_name_map_x1[PAPI_MAX_PRESET_EVENTS] = {
   {PAPI_L1_DCM, {0, {X1_P_EV_DCACHE_MISS, PAPI_NULL}, 0}}, /* Level 1 data cache misses */
   {PAPI_L1_ICM, {0, {X1_P_EV_ICACHE_MISS, PAPI_NULL}, 0}}, /* Level 1 instruction cache misses */
   {PAPI_TLB_IM, {0, {X1_P_EV_ITLB_MISS, PAPI_NULL}, 0}},     /* Instruction translation lookaside buffer misses */
   {PAPI_TLB_TL, {0, {X1_P_EV_TLB_MISS, PAPI_NULL}, 0}},       /* Total translation lookaside buffer misses */
   {PAPI_BR_CN, {0, {X1_P_EV_BRANCH_TAKEN, PAPI_NULL}, 0}},        /* Conditional branch instructions taken */
   {PAPI_BR_PRC, {0, {X1_P_EV_BHT_CORRECT, PAPI_NULL}, 0}}, /* Conditional branch instructions correctly predicted */
/* This event gives random counts that are either 0, a very small number, or a very large number when running sdsc.c
   I suspect it doesn't count what we think it counts. It should be deprecated pending further study. --dkt */
   /*   {PAPI_TOT_IIS, {0, {X1_P_EV_INST_DISPATCH, PAPI_NULL}, 0}}, */   /* Instructions issued */
   {PAPI_TOT_INS, {0, {X1_P_EV_INST_GRAD, PAPI_NULL}, 0}},    /* Instructions completed */
   {PAPI_INT_INS, {0, {X1_P_EV_INST_S_INT, PAPI_NULL}, 0}},  /* Integer instructions */
   {PAPI_FP_INS, {0, {X1_P_EV_INST_S_FP, PAPI_NULL}, 0}},     /* Floating point instructions */
   {PAPI_LD_INS, {0, {X1_P_EV_INST_LOAD, PAPI_NULL}, 0}},     /* Load instructions */
   {PAPI_BR_INS, {0, {X1_P_EV_INST_BRANCH, PAPI_NULL}, 0}}, /* Branch instructions */
   {PAPI_TOT_CYC, {0, {X1_P_EV_CYCLES, PAPI_NULL}, 0}},  /* Total cycles */
   {PAPI_SYC_INS, {0, {X1_P_EV_INST_SYNCS, PAPI_NULL}, 0}},  /* Synchronization instructions completed */
   {PAPI_L1_ICH, {0, {X1_P_EV_ICACHE_HITS, PAPI_NULL}, 0}}, /* Level 1 instruction cache hits */
   {0, {0, {PAPI_NULL}, 0}} /* end of list */
};

