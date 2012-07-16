/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    power5_events_map.c
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* This file MUST be kept synchronised with the events file.
*
*/
#include "perfctr-ppc64.h"

PPC64_native_map_t native_name_map[MAX_NATNAME_MAP_INDEX] = {
	{"PM_0INST_CLB_CYC", -1}
	,
	{"PM_1INST_CLB_CYC", -1}
	,
	{"PM_1PLUS_PPC_CMPL", -1}
	,
	{"PM_2INST_CLB_CYC", -1}
	,
	{"PM_3INST_CLB_CYC", -1}
	,
	{"PM_4INST_CLB_CYC", -1}
	,
	{"PM_5INST_CLB_CYC", -1}
	,
	{"PM_6INST_CLB_CYC", -1}
	,
	{"PM_BRQ_FULL_CYC", -1}
	,
	{"PM_BR_UNCOND", -1}
	,
	{"PM_CLB_FULL_CYC", -1}
	,
	{"PM_CR_MAP_FULL_CYC", -1}
	,
	{"PM_CYC", -1}
	,
	{"PM_DATA_FROM_L2", -1}
	,
	{"PM_DATA_FROM_L25_SHR", -1}
	,
	{"PM_DATA_FROM_L275_MOD", -1}
	,
	{"PM_DATA_FROM_L3", -1}
	,
	{"PM_DATA_FROM_L35_SHR", -1}
	,
	{"PM_DATA_FROM_L375_MOD", -1}
	,
	{"PM_DATA_FROM_RMEM", -1}
	,
	{"PM_DATA_TABLEWALK_CYC", -1}
	,
	{"PM_DSLB_MISS", -1}
	,
	{"PM_DTLB_MISS", -1}
	,
	{"PM_DTLB_MISS_16M", -1}
	,
	{"PM_DTLB_MISS_4K", -1}
	,
	{"PM_DTLB_REF_16M", -1}
	,
	{"PM_DTLB_REF_4K", -1}
	,
	{"PM_FAB_CMD_ISSUED", -1}
	,
	{"PM_FAB_DCLAIM_ISSUED", -1}
	,
	{"PM_FAB_HOLDtoNN_EMPTY", -1}
	,
	{"PM_FAB_HOLDtoVN_EMPTY", -1}
	,
	{"PM_FAB_M1toP1_SIDECAR_EMPTY", -1}
	,
	{"PM_FAB_P1toM1_SIDECAR_EMPTY", -1}
	,
	{"PM_FAB_PNtoNN_DIRECT", -1}
	,
	{"PM_FAB_PNtoVN_DIRECT", -1}
	,
	{"PM_FPR_MAP_FULL_CYC", -1}
	,
	{"PM_FPU0_1FLOP", -1}
	,
	{"PM_FPU0_DENORM", -1}
	,
	{"PM_FPU0_FDIV", -1}
	,
	{"PM_FPU0_FMA", -1}
	,
	{"PM_FPU0_FSQRT", -1}
	,
	{"PM_FPU0_FULL_CYC", -1}
	,
	{"PM_FPU0_SINGLE", -1}
	,
	{"PM_FPU0_STALL3", -1}
	,
	{"PM_FPU0_STF", -1}
	,
	{"PM_FPU1_1FLOP", -1}
	,
	{"PM_FPU1_DENORM", -1}
	,
	{"PM_FPU1_FDIV", -1}
	,
	{"PM_FPU1_FMA", -1}
	,
	{"PM_FPU1_FSQRT", -1}
	,
	{"PM_FPU1_FULL_CYC", -1}
	,
	{"PM_FPU1_SINGLE", -1}
	,
	{"PM_FPU1_STALL3", -1}
	,
	{"PM_FPU1_STF", -1}
	,
	{"PM_FPU_DENORM", -1}
	,
	{"PM_FPU_FDIV", -1}
	,
	{"PM_FPU_1FLOP", -1}
	,
	{"PM_FPU_FULL_CYC", -1}
	,
	{"PM_FPU_SINGLE", -1}
	,
	{"PM_FXU_IDLE", -1}
	,
	{"PM_GCT_NOSLOT_CYC", -1}
	,
	{"PM_GCT_FULL_CYC", -1}
	,
	{"PM_GCT_USAGE_00to59_CYC", -1}
	,
	{"PM_GRP_BR_REDIR", -1}
	,
	{"PM_GRP_BR_REDIR_NONSPEC", -1}
	,
	{"PM_GRP_DISP_REJECT", -1}
	,
	{"PM_GRP_DISP_VALID", -1}
	,
	{"PM_GRP_IC_MISS", -1}
	,
	{"PM_GRP_IC_MISS_BR_REDIR_NONSPEC", -1}
	,
	{"PM_GRP_IC_MISS_NONSPEC", -1}
	,
	{"PM_GRP_MRK", -1}
	,
	{"PM_IC_PREF_REQ", -1}
	,
	{"PM_IERAT_XLATE_WR", -1}
	,
	{"PM_INST_CMPL", -1}
	,
	{"PM_INST_DISP", -1}
	,
	{"PM_INST_FETCH_CYC", -1}
	,
	{"PM_INST_FROM_L2", -1}
	,
	{"PM_INST_FROM_L25_SHR", -1}
	,
	{"PM_INST_FROM_L3", -1}
	,
	{"PM_INST_FROM_L35_SHR", -1}
	,
	{"PM_ISLB_MISS", -1}
	,
	{"PM_ITLB_MISS", -1}
	,
	{"PM_L2SA_MOD_TAG", -1}
	,
	{"PM_L2SA_RCLD_DISP", -1}
	,
	{"PM_L2SA_RCLD_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SA_RCST_DISP", -1}
	,
	{"PM_L2SA_RCST_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SA_RC_DISP_FAIL_CO_BUSY", -1}
	,
	{"PM_L2SA_SHR_MOD", -1}
	,
	{"PM_L2SA_ST_REQ", -1}
	,
	{"PM_L2SB_MOD_TAG", -1}
	,
	{"PM_L2SB_RCLD_DISP", -1}
	,
	{"PM_L2SB_RCLD_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SB_RCST_DISP", -1}
	,
	{"PM_L2SB_RCST_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SB_RC_DISP_FAIL_CO_BUSY", -1}
	,
	{"PM_L2SB_SHR_MOD", -1}
	,
	{"PM_L2SB_ST_REQ", -1}
	,
	{"PM_L2SC_MOD_TAG", -1}
	,
	{"PM_L2SC_RCLD_DISP", -1}
	,
	{"PM_L2SC_RCLD_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SC_RCST_DISP", -1}
	,
	{"PM_L2SC_RCST_DISP_FAIL_RC_FULL", -1}
	,
	{"PM_L2SC_RC_DISP_FAIL_CO_BUSY", -1}
	,
	{"PM_L2SC_SHR_MOD", -1}
	,
	{"PM_L2SC_ST_REQ", -1}
	,
	{"PM_L3SA_ALL_BUSY", -1}
	,
	{"PM_L3SA_MOD_TAG", -1}
	,
	{"PM_L3SA_REF", -1}
	,
	{"PM_L3SB_ALL_BUSY", -1}
	,
	{"PM_L3SB_MOD_TAG", -1}
	,
	{"PM_L3SB_REF", -1}
	,
	{"PM_L3SC_ALL_BUSY", -1}
	,
	{"PM_L3SC_MOD_TAG", -1}
	,
	{"PM_L3SC_REF", -1}
	,
	{"PM_LARX_LSU0", -1}
	,
	{"PM_LR_CTR_MAP_FULL_CYC", -1}
	,
	{"PM_LSU0_BUSY_REJECT", -1}
	,
	{"PM_LSU0_DERAT_MISS", -1}
	,
	{"PM_LSU0_FLUSH_LRQ", -1}
	,
	{"PM_LSU0_FLUSH_SRQ", -1}
	,
	{"PM_LSU0_FLUSH_ULD", -1}
	,
	{"PM_LSU0_FLUSH_UST", -1}
	,
	{"PM_LSU0_REJECT_ERAT_MISS", -1}
	,
	{"PM_LSU0_REJECT_LMQ_FULL", -1}
	,
	{"PM_LSU0_REJECT_RELOAD_CDF", -1}
	,
	{"PM_LSU0_REJECT_SRQ_LHS", -1}
	,
	{"PM_LSU0_SRQ_STFWD", -1}
	,
	{"PM_LSU1_BUSY_REJECT", -1}
	,
	{"PM_LSU1_DERAT_MISS", -1}
	,
	{"PM_LSU1_FLUSH_LRQ", -1}
	,
	{"PM_LSU1_FLUSH_SRQ", -1}
	,
	{"PM_LSU1_FLUSH_ULD", -1}
	,
	{"PM_LSU1_FLUSH_UST", -1}
	,
	{"PM_LSU1_REJECT_ERAT_MISS", -1}
	,
	{"PM_LSU1_REJECT_LMQ_FULL", -1}
	,
	{"PM_LSU1_REJECT_RELOAD_CDF", -1}
	,
	{"PM_LSU1_REJECT_SRQ_LHS", -1}
	,
	{"PM_LSU1_SRQ_STFWD", -1}
	,
	{"PM_LSU_BUSY_REJECT", -1}
	,
	{"PM_LSU_FLUSH_LRQ_FULL", -1}
	,
	{"PM_LSU_FLUSH_SRQ", -1}
	,
	{"PM_LSU_FLUSH_ULD", -1}
	,
	{"PM_LSU_LRQ_S0_ALLOC", -1}
	,
	{"PM_LSU_LRQ_S0_VALID", -1}
	,
	{"PM_LSU_REJECT_ERAT_MISS", -1}
	,
	{"PM_LSU_REJECT_SRQ_LHS", -1}
	,
	{"PM_LSU_SRQ_S0_ALLOC", -1}
	,
	{"PM_LSU_SRQ_S0_VALID", -1}
	,
	{"PM_LSU_SRQ_STFWD", -1}
	,
	{"PM_MEM_FAST_PATH_RD_CMPL", -1}
	,
	{"PM_MEM_HI_PRIO_PW_CMPL", -1}
	,
	{"PM_MEM_HI_PRIO_WR_CMPL", -1}
	,
	{"PM_MEM_PWQ_DISP", -1}
	,
	{"PM_MEM_PWQ_DISP_BUSY2or3", -1}
	,
	{"PM_MEM_READ_CMPL", -1}
	,
	{"PM_MEM_RQ_DISP", -1}
	,
	{"PM_MEM_RQ_DISP_BUSY8to15", -1}
	,
	{"PM_MEM_WQ_DISP_BUSY1to7", -1}
	,
	{"PM_MEM_WQ_DISP_WRITE", -1}
	,
	{"PM_MRK_DATA_FROM_L2", -1}
	,
	{"PM_MRK_DATA_FROM_L25_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_L275_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_L3", -1}
	,
	{"PM_MRK_DATA_FROM_L35_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_L375_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_RMEM", -1}
	,
	{"PM_MRK_DTLB_MISS_16M", -1}
	,
	{"PM_MRK_DTLB_MISS_4K", -1}
	,
	{"PM_MRK_DTLB_REF_16M", -1}
	,
	{"PM_MRK_DTLB_REF_4K", -1}
	,
	{"PM_MRK_GRP_DISP", -1}
	,
	{"PM_MRK_GRP_ISSUED", -1}
	,
	{"PM_MRK_IMR_RELOAD", -1}
	,
	{"PM_MRK_LD_MISS_L1", -1}
	,
	{"PM_MRK_LD_MISS_L1_LSU0", -1}
	,
	{"PM_MRK_LD_MISS_L1_LSU1", -1}
	,
	{"PM_MRK_STCX_FAIL", -1}
	,
	{"PM_MRK_ST_CMPL", -1}
	,
	{"PM_MRK_ST_MISS_L1", -1}
	,
	{"PM_PMC4_OVERFLOW", -1}
	,
	{"PM_PMC5_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_L2", -1}
	,
	{"PM_PTEG_FROM_L25_SHR", -1}
	,
	{"PM_PTEG_FROM_L275_MOD", -1}
	,
	{"PM_PTEG_FROM_L3", -1}
	,
	{"PM_PTEG_FROM_L35_SHR", -1}
	,
	{"PM_PTEG_FROM_L375_MOD", -1}
	,
	{"PM_PTEG_FROM_RMEM", -1}
	,
	{"PM_RUN_CYC", -1}
	,
	{"PM_SNOOP_DCLAIM_RETRY_QFULL", -1}
	,
	{"PM_SNOOP_PW_RETRY_RQ", -1}
	,
	{"PM_SNOOP_RD_RETRY_QFULL", -1}
	,
	{"PM_SNOOP_RD_RETRY_RQ", -1}
	,
	{"PM_SNOOP_RETRY_1AHEAD", -1}
	,
	{"PM_SNOOP_TLBIE", -1}
	,
	{"PM_SNOOP_WR_RETRY_RQ", -1}
	,
	{"PM_STCX_FAIL", -1}
	,
	{"PM_STCX_PASS", -1}
	,
	{"PM_SUSPENDED", -1}
	,
	{"PM_TB_BIT_TRANS", -1}
	,
	{"PM_THRD_ONE_RUN_CYC", -1}
	,
	{"PM_THRD_PRIO_1_CYC", -1}
	,
	{"PM_THRD_PRIO_2_CYC", -1}
	,
	{"PM_THRD_PRIO_3_CYC", -1}
	,
	{"PM_THRD_PRIO_4_CYC", -1}
	,
	{"PM_THRD_PRIO_5_CYC", -1}
	,
	{"PM_THRD_PRIO_6_CYC", -1}
	,
	{"PM_THRD_PRIO_7_CYC", -1}
	,
	{"PM_TLB_MISS", -1}
	,
	{"PM_XER_MAP_FULL_CYC", -1}
	,
	{"PM_INST_FROM_L2MISS", -1}
	,
	{"PM_BR_PRED_TA", -1}
	,
	{"PM_CMPLU_STALL_DCACHE_MISS", -1}
	,
	{"PM_CMPLU_STALL_FDIV", -1}
	,
	{"PM_CMPLU_STALL_FXU", -1}
	,
	{"PM_CMPLU_STALL_LSU", -1}
	,
	{"PM_DATA_FROM_L25_MOD", -1}
	,
	{"PM_DATA_FROM_L35_MOD", -1}
	,
	{"PM_DATA_FROM_LMEM", -1}
	,
	{"PM_FPU_FSQRT", -1}
	,
	{"PM_FPU_FMA", -1}
	,
	{"PM_FPU_STALL3", -1}
	,
	{"PM_FPU_STF", -1}
	,
	{"PM_FXU_BUSY", -1}
	,
	{"PM_FXU_FIN", -1}
	,
	{"PM_GCT_NOSLOT_IC_MISS", -1}
	,
	{"PM_GCT_USAGE_60to79_CYC", -1}
	,
	{"PM_GRP_DISP", -1}
	,
	{"PM_HV_CYC", -1}
	,
	{"PM_INST_FROM_L1", -1}
	,
	{"PM_INST_FROM_L25_MOD", -1}
	,
	{"PM_INST_FROM_L35_MOD", -1}
	,
	{"PM_INST_FROM_LMEM", -1}
	,
	{"PM_LSU_DERAT_MISS", -1}
	,
	{"PM_LSU_FLUSH_LRQ", -1}
	,
	{"PM_LSU_FLUSH_UST", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_CYC", -1}
	,
	{"PM_LSU_REJECT_LMQ_FULL", -1}
	,
	{"PM_LSU_REJECT_RELOAD_CDF", -1}
	,
	{"PM_MRK_BRU_FIN", -1}
	,
	{"PM_MRK_DATA_FROM_L25_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_L25_SHR_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L275_SHR_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L2_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L35_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_L35_SHR_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L375_SHR_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L3_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_LMEM", -1}
	,
	{"PM_MRK_GRP_BR_REDIR", -1}
	,
	{"PM_MRK_ST_GPS", -1}
	,
	{"PM_PMC1_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_L25_MOD", -1}
	,
	{"PM_PTEG_FROM_L35_MOD", -1}
	,
	{"PM_PTEG_FROM_LMEM", -1}
	,
	{"PM_SLB_MISS", -1}
	,
	{"PM_GCT_EMPTY_CYC", -1}
	,
	{"PM_THRD_GRP_CMPL_BOTH_CYC", -1}
	,
	{"PM_BR_ISSUED", -1}
	,
	{"PM_BR_MPRED_CR", -1}
	,
	{"PM_BR_MPRED_TA", -1}
	,
	{"PM_BR_PRED_CR", -1}
	,
	{"PM_CRQ_FULL_CYC", -1}
	,
	{"PM_DATA_FROM_L275_SHR", -1}
	,
	{"PM_DATA_FROM_L375_SHR", -1}
	,
	{"PM_DC_INV_L2", -1}
	,
	{"PM_DC_PREF_DST", -1}
	,
	{"PM_DC_PREF_STREAM_ALLOC", -1}
	,
	{"PM_EE_OFF", -1}
	,
	{"PM_EE_OFF_EXT_INT", -1}
	,
	{"PM_FAB_CMD_RETRIED", -1}
	,
	{"PM_FAB_DCLAIM_RETRIED", -1}
	,
	{"PM_FAB_M1toVNorNN_SIDECAR_EMPTY", -1}
	,
	{"PM_FAB_P1toVNorNN_SIDECAR_EMPTY", -1}
	,
	{"PM_FAB_PNtoNN_SIDECAR", -1}
	,
	{"PM_FAB_PNtoVN_SIDECAR", -1}
	,
	{"PM_FAB_VBYPASS_EMPTY", -1}
	,
	{"PM_FLUSH_BR_MPRED", -1}
	,
	{"PM_FLUSH_IMBAL", -1}
	,
	{"PM_FLUSH", -1}
	,
	{"PM_FLUSH_SB", -1}
	,
	{"PM_FLUSH_SYNC", -1}
	,
	{"PM_FPU0_FEST", -1}
	,
	{"PM_FPU0_FIN", -1}
	,
	{"PM_FPU0_FMOV_FEST", -1}
	,
	{"PM_FPU0_FPSCR", -1}
	,
	{"PM_FPU0_FRSP_FCONV", -1}
	,
	{"PM_FPU1_FEST", -1}
	,
	{"PM_FPU1_FIN", -1}
	,
	{"PM_FPU1_FMOV_FEST", -1}
	,
	{"PM_FPU1_FRSP_FCONV", -1}
	,
	{"PM_FPU_FMOV_FEST", -1}
	,
	{"PM_FPU_FRSP_FCONV", -1}
	,
	{"PM_FXLS0_FULL_CYC", -1}
	,
	{"PM_FXLS1_FULL_CYC", -1}
	,
	{"PM_FXU0_BUSY_FXU1_IDLE", -1}
	,
	{"PM_FXU0_FIN", -1}
	,
	{"PM_FXU1_FIN", -1}
	,
	{"PM_GCT_NOSLOT_SRQ_FULL", -1}
	,
	{"PM_GCT_USAGE_80to99_CYC", -1}
	,
	{"PM_GPR_MAP_FULL_CYC", -1}
	,
	{"PM_GRP_CMPL", -1}
	,
	{"PM_GRP_DISP_BLK_SB_CYC", -1}
	,
	{"PM_GRP_DISP_SUCCESS", -1}
	,
	{"PM_IC_DEMAND_L2_BHT_REDIRECT", -1}
	,
	{"PM_IC_DEMAND_L2_BR_REDIRECT", -1}
	,
	{"PM_IC_PREF_INSTALL", -1}
	,
	{"PM_INST_FROM_L275_SHR", -1}
	,
	{"PM_INST_FROM_L375_SHR", -1}
	,
	{"PM_INST_FROM_PREF", -1}
	,
	{"PM_L1_DCACHE_RELOAD_VALID", -1}
	,
	{"PM_L1_PREF", -1}
	,
	{"PM_L1_WRITE_CYC", -1}
	,
	{"PM_L2SA_MOD_INV", -1}
	,
	{"PM_L2SA_RCLD_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SA_RCLD_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SA_RCST_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SA_RCST_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SA_RC_DISP_FAIL_CO_BUSY_ALL", -1}
	,
	{"PM_L2SA_SHR_INV", -1}
	,
	{"PM_L2SA_ST_HIT", -1}
	,
	{"PM_L2SB_MOD_INV", -1}
	,
	{"PM_L2SB_RCLD_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SB_RCLD_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SB_RCST_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SB_RCST_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SB_RC_DISP_FAIL_CO_BUSY_ALL", -1}
	,
	{"PM_L2SB_SHR_INV", -1}
	,
	{"PM_L2SB_ST_HIT", -1}
	,
	{"PM_L2SC_MOD_INV", -1}
	,
	{"PM_L2SC_RCLD_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SC_RCLD_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SC_RCST_DISP_FAIL_ADDR", -1}
	,
	{"PM_L2SC_RCST_DISP_FAIL_OTHER", -1}
	,
	{"PM_L2SC_RC_DISP_FAIL_CO_BUSY_ALL", -1}
	,
	{"PM_L2SC_SHR_INV", -1}
	,
	{"PM_L2SC_ST_HIT", -1}
	,
	{"PM_L2_PREF", -1}
	,
	{"PM_L3SA_HIT", -1}
	,
	{"PM_L3SA_MOD_INV", -1}
	,
	{"PM_L3SA_SHR_INV", -1}
	,
	{"PM_L3SA_SNOOP_RETRY", -1}
	,
	{"PM_L3SB_HIT", -1}
	,
	{"PM_L3SB_MOD_INV", -1}
	,
	{"PM_L3SB_SHR_INV", -1}
	,
	{"PM_L3SB_SNOOP_RETRY", -1}
	,
	{"PM_L3SC_HIT", -1}
	,
	{"PM_L3SC_MOD_INV", -1}
	,
	{"PM_L3SC_SHR_INV", -1}
	,
	{"PM_L3SC_SNOOP_RETRY", -1}
	,
	{"PM_LD_MISS_L1", -1}
	,
	{"PM_LD_MISS_L1_LSU0", -1}
	,
	{"PM_LD_MISS_L1_LSU1", -1}
	,
	{"PM_LD_REF_L1_LSU0", -1}
	,
	{"PM_LD_REF_L1_LSU1", -1}
	,
	{"PM_LSU0_LDF", -1}
	,
	{"PM_LSU0_NCLD", -1}
	,
	{"PM_LSU1_LDF", -1}
	,
	{"PM_LSU1_NCLD", -1}
	,
	{"PM_LSU_FLUSH", -1}
	,
	{"PM_LSU_FLUSH_SRQ_FULL", -1}
	,
	{"PM_LSU_LMQ_FULL_CYC", -1}
	,
	{"PM_LSU_LMQ_LHR_MERGE", -1}
	,
	{"PM_LSU_LMQ_S0_ALLOC", -1}
	,
	{"PM_LSU_LMQ_S0_VALID", -1}
	,
	{"PM_LSU_LRQ_FULL_CYC", -1}
	,
	{"PM_DC_PREF_STREAM_ALLOC_BLK", -1}
	,
	{"PM_LSU_SRQ_FULL_CYC", -1}
	,
	{"PM_LSU_SRQ_SYNC_CYC", -1}
	,
	{"PM_LWSYNC_HELD", -1}
	,
	{"PM_MEM_LO_PRIO_PW_CMPL", -1}
	,
	{"PM_MEM_LO_PRIO_WR_CMPL", -1}
	,
	{"PM_MEM_PW_CMPL", -1}
	,
	{"PM_MEM_PW_GATH", -1}
	,
	{"PM_MEM_RQ_DISP_BUSY1to7", -1}
	,
	{"PM_MEM_SPEC_RD_CANCEL", -1}
	,
	{"PM_MEM_WQ_DISP_BUSY8to15", -1}
	,
	{"PM_MEM_WQ_DISP_DCLAIM", -1}
	,
	{"PM_MRK_DATA_FROM_L275_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_L375_SHR", -1}
	,
	{"PM_MRK_DSLB_MISS", -1}
	,
	{"PM_MRK_DTLB_MISS", -1}
	,
	{"PM_MRK_FPU_FIN", -1}
	,
	{"PM_MRK_INST_FIN", -1}
	,
	{"PM_MRK_L1_RELOAD_VALID", -1}
	,
	{"PM_MRK_LSU0_FLUSH_LRQ", -1}
	,
	{"PM_MRK_LSU0_FLUSH_SRQ", -1}
	,
	{"PM_MRK_LSU0_FLUSH_UST", -1}
	,
	{"PM_MRK_LSU0_FLUSH_ULD", -1}
	,
	{"PM_MRK_LSU1_FLUSH_LRQ", -1}
	,
	{"PM_MRK_LSU1_FLUSH_SRQ", -1}
	,
	{"PM_MRK_LSU1_FLUSH_ULD", -1}
	,
	{"PM_MRK_LSU1_FLUSH_UST", -1}
	,
	{"PM_MRK_LSU_FLUSH_LRQ", -1}
	,
	{"PM_MRK_LSU_FLUSH_UST", -1}
	,
	{"PM_MRK_LSU_SRQ_INST_VALID", -1}
	,
	{"PM_MRK_ST_CMPL_INT", -1}
	,
	{"PM_PMC2_OVERFLOW", -1}
	,
	{"PM_PMC6_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_L275_SHR", -1}
	,
	{"PM_PTEG_FROM_L375_SHR", -1}
	,
	{"PM_SNOOP_PARTIAL_RTRY_QFULL", -1}
	,
	{"PM_SNOOP_PW_RETRY_WQ_PWQ", -1}
	,
	{"PM_SNOOP_RD_RETRY_WQ", -1}
	,
	{"PM_SNOOP_WR_RETRY_QFULL", -1}
	,
	{"PM_SNOOP_WR_RETRY_WQ", -1}
	,
	{"PM_STOP_COMPLETION", -1}
	,
	{"PM_ST_MISS_L1", -1}
	,
	{"PM_ST_REF_L1", -1}
	,
	{"PM_ST_REF_L1_LSU0", -1}
	,
	{"PM_ST_REF_L1_LSU1", -1}
	,
	{"PM_CLB_EMPTY_CYC", -1}
	,
	{"PM_THRD_L2MISS_BOTH_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_0_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_1or2_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_3or4_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_5or6_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus1or2_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus3or4_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus5or6_CYC", -1}
	,
	{"PM_THRD_SEL_OVER_CLB_EMPTY", -1}
	,
	{"PM_THRD_SEL_OVER_GCT_IMBAL", -1}
	,
	{"PM_THRD_SEL_OVER_ISU_HOLD", -1}
	,
	{"PM_THRD_SEL_OVER_L2MISS", -1}
	,
	{"PM_THRD_SEL_T0", -1}
	,
	{"PM_THRD_SEL_T1", -1}
	,
	{"PM_THRD_SMT_HANG", -1}
	,
	{"PM_THRESH_TIMEO", -1}
	,
	{"PM_TLBIE_HELD", -1}
	,
	{"PM_DATA_FROM_L2MISS", -1}
	,
	{"PM_MRK_DATA_FROM_L2MISS", -1}
	,
	{"PM_PTEG_FROM_L2MISS", -1}
	,
	{"PM_0INST_FETCH", -1}
	,
	{"PM_BR_PRED_CR_TA", -1}
	,
	{"PM_CMPLU_STALL_DIV", -1}
	,
	{"PM_CMPLU_STALL_ERAT_MISS", -1}
	,
	{"PM_CMPLU_STALL_FPU", -1}
	,
	{"PM_CMPLU_STALL_REJECT", -1}
	,
	{"PM_EXT_INT", -1}
	,
	{"PM_FPU_FEST", -1}
	,
	{"PM_FPU_FIN", -1}
	,
	{"PM_FXLS_FULL_CYC", -1}
	,
	{"PM_FXU1_BUSY_FXU0_IDLE", -1}
	,
	{"PM_GCT_NOSLOT_BR_MPRED", -1}
	,
	{"PM_INST_FROM_L275_MOD", -1}
	,
	{"PM_INST_FROM_L375_MOD", -1}
	,
	{"PM_INST_FROM_RMEM", -1}
	,
	{"PM_LD_REF_L1", -1}
	,
	{"PM_LSU_LDF", -1}
	,
	{"PM_LSU_SRQ_EMPTY_CYC", -1}
	,
	{"PM_MRK_CRU_FIN", -1}
	,
	{"PM_MRK_DATA_FROM_L25_MOD_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L275_MOD_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L35_MOD_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_L375_MOD_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_LMEM_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_RMEM_CYC", -1}
	,
	{"PM_MRK_GRP_CMPL", -1}
	,
	{"PM_MRK_GRP_IC_MISS", -1}
	,
	{"PM_MRK_GRP_TIMEO", -1}
	,
	{"PM_MRK_LSU_FIN", -1}
	,
	{"PM_MRK_LSU_FLUSH_SRQ", -1}
	,
	{"PM_MRK_LSU_FLUSH_ULD", -1}
	,
	{"PM_PMC3_OVERFLOW", -1}
	,
	{"PM_WORK_HELD", -1}
};
