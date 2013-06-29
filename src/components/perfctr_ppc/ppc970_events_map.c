/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    ppc970_events_map.c
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
	{"PM_BRQ_FULL_CYC", -1}
	,
	{"PM_CR_MAP_FULL_CYC", -1}
	,
	{"PM_CYC", -1}
	,
	{"PM_DATA_FROM_L2", -1}
	,
	{"PM_DATA_TABLEWALK_CYC", -1}
	,
	{"PM_DSLB_MISS", -1}
	,
	{"PM_DTLB_MISS", -1}
	,
	{"PM_FPR_MAP_FULL_CYC", -1}
	,
	{"PM_FPU0_ALL", -1}
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
	{"PM_FPU1_ALL", -1}
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
	{"PM_GCT_EMPTY_CYC", -1}
	,
	{"PM_GCT_FULL_CYC", -1}
	,
	{"PM_GRP_BR_MPRED", -1}
	,
	{"PM_GRP_BR_REDIR", -1}
	,
	{"PM_GRP_DISP_REJECT", -1}
	,
	{"PM_GRP_DISP_VALID", -1}
	,
	{"PM_IC_PREF_INSTALL", -1}
	,
	{"PM_IC_PREF_REQ", -1}
	,
	{"PM_IERAT_XLATE_WR", -1}
	,
	{"PM_INST_CMPL", -1}
	,
	{"PM_INST_DISP", -1}
	,
	{"PM_INST_FROM_L1", -1}
	,
	{"PM_INST_FROM_L2", -1}
	,
	{"PM_ISLB_MISS", -1}
	,
	{"PM_ITLB_MISS", -1}
	,
	{"PM_LARX_LSU0", -1}
	,
	{"PM_LR_CTR_MAP_FULL_CYC", -1}
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
	{"PM_LSU0_REJECT_SRQ", -1}
	,
	{"PM_LSU0_SRQ_STFWD", -1}
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
	{"PM_LSU1_REJECT_SRQ", -1}
	,
	{"PM_LSU1_SRQ_STFWD", -1}
	,
	{"PM_LSU_FLUSH_ULD", -1}
	,
	{"PM_LSU_LRQ_S0_ALLOC", -1}
	,
	{"PM_LSU_LRQ_S0_VALID", -1}
	,
	{"PM_LSU_REJECT_SRQ", -1}
	,
	{"PM_LSU_SRQ_S0_ALLOC", -1}
	,
	{"PM_LSU_SRQ_S0_VALID", -1}
	,
	{"PM_LSU_SRQ_STFWD", -1}
	,
	{"PM_MRK_DATA_FROM_L2", -1}
	,
	{"PM_MRK_GRP_DISP", -1}
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
	{"PM_PMC8_OVERFLOW", -1}
	,
	{"PM_RUN_CYC", -1}
	,
	{"PM_SNOOP_TLBIE", -1}
	,
	{"PM_STCX_FAIL", -1}
	,
	{"PM_STCX_PASS", -1}
	,
	{"PM_ST_MISS_L1", -1}
	,
	{"PM_SUSPENDED", -1}
	,
	{"PM_XER_MAP_FULL_CYC", -1}
	,
	{"PM_FPU_FMA", -1}
	,
	{"PM_FPU_STALL3", -1}
	,
	{"PM_GCT_EMPTY_SRQ_FULL", -1}
	,
	{"PM_GRP_DISP", -1}
	,
	{"PM_INST_FROM_MEM", -1}
	,
	{"PM_LSU_FLUSH_UST", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_CYC", -1}
	,
	{"PM_LSU_REJECT_LMQ_FULL", -1}
	,
	{"PM_MRK_BRU_FIN", -1}
	,
	{"PM_PMC1_OVERFLOW", -1}
	,
	{"PM_THRESH_TIMEO", -1}
	,
	{"PM_WORK_HELD", -1}
	,
	{"PM_BR_ISSUED", -1}
	,
	{"PM_BR_MPRED_CR", -1}
	,
	{"PM_BR_MPRED_TA", -1}
	,
	{"PM_CRQ_FULL_CYC", -1}
	,
	{"PM_DATA_FROM_MEM", -1}
	,
	{"PM_DC_INV_L2", -1}
	,
	{"PM_DC_PREF_OUT_OF_STREAMS", -1}
	,
	{"PM_DC_PREF_STREAM_ALLOC", -1}
	,
	{"PM_EE_OFF", -1}
	,
	{"PM_EE_OFF_EXT_INT", -1}
	,
	{"PM_FLUSH_BR_MPRED", -1}
	,
	{"PM_FLUSH_LSU_BR_MPRED", -1}
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
	{"PM_FPU_FEST", -1}
	,
	{"PM_FXLS0_FULL_CYC", -1}
	,
	{"PM_FXLS1_FULL_CYC", -1}
	,
	{"PM_FXU0_FIN", -1}
	,
	{"PM_FXU1_FIN", -1}
	,
	{"PM_FXU_FIN", -1}
	,
	{"PM_GPR_MAP_FULL_CYC", -1}
	,
	{"PM_GRP_DISP_BLK_SB_CYC", -1}
	,
	{"PM_HV_CYC", -1}
	,
	{"PM_INST_FROM_PREF", -1}
	,
	{"PM_L1_DCACHE_RELOAD_VALID", -1}
	,
	{"PM_L1_PREF", -1}
	,
	{"PM_L1_WRITE_CYC", -1}
	,
	{"PM_L2_PREF", -1}
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
	{"PM_LSU1_LDF", -1}
	,
	{"PM_LSU_FLUSH", -1}
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
	{"PM_LSU_SRQ_FULL_CYC", -1}
	,
	{"PM_LSU_SRQ_SYNC_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_MEM", -1}
	,
	{"PM_MRK_L1_RELOAD_VALID", -1}
	,
	{"PM_MRK_LSU0_FLUSH_LRQ", -1}
	,
	{"PM_MRK_LSU0_FLUSH_SRQ", -1}
	,
	{"PM_MRK_LSU0_FLUSH_ULD", -1}
	,
	{"PM_MRK_LSU0_FLUSH_UST", -1}
	,
	{"PM_MRK_LSU1_FLUSH_LRQ", -1}
	,
	{"PM_MRK_LSU1_FLUSH_SRQ", -1}
	,
	{"PM_MRK_LSU1_FLUSH_ULD", -1}
	,
	{"PM_MRK_LSU1_FLUSH_UST", -1}
	,
	{"PM_MRK_LSU_SRQ_INST_VALID", -1}
	,
	{"PM_MRK_ST_CMPL_INT", -1}
	,
	{"PM_MRK_VMX_FIN", -1}
	,
	{"PM_PMC2_OVERFLOW", -1}
	,
	{"PM_STOP_COMPLETION", -1}
	,
	{"PM_ST_REF_L1_LSU0", -1}
	,
	{"PM_ST_REF_L1_LSU1", -1}
	,
	{"PM_0INST_FETCH", -1}
	,
	{"PM_FPU_FIN", -1}
	,
	{"PM_FXU1_BUSY_FXU0_IDLE", -1}
	,
	{"PM_LSU_SRQ_EMPTY_CYC", -1}
	,
	{"PM_MRK_CRU_FIN", -1}
	,
	{"PM_MRK_GRP_CMPL", -1}
	,
	{"PM_PMC3_OVERFLOW", -1}
	,
	{"PM_1PLUS_PPC_CMPL", -1}
	,
	{"PM_DATA_FROM_L25_SHR", -1}
	,
	{"PM_FPU_ALL", -1}
	,
	{"PM_FPU_SINGLE", -1}
	,
	{"PM_FXU_IDLE", -1}
	,
	{"PM_GRP_DISP_SUCCESS", -1}
	,
	{"PM_GRP_MRK", -1}
	,
	{"PM_INST_FROM_L25_SHR", -1}
	,
	{"PM_LSU_FLUSH_SRQ", -1}
	,
	{"PM_LSU_REJECT_ERAT_MISS", -1}
	,
	{"PM_MRK_DATA_FROM_L25_SHR", -1}
	,
	{"PM_MRK_GRP_TIMEO", -1}
	,
	{"PM_PMC4_OVERFLOW", -1}
	,
	{"PM_DATA_FROM_L25_MOD", -1}
	,
	{"PM_FPU_FSQRT", -1}
	,
	{"PM_FPU_STF", -1}
	,
	{"PM_FXU_BUSY", -1}
	,
	{"PM_INST_FROM_L25_MOD", -1}
	,
	{"PM_LSU_DERAT_MISS", -1}
	,
	{"PM_LSU_FLUSH_LRQ", -1}
	,
	{"PM_LSU_REJECT_RELOAD_CDF", -1}
	,
	{"PM_MRK_DATA_FROM_L25_MOD", -1}
	,
	{"PM_MRK_FXU_FIN", -1}
	,
	{"PM_MRK_GRP_ISSUED", -1}
	,
	{"PM_MRK_ST_GPS", -1}
	,
	{"PM_PMC5_OVERFLOW", -1}
	,
	{"PM_FPU_FRSP_FCONV", -1}
	,
	{"PM_FXU0_BUSY_FXU1_IDLE", -1}
	,
	{"PM_GRP_CMPL", -1}
	,
	{"PM_MRK_FPU_FIN", -1}
	,
	{"PM_MRK_INST_FIN", -1}
	,
	{"PM_PMC6_OVERFLOW", -1}
	,
	{"PM_ST_REF_L1", -1}
	,
	{"PM_EXT_INT", -1}
	,
	{"PM_FPU_FMOV_FEST", -1}
	,
	{"PM_LD_REF_L1", -1}
	,
	{"PM_LSU_LDF", -1}
	,
	{"PM_MRK_LSU_FIN", -1}
	,
	{"PM_PMC7_OVERFLOW", -1}
	,
	{"PM_TB_BIT_TRANS", -1}
};
