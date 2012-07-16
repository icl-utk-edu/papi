/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    power6_events_map.c
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* (C) Copyright IBM Corporation, 2007.  All Rights Reserved.
* Contributed by Corey Ashford <cjashfor@us.ibm.com>
*
* This file MUST be kept synchronised with the events file.
*
*/
#include "perfctr-ppc64.h"

PPC64_native_map_t native_name_map[PAPI_MAX_NATIVE_EVENTS] = {
	{"PM_0INST_FETCH", -1}
	,
	{"PM_1PLUS_PPC_CMPL", -1}
	,
	{"PM_1PLUS_PPC_DISP", -1}
	,
	{"PM_BRU_FIN", -1}
	,
	{"PM_BR_MPRED_CCACHE", -1}
	,
	{"PM_BR_MPRED_COUNT", -1}
	,
	{"PM_BR_MPRED_CR", -1}
	,
	{"PM_BR_MPRED_TA", -1}
	,
	{"PM_BR_PRED", -1}
	,
	{"PM_BR_PRED_CCACHE", -1}
	,
	{"PM_BR_PRED_CR", -1}
	,
	{"PM_BR_PRED_LSTACK", -1}
	,
	{"PM_CYC", -1}
	,
	{"PM_DATA_FROM_L2", -1}
	,
	{"PM_DATA_FROM_L35_MOD", -1}
	,
	{"PM_DATA_FROM_MEM_DP", -1}
	,
	{"PM_DATA_FROM_RL2L3_MOD", -1}
	,
	{"PM_DATA_PTEG_1ST_HALF", -1}
	,
	{"PM_DATA_PTEG_2ND_HALF", -1}
	,
	{"PM_DATA_PTEG_SECONDARY", -1}
	,
	{"PM_DC_INV_L2", -1}
	,
	{"PM_DC_PREF_OUT_OF_STREAMS", -1}
	,
	{"PM_DC_PREF_STREAM_ALLOC", -1}
	,
	{"PM_DFU_ADD", -1}
	,
	{"PM_DFU_ADD_SHIFTED_BOTH", -1}
	,
	{"PM_DFU_BACK2BACK", -1}
	,
	{"PM_DFU_CONV", -1}
	,
	{"PM_DFU_ENC_BCD_DPD", -1}
	,
	{"PM_DFU_EXP_EQ", -1}
	,
	{"PM_DFU_FIN", -1}
	,
	{"PM_DFU_SUBNORM", -1}
	,
	{"PM_DPU_HELD_COMPLETION", -1}
	,
	{"PM_DPU_HELD_CR_LOGICAL", -1}
	,
	{"PM_DPU_HELD_CW", -1}
	,
	{"PM_DPU_HELD_FPQ", -1}
	,
	{"PM_DPU_HELD_FPU_CR", -1}
	,
	{"PM_DPU_HELD_FP_FX_MULT", -1}
	,
	{"PM_DPU_HELD_FXU_MULTI", -1}
	,
	{"PM_DPU_HELD_FXU_SOPS", -1}
	,
	{"PM_DPU_HELD_GPR", -1}
	,
	{"PM_DPU_HELD_INT", -1}
	,
	{"PM_DPU_HELD_ISYNC", -1}
	,
	{"PM_DPU_HELD_ITLB_ISLB", -1}
	,
	{"PM_DPU_HELD_LLA_END", -1}
	,
	{"PM_DPU_HELD_LSU", -1}
	,
	{"PM_DPU_HELD_LSU_SOPS", -1}
	,
	{"PM_DPU_HELD_MULT_GPR", -1}
	,
	{"PM_DPU_HELD_RESTART", -1}
	,
	{"PM_DPU_HELD_RU_WQ", -1}
	,
	{"PM_DPU_HELD_SMT", -1}
	,
	{"PM_DPU_HELD_SPR", -1}
	,
	{"PM_DPU_HELD_STCX_CR", -1}
	,
	{"PM_DPU_HELD_THERMAL", -1}
	,
	{"PM_DPU_HELD_THRD_PRIO", -1}
	,
	{"PM_DPU_HELD_XER", -1}
	,
	{"PM_DPU_HELD_XTHRD", -1}
	,
	{"PM_DSLB_MISS", -1}
	,
	{"PM_EE_OFF_EXT_INT", -1}
	,
	{"PM_FAB_ADDR_COLLISION", -1}
	,
	{"PM_FAB_CMD_ISSUED", -1}
	,
	{"PM_FAB_DCLAIM", -1}
	,
	{"PM_FAB_DMA", -1}
	,
	{"PM_FAB_MMIO", -1}
	,
	{"PM_FAB_NODE_PUMP", -1}
	,
	{"PM_FAB_RETRY_NODE_PUMP", -1}
	,
	{"PM_FAB_RETRY_SYS_PUMP", -1}
	,
	{"PM_FAB_SYS_PUMP", -1}
	,
	{"PM_FLUSH", -1}
	,
	{"PM_FLUSH_ASYNC", -1}
	,
	{"PM_FLUSH_FPU", -1}
	,
	{"PM_FLUSH_FXU", -1}
	,
	{"PM_FPU0_1FLOP", -1}
	,
	{"PM_FPU0_DENORM", -1}
	,
	{"PM_FPU0_FCONV", -1}
	,
	{"PM_FPU0_FEST", -1}
	,
	{"PM_FPU0_FIN", -1}
	,
	{"PM_FPU0_FLOP", -1}
	,
	{"PM_FPU0_FMA", -1}
	,
	{"PM_FPU0_FPSCR", -1}
	,
	{"PM_FPU0_FRSP", -1}
	,
	{"PM_FPU0_FSQRT_FDIV", -1}
	,
	{"PM_FPU0_FXDIV", -1}
	,
	{"PM_FPU0_FXMULT", -1}
	,
	{"PM_FPU0_SINGLE", -1}
	,
	{"PM_FPU0_STF", -1}
	,
	{"PM_FPU0_ST_FOLDED", -1}
	,
	{"PM_FPU1_1FLOP", -1}
	,
	{"PM_FPU1_DENORM", -1}
	,
	{"PM_FPU1_FCONV", -1}
	,
	{"PM_FPU1_FEST", -1}
	,
	{"PM_FPU1_FIN", -1}
	,
	{"PM_FPU1_FLOP", -1}
	,
	{"PM_FPU1_FMA", -1}
	,
	{"PM_FPU1_FPSCR", -1}
	,
	{"PM_FPU1_FRSP", -1}
	,
	{"PM_FPU1_FSQRT_FDIV", -1}
	,
	{"PM_FPU1_FXDIV", -1}
	,
	{"PM_FPU1_FXMULT", -1}
	,
	{"PM_FPU1_SINGLE", -1}
	,
	{"PM_FPU1_STF", -1}
	,
	{"PM_FPU1_ST_FOLDED", -1}
	,
	{"PM_FPU_1FLOP", -1}
	,
	{"PM_FPU_FCONV", -1}
	,
	{"PM_FPU_FIN", -1}
	,
	{"PM_FPU_FLOP", -1}
	,
	{"PM_FPU_FXDIV", -1}
	,
	{"PM_FPU_FXMULT", -1}
	,
	{"PM_FPU_ISSUE_0", -1}
	,
	{"PM_FPU_ISSUE_1", -1}
	,
	{"PM_FPU_ISSUE_2", -1}
	,
	{"PM_FPU_ISSUE_DIV_SQRT_OVERLAP", -1}
	,
	{"PM_FPU_ISSUE_OOO", -1}
	,
	{"PM_FPU_ISSUE_STALL_FPR", -1}
	,
	{"PM_FPU_ISSUE_STALL_ST", -1}
	,
	{"PM_FPU_ISSUE_STALL_THRD", -1}
	,
	{"PM_FPU_ISSUE_STEERING", -1}
	,
	{"PM_FPU_ISSUE_ST_FOLDED", -1}
	,
	{"PM_FXU_IDLE", -1}
	,
	{"PM_FXU_PIPELINED_MULT_DIV", -1}
	,
	{"PM_GCT_EMPTY_CYC", -1}
	,
	{"PM_GCT_FULL_CYC", -1}
	,
	{"PM_GCT_NOSLOT_CYC", -1}
	,
	{"PM_GXI_ADDR_CYC_BUSY", -1}
	,
	{"PM_GXI_CYC_BUSY", -1}
	,
	{"PM_GXI_DATA_CYC_BUSY", -1}
	,
	{"PM_GXO_ADDR_CYC_BUSY", -1}
	,
	{"PM_GXO_CYC_BUSY", -1}
	,
	{"PM_GXO_DATA_CYC_BUSY", -1}
	,
	{"PM_GX_DMA_READ", -1}
	,
	{"PM_GX_DMA_WRITE", -1}
	,
	{"PM_IBUF_FULL_CYC", -1}
	,
	{"PM_IC_DEMAND_L2_BHT_REDIRECT", -1}
	,
	{"PM_IC_DEMAND_L2_BR_REDIRECT", -1}
	,
	{"PM_IC_PREF_REQ", -1}
	,
	{"PM_IC_PREF_WRITE", -1}
	,
	{"PM_IC_RELOAD_SHR", -1}
	,
	{"PM_IC_REQ", -1}
	,
	{"PM_IERAT_MISS", -1}
	,
	{"PM_IFU_FIN", -1}
	,
	{"PM_INST_CMPL", -1}
	,
	{"PM_INST_DISP_LLA", -1}
	,
	{"PM_INST_FETCH_CYC", -1}
	,
	{"PM_INST_FROM_L1", -1}
	,
	{"PM_INST_FROM_L2", -1}
	,
	{"PM_INST_FROM_L35_MOD", -1}
	,
	{"PM_INST_FROM_MEM_DP", -1}
	,
	{"PM_INST_FROM_RL2L3_MOD", -1}
	,
	{"PM_INST_IMC_MATCH_CMPL", -1}
	,
	{"PM_INST_PTEG_1ST_HALF", -1}
	,
	{"PM_INST_PTEG_2ND_HALF", -1}
	,
	{"PM_INST_PTEG_SECONDARY", -1}
	,
	{"PM_INST_TABLEWALK_CYC", -1}
	,
	{"PM_ISLB_MISS", -1}
	,
	{"PM_ITLB_REF", -1}
	,
	{"PM_L1_ICACHE_MISS", -1}
	,
	{"PM_L1_PREF", -1}
	,
	{"PM_L1_WRITE_CYC", -1}
	,
	{"PM_L2SA_CASTOUT_MOD", -1}
	,
	{"PM_L2SA_CASTOUT_SHR", -1}
	,
	{"PM_L2SA_DC_INV", -1}
	,
	{"PM_L2SA_IC_INV", -1}
	,
	{"PM_L2SA_LD_HIT", -1}
	,
	{"PM_L2SA_LD_MISS_DATA", -1}
	,
	{"PM_L2SA_LD_MISS_INST", -1}
	,
	{"PM_L2SA_LD_REQ", -1}
	,
	{"PM_L2SA_LD_REQ_DATA", -1}
	,
	{"PM_L2SA_LD_REQ_INST", -1}
	,
	{"PM_L2SA_MISS", -1}
	,
	{"PM_L2SA_ST_HIT", -1}
	,
	{"PM_L2SA_ST_MISS", -1}
	,
	{"PM_L2SA_ST_REQ", -1}
	,
	{"PM_L2SB_CASTOUT_MOD", -1}
	,
	{"PM_L2SB_CASTOUT_SHR", -1}
	,
	{"PM_L2SB_DC_INV", -1}
	,
	{"PM_L2SB_IC_INV", -1}
	,
	{"PM_L2SB_LD_HIT", -1}
	,
	{"PM_L2SB_LD_MISS_DATA", -1}
	,
	{"PM_L2SB_LD_MISS_INST", -1}
	,
	{"PM_L2SB_LD_REQ", -1}
	,
	{"PM_L2SB_LD_REQ_DATA", -1}
	,
	{"PM_L2SB_LD_REQ_INST", -1}
	,
	{"PM_L2SB_MISS", -1}
	,
	{"PM_L2SB_ST_HIT", -1}
	,
	{"PM_L2SB_ST_MISS", -1}
	,
	{"PM_L2SB_ST_REQ", -1}
	,
	{"PM_L2_CASTOUT_MOD", -1}
	,
	{"PM_L2_LD_REQ_DATA", -1}
	,
	{"PM_L2_LD_REQ_INST", -1}
	,
	{"PM_L2_PREF_LD", -1}
	,
	{"PM_L2_PREF_ST", -1}
	,
	{"PM_L2_ST_MISS_DATA", -1}
	,
	{"PM_L3SA_HIT", -1}
	,
	{"PM_L3SA_MISS", -1}
	,
	{"PM_L3SA_REF", -1}
	,
	{"PM_L3SB_HIT", -1}
	,
	{"PM_L3SB_MISS", -1}
	,
	{"PM_L3SB_REF", -1}
	,
	{"PM_LARX", -1}
	,
	{"PM_LARX_L1HIT", -1}
	,
	{"PM_LD_MISS_L1", -1}
	,
	{"PM_LD_MISS_L1_CYC", -1}
	,
	{"PM_LD_REF_L1", -1}
	,
	{"PM_LD_REF_L1_BOTH", -1}
	,
	{"PM_LD_REQ_L2", -1}
	,
	{"PM_LSU0_DERAT_MISS", -1}
	,
	{"PM_LSU0_LDF", -1}
	,
	{"PM_LSU0_NCLD", -1}
	,
	{"PM_LSU0_NCST", -1}
	,
	{"PM_LSU0_REJECT", -1}
	,
	{"PM_LSU0_REJECT_DERAT_MPRED", -1}
	,
	{"PM_LSU0_REJECT_EXTERN", -1}
	,
	{"PM_LSU0_REJECT_L2MISS", -1}
	,
	{"PM_LSU0_REJECT_L2_CORR", -1}
	,
	{"PM_LSU0_REJECT_LHS", -1}
	,
	{"PM_LSU0_REJECT_NO_SCRATCH", -1}
	,
	{"PM_LSU0_REJECT_PARTIAL_SECTOR", -1}
	,
	{"PM_LSU0_REJECT_SET_MPRED", -1}
	,
	{"PM_LSU0_REJECT_STQ_FULL", -1}
	,
	{"PM_LSU0_REJECT_ULD", -1}
	,
	{"PM_LSU0_REJECT_UST", -1}
	,
	{"PM_LSU1_DERAT_MISS", -1}
	,
	{"PM_LSU1_LDF", -1}
	,
	{"PM_LSU1_REJECT", -1}
	,
	{"PM_LSU1_REJECT_DERAT_MPRED", -1}
	,
	{"PM_LSU1_REJECT_EXTERN", -1}
	,
	{"PM_LSU1_REJECT_L2_CORR", -1}
	,
	{"PM_LSU1_REJECT_LHS", -1}
	,
	{"PM_LSU1_REJECT_NO_SCRATCH", -1}
	,
	{"PM_LSU1_REJECT_PARTIAL_SECTOR", -1}
	,
	{"PM_LSU1_REJECT_SET_MPRED", -1}
	,
	{"PM_LSU1_REJECT_STQ_FULL", -1}
	,
	{"PM_LSU1_REJECT_ULD", -1}
	,
	{"PM_LSU1_REJECT_UST", -1}
	,
	{"PM_LSU_BOTH_BUS", -1}
	,
	{"PM_LSU_DERAT_MISS_CYC", -1}
	,
	{"PM_LSU_FLUSH_ALIGN", -1}
	,
	{"PM_LSU_FLUSH_DSI", -1}
	,
	{"PM_LSU_LDF_BOTH", -1}
	,
	{"PM_LSU_LMQ_FULL_CYC", -1}
	,
	{"PM_LSU_REJECT_L2_CORR", -1}
	,
	{"PM_LSU_REJECT_LHS", -1}
	,
	{"PM_LSU_REJECT_PARTIAL_SECTOR", -1}
	,
	{"PM_LSU_REJECT_STEAL", -1}
	,
	{"PM_LSU_REJECT_STQ_FULL", -1}
	,
	{"PM_LSU_REJECT_ULD", -1}
	,
	{"PM_LSU_REJECT_UST_BOTH", -1}
	,
	{"PM_LSU_ST_CHAINED", -1}
	,
	{"PM_LWSYNC", -1}
	,
	{"PM_MEM0_DP_CL_WR_GLOB", -1}
	,
	{"PM_MEM0_DP_CL_WR_LOC", -1}
	,
	{"PM_MEM0_DP_RQ_GLOB_LOC", -1}
	,
	{"PM_MEM0_DP_RQ_LOC_GLOB", -1}
	,
	{"PM_MEM1_DP_CL_WR_GLOB", -1}
	,
	{"PM_MEM1_DP_CL_WR_LOC", -1}
	,
	{"PM_MEM1_DP_RQ_GLOB_LOC", -1}
	,
	{"PM_MEM1_DP_RQ_LOC_GLOB", -1}
	,
	{"PM_MEM_DP_CL_WR_LOC", -1}
	,
	{"PM_MEM_DP_RQ_GLOB_LOC", -1}
	,
	{"PM_MRK_BR_TAKEN", -1}
	,
	{"PM_MRK_DATA_FROM_L2", -1}
	,
	{"PM_MRK_DATA_FROM_L2MISS", -1}
	,
	{"PM_MRK_DATA_FROM_L35_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_MEM_DP", -1}
	,
	{"PM_MRK_DATA_FROM_RL2L3_MOD", -1}
	,
	{"PM_MRK_DTLB_REF", -1}
	,
	{"PM_MRK_FPU0_FIN", -1}
	,
	{"PM_MRK_FPU1_FIN", -1}
	,
	{"PM_MRK_INST_DISP", -1}
	,
	{"PM_MRK_INST_ISSUED", -1}
	,
	{"PM_MRK_LSU0_REJECT_L2MISS", -1}
	,
	{"PM_MRK_LSU0_REJECT_LHS", -1}
	,
	{"PM_MRK_LSU0_REJECT_ULD", -1}
	,
	{"PM_MRK_LSU0_REJECT_UST", -1}
	,
	{"PM_MRK_LSU1_REJECT_LHS", -1}
	,
	{"PM_MRK_LSU1_REJECT_ULD", -1}
	,
	{"PM_MRK_LSU1_REJECT_UST", -1}
	,
	{"PM_MRK_LSU_REJECT_ULD", -1}
	,
	{"PM_MRK_PTEG_FROM_L2", -1}
	,
	{"PM_MRK_PTEG_FROM_L35_MOD", -1}
	,
	{"PM_MRK_PTEG_FROM_MEM_DP", -1}
	,
	{"PM_MRK_PTEG_FROM_RL2L3_MOD", -1}
	,
	{"PM_MRK_STCX_FAIL", -1}
	,
	{"PM_MRK_ST_CMPL", -1}
	,
	{"PM_MRK_VMX0_LD_WRBACK", -1}
	,
	{"PM_MRK_VMX1_LD_WRBACK", -1}
	,
	{"PM_MRK_VMX_COMPLEX_ISSUED", -1}
	,
	{"PM_MRK_VMX_FLOAT_ISSUED", -1}
	,
	{"PM_MRK_VMX_PERMUTE_ISSUED", -1}
	,
	{"PM_MRK_VMX_SIMPLE_ISSUED", -1}
	,
	{"PM_MRK_VMX_ST_ISSUED", -1}
	,
	{"PM_NO_ITAG_CYC", -1}
	,
	{"PM_PMC2_SAVED", -1}
	,
	{"PM_PMC4_OVERFLOW", -1}
	,
	{"PM_PMC4_REWIND", -1}
	,
	{"PM_PMC5_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_L2", -1}
	,
	{"PM_PTEG_FROM_L2MISS", -1}
	,
	{"PM_PTEG_FROM_L35_MOD", -1}
	,
	{"PM_PTEG_FROM_MEM_DP", -1}
	,
	{"PM_PTEG_FROM_RL2L3_MOD", -1}
	,
	{"PM_PTEG_RELOAD_VALID", -1}
	,
	{"PM_PURR", -1}
	,
	{"PM_RUN_CYC", -1}
	,
	{"PM_SLB_MISS", -1}
	,
	{"PM_STCX", -1}
	,
	{"PM_STCX_CANCEL", -1}
	,
	{"PM_STCX_FAIL", -1}
	,
	{"PM_ST_FIN", -1}
	,
	{"PM_ST_HIT_L2", -1}
	,
	{"PM_ST_MISS_L1", -1}
	,
	{"PM_ST_REF_L1", -1}
	,
	{"PM_SUSPENDED", -1}
	,
	{"PM_SYNC_CYC", -1}
	,
	{"PM_TB_BIT_TRANS", -1}
	,
	{"PM_THRD_L2MISS", -1}
	,
	{"PM_THRD_ONE_RUN_CYC", -1}
	,
	{"PM_THRD_PRIO_0_CYC", -1}
	,
	{"PM_THRD_PRIO_7_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_0_CYC", -1}
	,
	{"PM_THRD_SEL_T0", -1}
	,
	{"PM_TLB_REF", -1}
	,
	{"PM_VMX0_INST_ISSUED", -1}
	,
	{"PM_VMX0_LD_ISSUED", -1}
	,
	{"PM_VMX0_LD_WRBACK", -1}
	,
	{"PM_VMX0_STALL", -1}
	,
	{"PM_VMX1_INST_ISSUED", -1}
	,
	{"PM_VMX1_LD_ISSUED", -1}
	,
	{"PM_VMX1_LD_WRBACK", -1}
	,
	{"PM_VMX1_STALL", -1}
	,
	{"PM_VMX_COMPLEX_ISUED", -1}
	,
	{"PM_VMX_FLOAT_ISSUED", -1}
	,
	{"PM_VMX_FLOAT_MULTICYCLE", -1}
	,
	{"PM_VMX_PERMUTE_ISSUED", -1}
	,
	{"PM_VMX_RESULT_SAT_0_1", -1}
	,
	{"PM_VMX_RESULT_SAT_1", -1}
	,
	{"PM_VMX_SIMPLE_ISSUED", -1}
	,
	{"PM_VMX_ST_ISSUED", -1}
	,
	{"PM_0INST_FETCH_COUNT", -1}
	,
	{"PM_IBUF_FULL_COUNT", -1}
	,
	{"PM_GCT_FULL_COUNT", -1}
	,
	{"PM_NO_ITAG_COUNT", -1}
	,
	{"PM_INST_TABLEWALK_COUNT", -1}
	,
	{"PM_SYNC_COUNT", -1}
	,
	{"PM_RUN_COUNT", -1}
	,
	{"PM_THRD_ONE_RUN_COUNT", -1}
	,
	{"PM_LLA_CYC", -1}
	,
	{"PM_NOT_LLA_CYC", -1}
	,
	{"PM_LLA_COUNT", -1}
	,
	{"PM_DPU_HELD_THERMAL_COUNT", -1}
	,
	{"PM_GCT_NOSLOT_COUNT", -1}
	,
	{"PM_DERAT_REF_4K", -1}
	,
	{"PM_DERAT_MISS_4K", -1}
	,
	{"PM_IERAT_MISS_16G", -1}
	,
	{"PM_MRK_DERAT_REF_64K", -1}
	,
	{"PM_MRK_DERAT_MISS_64K", -1}
	,
	{"PM_BR_TAKEN", -1}
	,
	{"PM_DATA_FROM_DL2L3_SHR_CYC", -1}
	,
	{"PM_DATA_FROM_DMEM", -1}
	,
	{"PM_DATA_FROM_DMEM_CYC", -1}
	,
	{"PM_DATA_FROM_L21", -1}
	,
	{"PM_DATA_FROM_L25_SHR_CYC", -1}
	,
	{"PM_DATA_FROM_L2MISS", -1}
	,
	{"PM_DATA_FROM_L2_CYC", -1}
	,
	{"PM_DATA_FROM_L35_SHR", -1}
	,
	{"PM_DATA_FROM_L35_SHR_CYC", -1}
	,
	{"PM_DATA_FROM_L3_CYC", -1}
	,
	{"PM_DATA_FROM_LMEM_CYC", -1}
	,
	{"PM_DATA_FROM_RL2L3_SHR", -1}
	,
	{"PM_DATA_FROM_RL2L3_SHR_CYC", -1}
	,
	{"PM_DPU_HELD", -1}
	,
	{"PM_DPU_HELD_POWER", -1}
	,
	{"PM_DPU_WT_IC_MISS", -1}
	,
	{"PM_EXT_INT", -1}
	,
	{"PM_FAB_CMD_RETRIED", -1}
	,
	{"PM_FPU_DENORM", -1}
	,
	{"PM_FPU_FMA", -1}
	,
	{"PM_FPU_FPSCR", -1}
	,
	{"PM_FPU_FRSP", -1}
	,
	{"PM_FPU_FSQRT_FDIV", -1}
	,
	{"PM_FXU_BUSY", -1}
	,
	{"PM_HV_CYC", -1}
	,
	{"PM_IC_INV_L2", -1}
	,
	{"PM_INST_DISP", -1}
	,
	{"PM_INST_FROM_DMEM", -1}
	,
	{"PM_INST_FROM_L21", -1}
	,
	{"PM_INST_FROM_L35_SHR", -1}
	,
	{"PM_INST_FROM_RL2L3_SHR", -1}
	,
	{"PM_L2_CASTOUT_SHR", -1}
	,
	{"PM_L2_LD_MISS_DATA", -1}
	,
	{"PM_L2_LD_MISS_INST", -1}
	,
	{"PM_L2_MISS", -1}
	,
	{"PM_L2_ST_REQ_DATA", -1}
	,
	{"PM_LD_HIT_L2", -1}
	,
	{"PM_LSU_DERAT_MISS", -1}
	,
	{"PM_LSU_LDF", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_CYC", -1}
	,
	{"PM_LSU_REJECT_DERAT_MPRED", -1}
	,
	{"PM_LSU_REJECT_LHS_BOTH", -1}
	,
	{"PM_LSU_REJECT_NO_SCRATCH", -1}
	,
	{"PM_LSU_REJECT_SET_MPRED", -1}
	,
	{"PM_LSU_REJECT_SLOW", -1}
	,
	{"PM_LSU_REJECT_ULD_BOTH", -1}
	,
	{"PM_LSU_REJECT_UST", -1}
	,
	{"PM_MEM_DP_CL_WR_GLOB", -1}
	,
	{"PM_MEM_DP_RQ_LOC_GLOB", -1}
	,
	{"PM_MRK_DATA_FROM_DMEM", -1}
	,
	{"PM_MRK_DATA_FROM_L21", -1}
	,
	{"PM_MRK_DATA_FROM_L35_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_RL2L3_SHR", -1}
	,
	{"PM_MRK_FPU_FIN", -1}
	,
	{"PM_MRK_FXU_FIN", -1}
	,
	{"PM_MRK_IFU_FIN", -1}
	,
	{"PM_MRK_LD_MISS_L1", -1}
	,
	{"PM_MRK_LSU_REJECT_UST", -1}
	,
	{"PM_MRK_PTEG_FROM_DMEM", -1}
	,
	{"PM_MRK_PTEG_FROM_L21", -1}
	,
	{"PM_MRK_PTEG_FROM_L35_SHR", -1}
	,
	{"PM_MRK_PTEG_FROM_RL2L3_SHR", -1}
	,
	{"PM_MRK_ST_GPS", -1}
	,
	{"PM_PMC1_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_DMEM", -1}
	,
	{"PM_PTEG_FROM_L21", -1}
	,
	{"PM_PTEG_FROM_L35_SHR", -1}
	,
	{"PM_PTEG_FROM_RL2L3_SHR", -1}
	,
	{"PM_ST_REF_L1_BOTH", -1}
	,
	{"PM_ST_REQ_L2", -1}
	,
	{"PM_THRD_GRP_CMPL_BOTH_CYC", -1}
	,
	{"PM_THRD_PRIO_1_CYC", -1}
	,
	{"PM_THRD_PRIO_6_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_1or2_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus1or2_CYC", -1}
	,
	{"PM_HV_COUNT", -1}
	,
	{"PM_DPU_HELD_COUNT", -1}
	,
	{"PM_DPU_HELD_POWER_COUNT", -1}
	,
	{"PM_DPU_WT_IC_MISS_COUNT", -1}
	,
	{"PM_GCT_EMPTY_COUNT", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_COUNT", -1}
	,
	{"PM_DERAT_REF_64K", -1}
	,
	{"PM_DERAT_MISS_64K", -1}
	,
	{"PM_IERAT_MISS_16M", -1}
	,
	{"PM_MRK_DERAT_REF_4K", -1}
	,
	{"PM_MRK_DERAT_MISS_4K", -1}
	,
	{"PM_DATA_FROM_DL2L3_SHR", -1}
	,
	{"PM_DATA_FROM_L25_MOD", -1}
	,
	{"PM_DATA_FROM_L3", -1}
	,
	{"PM_DATA_FROM_L3MISS", -1}
	,
	{"PM_DATA_FROM_RMEM", -1}
	,
	{"PM_DPU_WT", -1}
	,
	{"PM_FPU_STF", -1}
	,
	{"PM_FPU_ST_FOLDED", -1}
	,
	{"PM_FREQ_DOWN", -1}
	,
	{"PM_FXU0_BUSY_FXU1_IDLE", -1}
	,
	{"PM_FXU0_FIN", -1}
	,
	{"PM_INST_FROM_DL2L3_SHR", -1}
	,
	{"PM_INST_FROM_L25_MOD", -1}
	,
	{"PM_INST_FROM_L3", -1}
	,
	{"PM_INST_FROM_L3MISS", -1}
	,
	{"PM_INST_FROM_RMEM", -1}
	,
	{"PM_L1_DCACHE_RELOAD_VALID", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_BOTH_CYC", -1}
	,
	{"PM_LSU_REJECT_EXTERN", -1}
	,
	{"PM_LSU_REJECT_FAST", -1}
	,
	{"PM_MRK_BR_MPRED", -1}
	,
	{"PM_MRK_DATA_FROM_DL2L3_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_L25_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_L3", -1}
	,
	{"PM_MRK_DATA_FROM_L3MISS", -1}
	,
	{"PM_MRK_DATA_FROM_RMEM", -1}
	,
	{"PM_MRK_DFU_FIN", -1}
	,
	{"PM_MRK_INST_FIN", -1}
	,
	{"PM_MRK_PTEG_FROM_DL2L3_SHR", -1}
	,
	{"PM_MRK_PTEG_FROM_L25_MOD", -1}
	,
	{"PM_MRK_PTEG_FROM_L3", -1}
	,
	{"PM_MRK_PTEG_FROM_L3MISS", -1}
	,
	{"PM_MRK_PTEG_FROM_RMEM", -1}
	,
	{"PM_MRK_ST_CMPL_INT", -1}
	,
	{"PM_PMC2_OVERFLOW", -1}
	,
	{"PM_PMC2_REWIND", -1}
	,
	{"PM_PMC4_SAVED", -1}
	,
	{"PM_PMC6_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_DL2L3_SHR", -1}
	,
	{"PM_PTEG_FROM_L25_MOD", -1}
	,
	{"PM_PTEG_FROM_L3", -1}
	,
	{"PM_PTEG_FROM_L3MISS", -1}
	,
	{"PM_PTEG_FROM_RMEM", -1}
	,
	{"PM_THERMAL_MAX", -1}
	,
	{"PM_THRD_CONC_RUN_INST", -1}
	,
	{"PM_THRD_PRIO_2_CYC", -1}
	,
	{"PM_THRD_PRIO_5_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_3or4_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus3or4_CYC", -1}
	,
	{"PM_THRESH_TIMEO", -1}
	,
	{"PM_DPU_WT_COUNT", -1}
	,
	{"PM_LSU_LMQ_SRQ_EMPTY_BOTH_COUNT", -1}
	,
	{"PM_DERAT_REF_16M", -1}
	,
	{"PM_DERAT_MISS_16M", -1}
	,
	{"PM_IERAT_MISS_64K", -1}
	,
	{"PM_MRK_DERAT_REF_16M", -1}
	,
	{"PM_MRK_DERAT_MISS_16M", -1}
	,
	{"PM_BR_MPRED", -1}
	,
	{"PM_DATA_FROM_DL2L3_MOD", -1}
	,
	{"PM_DATA_FROM_DL2L3_MOD_CYC", -1}
	,
	{"PM_DATA_FROM_L21_CYC", -1}
	,
	{"PM_DATA_FROM_L25_SHR", -1}
	,
	{"PM_DATA_FROM_L25_MOD_CYC", -1}
	,
	{"PM_DATA_FROM_L35_MOD_CYC", -1}
	,
	{"PM_DATA_FROM_LMEM", -1}
	,
	{"PM_DATA_FROM_MEM_DP_CYC", -1}
	,
	{"PM_DATA_FROM_RL2L3_MOD_CYC", -1}
	,
	{"PM_DATA_FROM_RMEM_CYC", -1}
	,
	{"PM_DPU_WT_BR_MPRED", -1}
	,
	{"PM_FPU_FEST", -1}
	,
	{"PM_FPU_SINGLE", -1}
	,
	{"PM_FREQ_UP", -1}
	,
	{"PM_FXU1_BUSY_FXU0_IDLE", -1}
	,
	{"PM_FXU1_FIN", -1}
	,
	{"PM_INST_FROM_DL2L3_MOD", -1}
	,
	{"PM_INST_FROM_L25_SHR", -1}
	,
	{"PM_INST_FROM_L2MISS", -1}
	,
	{"PM_INST_FROM_LMEM", -1}
	,
	{"PM_LSU_REJECT", -1}
	,
	{"PM_LSU_SRQ_EMPTY_CYC", -1}
	,
	{"PM_MRK_DATA_FROM_DL2L3_MOD", -1}
	,
	{"PM_MRK_DATA_FROM_L25_SHR", -1}
	,
	{"PM_MRK_DATA_FROM_LMEM", -1}
	,
	{"PM_MRK_INST_TIMEO", -1}
	,
	{"PM_MRK_LSU_DERAT_MISS", -1}
	,
	{"PM_MRK_LSU_FIN", -1}
	,
	{"PM_MRK_LSU_REJECT_LHS", -1}
	,
	{"PM_MRK_PTEG_FROM_DL2L3_MOD", -1}
	,
	{"PM_MRK_PTEG_FROM_L25_SHR", -1}
	,
	{"PM_MRK_PTEG_FROM_L2MISS", -1}
	,
	{"PM_MRK_PTEG_FROM_LMEM", -1}
	,
	{"PM_PMC3_OVERFLOW", -1}
	,
	{"PM_PTEG_FROM_DL2L3_MOD", -1}
	,
	{"PM_PTEG_FROM_L25_SHR", -1}
	,
	{"PM_PTEG_FROM_LMEM", -1}
	,
	{"PM_THRD_BOTH_RUN_CYC", -1}
	,
	{"PM_THRD_LLA_BOTH_CYC", -1}
	,
	{"PM_THRD_PRIO_3_CYC", -1}
	,
	{"PM_THRD_PRIO_4_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_5or6_CYC", -1}
	,
	{"PM_THRD_PRIO_DIFF_minus5or6_CYC", -1}
	,
	{"PM_THRD_BOTH_RUN_COUNT", -1}
	,
	{"PM_DPU_WT_BR_MPRED_COUNT", -1}
	,
	{"PM_LSU_SRQ_EMPTY_COUNT", -1}
	,
	{"PM_DERAT_REF_16G", -1}
	,
	{"PM_DERAT_MISS_16G", -1}
	,
	{"PM_IERAT_MISS_4K", -1}
	,
	{"PM_MRK_DERAT_REF_16G", -1}
	,
	{"PM_MRK_DERAT_MISS_16G", -1}
	,
	{"PM_RUN_PURR", -1}
	,
	{"PM_RUN_INST_CMPL", -1}
};
