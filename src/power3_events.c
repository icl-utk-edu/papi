/* 
* File:    power3_events.c
* CVS:     
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "power3_events.h"

PWR3_native_map_t native_name_map[MAX_NATNAME_MAP_INDEX] = {
   {"PM_CYC", -1}
   ,
   {"PM_INST_CMPL", -1}
   ,
   {"PM_TB_BIT_TRANS", -1}
   ,
   {"PM_INST_DISP", -1}
   ,
   {"PM_LD_CMPL", -1}
   ,
   {"PM_IC_MISS", -1}
   ,
   {"PM_LD_MISS_L2HIT", -1}
   ,
   {"PM_LD_MISS_EXCEED_NO_L2", -1}
   ,
   {"NUSED", -1}
   ,
   {"PM_ST_MISS_EXCEED_NO_L2", -1}
   ,
   {"PM_BURSTRD_L2MISS_W_INT", -1}
   ,
   {"PM_IC_MISS_USED", -1}
   ,
   {"PM_DU_ECAM_RCAM_OFFSET_HIT", -1}
   ,
   {"PM_GLOBAL_CANCEL_INST_DEL", -1}
   ,
   {"PM_CHAIN_1_TO_8", -1}
   ,
   {"PM_FPU0_BUSY", -1}
   ,
   {"PM_DSLB_MISS", -1}
   ,
   {"PM_LSU0_ISS_TAG_ST", -1}
   ,
   {"PM_TLB_MISS", -1}
   ,
   {"PM_EE_OFF", -1}
   ,
   {"PM_BRU_IDLE", -1}
   ,
   {"PM_SYNCHRO_INST", -1}
   ,
   {"PM_CYC_1STBUF_OCCP", -1}
   ,
   {"PM_SNOOP_L1_M_TO_E_OR_S", -1}
   ,
   {"PM_ST_CMPLBF_AT_GC", -1}
   ,
   {"PM_LINK_STACK_FULL", -1}
   ,
   {"PM_CBR_RESOLV_DISP", -1}
   ,
   {"PM_LD_CMPLBF_AT_GC", -1}
   ,
   {"PM_ENTRY_CMPLBF", -1}
   ,
   {"PM_BIU_ST_RTRY", -1}
   ,
   {"PM_EIEIO_WT_ST", -1}
   ,
   {"PM_I_1_ST_TO_BUS", -1}
   ,
   {"PM_CRB_BUSY_ENT", -1}
   ,
   {"PM_DC_PREF_STREAM_ALLOC_BLK", -1}
   ,
   {"PM_W_1_ST", -1}
   ,
   {"PM_LD_CI", -1}
   ,
   {"PM_4MISS", -1}
   ,
   {"PM_ST_GATH_BYTES", -1}
   ,
   {"PM_DC_HIT_UNDER_MISS", -1}
   ,
   {"PM_INTLEAVE_CONFL_STALLS", -1}
   ,
   {"PM_DU1_REQ_ST_ADDR_XTION", -1}
   ,
   {"PM_BTC_BTL_BLK", -1}
   ,
   {"PM_FPU_SUCCESS_OOO_INST_SCHED", -1}
   ,
   {"PM_FPU_LD_ST_ISSUES", -1}
   ,
   {"PM_FPU_EXEC_FPSCR", -1}
   ,
   {"PM_FPU0_EXEC_FSQRT", -1}
   ,
   {"PM_FPU0_EXEC_ESTIMATE", -1}
   ,
   {"PM_SNOOP_L2ACC", -1}
   ,
   {"PM_DU0_REQ_ST_ADDR_XTION", -1}
   ,
   {"PM_TAG_BURSTRD_L2MISS", -1}
   ,
   {"PM_FPU_IQ_FULL", -1}
   ,
   {"PM_BR_PRED", -1}
   ,
   {"PM_ST_L1MISS", -1}
   ,
   {"PM_LD_MISS_EXCEED_L2", -1}
   ,
   {"PM_L2ACC_BY_RWITM", -1}
   ,
   {"PM_ST_MISS_EXCEED_L2", -1}
   ,
   {"PM_ST_COND_FAIL", -1}
   ,
   {"PM_CI_ST_WT_CI_ST", -1}
   ,
   {"PM_CHAIN_2_TO_1", -1}
   ,
   {"PM_TAG_BURSTRD_L2MISS_W_INT", -1}
   ,
   {"PM_FXU2_IDLE", -1}
   ,
   {"PM_SC_INST", -1}
   ,
   {"PM_2CASTOUT_BF", -1}
   ,
   {"PM_BIU_LD_NORTRY", -1}
   ,
   {"PM_RESRV_RQ", -1}
   ,
   {"PM_SNOOP_E_TO_S", -1}
   ,
   {"PM_IBUF_EMPTY", -1}
   ,
   {"PM_SYNC_CMPLBF_CYC", -1}
   ,
   {"PM_TLBSYNC_CMPLBF_CYC", -1}
   ,
   {"PM_DC_PREF_L2_INV", -1}
   ,
   {"DC_PREF_FILT_1STR", -1}
   ,
   {"PM_ST_CI_PREGATH", -1}
   ,
   {"PM_ST_GATH_HW", -1}
   ,
   {"PM_LD_WT_ADDR_CONF", -1}
   ,
   {"PM_TAG_LD_DATA_RECV", -1}
   ,
   {"PM_FPU1_DENORM", -1}
   ,
   {"PM_FPU1_CMPL", -1}
   ,
   {"PM_FPU_FEST", -1}
   ,
   {"PM_FPU_LD", -1}
   ,
   {"PM_FPU0_FDIV", -1}
   ,
   {"PM_FPU0_FPSCR", -1}
   ,
   {"PM_LD_MISS_L1", -1}
   ,
   {"PM_TAG_ST_L2MISS", -1}
   ,
   {"PM_BRQ_FILLED_CYC", -1}
   ,
   {"PM_TAG_ST_L2MISS_W_INT", -1}
   ,
   {"PM_ST_CMPL", -1}
   ,
   {"PM_TAG_ST_CMPL", -1}
   ,
   {"PM_LD_NEXT", -1}
   ,
   {"PM_ST_L2MISS", -1}
   ,
   {"PM_TAG_BURSTRD_L2ACC", -1}
   ,
   {"PM_CHAIN_3_TO_2", -1}
   ,
   {"PM_UNALIGNED_ST", -1}
   ,
   {"PM_CORE_ST_N_COPYBACK", -1}
   ,
   {"PM_SYNC_RERUN", -1}
   ,
   {"PM_3CASTOUT_BF", -1}
   ,
   {"PM_BIU_RETRY_DU_LOST_RES", -1}
   ,
   {"PM_SNOOP_L2_E_OR_S_TO_I", -1}
   ,
   {"PM_FPU_FDIV", -1}
   ,
   {"PM_IO_INTERPT", -1}
   ,
   {"PM_DC_PREF_HIT", -1}
   ,
   {"PM_DC_PREF_FILT_2STR", -1}
   ,
   {"PM_PREF_MATCH_DEM_MISS", -1}
   ,
   {"PM_LSU1_IDLE", -1}
   ,
   {"PM_FPU0_DENORM", -1}
   ,
   {"PM_LSU0_ISS_TAG_LD", -1}
   ,
   {"PM_TAG_ST_L2ACC", -1}
   ,
   {"PM_LSU0_LD_DATA", -1}
   ,
   {"PM_ST_L2MISS_W_INT", -1}
   ,
   {"PM_SYNC", -1}
   ,
   {"PM_FXU2_BUSY", -1}
   ,
   {"PM_BIU_ST_NORTRY", -1}
   ,
   {"PM_CHAIN_4_TO_3", -1}
   ,
   {"PM_DC_ALIAS_HIT", -1}
   ,
   {"PM_FXU1_IDLE", -1}
   ,
   {"PM_UNALIGNED_LD", -1}
   ,
   {"PM_CMPLU_WT_LD", -1}
   ,
   {"PM_BIU_ARI_RTRY", -1}
   ,
   {"PM_FPU_FSQRT", -1}
   ,
   {"PM_BR_CMPL", -1}
   ,
   {"PM_DISP_BF_EMPTY", -1}
   ,
   {"PM_LNK_REG_STACK_ERR", -1}
   ,
   {"PM_CRLU_PROD_RES", -1}
   ,
   {"PM_TLBSYNC_RERUN", -1}
   ,
   {"PM_SNOOP_L2_M_TO_E_OR_S", -1}
   ,
   {"PM_DEM_FETCH_WT_PREF", -1}
   ,
   {"PM_FPU0_EXEC_FRSP_FCONV", -1}
   ,
   {"PM_IC_HIT", -1}
   ,
   {"PM_0INST_CMPL", -1}
   ,
   {"PM_FPU_DENORM", -1}
   ,
   {"PM_BURSTRD_L2ACC", -1}
   ,
   {"PM_FPU0_CMPL", -1}
   ,
   {"PM_LSU_IDLE", -1}
   ,
   {"PM_BTAC_HITS", -1}
   ,
   {"PM_STQ_FULL", -1}
   ,
   {"PM_BIU_WT_ST_BF", -1}
   ,
   {"PM_SNOOP_L2_M_TO_I", -1}
   ,
   {"PM_FRSP_FCONV_EXEC", -1}
   ,
   {"PM_BIU_ASI_RTRY", -1}
   ,
   {"PM_CHAIN_5_TO_4", -1}
   ,
   {"PM_DC_REQ_HIT_PREF_BUF", -1}
   ,
   {"PM_DC_PREF_FILT_3STR", -1}
   ,
   {"PM_3MISS", -1}
   ,
   {"PM_ST_GATH_WORD", -1}
   ,
   {"PM_LD_WT_ST_CONF", -1}
   ,
   {"PM_LSU1_ISS_TAG_ST", -1}
   ,
   {"PM_FPU1_BUSY", -1}
   ,
   {"PM_FPU0_FMOV_FEST", -1}
   ,
   {"PM_4CASTOUT_BUF", -1}
   ,
   {"PM_ST_L1HIT", -1}
   ,
   {"PM_FXU2_PROD_RESULT", -1}
   ,
   {"PM_BTAC_MISS", -1}
   ,
   {"PM_CBR_DISP", -1}
   ,
   {"PM_LQ_FULL", -1}
   ,
   {"PM_SNOOP_PUSH_INT", -1}
   ,
   {"PM_EE_OFF_EXT_INT", -1}
   ,
   {"PM_BIU_LD_RTRY", -1}
   ,
   {"PM_FPU_EXE_FCMP", -1}
   ,
   {"PM_DC_PREF_BF_INV", -1}
   ,
   {"PM_DC_PREF_FILT_4STR", -1}
   ,
   {"PM_CHAIN_6_TO_5", -1}
   ,
   {"PM_1MISS", -1}
   ,
   {"PM_ST_GATH_DW", -1}
   ,
   {"PM_LSU1_ISS_TAG_LD", -1}
   ,
   {"PM_FPU1_IDLE", -1}
   ,
   {"PM_FPU0_FMA", -1}
   ,
   {"PM_SNOOP_PUSH_BUF", -1}
   ,
   {"PM_FXU0_PROD_RESULT", -1}
   ,
   {"PM_BR_DISP", -1}
   ,
   {"PM_MPRED_BR_CAUSED_GC", -1}
   ,
   {"PM_SNOOP", -1}
   ,
   {"PM_0INST_DISP", -1}
   ,
   {"PM_FXU_IDLE", -1}
   ,
   {"PM_6XX_RTRY_CHNG_TRTP", -1}
   ,
   {"PM_EXEC_FMA", -1}
   ,
   {"PM_ST_DISP", -1}
   ,
   {"PM_DC_PREF_L2HIT", -1}
   ,
   {"PM_CHAIN_7_TO_6", -1}
   ,
   {"PM_DC_PREF_BLOCK_DEMAND_MISS", -1}
   ,
   {"PM_2MISS", -1}
   ,
   {"PM_DC_PREF_USED", -1}
   ,
   {"PM_LSU_WT_SNOOP_BUSY", -1}
   ,
   {"PM_IC_PREF_USED", -1}
   ,
   {"PM_FPU0_FADD_FCMP_FMUL", -1}
   ,
   {"PM_1WT_THRU_BUF_USED", -1}
   ,
   {"PM_SNOOP_L2HIT", -1}
   ,
   {"PM_BURSTRD_L2MISS", -1}
   ,
   {"PM_RESRV_CMPL", -1}
   ,
   {"PM_FXU1_PROD_RESULT", -1}
   ,
   {"PM_RETRY_BUS_OP", -1}
   ,
   {"PM_FPU_IDLE", -1}
   ,
   {"PM_FETCH_CORR_AT_DISPATCH", -1}
   ,
   {"PM_CMPLU_WT_ST", -1}
   ,
   {"PM_FPU_FADD_FMUL", -1}
   ,
   {"PM_LD_DISP", -1}
   ,
   {"PM_ALIGN_INT", -1}
   ,
   {"PM_2WT_THRU_BUF_USED", -1}
   ,
   {"PM_CHAIN_8_TO_7", -1}
   ,
   {"PM_DC_MISS", -1}
   ,
   {"PM_DTLB_MISS", -1}
   ,
   {"PM_ITLB_MISS", -1}
   ,
   {"PM_LD_MISS_CYC", -1}
   ,
   {"PM_BR_FINISH", -1}
   ,
   {"PM_BR_MPRED", -1}
   ,
   {"PM_FXU_CMPL", -1}
   ,
   {"PM_FPU_CMPL", -1}
   ,
   {"PM_LSU_EXEC", -1}
   ,
   {"PM_LD_MISS_DC_SHR", -1}
   ,
   {"PM_WR_HIT_SHR_KILL_BRC", -1}
   ,
   {"PM_MCI_IDLE", -1}
   ,
   {"PM_DPU_WT_IC_MISS", -1}
   ,
   {"PM_4INST_DISP", -1}
   ,
   {"PM_CMPLU_WT_UNF_INST", -1}
   ,
   {"PM_FPU_WT", -1}
   ,
   {"PM_SNOOP_RECV", -1}
   ,
   {"PM_LS_EXEC", -1}
   ,
   {"PM_ST_MISS", -1}
   ,
   {"PM_LSU_CMPL", -1}
   ,
   {"PM_STCX_SUCCESS", -1}
   ,
   {"PM_LARX", -1}
   ,
   {"PM_BR_MPRED_GC", -1}
   ,
   {"PM_FPU_FMA", -1}
   ,
   {"PM_BRU_FIN", -1}
   ,
   {"PM_ST_MISS_L1", -1}
};

pm_info_t pminfo;
native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];

/* to initialize the native_table */
void initialize_native_table()
{
   int i, j;

   memset(native_table, 0, PAPI_MAX_NATIVE_EVENTS * sizeof(native_event_entry_t));
   for (i = 0; i < PAPI_MAX_NATIVE_EVENTS; i++) {
      for (j = 0; j < MAX_COUNTERS; j++)
         native_table[i].resources.counter_cmd[j] = -1;
   }
}


/* to setup native_table values, and return number of entries */
int power3_setup_native_table()
{
   pm_events_t *wevp;
   pm_info_t *info;
   int pmc, ev, i, j, index, retval;

   info = &pminfo;
   index = 0;
   initialize_native_table();
   for (pmc = 0; pmc < info->maxpmcs; pmc++) {
      wevp = info->list_events[pmc];
      for (ev = 0; ev < info->maxevents[pmc]; ev++, wevp++) {
         for (i = 0; i < index; i++) {
            if (strcmp(wevp->short_name, native_table[i].name) == 0) {
               native_table[i].resources.selector |= 1 << pmc;
               native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
               break;
            }
         }
         if (i == index) {
            /*native_table[i].index=i; */
            native_table[i].resources.selector |= 1 << pmc;
            native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
            native_table[i].name = wevp->short_name;
            native_table[i].description = wevp->description;
            index++;
            for (j = 0; j < MAX_NATNAME_MAP_INDEX; j++) {
               if (strcmp(native_table[i].name, native_name_map[j].name) == 0)
                  native_name_map[j].index = i;
            }
         }
      }
   }
   return (PAPI_OK);
}
