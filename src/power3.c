/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This is a cloned file that handles the POWER3 architectures. These changes mostly involve naming differences in the event map.
  The switch between POWER3 and POWER4 is driven by the value _POWER4 which must be defined
  in the make file to conditionally compile for POWER3 or POWER4. Differences between these
  two counting architectures are substantial. Major blocks of conditional code are set off
  by comment lines containing '~~~~~~~~~~~~' characters. Routines that are significantly
  different (in addition to the event map) include:
    find_hwcounter -> find_hwcounter_gps
    setup_all_presets -> setup_p4_presets
    _papi_hwd_add_event
    _papi_hwd_merge
  Other routines also include minor conditionally compiled differences.
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

extern hwi_preset_t _papi_hwd_preset_map[];

/* These defines smooth out the differences between versions of pmtoolkit */

/* Put any modified metrics in the appropriate spot here */
#ifdef PMTOOLKIT_1_2
  #ifdef PMTOOLKIT_1_2_1
    /*#undef  PM_SNOOP*/
    #define PNE_PM_SNOOP       PNE_PM_SNOOP_RECV  /* The name in pre pmtoolkit-1.2.2 */
    #define PNE_PM_LSU_EXEC	   PNE_PM_LS_EXEC
    #define PNE_PM_ST_MISS_L1  PNE_PM_ST_MISS
    #define PNE_PM_MPRED_BR	   PNE_PM_MPRED_BR_CAUSED_GC
  #else
    #define PNE_PM_LSU_EXEC	   PNE_PM_LS_EXEC
    #define PNE_PM_ST_MISS_L1  PNE_PM_ST_MISS
    #define PNE_PM_MPRED_BR	   PNE_PM_MPRED_BR_CAUSED_GC
  #endif /*PMTOOLKIT_1_2_1*/
#else                                  /* pmtoolkit 1.3 and later */
  #ifdef _AIXVERSION_510	       /* AIX Version 5 */
    #define PNE_PM_LSU_EXEC   PNE_PM_LSU_CMPL
    #define PNE_PM_RESRV_CMPL PNE_PM_STCX_SUCCESS
    #define PNE_PM_RESRV_RQ	  PNE_PM_LARX
    #define PNE_PM_MPRED_BR	  PNE_PM_BR_MPRED_GC
    #define PNE_PM_EXEC_FMA	  PNE_PM_FPU_FMA
    #define PNE_PM_BR_FINISH  PNE_PM_BRU_FIN
  #else				       /* AIX Version 4 */
    #define PNE_PM_ST_MISS_L1 PNE_PM_ST_L1MISS
    #define PNE_PM_MPRED_BR	  PNE_PM_MPRED_BR_CAUSED_GC
  #endif /*_AIXVERSION_510*/
#endif /*PMTOOLKIT_1_2*/

#ifdef PAPI_POWER_604
static preset_search_t preset_name_map_604[PAPI_MAX_PRESET_EVENTS] = {
  {PAPI_L1_DCM,0,{PNE_PM_DC_MISS,0,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{PNE_PM_IC_MISS,0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{PNE_PM_DC_MISS,PNE_PM_IC_MISS,0,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PNE_PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_TLB_DM,0,{PNE_PM_DTLB_MISS,0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/	
  {PAPI_TLB_IM,0,{PNE_PM_ITLB_MISS,0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{PNE_PM_DTLB_MISS,PNE_PM_ITLB_MISS,0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/	
  {PAPI_L2_LDM,0,{PNE_PM_LD_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{PNE_PM_ST_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_CSR_SUC,0,{PNE_PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,DERIVED_SUB,{PNE_PM_RESRV_RQ,PNE_PM_RESRV_CMPL,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PNE_PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_RCY,0,{PNE_PM_LD_MISS_CYC,0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_BR_CN,0,{PNE_PM_BR_FINISH,0,0,0,0,0,0,0}}, /*Conditional branch instructions executed*/
  {PAPI_BR_MSP,0,{PNE_PM_BR_MPRED,0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_TOT_IIS,0,{PNE_PM_INST_DISP,0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{PNE_PM_INST_CMPL,0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{PNE_PM_FXU_CMPL,0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{PNE_PM_FPU_CMPL,0,0,0,0,0,0,0}}, /*Floating point instructions executed*/
  {PAPI_LD_INS,0,{PNE_PM_LD_CMPL,0,0,0,0,0,0,0}}, /*Load instructions executed*/
  {PAPI_BR_INS,0,{PNE_PM_BR_CMPL,0,0,0,0,0,0,0}},	/*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_PS,{PNE_PM_CYC,PNE_PM_FPU_CMPL,0,0,0,0,0,0}},	/*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{PNE_PM_CYC,0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{PNE_PM_CYC,PNE_PM_INST_CMPL,0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,0,{PNE_PM_LSU_EXEC,0,0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{PNE_PM_SYNC,0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};
preset_search_t *preset_search_map=preset_name_map_604;
#endif

#ifdef PAPI_POWER_604e
static preset_search_t preset_name_map_604e[PAPI_MAX_PRESET_EVENTS] = {
  {PAPI_L1_DCM,0,{PNE_PM_DC_MISS,0,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{PNE_PM_IC_MISS,0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{PNE_PM_DC_MISS,PNE_PM_IC_MISS,0,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PNE_PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_CA_SHR,0,{PNE_PM_LD_MISS_DC_SHR,0,0,0,0,0,0,0}}, /*Request for shared cache line (SMP)*/		 	
  {PAPI_CA_INV,0,{PNE_PM_WR_HIT_SHR_KILL_BRC,0,0,0,0,0,0,0}}, /*Request for cache line Invalidation (SMP)*/	
  {PAPI_CA_ITV,0,{PNE_PM_WR_HIT_SHR_KILL_BRC,0,0,0,0,0,0,0}}, /*Request for cache line Intervention (SMP)*/
  {PAPI_BRU_IDL,0,{PNE_PM_BRU_IDLE,0,0,0,0,0,0,0}}, /*Cycles branch units are idle*/
  {PAPI_FXU_IDL,0,{PNE_PM_MCI_IDLE,0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/	
  {PAPI_FPU_IDL,0,{PNE_PM_FPU_IDLE,0,0,0,0,0,0,0}}, /*Cycles floating point units are idle*/
  {PAPI_LSU_IDL,0,{PNE_PM_LSU_IDLE,0,0,0,0,0,0,0}}, /*Cycles load/store units are idle*/
  {PAPI_TLB_DM,0,{PNE_PM_DTLB_MISS,0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/	
  {PAPI_TLB_IM,0,{PNE_PM_ITLB_MISS,0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{PNE_PM_DTLB_MISS,PNE_PM_ITLB_MISS,0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/	
  {PAPI_L2_LDM,0,{PNE_PM_LD_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{PNE_PM_ST_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_CSR_SUC,0,{PNE_PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,DERIVED_SUB,{PNE_PM_RESRV_RQ,PNE_PM_RESRV_CMPL,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PNE_PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_SCY,DERIVED_ADD,{PNE_PM_CMPLU_WT_LD,PNE_PM_CMPLU_WT_ST,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Access*/
  {PAPI_MEM_RCY,0,{PNE_PM_CMPLU_WT_LD,0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_MEM_WCY,0,{PNE_PM_CMPLU_WT_ST,0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Write*/
  {PAPI_STL_ICY,0,{PNE_PM_DPU_WT_IC_MISS,0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {PAPI_FUL_ICY,0,{PNE_PM_4INST_DISP,0,0,0,0,0,0,0}}, /*Cycles with Maximum Instruction Issue*/
  {PAPI_STL_CCY,0,{PNE_PM_CMPLU_WT_UNF_INST,0,0,0,0,0,0,0}}, /*Cycles with No Instruction Completion*/
  {PAPI_FUL_CCY,0,{PNE_PM_4INST_DISP,0,0,0,0,0,0,0}}, /*Cycles with Maximum Instruction Completion*/
  {PAPI_BR_CN,0,{PNE_PM_BR_FINISH,0,0,0,0,0,0,0}}, /*Conditional branch instructions executed*/
  {PAPI_BR_MSP,0,{PNE_PM_BR_MPRED,0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_TOT_IIS,0,{PNE_PM_INST_DISP,0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{PNE_PM_INST_CMPL,0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{PNE_PM_FXU_CMPL,0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{PNE_PM_FPU_CMPL,0,0,0,0,0,0,0}}, /*Floating point instructions executed*/
  {PAPI_LD_INS,0,{PNE_PM_LD_CMPL,0,0,0,0,0,0,0}}, /*Load instructions executed*/
  {PAPI_BR_INS,0,{PNE_PM_BR_CMPL,0,0,0,0,0,0,0}},	/*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_PS,{PNE_PM_CYC,PNE_PM_FPU_CMPL,0,0,0,0,0,0}},	/*Floating Point instructions per second*/ 
  {PAPI_FP_STAL,0,{PNE_PM_FPU_WT,0,0,0,0,0,0,0}},	/*Cycles any FP units are stalled */	
  {PAPI_TOT_CYC,0,{PNE_PM_CYC,0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{PNE_PM_CYC,PNE_PM_INST_CMPL,0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,0,{PNE_PM_LSU_EXEC,0,0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{PNE_PM_SYNC,0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};
preset_search_t *preset_search_map=preset_name_map_604e;
#endif

#ifdef PAPI_POWER_630
static preset_search_t preset_name_map_630[PAPI_MAX_PRESET_EVENTS] = { 
  {PAPI_L1_DCM,DERIVED_ADD,{PNE_PM_LD_MISS_L1,PNE_PM_ST_MISS_L1,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{PNE_PM_IC_MISS,0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{PNE_PM_IC_MISS,PNE_PM_LD_MISS_L1,PNE_PM_ST_MISS_L1,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PNE_PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_CA_SHR,0,{PNE_PM_SNOOP_E_TO_S,0,0,0,0,0,0,0}}, /*Request for shared cache line (SMP)*/
  {PAPI_CA_ITV,0,{PNE_PM_SNOOP_PUSH_INT,0,0,0,0,0,0,0}}, /*Request for cache line Intervention (SMP)*/
  {PAPI_BRU_IDL,0,{PNE_PM_BRU_IDLE,0,0,0,0,0,0,0}}, /*Cycles branch units are idle*/
  {PAPI_FXU_IDL,0,{PNE_PM_FXU_IDLE,0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/
  {PAPI_FPU_IDL,0,{PNE_PM_FPU_IDLE,0,0,0,0,0,0,0}}, /*Cycles floating point units are idle*/
  {PAPI_LSU_IDL,0,{PNE_PM_LSU_IDLE,0,0,0,0,0,0,0}}, /*Cycles load/store units are idle*/
  {PAPI_TLB_TL,0,{PNE_PM_TLB_MISS,0,0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/
  {PAPI_L1_LDM,0,{PNE_PM_LD_MISS_L1,0,0,0,0,0,0,0}}, /*Level 1 load misses */
  {PAPI_L1_STM,0,{PNE_PM_ST_MISS_L1,0,0,0,0,0,0,0}}, /*Level 1 store misses */
  {PAPI_L2_LDM,0,{PNE_PM_LD_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{PNE_PM_ST_MISS_EXCEED_L2,0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_BTAC_M,0,{PNE_PM_BTAC_MISS,0,0,0,0,0,0,0}}, /*BTAC miss*/
  {PAPI_PRF_DM,0,{PNE_PM_PREF_MATCH_DEM_MISS,0,0,0,0,0,0,0}}, /*Prefetch data instruction caused a miss */
  {PAPI_TLB_SD,0,{PNE_PM_TLBSYNC_RERUN,0,0,0,0,0,0,0}}, /*Xlation lookaside buffer shootdowns (SMP)*/
  {PAPI_CSR_SUC,0,{PNE_PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,0,{PNE_PM_ST_COND_FAIL,0,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PNE_PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_SCY,DERIVED_ADD,{PNE_PM_CMPLU_WT_LD,PNE_PM_CMPLU_WT_ST,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Access*/
  {PAPI_MEM_RCY,0,{PNE_PM_CMPLU_WT_LD,0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_MEM_WCY,0,{PNE_PM_CMPLU_WT_ST,0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Write*/
  {PAPI_STL_ICY,0,{PNE_PM_0INST_DISP,0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {PAPI_STL_CCY,0,{PNE_PM_0INST_CMPL,0,0,0,0,0,0,0}}, /*Cycles with No Instruction Completion*/
  {PAPI_BR_CN,0,{PNE_PM_CBR_DISP,0,0,0,0,0,0}}, /*Conditional branch instructions executed*/    
  {PAPI_BR_MSP,0,{PNE_PM_MPRED_BR,0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_BR_PRC,0,{PNE_PM_BR_PRED,0,0,0,0,0,0,0}}, /*Conditional branch instructions corr. pred*/
  {PAPI_FMA_INS,0,{PNE_PM_EXEC_FMA,0,0,0,0,0,0,0}}, /*FMA instructions completed*/
  {PAPI_TOT_IIS,0,{PNE_PM_INST_DISP,0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{PNE_PM_INST_CMPL,0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,DERIVED_ADD,{PNE_PM_FXU0_PROD_RESULT,PNE_PM_FXU1_PROD_RESULT,PNE_PM_FXU2_PROD_RESULT,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,DERIVED_ADD,{PNE_PM_FPU0_CMPL,PNE_PM_FPU1_CMPL,0,0,0,0,0,0}}, /*Floating point instructions executed*/	
  {PAPI_LD_INS,0,{PNE_PM_LD_CMPL,0,0,0,0,0,0,0}},	/*Load instructions executed*/
  {PAPI_SR_INS,0,{PNE_PM_ST_CMPL,0,0,0,0,0,0,0}}, /*Store instructions executed*/
  {PAPI_BR_INS,0,{PNE_PM_BR_CMPL,0,0,0,0,0,0,0}}, /*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_ADD_PS,{PNE_PM_CYC,PNE_PM_FPU0_CMPL,PNE_PM_FPU1_CMPL,0,0,0,0,0}}, /*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{PNE_PM_CYC,0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{PNE_PM_CYC,PNE_PM_INST_CMPL,0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,DERIVED_ADD,{PNE_PM_LD_CMPL,PNE_PM_ST_CMPL,0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{PNE_PM_SYNC,0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {PAPI_FDV_INS,0,{PNE_PM_FPU_FDIV,0,0,0,0,0,0,0}}, /*FD ins */
  {PAPI_FSQ_INS,0,{PNE_PM_FPU_FSQRT,0,0,0,0,0,0,0}}, /*FSq ins */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};
preset_search_t *preset_search_map=preset_name_map_630;
#endif

/* Utility functions */

 #define DEBUG_SETUP 

/*static void print_state(hwd_control_state_t *s)
{
  int i;
  
  fprintf(stderr,"\n\n-----------------------------------------\nmaster_selector 0x%x\n",s->master_selector);
  for(i=0;i<POWER_MAX_COUNTERS;i++){
  	if(s->master_selector & (1<<i)) fprintf(stderr, "  1  ");
	else fprintf(stderr, "  0  ");
  }
  fprintf(stderr,"\nnative_event_name       %12s %12s %12s %12s %12s %12s %12s %12s\n",native_table[s->native[0].index].name,native_table[s->native[1].index].name,
    native_table[s->native[2].index].name,native_table[s->native[3].index].name,native_table[s->native[4].index].name,native_table[s->native[5].index].name,native_table[s->native[6].index].name,native_table[s->native[7].index].name);
  fprintf(stderr,"native_event_selectors    %12d %12d %12d %12d %12d %12d %12d %12d\n",native_table[s->native[0].index].selector,native_table[s->native[1].index].selector,
    native_table[s->native[2].index].selector,native_table[s->native[3].index].selector,native_table[s->native[4].index].selector,native_table[s->native[5].index].selector,native_table[s->native[6].index].selector,native_table[s->native[7].index].selector);
  fprintf(stderr,"native_event_position     %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].position,s->native[1].position,
    s->native[2].position,s->native[3].position,s->native[4].position,s->native[5].position,s->native[6].position,s->native[7].position);
  fprintf(stderr,"counters                  %12d %12d %12d %12d %12d %12d %12d %12d\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
  fprintf(stderr,"native links              %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].link,s->native[1].link,
    s->native[2].link,s->native[3].link,s->native[4].link,s->native[5].link,s->native[6].link,s->native[7].link);
  }
}
*/
static int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = MAX_COUNTERS - 1;
 
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if (a->counter_cmd.events[cntr] == b->counter_cmd.events[cntr])
    return(1);

  return(0);
}


/* this function recusively does Modified Bipartite Graph counter allocation 
     success  return 1
	 fail     return 0
*/
static int do_counter_allocation(PWR3_reg_alloc_t *event_list, int size)
{
	int i,j;
	PWR3_reg_alloc_t *queue[MAX_COUNTERS];
	int head, tail;
	
	/* if the event competes 1 counter only, it has priority, map it */
	head=0;
	tail=0;
	for(i=0;i<size;i++){ /* push rank=1 into queue */
		event_list[i].ra_mod=-1;
		if(event_list[i].ra_rank==1){
			queue[tail]=&event_list[i];
			event_list[i].ra_mod=i;
			tail++;
		}
	}
	
	while(head<tail){
		for(i=0;i<size;i++){
			if(i!=(*queue[head]).ra_mod){
				if(event_list[i].ra_selector & (*queue[head]).ra_selector){
					if(event_list[i].ra_rank==1){
						return 0; /* mapping fail, 2 events compete 1 counter only */
					}
					else{
						event_list[i].ra_selector ^= (*queue[head]).ra_selector;
						event_list[i].ra_rank--;
						if(event_list[i].ra_rank==1){
							queue[tail]=&event_list[i];
							event_list[i].ra_mod=i;
							tail++;
						}
					}
				}
			}
		}
		head++;
	}
	if(tail==size){
		return 1; /* successfully mapped */
	}
	else{
		PWR3_reg_alloc_t rest_event_list[MAX_COUNTERS];
		PWR3_reg_alloc_t copy_rest_event_list[MAX_COUNTERS];
		
		j=0;
		for(i=0;i<size;i++){
			if(event_list[i].ra_mod<0){
				memcpy(copy_rest_event_list+j, event_list+i, sizeof(PWR3_reg_alloc_t));
				copy_rest_event_list[j].ra_mod=i;
				j++;
			}
		}
		
		memcpy(rest_event_list, copy_rest_event_list, sizeof(PWR3_reg_alloc_t)*(size-tail));
		
		for(i=0;i<MAX_COUNTERS;i++){
			if(rest_event_list[0].ra_selector & (1<<i)){ /* pick first event on the list, set 1 to 0, to see whether there is an answer */
				for(j=0;j<size-tail;j++){
					if(j==0){
						rest_event_list[j].ra_selector = 1<<i;
						rest_event_list[j].ra_rank = 1;
					}
					else{
						if(rest_event_list[j].ra_selector & (1<<i)){
							rest_event_list[j].ra_selector ^= 1<<i;
							rest_event_list[j].ra_rank--;
						}
					}
				}
				if(do_counter_allocation(rest_event_list, size-tail))
					break;
				
				memcpy(rest_event_list, copy_rest_event_list, sizeof(PWR3_reg_alloc_t)*(size-tail));
			}
		}
		if(i==MAX_COUNTERS){
			return 0; /* fail to find mapping */
		}
		for(i=0;i<size-tail;i++){
			event_list[copy_rest_event_list[i].ra_mod].ra_selector=rest_event_list[i].ra_selector;
		}
		return 1;		
	}
}	
	

/* this function will be called when there are counters available 
     success  return 1
	 fail     return 0
*/      
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI)
{
  hwd_control_state_t *tmp_state = &ESI->machdep;
  unsigned char selector;
  int i, j, natNum;
  PWR3_reg_alloc_t event_list[MAX_COUNTERS];
  int position;

  
  /* not yet successfully mapped, but have enough slots for events */
	
  /* Initialize the local structure needed 
     for counter allocation and optimization. */
  natNum=ESI->NativeCount;
  for(i=0;i<natNum;i++){
    /* CAUTION: Since this is in the hardware layer, it's ok 
       to access the native table directly, but in general this is a bad idea */
    event_list[i].ra_selector = native_table[ESI->NativeInfoArray[i].ni_index].resources.selector;
    /* calculate native event rank, which is number of counters it can live on, this is power3 specific */
    for(j=0;j<MAX_COUNTERS;j++) {
      if(event_list[i].ra_selector & (1<<j))
	event_list[i].ra_rank++;
    }
    event_list[i].ra_mod = -1;
  }

  if(do_counter_allocation(event_list, natNum)){ /* successfully mapped */
	/* update tmp_state, reset... */
	for (i = 0; i <MAX_COUNTERS; i++) {
	    tmp_state->counter_cmd.events[i] = COUNT_NOTHING;
	}
		
	for(i=0;i<natNum;i++){
		/* update tmp_state->native->position */
		position=get_avail_hwcntr_num(event_list[i].ra_selector);
		ESI->NativeInfoArray[i].ni_position=position; 
		/* update tmp_state->counter_cmd */
		/* CAUTION: Since this is in the hardware layer, it's ok 
		   to access the native table directly, but in general this is a bad idea */
		tmp_state->counter_cmd.events[position] = native_table[ESI->NativeInfoArray[i].ni_index].resources.counter_cmd[position];
	}
	return 1;
  }
  else{
	return 0;
  }
}

/* This function clears out in the control structure whatever resources are allocated
    for this native event in the native info structure. */
void _papi_hwd_remove_native(hwd_control_state_t *this_state, NativeInfo_t *nativeInfo)
{
    /* POWER3 has an array of counter values that is sent to the driver. */

    /* Clear the counter command for this counter position */
    this_state->counter_cmd.events[nativeInfo->ni_position]=COUNT_NOTHING;
}


/* This used to be init_config, static to the substrate.
   Now its exposed to the hwi layer and called when an EventSet is allocated.
*/
void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  int i;

  for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
    ptr->counter_cmd.events[i] = COUNT_NOTHING;
 }
  set_domain(ptr,_papi_hwi_system_info.default_domain);
  set_granularity(ptr,_papi_hwi_system_info.default_granularity);
}


int _papi_hwd_update_shlib_info(void)
{
  return PAPI_ESBSTR;
}



