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

#include "allocate.h"

hwd_preset_t _papi_hwd_preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };


/* These defines smooth out the differences between versions of pmtoolkit */

/* Put base definitions of all disputed metrics here */
#define PM_SNOOP	"PM_SNOOP"
#define PM_LSU_EXEC	"PM_LS_EXEC"
#define PM_ST_MISS_L1	"PM_ST_MISS"
#define PM_RESRV_CMPL   "PM_RESRV_CMPL"
#define PM_RESRV_RQ	"PM_RESRV_RQ"
#define PM_MPRED_BR	"PM_MPRED_BR_CAUSED_GC"
#define PM_EXEC_FMA	"PM_EXEC_FMA"
#define PM_BR_FINISH	"PM_BR_FINISH"

/* Put any modified metrics in the appropriate spot here */
#ifdef PMTOOLKIT_1_2
  #ifdef PMTOOLKIT_1_2_1
    #undef  PM_SNOOP
    #define PM_SNOOP    "PM_SNOOP_RECV"  /* The name in pre pmtoolkit-1.2.2 */
  #endif /*PMTOOLKIT_1_2_1*/
#else                                  /* pmtoolkit 1.3 and later */
  #undef  PM_LSU_EXEC
  #undef  PM_ST_MISS_L1
  #ifdef _AIXVERSION_510	       /* AIX Version 5 */
    #undef  PM_RESRV_CMPL
    #undef  PM_RESRV_RQ
    #undef  PM_MPRED_BR
    #undef  PM_EXEC_FMA
    #undef  PM_BR_FINISHF
    #define PM_LSU_EXEC   "PM_LSU_CMPL"
    #define PM_ST_MISS_L1 "PM_ST_MISS_L1"
    #define PM_RESRV_CMPL "PM_STCX_SUCCESS"
    #define PM_RESRV_RQ	  "PM_LARX"
    #define PM_MPRED_BR	  "PM_BR_MPRED_GC"
    #define PM_EXEC_FMA	  "PM_FPU_FMA"
    #define PM_BR_FINISH  "PM_BRU_FIN"
  #else				       /* AIX Version 4 */
    #define PM_LSU_EXEC   "PM_LSU_EXEC"
    #define PM_ST_MISS_L1 "PM_ST_L1MISS"
  #endif /*_AIXVERSION_510*/
#endif /*PMTOOLKIT_1_2*/

static pmapi_search_t preset_name_map_604[PAPI_MAX_PRESET_EVENTS] = {
  {PAPI_L1_DCM,0,{"PM_DC_MISS",0,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{"PM_IC_MISS",0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{"PM_DC_MISS","PM_IC_MISS",0,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_TLB_DM,0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/	
  {PAPI_TLB_IM,0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/	
  {PAPI_L2_LDM,0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_CSR_SUC,0,{PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,DERIVED_SUB,{PM_RESRV_RQ,PM_RESRV_CMPL,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_RCY,0,{"PM_LD_MISS_CYC",0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_BR_CN,0,{PM_BR_FINISH,0,0,0,0,0,0,0}}, /*Conditional branch instructions executed*/
  {PAPI_BR_MSP,0,{"PM_BR_MPRED",0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_TOT_IIS,0,{"PM_INST_DISP",0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{"PM_INST_CMPL",0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{"PM_FXU_CMPL",0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{"PM_FPU_CMPL",0,0,0,0,0,0,0}}, /*Floating point instructions executed*/
  {PAPI_LD_INS,0,{"PM_LD_CMPL",0,0,0,0,0,0,0}}, /*Load instructions executed*/
  {PAPI_BR_INS,0,{"PM_BR_CMPL",0,0,0,0,0,0,0}},	/*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_PS,{"PM_CYC","PM_FPU_CMPL",0,0,0,0,0,0}},	/*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{"PM_CYC",0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,0,{PM_LSU_EXEC,0,0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{"PM_SYNC",0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};

static pmapi_search_t preset_name_map_604e[PAPI_MAX_PRESET_EVENTS] = {
  {PAPI_L1_DCM,0,{"PM_DC_MISS",0,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{"PM_IC_MISS",0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{"PM_DC_MISS","PM_IC_MISS",0,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_CA_SHR,0,{"PM_LD_MISS_DC_SHR",0,0,0,0,0,0,0}}, /*Request for shared cache line (SMP)*/		 	
  {PAPI_CA_INV,0,{"PM_WR_HIT_SHR_KILL_BRC",0,0,0,0,0,0,0}}, /*Request for cache line Invalidation (SMP)*/	
  {PAPI_CA_ITV,0,{"PM_WR_HIT_SHR_KILL_BRC",0,0,0,0,0,0,0}}, /*Request for cache line Intervention (SMP)*/
  {PAPI_BRU_IDL,0,{"PM_BRU_IDLE",0,0,0,0,0,0,0}}, /*Cycles branch units are idle*/
  {PAPI_FXU_IDL,0,{"PM_MCI_IDLE",0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/	
  {PAPI_FPU_IDL,0,{"PM_FPU_IDLE",0,0,0,0,0,0,0}}, /*Cycles floating point units are idle*/
  {PAPI_LSU_IDL,0,{"PM_LSU_IDLE",0,0,0,0,0,0,0}}, /*Cycles load/store units are idle*/
  {PAPI_TLB_DM,0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/	
  {PAPI_TLB_IM,0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/	
  {PAPI_L2_LDM,0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_CSR_SUC,0,{PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,DERIVED_SUB,{PM_RESRV_RQ,PM_RESRV_CMPL,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_SCY,DERIVED_ADD,{"PM_CMPLU_WT_LD","PM_CMPLU_WT_ST",0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Access*/
  {PAPI_MEM_RCY,0,{"PM_CMPLU_WT_LD",0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_MEM_WCY,0,{"PM_CMPLU_WT_ST",0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Write*/
  {PAPI_STL_ICY,0,{"PM_DPU_WT_IC_MISS",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {PAPI_FUL_ICY,0,{"PM_4INST_DISP",0,0,0,0,0,0,0}}, /*Cycles with Maximum Instruction Issue*/
  {PAPI_STL_CCY,0,{"PM_CMPLU_WT_UNF_INST",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Completion*/
  {PAPI_FUL_CCY,0,{"PM_4INST_DISP",0,0,0,0,0,0,0}}, /*Cycles with Maximum Instruction Completion*/
  {PAPI_BR_CN,0,{PM_BR_FINISH,0,0,0,0,0,0,0}}, /*Conditional branch instructions executed*/
  {PAPI_BR_MSP,0,{"PM_BR_MPRED",0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_TOT_IIS,0,{"PM_INST_DISP",0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{"PM_INST_CMPL",0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{"PM_FXU_CMPL",0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{"PM_FPU_CMPL",0,0,0,0,0,0,0}}, /*Floating point instructions executed*/
  {PAPI_LD_INS,0,{"PM_LD_CMPL",0,0,0,0,0,0,0}}, /*Load instructions executed*/
  {PAPI_BR_INS,0,{"PM_BR_CMPL",0,0,0,0,0,0,0}},	/*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_PS,{"PM_CYC","PM_FPU_CMPL",0,0,0,0,0,0}},	/*Floating Point instructions per second*/ 
  {PAPI_FP_STAL,0,{"PM_FPU_WT",0,0,0,0,0,0,0}},	/*Cycles any FP units are stalled */	
  {PAPI_TOT_CYC,0,{"PM_CYC",0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,0,{PM_LSU_EXEC,0,0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{"PM_SYNC",0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};

static pmapi_search_t preset_name_map_630[PAPI_MAX_PRESET_EVENTS] = { 
  {PAPI_L1_DCM,DERIVED_ADD,{"PM_LD_MISS_L1",PM_ST_MISS_L1,0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_ICM,0,{"PM_IC_MISS",0,0,0,0,0,0,0}}, /*Level 1 instruction cache misses*/ 
  {PAPI_L1_TCM,DERIVED_ADD,{"PM_IC_MISS","PM_LD_MISS_L1",PM_ST_MISS_L1,0,0,0,0,0}}, /*Level 1 total cache misses*/
  {PAPI_CA_SNP,0,{PM_SNOOP,0,0,0,0,0,0,0}}, /*Snoops*/
  {PAPI_CA_SHR,0,{"PM_SNOOP_E_TO_S",0,0,0,0,0,0,0}}, /*Request for shared cache line (SMP)*/
  {PAPI_CA_ITV,0,{"PM_SNOOP_PUSH_INT",0,0,0,0,0,0,0}}, /*Request for cache line Intervention (SMP)*/
  {PAPI_BRU_IDL,0,{"PM_BRU_IDLE",0,0,0,0,0,0,0}}, /*Cycles branch units are idle*/
  {PAPI_FXU_IDL,0,{"PM_FXU_IDLE",0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/
  {PAPI_FPU_IDL,0,{"PM_FPU_IDLE",0,0,0,0,0,0,0}}, /*Cycles floating point units are idle*/
  {PAPI_LSU_IDL,0,{"PM_LSU_IDLE",0,0,0,0,0,0,0}}, /*Cycles load/store units are idle*/
  {PAPI_TLB_TL,0,{"PM_TLB_MISS",0,0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/
  {PAPI_L1_LDM,0,{"PM_LD_MISS_L1",0,0,0,0,0,0,0}}, /*Level 1 load misses */
  {PAPI_L1_STM,0,{PM_ST_MISS_L1,0,0,0,0,0,0,0}}, /*Level 1 store misses */
  {PAPI_L2_LDM,0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 load misses */		
  {PAPI_L2_STM,0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}}, /*Level 2 store misses */		
  {PAPI_BTAC_M,0,{"PM_BTAC_MISS",0,0,0,0,0,0,0}}, /*BTAC miss*/
  {PAPI_PRF_DM,0,{"PM_PREF_MATCH_DEM_MISS",0,0,0,0,0,0,0}}, /*Prefetch data instruction caused a miss */
  {PAPI_TLB_SD,0,{"PM_TLBSYNC_RERUN",0,0,0,0,0,0,0}}, /*Xlation lookaside buffer shootdowns (SMP)*/
  {PAPI_CSR_SUC,0,{PM_RESRV_CMPL,0,0,0,0,0,0,0}}, /*Successful store conditional instructions*/	
  {PAPI_CSR_FAL,0,{"PM_ST_COND_FAIL",0,0,0,0,0,0,0}}, /*Failed store conditional instructions*/	
  {PAPI_CSR_TOT,0,{PM_RESRV_RQ,0,0,0,0,0,0,0}}, /*Total store conditional instructions*/		
  {PAPI_MEM_SCY,DERIVED_ADD,{"PM_CMPLU_WT_LD","PM_CMPLU_WT_ST",0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Access*/
  {PAPI_MEM_RCY,0,{"PM_CMPLU_WT_LD",0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Read*/
  {PAPI_MEM_WCY,0,{"PM_CMPLU_WT_ST",0,0,0,0,0,0,0}}, /*Cycles Stalled Waiting for Memory Write*/
  {PAPI_STL_ICY,0,{"PM_0INST_DISP",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {PAPI_STL_CCY,0,{"PM_0INST_CMPL",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Completion*/
  {PAPI_BR_CN,0,{"PM_CBR_DISP",0,0,0,0,0,0}}, /*Conditional branch instructions executed*/    
  {PAPI_BR_MSP,0,{PM_MPRED_BR,0,0,0,0,0,0,0}}, /*Conditional branch instructions mispred*/
  {PAPI_BR_PRC,0,{"PM_BR_PRED",0,0,0,0,0,0,0}}, /*Conditional branch instructions corr. pred*/
  {PAPI_FMA_INS,0,{PM_EXEC_FMA,0,0,0,0,0,0,0}}, /*FMA instructions completed*/
  {PAPI_TOT_IIS,0,{"PM_INST_DISP",0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{"PM_INST_CMPL",0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,DERIVED_ADD,{"PM_FXU0_PROD_RESULT","PM_FXU1_PROD_RESULT","PM_FXU2_PROD_RESULT",0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,DERIVED_ADD,{"PM_FPU0_CMPL","PM_FPU1_CMPL",0,0,0,0,0,0}}, /*Floating point instructions executed*/	
  {PAPI_LD_INS,0,{"PM_LD_CMPL",0,0,0,0,0,0,0}},	/*Load instructions executed*/
  {PAPI_SR_INS,0,{"PM_ST_CMPL",0,0,0,0,0,0,0}}, /*Store instructions executed*/
  {PAPI_BR_INS,0,{"PM_BR_CMPL",0,0,0,0,0,0,0}}, /*Total branch instructions executed*/
  {PAPI_FLOPS,DERIVED_ADD_PS,{"PM_CYC","PM_FPU0_CMPL","PM_FPU1_CMPL",0,0,0,0,0}}, /*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{"PM_CYC",0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_LST_INS,DERIVED_ADD,{"PM_LD_CMPL","PM_ST_CMPL",0,0,0,0,0,0}}, /*Total load/store inst. executed*/
  {PAPI_SYC_INS,0,{"PM_SYNC",0,0,0,0,0,0,0}}, /*Sync. inst. executed */
  {PAPI_FDV_INS,0,{"PM_FPU_FDIV",0,0,0,0,0,0,0}}, /*FD ins */
  {PAPI_FSQ_INS,0,{"PM_FPU_FSQRT",0,0,0,0,0,0,0}}, /*FSq ins */
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};

/* Utility functions */

/* Find all the hwcntrs that name lives on */

static int find_hwcounter(pm_info_t *info, char *name, hwd_preset_t *preset, int index)
{
  int did_something = 0, pmc, ev;
  pm_events_t *wevp;

  preset->metric_count++;

#ifdef DEBUG_SETUP
	  DBG((stderr,"find_hwcounter( %s, %d, %d)\n",name, index, preset->metric_count));
#endif

  preset->rank[index] = 0;  /* this value accumulates if initializes more than once */
  for (pmc = 0; pmc < info->maxpmcs; pmc++) 
    {
      preset->counter_cmd[index][pmc] = COUNT_NOTHING;
      wevp = info->list_events[pmc];
      for (ev = 0; ev < info->maxevents[pmc]; ev++, wevp++) 
	{
	  if (strcmp(name, wevp->short_name) == 0) 
	    {
	      preset->counter_cmd[index][pmc] = wevp->event_id;
	      preset->selector[index] |= 1 << pmc;
	      preset->rank[index]++;
	      did_something++;
	      break;
	    }
	}
    }

  if (did_something)
    {
      return(1);
    }
  else
    abort();
}

 #define DEBUG_SETUP 

int setup_all_presets(pm_info_t *info)
{
  pmapi_search_t *findem;
  int pnum,did_something = 0,pmc,derived;
  int preset_index;

  if (__power_630())
    findem = preset_name_map_630;
  else if (__power_604())
    {
      if (strstr(info->proc_name,"604e"))
	findem = preset_name_map_604e;
      else
	findem = preset_name_map_604;
    }
  else
    return(PAPI_ESBSTR);

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      /* dense array of events is terminated with a 0 preset */
      if (findem[pnum].preset == 0)
	break;

      preset_index = findem[pnum].preset & PRESET_AND_MASK; 
      /* If it's not derived */
      if (findem[pnum].derived == 0)
	{
	  /* If we find it, then on to the next preset */
#ifdef DEBUG_SETUP
	  DBG((stderr,"Looking for preset %d, %s\n",preset_index,findem[pnum].findme[0]));
#endif
	  find_hwcounter(info,findem[pnum].findme[0],&_papi_hwd_preset_map[preset_index], 0);
	  strncpy(_papi_hwd_preset_map[preset_index].note,findem[pnum].findme[0], PAPI_MAX_STR_LEN);
	  did_something++;
	}
      else 
	{
	  hwd_preset_t tmp;
	  int free_hwcntrs, need_one_hwcntr, hwcntr_num, err = 0, all_selector = 0;
	  int i,j;
	  
	  memset(&tmp,0x00,sizeof(tmp));
	  tmp.derived = findem[pnum].derived;
	  /* Find info for all the metrics (up to 8!) in this derived event */
	  pmc = 0;
	  while (findem[pnum].findme[pmc])
	    {
#ifdef DEBUG_SETUP
	      DBG((stderr,"Looking for preset %d, %s\n",pnum,findem[pnum].findme[pmc]));
#endif
	      find_hwcounter(info,findem[pnum].findme[pmc],&tmp, pmc);
	      /* append the metric name to the event descriptor */
	      if (strlen(tmp.note)+strlen(findem[pnum].findme[pmc]+1) < PAPI_MAX_STR_LEN)
		{
		  strcat(tmp.note,findem[pnum].findme[pmc]);
		  strcat(tmp.note,",");
		}
#ifdef DEBUG_SETUP
	    if (findem[pnum].preset == PAPI_TOT_CYC) {
	      DBG((stderr,"selector[%d] = 0x%x, rank[%d] = %d\n",pmc,tmp.selector[pmc],pmc,tmp.rank[pmc]));
	      DBG((stderr,"cmd[%d][] = %d %d %d %d %d %d %d %d\n",pmc,tmp.counter_cmd[pmc][0],
		tmp.counter_cmd[pmc][1],tmp.counter_cmd[pmc][2],tmp.counter_cmd[pmc][3],
		tmp.counter_cmd[pmc][4],tmp.counter_cmd[pmc][5],tmp.counter_cmd[pmc][6],
		tmp.counter_cmd[pmc][7]));
	    }
#endif
	      pmc++;
	    }
	  /* For all the metrics, verify that a valid counter mapping exists.
	      Rank is the number of counters a metric can live on.
	      Scan metrics from lowest to highest rank, assigning counters
	      as you go. This guarantees that the most restricted metrics get
	      mapped first. The result may not be the only valid mapping, and
	      may not find all possible mappings, but should do a pretty good job,
	      esp if metrics are mapped sparsely onto counters.
	  */
	  for (i=1; i<POWER_MAX_COUNTERS+1 && !err; i++) /* scan across rank */
	    {
	      for (j=0; j<pmc && !err; j++) /* scan across available metrics */
		{
		  if (tmp.rank[j] == i)
		    {
		      /* selector[j] contains all the counters with findme[j] */
		      /* first, find what's currently available */
		      free_hwcntrs = ~all_selector;
		      /* second, of those available, what can we choose */
		      need_one_hwcntr = free_hwcntrs & tmp.selector[j];
#ifdef DEBUG_SETUP
		      DBG((stderr,"rank %d = %d; need_one = 0x%x\n",j, i, need_one_hwcntr));
#endif
		      if (need_one_hwcntr == 0)
			{
			  err = 1;
			  break;
			}
		      /* third, pick one */
		      hwcntr_num = get_avail_hwcntr_num(need_one_hwcntr);
		      need_one_hwcntr = 1 << hwcntr_num;
		      /* fourth, add it to our set */
		      all_selector |= need_one_hwcntr;
		    }
		}
	    }

	  /* If we're successful */
	  if (err == 0)
	    {
	      tmp.note[strlen(tmp.note)-1] = '\0';
	      _papi_hwd_preset_map[preset_index] = tmp;
#ifdef DEBUG_SETUP
	      DBG((stderr,"Found compound preset %d on 0x%x\n",preset_index,all_selector));
	      DBG((stderr,"preset->metric_count: %d\n",_papi_hwd_preset_map[preset_index].metric_count));
#endif
	      did_something++;
	      continue;
	    }
	  fprintf(stderr,"Did not find compound preset %d on 0x%x\n",preset_index,all_selector);	  
	  abort();
	}
    }
  return(did_something ? 0 : PAPI_ESBSTR);
}


void init_config(hwd_control_state_t *ptr)
{
  int i, j;

  memset(ptr->native, 0, sizeof(hwd_native_t)*POWER_MAX_COUNTERS);

  for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
    ptr->preset[i] = COUNT_NOTHING;
    ptr->counter_cmd.events[i] = COUNT_NOTHING;
 	ptr->native[i].position=COUNT_NOTHING;
	/*ptr->native[i].link=COUNT_NOTHING;*/
 }
  for(i=0;i<POWER_MAX_COUNTERS_MAPPING;i++){
    ptr->allevent[i]=COUNT_NOTHING;
	for (j = 0; j < _papi_hwi_system_info.num_cntrs; j++) {
	  ptr->emap[i][j]=COUNT_NOTHING;
	}
  }
  ptr->hwd_idx=0;
  ptr->hwd_idx_a=0;
  ptr->native_idx=0;
  set_domain(ptr,_papi_hwi_system_info.default_domain);
  set_granularity(ptr,_papi_hwi_system_info.default_granularity);
}


static void print_state(hwd_control_state_t *s)
{
  int i;
  
  fprintf(stderr,"\n\n-----------------------------------------\nmaster_selector 0x%x\n",s->master_selector);
  for(i=0;i<POWER_MAX_COUNTERS;i++){
  	if(s->master_selector & (1<<i)) fprintf(stderr, "  1  ");
	else fprintf(stderr, "  0  ");
  }
  fprintf(stderr,"\nnative_event_name       %12s %12s %12s %12s %12s %12s %12s %12s\n",s->native[0].name,s->native[1].name,
    s->native[2].name,s->native[3].name,s->native[4].name,s->native[5].name,s->native[6].name,s->native[7].name);
  fprintf(stderr,"native_event_selectors    %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].selector,s->native[1].selector,
    s->native[2].selector,s->native[3].selector,s->native[4].selector,s->native[5].selector,s->native[6].selector,s->native[7].selector);
  fprintf(stderr,"native_event_position     %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].position,s->native[1].position,
    s->native[2].position,s->native[3].position,s->native[4].position,s->native[5].position,s->native[6].position,s->native[7].position);
  fprintf(stderr,"counters                  %12d %12d %12d %12d %12d %12d %12d %12d\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
  fprintf(stderr,"native links              %12d %12d %12d %12d %12d %12d %12d %12d\n",s->native[0].link,s->native[1].link,
    s->native[2].link,s->native[3].link,s->native[4].link,s->native[5].link,s->native[6].link,s->native[7].link);
  for(i=0;i<s->hwd_idx_a;i++){
  	fprintf(stderr,"event_codes %x\n",s->allevent[i]);
  }
}

static int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = POWER_MAX_COUNTERS - 1;
 
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


#if 1
static void dump_state(hwd_control_state_t *s)
{
  fprintf(stderr,"master_selector %x\n",s->master_selector);
  fprintf(stderr,"event_codes %x %x %x %x %x %x %x %x\n",s->preset[0],s->preset[1],
    s->preset[2],s->preset[3],s->preset[4],s->preset[5],s->preset[6],s->preset[7]);
  fprintf(stderr,"event_selectors %x %x %x %x %x %x %x %x\n",s->selector[0],s->selector[1],
    s->selector[2],s->selector[3],s->selector[4],s->selector[5],s->selector[6],s->selector[7]);
  fprintf(stderr,"counters %x %x %x %x %x %x %x %x\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
}
#endif


int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int selector;
  int avail;
  int out_command, out_operand_index;
  hwd_control_state_t tmp_state;
  hwd_preset_t *this_preset;
  hwd_native_t new_native;
  int hwd_idx = 0, i, j;
  unsigned int event_code;
  int hwcntr_num, rank, metric, max_rank, total_metrics;
  EventInfo_t *zeroth;
  void *v;

  DBG((stderr,"EventCode %x \n",EventCode));
  
#ifdef HAS_NATIVE_MAP
  if(EventCode & NATIVE_MASK)
    EventCode = EventCode ^ NATIVE_MASK;
#endif

  /* Do a preliminary check to eliminate preset events that aren't
     supported on this platform */
  if (EventCode & PRESET_MASK)
    {
      if (_papi_hwd_preset_map[(EventCode & PRESET_AND_MASK)].selector[0] == 0)
	 	return(PAPI_ENOEVNT);
    }
  else{ /* also need to check the native event is eligible */ 
		pm_events_t *pe;
		int found_native=0, pmc;
		event_code=EventCode>>8;
		hwcntr_num = EventCode & 0xff;
		pe=pminfo.list_events[hwcntr_num];
		for(i=0;i<pminfo.maxevents[hwcntr_num];i++, pe++){
			if(pe->event_id==event_code){
				strcpy(new_native.name, pe->short_name);
				found_native=1;
				break;
			}
		}
		
		if(!found_native){ /* no such native event */ 
			return(PAPI_ENOEVNT);
		}
		else{
			new_native.selector=0;
			new_native.rank=0;
			/*memset(&new_native, 0, sizeof(hwd_native_t));*/
			new_native.position=COUNT_NOTHING;
			new_native.link=0;
			for (pmc = 0; pmc < pminfo.maxpmcs; pmc++){
				new_native.counter_cmd[pmc] = COUNT_NOTHING;
      			pe = pminfo.list_events[pmc];
				for (i = 0; i < pminfo.maxevents[pmc]; i++, pe++){
					if (strcmp(new_native.name, pe->short_name) == 0){
						new_native.counter_cmd[pmc]=pe->event_id;
						new_native.selector |=1<<pmc;
						new_native.rank++;
						break;
					}
				}
			}
		}
	}

  /* Copy this_state into tmp_state. We can muck around with tmp and
     bail in case of failure and leave things unchanged. tmp_state 
     gets written back to this_state only if everything goes OK. */
  tmp_state = *this_state;

  /* If all slots are empty, initialize the state. */
  if (tmp_state.master_selector == 0)
    init_config(&tmp_state);

  if (EventCode & PRESET_MASK){
  	this_preset=&(_papi_hwd_preset_map[EventCode & PRESET_AND_MASK]);
  	v=(void *)this_preset;
  }
  else
  	v=(void *)&new_native;

  if(_papi_hwd_event_precheck(&tmp_state, EventCode, out, v)){
  int j;
	*this_state=tmp_state;
  	return(PAPI_OK);
  }
  
  
  /* If all counters are full, return a conflict error. */
  if ((tmp_state.master_selector & 0xff) == 0xff)
   	return(PAPI_ECNFLCT);

  if(_papi_hwd_counter_mapping(&tmp_state, EventCode, out, v)){
	*this_state=tmp_state;
	out->bits.event_code = EventCode;
  	return(PAPI_OK);
  }
  else{
   	return(PAPI_ECNFLCT);
  }
#if 0
  DBG((stderr,"success \n"));
  dump_state(this_state);
#endif
}



/*int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)*/
int _papi_hwd_remove_event(hwd_register_map_t *chosen, unsigned int hardware_index, hwd_control_state_t *out)
{
  int i, j, selector, used, preset_index, EventCode, metric, zero;
  int allevent[POWER_MAX_COUNTERS_MAPPING];
  int found;
  hwd_control_state_t new_state;
  hwd_preset_t *this_preset;
  hwd_native_t *this_native;
  

  EventCode = chosen->event_code;
  zero=0;

	/*print_state(out);*/

  if(EventCode!=out->allevent[hardware_index])
  	return(PAPI_ENOEVNT);

	/* preset */
	if(EventCode & PRESET_MASK){
		this_preset = &(_papi_hwd_preset_map[EventCode & PRESET_AND_MASK]);
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<POWER_MAX_COUNTERS; j++) {
				if (out->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (out->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<POWER_MAX_COUNTERS){ /* found mapping */ 
				for(i=0;i<out->native_idx;i++)
					if(j==out->native[i].position){
						out->native[i].link--;
						if(out->native[i].link==0)
							zero++;
						break;
					}
			}
			else{/* should never happen*/
				fprintf(stderr, "error in _papi_hwd_rem_event()\n");
				exit(1);
			}		
		}
	}
	else{
		unsigned int code;
		int hwcntr_num;

		code=EventCode>>8;
		hwcntr_num = EventCode & 0xff;

		/* to find the native event from the native events list */
		for(i=0; i<out->native_idx;i++){
			if(code==out->native[i].counter_cmd[hwcntr_num]){
				out->native[i].link--; 
				if(out->native[i].link==0)
					zero++;
				break;
			}
		}
	}

	/* to reset hwd_control_state values */
	out->allevent[hardware_index]=COUNT_NOTHING;
	out->hwd_idx_a--;
	out->native_idx-=zero;
	for (j = 0; j < _papi_hwi_system_info.num_cntrs; j++) {
	  out->emap[hardware_index][j]=COUNT_NOTHING;
	}

	/*	print_state(out);*/
/* to move correspond native structures */
	for(found=0; found<zero; found++){
	for(i=0;i<_papi_hwi_system_info.num_cntrs;i++){
		if(out->native[i].link==0 && out->native[i].position!=COUNT_NOTHING ){
			int copy=0;
			out->master_selector^=1<<out->native[i].position;
			out->counter_cmd.events[out->native[i].position]=COUNT_NOTHING;
			for(j=_papi_hwi_system_info.num_cntrs-1;j>i;j--){
				if(out->native[j].position==COUNT_NOTHING)
					continue;
				else{
					memcpy(out->native+i, out->native+j, sizeof(hwd_native_t));
					memset(out->native+j, 0, sizeof(hwd_native_t));
					out->native[j].position=COUNT_NOTHING;
					/*out->native[j].link=COUNT_NOTHING;*/
					copy++;
					break;
				}
			}
			if(copy==0){
				memset(out->native+i, 0, sizeof(hwd_native_t));
				out->native[i].position=COUNT_NOTHING;
				/*out->native[i].link=COUNT_NOTHING;*/
			}
			
			/*found++;
			if(found==zero)
				break;*/
		}
	}
	}
	/*print_state(out);
	fprintf(stderr, "*********out->native_idx=%d, zero=%d,  found=%d\n", out->native_idx,zero, found); */


#if 1
  dump_state(out);
#endif

  return(PAPI_OK);
}

int _papi_hwd_update_shlib_info(void)
{
  return PAPI_ESBSTR;
}


int _papi_hwd_query(int preset_index, int *flags, char **note)
{
  int events;

  events = _papi_hwd_preset_map[preset_index].selector[0];

  if (events == 0)
    return(0);
  if (_papi_hwd_preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (_papi_hwd_preset_map[preset_index].note)
    *note = _papi_hwd_preset_map[preset_index].note;
  return(1);
}


