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

#include "power3.h"


static int maxgroups = 0;
static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };


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
    #undef  PM_BR_FINISH
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
	  find_hwcounter(info,findem[pnum].findme[0],&preset_map[preset_index], 0);
	  strncpy(preset_map[preset_index].note,findem[pnum].findme[0], PAPI_MAX_STR_LEN);
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
	      preset_map[preset_index] = tmp;
#ifdef DEBUG_SETUP
	      DBG((stderr,"Found compound preset %d on 0x%x\n",preset_index,all_selector));
	      DBG((stderr,"preset->metric_count: %d\n",preset_map[preset_index].metric_count));
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

  for (i = 0; i < _papi_system_info.num_cntrs; i++) {
    ptr->preset[i] = COUNT_NOTHING;
    ptr->counter_cmd.events[i] = COUNT_NOTHING;
 	ptr->native[i].position=COUNT_NOTHING;
	/*ptr->native[i].link=COUNT_NOTHING;*/
 }
  for(i=0;i<POWER_MAX_COUNTERS_MAPPING;i++){
    ptr->allevent[i]=COUNT_NOTHING;
	for (j = 0; j < _papi_system_info.num_cntrs; j++) {
	  ptr->emap[i][j]=COUNT_NOTHING;
	}
  }
  ptr->hwd_idx=0;
  ptr->hwd_idx_a=0;
  ptr->native_idx=0;
  set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity);
}

static int get_system_info(void)
{
  int retval;
 /* pm_info_t pminfo;*/
  struct procsinfo psi = { 0 };
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN];

#ifdef _AIXVERSION_510
  pm_groups_info_t pmgroups;
#endif

  #define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);
  psi.pi_pid = pid;
  retval = getargs(&psi,sizeof(psi),maxargs,PAPI_MAX_STR_LEN);
  if (retval == -1)
    return(PAPI_ESYS);
  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,maxargs);
  strncpy(_papi_system_info.exe_info.name,basename(maxargs),PAPI_MAX_STR_LEN);

#ifdef _AIXVERSION_510
  DBG((stderr,"Calling AIX 5 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS, &pminfo, &pmgroups);
#else
  DBG((stderr,"Calling AIX 4 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS,&pminfo);
#endif
  DBG((stderr,"...Back from pm_init\n"));

  if (retval > 0)
    return(retval);

  _papi_system_info.hw_info.ncpu = _system_configuration.ncpus;
  _papi_system_info.hw_info.totalcpus = 
    _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"IBM");
  _papi_system_info.hw_info.model = _system_configuration.implementation;
  strcpy(_papi_system_info.hw_info.model_string,pminfo.proc_name);
  _papi_system_info.hw_info.revision = (float)_system_configuration.version;
  _papi_system_info.hw_info.mhz = (float)(pm_cycles() / 1000000.0);
  _papi_system_info.num_gp_cntrs = pminfo.maxpmcs;
  _papi_system_info.num_cntrs = pminfo.maxpmcs;
  _papi_system_info.cpunum = mycpu();
/*  _papi_system_info.exe_info.text_end = (caddr_t)&_etext;*/

  retval = setup_all_presets(&pminfo);

  if (retval)
    return(retval);

  return(PAPI_OK);
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




/* this function try to find out whether native events contained by this preset have already been mapped. If it is, mapping is done */
int _papi_hwd_event_precheck(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v)
{
	int metric, i, j, found_native=0, hwd_idx=0;
	hwd_preset_t *this_preset;
	hwd_native_t *this_native;
	int counter_mapping[POWER_MAX_COUNTERS];
	unsigned char selector;
	  
	/* to find first empty slot */
  hwd_idx=out->index;
		
	/* preset event */
	if(EventCode & PRESET_MASK){
		this_preset=(hwd_preset_t *)v;
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<POWER_MAX_COUNTERS; j++) {
				if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<POWER_MAX_COUNTERS){ /* found mapping */ 
				counter_mapping[metric]=j;
				for(i=0;i<tmp_state->native_idx;i++)
					if(j==tmp_state->native[i].position){
						tmp_state->native[i].link++;
						break;
					}
			}
			else{
				return 0;
			}		
		}
		
		/* successfully found mapping. Write to EventInfo_t *out, return 1 */
		tmp_state->allevent[hwd_idx]=EventCode;
		selector=0;
		for(j=0;j<metric;j++){
			tmp_state->emap[hwd_idx][j]=counter_mapping[j];
			selector|=1<<counter_mapping[j];
		}
	
		/* update EventInfo_t *out */
		out->code = EventCode;
		out->selector = selector;
		out->command = this_preset->derived;
		out->operand_index = tmp_state->emap[hwd_idx][0];
		tmp_state->hwd_idx_a++;
	
		return 1;
	}
	else{
		this_native=(hwd_native_t *)v;

		/* to find the native event from the native events list */
		for(i=0; i<tmp_state->native_idx;i++){
			if(strcmp(this_native->name, tmp_state->native[i].name)==0){
				found_native=1;
				break;
			}
		}
		if(found_native){
			tmp_state->allevent[hwd_idx]=EventCode;
			tmp_state->emap[hwd_idx][0]=tmp_state->native[i].position;
			tmp_state->native[i].link++;
			/* update EventInfo_t *out */
			out->code = EventCode;
			out->selector |= 1<<tmp_state->native[i].position;
			out->command = NOT_DERIVED;
			out->operand_index = tmp_state->emap[hwd_idx][0];
			tmp_state->hwd_idx_a++;
			return 1;
		}
		else{
			return 0;
		}
	}
}	  



/* this function is called after mapping is done */
int _papi_hwd_event_mapafter(hwd_control_state_t *tmp_state, int index, EventInfo_t *out)
{
	int metric, j;
	hwd_preset_t *this_preset;
	int counter_mapping[POWER_MAX_COUNTERS];
	unsigned char selector;
	unsigned int EventCode;
	  
  	EventCode=tmp_state->allevent[index];
	/* preset */
	if(EventCode & PRESET_MASK){
		this_preset = &(preset_map[EventCode & PRESET_AND_MASK]);
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<POWER_MAX_COUNTERS; j++) {
				if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<POWER_MAX_COUNTERS){ /* found mapping */
				counter_mapping[metric]=j;
			}
			else{
				return 0;
			}		
		}
	
		/* successfully found mapping. Write to EventInfo_t *out, return 1 */
		selector=0;
		for(j=0;j<metric;j++){
			tmp_state->emap[index][j]=counter_mapping[j];
			selector|=1<<counter_mapping[j];
		}
	
		out->selector = selector;
		out->operand_index = tmp_state->emap[index][0];
		return 1;
	}
	else{
		pm_events_t *pe;
		int found_native=0, pmc, hwcntr_num, i;
		unsigned int event_code;
		char name[PAPI_MAX_STR_LEN];
		
		/* to get pm event name */ 
		event_code=EventCode>>8;
		hwcntr_num = EventCode & 0xff;
		pe=pminfo.list_events[hwcntr_num];
		for(i=0;i<pminfo.maxevents[hwcntr_num];i++, pe++){
			if(pe->event_id==event_code){
				strcpy(name, pe->short_name); /* will be found */
				break;
			}
		}
		
		/* to find the native event from the native events list */
		for(i=0; i<POWER_MAX_COUNTERS;i++){
			if(strcmp(name, tmp_state->native[i].name)==0){
				found_native=1;
				break;
			}
		}
		if(found_native){
			tmp_state->emap[index][0]=tmp_state->native[i].position;
	
			/* update EventInfo_t *out */
			out->selector |= 1<<tmp_state->native[i].position;
			out->operand_index = tmp_state->emap[index][0];
			return 1;
		}
		else{
			return 0;
		}
	}
}	  

int do_counter_mapping(hwd_native_t *event_list, int size)
{
	int i,j;
	hwd_native_t *queue[POWER_MAX_COUNTERS];
	int head, tail;
	
	/* if the event competes 1 counter only, it has priority, map it */
	head=0;
	tail=0;
	for(i=0;i<size;i++){ /* push rank=1 into queue */
		event_list[i].mod=-1;
		if(event_list[i].rank==1){
			queue[tail]=&event_list[i];
			event_list[i].mod=i;
			tail++;
		}
	}
	
	while(head<tail){
		for(i=0;i<size;i++){
			if(i!=(*queue[head]).mod){
				if(event_list[i].selector & (*queue[head]).selector){
					if(event_list[i].rank==1){
						return 0; /* mapping fail, 2 events compete 1 counter only */
					}
					else{
						event_list[i].selector ^= (*queue[head]).selector;
						event_list[i].rank--;
						if(event_list[i].rank==1){
							queue[tail]=&event_list[i];
							event_list[i].mod=i;
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
		hwd_native_t rest_event_list[POWER_MAX_COUNTERS];
		hwd_native_t copy_rest_event_list[POWER_MAX_COUNTERS];
		
		j=0;
		for(i=0;i<size;i++){
			if(event_list[i].mod<0){
				memcpy(copy_rest_event_list+j, event_list+i, sizeof(hwd_native_t));
				copy_rest_event_list[j].mod=i;
				j++;
			}
		}
		
		memcpy(rest_event_list, copy_rest_event_list, sizeof(hwd_native_t)*(size-tail));
		
		for(i=0;i<POWER_MAX_COUNTERS;i++){
			if(rest_event_list[0].selector & (1<<i)){ /* pick first event on the list, set 1 to 0, to see whether there is an answer */
				for(j=0;j<size-tail;j++){
					if(j==0){
						rest_event_list[j].selector = 1<<i;
						rest_event_list[j].rank = 1;
					}
					else{
						if(rest_event_list[j].selector & (1<<i)){
							rest_event_list[j].selector ^= 1<<i;
							rest_event_list[j].rank--;
						}
					}
				}
				if(do_counter_mapping(rest_event_list, size-tail))
					break;
				
				memcpy(rest_event_list, copy_rest_event_list, sizeof(hwd_native_t)*(size-tail));
			}
		}
		if(i==POWER_MAX_COUNTERS){
			return 0; /* fail to find mapping */
		}
		for(i=0;i<size-tail;i++){
			event_list[copy_rest_event_list[i].mod].selector=rest_event_list[i].selector;
		}
		return 1;		
	}
}	
	

/* this function will be called when there are counters available, (void *) is the pointer to adding event structure
   (hwd_preset_t *)  or (hwd_native_t *)  
*/      
int _papi_hwd_counter_mapping(hwd_control_state_t *tmp_state, unsigned int EventCode, EventInfo_t *out, void *v)
{
  hwd_preset_t *this_preset;
  hwd_native_t *this_native;
  unsigned char selector;
  int metric, i, j, k, getname=1, ncount=0, hwd_idx=0, triger=0, natNum;
  pm_events_t *pe;
  int tr;
  hwd_control_state_t ttmp_state;
  EventInfo_t *zeroth;


  hwd_idx=out->index;
  
  tmp_state->allevent[hwd_idx]=EventCode;
  selector=0;
  natNum=tmp_state->native_idx;
  
  if(EventCode & PRESET_MASK){
	this_preset=(hwd_preset_t *)v;
	
	
	/* try to find unmapped native events, then put then on to native list */ 
	for( metric=0; metric<this_preset->metric_count; metric++){
		for (j=0; j<POWER_MAX_COUNTERS; j++) {
			if (tmp_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
				if (tmp_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j]){
					selector |= 1<<j;
					tmp_state->emap[hwd_idx][metric]=j;
					if(triger){
						for(i=0;i<natNum;i++)
							if(j==tmp_state->native[i].position){
								tmp_state->native[i].link++;
								break;
							}
					}
					break;
				}
		  	}
		}
		if(j==POWER_MAX_COUNTERS){ /* not found mapping from existed mapped native events */
			if(tmp_state->native_idx==POWER_MAX_COUNTERS){ /* can not do mapping, no counter available */
				return 0;
			}
			triger=1;
			tmp_state->native[tmp_state->native_idx].selector=this_preset->selector[metric];
			tmp_state->native[tmp_state->native_idx].rank=this_preset->rank[metric];
			getname=1;
			
			for(i=0;i<POWER_MAX_COUNTERS;i++){
				tmp_state->native[tmp_state->native_idx].counter_cmd[i]=this_preset->counter_cmd[metric][i];
				if(getname && tmp_state->native[tmp_state->native_idx].counter_cmd[i]!=COUNT_NOTHING){
					/* get the native event's name */
					pe=pminfo.list_events[i];
					/*printf("tmp_state->native[%d].counter_cmd[%d]=%d\n", tmp_state->native_idx, i, tmp_state->native[tmp_state->native_idx].counter_cmd[i]  );*/
					for(k=0;k<pminfo.maxevents[i];k++, pe++){ 
						if(pe->event_id==tmp_state->native[tmp_state->native_idx].counter_cmd[i]){
							strcpy(tmp_state->native[tmp_state->native_idx].name, pe->short_name);
							tmp_state->native[tmp_state->native_idx].link++;
							getname=0;
							break;
						}
					}
				}
			}
			tmp_state->native_idx++;
		}
	}
  }
  else{
	
  	this_native=(hwd_native_t *)v;
	this_native->link++;
	memcpy(tmp_state->native+tmp_state->native_idx, this_native, sizeof(hwd_native_t));
	tmp_state->native_idx++;
  }

  { /* not successfully mapped, but have enough slots for events */
  	hwd_native_t event_list[POWER_MAX_COUNTERS];
	
	memcpy(event_list, tmp_state->native, sizeof(hwd_native_t)*(tmp_state->native_idx));
	
	if(do_counter_mapping(event_list, tmp_state->native_idx)){ /* successfully mapped */
		/* update tmp_state, reset... */
		tmp_state->master_selector=0;
		for (i = 0; i <POWER_MAX_COUNTERS; i++) {
		    tmp_state->counter_cmd.events[i] = COUNT_NOTHING;
		}
		
		for(i=0;i<tmp_state->native_idx;i++){
			tmp_state->master_selector |= event_list[i].selector;
			/* update tmp_state->native->position */
			tmp_state->native[i].position=get_avail_hwcntr_num(event_list[i].selector); 
			/* update tmp_state->counter_cmd */
			tmp_state->counter_cmd.events[tmp_state->native[i].position] = tmp_state->native[i].counter_cmd[tmp_state->native[i].position];
		}
		
		/* copy new value to out */
		zeroth = out-out->index;
		j=0;
		for(i=0;i<=tmp_state->hwd_idx_a;i++){
			while(tmp_state->allevent[j]==COUNT_NOTHING)
				j++;
			tr=_papi_hwd_event_mapafter(tmp_state, j, zeroth+j);
			if(!tr)
				printf("************************not possible!  j=%d\n", j);
			j++;
		}
		
		out->code = EventCode;
		if(EventCode & PRESET_MASK)
			out->command = this_preset->derived;
		else
			out->command = NOT_DERIVED;
		/*out->index=tmp_state->hwd_idx_a;*/
	
		tmp_state->hwd_idx++;
		tmp_state->hwd_idx_a++;
		return 1;
	}
	else{
		DBG((stderr,"--------fail 1: %x  %d \n",EventCode, tmp_state->hwd_idx_a));
		return 0;
	}
  }

}


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
  
  /* Do a preliminary check to eliminate preset events that aren't
     supported on this platform */
  if (EventCode & PRESET_MASK)
    {
      if (preset_map[(EventCode & PRESET_AND_MASK)].selector[0] == 0)
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
  	this_preset=&(preset_map[EventCode & PRESET_AND_MASK]);
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



int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int i, j, selector, used, preset_index, EventCode, metric, zero;
  int allevent[POWER_MAX_COUNTERS_MAPPING];
  int found;
  EventInfo_t *zeroth;
  hwd_control_state_t new_state;
  hwd_preset_t *this_preset;
  hwd_native_t *this_native;
  
  zeroth=&(in[-in->index]); 
  EventCode = in->code;
  zero=0;

	/*print_state(this_state);*/
  if(EventCode!=this_state->allevent[in->index])
  	return(PAPI_ENOEVNT);

	/* preset */
	if(EventCode & PRESET_MASK){
		this_preset = &(preset_map[EventCode & PRESET_AND_MASK]);
		for( metric=0; metric<this_preset->metric_count; metric++){
			for (j=0; j<POWER_MAX_COUNTERS; j++) {
				if (this_state->master_selector & (1<<j) && this_preset->selector[metric] & (1<<j)) {
					if (this_state->counter_cmd.events[j] == this_preset->counter_cmd[metric][j])
						break;
			  	}
			}
		
			if(j<POWER_MAX_COUNTERS){ /* found mapping */ 
				for(i=0;i<this_state->native_idx;i++)
					if(j==this_state->native[i].position){
						this_state->native[i].link--;
						if(this_state->native[i].link==0)
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
		for(i=0; i<this_state->native_idx;i++){
			if(code==this_state->native[i].counter_cmd[hwcntr_num]){
				this_state->native[i].link--; 
				if(this_state->native[i].link==0)
					zero++;
				break;
			}
		}
	}

	/* to reset hwd_control_state values */
	this_state->allevent[in->index]=COUNT_NOTHING;
	this_state->hwd_idx_a--;
	this_state->native_idx-=zero;
	for (j = 0; j < _papi_system_info.num_cntrs; j++) {
	  this_state->emap[in->index][j]=COUNT_NOTHING;
	}

	/*	print_state(this_state);*/
/* to move correspond native structures */
	for(found=0; found<zero; found++){
	for(i=0;i<_papi_system_info.num_cntrs;i++){
		if(this_state->native[i].link==0 && this_state->native[i].position!=COUNT_NOTHING ){
			int copy=0;
			this_state->master_selector^=1<<this_state->native[i].position;
			this_state->counter_cmd.events[this_state->native[i].position]=COUNT_NOTHING;
			for(j=_papi_system_info.num_cntrs-1;j>i;j--){
				if(this_state->native[j].position==COUNT_NOTHING)
					continue;
				else{
					memcpy(this_state->native+i, this_state->native+j, sizeof(hwd_native_t));
					memset(this_state->native+j, 0, sizeof(hwd_native_t));
					this_state->native[j].position=COUNT_NOTHING;
					/*this_state->native[j].link=COUNT_NOTHING;*/
					copy++;
					break;
				}
			}
			if(copy==0){
				memset(this_state->native+i, 0, sizeof(hwd_native_t));
				this_state->native[i].position=COUNT_NOTHING;
				/*this_state->native[i].link=COUNT_NOTHING;*/
			}
			
			/*found++;
			if(found==zero)
				break;*/
		}
	}
	}
	/*print_state(this_state);
	fprintf(stderr, "*********this_state->native_idx=%d, zero=%d,  found=%d\n", this_state->native_idx,zero, found); */


#if 0
  dump_state(this_state);
#endif

  return(PAPI_OK);
}




/* EventSet zero contains the 'current' state of the counting hardware */
int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we are nested, merge the global counter structure
     with the current eventset */

#if 0
DBG((stderr, "Merge\n"));
dump_state(this_state);
dump_state(current_state);
#endif

  if (current_state->master_selector)
    {
      int hwcntrs_in_both, hwcntr;

      /* Stop the current context */

      DBG((stderr,"Stopping the thread\n"));
      retval = pm_stop_mythread();
      if (retval > 0) 
	return(retval); 
  
      /* Update the global values */
      DBG((stderr,"Updating Global hwcounters\n"));
      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      retval = pm_delete_program_mythread();
      if (retval > 0)
	return(retval);

      hwcntrs_in_both = this_state->master_selector & current_state->master_selector;

      for (i = 0; i < _papi_system_info.num_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */
	  
	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (counter_shared(this_state, current_state, i))
		zero->multistart.SharedDepth[i]++;
	      else
		return(PAPI_ECNFLCT);
	      ESI->hw_start[i] = zero->hw_start[i];
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->master_selector & hwcntr)
	    {
	      current_state->master_selector |= hwcntr;
	      current_state->counter_cmd.mode.w = this_state->counter_cmd.mode.w;
	      current_state->counter_cmd.events[i] = this_state->counter_cmd.events[i];
	      ESI->hw_start[i] = 0;
	      zero->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */
      DBG((stderr,"Copying states\n"));

      current_state->master_selector = this_state->master_selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(pm_prog_t));

    }

  /* Set up the new merged control structure */
  
#if 0
  dump_state(this_state);
  dump_state(current_state);
  dump_cmd(&current_state->counter_cmd);
#endif
      
  retval = pm_set_program_mythread(&current_state->counter_cmd);
  if (retval > 0) 
    return(retval);

  /* (Re)start the counters */
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
} 



int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval = pm_stop_mythread();
  if (retval > 0) 
    return(retval); 
  
  for (i = 0; i < _papi_system_info.num_cntrs; i++)
    {
      /* Check for events that are NOT shared between eventsets and 
	 therefore require modification to the control state. */
      
      hwcntr = 1 << i;
      if (hwcntr & this_state->master_selector)
	{
	  if (zero->multistart.SharedDepth[i] - 1 < 0)
	    current_state->master_selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i]--;
	}
    }

  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. */

  if (zero->multistart.num_runners - 1)
    {
      retval = pm_start_mythread();
      if (retval > 0) 
	return(retval);
    }
  else
    {
      retval = pm_delete_program_mythread();
      if (retval > 0) 
	return(retval);
    }

  return(PAPI_OK);
}




int _papi_hwd_query(int preset_index, int *flags, char **note)
{
  int events;

  events = preset_map[preset_index].selector[0];

  if (events == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
}


