/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This is a merged file that handles POWER3 and POWER4 architectures and supports
  both AIX 4 and AIX 5. The switch between AIX 4 and 5 is driven by the system defined
  value _AIX_VERSION_510. These changes mostly involve naming differences in the event map.
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

#include "aix-power.h"

static int maxgroups = 0;
static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };
static pm_info_t pminfo;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef _POWER4

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

static int setup_all_presets(pm_info_t *info)
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

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#else
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

static hwd_groups_t group_map[MAX_GROUPS] = { 0 };
static pmapi_search_t preset_name_map_P4[PAPI_MAX_PRESET_EVENTS] = { 
  {PAPI_L1_DCM,DERIVED_ADD,{"PM_LD_MISS_L1","PM_ST_MISS_L1",0,0,0,0,0,0}}, /*Level 1 data cache misses*/
  {PAPI_L1_DCA,DERIVED_ADD,{"PM_LD_REF_L1","PM_ST_REF_L1",0,0,0,0,0,0}}, /*Level 1 data cache access*/
  {PAPI_FXU_IDL,0,{"PM_FXU_IDLE",0,0,0,0,0,0,0}}, /*Cycles integer units are idle*/
  {PAPI_L1_LDM,0,{"PM_LD_MISS_L1",0,0,0,0,0,0,0}}, /*Level 1 load misses */
  {PAPI_L1_STM,0,{"PM_ST_MISS_L1",0,0,0,0,0,0,0}}, /*Level 1 store misses */
  {PAPI_L1_DCW,0,{"PM_ST_REF_L1",0,0,0,0,0,0,0}}, /*Level 1 D cache write */
  {PAPI_L1_DCR,0,{"PM_LD_REF_L1",0,0,0,0,0,0,0}}, /*Level 1 D cache read */
  {PAPI_FMA_INS,0,{"PM_FPU_FMA",0,0,0,0,0,0,0}}, /*FMA instructions completed*/
  {PAPI_TOT_IIS,0,{"PM_INST_DISP",0,0,0,0,0,0,0}}, /*Total instructions issued*/
  {PAPI_TOT_INS,0,{"PM_INST_CMPL",0,0,0,0,0,0,0}}, /*Total instructions executed*/
  {PAPI_INT_INS,0,{"PM_FXU_FIN",0,0,0,0,0,0,0}}, /*Integer instructions executed*/
  {PAPI_FP_INS,0,{"PM_FPU_FIN",0,0,0,0,0,0,0}}, /*Floating point instructions executed*/	
/*  {PAPI_FP_INS,DERIVED_ADD,{"PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
    "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0,0}},*/ /*Floating point instructions executed*/	
  {PAPI_FLOPS,DERIVED_PS,{"PM_CYC","PM_FPU_FIN",0,0,0,0,0,0}}, /*Floating Point instructions per second*/ 
 /* {PAPI_FLOPS,DERIVED_ADD_PS,{"PM_CYC","PM_FPU0_ALL","PM_FPU1_ALL","PM_FPU0_FIN",
    "PM_FPU1_FIN","PM_FPU0_FMA","PM_FPU1_FMA",0}},*/ /*Floating Point instructions per second*/ 
  {PAPI_TOT_CYC,0,{"PM_CYC",0,0,0,0,0,0,0}}, /*Total cycles*/
  {PAPI_IPS,DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}}, /*Instructions executed per second*/
  {PAPI_FDV_INS,0,{"PM_FPU_FDIV",0,0,0,0,0,0,0}}, /*FD ins */
  {PAPI_FSQ_INS,0,{"PM_FPU_FSQRT",0,0,0,0,0,0,0}}, /*FSq ins */
  {PAPI_TLB_DM,0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}}, /*Data translation lookaside buffer misses*/
  {PAPI_TLB_IM,0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}}, /*Instr translation lookaside buffer misses*/
  {PAPI_TLB_TL,DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}}, /*Total translation lookaside buffer misses*/
  {PAPI_HW_INT,0,{"PM_EXT_INT",0,0,0,0,0,0,0}}, /*Hardware interrupts*/
  {PAPI_STL_ICY,0,{"PM_0INST_FETCH",0,0,0,0,0,0,0}}, /*Cycles with No Instruction Issue*/
  {0,0,{0,0,0,0,0,0,0,0}} /* end of list */
};

/* Utility functions */

/* Find all the groups that name lives on */

static int find_hwcounter_gps(pm_info_t *pminfo, char *name, hwd_preset_t *preset, int index)
{
  int did_something = 0, pmc, g, ev;
  pm_events_t *wevp;
  unsigned char *p;

  /* dereference counter array for this metric */
  p = &(preset->counter_cmd[index][0]);

  /* fill the counter array by scanning all metrics (ev) 
     on all counters (pmc) */
  for (pmc = 0; pmc < pminfo->maxpmcs; pmc++) 
    {
     DBG((stderr,"maxpmc: %d pmc: %d maxevents: %d\n",pminfo->maxpmcs, pmc, pminfo->maxevents[pmc]));
     p[pmc] = INVALID_EVENT;
      wevp = pminfo->list_events[pmc];
      for (ev = 0; ev < pminfo->maxevents[pmc]; ev++, wevp++) 
	{
	  DBG((stderr,"wevp->short_name[%d, %d] = %s \n",pmc,ev,wevp->short_name));
	  if (strcmp(name, wevp->short_name) == 0) 
	    {
	      p[pmc] = wevp->event_id;
	      did_something++;
	      DBG((stderr,"Found %s on hardware counter %d, event %d\n",name,pmc,wevp->event_id));
	      break;
	    }
	}
    }

  /* exit with error if metric wasn't found anywhere */
  if (did_something)
    did_something = 0;
  else
    return(0);

  /* fill the group bit array by scanning all groups 
     for this metric from the counter array */
  preset->gps[0] = 0;
  preset->gps[1] = 0;
  for (g = 0; g < maxgroups; g++) 
    {
      for (pmc = 0; pmc < POWER_MAX_COUNTERS; pmc++) 
	{
	  if (p[pmc] == group_map[g].counter_cmd[pmc]) 
	    {
	      preset->gps[g/32] |= 1 << (g%32);
	      did_something++;
	      DBG((stderr,"Found %s on group %d, counter %d\n",name,g,pmc));
	      break;
	    }
	}
    }
    DBG((stderr,"Found %s in groups %x %x\n",name, preset->gps[1], preset->gps[0]));

  return(did_something);
}

static int setup_p4_presets(pm_info_t *pminfo, pm_groups_info_t *pmgroups)
{
  pmapi_search_t *findem;
  pm_groups_t    *eg;
  int pnum,gnum,did_something = 0,pmc,derived;
  int preset_index, found;
  
  findem = preset_name_map_P4;
  
  maxgroups = pmgroups->maxgroups;
  DBG((stderr,"Found %d groups\n",maxgroups));
  eg = pmgroups->event_groups;
  for (gnum = 0; gnum < maxgroups; gnum++)
    {
      /* Copy the group id for this group */
      group_map[gnum].group_id = eg[gnum].group_id;
      for (pmc=0; pmc < pminfo->maxpmcs; pmc++)
	{
	  /* Copy all the counter commands for this group */
          group_map[gnum].counter_cmd[pmc] = eg[gnum].events[pmc];
	}
    }

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
	  DBG((stderr,"Looking for preset %d, %s\n",preset_index,findem[pnum].findme[0]));
	  found = find_hwcounter_gps(pminfo,findem[pnum].findme[0],&preset_map[preset_index], 0);
	  if (!found) {
	    fprintf(stderr,"Did not find simple preset %d\n",preset_index);
	    abort();
	  }
	  preset_map[preset_index].metric_count = 1; /* one metric if not derived */
	  strncpy(preset_map[preset_index].note,findem[pnum].findme[0], PAPI_MAX_STR_LEN);
	  did_something++;
	}
      else 
	{
	  hwd_preset_t tmp;
	  unsigned int tmp_gps[2] = {0xffffffff, 0xffffffff}; /* all groups true */
	  
	  memset(&tmp,0x00,sizeof(tmp));
	  tmp.derived = findem[pnum].derived;
	  /* Find info for all the metrics (up to 8!) in this derived event */
	  for (pmc = 0; pmc < POWER_MAX_COUNTERS && findem[pnum].findme[pmc]; pmc++)
	    {
	      DBG((stderr,"Looking for preset %d, %s\n",pnum,findem[pnum].findme[pmc]));
	      found = find_hwcounter_gps(pminfo,findem[pnum].findme[pmc],&tmp, pmc);
	      if (!found) {
		fprintf(stderr,"Did not find compund event %s\n",findem[pnum].findme[pmc]);
		abort();
	      }
	      /* append the metric name to the event descriptor */
	      if (strlen(tmp.note)+strlen(findem[pnum].findme[pmc]+1) < PAPI_MAX_STR_LEN)
		{
		  strcat(tmp.note,findem[pnum].findme[pmc]);
		  strcat(tmp.note,",");
		}
	      if (findem[pnum].preset == PAPI_TOT_CYC) {
		DBG((stderr,"cmd[%d][] = %d %d %d %d %d %d %d %d\n",pmc,tmp.counter_cmd[pmc][0],
		  tmp.counter_cmd[pmc][1],tmp.counter_cmd[pmc][2],tmp.counter_cmd[pmc][3],
		  tmp.counter_cmd[pmc][4],tmp.counter_cmd[pmc][5],tmp.counter_cmd[pmc][6],
		  tmp.counter_cmd[pmc][7]));
	      }
	      /* Collect available groups containing EVERY metric */
	      tmp_gps[0] &= tmp.gps[0];
	      tmp_gps[1] &= tmp.gps[1];
	    }

	  /* If we've got at least one group left... */
	  if ((tmp_gps[0] | tmp_gps[1]) != 0)
	    {
	      tmp.gps[0] = tmp_gps[0];
	      tmp.gps[1] = tmp_gps[1];
	      tmp.note[strlen(tmp.note)-1] = '\0';
	      tmp.metric_count = pmc;
	      preset_map[preset_index] = tmp;
	      DBG((stderr,"Found compound preset %d in groups 0x%x 0x%x\n",preset_index,preset_map[preset_index].gps[1],preset_map[preset_index].gps[0]));
	      did_something++;
	      continue;
	    }
	  fprintf(stderr,"Did not find compound preset %d\n",preset_index);	  
	  abort();
	}
    }
  return(did_something ? 0 : PAPI_ESBSTR);
}

#endif
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->counter_cmd.events[arg1] = arg2;
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  ptr->counter_cmd.events[arg1] = 0;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if (a->counter_cmd.events[cntr] == b->counter_cmd.events[cntr])
    return(1);

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval;
  pm_data_t data;

  retval = pm_get_data_mythread(&data);
  if (retval > 0)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+data.accu[i],global->hw_start[i],data.accu[i]));
#endif
      global->hw_start[i] = global->hw_start[i] + data.accu[i];
    }

  retval = pm_reset_data_mythread();
  if (retval > 0)
    return(retval);
   
  return(0);
}

static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
#endif
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

static int set_domain(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_DOM_USER:
      mode->b.user = 1;
      mode->b.kernel = 0;
      break;
    case PAPI_DOM_KERNEL:
      mode->b.user = 0;
      mode->b.kernel = 1;
      break;
    case PAPI_DOM_ALL:
      mode->b.user = 1;
      mode->b.kernel = 1;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_GRN_THR:
      mode->b.process = 0;
      mode->b.proctree = 0;
      break;
    /* case PAPI_GRN_PROC:
      mode->b.process = 1;
      mode->b.proctree = 0;
      break;
    case PAPI_GRN_PROCG:
      mode->b.process = 0;
      mode->b.proctree = 1;
      break; */
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_default_domain(EventSetInfo *zero, int domain)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain));
}

static int set_default_granularity(EventSetInfo *zero, int granularity)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_granularity(current_state,granularity));
}

static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

static void init_config(hwd_control_state_t *ptr)
{
  int i, j;

#ifdef _POWER4
  /* Power4 machines must count by groups */
  ptr->counter_cmd.mode.b.is_group = 1;
#endif
#ifndef _POWER4
  memset(ptr->native, 0, sizeof(hwd_native_t)*POWER_MAX_COUNTERS);
#endif

  for (i = 0; i < _papi_system_info.num_cntrs; i++) {
    ptr->preset[i] = COUNT_NOTHING;
    ptr->counter_cmd.events[i] = COUNT_NOTHING;
#ifndef _POWER4
 	ptr->native[i].position=COUNT_NOTHING;
#endif
	/*ptr->native[i].link=COUNT_NOTHING;*/
 }
#ifndef _POWER4
  for(i=0;i<POWER_MAX_COUNTERS_MAPPING;i++){
    ptr->allevent[i]=COUNT_NOTHING;
	for (j = 0; j < _papi_system_info.num_cntrs; j++) {
	  ptr->emap[i][j]=COUNT_NOTHING;
	}
  }
  ptr->hwd_idx=0;
  ptr->hwd_idx_a=0;
  ptr->native_idx=0;
#endif
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

#ifdef _POWER4
  #define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS
#else
  #define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT
#endif

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
  _papi_system_info.exe_info.text_end = (caddr_t)&_etext;

#ifdef _POWER4
  retval = setup_p4_presets(&pminfo, &pmgroups);
#else
  retval = setup_all_presets(&pminfo);
#endif

  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  timebasestruct_t t;
  long long retval;

  read_real_time(&t,TIMEBASE_SZ);
  time_base_to_time(&t,TIMEBASE_SZ);
  retval = (t.tb_high * 1000000) + t.tb_low / 1000;
  return(retval);
}

long long _papi_hwd_get_real_cycles (void)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

long long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  long long retval;
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);
  return(retval);
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(zero);
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error");
  pm_error(where,error);
}

int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  retval = get_memory_info(&_papi_system_info.mem_info);
  if (retval)
    return(retval);

  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_system_info.hw_info.totalcpus,
       _papi_system_info.hw_info.vendor_string,
       _papi_system_info.hw_info.model_string,
       _papi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Initialize our global machdep. */

  init_config(zero->machdep);

  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (POWER_MAX_COUNTERS-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
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

static void set_hwcntr_codes(int selector, unsigned char *from, int *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = from[i];
	}
    }
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
  
#ifndef _POWER4
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
#endif
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef _POWER4

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

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#else
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int master_selector;
  int out_command, out_operand_index;
  int i,g;
  hwd_control_state_t tmp_state;
  hwd_preset_t *this_preset;
  int hwd_idx = 0;
  unsigned int event_code;
  unsigned int tmp_gps[2];  
  int hwcntr_num, metric;
  EventInfo_t *zeroth;

  DBG((stderr,"EventCode %x \n",EventCode));

  /* mask off the preset bit */
  event_code = EventCode & PRESET_AND_MASK;

  if (EventCode == PAPI_FP_INS)
    DBG((stderr,"PAPI_FP_INS Groups: 0x%x 0x%x\n", preset_map[event_code].gps[1],preset_map[event_code].gps[0]));

  /* Do a preliminary check to eliminate preset events that aren't
     supported on this platform */
  if (EventCode & PRESET_MASK)
    {
      /* Make sure it lives in at least one group */
      if ((preset_map[event_code].gps[0] == 0)
	&& (preset_map[event_code].gps[1] == 0))
	return(PAPI_ENOEVNT);
    }

  /* Copy this_state into tmp_state. We can muck around with tmp and
     bail in case of failure and leave things unchanged. tmp_state 
     gets written back to this_state only if everything goes OK. */
  tmp_state = *this_state;

  /* If all slots are empty, initialize the state. */
  if (tmp_state.master_selector == 0)
    init_config(&tmp_state);

  /* If all counters are full, return a conflict error. */
  if ((tmp_state.master_selector & 0xff) == 0xff)
    return(PAPI_ECNFLCT);

  /* Find the first available slot in the state map. Each filled slot
     has a non-zero selector associated with it. Slots can be filled
     with simple events, derived events (multiple metrics), or native
     events. Because of derived events, the counters may fill before
     all slots are full. But because derived metrics can overlap, slots
     may fill before counters... */ 
  while ((tmp_state.selector[hwd_idx]) && (hwd_idx < POWER_MAX_COUNTERS))
    hwd_idx++;

  if (hwd_idx == POWER_MAX_COUNTERS) 
    return(PAPI_ECNFLCT); /* This should never happen unless the mapping code fails */

  /* Add the new event code to the list */
  tmp_state.preset[hwd_idx] = EventCode;

#if 0
  DBG((stderr,"hwd_idx %d \n",hwd_idx));
  dump_state(this_state);
#endif

  /* Scan the list and look for a common group */

  /* First, clear all selectors and counter commands */
  tmp_state.master_selector = 0;
  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++)
    {
      tmp_state.counter_cmd.events[hwd_idx] = COUNT_NOTHING;
      tmp_state.selector[hwd_idx] = 0;
    }

  /* Second, scan events to collect candidate groups */
  tmp_gps[0] = tmp_gps[1] = 0xffffffff;
  for (hwd_idx=0; hwd_idx < POWER_MAX_COUNTERS; hwd_idx++)
    {
      event_code = tmp_state.preset[hwd_idx];
      if (event_code == COUNT_NOTHING) break;

      /* look for native events separately */
      if ((event_code & PRESET_MASK) == 0)
	{
	  int native_gps[2] = {0, 0};

	  hwcntr_num = event_code & 0xff;
	  metric = event_code >> 8;
	  for (g = 0; g < MAX_GROUPS; g++)
	    {
	      /* scan all groups for this metric in this counter */
	      if (group_map[g].counter_cmd[hwcntr_num] == metric)
		{
		  native_gps[g/32] |= 1 << (g%32);
		}
	    }
	  tmp_gps[0] &= native_gps[0];
	  tmp_gps[1] &= native_gps[1];
          DBG((stderr,"native -- hwd_idx: %d, Groups: 0x%x 0x%x\n",hwd_idx, tmp_gps[1],tmp_gps[0]));
	}
      /* simple presets and derived events have predefined groups */
      else
	{
	  event_code &= PRESET_AND_MASK;
	  tmp_gps[0] &= preset_map[event_code].gps[0];
	  tmp_gps[1] &= preset_map[event_code].gps[1];
          DBG((stderr,"preset -- hwd_idx: %d, Groups: 0x%x 0x%x\n",hwd_idx, tmp_gps[1],tmp_gps[0]));
	}
    }

  if (tmp_gps[0] == 0 && tmp_gps[1] == 0) {
    return(PAPI_ECNFLCT); /* No group exists that contains all these metrics */
  }

  /* Third, pick the first available group (no particular reason) and its group id */
  for (g = 0; g < MAX_GROUPS; g++)
    {
      if (tmp_gps[g/32] & (1 << (g%32)))
	break;
    }
  /* for programming by groups, the first counter entry gets the group id */
  tmp_state.counter_cmd.events[0] = group_map[g].group_id;

  /* Fourth, rescan all available events to identify selector mask for each.
     This allows us to deconstruct the counter values on read. */

  /* We do this by comparing the possible counter_cmd values for each metric 
     of an event against the actual counter_cmd values of the selected group.
     If they match, that counter is used for this metric and event. */

  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++)
  {
    event_code = tmp_state.preset[hwd_idx];
    if (event_code == COUNT_NOTHING) break;

    /* process native events */
    if ((event_code & PRESET_MASK) == 0)
    {
      /* if we got this far, the native event
	 MUST be in the specified counter */
      hwcntr_num = event_code & 0xff;
      tmp_state.selector[hwd_idx] = 1 << hwcntr_num;
      out_command = NOT_DERIVED;
   }
    else /* its a preset event */
    {
      /* capture the derived state of the current event code */
      if (event_code == EventCode)
	out_command = preset_map[event_code & PRESET_AND_MASK].derived;
      else out_command = NOT_DERIVED;

      /* Dereference this preset for cleaner access */
      this_preset = &(preset_map[event_code & PRESET_AND_MASK]);

      /* Process all available metrics for this event.
	 This may be as many as 8 for derived events */
      for (metric=0; metric < this_preset->metric_count; metric++)
      {
  DBG((stderr,"pm_codes %d %d %d %d %d %d %d %d\n",
    this_preset->counter_cmd[metric][0],this_preset->counter_cmd[metric][1],
    this_preset->counter_cmd[metric][2],this_preset->counter_cmd[metric][3],
    this_preset->counter_cmd[metric][4],this_preset->counter_cmd[metric][5],
    this_preset->counter_cmd[metric][6],this_preset->counter_cmd[metric][7]));
  DBG((stderr,"pm_codes %d %d %d %d %d %d %d %d\n",
    group_map[g].counter_cmd[0],group_map[g].counter_cmd[1],
    group_map[g].counter_cmd[2],group_map[g].counter_cmd[3],
    group_map[g].counter_cmd[4],group_map[g].counter_cmd[5],
    group_map[g].counter_cmd[6],group_map[g].counter_cmd[7]));
	for (i=0;i<POWER_MAX_COUNTERS; i++) {
	  if (this_preset->counter_cmd[metric][i] == group_map[g].counter_cmd[i]) {
	    tmp_state.selector[hwd_idx] |= 1 << i;
	    if (out_command && (metric == 0))
	      out_operand_index = i;
	    break;
	  }
	}
      }
    }
    tmp_state.master_selector |= tmp_state.selector[hwd_idx];
  }

  /* Everything worked. Copy temporary state back to current state */

  /* First, find out which event in the event array is this one */
  for (hwd_idx=0; hwd_idx<POWER_MAX_COUNTERS; hwd_idx++) /* scan across available events */
    {
      if (EventCode == tmp_state.preset[hwd_idx]) break;
    }
  
  /* Next, update the high level selectors for all earlier events, 
     in case a remapping occurred. This is REALLY UGLY code, because it
     requires that one assume the out pointer is the ith member of a 
     contiguous array of EventInfo_t structures and computes the address
     of the 0th member... */
  zeroth = &(out[-hwd_idx]);
  for (i=0; i<hwd_idx; i++)
    {
      zeroth[i].selector = tmp_state.selector[i];
    }

  /* Finally, inform the upper level of the necessary info for this event. */
  out->code = EventCode;
  out->selector = tmp_state.selector[hwd_idx];
  out->command = out_command;
  out->operand_index = out_operand_index;
  *this_state = tmp_state;
 
#if 0
  DBG((stderr,"success \n"));
  dump_state(this_state);
#endif

  return(PAPI_OK);
}

#endif
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
#ifndef _POWER4
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

#else
  int i, selector, used, preset_index, EventCode;
  
  /* Find out which counters used. */
  used = in->selector;
  EventCode = in->code;

  /* scan across events in this set */
  for (i=0; i<POWER_MAX_COUNTERS; i++)
    {
      if (EventCode == this_state->preset[i]) break;
    }

  /* Make sure the event was found */
  if (i == POWER_MAX_COUNTERS)
    return(PAPI_ENOEVNT);

  selector = this_state->selector[i];

  /* Make sure the selector is set. */
  if (selector == 0)
    return(PAPI_ENOEVNT);

  /* Check if these counters aren't used. */
  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out selector bits that are part of this event. */
  this_state->master_selector ^= selector;
  this_state->selector[i] = 0;

  /* Clear out the preset for this event */
  this_state->preset[i] = COUNT_NOTHING;
#endif

#if 0
  dump_state(this_state);
#endif

  return(PAPI_OK);
}


int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)

{
  return(PAPI_ESBSTR);
}

void dump_cmd(pm_prog_t *t)
{
  fprintf(stderr,"mode.b.threshold %d\n",t->mode.b.threshold);
  fprintf(stderr,"mode.b.spare %d\n",t->mode.b.spare);
  fprintf(stderr,"mode.b.process %d\n",t->mode.b.process);
  fprintf(stderr,"mode.b.kernel %d\n",t->mode.b.kernel);
  fprintf(stderr,"mode.b.user %d\n",t->mode.b.user);
  fprintf(stderr,"mode.b.count %d\n",t->mode.b.count);
  fprintf(stderr,"mode.b.proctree %d\n",t->mode.b.proctree);
  fprintf(stderr,"events[0] %d\n",t->events[0]);
  fprintf(stderr,"events[1] %d\n",t->events[1]);
  fprintf(stderr,"events[2] %d\n",t->events[2]);
  fprintf(stderr,"events[3] %d\n",t->events[3]);
  fprintf(stderr,"events[4] %d\n",t->events[4]);
  fprintf(stderr,"events[5] %d\n",t->events[5]);
  fprintf(stderr,"events[6] %d\n",t->events[6]);
  fprintf(stderr,"events[7] %d\n",t->events[7]);
  fprintf(stderr,"reserved %d\n",t->reserved);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef _POWER4

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

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#else
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

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
      
      /* only merge if it's the same group */
      if (this_state->counter_cmd.events[0] != current_state->counter_cmd.events[0])
	 return(PAPI_ECNFLCT);

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

      hwcntrs_in_both = this_state->master_selector & current_state->master_selector;

      for (i = 0; i < _papi_system_info.num_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */
	  
	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      zero->multistart.SharedDepth[i]++;
	      ESI->hw_start[i] = zero->hw_start[i];
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->master_selector & hwcntr)
	    {
	      current_state->master_selector |= hwcntr;
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
      memcpy(current_state,this_state,sizeof(hwd_control_state_t));

      retval = pm_set_program_mythread(&current_state->counter_cmd);
      if (retval > 0) 
        return(retval);

   }

  /* Set up the new merged control structure */
  
#if 0
  dump_state(this_state);
  dump_state(current_state);
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* (Re)start the counters */
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
} 

#endif
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


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

int _papi_hwd_reset(EventSetInfo *ESI, EventSetInfo *zero)
{
  int i, retval;

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    ESI->hw_start[i] = zero->hw_start[i];

  return(PAPI_OK);
}

static long long handle_derived_add(int selector, long long *from)
{
  int pos;
  long long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while (pos = ffs(selector))
    {
      DBG((stderr,"Compound event, subtracting %lld to %lld\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  return((long long)((float)units * _papi_system_info.hw_info.mhz * 1000000.0 / (float)cycles));
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  assert(pos >= 0);

  return(units_per_second(from[pos],from[operand_index]));
}

static long long handle_derived_add_ps(int operand_index, int selector, long long *from)
{
  int add_selector = selector ^ (1 << operand_index);
  long long tmp = handle_derived_add(add_selector, from);
  return(units_per_second(tmp, from[operand_index]));
}

static long long handle_derived(EventInfo_t *cmd, long long *from)
{
  switch (cmd->command)
    {
    case DERIVED_ADD: 
      return(handle_derived_add(cmd->selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(cmd->operand_index, cmd->selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(cmd->operand_index, cmd->selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(cmd->operand_index, cmd->selector, from));
    default:
      abort();
    }
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[POWER_MAX_COUNTERS];
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  retval = correct_local_hwcounters(zero, ESI, correct);
  if (retval)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == 0)
	continue;

     DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));
#ifdef _POWER4
     DBG((stderr,"Group is %d\n",this_state->counter_cmd.events[0]));
#endif
     assert(selector != 0);

      /* If this is not a derived event */

      DBG((stderr,"Derived: %d\n", ESI->EventInfoArray[i].command));
      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  assert(shift_cnt >= 0);
	  events[j] = correct[shift_cnt];
	}
      
      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct);
	
      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
	return(PAPI_OK);
    }

  /* Should never get here */

  return(PAPI_EBUG);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
    case PAPI_SET_GRANUL:
      return(set_granularity(option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
#endif
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(EventSetInfo *master, EventSetInfo *ESI, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  pm_delete_program_mythread();
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{
  int events;

#ifdef _POWER4
  events = preset_map[preset_index].metric_count;
#else
  events = preset_map[preset_index].selector[0];
#endif

  if (events == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *i)
{
#ifdef DEBUG
  ucontext_t *info;
  info = (ucontext_t *)i;
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info->uc_mcontext.jmp_context.iar));
#endif

  _papi_hwi_dispatch_overflow_signal(i); 
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  if (overflow_option->threshold == 0)
    {
      this_state->timer_ms = 0;
      overflow_option->timer_ms = 0;
    }
  else
    {
      this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
      overflow_option->timer_ms = 1;
    }

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

int _papi_hwd_stop_profiling(EventSetInfo *ESI, EventSetInfo *master)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_jmpbuf.jmp_context.iar;

  return(location);
}

static volatile int lock_var = 0;
static atomic_p lock;

void _papi_hwd_lock_init(void)
{
  lock = (int *)&lock_var;
}

void _papi_hwd_lock(void)
{
  while (_check_lock(lock,0,1) == TRUE)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
}

void _papi_hwd_unlock(void)
{
  _clear_lock(lock, 0);
}

/* Machine info structure. -1 is initialized by _papi_hwd_init. */

papi_mdi _papi_system_info = { "$Id$",
			      1.0, /*  version */
			       -1,  /*  cpunum */
			       { 
				 -1,  /*  ncpu */
				  1,  /*  nnodes */
				 -1,  /*  totalcpus */
				 -1,  /*  vendor */
				 "",  /*  vendor string */
				 -1,  /*  model */
				 "",  /*  model string */
				0.0,  /*  revision */
				 -1  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)0x10000200,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 ""
			       },
                               { 0,  /*total_tlb_size*/
                                 0,  /*itlb_size */
                                 0,  /*itlb_assoc*/
                                 0,  /*dtlb_size */
                                 0, /*dtlb_assoc*/
                                 0, /*total_L1_size*/
                                 0, /*L1_icache_size*/
                                 0, /*L1_icache_assoc*/
                                 0, /*L1_icache_lines*/
                                 0, /*L1_icache_linesize*/
                                 0, /*L1_dcache_size */
                                 0, /*L1_dcache_assoc*/
                                 0, /*L1_dcache_lines*/
                                 0, /*L1_dcache_linesize*/
                                 0, /*L2_cache_size*/
                                 0, /*L2_cache_assoc*/
                                 0, /*L2_cache_lines*/
                                 0, /*L2_cache_linesize*/
                                 0, /*L3_cache_size*/
                                 0, /*L3_cache_assoc*/
                                 0, /*L3_cache_lines*/
                                 0  /*L3_cache_linesize*/
                               },
			       -1,  /*  num_cntrs */
			       -1,  /*  num_gp_cntrs */
			       -1,  /*  grouped_counters */
			       -1,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_THR,  /* default granularity */
			        0,  /* We can use add_prog_event */
			        0,  /* We can write the counters */
			        0,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        0,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW Read also resets the counters */
			        sizeof(hwd_control_state_t), 
			        NULL };

