/*
 * File:   	linux-ia64.c
 *
 * Mods:	Kevin London
 *		london@cs.utk.edu
 */

#include SUBSTRATE

#ifdef PFM06A
static preset_search_t preset_search_map[] = { 
  {PAPI_L1_TCM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L1_ICM,0,{"L2_INST_DEMAND_READS",0,0,0}},
  {PAPI_L1_DCM,0,{"L1D_READ_MISSES_RETIRED",0,0,0}},
  {PAPI_L2_TCM,0,{"L2_MISSES",0,0,0}},
  {PAPI_L2_DCM,DERIVED_SUB,{"L2_MISSES","L3_READS.INST_READS.ALL",0,0}},
  {PAPI_L2_ICM,0,{"L3_READS.INST_READS.ALL",0,0,0}},
  {PAPI_L3_TCM,0,{"L3_MISSES",0,0,0}},
  {PAPI_L3_ICM,0,{"L3_READS.INST_READS.MISS",0,0,0}},
  {PAPI_L3_DCM,DERIVED_ADD,{"L3_READS.DATA_READS.MISS","L3_WRITES.DATA_WRITES.MISS",0,0}},
  {PAPI_L3_LDM,DERIVED_ADD,{"L3_READS.DATA_READS.MISS","L3_READS.INST_READS.MISS",0,0}},
  {PAPI_L3_STM,0,{"L3_WRITES.DATA_WRITES.MISS",0,0,0}},
  {PAPI_L1_LDM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L2_LDM,DERIVED_ADD,{"L3_READS.DATA_READS.ALL","L3_READS.INST_READS.ALL",0,0}},
  {PAPI_L2_STM,0,{"L3_WRITES.ALL_WRITES.ALL",0,0,0}},
  {PAPI_L3_DCH,DERIVED_ADD,{"L3_READS.DATA_READS.HIT","L3_WRITES.DATA_WRITES.HIT",0,0}},
  {PAPI_L1_DCH,DERIVED_SUB,{"L1D_READS_RETIRED","L1D_READ_MISSES_RETIRED",0,0}},
  {PAPI_L1_DCA,0,{"L1D_READS_RETIRED",0,0,0}},
  {PAPI_L2_DCA,0,{"L2_DATA_REFERENCES.ALL",0,0,0}},
  {PAPI_L3_DCA,DERIVED_ADD,{"L3_READS.DATA_READS.ALL","L3_WRITES.DATA_WRITES.ALL",0,0}},
  {PAPI_L2_DCR,0,{"L2_DATA_REFERENCES.READS",0,0,0}},
  {PAPI_L3_DCR,0,{"L3_READS.DATA_READS.ALL",0,0,0}},
  {PAPI_L2_DCW,0,{"L2_DATA_REFERENCES.WRITES",0,0,0}},
  {PAPI_L3_DCW,0,{"L3_WRITES.DATA_WRITES.ALL",0,0,0}},
  {PAPI_L3_ICH,0,{"L3_READS.INST_READS.HIT",0,0,0}},
  {PAPI_L1_ICR,DERIVED_ADD,{"L1I_PREFETCH_READS","L1I_DEMAND_READS",0,0}},
  {PAPI_L2_ICR,DERIVED_ADD,{"L2_INST_DEMAND_READS","L2_INST_PREFETCH_READS",0,0}},
  {PAPI_L3_ICR,0,{"L3_READS.INST_READS.ALL",0,0,0}},
  {PAPI_TLB_DM,0,{"DTLB_MISSES",0,0,0}},
  {PAPI_TLB_IM,0,{"ITLB_MISSES_FETCH",0,0,0}},
  {PAPI_MEM_SCY,0,{"MEMORY_CYCLE",0,0,0}},
  {PAPI_STL_ICY,0,{"INST_FETCH_CYCLE",0,0,0}},
  {PAPI_BR_INS,0,{"BRANCH_EVENT",0,0,0}},
  {PAPI_BR_NTK,DERIVED_ADD,{"BRANCH_PATH.ALL.NT_OUTCOMES_CORRECTLY_PREDICTED","BRANCH_PATH.ALL.TK_OUTCOMES_INCORRECTLY_PREDICTED",0,0}},
  {PAPI_BR_TKN,DERIVED_ADD,{"BRANCH_PATH.ALL.TK_OUTCOMES_CORRECTLY_PREDICTED","BRANCH_PATH.ALL.NT_OUTCOMES_INCORRECTLY_PREDICTED",0,0}},
  {PAPI_BR_PRC,0,{"BRANCH_PREDICATOR.ALL.CORRECT_PREDICTIONS",0,0,0}},
  {PAPI_BR_MSP,DERIVED_ADD,{"BRANCH_PREDICATOR.ALL.WRONG_PATH","BRANCH_PREDICATOR.ALL.WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,DERIVED_ADD,{"FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0,0}},
  {PAPI_TOT_INS,0,{"IA64_INST_RETIRED",0,0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_FLOPS,DERIVED_ADD_PS,{"CPU_CYCLES","FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0}},
  {0,0,{0,0,0,0}}};
#else
#ifndef ITANIUM2
static preset_search_t preset_search_map[] = { 
  {PAPI_L1_TCM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L1_ICM,0,{"L2_INST_DEMAND_READS",0,0,0}},
  {PAPI_L1_DCM,0,{"L1D_READ_MISSES_RETIRED",0,0,0}},
  {PAPI_L2_TCM,0,{"L2_MISSES",0,0,0}},
  {PAPI_L2_DCM,DERIVED_SUB,{"L2_MISSES","L3_READS_INST_READS_ALL",0,0}},
  {PAPI_L2_ICM,0,{"L3_READS_INST_READS_ALL",0,0,0}},
  {PAPI_L3_TCM,0,{"L3_MISSES",0,0,0}},
  {PAPI_L3_ICM,0,{"L3_READS_INST_READS_MISS",0,0,0}},
  {PAPI_L3_DCM,DERIVED_ADD,{"L3_READS_DATA_READS_MISS","L3_WRITES_DATA_WRITES_MISS",0,0}},
  {PAPI_L3_LDM,DERIVED_ADD,{"L3_READS_DATA_READS_MISS","L3_READS_INST_READS_MISS",0,0}},
  {PAPI_L3_STM,0,{"L3_WRITES_DATA_WRITES_MISS",0,0,0}},
  {PAPI_L1_LDM,DERIVED_ADD,{"L1D_READ_MISSES_RETIRED","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L2_LDM,DERIVED_ADD,{"L3_READS_DATA_READS_ALL","L3_READS_INST_READS_ALL",0,0}},
  {PAPI_L2_STM,0,{"L3_WRITES_ALL_WRITES_ALL",0,0,0}},
  {PAPI_L3_DCH,DERIVED_ADD,{"L3_READS_DATA_READS_HIT","L3_WRITES_DATA_WRITES_HIT",0,0}},
  {PAPI_L1_DCH,DERIVED_SUB,{"L1D_READS_RETIRED","L1D_READ_MISSES_RETIRED",0,0}},
  {PAPI_L1_DCA,0,{"L1D_READS_RETIRED",0,0,0}},
  {PAPI_L2_DCA,0,{"L2_DATA_REFERENCES_ALL",0,0,0}},
  {PAPI_L3_DCA,DERIVED_ADD,{"L3_READS_DATA_READS_ALL","L3_WRITES_DATA_WRITES_ALL",0,0}},
  {PAPI_L2_DCR,0,{"L2_DATA_REFERENCES_READS",0,0,0}},
  {PAPI_L3_DCR,0,{"L3_READS_DATA_READS_ALL",0,0,0}},
  {PAPI_L2_DCW,0,{"L2_DATA_REFERENCES_WRITES",0,0,0}},
  {PAPI_L3_DCW,0,{"L3_WRITES_DATA_WRITES_ALL",0,0,0}},
  {PAPI_L3_ICH,0,{"L3_READS_INST_READS_HIT",0,0,0}},
  {PAPI_L1_ICR,DERIVED_ADD,{"L1I_PREFETCH_READS","L1I_DEMAND_READS",0,0}},
  {PAPI_L2_ICR,DERIVED_ADD,{"L2_INST_DEMAND_READS","L2_INST_PREFETCH_READS",0,0}},
  {PAPI_L3_ICR,0,{"L3_READS_INST_READS_ALL",0,0,0}},
  {PAPI_TLB_DM,0,{"DTLB_MISSES",0,0,0}},
  {PAPI_TLB_IM,0,{"ITLB_MISSES_FETCH",0,0,0}},
  {PAPI_MEM_SCY,0,{"MEMORY_CYCLE",0,0,0}},
  {PAPI_STL_ICY,0,{"UNSTALLED_BACKEND_CYCLE",0,0,0}},
  {PAPI_BR_INS,0,{"BRANCH_EVENT",0,0,0}},
  {PAPI_BR_NTK,DERIVED_ADD,{"BRANCH_PATH_ALL_NT_OUTCOMES_CORRECTLY_PREDICTED","BRANCH_PATH_ALL_TK_OUTCOMES_INCORRECTLY_PREDICTED",0,0}},
  {PAPI_BR_TKN,DERIVED_ADD,{"BRANCH_PATH_ALL_TK_OUTCOMES_CORRECTLY_PREDICTED","BRANCH_PATH_ALL_NT_OUTCOMES_INCORRECTLY_PREDICTED",0,0}},
  {PAPI_BR_PRC,0,{"BRANCH_PREDICTOR_ALL_ALL_PREDICTIONS",0,0,0}}, 
  {PAPI_BR_MSP,DERIVED_ADD,{"BRANCH_PREDICTOR_ALL_WRONG_PATH","BRANCH_PREDICTOR_ALL_WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,DERIVED_ADD,{"FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0,0}},
  {PAPI_TOT_INS,0,{"IA64_INST_RETIRED",0,0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_LST_INS,DERIVED_ADD,{"LOADS_RETIRED","STORES_RETIRED",0,0}},
  {PAPI_FLOPS,DERIVED_ADD_PS,{"CPU_CYCLES","FP_OPS_RETIRED_HI","FP_OPS_RETIRED_LO",0}},
  {0,0,{0,0,0,0}}};
#else
static preset_search_t preset_search_map[] = {
  {PAPI_L1_TCM,DERIVED_ADD,{"L2_INST_DEMAND_READS","L1D_READ_MISSES_ALL",0,0}},
  {PAPI_L1_ICM,0,{"L2_INST_DEMAND_READS",0,0,0}},
  {PAPI_L1_DCM,0,{"L1D_READ_MISSES_ALL",0,0,0}},
  {PAPI_L2_TCM,0,{"L2_MISSES",0,0,0}},
  {PAPI_L2_DCM,0,{"L3_READS_DATA_READ_ALL",0,0,0}},
  {PAPI_L2_ICM,0,{"L3_READS_INST_FETCH_ALL",0,0,0}},
  {PAPI_L3_TCM,0,{"L3_MISSES",0,0,0}},
  {PAPI_L3_ICM,0,{"L3_READS_INST_FETCH_MISS",0,0,0}},
  {PAPI_L3_DCM,DERIVED_ADD,{"L3_READS_DATA_READ_MISS","L3_WRITES_DATA_WRITE_MISS",0,0}},
  {PAPI_L3_LDM,0,{"L3_READS_ALL_MISS",0,0,0}},
  {PAPI_L3_STM,0,{"L3_WRITES_DATA_WRITE_MISS",0,0,0}},
  {PAPI_L1_LDM,DERIVED_ADD,{"L1D_READ_MISSES_ALL","L2_INST_DEMAND_READS",0,0}},
  {PAPI_L2_LDM,0,{"L3_READS_ALL_ALL",0,0,0}},
  {PAPI_L2_STM,0,{"L3_WRITES_ALL_ALL",0,0,0}},
  {PAPI_L1_DCH,DERIVED_SUB,{"L1D_READS_SET1","L1D_READ_MISSES_ALL",0,0}},
  {PAPI_L2_DCH,DERIVED_SUB,{"L2_DATA_REFERENCES_L2_ALL","L2_MISSES",0,0}},
  {PAPI_L3_DCH,DERIVED_ADD,{"L3_READS_DATA_READ_HIT","L3_WRITES_DATA_WRITE_HIT",0,0}},
  {PAPI_L1_DCA,0,{"L1D_READS_SET1",0,0,0}},
  {PAPI_L2_DCA,0,{"L2_DATA_REFERENCES_L2_ALL",0,0,0}},
  {PAPI_L3_DCA,0,{"L3_REFERENCES",0,0,0}},
  {PAPI_L1_DCR,0,{"L1D_READS_SET1",0,0,0}},
  {PAPI_L2_DCR,0,{"L2_DATA_REFERENCES_L2_DATA_READS",0,0,0}},
  {PAPI_L3_DCR,0,{"L3_READS_DATA_READ_ALL",0,0,0}},
  {PAPI_L2_DCW,0,{"L2_DATA_REFERENCES_L2_DATA_WRITES",0,0,0}},
  {PAPI_L3_DCW,0,{"L3_WRITES_DATA_WRITE_ALL",0,0,0}},
  {PAPI_L3_ICH,0,{"L3_READS_DINST_FETCH_HIT",0,0,0}},
  {PAPI_L1_ICR,DERIVED_ADD,{"L1I_PREFETCHES","L1I_READS",0,0}},
  {PAPI_L2_ICR,DERIVED_ADD,{"L2_INST_DEMAND_READS","L2_INST_PREFETCHES",0,0}},
  {PAPI_L3_ICR,0,{"L3_READS_INST_FETCH_ALL",0,0,0}},
  {PAPI_L1_ICA,DERIVED_ADD,{"L1I_PREFETCHES","L1I_READS",0,0}},
  {PAPI_L2_TCH,DERIVED_SUB,{"L2_REFERENCES","L2_MISSES",0,0}},
  {PAPI_L3_TCH,DERIVED_SUB,{"L3_REFERENCES","L3_MISSES",0,0}},
  {PAPI_L2_TCA,0,{"L2_REFERENCES",0,0,0}},
  {PAPI_L3_TCA,0,{"L3_REFERENCES",0,0,0}},
  {PAPI_L3_TCR,0,{"L3_READS_ALL_ALL",0,0,0}},
  {PAPI_L3_TCW,0,{"L3_WRITES_ALL_ALL",0,0,0}},
  {PAPI_TLB_DM,0,{"L2DTLB_MISSES",0,0,0}},
  {PAPI_TLB_IM,0,{"ITLB_MISSES_FETCH_L2ITLB",0,0,0}},
  {PAPI_BR_INS,0,{"BRANCH_EVENT",0,0,0}},
  {PAPI_BR_NTK,DERIVED_ADD,{"BR_PATH_PRED_ALL_MISPRED_NOTTAKEN","BR_PATH_PRED_ALL_OKPRED_NOTTAKEN",0,0}},
  {PAPI_BR_TKN,DERIVED_ADD,{"BR_PATH_PRED_ALL_OKPRED_TAKEN","BR_PATH_PRED_ALL_MISPRED_TAKEN",0,0}},
  {PAPI_BR_PRC,0,{"BR_MISPRED_DETAIL_ALL_CORRECT_PRED",0,0,0}},
  {PAPI_BR_MSP,DERIVED_ADD,{"BR_MISPRED_DETAIL_ALL_WRONG_PATH","BR_MISPRED_DETAIL_ALL_WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,0,{"FP_OPS_RETIRED",0,0,0}},
  {PAPI_TOT_INS,DERIVED_ADD,{"IA64_INST_RETIRED","IA32_INST_RETIRED",0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_FLOPS,DERIVED_PS,{"CPU_CYCLES","FP_OPS_RETIRED",0,0}},
  {0,0,{0,0,0,0}}};
#endif
#endif
static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS];

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Low level functions, should not handle errors, just return codes. */

#ifdef PFM06A
void
pfm_start(void)
{
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
}

/*
 * Stops monitoring for user-level monitors
 */
void
pfm_stop(void)
{
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
}
#endif

static inline char *search_cpu_info(FILE *f, char *search_str, char *line)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

  char *s;

  while (fgets(line, 256, f) != NULL)
    {
      if (strstr(line, search_str) != NULL)
	{
	  /* ignore all characters in line up to : */
	  for (s = line; *s && (*s != ':'); ++s)
	    ;
	  if (*s)
	    return(s);
	}
    }
  return(NULL);

  /* End stolen code */
}

static inline unsigned long get_cycles(void)
{
	unsigned long tmp;

	/* XXX: need more to adjust for Itanium itc bug */
	__asm__ __volatile__("mov %0=ar.itc" : "=r"(tmp) :: "memory");

	return tmp;
}

/* Dumb hack to make sure I get the cycle time correct. */

inline static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  float correction = 4000.0, mhz;

  /* Warm the cache */

  ostamp = get_cycles();
  usleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  sleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  return(mhz);
}

/* Begin blatantly stolen code  
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com> */

#ifdef PFM06A
static inline int
gen_events(char **arg, pfm_event_config_t *evt)
#else
static inline int
gen_events(char **arg, pfmlib_param_t *evt)
#endif
{
	int ev;
	int cnt=0;

	if (arg == NULL) return -1;

	while (*arg) {

		if (cnt == PMU_MAX_COUNTERS) goto too_many;
		/* must match vcode only */
#ifdef PFM06A
		if ((ev = pfm_findevent(*arg,0)) == -1) goto error;
		evt->pec_evt[cnt++] = ev;
#else
		if ((ev = pfm_find_event(*arg,0,&(evt->pfp_evt[cnt++]))) 
			!= PFMLIB_SUCCESS) goto error;
#endif

		arg++;
	}
#ifdef PFM06A
	evt->pec_count = cnt;
#else
	evt->pfp_count = cnt;
#endif
	return 0;
error:
	return -1;
too_many:
	return -1;
}

/* End blatantly stolen code 
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com> */

static inline int setup_all_presets()
{
  int pnum, i, preset_index;
  char **name = NULL, note[PAPI_MAX_STR_LEN];

  memset(preset_map,0x0,sizeof(preset_map));

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if (preset_search_map[pnum].preset == 0)
	break;
      preset_index = preset_search_map[pnum].preset ^ PRESET_MASK; 
      if (gen_events(preset_search_map[pnum].findme, &preset_map[preset_index].evt) == -1)
	abort();
      preset_map[preset_index].present = 1;
      preset_map[preset_index].derived = preset_search_map[pnum].derived;

      /* Itanium specific code, irrelevant for now */
      if (preset_search_map[pnum].preset == PAPI_FLOPS)
	preset_map[preset_index].operand_index = 0;
      else
	preset_map[preset_index].operand_index = 0;
      /* End itanium specific code */
	  
      strcpy(note,"");
      name = preset_search_map[pnum].findme;
      i = 0;
      while (name[i])
	{
	  strcat(note,name[i]);
	  if (name[++i])
	    strcat(note,",");
	  else
	    break;
	}
      strcpy(preset_map[preset_index].note,note);
    }
  if (pnum == 0)
    abort();
  return(PAPI_OK);
}

/* Utility functions */

/* Return new counter mask */

#ifdef PFM06A
inline static int set_hwcntr_codes(hwd_control_state_t *this_state, const pfm_event_config_t *from)
#else
inline static int set_hwcntr_codes(hwd_control_state_t *this_state, const pfmlib_param_t *from)
#endif
{
#ifdef PFM06A
  perfmon_req_t *pc = this_state->pc;
  pfm_event_config_t *evt = &this_state->evt;
  int i, orig_cnt = evt->pec_count;
#else
  pfarg_reg_t *pc = this_state->pc;
  pfmlib_param_t *evt = &this_state->evt;
  int i, orig_cnt = evt->pfp_count;
#endif
  int selector = 0;

  if (from)
    {
      /* Called from add_event */
      /* Merge the two evt structures into the old one */
      
#ifdef PFM06A
      for (i=0;i<from->pec_count;i++)
	evt->pec_evt[evt->pec_count++] = from->pec_evt[i];
      
      if ((from->pec_count) > PMU_MAX_COUNTERS)
	{
	bail:
	  evt->pec_count = orig_cnt;
	  return(PAPI_ECNFLCT);
#else
      for (i=0;i<from->pfp_count;i++)
	evt->pfp_evt[evt->pfp_count++] = from->pfp_evt[i];
      
      if ((from->pfp_count) > PMU_MAX_COUNTERS)
	{
	bail:
	  evt->pfp_count = orig_cnt;
	  return(PAPI_ECNFLCT);
#endif
	}
    }

  /* Recalcuate the perfmon_req_t structure, may also signal conflict */
#ifdef PFM06A
  if (pfm_dispatch_events(evt,pc,&evt->pec_count))
#else
  if (pfm_dispatch_events(evt,pc,&evt->pfp_count))
#endif
    {
      goto bail;
      return(PAPI_ECNFLCT);
    }

#ifdef PFM06A
   for (i=0;i<evt->pec_count;i++)
    {
      selector |= 1 << pc[i].pfr_reg.reg_num;
#else
   for (i=0;i<evt->pfp_count;i++)
    {
      selector |= 1 << pc[i].reg_num;
#endif
      DBG((stderr,"Selector is now 0x%x\n",selector));
    }

  return(selector);
} 

inline static int set_domain(hwd_control_state_t *this_state, int domain)
{
  int mode = 0, did = 0, i;
  
  if (domain & PAPI_DOM_USER)
    {
      did = 1;
      mode |= PFM_PLM3;
    }
  if (domain & PAPI_DOM_KERNEL)
    {
      did = 1;
      mode |= PFM_PLM0;
    }

  if (!did)
    return(PAPI_EINVAL);

#ifdef PFM06A
  this_state->evt.pec_plm = mode;
#else
  this_state->evt.pfp_dfl_plm = mode;
#endif

  /* Bug fix in case we don't call pfm_dispatch_events after this code */

  for (i=0;i<PMU_MAX_COUNTERS;i++)
    {
#ifdef PFM06A
      if (this_state->pc[i].pfr_reg.reg_num)
	{
	  perfmon_reg_t value;
	  DBG((stderr,"slot %d, register %ld active, config value 0x%lx\n",
	       i,this_state->pc[i].pfr_reg.reg_num,this_state->pc[i].pfr_reg.reg_value));

	  value.pmu_reg = 
	    this_state->pc[i].pfr_reg.reg_value;
	  value.pmc_plm = mode;
	  this_state->pc[i].pfr_reg.reg_value = 
	    value.pmu_reg;

	  DBG((stderr,"new config value 0x%lx\n",this_state->pc[i].pfr_reg.reg_value));
	}
#else
      if (this_state->pc[i].reg_num)
	{
#if defined(ITANIUM2)
	  pfm_ita2_reg_t value;
#else
	  pfm_ita_reg_t value;
#endif

	  DBG((stderr,"slot %d, register %d active, config value 0x%lx\n",
	       i,this_state->pc[i].reg_num,this_state->pc[i].reg_value));

	  value.reg_val = this_state->pc[i].reg_value;
	  value.pmc_plm = mode;
	  this_state->pc[i].reg_value = value.reg_val;

	  DBG((stderr,"new config value 0x%lx\n",this_state->pc[i].reg_value));
	}
#endif
    }
	
  return(PAPI_OK);
}

inline static void init_config(hwd_control_state_t *ptr)
{
  ptr->pid = getpid();
  set_domain(ptr,_papi_system_info.default_domain);
} 

static int get_system_info(void)
{
  pid_t pid;
  int tmp;
  float mhz;
  char maxargs[PAPI_MAX_STR_LEN], *t, *s;
  FILE *f;

  /* Path and args */

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  sprintf(maxargs,"/proc/%d/exe",(int)getpid());
  if (readlink(maxargs,_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == -1)
    return(PAPI_ESYS);
  sprintf(_papi_system_info.exe_info.name,"%s",basename(_papi_system_info.exe_info.fullname));

  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
    return -1;
 
  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.vendor = -1;

  rewind(f);
  s = search_cpu_info(f,"vendor",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.vendor_string,s+2);
    }

  rewind(f);
  s = search_cpu_info(f,"revision",maxargs);
  if (s)
    sscanf(s+1, "%d", &tmp);
  _papi_system_info.hw_info.revision = (float)tmp;

  rewind(f);
  s = search_cpu_info(f,"model",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.model_string,s+2);
    }

  rewind(f);
  s = search_cpu_info(f,"cpu MHz",maxargs);
  if (s)
    sscanf(s+1, "%f", &mhz);
  _papi_system_info.hw_info.mhz = mhz;

  DBG((stderr,"Detected MHZ is %f\n",_papi_system_info.hw_info.mhz));
  mhz = calc_mhz();
  DBG((stderr,"Calculated MHZ is %f\n",mhz));
  if (_papi_system_info.hw_info.mhz < mhz)
    _papi_system_info.hw_info.mhz = mhz;
  {
    int tmp = (int)_papi_system_info.hw_info.mhz;
    _papi_system_info.hw_info.mhz = (float)tmp;
  }
  DBG((stderr,"Actual MHZ is %f\n",_papi_system_info.hw_info.mhz));
  _papi_system_info.num_cntrs = 4;
  _papi_system_info.num_gp_cntrs = 4;

  /* Setup presets */

  tmp = setup_all_presets();
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

#ifdef PFM06A
inline static int counter_event_shared(const pfm_event_config_t *a, const pfm_event_config_t *b, int cntr)
#else
inline static int counter_event_shared(const pfmlib_param_t *a, const pfmlib_param_t *b, int cntr)
#endif
{
#ifdef PFM06A
  DBG((stderr,"%d %x vs %x \n",cntr,a->pec_evt[cntr],b->pec_evt[cntr]));
  if (a->pec_evt[cntr] == b->pec_evt[cntr])
#else
  DBG((stderr,"%d %x vs %x \n",cntr,a->pfp_evt[cntr],b->pfp_evt[cntr]));
  if (a->pfp_evt[cntr] == b->pfp_evt[cntr])
#endif
    return(1);

  return(0);
}

#ifdef PFM06A
inline static int counter_event_compat(const pfm_event_config_t *a, const pfm_event_config_t *b, int cntr)
#else
inline static int counter_event_compat(const pfmlib_param_t *a, const pfmlib_param_t *b, int cntr)
#endif
{
#ifdef PFM06A
  DBG((stderr,"%d %d vs. %d\n",cntr,a->pec_plm,b->pec_plm));
  if (a->pec_plm == b->pec_plm)
#else
  DBG((stderr,"%d %d vs. %d\n",cntr,a->pfp_dfl_plm,b->pfp_dfl_plm));
  if (a->pfp_plm == b->pfp_plm)
#endif
    return(1);

  return(0);
}

#ifdef PFM06A
inline static void counter_event_copy(const pfm_event_config_t *a, pfm_event_config_t *b, int cntr)
#else
inline static void counter_event_copy(const pfmlib_param_t *a, pfmlib_param_t *b, int cntr)
#endif
{
  DBG((stderr,"%d\n",cntr));
#ifdef PFM06A
  b->pec_evt[cntr] = a->pec_evt[cntr];
  b->pec_count++;
#else
  b->pfp_evt[cntr] = a->pfp_evt[cntr];
  b->pfp_count++;
#endif
}

inline static int update_global_hwcounters(EventSetInfo *local, EventSetInfo *global)
{
  hwd_control_state_t *machdep = global->machdep;
  int i, selector = 0, hwcntr;
#ifdef PFM06A
  perfmon_reg_t flop_hack;
  perfmon_req_t readem[PMU_MAX_COUNTERS], writeem[PMU_MAX_COUNTERS];
  memset(writeem,0x0,sizeof(perfmon_req_t)*PMU_MAX_COUNTERS);
#else
#ifdef ITANIUM2
  pfm_ita2_reg_t flop_hack;
#else
  pfm_ita_reg_t flop_hack;
#endif
  pfarg_reg_t readem[PMU_MAX_COUNTERS], writeem[PMU_MAX_COUNTERS];
  memset(writeem,0x0,sizeof(pfarg_reg_t)*PMU_MAX_COUNTERS);
#endif


  for(i=0; i < PMU_MAX_COUNTERS; i++)
    {
      /* Bug fix, we must read the counters out in the same order we programmed them. */
      /* pfm_dispatch_events may request registers out of order. */

#ifdef PFM06A
      readem[i].pfr_reg.reg_num = machdep->pc[i].pfr_reg.reg_num;

      /* Writing doesn't matter, we're just zeroing the counter. */ 

      writeem[i].pfr_reg.reg_num = PMU_MAX_COUNTERS+i;
#else
      readem[i].reg_num = machdep->pc[i].reg_num;

      /* Writing doesn't matter, we're just zeroing the counter. */ 

      writeem[i].reg_num = PMU_MAX_COUNTERS+i;
#endif
    }

#ifdef PFM06A
  if (perfmonctl(machdep->pid, PFM_READ_PMDS, 0, readem, PMU_MAX_COUNTERS) == -1) 
#else
#ifdef ITANIUM2
  if (perfmonctl(machdep->pid, PFM_READ_PMDS, readem, machdep->evt.pfp_count) == -1)
#else
  if (perfmonctl(machdep->pid, PFM_READ_PMDS, readem, PMU_MAX_COUNTERS) == -1)
#endif
#endif
    {
      DBG((stderr,"perfmonctl error READ_PMDS errno %d\n",errno));
      return PAPI_ESYS;
    }

  if ((local->state & PAPI_OVERFLOWING) && (_papi_system_info.supports_hw_overflow))

    {
      selector = local->EventInfoArray[local->overflow.EventIndex].selector;
      while ((i = ffs(selector)))
	{
	  hwcntr = 1 << (i-1);
	  if (hwcntr & selector)
	    {
	      DBG((stderr,"counter %d used in overflow, threshold %d\n",i-1-PMU_MAX_COUNTERS,local->overflow.threshold));
	      /* Correct value read from kernel */
#ifdef PFM06A
	      readem[i-1-PMU_MAX_COUNTERS].pfr_reg.reg_value += (unsigned long)local->overflow.threshold;
	      /* Ready the new structure */
	      writeem[i-1-PMU_MAX_COUNTERS].pfr_reg.reg_value = (~0UL) - (unsigned long)local->overflow.threshold;
	      writeem[i-1-PMU_MAX_COUNTERS].pfr_reg.reg_smpl_reset = (~0UL) - (unsigned long)local->overflow.threshold;
#else
	      readem[i-1-PMU_MAX_COUNTERS].reg_value += (unsigned long)local->overflow.threshold;
	      /* Ready the new structure */
	      writeem[i-1-PMU_MAX_COUNTERS].reg_value = (~0UL) - (unsigned long)local->overflow.threshold;
	      writeem[i-1-PMU_MAX_COUNTERS].reg_long_reset = (~0UL) - (unsigned long)local->overflow.threshold;
#endif
	    }
	  selector ^= 1 << (i-1); 
	}
    }

  /* We need to scale FP_OPS_HI, dammit */ 

  for(i=0; i < PMU_MAX_COUNTERS; i++)
    {
#ifdef PFM06A
      flop_hack.pmu_reg = machdep->pc[i].pfr_reg.reg_value;
      if (flop_hack.pmc_es == 0xa)
	readem[i].pfr_reg.reg_value = readem[i].pfr_reg.reg_value * 4;
#else
      flop_hack.reg_val = machdep->pc[i].reg_value;
      if (flop_hack.pmc_es == 0xa)
	readem[i].reg_value = readem[i].reg_value * 4;
#endif
    }

  for(i=0; i < PMU_MAX_COUNTERS; i++)
    {
#ifdef PFM06A
      DBG((stderr,"update_global_hwcounters() %d: G%ld = G%lld + C%ld\n",i+4,
	   global->hw_start[i]+readem[i].pfr_reg.reg_value,
	   global->hw_start[i],readem[i].pfr_reg.reg_value));
      global->hw_start[i] = global->hw_start[i] + readem[i].pfr_reg.reg_value;
#else
      DBG((stderr,"update_global_hwcounters() %d: G%ld = G%lld + C%ld\n",i+4,
	   global->hw_start[i]+readem[i].reg_value,
	   global->hw_start[i],readem[i].reg_value));
      global->hw_start[i] = global->hw_start[i] + readem[i].reg_value;
#endif
    }

#ifdef PFM06A
  if (perfmonctl(machdep->pid, PFM_WRITE_PMDS, 0, writeem, PMU_MAX_COUNTERS) == -1) {
#else
  if (perfmonctl(machdep->pid, PFM_WRITE_PMDS, writeem, PMU_MAX_COUNTERS) == -1) {
#endif
    fprintf(stderr, "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
    return PAPI_ESYS;
  }

  return(PAPI_OK);
}

inline static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

inline static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  switch (domain)
    {
    case PAPI_GRN_THR:
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

inline static int set_default_domain(EventSetInfo *zero, int domain)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain));
}

inline static int set_default_granularity(EventSetInfo *zero, int granularity)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_granularity(current_state,granularity));
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

// struct perfctr_dev *dev;
int _papi_hwd_init_global(void)
{
  int retval;
  pfmlib_options_t pfmlib_options;
#ifdef DEBUG
  extern int papi_debug;
#endif

  /* Opened once for all threads. */

#ifndef PFM06A
  if (pfm_initialize() != PFMLIB_SUCCESS ) 
    return(PAPI_ESYS);
#endif

  memset(&pfmlib_options, 0, sizeof(pfmlib_options));
#ifdef DEBUG
  if (papi_debug)
    pfmlib_options.pfm_debug = 1;
#endif
#ifdef PFM06A
  if (pfmlib_config(&pfmlib_options))
#else
  if (pfm_set_options(&pfmlib_options))
#endif
    return(PAPI_ESYS);

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

int _papi_hwd_shutdown_global(void)
{
  /* Need to pass in pid for _papi_hwd_shutdown_globabl in the future -KSL */
#ifdef PFM06A
  perfmonctl(getpid(), PFM_DISABLE, 0, NULL, 0);
#else
  perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0);
#endif
  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
#ifdef PFM06A
  perfmon_req_t ctx[1];
#else
  pfarg_context_t ctx[1];
#endif
  hwd_control_state_t *machdep = zero->machdep;
  
  memset(ctx, 0, sizeof(ctx));
#ifdef PFM06A
  ctx[0].pfr_ctx.notify_pid = getpid();
  ctx[0].pfr_ctx.notify_sig = SIGPROF;
  ctx[0].pfr_ctx.flags      = PFM_FL_INHERIT_NONE; 
#else
  ctx[0].ctx_notify_pid = getpid();
  ctx[0].ctx_flags      = PFM_FL_INHERIT_NONE; 
#endif

#ifdef PFM06A
  if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, 0 , ctx, 1) == -1 ) {
#else
  if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
#endif
    fprintf(stderr,"PID %d: perfmonctl error PFM_CREATE_CONTENT %d\n", getpid(), errno);
  }

  /* 
   * reset PMU (guarantee not active on return) and unfreeze
   * must be done before writing to any PMC/PMD
   */ 

#ifdef PFM06A
  if (perfmonctl(getpid(), PFM_ENABLE, 0, 0, 0) == -1) {
#else
  if (perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
#endif
    if (errno == ENOSYS) 
      fprintf(stderr,"Your kernel does not have performance monitoring support !\n");
    fprintf(stderr,"PID %d: perfmonctl error PFM_ENABLE %d\n",getpid(),errno);
  }

  /* Initialize our global machdep. */

  init_config(machdep);

  return(PAPI_OK);
}

long long _papi_hwd_get_real_usec (void)
{
  long long cyc;

  cyc = get_cycles()*(unsigned long long)1000;
  cyc = cyc / (long long)_papi_system_info.hw_info.mhz;
  return(cyc / (long long)1000);
}

long long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
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
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int nselector = 0;
  int retval = 0;
  int selector = 0;
#ifdef PFM06A
  pfm_event_config_t tmp_cmd, *codes;
#else
  pfmlib_param_t tmp_cmd, *codes;
#endif

  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode ^ PRESET_MASK; 

      if (!preset_map[preset_index].present)
	return(PAPI_ENOEVNT);

      derived = preset_map[preset_index].derived;

      /* Get the codes used for this event */

      codes = &preset_map[preset_index].evt;
      out->command = derived;
      out->operand_index 
	= preset_map[preset_index].operand_index;
    }
  else
    {
      unsigned long hwcntr_num;
      int ev;
#ifdef PFM06A
      pme_entry_code_t tmp;
      extern int pfm_findeventbyvcode(int code);
#else
      pme_ita_code_t tmp;
      extern int pfm_find_event_byvcode(int code, int *idx);
#endif

      tmp.pme_vcode = 0;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if ((hwcntr_num < PMU_FIRST_COUNTER) ||
	  (hwcntr_num >= PMU_MAX_COUNTERS+PMU_FIRST_COUNTER))
	return(PAPI_EINVAL);
      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

#ifdef PFM06A
      tmp.pme_codes.pme_mcode = (EventCode >> 8) & 0xff; /* bits 8 through 15 */
      tmp.pme_codes.pme_ear = (EventCode >> 16) & 0x1; 
      tmp.pme_codes.pme_dear = (EventCode >> 17) & 0x1; 
      tmp.pme_codes.pme_tlb = (EventCode >> 18) & 0x1; 
      tmp.pme_codes.pme_umask = (EventCode >> 19) & 0x1fff; 
      ev = pfm_findeventbyvcode(tmp.pme_vcode);
      if (ev == -1)
	return(PAPI_EINVAL);
      tmp_cmd.pec_count = 1;
      tmp_cmd.pec_evt[0] = ev;
#else
      tmp.pme_ita_code.pme_code = (EventCode >> 8) & 0xff; /* bits 8 through 15 */
      tmp.pme_ita_code.pme_ear = (EventCode >> 16) & 0x1; 
      tmp.pme_ita_code.pme_dear = (EventCode >> 17) & 0x1; 
      tmp.pme_ita_code.pme_tlb = (EventCode >> 18) & 0x1; 
      tmp.pme_ita_code.pme_umask = (EventCode >> 19) & 0x1fff; 
      ev = pfm_find_event_byvcode(tmp.pme_vcode, &(tmp_cmd.pfp_evt[0]));
      if (ev != PFMLIB_SUCCESS )
	return(PAPI_EINVAL);
      tmp_cmd.pfp_count = 1;
#endif
      codes = &tmp_cmd;
    }

  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the control codes and get the new bits required */

  nselector = set_hwcntr_codes(this_state,codes);
  if (nselector < 0)
    return retval;
  if (nselector == 0)
    abort();

  /* Only the new fields */

  selector = this_state->selector ^ nselector;
  DBG((stderr,"This new event has selector 0x%x of 0x%x\n",selector,nselector));

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  out->code = EventCode;
  out->selector = selector;

  /* Update the new counter select field */

  this_state->selector = nselector;
  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int used,preset_index,i,j;

  /* Find out which counters used. */
  
  used = in->selector;

  /* Clear out counters that are part of this event. */

  this_state->selector = this_state->selector ^ used;
  /* We need to remove the count from this event, do we need to
   * reset the index of values too? -KSL 
   * Apparently so. -KSL */
  preset_index = in->code ^ PRESET_MASK;
  for(i=0;i<PMU_MAX_COUNTERS;i++){
#ifdef PFM06A
    if ( this_state->evt.pec_evt[i] & used ) {
       for ( j=i;j<(PMU_MAX_COUNTERS-1);j++ )
           this_state->evt.pec_evt[j] = this_state->evt.pec_evt[j+1];
#else
    if ( this_state->evt.pfp_evt[i] & used ) {
       for ( j=i;j<(PMU_MAX_COUNTERS-1);j++ )
           this_state->evt.pfp_evt[j] = this_state->evt.pfp_evt[j+1];
#endif
    }
  } 
#ifdef PFM06A
  this_state->evt.pec_count-=preset_map[preset_index].evt.pec_count;
#else
  this_state->evt.pfp_count-=preset_map[preset_index].evt.pfp_count;
#endif

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
#ifdef PFM06A
  perfmon_req_t pd[PMU_MAX_COUNTERS];
#else
  pfarg_reg_t pd[PMU_MAX_COUNTERS];
#endif

  /* If we ARE NOT nested, 
     just copy the global counter structure to the current eventset */

  if (current_state->selector == 0x0)
    {
      int selector, hwcntr;

      pfm_stop();

      current_state->selector = this_state->selector;
#ifdef PFM06A
      memcpy(&current_state->evt,&this_state->evt,sizeof(pfm_event_config_t));
      memcpy(current_state->pc,this_state->pc,sizeof(perfmon_req_t)*PMU_MAX_COUNTERS);
#else
      memcpy(&current_state->evt,&this_state->evt,sizeof(pfmlib_param_t));
      memcpy(current_state->pc,this_state->pc,sizeof(pfarg_reg_t)*PMU_MAX_COUNTERS);
#endif

    restart_pm_hardware:

#ifdef PFM06A
      if (perfmonctl(current_state->pid, PFM_WRITE_PMCS, 0, current_state->pc, current_state->evt.pec_count) == -1) {
#else
      if (perfmonctl(current_state->pid, PFM_WRITE_PMCS, current_state->pc, current_state->evt.pfp_count) == -1) {
#endif
	fprintf(stderr,"child: perfmonctl error WRITE_PMCS errno %d\n",errno); pfm_start(); return(PAPI_ESYS);
      }

#ifdef PFM06A
      memset(pd, 0, sizeof(perfmon_req_t)*PMU_MAX_COUNTERS);
#else
      memset(pd, 0, sizeof(pfarg_context_t)*PMU_MAX_COUNTERS);
#endif
      for(i=0; i < PMU_MAX_COUNTERS; i++) 
	{
#ifdef PFM06A
	  pd[i].pfr_reg.reg_num = PMU_MAX_COUNTERS+i;
#else
	  pd[i].reg_num = PMU_MAX_COUNTERS+i;
#endif
	}

      if ((ESI->state & PAPI_OVERFLOWING) && (_papi_system_info.supports_hw_overflow))
	{
	  selector = ESI->EventInfoArray[ESI->overflow.EventIndex].selector;
	  while ((i = ffs(selector)))
	    {
	      hwcntr = 1 << (i-1);
	      if (hwcntr & selector)
		{
		  DBG((stderr,"counter %d used in overflow, threshold %d\n",i-1-PMU_MAX_COUNTERS,ESI->overflow.threshold));
#ifdef PFM06A
		  pd[i-1-PMU_MAX_COUNTERS].pfr_reg.reg_value = (~0UL) - (unsigned long)ESI->overflow.threshold;
		  pd[i-1-PMU_MAX_COUNTERS].pfr_reg.reg_smpl_reset = (~0UL) - (unsigned long)ESI->overflow.threshold;
#else
		  pd[i-1-PMU_MAX_COUNTERS].reg_value = (~0UL) - (unsigned long)ESI->overflow.threshold;
		  pd[i-1-PMU_MAX_COUNTERS].reg_long_reset = (~0UL) - (unsigned long)ESI->overflow.threshold;
#endif
		}
	      selector ^= 1 << (i-1); 
	    }
	}
      
#ifdef PFM06A
      if (perfmonctl(current_state->pid, PFM_WRITE_PMDS, 0, pd, PMU_MAX_COUNTERS) == -1) {
#else
      if (perfmonctl(current_state->pid, PFM_WRITE_PMDS, pd, current_state->evt.pfp_count) == -1) {
#endif
	fprintf(stderr,"child: perfmonctl error WRITE_PMDS errno %d\n",errno); pfm_start(); return(PAPI_ESYS);
      }
      
      pfm_start();
      
      return(PAPI_OK);
    }

  /* If we ARE nested, 

     For all shared events:

     1) move selector bits in old ESI->EventInfoArray[].selector
     2) move operand index in old ESI->EventInfoArray[].operand_index
     3) move reg num in old machdep->pc kernel structure
     4) move selecror in old machdep->selector structure
     5) copy shared start values from master to old

     For all unshared events whether or not they share specific register:

     1) add to end of machdep->evt list in current
     2) move selector bits in old ESI->EventInfoArray[].selector
     3) move operand index in old ESI->EventInfoArray[].operand_index
     4) move reg num in old machdep->pc kernel structure
     5) move selecror in old machdep->selector structure
     6) start values = 0 to old */
  
  else
    {
      /* Stop the current context */

      if (ESI->state & PAPI_OVERFLOWING)
	{
	  fprintf(stderr,"Simultaneous event sets not supported when overflowing.\n");
	  return(PAPI_ESBSTR);
	}

      pfm_stop();

      /* Update the global values */

      retval = update_global_hwcounters(ESI,zero);
      if (retval)
	{
	  pfm_start();
	  return(retval);
	}

      {
	int j, nselector = 0, selector = 0, shared = 0;
	int tmp, hwcntr, done = 0, not_shared = 0;
	int index_in_current[PMU_MAX_COUNTERS] = { -1, };
	int index_in_this[PMU_MAX_COUNTERS] = { -1, };
	int index_in_this_not_shared[PMU_MAX_COUNTERS] = { -1, };
	int new_selector_in_this[PMU_MAX_COUNTERS] = { 0, };
	int new_selector_in_this_not_shared[PMU_MAX_COUNTERS] = { 0, };
	int new_operand_in_this[PMU_MAX_COUNTERS] = { 0, };
	int new_operand_in_this_not_shared[PMU_MAX_COUNTERS] = { 0, };

	/* How many and where are the shared and unshared events */

#ifdef PFM06A
	for (j=0;j<this_state->evt.pec_count;j++)
	  {
	    for (i=0;i<current_state->evt.pec_count;i++)
	      {
		if (this_state->evt.pec_evt[j] == current_state->evt.pec_evt[i])
#else
	for (j=0;j<this_state->evt.pfp_count;j++)
	  {
	    for (i=0;i<current_state->evt.pfp_count;i++)
	      {
		if (this_state->evt.pfp_evt[j] == current_state->evt.pfp_evt[i])
#endif
		  {
		    index_in_current[shared] = i+PMU_FIRST_COUNTER;
		    index_in_this[shared] = j+PMU_FIRST_COUNTER;
		    shared++;
		    done = 0;
		    break;
		  }
		else
		  done = 1;
	      }
	    if (done)
	      {
		index_in_this_not_shared[not_shared] = j+PMU_FIRST_COUNTER;
		not_shared++;
	      }
	  }

	/* Turn off the selector bits for the unshared events */
	/* Gather up the data for later */

	if (not_shared)
	  {
	    for (j=0;j<not_shared;j++)
	      {
		for (i=0;i<ESI->NumberOfEvents;i++)
		  {
		    if (ESI->EventInfoArray[i].selector & (1 << index_in_this_not_shared[j]))
		      {
			ESI->EventInfoArray[i].selector ^= 1 << index_in_this_not_shared[j];
#ifdef PFM06A
			new_selector_in_this_not_shared[i] |= 1 << (current_state->evt.pec_count+j+PMU_FIRST_COUNTER);
			new_operand_in_this_not_shared[i] = current_state->evt.pec_count+j+PMU_FIRST_COUNTER;
#else
			new_selector_in_this_not_shared[i] |= 1 << (current_state->evt.pfp_count+j+PMU_FIRST_COUNTER);
			new_operand_in_this_not_shared[i] = current_state->evt.pfp_count+j+PMU_FIRST_COUNTER;
#endif
		      }
		  }
	      }
	  }

	/* Turn off the selector bits for the shared events */
	/* Gather up the data for later */

	if (shared)
	  {
	    /* Turn off the old bits and gather the new ones */
	    for (j=0;j<shared;j++)
	      {
		for (i=0;i<ESI->NumberOfEvents;i++)
		  {
		    if (ESI->EventInfoArray[i].selector & (1 << index_in_this[j]))
		      {
			ESI->EventInfoArray[i].selector ^= (1 << index_in_this[j]);
			new_selector_in_this[i] |= (1 << index_in_current[j]);
			new_operand_in_this[i] = index_in_current[j];
		      }
		  }
	      }
	  }

	/* Turn on the new bits in the modified structure to be merged with */
      
	for (i=0;i<ESI->NumberOfEvents;i++)
	  {
	    DBG((stderr,"new_selector_in_this[%d] = 0x%x, new_selector_in_this_not_shared[%d] = 0x%x\n",i,new_selector_in_this[i],i,new_selector_in_this_not_shared[i]));
	    ESI->EventInfoArray[i].selector = new_selector_in_this[i] | new_selector_in_this_not_shared[i];
	    DBG((stderr,"new_operand_in_this[%d] = %d, new_operand_in_this_not_shared[%d] = %d\n",i,new_operand_in_this[i],i,new_operand_in_this_not_shared[i]));
	    ESI->EventInfoArray[i].operand_index = new_operand_in_this[i] | new_operand_in_this_not_shared[i];
	  }
	for (i=0;i<shared;i++)
	  {
	    if (index_in_current[i])
	      {
#ifdef PFM06A
		this_state->pc[index_in_this[i]-PMU_FIRST_COUNTER].pfr_reg.reg_num = index_in_current[i];
#else
		this_state->pc[index_in_this[i]-PMU_FIRST_COUNTER].reg_num = index_in_current[i];
#endif
		this_state->selector ^= 1 << index_in_this[i];
	      }
	  }
	for (i=0;i<not_shared;i++)
	  {
	    if (index_in_this_not_shared[i])
	      {
#ifdef PFM06A
		this_state->pc[index_in_this_not_shared[i]-PMU_FIRST_COUNTER].pfr_reg.reg_num = current_state->evt.pec_count+i+PMU_FIRST_COUNTER;
#else
		this_state->pc[index_in_this_not_shared[i]-PMU_FIRST_COUNTER].reg_num = current_state->evt.pfp_count+i+PMU_FIRST_COUNTER;
#endif
		this_state->selector ^= 1 << index_in_this_not_shared[i];
	      }
	  }
	for (i=0;i<shared;i++)
	  this_state->selector |= 1 << index_in_current[i];
	for (i=0;i<not_shared;i++)
	  {
#ifdef PFM06A
	    this_state->selector |= 1 << (current_state->evt.pec_count+i+PMU_FIRST_COUNTER);
	    /* Add the not shared events to the end of the list of the current structure running */
	    current_state->evt.pec_evt[current_state->evt.pec_count+i] = this_state->evt.pec_evt[index_in_this_not_shared[i]-PMU_FIRST_COUNTER];
	  }
	current_state->evt.pec_count += not_shared;
#else
	    this_state->selector |= 1 << (current_state->evt.pfp_count+i+PMU_FIRST_COUNTER);
	    /* Add the not shared events to the end of the list of the current structure running */
	    current_state->evt.pfp_evt[current_state->evt.pfp_count+i] = this_state->evt.pfp_evt[index_in_this_not_shared[i]-PMU_FIRST_COUNTER];
	  }
	current_state->evt.pfp_count += not_shared;
#endif

	/* Re-encode the command structure, return the new selector */
	nselector = set_hwcntr_codes(current_state,NULL);
	if (nselector < 0)
	  {
	    pfm_start();
	    return retval;
	  }
	if (nselector == 0)
	  abort();

	/* Get the new fields */
	selector = current_state->selector ^ nselector;
	DBG((stderr,"The new merged structure added selector 0x%x to 0x%x\n",selector,nselector));

	/* Now everything is good, so actually do the merge */

	/* First for shared counters */

	tmp = current_state->selector & this_state->selector;
	while ((i = ffs(tmp)))
	  {
	    hwcntr = 1 << (i-1);
	    if (hwcntr & tmp)
	      {
		DBG((stderr,"counter %d used in both\n",i-1-PMU_MAX_COUNTERS));
		ESI->hw_start[i-1-PMU_MAX_COUNTERS] = zero->hw_start[i-1-PMU_MAX_COUNTERS];
		zero->multistart.SharedDepth[i-1-PMU_MAX_COUNTERS]++; 
	      }
	    tmp ^= 1 << (i-1); 
	  }

	/* Now for unshared counters */

	tmp = nselector & selector;
	while ((i = ffs(tmp)))
	  {
	    hwcntr = 1 << (i-1);
	    if (hwcntr & tmp)
	      {
		DBG((stderr,"counter %d added\n",i-1-PMU_MAX_COUNTERS));
		current_state->selector |= hwcntr;
		ESI->hw_start[i-1-PMU_MAX_COUNTERS] = 0;
		zero->hw_start[i-1-PMU_MAX_COUNTERS] = 0;
	      }
	    tmp ^= 1 << (i-1); 
	  }
      }
      pfm_start();
    }

  goto restart_pm_hardware;
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, tmp;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  /* Check for events that are NOT shared between eventsets and 
     therefore require modification to the selection mask. */

  if ((zero->multistart.num_runners - 1) == 0)
    {
      current_state->selector = 0;
      return(PAPI_OK);
    }
  else
    {
      tmp = this_state->selector;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1-PMU_MAX_COUNTERS);
	  if (zero->multistart.SharedDepth[i-1-PMU_MAX_COUNTERS] - 1 < 0)
	    current_state->selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i-1-PMU_MAX_COUNTERS]--;
	  tmp ^= 1 << (i-1); 
	}
      return(PAPI_OK);
    }
}

int _papi_hwd_reset(EventSetInfo *ESI, EventSetInfo *zero)
{
  int i, retval;
  
  retval = update_global_hwcounters(ESI,zero);
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
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, subtracting %lld to %lld\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  float tmp;

  tmp = (float)units * _papi_system_info.hw_info.mhz * 1000000.0;
  tmp = tmp / (float) cycles;
  return((long long)tmp);
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
  int selector = cmd->selector >> 4;

  switch (cmd->command)
    {
    case DERIVED_ADD: 
      return(handle_derived_add(selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(cmd->operand_index, selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(cmd->operand_index, selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(cmd->operand_index, selector, from));
    default:
      abort();
    }
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long events[])
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[PMU_MAX_COUNTERS];

  retval = update_global_hwcounters(ESI,zero);
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
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector >> 4) - 1;
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
  // hwd_control_state_t *machdep = zero->machdep;
  // vperfctr_unlink(machdep->self);
  return(PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{ 
  if (preset_map[preset_index].present == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
}

/* This function only used when hardware interrupts ARE NOT working */

void _papi_hwd_dispatch_timer(int signal, siginfo_t* info, void * tmp)
{
  struct ucontext *uc;
  struct sigcontext *mc;
  struct ucontext realc;

  pfm_stop();
  uc = (struct ucontext *) tmp;
  realc = *uc;
  mc = &uc->uc_mcontext;
  DBG((stderr,"Start at 0x%lx\n",mc->sc_ip));
  _papi_hwi_dispatch_overflow_signal((void *)mc); 
  DBG((stderr,"Finished at 0x%lx\n",mc->sc_ip));
  pfm_start();
}

/* This function only used when hardware interrupts ARE working */

#ifdef PFM06A
static void ia64_dispatch_sigprof(int n, struct mysiginfo *info, struct sigcontext *context)
#else
static void ia64_dispatch_sigprof(int n, pfm_siginfo_t *info, struct sigcontext *context)
#endif
{
  pfm_stop();
#ifdef PFM06A
  DBG((stderr,"pid=%d @0x%lx bv=0x%lx\n", info->sy_pid, context->sc_ip, info->sy_pfm_ovfl));
#else
  DBG((stderr,"pid=%d @0x%lx bv=0x%lx\n", info->sy_pid, context->sc_ip, info->sy_pfm_ovfl[0]));
  if (info->sy_code != PROF_OVFL) {
    fprintf(stderr,"PAPI: received spurious SIGPROF si_code=%d\n", info->sy_code);
    return;
  } 
#endif
  _papi_hwi_dispatch_overflow_signal((void *)context); 
#ifdef PFM06A
  perfmonctl(info->sy_pid, PFM_RESTART, 0, 0, 0);
#else
  perfmonctl(info->sy_pid, PFM_RESTART, 0, 0);
#endif
  pfm_start();
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
#ifdef PFM06A
  perfmon_req_t *pc = this_state->pc;
#else
  pfarg_reg_t *pc = this_state->pc;
#endif
  int i, selector, hwcntr, retval = PAPI_OK;

  if (overflow_option->threshold == 0)
    {
      /* Remove the overflow notifier on the proper event. Remember that selector
         contains the actual hardware register, not the index in the command structure. */

      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      while ((hwcntr = ffs(selector)))
	{
	  hwcntr = hwcntr - 1;
	  for (i=0;i<PMU_MAX_COUNTERS;i++)
	    {
#ifdef PFM06A
	      if (pc[i].pfr_reg.reg_num == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d, flags %d\n",hwcntr,i,pc[i].pfr_reg.reg_flags));
		  pc[i].pfr_reg.reg_flags = 0;
#else
	      if (pc[i].reg_num == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d, flags %d\n",hwcntr,i,pc[i].reg_flags));
		  pc[i].reg_flags = 0;
#endif
		  break;
		}
	    }
	  selector ^= 1 << hwcntr;
	}

      /* Remove the signal handler */

      PAPI_lock();
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0)
	{
	  if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	    retval = PAPI_ESYS;
	}
      PAPI_unlock();
    }
  else
    {
      struct sigaction act;
      void *tmp;

      tmp = (void *)signal(SIGPROF, SIG_IGN);
      if ((tmp != (void *)SIG_DFL) && (tmp != (void *)ia64_dispatch_sigprof))
	return(PAPI_EMISC);

      /* Set up the signal handler */

      memset(&act,0x0,sizeof(struct sigaction));
      act.sa_handler = (sig_t)ia64_dispatch_sigprof;
      act.sa_flags = SA_SIGINFO;
      if (sigaction(SIGPROF, &act, NULL) == -1)
	return(PAPI_ESYS);

      /* Set up the overflow notifier on the proper event. Remember that selector
         contains the actual hardware register, not the index in the command structure. */

      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      while ((hwcntr = ffs(selector)))
	{
	  hwcntr = hwcntr - 1;
	  for (i=0;i<PMU_MAX_COUNTERS;i++)
	    {
#ifdef PFM06A
	      if (pc[i].pfr_reg.reg_num == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d\n",hwcntr,i));
		  pc[i].pfr_reg.reg_flags = PFM_REGFL_OVFL_NOTIFY;
		  break;
		}
#else
	      if (pc[i].reg_num == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d\n",hwcntr,i));
		  pc[i].reg_flags = PFM_REGFL_OVFL_NOTIFY;
		  break;
		}
#endif
	    }
	  selector ^= 1 << hwcntr;
	}

      PAPI_lock();
      _papi_hwi_using_signal++;
      PAPI_unlock();
    }
  return(retval);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_ip;

  return(location);
}

#define __SMP__
#define CONFIG_SMP
#include <asm/atomic.h>
static atomic_t lock;

void _papi_hwd_lock_init(void)
{
  atomic_set(&lock,1);
}

void _papi_hwd_lock(void)
{
  if (atomic_dec_and_test(&lock))
    return;
  else
    {
#ifdef DEBUG
      volatile int waitcyc = 0;
#endif
      while (atomic_dec_and_test(&lock))
	{
	  DBG((stderr,"Waiting..."));
#ifdef DEBUG
	  waitcyc++;
#endif
	  atomic_inc(&lock);
	}
    }
}

void _papi_hwd_unlock(void)
{
  atomic_set(&lock, 1);
}

/* Machine info structure. -1 is unused. */

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
				0.0  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)&_init,
				 (caddr_t)&_etext,
				 (caddr_t)&_etext+1,
				 (caddr_t)&_edata,
				 (caddr_t)&__bss_start,
				 (caddr_t)NULL,
				 "LD_PRELOAD" /* How to preload libs */
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
			        1,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        1,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, } };

