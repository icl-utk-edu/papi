/*
* File:    linux-ia64.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:	   Kevin London
*	   london@cs.utk.edu
*          Per Ekman
*          pek@pdc.kth.se
*          Zhou Min
*          min@cs.utk.edu
*/

#include SUBSTRATE

#include "pfmwrap.h"

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
  {PAPI_CA_SNP,0,{"BUS_SNOOPS_SELF",0,0,0}},
  {PAPI_CA_INV,DERIVED_ADD,{"BUS_MEM_READ_BRIL_SELF","BUS_MEM_READ_BIL_SELF",0,0}},
  {PAPI_TLB_TL,DERIVED_ADD,{"ITLB_MISSES_FETCH_L2ITLB","L2DTLB_MISSES",0,0}},
  {PAPI_STL_ICY,0,{"DISP_STALLED",0,0,0}},
  {PAPI_STL_CCY,0,{"BACK_END_BUBBLE_ALL",0,0,0}},
  {PAPI_TOT_IIS,0,{"INST_DISPERSED",0,0,0}},
  {PAPI_RES_STL,0,{"BE_EXE_BUBBLE_ALL",0,0,0}},
  {PAPI_FP_STAL,0,{"BE_EXE_BUBBLE_FRALL",0,0,0}},
  {PAPI_L2_TCR,DERIVED_ADD,{"L2_DATA_REFERENCES_L2_DATA_READS","L2_INST_DEMAND_READS","L2_INST_PREFETCHES",0}},
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
  {PAPI_BR_PRC,0,{"BR_MISPRED_DETAIL_ALL_CORRECT_PRED",0,0,0}},
  {PAPI_BR_MSP,DERIVED_ADD,{"BR_MISPRED_DETAIL_ALL_WRONG_PATH","BR_MISPRED_DETAIL_ALL_WRONG_TARGET",0,0}},
  {PAPI_TOT_CYC,0,{"CPU_CYCLES",0,0,0}},
  {PAPI_FP_INS,0,{"FP_OPS_RETIRED",0,0,0}},
  {PAPI_TOT_INS,DERIVED_ADD,{"IA64_INST_RETIRED","IA32_INST_RETIRED",0,0}},
  {PAPI_LD_INS,0,{"LOADS_RETIRED",0,0,0}},
  {PAPI_SR_INS,0,{"STORES_RETIRED",0,0,0}},
  {PAPI_FLOPS,DERIVED_PS,{"CPU_CYCLES","FP_OPS_RETIRED",0,0}},
  /* First byte selects type (M, I, F, B), bits 3-30 set to 1 to mask the whole opcode,
   * bits 1 (ig_ad) and 2 (mandatory 1) are set */
  {PAPI_INT_INS,0,{"400000003FFFFFFF@IA64_TAGGED_INST_RETIRED_IBRP0_PMC8",0,0,0}},
  {PAPI_FSQ_INS,0,{"2890000001BFFFFF@IA64_TAGGED_INST_RETIRED_IBRP0_PMC8",0,0,0}},
  {0,0,{0,0,0,0}}};
#endif
#endif
static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS];

#ifdef ITANIUM2
static pfmlib_ita2_param_t ita2_param[PAPI_MAX_PRESET_EVENTS];
#endif

pfmw_ita_param_t ear_ita_param;
static void *smpl_vaddr;
extern void dispatch_profile(EventSetInfo *ESI, void *context,
                 long_long over, long_long threshold);

int set_dear_ita_param(int EventCode)
{
#ifdef ITANIUM2
  ear_ita_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
  ear_ita_param.pfp_ita2_dear.ear_used   = 1;
  ear_ita_param.pfp_ita2_dear.ear_mode = 0;
  ear_ita_param.pfp_ita2_dear.ear_plm = PFM_PLM3;
  ear_ita_param.pfp_ita2_dear.ear_ism = PFMLIB_ITA2_ISM_IA64; /* ia64 only */
  ear_ita_param.pfp_ita2_dear.ear_umask =  (EventCode >> 19) & 0x1fff;
#else 
  ear_ita_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
  ear_ita_param.pfp_ita_dear.ear_used   = 1;
  ear_ita_param.pfp_ita_dear.ear_is_tlb = 0;
  ear_ita_param.pfp_ita_dear.ear_plm = PFM_PLM3;
  ear_ita_param.pfp_ita_dear.ear_ism = PFMLIB_ITA_ISM_IA64; /* ia64 only */
  ear_ita_param.pfp_ita_dear.ear_umask =  (EventCode >> 19) & 0x1fff;
#endif
  return PAPI_OK;
}

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
#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h> 
	tmp = __getReg(_IA64_REG_AR_ITC);
 
#else /* GCC */
	/* XXX: need more to adjust for Itanium itc bug */
       __asm__ __volatile__("mov %0=ar.itc" : "=r"(tmp) :: "memory"); 
#endif
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

static inline int
gen_events(char **arg, pfmw_param_t *evt)
{
	int ev;
	int cnt=0;
        char *p;
#ifdef ITANIUM2 
        unsigned long mask = 0;
#endif

	if (arg == NULL) return -1;

	while (*arg) {
		p = *arg;
		if (cnt == PMU_MAX_COUNTERS) goto too_many;
#ifdef ITANIUM2 /* The following case was added by pek@pdc.kth.se. Don't
		   blame Stephane Eranian for it. */
		/* Hack to extract the mask for opcode matching */
		mask = strtol(*arg, &p, 16);
		if (p && p[0] == '@') {
			((pfmlib_ita2_param_t *)evt->pfp_model)->pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
			((pfmlib_ita2_param_t *)evt->pfp_model)->pfp_ita2_pmc8.opcm_used = 1;
			((pfmlib_ita2_param_t *)evt->pfp_model)->pfp_ita2_pmc8.pmc_val = mask;
			p++;
	    	}
		else {
			mask = 0;
			p = *arg;
			evt->pfp_model = NULL;
	    	}
#endif
		/* must match vcode only */
		if ((ev = pfmw_find_event(p,0,&(PFMW_PEVT_EVENT(evt, cnt)))) 
	    		!= PFMLIB_SUCCESS) goto error;
		cnt++;
		arg++;
	}
	PFMW_PEVT_EVTCOUNT(evt) = cnt;

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
      preset_index = preset_search_map[pnum].preset & PRESET_AND_MASK; 
#ifdef ITANIUM2
      preset_map[preset_index].evt.pfp_model = &ita2_param[pnum];
#endif
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
inline static int set_hwcntr_codes(hwd_control_state_t *this_state, const pfmw_param_t *from)
{
  pfmw_reg_t *pc = this_state->pc;
  pfmw_param_t *evt = &this_state->evt;
  int i, orig_cnt = PFMW_PEVT_EVTCOUNT(evt);  
  int cnt = PMU_MAX_PMCS;
  int selector = 0;

  if (from)
    {
      /* Called from add_event */
      /* Merge the two evt structures into the old one */
      
      for (i=0;i<PFMW_PEVT_EVTCOUNT(from);i++) {
	PFMW_PEVT_EVENT(evt,PFMW_PEVT_EVTCOUNT(evt)) = PFMW_PEVT_EVENT(from,i);
	PFMW_PEVT_EVTCOUNT(evt)++;
      }
      
      if ((PFMW_PEVT_EVTCOUNT(evt)) > PMU_MAX_COUNTERS)
	{
	bail:
	  PFMW_PEVT_EVTCOUNT(evt) = orig_cnt;
	  return(PAPI_ECNFLCT);
	}
#ifdef ITANIUM2
      evt->pfp_model = from->pfp_model;
#endif
    }


  /* Recalcuate the pfmw_param_t structure, may also signal conflict */
  if (pfmw_dispatch_events(evt,pc,&cnt))
    {
      goto bail;
      return(PAPI_ECNFLCT);
    }

   this_state->pc_count = cnt;
   for (i=0;i<PFMW_PEVT_EVTCOUNT(evt);i++)
    {
      selector |= 1 << PFMW_REG_REGNUM(pc[i]);

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

  PFMW_EVT_DFLPLM(this_state->evt) = mode;

  /* Bug fix in case we don't call pfmw_dispatch_events after this code */

  for (i=0;i<PMU_MAX_COUNTERS;i++)
    {
      if (PFMW_REG_REGNUM(this_state->pc[i]))
	{
	  pfmw_arch_reg_t value;
	  DBG((stderr,"slot %d, register %lud active, config value 0x%lx\n",
	       i,(unsigned long)PFMW_REG_REGNUM(this_state->pc[i]),PFMW_REG_REGVAL(this_state->pc[i])));

	  PFMW_ARCH_REG_REGVAL(value) = 
	    PFMW_REG_REGVAL(this_state->pc[i]);
	  PFMW_ARCH_REG_PMCPLM(value) = mode;
	  PFMW_REG_REGVAL(this_state->pc[i]) = 
	    PFMW_ARCH_REG_REGVAL(value);

	  DBG((stderr,"new config value 0x%lx\n",PFMW_REG_REGVAL(this_state->pc[i])));
	}
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
  s = search_cpu_info(f,"family",maxargs);
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

inline static int counter_event_shared(const pfmw_param_t *a, const pfmw_param_t *b, int cntr)
{
  DBG((stderr,"%d %x vs %x \n",cntr,PFMW_PEVT_EVENT(a,cntr),PFMW_PEVT_EVENT(b,cntr)));
  if (PFMW_PEVT_EVENT(a,cntr) == PFMW_PEVT_EVENT(b,cntr))
    return(1);

  return(0);
}

inline static int counter_event_compat
(const pfmw_param_t *a, const pfmw_param_t *b, int cntr)
{
  DBG((stderr,"%d %d vs. %d\n",cntr,PFMW_PEVT_PLM(a,cntr),PFMW_PEVT_PLM(b,cntr)));
  if (PFMW_PEVT_PLM(a,cntr) == PFMW_PEVT_PLM(b,cntr))
    return(1);

  return(0);
}

inline static void counter_event_copy(const pfmw_param_t *a, pfmw_param_t *b, int cntr)
{
  DBG((stderr,"%d\n",cntr));
  PFMW_PEVT_EVENT(b,cntr) = PFMW_PEVT_EVENT(a,cntr);
  PFMW_PEVT_EVTCOUNT(b)++;
}

inline static int update_global_hwcounters(EventSetInfo *local, EventSetInfo *global)
{
  hwd_control_state_t *machdep = global->machdep;
  int i, selector = 0, hwcntr;
  pfmw_arch_reg_t flop_hack;
  pfmw_reg_t readem[PMU_MAX_COUNTERS], writeem[PMU_MAX_COUNTERS];
  memset(writeem, 0x0, sizeof writeem);
  memset(readem, 0x0, sizeof readem);

  for(i=0; i < PMU_MAX_COUNTERS; i++)
    {
      /* Bug fix, we must read the counters out in the same order we programmed them. */
      /* pfmw_dispatch_events may request registers out of order. */

      PFMW_REG_REGNUM(readem[i]) = PFMW_REG_REGNUM(machdep->pc[i]);

      /* Writing doesn't matter, we're just zeroing the counter. */ 

      PFMW_REG_REGNUM(writeem[i]) = PMU_MAX_COUNTERS+i;

    }

  if (pfmw_perfmonctl(machdep->pid, PFM_READ_PMDS, readem, PFMW_EVT_EVTCOUNT(machdep->evt)) == -1) 
    {
      DBG((stderr,"perfmonctl error READ_PMDS errno %d\n",errno));
      return PAPI_ESYS;
    }


  if ((local->state & PAPI_PROFILING) && (_papi_system_info.supports_hw_profile))
    {
      selector = local->EventInfoArray[local->profile.EventIndex].selector;
      while ((i = ffs(selector)))
	{
	  hwcntr = 1 << (i-1);
	  if (hwcntr & selector)
	    {
	      /* Correct value read from kernel */
	      PFMW_REG_REGVAL(readem[i-1-PMU_MAX_COUNTERS]) += (unsigned long)local->profile.threshold ;
	      /* Ready the new structure */
	      PFMW_REG_REGVAL(writeem[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)local->profile.threshold;
	      PFMW_REG_SMPLLRST(writeem[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)local->profile.threshold;
	      PFMW_REG_SMPLSRST(writeem[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)local->profile.threshold;
	    }
	  selector ^= 1 << (i-1); 
	}
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
	      PFMW_REG_REGVAL(readem[i-1-PMU_MAX_COUNTERS]) += (unsigned long)local->overflow.threshold;
	      /* Ready the new structure */
	      PFMW_REG_REGVAL(writeem[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)local->overflow.threshold;
	      PFMW_REG_SMPLLRST(writeem[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)local->overflow.threshold;
	    }
	  selector ^= 1 << (i-1); 
	}
    }

  /* We need to scale FP_OPS_HI */ 

  for(i=0; i < PMU_MAX_COUNTERS; i++)
    {
      PFMW_ARCH_REG_REGVAL(flop_hack) = PFMW_REG_REGVAL(machdep->pc[i]);
      if (PFMW_ARCH_REG_PMCES(flop_hack) == 0xa)
	PFMW_REG_REGVAL(readem[i]) = PFMW_REG_REGVAL(readem[i]) * 4;
    }

/* this code is not right, just temporary solution --  min  */
  if ((local->state & PAPI_PROFILING) && (_papi_system_info.supports_hw_profile))
  {
   selector = local->EventInfoArray[local->profile.EventIndex].selector;
   while ((i = ffs(selector)))
    {
      hwcntr = 1 << (i-1);
      if (hwcntr & selector)
        {
          PFMW_REG_REGVAL(readem[i-1-PMU_MAX_COUNTERS]) += local->profile.threshold*local->profile.overflowcount ;
		  break;
        }
    }
	DBG((stderr,"profile overflowcount=%d\n", local->profile.overflowcount));
	 local->profile.overflowcount=0;
  }

  /* Store the results in register order (they are read out LSb first from
     the selector) */
  for(i=0; i < PMU_MAX_COUNTERS && PFMW_REG_REGNUM(readem[i]); i++)
    {
      int papireg;
      DBG((stderr,"update_global_hwcounters() %d: G%ld = G%lld + C%ld\n",i+4,
	   global->hw_start[i]+PFMW_REG_REGVAL(readem[i]),
	   global->hw_start[i],PFMW_REG_REGVAL(readem[i])));
      papireg = PFMW_REG_REGNUM(readem[i]) - 4;
      global->hw_start[papireg] += PFMW_REG_REGVAL(readem[i]);
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
  int retval, type;
  unsigned int version;
  pfmlib_options_t pfmlib_options;
#ifdef DEBUG
  extern int papi_debug;
#endif

  /* Opened once for all threads. */

  if (pfm_initialize() != PFMLIB_SUCCESS ) 
    return(PAPI_ESYS);

  if (pfm_get_pmu_type(&type) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);

#ifdef ITANIUM2
  if (type != PFMLIB_ITANIUM2_PMU)
    {
      fprintf(stderr,"Intel Itanium I is not supported by this substrate.\n");
      return(PAPI_ESBSTR);
    }
#else
  if (type != PFMLIB_ITANIUM_PMU)
    {
      fprintf(stderr,"Intel Itanium II is not supported by this substrate.\n");
      return(PAPI_ESBSTR);
    }
#endif

  if (pfm_get_version(&version) != PFMLIB_SUCCESS)
    return(PAPI_ESBSTR);

  if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION))
    {
      fprintf(stderr,"Version mismatch of libpfm: compiled %x vs. installed %x\n",PFM_VERSION_MAJOR(PFMLIB_VERSION),PFM_VERSION_MAJOR(version));
      return(PAPI_ESBSTR);
    }
  memset(&pfmlib_options, 0, sizeof(pfmlib_options));
#ifdef DEBUG
  if (papi_debug)
    pfmlib_options.pfm_debug = 1;
#endif

  if (pfmw_set_options(&pfmlib_options))
    return(PAPI_ESYS);

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  /* get_memory_info has a CPU model argument that is not used,
   * fakining it here with hw_info.model which is not set by this
   * substrate 
   */
  retval = get_memory_info(&_papi_system_info.mem_info,
			   _papi_system_info.hw_info.model);
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
  pfmw_perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0);

  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  pfmw_context_t ctx[1];
  hwd_control_state_t *machdep = zero->machdep;
  
  memset(ctx, 0, sizeof(ctx));

  PFMW_CTX_NOTIFYPID(ctx[0]) = getpid();
  PFMW_CTX_FLAGS(ctx[0])     = PFM_FL_INHERIT_NONE;
#ifdef PFM06A
  ctx[0].pfr_ctx.notify_sig = SIGPROF;
#endif

  if (pfmw_perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
    fprintf(stderr,"PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", getpid(), errno);
  }

  /* 
   * reset PMU (guarantee not active on return) and unfreeze
   * must be done before writing to any PMC/PMD
   */ 

  if (pfmw_perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
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

int _papi_recalc_selectors(hwd_control_state_t *this_state, EventInfo_t *eventinfo) {
    EventSetInfo *ESI = get_my_EventSetInfo(eventinfo);
    int i, j, k;
    unsigned int EventCode, preset_index;
    pfmlib_param_t *old_evt, *cur_evt = &this_state->evt;
    int selector;

    if (ESI == NULL)
        return(PAPI_ESBSTR);
    
    this_state->selector = 0;
    for (i = 0; i < ESI->NumberOfEvents; i++) {
        EventCode = ESI->EventInfoArray[i].code;
        selector = 0;
        if (EventCode & PRESET_MASK) {
            preset_index = EventCode & PRESET_AND_MASK; 
            old_evt = &preset_map[preset_index].evt;
            for (j = 0; j < PFMW_PEVT_EVTCOUNT(old_evt); j++) {
                for (k = 0; k < PFMW_PEVT_EVTCOUNT(cur_evt); k++) {
                    if (PFMW_PEVT_EVENT(old_evt,j) == PFMW_PEVT_EVENT(cur_evt,k))
                        selector |= (1 << PFMW_REG_REGNUM(this_state->pc[k]));
                }
            }
            ESI->EventInfoArray[i].selector = selector;
            this_state->selector |= selector;
        }
        else {
            /* SOL */    
        }
    }
    return 0;
}

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int nselector = 0;
  int retval = 0;
  int selector = 0;
  pfmw_param_t tmp_cmd, *codes;

  if (EventCode & PRESET_MASK)
    { 
      unsigned int preset_index;
      int derived;

      preset_index = EventCode & PRESET_AND_MASK; 

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
      pfmw_code_t tmp;
#ifdef PFM06A
      extern int pfm_findeventbyvcode(int code);
#else
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
      tmp_cmd.pec_evt[0] = ev;
#elif PFM20
	  return(PAPI_ESBSTR);
      memset(&tmp_cmd, 0, sizeof tmp_cmd);
      tmp.pme_ita_code.pme_code = (EventCode >> 8) & 0xff; /* bits 8 through 15 */
      tmp.pme_ita_code.pme_ear = (EventCode >> 16) & 0x1; 
      tmp.pme_ita_code.pme_dear = (EventCode >> 17) & 0x1; 
      tmp.pme_ita_code.pme_tlb = (EventCode >> 18) & 0x1; 
      tmp.pme_ita_code.pme_umask = (EventCode >> 19) & 0x1fff; 
/*
      ev = pfm_find_event_byvcode(tmp.pme_vcode, &(PFMW_EVT_EVENT(tmp_cmd, 0)));
*/
      ev = pfm_find_event_bycode(tmp.pme_ita_code.pme_code, &(PFMW_EVT_EVENT(tmp_cmd, 0)));
      if (ev != PFMLIB_SUCCESS )
	return(PAPI_EINVAL);
#else
      memset(&tmp_cmd, 0, sizeof tmp_cmd);
      tmp.pme_ita_code.pme_code = (EventCode >> 8) & 0xff; /* bits 8 through 15 */
      tmp.pme_ita_code.pme_ear = (EventCode >> 16) & 0x1; 
      tmp.pme_ita_code.pme_dear = (EventCode >> 17) & 0x1; 
      tmp.pme_ita_code.pme_tlb = (EventCode >> 18) & 0x1; 
      tmp.pme_ita_code.pme_umask = (EventCode >> 19) & 0x1fff; 
      ev = pfm_find_event_byvcode(tmp.pme_vcode, &(tmp_cmd.pfp_evt[0]));
      if (ev != PFMLIB_SUCCESS )
	return(PAPI_EINVAL);
#endif
      PFMW_EVT_EVTCOUNT(tmp_cmd) = 1;

      codes = &tmp_cmd;
    }

  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
/* tested by zhou */
   if (((EventCode >> 8) & 0xff) == 0x67) {
		set_dear_ita_param(EventCode);
		this_state->evt.pfp_model=&ear_ita_param;
	}

  /* Turn on the control codes and get the new bits required */

  nselector = set_hwcntr_codes(this_state,codes);
  if (nselector < 0)
    return retval;
  if (nselector == 0)
    abort();

  out->code = EventCode;
  /* Recalculate the selectors in this EventSet (the order may have changed). */
  _papi_recalc_selectors(this_state, out);

  /* Only the new fields */

  selector = this_state->selector ^ nselector;
  DBG((stderr,"This new event has selector 0x%x of 0x%x\n",selector,nselector));

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  out->selector = selector;

  /* Update the new counter select field */

  this_state->selector = nselector;
  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int used,i,j;
  unsigned int preset_index;

  /* Find out which counters used. */
  
  used = in->selector;

  /* Clear out counters that are part of this event. */

  this_state->selector = this_state->selector ^ used;
  /* We need to remove the count from this event, do we need to
   * reset the index of values too? -KSL 
   * Apparently so. -KSL */
  preset_index = in->code & PRESET_AND_MASK;
  for(i=0;i<PMU_MAX_COUNTERS;i++){
    if ( PFMW_EVT_EVENT(this_state->evt,i) & used ) {
      for ( j=i;j<(PMU_MAX_COUNTERS-1);j++ )
         PFMW_EVT_EVENT(this_state->evt,j) = PFMW_EVT_EVENT(this_state->evt,j+1);
    }
  } 
  PFMW_EVT_EVTCOUNT(this_state->evt)-=PFMW_EVT_EVTCOUNT(preset_map[preset_index].evt);

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

/*
int write_pmc_pmd()
{
      if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMCS, current_state->pc,
 current_state->pc_count) == -1) {
    fprintf(stderr,"child: perfmonctl error WRITE_PMCS errno %d\n",errno); pfm_s
tart(); return(PAPI_ESYS);
      }
}
*/


int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  pfmw_reg_t pd[PMU_MAX_COUNTERS];
  
/* for hardware profile, currently only support one eventset */
  if ((ESI->state & PAPI_PROFILING) && (_papi_system_info.supports_hw_profile))
  {
      int selector, hwcntr;

      pfm_stop();
      current_state->selector = this_state->selector;
      memcpy(&current_state->evt,&this_state->evt,sizeof this_state->evt);
      memcpy(current_state->pc,this_state->pc,sizeof this_state->pc);
      current_state->pc_count = this_state->pc_count;

      if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMCS, current_state->pc,
 current_state->pc_count) == -1) {
        fprintf(stderr,"child: perfmonctl error WRITE_PMCS errno %d\n",errno); 
        pfm_start(); 
		return(PAPI_ESYS);
      }

      memset(pd, 0, sizeof pd);
      for(i=0; i < PMU_MAX_COUNTERS; i++)
      {
        PFMW_REG_REGNUM(pd[i]) = PMU_MAX_COUNTERS+i;
      }
      selector = ESI->EventInfoArray[ESI->profile.EventIndex].selector;
      while ((i = ffs(selector)))
        {
          hwcntr = 1 << (i-1);
          if (hwcntr & selector)
          {
          DBG((stderr,"counter %d used in overflow, threshold %d\n",i-1-PMU_MAX_COUNTERS,ESI->profile.threshold));
          PFMW_REG_REGVAL(pd[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)ESI->profile.threshold;
          PFMW_REG_SMPLLRST(pd[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)ESI->profile.threshold;
          PFMW_REG_SMPLSRST(pd[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)ESI->profile.threshold;
          }
          selector ^= 1 << (i-1);
        }
      if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMDS, pd, PMU_MAX_COUNTERS) == -1) {
         fprintf(stderr,"child: perfmonctl error WRITE_PMDS errno %d\n",errno);
		 pfm_start(); return(PAPI_ESYS);
      }

    pfm_start();
    return(PAPI_OK);
  }

  /* If we ARE NOT nested, 
     just copy the global counter structure to the current eventset */

  if (current_state->selector == 0x0)
    {
      int selector, hwcntr;

      pfm_stop();

      current_state->selector = this_state->selector;

      memcpy(&current_state->evt,&this_state->evt,sizeof this_state->evt);
      memcpy(current_state->pc,this_state->pc,sizeof this_state->pc);
      current_state->pc_count = this_state->pc_count;

    restart_pm_hardware:

      if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMCS, current_state->pc, current_state->pc_count) == -1) {
	fprintf(stderr,"child: perfmonctl error WRITE_PMCS errno %d\n",errno); pfm_start(); return(PAPI_ESYS);
      }

      memset(pd, 0, sizeof pd);

      for(i=0; i < PMU_MAX_COUNTERS; i++) 
	{
	  PFMW_REG_REGNUM(pd[i]) = PMU_MAX_COUNTERS+i;  
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
		  PFMW_REG_REGVAL(pd[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)ESI->overflow.threshold;
		  PFMW_REG_SMPLLRST(pd[i-1-PMU_MAX_COUNTERS]) = (~0UL) - (unsigned long)ESI->overflow.threshold;
		}
	      selector ^= 1 << (i-1); 
	    }
	}
      
      if (pfmw_perfmonctl(current_state->pid, PFM_WRITE_PMDS, pd, PMU_MAX_COUNTERS) == -1) {
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

	for (j=0;j<PFMW_EVT_EVTCOUNT(this_state->evt);j++)
	  {
	    for (i=0;i<PFMW_EVT_EVTCOUNT(current_state->evt);i++)
	      {
		if (PFMW_EVT_EVENT(this_state->evt,j) == PFMW_EVT_EVENT(current_state->evt,i))
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
			new_selector_in_this_not_shared[i] |= 1 << (PFMW_EVT_EVTCOUNT(current_state->evt)+j+PMU_FIRST_COUNTER);
			new_operand_in_this_not_shared[i] = PFMW_EVT_EVTCOUNT(current_state->evt)+j+PMU_FIRST_COUNTER;
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
		PFMW_REG_REGNUM(this_state->pc[index_in_this[i]-PMU_FIRST_COUNTER]) = index_in_current[i];
		this_state->selector ^= 1 << index_in_this[i];
	      }
	  }
	for (i=0;i<not_shared;i++)
	  {
	    if (index_in_this_not_shared[i])
	      {
		PFMW_REG_REGNUM(this_state->pc[index_in_this_not_shared[i]-PMU_FIRST_COUNTER]) = 
		    PFMW_EVT_EVTCOUNT(current_state->evt)+i+PMU_FIRST_COUNTER;
		this_state->selector ^= 1 << index_in_this_not_shared[i];
	      }
	  }
	for (i=0;i<shared;i++)
	  this_state->selector |= 1 << index_in_current[i];
	for (i=0;i<not_shared;i++)
	  {
	    this_state->selector |= 1 << (PFMW_EVT_EVTCOUNT(current_state->evt)+i+PMU_FIRST_COUNTER);
	    /* Add the not shared events to the end of the list of the current structure running */
	    PFMW_EVT_EVENT(current_state->evt,PFMW_EVT_EVTCOUNT(current_state->evt)+i) = 
		PFMW_EVT_EVENT(this_state->evt,index_in_this_not_shared[i]-PMU_FIRST_COUNTER);
	  }
	PFMW_EVT_EVTCOUNT(current_state->evt) += not_shared;

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


int ia64_process_profile_entry()
{
    EventSetInfo *master_event_set;
    EventSetInfo *ESI;
    perfmon_smpl_hdr_t *hdr = (perfmon_smpl_hdr_t *)smpl_vaddr;
    perfmon_smpl_entry_t *ent;
    unsigned long pos;
	pfmw_arch_reg_t *reg;
/*
    unsigned long smpl_entry = 0;
*/
    int i, ret;
    struct sigcontext info;

    /*
     * Make sure the kernel uses the format we understand
     */
    if (PFM_VERSION_MAJOR(hdr->hdr_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
        fprintf(stderr,"Perfmon v%u.%u sampling format is not supported\n",
                PFM_VERSION_MAJOR(hdr->hdr_version),
                PFM_VERSION_MINOR(hdr->hdr_version));
    }

	master_event_set = _papi_hwi_lookup_in_master_list();
    if (master_event_set == NULL)
		return(PAPI_ESYS);
    if ((ESI = master_event_set->event_set_profiling)==NULL)
		return(PAPI_ESYS);
   /*
     * walk through all the entries recored in the buffer
     */
    pos = (unsigned long)(hdr+1);
    for(i=0; i < hdr->hdr_count; i++) {
        ret = 0;
        ent = (perfmon_smpl_entry_t *)pos;
		if ( ent->regs != 0 )
			ESI->profile.overflowcount++;
        /*
         * print entry header
         */
		if (((ESI->profile.EventCode >> 8) & 0xff) == 0x67) {
			reg = (pfmw_arch_reg_t*)(ent+1);
			reg++;
			reg++;
#ifdef ITANIUM2
            info.sc_ip = ((reg->pmd17_ita2_reg.dear_iaddr+reg->pmd17_ita2_reg.dear_bn) << 4) | reg->pmd17_ita2_reg.dear_slot;

#else
			info.sc_ip= (reg->pmd17_ita_reg.dear_iaddr<<4) | (reg->pmd17_ita_reg.dear_slot);
#endif
		} else info.sc_ip=ent->ip;

/*
        printf("Entry %ld PID:%d CPU:%d regs:0x%lx IIP:0x%016lx\n",
            smpl_entry++,
            ent->pid,
            ent->cpu,
            ent->regs,
            info.sc_ip);
*/

    	dispatch_profile(ESI, (caddr_t)&info, 0, ESI->profile.threshold);
        /*
         * move to next entry
         */
        pos += hdr->hdr_entry_size;
	}
    return(PAPI_OK);
}

#ifdef PFM06A
static void ia64_process_sigprof(int n, struct mysiginfo *info, struct sigconte
xt *context)
#else
static void ia64_process_sigprof(int n, pfm_siginfo_t *info, struct sigcontext
*context)
#endif
{
  if (info->sy_code != PROF_OVFL) {
    fprintf(stderr,"PAPI: received spurious SIGPROF si_code=%d\n", info->sy_code
);
    return;
  }
  ia64_process_profile_entry();
  pfmw_perfmonctl(info->sy_pid, PFM_RESTART, 0, 0);
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
  pfmw_perfmonctl(info->sy_pid, PFM_RESTART, 0, 0);

  pfm_start();
}

int set_notify(EventSetInfo *ESI, int index, int value)
{
	int selector, hwcntr, i;
    hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
    pfmw_reg_t *pc = this_state->pc;

    selector = ESI->EventInfoArray[index].selector;
    while ((hwcntr = ffs(selector)))
    {
      hwcntr = hwcntr - 1;
      for (i=0;i<PMU_MAX_COUNTERS;i++)
        {
          if (PFMW_REG_REGNUM(pc[i]) == hwcntr)
        {
          DBG((stderr,"Found hw counter %d in %d, flags %d\n",hwcntr,i,PFMW_REG_REGNUM(pc[i])));
          PFMW_REG_REGFLAGS(pc[i]) = value;
          break;
        }
        }
      selector ^= 1 << hwcntr;
    }
	return(PAPI_OK);
}

int _papi_hwd_stop_profiling(EventSetInfo *ESI, EventSetInfo *master)
{
  if (_papi_system_info.supports_hw_profile) {
	  ia64_process_profile_entry();
      master->event_set_profiling=NULL;
  }
   return(PAPI_OK);
}


int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
    struct sigaction act;
    void *tmp;
/*
    hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
    pfmw_reg_t *pc = this_state->pc;
*/
    pfmw_context_t ctx[1];

	if (profile_option->threshold == 0 ) {
/* unset notify */
		set_notify(ESI, profile_option->EventIndex, 0);
/* remove the signal handler */
        if (sigaction(SIGPROF, NULL, NULL) == -1)
            return(PAPI_ESYS);
	} else {
    	tmp = (void *)signal(SIGPROF, SIG_IGN);
    	if ((tmp != (void *)SIG_DFL) && (tmp != (void *)ia64_process_sigprof) )
           return(PAPI_EMISC);

      /* Set up the signal handler */

    	memset(&act,0x0,sizeof(struct sigaction));
    	act.sa_handler = (sig_t)ia64_process_sigprof;
    	act.sa_flags = SA_RESTART;
    	if (sigaction(SIGPROF, &act, NULL) == -1)
      		return(PAPI_ESYS);

     /* Set up the overflow notifier on the proper event.  */

		set_notify(ESI, profile_option->EventIndex, PFM_REGFL_OVFL_NOTIFY);

  /* need to rebuild the context */
    pfmw_perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0);
    memset(ctx, 0, sizeof(ctx));
    PFMW_CTX_NOTIFYPID(ctx[0]) = getpid();
    PFMW_CTX_FLAGS(ctx[0])     = PFM_FL_INHERIT_NONE;
#ifdef PFM06A
    ctx[0].pfr_ctx.notify_sig = SIGPROF;
#endif
    ctx[0].ctx_smpl_entries = SMPL_BUF_NENTRIES;

/* DEAR events */
    if (((profile_option->EventCode >> 8) & 0xff )==0x67) {
        ctx[0].ctx_smpl_regs[0] = DEAR_REGS_MASK;
	}

 if (pfmw_perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
    fprintf(stderr,"PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", getpid(),
 errno);
      		return(PAPI_ESYS);
  }
  DBG((stderr,"Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr));

  smpl_vaddr = ctx[0].ctx_smpl_vaddr;

  /*
   * reset PMU (guarantee not active on return) and unfreeze
   * must be done before writing to any PMC/PMD
   */
  if (pfmw_perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
    if (errno == ENOSYS)
      fprintf(stderr,"Your kernel does not have performance monitoring support !
\n");
    fprintf(stderr,"PID %d: perfmonctl error PFM_ENABLE %d\n",getpid(),errno);
  }
 }
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  pfmw_reg_t *pc = this_state->pc;
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
	      if (PFMW_REG_REGNUM(pc[i]) == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d, flags %d\n",hwcntr,i,PFMW_REG_REGNUM(pc[i])));
		  PFMW_REG_REGFLAGS(pc[i]) = 0;
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
	      if (PFMW_REG_REGNUM(pc[i]) == hwcntr)
		{
		  DBG((stderr,"Found hw counter %d in %d\n",hwcntr,i));
		  PFMW_REG_REGFLAGS(pc[i]) = PFM_REGFL_OVFL_NOTIFY;
		  break;
		}
	    }
	  selector ^= 1 << hwcntr;
	}

      PAPI_lock();
      _papi_hwi_using_signal++;
      PAPI_unlock();
    }
  return(retval);
}


void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_ip;

  return(location);
}

static volatile unsigned int lock = 0;
static volatile unsigned int *lock_addr = &lock;

void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(void)
{
  while (1)
    {
      if (test_and_set_bit(0,lock_addr))
	{
	  __asm__ __volatile__ ("mf" ::: "memory");
	  return;
	}
    }
}

void _papi_hwd_unlock(void)
{
  clear_bit(0, lock_addr);
  __asm__ __volatile__ ("mf" ::: "memory");
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
				 "LD_PRELOAD", /* How to preload libs */
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
			        1,  /* supports HW overflow */
#if defined(PFM20) && !defined(ITANIUM2) /* Only Libpfm 2.0+ and Itanium supports hardware profiling */
			        1,  /* supports HW profile */
#else
			        0,  /* supports HW profile */
#endif
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

