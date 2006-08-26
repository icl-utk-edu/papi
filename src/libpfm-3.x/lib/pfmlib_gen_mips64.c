/*
 * pfmlib_generic_mips64.c : support for the generic MIPS64 PMU family
 *
 * Contributed by Philip Mucci <mucci@cs.utk.edu> based on code from
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* public headers */
#include <perfmon/pfmlib_gen_mips64.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_gen_mips64_priv.h"		/* architecture private */
#include "gen_mips64_events.h"		/* PMU private */

/* let's define some handy shortcuts! */
#define sel_event_mask	perfsel.sel_event_mask
#define sel_exl		perfsel.sel_exl
#define sel_os		perfsel.sel_os
#define sel_usr		perfsel.sel_usr
#define sel_sup		perfsel.sel_sup
#define sel_int		perfsel.sel_int

static pme_gen_mips64_entry_t *gen_mips64_pe = NULL;

static char * pfm_gen_mips64_get_event_name(unsigned int i);

pfm_pmu_support_t generic_mips64_support;

static int
pfm_gen_mips64_detect(void)
{
	static char mips20k_name[] = "MIPS20K";
	static char mips5k_name[] = "MIPS5K";
	int ret;
	char buffer[128];

	ret = __pfm_getcpuinfo_attr("cpu model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	if (strstr(buffer,"MIPS 25Kf") || strstr(buffer,"MIPS 20Kc"))
	  {
	    gen_mips64_pe = gen_mips64_20k_pe;
	    generic_mips64_support.pme_count = (sizeof(gen_mips64_20k_pe)/sizeof(pme_gen_mips64_entry_t));
	    generic_mips64_support.pmu_name = mips20k_name;
	    generic_mips64_support.pmc_count = 1;
	    generic_mips64_support.pmd_count = 1;
	  }
#if 0
	else if (strstr(tmp,"MIPS 5Kc"))
#else
	  else
#endif
	  {
	    DPRINT(("warning: assuming MIPS5Kc\n"));
	    gen_mips64_pe = gen_mips64_5k_pe;
	    generic_mips64_support.pme_count = (sizeof(gen_mips64_5k_pe)/sizeof(pme_gen_mips64_entry_t));
	    generic_mips64_support.pmu_name = mips5k_name;
	    generic_mips64_support.pmc_count = 2;
	    generic_mips64_support.pmd_count = 2;
	  }
#if 0
	else
	  {
	    return PFMLIB_ERR_NOTSUPP;
	  }
#endif
	generic_mips64_support.num_cnt = generic_mips64_support.pmd_count;
	return PFMLIB_SUCCESS;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_regt structure is ready to be submitted to kernel
 */
static int
pfm_gen_mips64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_gen_mips64_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
        /* pfmlib_gen_mips64_input_param_t *param = mod_in; */
	pfm_gen_mips64_sel_reg_t reg;
	pfmlib_event_t *e = inp->pfp_events;
	pfmlib_reg_t *pc = outp->pfp_pmcs;
	unsigned long plm;
	unsigned int j, cnt = inp->pfp_event_count;
	unsigned int assign[PMU_GEN_MIPS64_NUM_COUNTERS];
	unsigned int used = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	/* Degree 2 rank based allocation */
	if (cnt > generic_mips64_support.pmc_count) return PFMLIB_ERR_TOOMANY;

	if (PFMLIB_DEBUG()) {
	  for (j=0; j < cnt; j++) {
	    DPRINT(("ev[%d]=%s, counters=0x%x\n", j, gen_mips64_pe[e[j].event].pme_name,gen_mips64_pe[e[j].event].pme_counters));
	  }
	}

	/* First fine out which events live on only 1 counter. */
	for (j=0; j < cnt ; j++ ) {
	  uint32_t tmp = gen_mips64_pe[e[j].event].pme_counters;
	  if ((tmp & 0x3) == 0x3)
	      assign[j] = 0;
	  else 
	      assign[j] = gen_mips64_pe[e[j].event].pme_counters;
	}

	/* Assign them first */
	for (j=0; j < cnt ; j++ ) {
	  if ((assign[j] & used) == 0) {
		reg.val    = 0; /* assume reserved bits are zerooed */
		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;
		reg.sel_usr = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os  = plm & PFM_PLM2 ? 1 : 0;
		reg.sel_sup = plm & PFM_PLM1 ? 1 : 0;
		reg.sel_exl = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_int = 1; /* force int to 1 */
		reg.sel_event_mask = (gen_mips64_pe[e[j].event].pme_entry_code.pme_code.pme_emask >> ((assign[j]-1)*4)) & 0xf;
		pc[j].reg_num     = ffs(assign[j]) - 1;
		pc[j].reg_pmd_num = ffs(assign[j]) - 1;
		pc[j].reg_evt_idx = ffs(assign[j]) - 1;
		pc[j].reg_value   = reg.val;
		used |= assign[j];
		DPRINT(("Degree 1: Used is now %x\n",used));
	  }
	  else {
	    return PFMLIB_ERR_NOASSIGN;
	  }
	}

	/* Now assign those that live on two counters. */
	for (j=0; j < cnt ; j++ ) {
	  if (assign[j] == 0) {
	    /* Which counters are available */
	    unsigned int avail = (~used & 0x3);
	    DPRINT(("Counters available: %x\n",avail));
	    if (avail == 0x0)
	      return PFMLIB_ERR_NOASSIGN;
	    /* Pick one */
	    avail = 1 << (ffs(avail) - 1);
	    DPRINT(("Selected: %x\n",used));
	    reg.val    = 0; /* assume reserved bits are zerooed */
	    /* if plm is 0, then assume not specified per-event and use default */
	    plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;
	    reg.sel_usr = plm & PFM_PLM3 ? 1 : 0;
	    reg.sel_os  = plm & PFM_PLM2 ? 1 : 0;
	    reg.sel_sup = plm & PFM_PLM1 ? 1 : 0;
	    reg.sel_exl = plm & PFM_PLM0 ? 1 : 0;
	    reg.sel_int = 1; /* force int to 1 */
	    reg.sel_event_mask = (gen_mips64_pe[e[j].event].pme_entry_code.pme_code.pme_emask >> ((avail-1)*4)) & 0xf;;
	    pc[j].reg_num     = ffs(avail) - 1;
	    pc[j].reg_pmd_num = ffs(avail) - 1;
	    pc[j].reg_evt_idx = ffs(avail) - 1;
	    pc[j].reg_value   = reg.val;
	    used |= avail;
	    DPRINT(("Degree 2: Used is now %x\n",used));
	  }
	}

	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_mips64_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_gen_mips64_input_param_t *mod_in  = (pfmlib_gen_mips64_input_param_t *)model_in;
	/* All PLMS are valid */
#if 0
	if (inp->pfp_dfl_plm & (PFM_PLM0|PFM_PLM1|PFM_PLM2|PFM_PLM3)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
#endif
	return pfm_gen_mips64_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_gen_mips64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	extern pfm_pmu_support_t generic_mips64_support;

	/* check validity of counter index */
	if (cnt != PFMLIB_CNT_FIRST) {
	  if (cnt < 0 || cnt >= generic_mips64_support.pmc_count)
	    return PFMLIB_ERR_INVAL; }
	else 	  {
	    cnt = ffs(gen_mips64_pe[i].pme_counters)-1;
	    if (cnt == -1)
	      return(PFMLIB_ERR_INVAL);
	  }
 
	/* if cnt == 1, shift right by 0, if cnt == 2, shift right by 4 */
	/* Works on both 5k anf 20K */

	if (gen_mips64_pe[i].pme_counters & (1<< cnt))
	  *code = 0xf & (gen_mips64_pe[i].pme_entry_code.pme_code.pme_emask >> (cnt*4));
	else
	  return PFMLIB_ERR_INVAL;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_gen_mips64_get_event_umask(unsigned int i, unsigned long *umask)
{
	extern pfm_pmu_support_t generic_mips64_support;
	if (i >= generic_mips64_support.pme_count || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0; //evt_umask(i);
	return PFMLIB_SUCCESS;
}
	
static void
pfm_gen_mips64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;
	extern pfm_pmu_support_t generic_mips64_support;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < generic_mips64_support.pmc_count; i++) {
		pfm_regmask_set(counters, i);
	}
}


static void
pfm_gen_mips64_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* all pmcs are contiguous */
	for(i=0; i < generic_mips64_support.pmc_count; i++) pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_gen_mips64_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* all pmds are contiguous */
	for(i=0; i < generic_mips64_support.pmd_count; i++) pfm_regmask_set(impl_pmds, i);
}

static void
pfm_gen_mips64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;
	extern pfm_pmu_support_t generic_mips64_support;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* counting pmds are contiguous */
	for(i=0; i < generic_mips64_support.pmc_count; i++) pfm_regmask_set(impl_counters, i);
}

static void
pfm_gen_mips64_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_GEN_MIPS64_COUNTER_WIDTH;
}

static char *
pfm_gen_mips64_get_event_name(unsigned int i)
{
	return gen_mips64_pe[i].pme_name;
}

static int
pfm_gen_mips64_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = gen_mips64_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t generic_mips64_support={
	.pmu_name		= "",
	.pmu_type		= PFMLIB_GEN_MIPS64_PMU,
	.pme_count		= 0,
	.pmc_count		= 0,
	.pmd_count		= 0,
	.num_cnt		= 0,
	.flags			= PFMLIB_MULT_CODE_EVENT,
	.cycle_event		= PME_GEN_MIPS64_CYC,
	.inst_retired_event	= PME_GEN_MIPS64_INST,
	.get_event_code		= pfm_gen_mips64_get_event_code,
	.get_event_name		= pfm_gen_mips64_get_event_name,
	.get_event_counters	= pfm_gen_mips64_get_event_counters,
	.dispatch_events	= pfm_gen_mips64_dispatch_events,
	.pmu_detect		= pfm_gen_mips64_detect,
	.get_impl_pmcs		= pfm_gen_mips64_get_impl_perfsel,
	.get_impl_pmds		= pfm_gen_mips64_get_impl_perfctr,
	.get_impl_counters	= pfm_gen_mips64_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_mips64_get_hw_counter_width,
	.get_event_desc         = pfm_gen_mips64_get_event_description
};
