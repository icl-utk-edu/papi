/*
 * pfmlib_i386_pm.c : support for the P6 processor family (family=6)
 * 		      incl. Pentium II, Pentium III, Pentium Pro, Pentium M
 *
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

/* public headers */
#include <perfmon/pfmlib_i386_p6.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_i386_p6_priv.h"		/* architecture private */
#include "i386_p6_events.h"			/* event tables */

/* let's define some handy shortcuts! */
#define sel_event_mask	perfsel.sel_event_mask
#define sel_unit_mask	perfsel.sel_unit_mask
#define sel_usr		perfsel.sel_usr
#define sel_os		perfsel.sel_os
#define sel_edge	perfsel.sel_edge
#define sel_pc		perfsel.sel_pc
#define sel_int		perfsel.sel_int
#define sel_en		perfsel.sel_en
#define sel_inv		perfsel.sel_inv
#define sel_cnt_mask	perfsel.sel_cnt_mask

static char * pfm_i386_p6_get_event_name(unsigned int i);
static pme_i386_p6_entry_t *i386_pe;
static unsigned int i386_p6_num_events;

static inline unsigned int cpuid_eax(unsigned int op)
{
	unsigned int eax;

	__asm__("pushl %%ebx; cpuid; popl %%ebx"
			: "=a" (eax)
			: "0" (op)
			: "cx", "dx");
	return eax;
}

/*
 * Verify we are on an Intel processor
 * XXX: siwtch to CPUID to avoid dependency on /proc/cpuinfo
 */
static int
check_intel_vendor(void)
{
	FILE *fp1;	
	char buffer[128], *p, *value;

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return 0;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL)
			break;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			break;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp("vendor_id", buffer, 9)) {
			return strcmp(value, "GenuineIntel");
		}
	}
	fclose(fp1);
	return -1;
}

/*
 * detect only non Pentium P6 cores
 */
static int
pfm_i386_p6_detect(void)
{
	unsigned long eax;
	unsigned long family, model;

	/*
	 * check that the core library supports enough registers
	 */
	if (PFMLIB_MAX_PMCS < PMU_I386_P6_NUM_PERFSEL) return PFMLIB_ERR_NOTSUPP;
	if (PFMLIB_MAX_PMDS < PMU_I386_P6_NUM_PERFCTR) return PFMLIB_ERR_NOTSUPP;

	eax = cpuid_eax(1);

	/*
	 * we cannot check MSR_MISC_ENABLE from user level. Host
	 * processoor may not have PMU.
	 */
	family = (eax>>8) & 0xf;
	model  = (eax>>4) & 0xf;

	if (family != 0x6)
		return PFMLIB_ERR_NOTSUPP;

	if (check_intel_vendor())
		return PFMLIB_ERR_NOTSUPP;

	switch(model) {
                case 3: /* Pentium II */
		case 7: /* Pentium III Katmai */
		case 8: /* Pentium III Coppermine */
		case 9: /* Mobile Pentium III */
		case 10:/* Pentium III Cascades */
		case 11:/* Pentium III Tualatin */
			break;
		default:
			return PFMLIB_ERR_NOTSUPP;
	}
	i386_pe = i386_p6_pe;
	i386_p6_num_events = PME_I386_P6_EVENT_COUNT;
	return PFMLIB_SUCCESS;
}

static int
pfm_i386_pm_detect(void)
{
	unsigned long eax;
	unsigned long family, model;

	/*
	 * check that the core library supports enough registers
	 */
	if (PFMLIB_MAX_PMCS < PMU_I386_P6_NUM_PERFSEL) return PFMLIB_ERR_NOTSUPP;
	if (PFMLIB_MAX_PMDS < PMU_I386_P6_NUM_PERFCTR) return PFMLIB_ERR_NOTSUPP;

	eax = cpuid_eax(1);

	/*
	 * we cannot check MSR_MISC_ENABLE from user level. Host
	 * processor may not have PMU.
	 */
	family = (eax>>8) & 0xf;
	model  = (eax>>4) & 0xf;

	/*
	 * only looking for Pentium M
	 */
	if (family != 0x6 || model != 13) return PFMLIB_ERR_NOTSUPP;

	i386_pe = i386_pm_pe;
	i386_p6_num_events = PME_I386_PM_EVENT_COUNT;

	return PFMLIB_SUCCESS;
}
/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_regt structure is ready to be submitted to kernel
 */
static int
pfm_i386_p6_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_i386_p6_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_i386_p6_input_param_t *param = mod_in;
	pfmlib_i386_p6_counter_t *cntrs;
	pfm_i386_p6_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, cnt;
	unsigned int assign[PMU_I386_P6_NUM_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_i386_p6_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, i386_pe[e[j].event].pme_name));
		}
	}

	if (cnt > PMU_I386_P6_NUM_COUNTERS) return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++) {
		/*
		 * P6 only supports two priv levels for perf counters
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT(("event=%d invalid plm=%d\n", e[j].event, e[j].plm));
			return PFMLIB_ERR_INVAL;
		}
		
		/*
		 * exclude restricted registers from assignement
		 */
		while(i < PMU_I386_P6_NUM_COUNTERS && pfm_regmask_isset(r_pmcs, i)) i++;

		if (i == PMU_I386_P6_NUM_COUNTERS)
			return PFMLIB_ERR_NOASSIGN;

		assign[j] = i++;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val    = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		reg.sel_event_mask = i386_pe[e[j].event].pme_entry_code.pme_code.pme_emask;
		reg.sel_unit_mask  = i386_pe[e[j].event].pme_entry_code.pme_code.pme_umask;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */
		if (cntrs) {
			reg.sel_cnt_mask   = cntrs[j].cnt_mask;

			/*
			 * certain events require edge to be set
			 */
			reg.sel_edge	   = cntrs[j].flags & PFM_I386_P6_SEL_EDGE
					   ||  i386_pe[e[j].event].pme_entry_code.pme_code.pme_edge ? 1 : 0;

			reg.sel_inv	   = cntrs[j].flags & PFM_I386_P6_SEL_INV ? 1 : 0;
		}

		/*
		 * XXX: assumes perfmon2 mappings
		 */
		pc[j].reg_num     = assign[j];
		pc[j].reg_pmd_num = assign[j];
		pc[j].reg_evt_idx = j;
		pc[j].reg_value   = reg.val;

		__pfm_vbprintf("[perfsel%u=0x%lx emask=0x%lx umask=0x%lx os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
			reg.val,
			reg.sel_event_mask,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask,
			i386_pe[e[j].event].pme_name);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_i386_p6_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_i386_p6_input_param_t *mod_in  = (pfmlib_i386_p6_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_i386_p6_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_i386_p6_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > 2)
		return PFMLIB_ERR_INVAL;
	
	*code = i386_pe[i].pme_entry_code.pme_code.pme_emask;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_i386_p6_get_event_umask(unsigned int i, unsigned long *umask)
{
	if (i >= i386_p6_num_events || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0;
	return PFMLIB_SUCCESS;
}
	
static void
pfm_i386_p6_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < PMU_I386_P6_NUM_COUNTERS; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_i386_p6_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* all pmcs are contiguous */
	for(i=0; i < PMU_I386_P6_NUM_PERFSEL; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_i386_p6_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* all pmds are contiguous */
	for(i=0; i < PMU_I386_P6_NUM_PERFCTR; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_i386_p6_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* counting pmds are contiguous */
	for(i=0; i < 4; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_i386_p6_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_I386_P6_COUNTER_WIDTH;
}

static char *
pfm_i386_p6_get_event_name(unsigned int i)
{
	return i386_pe[i].pme_name;
}

static int
pfm_i386_p6_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = i386_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

/* Generic P6 processor support (not incl. Pentium M) */
pfm_pmu_support_t i386_p6_support={
	.pmu_name		= "P6 Processor Family",
	.pmu_type		= PFMLIB_I386_P6_PMU,
	.pme_count		= PME_I386_P6_EVENT_COUNT,
	.pmc_count		= PMU_I386_P6_NUM_PERFSEL,
	.pmd_count		= PMU_I386_P6_NUM_PERFCTR,
	.num_cnt		= PMU_I386_P6_NUM_COUNTERS,
	.cycle_event		= PME_I386_P6_CPU_CLK_UNHALTED,
	.inst_retired_event	= PME_I386_P6_INST_RETIRED,
	.get_event_code		= pfm_i386_p6_get_event_code,
	.get_event_name		= pfm_i386_p6_get_event_name,
	.get_event_counters	= pfm_i386_p6_get_event_counters,
	.dispatch_events	= pfm_i386_p6_dispatch_events,
	.pmu_detect		= pfm_i386_p6_detect,
	.get_impl_pmcs		= pfm_i386_p6_get_impl_perfsel,
	.get_impl_pmds		= pfm_i386_p6_get_impl_perfctr,
	.get_impl_counters	= pfm_i386_p6_get_impl_counters,
	.get_hw_counter_width	= pfm_i386_p6_get_hw_counter_width,
	.get_event_desc         = pfm_i386_p6_get_event_description
};

/* Pentium M support */
pfm_pmu_support_t i386_pm_support={
	.pmu_name		= "Pentium M",
	.pmu_type		= PFMLIB_I386_P6_PMU,
	.pme_count		= PME_I386_PM_EVENT_COUNT,
	.pmc_count		= PMU_I386_P6_NUM_PERFSEL,
	.pmd_count		= PMU_I386_P6_NUM_PERFCTR,
	.num_cnt		= PMU_I386_P6_NUM_COUNTERS,
	.cycle_event		= PME_I386_PM_CPU_CLK_UNHALTED,
	.inst_retired_event	= PME_I386_PM_INST_RETIRED,
	.get_event_code		= pfm_i386_p6_get_event_code,
	.get_event_name		= pfm_i386_p6_get_event_name,
	.get_event_counters	= pfm_i386_p6_get_event_counters,
	.dispatch_events	= pfm_i386_p6_dispatch_events,
	.pmu_detect		= pfm_i386_pm_detect,
	.get_impl_pmcs		= pfm_i386_p6_get_impl_perfsel,
	.get_impl_pmds		= pfm_i386_p6_get_impl_perfctr,
	.get_impl_counters	= pfm_i386_p6_get_impl_counters,
	.get_hw_counter_width	= pfm_i386_p6_get_hw_counter_width,
	.get_event_desc         = pfm_i386_p6_get_event_description
};
