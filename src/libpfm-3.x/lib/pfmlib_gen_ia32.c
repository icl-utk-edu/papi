/*
 * pfmlib_gen_ia32.c : support for architected IA-32 PMU
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
 *
 *
 * This file implements supports for the IA-32 architected PMU as specified in the
 * following document:
 * 	"IA-32 Intel Architecture Software Developer's Manual - Volume 3B: System
 * 	Programming Guide, Part 2"
 * 	Order Number: 253669-019
 * 	Date: March 2006
 * 	Section  18.11.1
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_gen_ia32.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_gen_ia32_priv.h"		/* architecture private */

/* let's define some handy shortcuts! */
#define sel_event_select perfevtsel.sel_event_select
#define sel_unit_mask	 perfevtsel.sel_unit_mask
#define sel_usr		 perfevtsel.sel_usr
#define sel_os		 perfevtsel.sel_os
#define sel_edge	 perfevtsel.sel_edge
#define sel_pc		 perfevtsel.sel_pc
#define sel_int		 perfevtsel.sel_int
#define sel_en		 perfevtsel.sel_en
#define sel_inv		 perfevtsel.sel_inv
#define sel_cnt_mask	 perfevtsel.sel_cnt_mask

pfm_pmu_support_t gen_ia32_support;

static char * pfm_gen_ia32_get_event_name(unsigned int i);

static pme_gen_ia32_entry_t gen_ia32_all_pe[]={
#define PME_GEN_IA32_UNHALTED_CORE_CYCLES 0
	{.pme_name = "UNHALTED_CORE_CYCLES",
	 .pme_entry_code.pme_vcode = 0x003c,
	 .pme_desc =  "count core clock cycles whenever the clock signal on the specific core is running (not halted)"
	},
#define PME_GEN_IA32_INSTRUCTIONS_RETIRED 1
	{.pme_name = "INSTRUCTIONS_RETIRED",
	 .pme_entry_code.pme_vcode = 0x00c0,
	 .pme_desc =  "count the number of instructions at retirement. For instructions that consists of multiple mnicro-ops, this event counts the retirement of the last micro-op of the instruction",
	},
	{.pme_name = "UNHALTED_REFERENCE_CYCLES",
	 .pme_entry_code.pme_vcode = 0x013c,
	 .pme_desc =  "count reference clock cycles while the clock signal on the specific core is running. The reference clock operates at a fixed frequency, irrespective of core freqeuncy changes due to performance state transitions",
	},
	{.pme_name = "LAST_LEVEL_CACHE_REFERENCES",
	 .pme_entry_code.pme_vcode = 0x4f2e,
	 .pme_desc =  "count each request originating from the core to reference a cache line in the last level cache. The count may include speculation, but excludes cache line fills due to hardware prefetch",
	},
	{.pme_name = "LAST_LEVEL_CACHE_MISSES",
	 .pme_entry_code.pme_vcode = 0x412e,
	 .pme_desc =  "count each cache miss condition for references to the last level cache. The event count may include speculation, but excludes cache line fills due to hardware prefetch",
	},
	{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	 .pme_entry_code.pme_vcode = 0x00c4,
	 .pme_desc =  "count branch instructions at retirement. Specifically, this event counts the retirement of the last micro-op of a branch instruction",
	},
	{.pme_name = "ALL_BRANCH_MISPREDICT_RETIRED",
	 .pme_entry_code.pme_vcode = 0x00c5,
	 .pme_desc =  "count mispredicted branch instructions at retirement. Specifically, this event counts at retirement of the last micro-op of a branch instruction in the architectural path of the execution and experienced misprediction in the branch prediction hardware",
	}
};
#define PFMLIB_GEN_IA32_DEF_NUM_EVENTS	(sizeof(gen_ia32_all_pe)/sizeof(pme_gen_ia32_entry_t))

static pme_gen_ia32_entry_t *gen_ia32_pe;

static inline unsigned int cpuid_eax(unsigned int op)
{
	unsigned int eax;

	__asm__("pushl %%ebx; cpuid; popl %%ebx"
			: "=a" (eax)
			: "0" (op)
			: "ecx", "edx");
	return eax;
}

static inline void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx)
{
	/*
	 * because ebx is used in Pic mode, we need to save/restore because
	 * cpuid clobbers it. I could not figure out a way to get ebx out in
	 * one cpuid instruction. To extract ebx, we need to  move it to another
	 * register (here eax)
	 */
	__asm__("pushl %%ebx;cpuid; popl %%ebx"
			:"=a" (*eax)
			: "a" (op)
			: "ecx", "edx");

	__asm__("pushl %%ebx;cpuid; movl %%ebx, %%eax;popl %%ebx"
			:"=a" (*ebx)
			: "a" (op)
			: "ecx", "edx");
}

/*
 * detect presence of architected PMU
 */
static int
pfm_gen_ia32_detect(void)
{
	union {
		unsigned int val;
		pmu_eax_t eax;
	} eax;
	pme_gen_ia32_entry_t *pe;
	unsigned int ebx = 0, mask;
	unsigned int i, num_events;

	/*
	 * check if CPU supoprt 0xa function of CPUID
	 * 0xa started with Core Duo. Needed to detect if
	 * architected PMU is present
	 */
	cpuid(0x0, &eax.val, &ebx);

	if (eax.val < 0xa)
		return PFMLIB_ERR_NOTSUPP;
	/*
	 * extract architected PMU information
	 */
	cpuid(0xa, &eax.val, &ebx);
	/*
	 * check version. must be greater than zero
	 */
	if (eax.eax.version < 1)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * sanity check number of counters
	 */
	if (eax.eax.num_cnt == 0)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * only support counter which air paired PERFEVTSEL/PMC
	 * In the Intel terminology a counting PMD register is a PMC
	 */
	gen_ia32_support.pmc_count = eax.eax.num_cnt;
	gen_ia32_support.pmd_count = eax.eax.num_cnt;
	gen_ia32_support.num_cnt   = eax.eax.num_cnt;

	num_events = 0;
	mask = ebx;
	/*
	 * first pass: count the number of supported events
	 */
	for(i=0; i < 7; i++, mask>>=1) {
		if ((mask & 0x1)  == 0)
			num_events++;
	}

	gen_ia32_support.pme_count = num_events;

	/*
	 * alloc event table
	 */
	gen_ia32_pe = malloc(num_events*sizeof(pme_gen_ia32_entry_t));
	if (gen_ia32_pe == NULL)
		return PFMLIB_ERR_NOTSUPP;
	/*
	 * second pass: populate event table
	 */
	mask = ebx;
	for(i=0, pe = gen_ia32_pe; i < 7; i++, mask>>=1) {
		if ((mask & 0x1)  == 0) {
			*pe = gen_ia32_all_pe[i];
			/*
			 * setup default event: cycles and inst_retired
			 */
			if (i == PME_GEN_IA32_UNHALTED_CORE_CYCLES)
				gen_ia32_support.cycle_event = pe - gen_ia32_pe;
			if (i == PME_GEN_IA32_INSTRUCTIONS_RETIRED)
				gen_ia32_support.inst_retired_event = pe - gen_ia32_pe;
			pe++;
		}
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_gen_ia32_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_gen_ia32_input_param_t *param = mod_in;
	pfmlib_gen_ia32_counter_t *cntrs;
	pfm_gen_ia32_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, cnt;
	unsigned int assign[PMU_GEN_IA32_MAX_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_gen_ia32_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, gen_ia32_pe[e[j].event].pme_name));
		}
	}

	if (cnt > gen_ia32_support.pmc_count) return PFMLIB_ERR_TOOMANY;

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
		while(i < gen_ia32_support.pmc_count && pfm_regmask_isset(r_pmcs, i)) i++;

		if (i == gen_ia32_support.pmc_count)
			return PFMLIB_ERR_NOASSIGN;

		/*
		 * events can be assigned to any counter
		 */
		assign[j] = i++;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val    = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		reg.sel_event_select = gen_ia32_pe[e[j].event].pme_entry_code.pme_code.pme_sel;
		reg.sel_unit_mask    = gen_ia32_pe[e[j].event].pme_entry_code.pme_code.pme_umask;
		reg.sel_usr          = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os           = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en           = 1; /* force enable bit to 1 */
		reg.sel_int          = 1; /* force APIC int to 1 */
		if (cntrs) {
			reg.sel_cnt_mask   = cntrs[j].cnt_mask;

			/*
			 * certain events require edge to be set
			 */
			reg.sel_edge	   = cntrs[j].flags & PFM_GEN_IA32_SEL_EDGE ? 1 : 0;
			reg.sel_inv	   = cntrs[j].flags & PFM_GEN_IA32_SEL_INV ? 1 : 0;
		}

		/*
		 * XXX: assumes perfmon2 mappings
		 */
		pc[j].reg_num     = assign[j];
		pc[j].reg_pmd_num = assign[j];
		pc[j].reg_evt_idx = j;
		pc[j].reg_value   = reg.val;

		__pfm_vbprintf("[perfevtsel%u=0x%llx event_sel=0x%lx umask=0x%lx os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
			reg.val,
			reg.sel_event_select,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_en,
			reg.sel_int,
			reg.sel_inv,
			reg.sel_edge,
			reg.sel_cnt_mask,
			gen_ia32_pe[e[j].event].pme_name);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_gen_ia32_input_param_t *mod_in  = (pfmlib_gen_ia32_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_gen_ia32_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_gen_ia32_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > gen_ia32_support.pmc_count)
		return PFMLIB_ERR_INVAL;

	*code = gen_ia32_pe[i].pme_entry_code.pme_code.pme_sel;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_gen_ia32_get_event_umask(unsigned int i, unsigned long *umask)
{
	if (i >= gen_ia32_support.pme_count || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0;
	return PFMLIB_SUCCESS;
}
	
static void
pfm_gen_ia32_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < gen_ia32_support.pmc_count; i++)
		pfm_regmask_set(counters, i);
}


static void
pfm_gen_ia32_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* all pmcs are contiguous */
	for(i=0; i < gen_ia32_support.pmc_count; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_gen_ia32_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* all pmds are contiguous */
	for(i=0; i < gen_ia32_support.pmc_count; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_gen_ia32_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* counting pmds are contiguous */
	for(i=0; i < 4; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_gen_ia32_get_hw_counter_width(unsigned int *width)
{
	/*
	 * Even though, CPUID 0xa returns in eax the actual counter
	 * width, the architecture specifies that writes are limited
	 * to lower 32-bits. As such, only the lower 32-bit have full
	 * degree of freedom. That is the "useable" counter width.
	 */
	*width = PMU_GEN_IA32_COUNTER_WIDTH;
}

static char *
pfm_gen_ia32_get_event_name(unsigned int i)
{
	return gen_ia32_pe[i].pme_name;
}

static int
pfm_gen_ia32_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = gen_ia32_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

/* architected IA-32 PMU */
pfm_pmu_support_t gen_ia32_support={
	.pmu_name		= "IA-32 Generic",
	.pmu_type		= PFMLIB_GEN_IA32_PMU,
	.pme_count		= 0,
	.pmc_count		= 0,
	.pmd_count		= 0,
	.num_cnt		= 0,
	.cycle_event		= PFMLIB_NO_EVT,
	.inst_retired_event	= PFMLIB_NO_EVT,
	.get_event_code		= pfm_gen_ia32_get_event_code,
	.get_event_name		= pfm_gen_ia32_get_event_name,
	.get_event_counters	= pfm_gen_ia32_get_event_counters,
	.dispatch_events	= pfm_gen_ia32_dispatch_events,
	.pmu_detect		= pfm_gen_ia32_detect,
	.get_impl_pmcs		= pfm_gen_ia32_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_ia32_get_impl_pmds,
	.get_impl_counters	= pfm_gen_ia32_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_ia32_get_hw_counter_width,
	.get_event_desc         = pfm_gen_ia32_get_event_description
};
