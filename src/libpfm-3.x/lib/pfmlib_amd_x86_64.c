/*
 * pfmlib_generic_x86_64.c : support for the AMD X86-64 PMU family
 * 			     (for both 64 and 32 bit modes)
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
#include <perfmon/pfmlib_amd_x86_64.h>

/* private headers */
#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_amd_x86_64_priv.h"	/* architecture private */
#include "amd_x86_64_events.h"		/* PMU private */

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

static char * pfm_amd_x86_64_get_event_name(unsigned int i);

#ifdef __x86_64__
static inline void cpuid(int op, unsigned int *eax,
				 unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{
	__asm__("cpuid"
			: "=a" (*eax),
			"=b" (*ebx),
			"=c" (*ecx),
			"=d" (*edx)
			: "0" (op));
}

static inline unsigned int cpuid_family(void)
{
	unsigned int eax, ebx, ecx, edx;
	cpuid(1, &eax, &ebx, &ecx, &edx);
	return eax;
}

static inline void cpuid_vendor(int op, unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{
	unsigned int eax;

	cpuid(op, &eax, ebx, ecx, edx);
}
#endif

#ifdef __i386__
static inline unsigned int cpuid_family(void)
{
	unsigned int eax, op = 1;

	__asm__("pushl %%ebx; cpuid; popl %%ebx"
			: "=a" (eax)
			: "0" (op)
			: "ecx", "edx");
	return eax;
}

static inline void cpuid_vendor(int op,
				 unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{	
	/*
	 * because ebx is used in Pic mode, we need to save/restore because
	 * cpuid clobbers it. I could not figure out a way to get ebx out in
	 * one cpuid instruction. To extract ebx, we need to  move it to another
	 * register (here eax)
	 */
	__asm__("pushl %%ebx;cpuid; movl %%ebx, %%eax;popl %%ebx"
			:"=a" (*ebx)
			: "a" (op)
			: "ecx", "edx");

	__asm__("pushl %%ebx;cpuid; popl %%ebx"
			:"=c" (*ecx), "=d" (*edx)
			: "a" (op));
}
#endif

static int
pfm_amd_x86_64_detect(void)
{
	unsigned int eax;
	char vendor_id[16];
	int ret = PFMLIB_ERR_NOTSUPP;

	memset(vendor_id, 0, sizeof(vendor_id));

	/*
	 * check that the core library supports enough registers
	 */
	if (PFMLIB_MAX_PMCS < PMU_AMD_X86_64_NUM_PERFSEL) return ret;
	if (PFMLIB_MAX_PMDS < PMU_AMD_X86_64_NUM_PERFCTR) return ret;

	eax = cpuid_family();
	cpuid_vendor(0, (unsigned int *)&vendor_id[0],
			(unsigned int *)&vendor_id[8],
			(unsigned int *)&vendor_id[4]);

	/*
	 * this file only supports AMD X86-64. Intel EM64T support is outside
	 * the scope of this libpfm module.
	 *
	 * accept family 15 AMD processors (all models)
	 */
	if (((eax>>8) & 0xf) == 15 && !strcmp(vendor_id, "AuthenticAMD")) {
		ret = PFMLIB_SUCCESS;
	}

	return ret;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 */
static int
pfm_amd_x86_64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_amd_x86_64_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_amd_x86_64_input_param_t *param = mod_in;
	pfmlib_amd_x86_64_counter_t *cntrs;
	pfm_amd_x86_64_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, cnt;
	unsigned int assign[PMU_AMD_X86_64_NUM_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_amd_x86_64_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, amd_x86_64_pe[e[j].event].pme_name));
		}
	}

	if (cnt > PMU_AMD_X86_64_NUM_COUNTERS) return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++) {
		/*
		 * X86-64 only supports two priv levels for perf counters
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT(("event=%d invalid plm=%d\n", e[j].event, e[j].plm));
			return PFMLIB_ERR_INVAL;
		}

		if (cntrs && (cntrs[j].cnt_mask >= PMU_AMD_X86_64_CNT_MASK_MAX)) {
			DPRINT(("event=%d invalid cnt_mask=%d: must be < %u\n",
				e[j].event,
				cntrs[j].cnt_mask,
				PMU_AMD_X86_64_CNT_MASK_MAX));
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * exclude restricted registers from assignement
		 */
		while(i < PMU_AMD_X86_64_NUM_COUNTERS && pfm_regmask_isset(r_pmcs, i)) i++;

		if (i == PMU_AMD_X86_64_NUM_COUNTERS)
			return PFMLIB_ERR_NOASSIGN;

		assign[j] = i++;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val    = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		reg.sel_event_mask = amd_x86_64_pe[e[j].event].pme_entry_code.pme_code.pme_emask;
		reg.sel_unit_mask  = amd_x86_64_pe[e[j].event].pme_entry_code.pme_code.pme_umask;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */
		if (cntrs) {
			reg.sel_cnt_mask   = cntrs[j].cnt_mask;
			reg.sel_edge	   = cntrs[j].flags & PFM_AMD_X86_64_SEL_EDGE ? 1 : 0;
			reg.sel_inv	   = cntrs[j].flags & PFM_AMD_X86_64_SEL_INV ? 1 : 0;
		}

		pc[j].reg_num     = assign[j];
		/*
		 * XXX: assumes perfmon2 X86-64 mappings!
		 */
		pc[j].reg_pmd_num = assign[j];
		pc[j].reg_evt_idx = j;

		pc[j].reg_value   = reg.val;

		__pfm_vbprintf("[perfsel%u=0x%llx emask=0x%lx umask=0x%lx os=%d usr=%d inv=%d en=%d int=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
			reg.val,
			reg.sel_event_mask,
			reg.sel_unit_mask,
			reg.sel_os,
			reg.sel_usr,
			reg.sel_inv,
			reg.sel_en,
			reg.sel_int,
			reg.sel_edge,
			reg.sel_cnt_mask,
			amd_x86_64_pe[e[j].event].pme_name);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_amd_x86_64_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_amd_x86_64_input_param_t *mod_in  = (pfmlib_amd_x86_64_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_amd_x86_64_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_amd_x86_64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > 3)
		return PFMLIB_ERR_INVAL;

	*code = amd_x86_64_pe[i].pme_entry_code.pme_code.pme_emask;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_amd_x86_64_get_event_umask(unsigned int i, unsigned long *umask)
{
	if (i >= PME_AMD_X86_64_EVENT_COUNT || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0; //evt_umask(i);
	return PFMLIB_SUCCESS;
}
	
static void
pfm_amd_x86_64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < PMU_AMD_X86_64_NUM_COUNTERS; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_amd_x86_64_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* all pmcs are contiguous */
	for(i=0; i < PMU_AMD_X86_64_NUM_PERFSEL; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_amd_x86_64_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* all pmds are contiguous */
	for(i=0; i < PMU_AMD_X86_64_NUM_PERFCTR; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_amd_x86_64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* counting pmds are contiguous */
	for(i=0; i < 4; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_amd_x86_64_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_AMD_X86_64_COUNTER_WIDTH;
}

static char *
pfm_amd_x86_64_get_event_name(unsigned int i)
{
	return amd_x86_64_pe[i].pme_name;
}

static int
pfm_amd_x86_64_get_event_description(unsigned int ev, char **str)
{
	char *s;
	s = amd_x86_64_pe[ev].pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t amd_x86_64_support={
	.pmu_name		= "AMD X86-64",
	.pmu_type		= PFMLIB_AMD_X86_64_PMU,
	.pme_count		= PME_AMD_X86_64_EVENT_COUNT,
	.pmc_count		= PMU_AMD_X86_64_NUM_PERFSEL,
	.pmd_count		= PMU_AMD_X86_64_NUM_PERFCTR,
	.num_cnt		= PMU_AMD_X86_64_NUM_COUNTERS,
	.cycle_event		= PME_AMD_X86_64_CPU_CLK_UNHALTED,
	.inst_retired_event	= PME_AMD_X86_64_RETIRED_X86_INST,
	.get_event_code		= pfm_amd_x86_64_get_event_code,
	.get_event_name		= pfm_amd_x86_64_get_event_name,
	.get_event_counters	= pfm_amd_x86_64_get_event_counters,
	.dispatch_events	= pfm_amd_x86_64_dispatch_events,
	.pmu_detect		= pfm_amd_x86_64_detect,
	.get_impl_pmcs		= pfm_amd_x86_64_get_impl_perfsel,
	.get_impl_pmds		= pfm_amd_x86_64_get_impl_perfctr,
	.get_impl_counters	= pfm_amd_x86_64_get_impl_counters,
	.get_hw_counter_width	= pfm_amd_x86_64_get_hw_counter_width,
	.get_event_desc         = pfm_amd_x86_64_get_event_description
};
