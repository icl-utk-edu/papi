/*
 * pfmlib_gen_ia32.c : Intel architectural PMU v1
 *
 * The file provides support for the Intel architectural PMU v1.
 *
 * It also provides support for Core Duo/Core Solo processors which
 * implement the architectural PMU with more than architected events.
 *
 * Copyright (c) 2005-2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
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
 * This file implements supports for the IA-32 architectural PMU as specified
 * in the following document:
 * 	"IA-32 Intel Architecture Software Developer's Manual - Volume 3B: System
 * 	Programming Guide"
 */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public headers */
#include <perfmon/pfmlib_gen_ia32.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_gen_ia32_priv.h"		/* architecture private */

#include "gen_ia32_events.h"			/* architected event table */
#include "coreduo_events.h"			/* Core Duo/Core Solo event table */

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

pfm_pmu_support_t coreduo_support;
pfm_pmu_support_t gen_ia32_support;
pfm_pmu_support_t *gen_support;

/*
 * Description of the PMC register mappings use by
 * this module (as reported in pfmlib_reg_t.reg_num):
 *
 * 0 -> PMC0 -> PERFEVTSEL0 -> MSR @ 0x186
 * 1 -> PMC1 -> PERFEVTSEL1 -> MSR @ 0x187
 * n -> PMCn -> PERFEVTSELn -> MSR @ 0x186+n
 *
 * 0 -> PMD0 -> IA32_PMC0   -> MSR @ 0xc1
 * 1 -> PMD1 -> IA32_PMC1   -> MSR @ 0xc2
 * 2 -> PMDn -> IA32_PMCn   -> MSR @ 0xc1+n
 */
#define GEN_IA32_SEL_BASE 0x186
#define GEN_IA32_CTR_BASE 0xc1

#define PFMLIB_GEN_IA32_ALL_FLAGS \
	(PFM_GEN_IA32_SEL_INV|PFM_GEN_IA32_SEL_EDGE)

static char * pfm_gen_ia32_get_event_name(unsigned int i);

static pme_gen_ia32_entry_t *gen_ia32_pe;

static int gen_ia32_cycle_event, gen_ia32_inst_retired_event;

#ifdef __i386__
static inline void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx,
			 unsigned int *ecx, unsigned int *edx)
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
#else
static inline void cpuid(unsigned int op, unsigned int *eax, unsigned int *ebx,
			 unsigned int *ecx, unsigned int *edx)
{
        __asm__("cpuid"
                        : "=a" (*eax),
                        "=b" (*ebx),
                        "=c" (*ecx),
                        "=d" (*edx)
                        : "0" (op), "c"(0));
}
#endif

/*
 * create architected event table
 */
static int
create_arch_event_table(unsigned int mask)
{
	pme_gen_ia32_entry_t *pe;
	unsigned int i, num_events = 0;
	unsigned int m;

	/*
	 * first pass: count the number of supported events
	 */
	m = mask;
	for(i=0; i < 7; i++, m>>=1) {
		if ((m & 0x1)  == 0)
			num_events++;
	}
	gen_ia32_support.pme_count = num_events;

	gen_ia32_pe = calloc(num_events, sizeof(pme_gen_ia32_entry_t));
	if (gen_ia32_pe == NULL)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * second pass: populate the table
	 */
	gen_ia32_cycle_event = gen_ia32_inst_retired_event = -1;
	m = mask;
	for(i=0, pe = gen_ia32_pe; i < 7; i++, m>>=1) {
		if ((m & 0x1)  == 0) {
			*pe = gen_ia32_all_pe[i];
			/*
			 * setup default event: cycles and inst_retired
			 */
			if (i == PME_GEN_IA32_UNHALTED_CORE_CYCLES)
				gen_ia32_cycle_event = pe - gen_ia32_pe;
			if (i == PME_GEN_IA32_INSTRUCTIONS_RETIRED)
				gen_ia32_inst_retired_event = pe - gen_ia32_pe;
			pe++;
		}
	}
	return PFMLIB_SUCCESS;
}

static int
check_arch_pmu(int family)
{
	union {
		unsigned int val;
		pmu_eax_t eax;
	} eax, ecx, edx, ebx;
	int ret;

	/*
	 * check family number to reject for processors
	 * older than Pentium (family=5). Those processors
	 * did not have the CPUID instruction
	 */
	if (family < 5)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * check if CPU supports 0xa function of CPUID
	 * 0xa started with Core Duo. Needed to detect if
	 * architected PMU is present
	 */
	cpuid(0x0, &eax.val, &ebx.val, &ecx.val, &edx.val);
	if (eax.val < 0xa)
		return PFMLIB_ERR_NOTSUPP;

	/*
	 * extract architected PMU information
	 */
	cpuid(0xa, &eax.val, &ebx.val, &ecx.val, &edx.val);

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

	gen_ia32_support.pmc_count = eax.eax.num_cnt;
	gen_ia32_support.pmd_count = eax.eax.num_cnt;
	gen_ia32_support.num_cnt   = eax.eax.num_cnt;

	ret = create_arch_event_table(ebx.val);
	if (ret != PFMLIB_SUCCESS)
		return ret;

	gen_support = &gen_ia32_support;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_detect(void)
{
	int ret, family;
	char buffer[128];

	ret = __pfm_getcpuinfo_attr("vendor_id", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	if (strcmp(buffer, "GenuineIntel"))
		return PFMLIB_ERR_NOTSUPP;

	ret = __pfm_getcpuinfo_attr("cpu family", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	family = atoi(buffer);

	return check_arch_pmu(family);
}

static int
pfm_coreduo_detect(void)
{
	int ret, family, model;
	char buffer[128];

	ret = __pfm_getcpuinfo_attr("vendor_id", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	if (strcmp(buffer, "GenuineIntel"))
		return PFMLIB_ERR_NOTSUPP;

	ret = __pfm_getcpuinfo_attr("cpu family", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	family = atoi(buffer);

	ret = __pfm_getcpuinfo_attr("model", buffer, sizeof(buffer));
	if (ret == -1)
		return PFMLIB_ERR_NOTSUPP;

	model = atoi(buffer);

	/*
	 * check for core solo/core duo
	 */
	if (family == 6 && model == 14) {
		gen_ia32_pe = coreduo_pe;
		gen_support = &coreduo_support;
		gen_ia32_cycle_event = PME_COREDUO_UNHALTED_CORE_CYCLES;
		gen_ia32_inst_retired_event = PME_COREDUO_INSTRUCTIONS_RETIRED;
		return PFMLIB_SUCCESS;
	}
	return PFMLIB_ERR_NOTSUPP;
}

static int
pfm_gen_ia32_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_gen_ia32_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_gen_ia32_input_param_t *param = mod_in;
	pfmlib_gen_ia32_counter_t *cntrs;
	pfm_gen_ia32_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, cnt, k, umask;
	unsigned int assign[PMU_GEN_IA32_MAX_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_gen_ia32_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, gen_ia32_pe[e[j].event].pme_name));
		}
	}

	if (cnt > gen_support->pmc_count)
		return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++) {
		/*
		 * P6 only supports two priv levels for perf counters
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT(("event=%d invalid plm=%d\n", e[j].event, e[j].plm));
			return PFMLIB_ERR_INVAL;
		}

		if (e[j].flags & ~PFMLIB_GEN_IA32_ALL_FLAGS) {
			DPRINT(("event=%d invalid flags=0x%lx\n", e[j].event, e[j].flags));
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * exclude restricted registers from assignment
		 */
		while(i < gen_support->pmc_count && pfm_regmask_isset(r_pmcs, i)) i++;

		if (i == gen_support->pmc_count)
			return PFMLIB_ERR_TOOMANY;

		/*
		 * events can be assigned to any counter
		 */
		assign[j] = i++;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		reg.sel_event_select = gen_ia32_pe[e[j].event].pme_code & 0xff;

		umask = (gen_ia32_pe[e[j].event].pme_code >> 8) & 0xff;

		for(k=0; k < e[j].num_masks; k++) {
			umask |= gen_ia32_pe[e[j].event].pme_umasks[e[j].unit_masks[k]].pme_ucode;
		}
		reg.sel_unit_mask  = umask;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */

		if (cntrs) {
			reg.sel_cnt_mask = cntrs[j].cnt_mask;
			reg.sel_edge	 = cntrs[j].flags & PFM_GEN_IA32_SEL_EDGE ? 1 : 0;
			reg.sel_inv	 = cntrs[j].flags & PFM_GEN_IA32_SEL_INV ? 1 : 0;
		}

		pc[j].reg_num     = assign[j];
		pc[j].reg_addr    = GEN_IA32_SEL_BASE+assign[j];
		pc[j].reg_value   = reg.val;

		pd[j].reg_num  = assign[j];
		pd[j].reg_addr = GEN_IA32_CTR_BASE+assign[j];

		__pfm_vbprintf("[PERFEVTSEL%u(pmc%u)=0x%llx event_sel=0x%x umask=0x%x os=%d usr=%d en=%d int=%d inv=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
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

		__pfm_vbprintf("[PMC%u(pmd%u)]\n", pd[j].reg_num, pd[j].reg_num);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;
	outp->pfp_pmd_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_gen_ia32_input_param_t *mod_in  = model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_gen_ia32_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_gen_ia32_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > gen_support->pmc_count)
		return PFMLIB_ERR_INVAL;

	*code = gen_ia32_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_gen_ia32_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < gen_support->pmc_count; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_gen_ia32_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	/* all pmcs are contiguous */
	for(i=0; i < gen_support->pmc_count; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_gen_ia32_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	/* all pmds are contiguous */
	for(i=0; i < gen_support->pmc_count; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_gen_ia32_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

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

static char *
pfm_gen_ia32_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return gen_ia32_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_gen_ia32_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = gen_ia32_pe[ev].pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_gen_ia32_get_num_event_masks(unsigned int ev)
{
	return gen_ia32_pe[ev].pme_numasks;
}

static int
pfm_gen_ia32_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code =gen_ia32_pe[ev].pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia32_get_cycle_event(pfmlib_event_t *e)
{
	if (gen_ia32_cycle_event == -1)
		return PFMLIB_ERR_NOTSUPP;

	e->event = gen_ia32_cycle_event;
	return PFMLIB_SUCCESS;

}

static int
pfm_gen_ia32_get_inst_retired(pfmlib_event_t *e)
{
	if (gen_ia32_inst_retired_event == -1)
		return PFMLIB_ERR_NOTSUPP;

	e->event = gen_ia32_inst_retired_event;
	return PFMLIB_SUCCESS;
}

/* architected PMU */
pfm_pmu_support_t gen_ia32_support={
	.pmu_name		= "Intel architectural PMU v1",
	.pmu_type		= PFMLIB_GEN_IA32_PMU,
	.pme_count		= 0,
	.pmc_count		= 0,
	.pmd_count		= 0,
	.num_cnt		= 0,
	.get_event_code		= pfm_gen_ia32_get_event_code,
	.get_event_name		= pfm_gen_ia32_get_event_name,
	.get_event_counters	= pfm_gen_ia32_get_event_counters,
	.dispatch_events	= pfm_gen_ia32_dispatch_events,
	.pmu_detect		= pfm_gen_ia32_detect,
	.get_impl_pmcs		= pfm_gen_ia32_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_ia32_get_impl_pmds,
	.get_impl_counters	= pfm_gen_ia32_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_ia32_get_hw_counter_width,
	.get_event_desc         = pfm_gen_ia32_get_event_description,
	.get_cycle_event	= pfm_gen_ia32_get_cycle_event,
	.get_inst_retired_event = pfm_gen_ia32_get_inst_retired,
	.get_num_event_masks	= pfm_gen_ia32_get_num_event_masks,
	.get_event_mask_name	= pfm_gen_ia32_get_event_mask_name,
	.get_event_mask_code	= pfm_gen_ia32_get_event_mask_code,
	.get_event_mask_desc	= pfm_gen_ia32_get_event_mask_desc
};

pfm_pmu_support_t coreduo_support={
	.pmu_name		= "Intel Core Duo/Core Solo",
	.pmu_type		= PFMLIB_COREDUO_PMU,
	.pme_count		= PME_COREDUO_EVENT_COUNT,
	.pmc_count		= 2,
	.pmd_count		= 2,
	.num_cnt		= 2,
	.get_event_code		= pfm_gen_ia32_get_event_code,
	.get_event_name		= pfm_gen_ia32_get_event_name,
	.get_event_counters	= pfm_gen_ia32_get_event_counters,
	.dispatch_events	= pfm_gen_ia32_dispatch_events,
	.pmu_detect		= pfm_coreduo_detect,
	.get_impl_pmcs		= pfm_gen_ia32_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_ia32_get_impl_pmds,
	.get_impl_counters	= pfm_gen_ia32_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_ia32_get_hw_counter_width,
	.get_event_desc         = pfm_gen_ia32_get_event_description,
	.get_num_event_masks	= pfm_gen_ia32_get_num_event_masks,
	.get_event_mask_name	= pfm_gen_ia32_get_event_mask_name,
	.get_event_mask_code	= pfm_gen_ia32_get_event_mask_code,
	.get_event_mask_desc	= pfm_gen_ia32_get_event_mask_desc,
	.get_cycle_event	= pfm_gen_ia32_get_cycle_event,
	.get_inst_retired_event = pfm_gen_ia32_get_inst_retired
};
