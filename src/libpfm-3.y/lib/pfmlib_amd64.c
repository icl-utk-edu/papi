/*
 * pfmlib_amd64.c : support for the AMD64 architected PMU
 * 		    (for both 64 and 32 bit modes)
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
#include <stdlib.h>

/* public headers */
#include <perfmon/pfmlib_amd64.h>

/* private headers */
#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_amd64_priv.h"	/* architecture private */
#include "amd64_events.h"		/* PMU private */

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
#define sel_event_mask2 perfsel.sel_event_mask2
#define sel_guest	perfsel.sel_guest
#define sel_host	perfsel.sel_host

#define IS_FAM10H_ONLY(reg) \
	((reg).sel_event_mask2 || (reg).sel_guest || (reg).sel_host)

#define PFMLIB_AMD64_HAS_COMBO(_e) \
	((pfm_amd64_get_event_entry(_e)->pme_flags & PFMLIB_AMD64_UMASK_COMBO) != 0)

/*
 * Description of the PMC register mappings use by
 * this module:
 * pfp_pmcs[].reg_num:
 * 	0 -> PMC0 -> PERFEVTSEL0 -> MSR @ 0xc0010000
 * 	1 -> PMC1 -> PERFEVTSEL1 -> MSR @ 0xc0010001
 * 	...
 * pfp_pmds[].reg_num:
 * 	0 -> PMD0 -> PERCTR0 -> MSR @ 0xc0010004
 * 	1 -> PMD1 -> PERCTR1 -> MSR @ 0xc0010005
 * 	...
 */
#define AMD64_SEL_BASE	0xc0010000
#define AMD64_CTR_BASE	0xc0010004

#define AMD64_FAM10H AMD64_FAM10H_REV_B

typedef enum {
	AMD64_CPU_UN, 
	AMD64_K8_REV_B,
	AMD64_K8_REV_C,
	AMD64_K8_REV_D,
	AMD64_K8_REV_E,
	AMD64_K8_REV_F,
	AMD64_K8_REV_G,
	AMD64_FAM10H_REV_B,
	AMD64_FAM10H_REV_C,
} amd64_rev_t;

static const char *amd64_rev_strs[]= {
	"?", "B", "C", "D", "E", "F", "G", "B", "C"
};

static const char *amd64_cpu_strs[]= {
	"unknown model",
	"K8 RevB",
	"K8 RevC",
	"K8 RevD",
	"K8 RevE",
	"K8 RevF",
	"K8 RevG",
	"Barcelona RevB",
	"Barcelona RevC",
};

#define NAME_SIZE 32
static struct {
	amd64_rev_t	revision;
	char		name[NAME_SIZE];
	unsigned int	cpu_clks;
	unsigned int	ret_inst;
	pme_amd64_entry_t *events;
} amd64_pmu;

pfm_pmu_support_t amd64_support;

#define amd64_revision    amd64_pmu.revision
#define amd64_event_count amd64_support.pme_count
#define amd64_cpu_clks    amd64_pmu.cpu_clks
#define amd64_ret_inst    amd64_pmu.ret_inst
#define amd64_events      amd64_pmu.events

#define IS_FAMILY_10H() (amd64_pmu.revision >= AMD64_FAM10H)

static amd64_rev_t
amd64_get_revision(int family, int model, int stepping)
{
	if (family == 15) {
		switch (model >> 4) {
		case 0:
			if (model == 5 && stepping < 2)
				return AMD64_K8_REV_B;
			if (model == 4 && stepping == 0)
				return AMD64_K8_REV_B;
			return AMD64_K8_REV_C;
		case 1:
			return AMD64_K8_REV_D;
		case 2:
		case 3:
			return AMD64_K8_REV_E;
		case 4:
		case 5:
		case 0xc:
			return AMD64_K8_REV_F;
		case 6:
		case 7:
		case 8:
			return AMD64_K8_REV_G;
		default:
			return AMD64_K8_REV_B;
		}
	} else if (family == 16) {
		if (model <= 3)
			return AMD64_FAM10H_REV_B;
		if (model <= 6)
			return AMD64_FAM10H_REV_C;
		return AMD64_FAM10H_REV_B;
	}

	return AMD64_CPU_UN;
}

/*
 * .byte 0x53 == push ebx. it's universal for 32 and 64 bit
 * .byte 0x5b == pop ebx.
 * Some gcc's (4.1.2 on Core2) object to pairing push/pop and ebx in 64 bit mode.
 * Using the opcode directly avoids this problem.
 */
static inline void cpuid(unsigned int op, unsigned int *a, unsigned int *b,
                  unsigned int *c, unsigned int *d)
{
  __asm__ __volatile__ (".byte 0x53\n\tcpuid\n\tmovl %%ebx, %%esi\n\t.byte 0x5b"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op));
}

static void
pfm_amd64_setup(int revision)
{
	amd64_pmu.revision = revision;
	snprintf(amd64_pmu.name, NAME_SIZE, "AMD64 (%s)",
		 amd64_cpu_strs[revision]);
	if (IS_FAMILY_10H()) {
		amd64_pmu.events = amd64_fam10h_table.events;
		amd64_support.pme_count = amd64_fam10h_table.num;
		amd64_pmu.cpu_clks = amd64_fam10h_table.cpu_clks;
		amd64_pmu.ret_inst = amd64_fam10h_table.ret_inst;
	} else {
		amd64_pmu.events = amd64_k8_table.events;
		amd64_support.pme_count = amd64_k8_table.num;
		amd64_pmu.cpu_clks = amd64_k8_table.cpu_clks;
		amd64_pmu.ret_inst = amd64_k8_table.ret_inst;
	}
	amd64_support.pmu_name		= amd64_pmu.name;
	amd64_support.pmu_type		= PFMLIB_AMD64_PMU;
	amd64_support.pmc_count		= PMU_AMD64_NUM_PERFSEL;
	amd64_support.pmd_count		= PMU_AMD64_NUM_PERFCTR;
	amd64_support.num_cnt		= PMU_AMD64_NUM_COUNTERS;
}

static int
pfm_amd64_detect(void)
{
	unsigned int a, b, c, d;
	int family, model, stepping;
	char buffer[128];

	cpuid(0, &a, &b, &c, &d);
	strncpy(&buffer[0], (char *)(&b), 4);
	strncpy(&buffer[4], (char *)(&d), 4);
	strncpy(&buffer[8], (char *)(&c), 4);
	buffer[12] = '\0';

	if (strcmp(buffer, "AuthenticAMD"))
		return PFMLIB_ERR_NOTSUPP;

	cpuid(1, &a, &b, &c, &d);
	family = (a >> 8) & 0x0000000f;  // bits 11 - 8
	model  = (a >> 4) & 0x0000000f;  // Bits  7 - 4
	if (family == 0xf) {
		family += (a >> 20) & 0x000000ff; // Extended family
		model  |= (a >> 12) & 0x000000f0; // Extended model
	}
	stepping= a & 0x0000000f;  // bits  3 - 0

	amd64_revision = amd64_get_revision(family, model, stepping);

	if (amd64_revision == AMD64_CPU_UN)
		return PFMLIB_ERR_NOTSUPP;

	__pfm_vbprintf("AMD family=%d model=0x%x stepping=0x%x rev=%s (%s)\n",
		       family,
		       model,
		       stepping,
		       amd64_rev_strs[amd64_revision],
		       amd64_cpu_strs[amd64_revision]);

	pfm_amd64_setup(amd64_revision);

	return PFMLIB_SUCCESS;
}

static int
is_valid_rev(int flags)
{
	if (flags & PFMLIB_AMD64_K8_REV_D
	   && amd64_revision < AMD64_K8_REV_D)
	   	return 0;

	if (flags & PFMLIB_AMD64_K8_REV_E
	   && amd64_revision < AMD64_K8_REV_E)
	   	return 0;

	if (flags & PFMLIB_AMD64_K8_REV_F
	   && amd64_revision < AMD64_K8_REV_F)
	   	return 0;

	/* no restrictions or matches restrictions */
	return 1;
}

static inline pme_amd64_entry_t
*pfm_amd64_get_event_entry(unsigned int i)
{
	unsigned int index = i;

	if (index >= amd64_event_count)
		return NULL;

	return &amd64_events[index];
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 */
static int
pfm_amd64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_amd64_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_amd64_input_param_t *param = mod_in;
	pfmlib_amd64_counter_t *cntrs;
	pfm_amd64_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, k, cnt, umask;
	unsigned int assign[PMU_AMD64_NUM_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_amd64_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, pfm_amd64_get_event_entry(e[j].event)->pme_name));
		}
	}

	if (cnt > PMU_AMD64_NUM_COUNTERS) return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++, i++) {
		/*
		 * AMD64 only supports two priv levels for perf counters
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT(("event=%d invalid plm=%d\n", e[j].event, e[j].plm));
			return PFMLIB_ERR_INVAL;
		}
		/*
		 * check illegal unit masks combination
		 */
		if (e[j].num_masks > 1 && PFMLIB_AMD64_HAS_COMBO(e[j].event) == 0) {
			DPRINT(("event does not supports unit mask combination\n"));
			return PFMLIB_ERR_FEATCOMB;
		}

		/*
		 * check revision restrictions at the event level
		 * (check at the umask level later)
		 */
		if (!is_valid_rev(pfm_amd64_get_event_entry(e[i].event)->pme_flags)) {
			DPRINT(("CPU does not have correct revision level\n"));
			return PFMLIB_ERR_BADHOST;
		}

		if (cntrs && (cntrs[j].cnt_mask >= PMU_AMD64_CNT_MASK_MAX)) {
			DPRINT(("event=%d invalid cnt_mask=%d: must be < %u\n",
				e[j].event,
				cntrs[j].cnt_mask,
				PMU_AMD64_CNT_MASK_MAX));
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * exclude unavailable registers from assignment
		 */
		while(i < PMU_AMD64_NUM_COUNTERS && pfm_regmask_isset(r_pmcs, i))
			i++;

		if (i == PMU_AMD64_NUM_COUNTERS)
			return PFMLIB_ERR_NOASSIGN;

		assign[j] = i;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		if (!is_valid_rev(pfm_amd64_get_event_entry(e[j].event)->pme_flags))
			return PFMLIB_ERR_BADHOST;
		reg.sel_event_mask  = pfm_amd64_get_event_entry(e[j].event)->pme_code;
		reg.sel_event_mask2 = pfm_amd64_get_event_entry(e[j].event)->pme_code >> 8;

		umask = 0;
		for(k=0; k < e[j].num_masks; k++) {
			/* check unit mask revision restrictions */
			if (!is_valid_rev(pfm_amd64_get_event_entry(e[j].event)->pme_umasks[e[j].unit_masks[k]].pme_uflags))
				return PFMLIB_ERR_BADHOST;

			umask |= pfm_amd64_get_event_entry(e[j].event)->pme_umasks[e[j].unit_masks[k]].pme_ucode;
		}
		reg.sel_unit_mask  = umask;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */
		if (cntrs) {
			reg.sel_cnt_mask = cntrs[j].cnt_mask;
			reg.sel_edge	 = cntrs[j].flags & PFM_AMD64_SEL_EDGE ? 1 : 0;
			reg.sel_inv	 = cntrs[j].flags & PFM_AMD64_SEL_INV ? 1 : 0;
			reg.sel_guest	 = cntrs[j].flags & PFM_AMD64_SEL_GUEST ? 1 : 0;
			reg.sel_host	 = cntrs[j].flags & PFM_AMD64_SEL_HOST ? 1 : 0;
		}
		pc[j].reg_num   = assign[j];
		if ((IS_FAM10H_ONLY(reg)) && !IS_FAMILY_10H())
			return PFMLIB_ERR_BADHOST;
		pc[j].reg_value = reg.val;
		pc[j].reg_addr	= AMD64_SEL_BASE+assign[j];
		pc[j].reg_alt_addr = AMD64_SEL_BASE+assign[j];


		pd[j].reg_num  = assign[j];
		pd[j].reg_addr = AMD64_CTR_BASE+assign[j];
		/* index to use with RDPMC */
		pd[j].reg_alt_addr = assign[j];

		__pfm_vbprintf("[PERFSEL%u(pmc%u)=0x%llx emask=0x%x umask=0x%x os=%d usr=%d inv=%d en=%d int=%d edge=%d cnt_mask=%d] %s\n",
			assign[j],
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
			pfm_amd64_get_event_entry(e[j].event)->pme_name);

		__pfm_vbprintf("[PERFCTR%u(pmd%u)]\n", pd[j].reg_num, pd[j].reg_num);
	}
	/* number of evtsel/ctr registers programmed */
	outp->pfp_pmc_count = cnt;
	outp->pfp_pmd_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_amd64_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_amd64_input_param_t *mod_in  = (pfmlib_amd64_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_amd64_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_amd64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > 3)
		return PFMLIB_ERR_INVAL;

	*code = pfm_amd64_get_event_entry(i)->pme_code;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_amd64_get_event_umask(unsigned int i, unsigned long *umask)
{
	if (i >= amd64_event_count || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0; //evt_umask(i);
	return PFMLIB_SUCCESS;
}
	
static void
pfm_amd64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < PMU_AMD64_NUM_COUNTERS; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_amd64_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	/* all pmcs are contiguous */
	for(i=0; i < PMU_AMD64_NUM_PERFSEL; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_amd64_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	/* all pmds are contiguous */
	for(i=0; i < PMU_AMD64_NUM_PERFCTR; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_amd64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	/* counting pmds are contiguous */
	for(i=0; i < 4; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_amd64_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_AMD64_COUNTER_WIDTH;
}

static char *
pfm_amd64_get_event_name(unsigned int i)
{
	return pfm_amd64_get_event_entry(i)->pme_name;
}

static int
pfm_amd64_get_event_desc(unsigned int ev, char **str)
{
	char *s;
	s = pfm_amd64_get_event_entry(ev)->pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static char *
pfm_amd64_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return pfm_amd64_get_event_entry(ev)->pme_umasks[midx].pme_uname;
}

static int
pfm_amd64_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = pfm_amd64_get_event_entry(ev)->pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_amd64_get_num_event_masks(unsigned int ev)
{
	return pfm_amd64_get_event_entry(ev)->pme_numasks;
}

static int
pfm_amd64_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code = pfm_amd64_get_event_entry(ev)->pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_amd64_get_cycle_event(pfmlib_event_t *e)
{
	e->event = amd64_cpu_clks;
	return PFMLIB_SUCCESS;

}

static int
pfm_amd64_get_inst_retired(pfmlib_event_t *e)
{
	e->event = amd64_ret_inst;
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t amd64_support = {
	.pmu_name		= "AMD64",
	.pmu_type		= PFMLIB_AMD64_PMU,
	.pme_count		= 0,
	.pmc_count		= PMU_AMD64_NUM_PERFSEL,
	.pmd_count		= PMU_AMD64_NUM_PERFCTR,
	.num_cnt		= PMU_AMD64_NUM_COUNTERS,
	.get_event_code		= pfm_amd64_get_event_code,
	.get_event_name		= pfm_amd64_get_event_name,
	.get_event_counters	= pfm_amd64_get_event_counters,
	.dispatch_events	= pfm_amd64_dispatch_events,
	.pmu_detect		= pfm_amd64_detect,
	.get_impl_pmcs		= pfm_amd64_get_impl_perfsel,
	.get_impl_pmds		= pfm_amd64_get_impl_perfctr,
	.get_impl_counters	= pfm_amd64_get_impl_counters,
	.get_hw_counter_width	= pfm_amd64_get_hw_counter_width,
	.get_event_desc         = pfm_amd64_get_event_desc,
	.get_num_event_masks	= pfm_amd64_get_num_event_masks,
	.get_event_mask_name	= pfm_amd64_get_event_mask_name,
	.get_event_mask_code	= pfm_amd64_get_event_mask_code,
	.get_event_mask_desc	= pfm_amd64_get_event_mask_desc,
	.get_cycle_event	= pfm_amd64_get_cycle_event,
	.get_inst_retired_event = pfm_amd64_get_inst_retired
};
