/*
 * pfmlib_coreduo.c : support for Intel Core Duo/Core Solo processors
 *		      (using architectural perfmon)
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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

/*
 * XXX: needs to be merged with i386_p6
 */
/* public headers */
#include <perfmon/pfmlib_coreduo.h>

/* private headers */
#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_coreduo_priv.h"	/* architecture private */
#include "coreduo_events.h"		/* PMU private */

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

static char * pfm_coreduo_get_event_name(unsigned int i);

#define PFMLIB_COREDUO_HAS_COMBO(_e) ((coreduo_pe[_e].pme_flags & PFMLIB_COREDUO_UMASK_COMBO) != 0)

#define PFMLIB_COREDUO_ALL_FLAGS \
	(PFM_COREDUO_SEL_INV|PFM_COREDUO_SEL_EDGE)

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

	return family != 6 || model != 14 ? PFMLIB_ERR_NOTSUPP : PFMLIB_SUCCESS;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 */
static int
pfm_coreduo_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_coreduo_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_coreduo_input_param_t *param = mod_in;
	pfmlib_coreduo_counter_t *cntrs;
	pfm_coreduo_perfevtsel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, k, cnt, umask;
	unsigned int assign[PMU_COREDUO_NUM_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_coreduo_counters : NULL;

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT(("ev[%d]=%s\n", j, coreduo_pe[e[j].event].pme_name));
		}
	}

	if (cnt > PMU_COREDUO_NUM_COUNTERS) return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++, i++) {
		/*
		 * COREDUO only supports two priv levels
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT(("event=%d invalid plm=%d\n", e[j].event, e[j].plm));
			return PFMLIB_ERR_INVAL;
		}

		if (e[j].flags & ~PFMLIB_COREDUO_ALL_FLAGS) {
			DPRINT(("event=%d invalid flags=0x%lx\n", e[j].event, e[j].flags));
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * check illegal unit masks combination
		 */
		if (e[j].num_masks > 1 && PFMLIB_COREDUO_HAS_COMBO(e[j].event) == 0) {
			DPRINT(("event does not supports unit mask combination\n"));
			return PFMLIB_ERR_FEATCOMB;
		}

		if (cntrs && (cntrs[j].cnt_mask >= PMU_COREDUO_CNT_MASK_MAX)) {
			DPRINT(("event=%d invalid cnt_mask=%d: must be < %u\n",
				e[j].event,
				cntrs[j].cnt_mask,
				PMU_COREDUO_CNT_MASK_MAX));
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * exclude unavailable registers from assignment
		 */
		while(i < PMU_COREDUO_NUM_COUNTERS && pfm_regmask_isset(r_pmcs, i))
			i++;

		if (i == PMU_COREDUO_NUM_COUNTERS)
			return PFMLIB_ERR_NOASSIGN;

		assign[j] = i;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		reg.sel_event_mask = coreduo_pe[e[j].event].pme_code;
		/*
		 * some events have only a single umask. We do not create
		 * specific umask entry in this case. The umask code is taken
		 * out of the (extended) event code (2nd byte)
		 */
		umask = (coreduo_pe[e[j].event].pme_code >> 8) & 0xff;

		for(k=0; k < e[j].num_masks; k++) {
			umask |= coreduo_pe[e[j].event].pme_umasks[e[j].unit_masks[k]].pme_ucode;
		}
		reg.sel_unit_mask  = umask;
		reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
		reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
		reg.sel_en         = 1; /* force enable bit to 1 */
		reg.sel_int        = 1; /* force APIC int to 1 */
		if (cntrs) {
			reg.sel_cnt_mask = cntrs[j].cnt_mask;
			reg.sel_edge	 = cntrs[j].flags & PFM_COREDUO_SEL_EDGE ? 1 : 0;
			reg.sel_inv	 = cntrs[j].flags & PFM_COREDUO_SEL_INV ? 1 : 0;
		}
		pc[j].reg_num = assign[j];
		/*
		 * XXX: assumes perfmon2 COREDUO mappings!
		 */
		pc[j].reg_pmd_num = assign[j];
		pc[j].reg_evt_idx = j;
		pc[j].reg_value   = reg.val;

		__pfm_vbprintf("[perfevtsel%u=0x%llx emask=0x%lx umask=0x%lx os=%d usr=%d inv=%d en=%d int=%d edge=%d cnt_mask=%d] %s\n",
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
			coreduo_pe[e[j].event].pme_name);
	}
	/* number of evtsel registers programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_coreduo_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	pfmlib_coreduo_input_param_t *mod_in  = (pfmlib_coreduo_input_param_t *)model_in;

	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT(("invalid plm=%x\n", inp->pfp_dfl_plm));
		return PFMLIB_ERR_INVAL;
	}
	return pfm_coreduo_dispatch_counters(inp, mod_in, outp);
}

static int
pfm_coreduo_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt > 3)
		return PFMLIB_ERR_INVAL;

	/*
	 * we return the full value.
	 * Event with a single umask do not have explicit umask
	 * table. In this case, the unit mask value if merged with the
	 * event code value. So this function may return more than just
	 * the plain event code.
	 */
	*code = coreduo_pe[i].pme_code;

	return PFMLIB_SUCCESS;
}

static void
pfm_coreduo_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < PMU_COREDUO_NUM_COUNTERS; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_coreduo_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* all pmcs are contiguous */
	for(i=0; i < PMU_COREDUO_NUM_PERFSEL; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_coreduo_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* all pmds are contiguous */
	for(i=0; i < PMU_COREDUO_NUM_PERFCTR; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_coreduo_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* counting pmds are contiguous */
	for(i=0; i < 4; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_coreduo_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_COREDUO_COUNTER_WIDTH;
}

static char *
pfm_coreduo_get_event_name(unsigned int i)
{
	return coreduo_pe[i].pme_name;
}

static int
pfm_coreduo_get_event_desc(unsigned int ev, char **str)
{
	char *s;

	s = coreduo_pe[ev].pme_desc;
	if (s)
		*str = strdup(s);
	else
		*str = NULL;
	return PFMLIB_SUCCESS;
}

static char *
pfm_coreduo_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	return coreduo_pe[ev].pme_umasks[midx].pme_uname;
}

static int
pfm_coreduo_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = coreduo_pe[ev].pme_umasks[midx].pme_udesc;
	if (s)
		*str = strdup(s);
	else
		*str = NULL;
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_coreduo_get_num_event_masks(unsigned int ev)
{
	return coreduo_pe[ev].pme_numasks;
}

static int
pfm_coreduo_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code = coreduo_pe[ev].pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t coreduo_support={
	.pmu_name		= "Intel Core Duo/Core Solo",
	.pmu_type		= PFMLIB_COREDUO_PMU,
	.pme_count		= PME_COREDUO_EVENT_COUNT,
	.pmc_count		= PMU_COREDUO_NUM_PERFSEL,
	.pmd_count		= PMU_COREDUO_NUM_PERFCTR,
	.num_cnt		= PMU_COREDUO_NUM_COUNTERS,
	.cycle_event		= PME_COREDUO_CPU_CLK_UNHALTED,
	.inst_retired_event	= PME_COREDUO_RETIRED_INSTRUCTIONS,
	.get_event_code		= pfm_coreduo_get_event_code,
	.get_event_name		= pfm_coreduo_get_event_name,
	.get_event_counters	= pfm_coreduo_get_event_counters,
	.dispatch_events	= pfm_coreduo_dispatch_events,
	.pmu_detect		= pfm_coreduo_detect,
	.get_impl_pmcs		= pfm_coreduo_get_impl_perfsel,
	.get_impl_pmds		= pfm_coreduo_get_impl_perfctr,
	.get_impl_counters	= pfm_coreduo_get_impl_counters,
	.get_hw_counter_width	= pfm_coreduo_get_hw_counter_width,
	.get_event_desc         = pfm_coreduo_get_event_desc,
	.get_num_event_masks	= pfm_coreduo_get_num_event_masks,
	.get_event_mask_name	= pfm_coreduo_get_event_mask_name,
	.get_event_mask_code	= pfm_coreduo_get_event_mask_code,
	.get_event_mask_desc	= pfm_coreduo_get_event_mask_desc
};
