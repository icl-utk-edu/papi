/*
 * pfmlib_generic_ia64.c : support default architected IA-64 PMU features
 *
 * Copyright (C) 2001-2003 Hewlett-Packard Co
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
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
 */

#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_generic_ia64.h>

#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_priv_ia64.h"		/* architecture private */

#define PMU_GEN_IA64_MAX_COUNTERS	4

/*
 * number of events by default
 */
#define PME_GEN_COUNT	2

/*
 * generic event as described by architecture
 */
typedef	struct {
	unsigned long pme_code:8;	/* major event code */
	unsigned long pme_ig:56;	/* ignored */
} pme_gen_code_t;

/*
 *  union of all possible entry codes. All encodings must fit in 64bit
 */
typedef union {
	unsigned long  pme_vcode;
	pme_gen_code_t pme_gen_code;
} pme_gen_entry_code_t;

/*
 * entry in the event table (one table per implementation)
 */
typedef struct pme_entry {
	char		 	*pme_name;
	pme_gen_entry_code_t	pme_gen_entry_code;	/* event code */
	unsigned long		pme_counters;		/* supported counters */
} pme_gen_entry_t;

/* let's define some handy shortcuts ! */
#define pmc_plm		pmc_gen_count_reg.pmc_plm
#define pmc_ev		pmc_gen_count_reg.pmc_ev
#define pmc_oi		pmc_gen_count_reg.pmc_oi
#define pmc_pm		pmc_gen_count_reg.pmc_pm
#define pmc_es		pmc_gen_count_reg.pmc_es



/*
 * this table is patched by initialization code
 */
static pme_gen_entry_t generic_pe[PME_GEN_COUNT]={
	{ "CPU_CYCLES", {0}, 0},
	{ "IA64_INST_RETIRED", {0}, 0},
};

static int pfm_gen_counter_width;	/* probed via /proc/pal */

static int
parse_counter_range(char *range, unsigned long *bitmask)
{
	char *p;
	int start, end;

	p = strchr(range, '-');

	start = atoi(range);

	if (start >= 64) {
		printf("%s.%s : bitmask too small need %d bits\n", __FILE__, __FUNCTION__, start);
		return -1;
	}
	*bitmask = 1UL << start;
	if (p == NULL) return 0;

	end = atoi(p+1);

	if (end >= 64) {
		printf("%s.%s: bitmask too small need %d bits\n", __FILE__, __FUNCTION__, end);
		return -1;
	}
	*bitmask = (~((1UL << start) -1)) & ((1UL << end) | ((1UL << end)-1));

	return 0;
}

static int
pfm_gen_initialize(void)
{
	FILE *fp;	
	char *p;
	char buffer[64];
	int matches = 0;

	/*
	 * check that the core library supports enough registers
	 */
	if (PFMLIB_MAX_PMCS < PMU_GEN_IA64_MAX_COUNTERS) return PFMLIB_ERR_NOTSUPP;
	if (PFMLIB_MAX_PMDS < PMU_GEN_IA64_MAX_COUNTERS) return PFMLIB_ERR_NOTSUPP;

	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL) return PFMLIB_ERR_NOTSUPP;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);

		if (p == NULL) break;

		if ((p = strchr(buffer, ':')) == NULL) break;

		*p = '\0';

		if (!strncmp("Counter width", buffer, 13)) {
			pfm_gen_counter_width = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Cycle event number", buffer, 18)) {
			generic_pe[0].pme_gen_entry_code.pme_vcode = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Retired event number", buffer, 20)) {
			generic_pe[1].pme_gen_entry_code.pme_vcode = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Cycles count capable", buffer, 20)) {
			if (parse_counter_range(p+1, &generic_pe[0].pme_counters) == -1) return -1;
			matches++;
			continue;
		}
		if (!strncmp("Retired bundles count capable", buffer, 29)) {
			if (parse_counter_range(p+1, &generic_pe[1].pme_counters) == -1) return -1;
			matches++;
			continue;
		}
	}
	fclose(fp);
	return matches == 5 ? PFMLIB_SUCCESS : PFMLIB_ERR_NOTSUPP;
}

static int
pfm_gen_detect(void)
{
	static int initialization_done;

	if (initialization_done) return 0;

	/* always match */
	if (pfm_gen_initialize() == -1) return PFMLIB_ERR_NOTSUPP;

	initialization_done = 1;

	return PFMLIB_SUCCESS;
}

static int
valid_assign(unsigned int *as, unsigned int cnt)
{
	unsigned int i;
	for(i=0; i < cnt; i++) if (as[i]==0) return 0;
	return 1;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_reg_t structure is ready to be submitted to kernel
 */
static int
pfm_gen_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_output_param_t *outp)
{
	unsigned int max_l0, max_l1, max_l2, max_l3;
	unsigned int assign[PMU_GEN_IA64_MAX_COUNTERS];
	pfm_gen_ia64_pmc_reg_t reg;
	pfmlib_event_t *e= inp->pfp_events;
	pfmlib_pmc_t *pc  = outp->pfp_pmcs;
	unsigned int i,j,k,l;
	unsigned int cnt = inp->pfp_event_count;

#define	has_counter(e,b)	(generic_pe[e].pme_counters & (1 << (b)) ? (b) : 0)

	if (PFMLIB_DEBUG()) {
		for (i=0; i < cnt; i++) {
			DPRINT(("ev[%d]=%s counters=0x%lx\n",
				i,
				generic_pe[e[i].event].pme_name,
				generic_pe[e[i].event].pme_counters));
		}
	}

	if (cnt > PMU_GEN_IA64_MAX_COUNTERS) return PFMLIB_ERR_TOOMANY;

	max_l0 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS;
	max_l1 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>3);

	if (PFMLIB_DEBUG()) {
		printf("max_l0=%u max_l1=%u max_l2=%u max_l3=%u\n", max_l0, max_l1, max_l2, max_l3);
	}
	/*
	 *  This code needs fixing. It is not very pretty and
	 *  won't handle more than 4 counters if more become
	 *  available !
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=PMU_GEN_IA64_FIRST_COUNTER; i < max_l0; i++) {

		assign[0]= has_counter(e[0].event,i);

		if (max_l1 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, cnt)) goto done;

		for (j=PMU_GEN_IA64_FIRST_COUNTER; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(e[1].event,j);

			if (max_l2 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, cnt)) goto done;

			for (k=PMU_GEN_IA64_FIRST_COUNTER; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(e[2].event,k);

				if (max_l3 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, cnt)) goto done;
				for (l=PMU_GEN_IA64_FIRST_COUNTER; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(e[3].event,l);

					if (valid_assign(assign, cnt)) goto done;
				}
			}
		}
	}
	/* we cannot satisfy the constraints */
	return PFMLIB_ERR_NOASSIGN;
done:

	for (j=0; j < cnt ; j++ ) {
		reg.pmc_val    = 0; /* clear all */
		/* if not specified per event, then use default (could be zero: measure nothing) */
		reg.pmc_plm    = e[j].plm ? e[j].plm: inp->pfp_dfl_plm;
		reg.pmc_oi     = 1; /* overflow interrupt */
		reg.pmc_pm     = inp->pfp_flags & PFMLIB_PFP_SYSTEMWIDE? 1 : 0;
		reg.pmc_es     = generic_pe[e[j].event].pme_gen_entry_code.pme_gen_code.pme_code;

		pc[j].reg_num   = assign[j];
		pc[j].reg_value = reg.pmc_val;

		if (PFMLIB_DEBUG()) {
			DPRINT(("[pmc%u=0x%lx,es=0x%02x,plm=%d pm=%d] %s\n",
					assign[j], reg.pmc_val,
					reg.pmc_es,reg.pmc_plm,
					reg.pmc_pm,
					generic_pe[e[j].event].pme_name));
		}
	}
	/* number of PMC programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_dispatch_events(pfmlib_input_param_t *inp, void *dummy1, pfmlib_output_param_t *outp, void *dummy2)
{
	return pfm_gen_dispatch_counters(inp, outp);
}

static int
pfm_gen_get_event_code(unsigned int i)
{
	return generic_pe[i].pme_gen_entry_code.pme_gen_code.pme_code;
}

static unsigned long
pfm_gen_get_event_vcode(unsigned int i)
{
	return generic_pe[i].pme_gen_entry_code.pme_vcode;
}

static char *
pfm_gen_get_event_name(unsigned int i)
{
	return generic_pe[i].pme_name;
}

static void
pfm_gen_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;
	unsigned long m;

	memset(counters, 0, sizeof(*counters));

	m = generic_pe[j].pme_counters;
	for(i=0; m ; i++, m>>=1) {
		if (m & 0x1) PFMLIB_REGMASK_SET(counters, i);
	}
}

static void
pfm_gen_num_counters(unsigned int *num)
{
	*num = PMU_GEN_IA64_MAX_COUNTERS;
}

static void
pfm_gen_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	/* pmc0-pmc7 */
	for(i=0; i < 8; i++) PFMLIB_REGMASK_SET(impl_pmcs, i);
}

static void
pfm_gen_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	/* pmd4-pmd7 */
	for(i=4; i < 8; i++) PFMLIB_REGMASK_SET(impl_pmds, i);
}

static void
pfm_gen_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* pmd4-pmd7 */
	for(i=4; i < 8; i++) PFMLIB_REGMASK_SET(impl_counters, i);
}

static void
pfm_gen_get_hw_counter_width(unsigned int *width)
{
	*width = pfm_gen_counter_width;
}

static void
pfm_gen_num_pmcs(unsigned int *num)
{
	*num = 4 + 4;
}

static void
pfm_gen_num_pmds(unsigned int *num)
{
	*num = PMU_GEN_IA64_MAX_COUNTERS;
}


pfm_pmu_support_t generic_ia64_support={
	.pmu_name		="generic IA-64",
	.pmu_type		= PFMLIB_GENERIC_IA64_PMU,
	.pme_count		= PME_GEN_COUNT,
	.get_event_code		= pfm_gen_get_event_code,
	.get_event_vcode	= pfm_gen_get_event_vcode,
	.get_event_name		= pfm_gen_get_event_name,
	.get_event_counters	= pfm_gen_get_event_counters,
	.dispatch_events	= pfm_gen_dispatch_events,
	.get_num_counters	= pfm_gen_num_counters,
	.get_num_pmcs		= pfm_gen_num_pmcs,
	.get_num_pmds		= pfm_gen_num_pmds,
	.pmu_detect		= pfm_gen_detect,
	.get_impl_pmcs		= pfm_gen_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_get_impl_pmds,
	.get_impl_counters	= pfm_gen_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_get_hw_counter_width
};
