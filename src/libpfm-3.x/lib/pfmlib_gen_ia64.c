/*
 * pfmlib_gen_ia64.c : support default architected IA-64 PMU features
 *
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_gen_ia64.h>

#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_priv_ia64.h"		/* architecture private */

#define PMU_GEN_IA64_MAX_COUNTERS	4
/*
 * number of architected events
 */
#define PME_GEN_COUNT	2

/*
 * generic event as described by architecture
 */
typedef	struct {
	unsigned long pme_code:8;	/* major event code */
	unsigned long pme_ig:56;	/* ignored */
} pme_gen_ia64_code_t;

/*
 *  union of all possible entry codes. All encodings must fit in 64bit
 */
typedef union {
	unsigned long  	    pme_vcode;
	pme_gen_ia64_code_t pme_gen_code;
} pme_gen_ia64_entry_code_t;

/*
 * entry in the event table (one table per implementation)
 */
typedef struct pme_entry {
	char		 		*pme_name;
	pme_gen_ia64_entry_code_t	pme_entry_code;	/* event code */
	pfmlib_regmask_bits_t		pme_counters[PFMLIB_REG_BV]; /* counter bitmask */
} pme_gen_ia64_entry_t;

/* let's define some handy shortcuts ! */
#define pmc_plm		pmc_gen_count_reg.pmc_plm
#define pmc_ev		pmc_gen_count_reg.pmc_ev
#define pmc_oi		pmc_gen_count_reg.pmc_oi
#define pmc_pm		pmc_gen_count_reg.pmc_pm
#define pmc_es		pmc_gen_count_reg.pmc_es

/*
 * this table is patched by initialization code
 */
static pme_gen_ia64_entry_t generic_pe[PME_GEN_COUNT]={
#define PME_IA64_GEN_CPU_CYCLES	0
	{ "CPU_CYCLES", },
#define PME_IA64_GEN_INST_RETIRED 1
	{ "IA64_INST_RETIRED", },
};

static int pfm_gen_ia64_counter_width;
static int pfm_gen_ia64_counters;
static pfmlib_regmask_bits_t pfm_gen_ia64_impl_pmcs[PFMLIB_REG_BV];
static pfmlib_regmask_bits_t pfm_gen_ia64_impl_pmds[PFMLIB_REG_BV];

/*
 * convert text range (e.g. 4-15 18 12-26) into actual bitmask
 * range argument is modified
 */
static int
parse_counter_range(char *range, unsigned long *bitmask)
{
	char *p, c;
	int start, end;

	if (range[strlen(range)-1] == '\n')
		range[strlen(range)-1] = '\0';

	/*
	 * reset bitmask argument
	 */
	for (start=0; start < PFMLIB_REG_BV; start++)
		bitmask[start] = 0;

	while(range) {
		p = range;
		while (*p && *p != ' ' && *p != '-') p++;

		if (*p == '\0') break;

		c = *p;
		*p = '\0';
		start = atoi(range);
		range = p+1;

		if (c == '-') {
			p++;
			while (*p && *p != ' ' && *p != '-') p++;
			if (*p) *p++ = '\0';
			end = atoi(range);
			range = p;
		} else {
			end = start;
		}

		if (end  >= PFMLIB_REG_MAX|| start >= PFMLIB_REG_MAX)
			goto invalid;
		for (; start <= end; start++)
			bitmask[__PFMLIB_REGMASK_EL(start)] |=  __PFMLIB_REGMASK_MASK(start);
	}
	return 0;
invalid:
	fprintf(stderr, "%s.%s : bitmask too small need %d bits\n", __FILE__, __FUNCTION__, start);
	return -1;
}

static int
pfm_gen_ia64_initialize(void)
{
	FILE *fp;	
	char *p;
	char buffer[64];
	int matches = 0;

	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL) return PFMLIB_ERR_NOTSUPP;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);

		if (p == NULL) break;

		if ((p = strchr(buffer, ':')) == NULL) break;

		*p = '\0';

		if (!strncmp("Counter width", buffer, 13)) {
			pfm_gen_ia64_counter_width = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("PMC/PMD pairs", buffer, 13)) {
			pfm_gen_ia64_counters = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Cycle event number", buffer, 18)) {
			generic_pe[0].pme_entry_code.pme_vcode = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Retired event number", buffer, 20)) {
			generic_pe[1].pme_entry_code.pme_vcode = atoi(p+2);
			matches++;
			continue;
		}
		if (!strncmp("Cycles count capable", buffer, 20)) {
			if (parse_counter_range(p+2, generic_pe[0].pme_counters) == -1) return -1;
			matches++;
			continue;
		}
		if (!strncmp("Retired bundles count capable", buffer, 29)) {
			if (parse_counter_range(p+2, generic_pe[1].pme_counters) == -1) return -1;
			matches++;
			continue;
		}
		if (!strncmp("Implemented PMC", buffer, 15)) {
			if (parse_counter_range(p+2, pfm_gen_ia64_impl_pmcs) == -1) return -1;
			matches++;
			continue;
		}
		if (!strncmp("Implemented PMD", buffer, 15)) {
			if (parse_counter_range(p+2, pfm_gen_ia64_impl_pmds) == -1) return -1;
			matches++;
			continue;
		}
	}

	fclose(fp);
	return matches == 8 ? PFMLIB_SUCCESS : PFMLIB_ERR_NOTSUPP;
}

static int
pfm_gen_ia64_detect(void)
{
	static int initialization_done;

	if (initialization_done) return 0;

	/* always match */
	if (pfm_gen_ia64_initialize() == -1) return PFMLIB_ERR_NOTSUPP;

	initialization_done = 1;

	return PFMLIB_SUCCESS;
}

static int
valid_assign(unsigned int *as, pfmlib_regmask_t *r_pmcs, unsigned int cnt)
{
	unsigned int i;
	for(i=0; i < cnt; i++) {
		if (as[i]==0) return 0;
		/*
		 * take care of restricted PMC registers
		 */
		if (pfm_regmask_isset(r_pmcs, as[i]))
			return 0;
	}
	return 1;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_reg_t structure is ready to be submitted to kernel
 */
static int
pfm_gen_ia64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_output_param_t *outp)
{
#define	has_counter(e,b)	(generic_pe[e].pme_counters[__PFMLIB_REGMASK_EL(b)] & __PFMLIB_REGMASK_MASK(b) ? (b) : 0)
	unsigned int max_l0, max_l1, max_l2, max_l3;
	unsigned int assign[PMU_GEN_IA64_MAX_COUNTERS];
	pfm_gen_ia64_pmc_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_pmc_t *pc;
	pfmlib_regmask_t *r_pmcs;
	unsigned int i,j,k,l;
	unsigned int cnt;

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;

	if (PFMLIB_DEBUG()) {
		for (i=0; i < cnt; i++) {
			DPRINT(("ev[%d]=%s counters=0x%lx\n",
				i,
				generic_pe[e[i].event].pme_name,
				generic_pe[e[i].event].pme_counters[0]));
		}
	}

	if (cnt > PMU_GEN_IA64_MAX_COUNTERS) return PFMLIB_ERR_TOOMANY;

	max_l0 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS;
	max_l1 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_GEN_IA64_FIRST_COUNTER + PMU_GEN_IA64_MAX_COUNTERS*(cnt>3);

	if (PFMLIB_DEBUG()) {
		DPRINT(("max_l0=%u max_l1=%u max_l2=%u max_l3=%u\n", max_l0, max_l1, max_l2, max_l3));
	}
	/*
	 *  This code needs fixing. It is not very pretty and
	 *  won't handle more than 4 counters if more become
	 *  available !
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=PMU_GEN_IA64_FIRST_COUNTER; i < max_l0; i++) {

		assign[0]= has_counter(e[0].event,i);

		if (max_l1 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, r_pmcs, cnt)) goto done;

		for (j=PMU_GEN_IA64_FIRST_COUNTER; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(e[1].event,j);

			if (max_l2 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, r_pmcs, cnt)) goto done;

			for (k=PMU_GEN_IA64_FIRST_COUNTER; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(e[2].event,k);

				if (max_l3 == PMU_GEN_IA64_FIRST_COUNTER && valid_assign(assign, r_pmcs, cnt)) goto done;

				for (l=PMU_GEN_IA64_FIRST_COUNTER; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(e[3].event,l);

					if (valid_assign(assign, r_pmcs, cnt)) goto done;
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
		reg.pmc_es     = generic_pe[e[j].event].pme_entry_code.pme_gen_code.pme_code;

		pc[j].reg_num     = assign[j];
		pc[j].reg_pmd_num = assign[j];
		pc[j].reg_evt_idx = j;
		pc[j].reg_value   = reg.pmc_val;

		__pfm_vbprintf("[pmc%u=0x%lx,es=0x%02x,plm=%d pm=%d] %s\n",
				assign[j], reg.pmc_val,
				reg.pmc_es,reg.pmc_plm,
				reg.pmc_pm,
				generic_pe[e[j].event].pme_name);
	}
	/* number of PMC programmed */
	outp->pfp_pmc_count = cnt;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_ia64_dispatch_events(pfmlib_input_param_t *inp, void *dummy1, pfmlib_output_param_t *outp, void *dummy2)
{
	return pfm_gen_ia64_dispatch_counters(inp, outp);
}

static int
pfm_gen_ia64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && (cnt < 4 || cnt > 7))
		return PFMLIB_ERR_INVAL;

	*code = generic_pe[i].pme_entry_code.pme_gen_code.pme_code;

	return PFMLIB_SUCCESS;
}

static char *
pfm_gen_ia64_get_event_name(unsigned int i)
{
	return generic_pe[i].pme_name;
}

static void
pfm_gen_ia64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;
	unsigned long m;

	memset(counters, 0, sizeof(*counters));

	m = generic_pe[j].pme_counters[0];
	for(i=0; m ; i++, m>>=1) {
		if (m & 0x1) pfm_regmask_set(counters, i);
	}
}

static void
pfm_gen_ia64_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	memcpy(impl_pmcs->bits, pfm_gen_ia64_impl_pmcs, sizeof(impl_pmcs->bits));
}

static void
pfm_gen_ia64_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	memcpy(impl_pmds->bits, pfm_gen_ia64_impl_pmds, sizeof(impl_pmds->bits));
}

static void
pfm_gen_ia64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	memset(impl_counters, 0, sizeof(*impl_counters));

	/* pmd4-pmd7 */
	for(i=4; i < 8; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_gen_ia64_get_hw_counter_width(unsigned int *width)
{
	*width = pfm_gen_ia64_counter_width;
}

static int
pfm_gen_ia64_get_event_desc(unsigned int ev, char **str)
{
	switch(ev) {
		case PME_IA64_GEN_CPU_CYCLES:
				*str = strdup("CPU cycles");
				break;
		case PME_IA64_GEN_INST_RETIRED:
				*str = strdup("IA-64 instructions retired");
				break;
		default:
				*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t generic_ia64_support={
	.pmu_name		="IA-64",
	.pmu_type		= PFMLIB_GEN_IA64_PMU,
	.pme_count		= PME_GEN_COUNT,
	.pmc_count		= 4+4,
	.pmd_count		= PMU_GEN_IA64_MAX_COUNTERS,
	.num_cnt		= PMU_GEN_IA64_MAX_COUNTERS,
	.cycle_event		= PME_IA64_GEN_CPU_CYCLES,
	.inst_retired_event	= PME_IA64_GEN_INST_RETIRED,
	.get_event_code		= pfm_gen_ia64_get_event_code,
	.get_event_name		= pfm_gen_ia64_get_event_name,
	.get_event_counters	= pfm_gen_ia64_get_event_counters,
	.dispatch_events	= pfm_gen_ia64_dispatch_events,
	.pmu_detect		= pfm_gen_ia64_detect,
	.get_impl_pmcs		= pfm_gen_ia64_get_impl_pmcs,
	.get_impl_pmds		= pfm_gen_ia64_get_impl_pmds,
	.get_impl_counters	= pfm_gen_ia64_get_impl_counters,
	.get_hw_counter_width	= pfm_gen_ia64_get_hw_counter_width,
	.get_event_desc		= pfm_gen_ia64_get_event_desc
};
