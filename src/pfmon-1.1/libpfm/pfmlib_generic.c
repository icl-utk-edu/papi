/*
 * pfmlib_generic.c : support default architected PMU features
 *
 * Copyright (C) 2001-2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */

#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"

#define PMU_GEN_MAX_COUNTERS	4

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
	unsigned long		pme_counters;	/* supported counters */
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
#ifdef NUE_HACK
	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL)
		fp = fopen("/tmp/pal/cpu0/perfmon_info", "r");
	else
#endif
	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL) return -1;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);

		if (p == NULL) break;
	       
		if ((p = strchr(buffer, ':')) == NULL) break;

		*p = '\0'; 

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
	return matches == 4 ? 0 : -1;
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

/*
 * Part of the following code will eventually go into a perfmon library
 */
static int
valid_assign(int *as, int cnt)
{
	int i;
	for(i=0; i < cnt; i++) if (as[i]==0) return 0;
	return 1;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_reg_t structure is ready to be submitted to kernel
 */
static int
pfm_gen_dispatch_counters(pfmlib_param_t *evt, pfarg_reg_t *pc)
{
	int i,j,k,l, m;
	unsigned int max_l0, max_l1, max_l2, max_l3;
	int assign[PMU_GEN_MAX_COUNTERS];
	pmu_reg_t reg;
	int *cnt_list = evt->pfp_evt;
	unsigned int cnt = evt->pfp_count;

#define	has_counter(e,b)	(generic_pe[e].pme_counters & (1 << (b)) ? (b) : 0)

	if (PFMLIB_DEBUG())
		for (m=0; m < cnt; m++) {
			printf("ev[%d]=%s counters=0x%lx\n", m, generic_pe[cnt_list[m]].pme_name, generic_pe[cnt_list[m]].pme_counters);
		}
	if (cnt > PMU_GEN_MAX_COUNTERS) return PFMLIB_ERR_TOOMANY;

	max_l0 = PMU_FIRST_COUNTER + PMU_GEN_MAX_COUNTERS;
	max_l1 = PMU_FIRST_COUNTER + PMU_GEN_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_FIRST_COUNTER + PMU_GEN_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_FIRST_COUNTER + PMU_GEN_MAX_COUNTERS*(cnt>3);

	if (PFMLIB_DEBUG()) {
		printf("max_l0=%u max_l1=%u max_l2=%u max_l3=%u\n", max_l0, max_l1, max_l2, max_l3);
	}
	/*
	 *  This code needs fixing. It is not very pretty and 
	 *  won't handle more than 4 counters if more become
	 *  available !
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=PMU_FIRST_COUNTER; i < max_l0; i++) {

		assign[0]= has_counter(cnt_list[0],i);

		if (max_l1 == PMU_FIRST_COUNTER && valid_assign(assign,cnt)) goto done;

		for (j=PMU_FIRST_COUNTER; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(cnt_list[1],j);

			if (max_l2 == PMU_FIRST_COUNTER && valid_assign(assign,cnt)) goto done;

			for (k=PMU_FIRST_COUNTER; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(cnt_list[2],k);

				if (max_l3 == PMU_FIRST_COUNTER && valid_assign(assign,cnt)) goto done;
				for (l=PMU_FIRST_COUNTER; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(cnt_list[3],l);

					if (valid_assign(assign,cnt)) goto done;
				}
			}
		}
	}
	/* we cannot satisfy the constraints */
	return PFMLIB_ERR_NOASSIGN;
done:
	/* cleanup the array */
	memset(pc, 0, cnt*sizeof(pfarg_reg_t));

	for (j=0; j < cnt ; j++ ) {
		reg.reg_val    = 0; /* clear all */
		/* if not specified per event, then use default (could be zero: measure nothing) */
		reg.pmc_plm    = evt->pfp_plm[j] ? evt->pfp_plm[j]: evt->pfp_dfl_plm; 
		reg.pmc_oi     = 1; /* overflow interrupt */
		reg.pmc_pm     = evt->pfp_pm; 
		reg.pmc_es     = generic_pe[cnt_list[j]].pme_gen_entry_code.pme_gen_code.pme_code;

		pc[j].reg_num   = assign[j]; 
		pc[j].reg_value = reg.reg_val;

		if (PFMLIB_DEBUG()) {
			printf("[cnt=0x%x,val=0x%lx,es=0x%x,plm=%d pm=%d] %s\n", 
					assign[j], reg.reg_val, 
					reg.pmc_es,reg.pmc_plm, 
					reg.pmc_pm,
					generic_pe[cnt_list[j]].pme_name);
		}
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_gen_dispatch_events(pfmlib_param_t *evt, pfarg_reg_t *pc, int *count)
{
	int ret;

	/* not enough slots in command */
	if (*count < evt->pfp_count) return -1;

	/* check for nothing to do */
	if (evt->pfp_count == 0) return 0;

	ret = pfm_gen_dispatch_counters(evt, pc);

	if (ret == PFMLIB_SUCCESS) *count = evt->pfp_count;

	return ret;
}

static unsigned long
pfm_gen_get_event_code(int i)
{
	return generic_pe[i].pme_gen_entry_code.pme_gen_code.pme_code;
}

static unsigned long
pfm_gen_get_event_vcode(int i)
{
	return generic_pe[i].pme_gen_entry_code.pme_vcode;
}

static char *
pfm_gen_get_event_name(int i)
{
	return generic_pe[i].pme_name;
}

static unsigned long
pfm_gen_get_event_counters(int i)
{
	return generic_pe[i].pme_counters;
}

static int
pfm_gen_num_counters(void)
{
	return PMU_GEN_MAX_COUNTERS;
}

pfm_pmu_support_t generic_support={
		"generic",
		PFMLIB_GENERIC_PMU,
		PME_GEN_COUNT,
		pfm_gen_get_event_code,
		pfm_gen_get_event_vcode,
		pfm_gen_get_event_name,
		pfm_gen_get_event_counters,
		NULL,
		pfm_gen_dispatch_events,
		pfm_gen_num_counters,
		pfm_gen_detect
};
