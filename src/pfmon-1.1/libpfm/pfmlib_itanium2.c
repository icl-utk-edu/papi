/*
 * pfmlib_itanium2.c : support for the Itanium2 PMU family
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* public headers */
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium2.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_itanium2_priv.h"
#include "itanium2_events.h"

#define is_ear(i)	event_is_ear(itanium2_pe+(i))
#define is_ear_tlb(i)	event_is_ear_tlb(itanium2_pe+(i))
#define is_ear_alat(i)	event_is_ear_alat(itanium2_pe+(i))
#define is_ear_cache(i)	event_is_ear_cache(itanium2_pe+(i))
#define is_iear(i)	event_is_iear(itanium2_pe+(i))
#define is_dear(i)	event_is_dear(itanium2_pe+(i))
#define is_btb(i)	event_is_btb(itanium2_pe+(i))
#define has_opcm(i)	event_opcm_ok(itanium2_pe+(i))
#define has_iarr(i)	event_iarr_ok(itanium2_pe+(i))
#define has_darr(i)	event_darr_ok(itanium2_pe+(i))

#define evt_use_opcm(e)		((e)->pfp_ita2_pmc8.opcm_used != 0 || (e)->pfp_ita2_pmc9.opcm_used !=0)
#define evt_use_irange(e)	((e)->pfp_ita2_irange.rr_used)
#define evt_use_drange(e)	((e)->pfp_ita2_drange.rr_used)

#define evt_grp(e)	itanium2_pe[e].pme_qualifiers.pme_qual.pme_group
#define evt_set(e)	itanium2_pe[e].pme_qualifiers.pme_qual.pme_set
#define evt_umask(e)	itanium2_pe[e].pme_umask


#define FINE_MODE_BOUNDARY_BITS	12
#define FINE_MODE_MASK	~((1UL<<12)-1)

static char * pfm_ita2_get_event_name(int i);
/* 
 * The Itanium2 PMU has a bug in the fine mode implementation. 
 * It only sees ranges with a granularity of two bundles. 
 * So we prepare for the day they fix it.
 */
static int has_fine_mode_bug;

/*
 * find last bit set
 */
#ifdef __GNUC__
static inline int
ia64_fls (unsigned long x)
{
	double d = x;
	long exp;

	__asm__ ("getf.exp %0=%1" : "=r"(exp) : "f"(d));
	return exp - 0xffff;

}
#else
static inline int
ia64_fls (unsigned long x)
{
	int i = 63;
	unsigned long m = 1UL<<63;

	for (i=63; i > -1; i++, m>>=1) {
		if (x & m) return i;
	}
	return -1;
}
#endif


static int
pfm_ita2_detect(void)
{
	int tmp; 
	int ret = PFMLIB_ERR_NOTSUPP;

	tmp = pfm_get_cpu_family();
	if (tmp == 0x1f) {
		has_fine_mode_bug = 1;
		ret = PFMLIB_SUCCESS;
	}
	return ret;
}

/*
 * Check the event for incompatibilities. This is useful
 * for L1 and L2 related events. Due to wire limitations,
 * some caches events are separated into sets. There
 * are 5 sets for the L1D cache group and 6 sets for L2 group.
 * It is NOT possible to simultaneously measure events from 
 * differents sets within a group. For instance, you cannot
 * measure events from set0 and set1 in L1D cache group. However
 * it is possible to measure set0 in L1D and set1 in L2 at the same
 * time. 
 *
 * This function verifies that the set constraint are respected.
 */
static int
check_cross_groups_and_umasks(pfmlib_param_t *evt)
{
	unsigned long ref_umask, umask;
	int g, s;
	unsigned int cnt = evt->pfp_count;
	int *cnt_list = evt->pfp_evt;
	int i, j;

	/*
	 * XXX: could possibly be optimized
	 */
	for (i=0; i < cnt; i++) {
		g = evt_grp(cnt_list[i]);
		s = evt_set(cnt_list[i]);

		if (g == PFMLIB_ITA2_EVT_NO_GRP) continue;

		ref_umask = evt_umask(cnt_list[i]);

		for (j=i+1; j < cnt; j++) {
			if (evt_grp(cnt_list[j]) != g) continue;
			if (evt_set(cnt_list[j]) != s) return PFMLIB_ERR_EVTSET;
			
			/* only care about L2 cache group */
			if (g != PFMLIB_ITA2_EVT_L2_CACHE_GRP || (s == 1 || s == 2)) continue;

			umask = evt_umask(cnt_list[j]); 
			/*
			 * there is no assignement possible if the event in PMC4
			 * has a umask (ref_umask) and an event (from the same
			 * set) also has a umask AND it is different. For some
			 * sets, the umasks are shared, therefore the value 
			 * programmed into PMC4 determines the umask for all
			 * the other events (with umask) from the set.
			 */
			if (umask && ref_umask != umask) return PFMLIB_ERR_NOASSIGN;

		}
	}
	return PFMLIB_SUCCESS;
}

/*
 * Certain prefetch events must be treated specially when instruction range restriction
 * is in use because they can only be constrained by IBRP1 in fine-mode. Other events
 * will use IBRP0 if tagged as a demand fetch OR IBPR1 if tagged as a prefetch match.
 * From the library's point of view there is no way of distinguishing this, so we leave
 * it up to the user to interpret the results.
 *
 * Events which can be qualified by the two pairs depending on their tag:
 * 	- IBP_BUNPAIRS_IN
 * 	- L1I_FETCH_RAB_HIT
 *	- L1I_FETCH_ISB_HIT
 * 	- L1I_FILLS
 *
 * This function returns the number of qualifying prefetch events found
 *
 * XXX: not clear which events do qualify as prefetch events.
 */
static int prefetch_events[]={
	PME_ITA2_L1I_PREFETCHES,
	PME_ITA2_L1I_STRM_PREFETCHES,
	PME_ITA2_L2_INST_PREFETCHES
};
#define NPREFETCH_EVENTS	sizeof(prefetch_events)/sizeof(int)

static int
check_prefetch_events(pfmlib_param_t *evt)
{
	int code;
	int prefetch_codes[NPREFETCH_EVENTS];
	int i, j, c, found = 0;

	for(i=0; i < NPREFETCH_EVENTS; i++) {
		pfm_get_event_code(prefetch_events[i], &code);
		prefetch_codes[i] = code;
	}

	for(i=0; i < evt->pfp_count; i++) {
		pfm_get_event_code(evt->pfp_evt[i], &c);
		for(j=0; j < NPREFETCH_EVENTS; j++) {
			if (c == prefetch_codes[j]) found++;
		}
	}
	return found;
}


/*
 * IA64_INST_RETIRED (and subevents) is the only event which can be measured on all
 * 4 IBR when non-fine mode is not possible.
 *
 * This function returns:
 * 	the number of events match the IA64_INST_RETIRED code
 */
static int
check_inst_retired_events(pfmlib_param_t *evt)
{
	int code;
	int i, c, found = 0;

	pfm_get_event_code(PME_ITA2_IA64_INST_RETIRED_THIS, &code);

	for(i=0; i < evt->pfp_count; i++) {
		pfm_get_event_code(evt->pfp_evt[i], &c);
		if (c == code)  found++;
	}
	return found;
}

static int
check_fine_mode_possible(pfmlib_ita2_rr_t *rr, int n)
{
	pfmlib_ita2_rr_desc_t *lim = rr->rr_limits;
	int i;

	for(i=0; i < n; i++) {
		if ((lim[i].rr_start & FINE_MODE_MASK) != (lim[i].rr_end & FINE_MODE_MASK)) 
			return 0;
	}
	return 1;
}

/*
 * mode = 0 -> check code (enforce bundle alignment)
 * mode = 1 -> check data
 */
static int
check_intervals(pfmlib_ita2_rr_t *rr, int mode, int *n_intervals)
{
	int i;
	pfmlib_ita2_rr_desc_t *lim = rr->rr_limits;

	for(i=0; i < 4; i++) {
		/* end marker */
		if (lim[i].rr_start == 0 && lim[i].rr_end == 0) break;

		/* invalid entry */
		if (lim[i].rr_start >= lim[i].rr_end) return PFMLIB_ERR_IRRINVAL;

		if (mode == 0 && (lim[i].rr_start & 0xf || lim[i].rr_end & 0xf)) 
			return PFMLIB_ERR_IRRALIGN;
	}
	*n_intervals = i;
	return PFMLIB_SUCCESS;
}


static int
valid_assign(int *cnt_list, int *as, int cnt)
{
	unsigned long pmc4_umask = 0, umask;
	char *name;
	int l1_grp_present = 0, l2_grp_present = 0;
	int i, failure;
	int e;
	int need_pmc5, need_pmc4;
	int pmc5_evt = -1, pmc4_evt = -1;

	if (PFMLIB_DEBUG()) {
		int j;
		for(j=0;j<cnt; j++) {
			pfm_get_event_name(cnt_list[j], &name);
			printf("%-2d (%d,%d): %s\n", 
				as[j], 
				evt_grp(cnt_list[j]) == PFMLIB_ITA2_EVT_NO_GRP ? -1 : evt_grp(cnt_list[j]), 
				evt_set(cnt_list[j]) == 0xf ? -1 : evt_set(cnt_list[j]),
				name);

		}
	}
	failure = 1;
	/*
	 * first: check that all events have an assigned counter
	 */
	for(i=0; i < cnt; i++) if (as[i]==0) goto do_failure;

	/*
	 * second: scan list of events for the presence of groups
	 * at this point, we know that there can be no set crossing per group
	 * because this has been tested earlier.
	 */
	for(i=0; i < cnt; i++) {

		e = cnt_list[i];

		if (evt_grp(e) == PFMLIB_ITA2_EVT_L1_CACHE_GRP) l1_grp_present = 1;

		if (evt_grp(e) == PFMLIB_ITA2_EVT_L2_CACHE_GRP) l2_grp_present = 1;
	}

	/*
	 * third: scan assignements and make sure that there is at least one
	 * member of a special group assigned to either PMC4 or PMC5 depending
	 * on the constraint for that group
	 */
	if (l1_grp_present || l2_grp_present) {

		need_pmc5 = l1_grp_present;
		need_pmc4 = l2_grp_present;

		for(i=0; i < cnt; i++) {

			if (need_pmc5 && as[i] == 5 && evt_grp(cnt_list[i]) == PFMLIB_ITA2_EVT_L1_CACHE_GRP) {
				need_pmc5 = 0;
				pmc5_evt = cnt_list[i];
			}

			if (need_pmc4 && as[i] == 4 && evt_grp(cnt_list[i]) == PFMLIB_ITA2_EVT_L2_CACHE_GRP) {
				need_pmc4 = 0;
				pmc4_evt = cnt_list[i];
			}

			if (need_pmc4 == 0 && need_pmc5 == 0) break;
		}
		failure = 2;
		if (need_pmc4) goto do_failure;

		failure = 3;
		if (need_pmc5) goto do_failure;
	}
	/*
	 * fourth: for the L2 cache event group, you must make sure that there is no
	 * umask conflict, except for sets 1 and 2 which do not suffer from this restriction. 
	 * The umask in PMC4 determines the umask for all the other events in the same set. 
	 * It is ignored if the event does no belong to a set or if the event has no 
	 * umask (don't care umask).
	 *
	 * XXX: redudant, already checked in check_cross_groups_and_umasks(pfmlib_param_t *evt)
	 */
	if (l2_grp_present && evt_set(pmc4_evt) != 1 && evt_set(pmc4_evt) != 2) {

		/*
		 * extract the umask of the "key" event
		 */
		pmc4_umask = evt_umask(pmc4_evt);

		failure = 4;

		for(i=0; i < cnt; i++) {

			umask = evt_umask(cnt_list[i]);

			DPRINT(("pmc4_evt=%d pmc4_umask=0x%lx cnt_list[%d]=%d grp=%d umask=0x%lx\n", pmc4_evt, pmc4_umask, i, cnt_list[i],evt_grp(cnt_list[i]), umask));

			if (as[i] != 4 && evt_grp(cnt_list[i]) == PFMLIB_ITA2_EVT_L2_CACHE_GRP && umask != 0 && umask != pmc4_umask) break;
		}
		if (i != cnt) goto do_failure;
	}

	return PFMLIB_SUCCESS;
do_failure:
	DPRINT(("%s : failure %d\n", __FUNCTION__, failure));
	return PFMLIB_ERR_NOASSIGN;
}

/*
 * It is not possible to measure more than one of the
 * L2_OZQ_CANCELS0, L2_OZQ_CANCELS1, L2_OZQ_CANCELS2 at the
 * same time.
 */

static int cancel_events[]=
{
	PME_ITA2_L2_OZQ_CANCELS0_ANY,
	PME_ITA2_L2_OZQ_CANCELS1_REL,
	PME_ITA2_L2_OZQ_CANCELS2_ACQ
};

#define NCANCEL_EVENTS	sizeof(cancel_events)/sizeof(int)

static int
check_cancel_events(pfmlib_param_t *evt)
{
	int i, j, code;
	int cancel_codes[NCANCEL_EVENTS];
	int idx = -1;

	for(i=0; i < NCANCEL_EVENTS; i++) {
		pfm_get_event_code(cancel_events[i], &code);
		cancel_codes[i] = code;
	}
	for(i=0; i < evt->pfp_count; i++) {
		for (j=0; j < NCANCEL_EVENTS; j++) {
			pfm_get_event_code(evt->pfp_evt[i], &code);
			if (code == cancel_codes[j]) {
				if (idx != -1) {
					return PFMLIB_ERR_INVAL;
				}
				idx = evt->pfp_evt[i];
			}
		}
	}
	return PFMLIB_SUCCESS;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_regt structure is ready to be submitted to kernel
 */
static int
pfm_ita2_dispatch_counters(pfmlib_param_t *evt, pfarg_reg_t *pc)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfm_ita2_reg_t reg;
	int *cnt_list = evt->pfp_evt;
	int i,j,k,l, m, ret;
	unsigned int max_l0, max_l1, max_l2, max_l3;
	int assign[PMU_ITA2_MAX_COUNTERS];
	unsigned int cnt = evt->pfp_count;

#define	has_counter(e,b)	(itanium2_pe[e].pme_counters & (1 << (b)) ? (b) : 0)

	/*
	 * first make sure that the event indexes are valid
	 */
	for (i=0; i < cnt; i++) {
		if (cnt_list[i] < 0 || cnt_list[i] >= PME_ITA2_EVENT_COUNT) return PFMLIB_ERR_INVAL;
	}

	if (PFMLIB_DEBUG())
		for (m=0; m < cnt; m++) {
			printf("ev[%d]=%s counters=0x%lx\n", m, itanium2_pe[cnt_list[m]].pme_name, 
				itanium2_pe[cnt_list[m]].pme_counters);
		}

	if (cnt > PMU_ITA2_MAX_COUNTERS) return PFMLIB_ERR_TOOMANY;

	ret = check_cross_groups_and_umasks(evt);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = check_cancel_events(evt);
	if (ret != PFMLIB_SUCCESS) return ret;

	max_l0 = PMU_FIRST_COUNTER + PMU_ITA2_MAX_COUNTERS;
	max_l1 = PMU_FIRST_COUNTER + PMU_ITA2_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_FIRST_COUNTER + PMU_ITA2_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_FIRST_COUNTER + PMU_ITA2_MAX_COUNTERS*(cnt>3);

	DPRINT(("max_l0=%d max_l1=%d max_l2=%d max_l3=%d\n", max_l0, max_l1, max_l2, max_l3));
	/*
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=PMU_FIRST_COUNTER; i < max_l0; i++) {

		assign[0] = has_counter(cnt_list[0],i);

		if (max_l1 == PMU_FIRST_COUNTER && valid_assign(cnt_list, assign,cnt) == PFMLIB_SUCCESS) goto done;

		for (j=PMU_FIRST_COUNTER; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(cnt_list[1],j);

			if (max_l2 == PMU_FIRST_COUNTER && valid_assign(cnt_list, assign,cnt) == PFMLIB_SUCCESS) goto done;

			for (k=PMU_FIRST_COUNTER; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(cnt_list[2],k);

				if (max_l3 == PMU_FIRST_COUNTER && valid_assign(cnt_list, assign,cnt) == PFMLIB_SUCCESS) goto done;
				for (l=PMU_FIRST_COUNTER; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(cnt_list[3],l);

					if (valid_assign(cnt_list, assign,cnt) == PFMLIB_SUCCESS) goto done;
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
		reg.reg_val = 0; /* clear all */
		/* if not specified per event, then use default (could be zero: measure nothing) */
		reg.pmc_plm    = evt->pfp_plm[j] ? evt->pfp_plm[j]: evt->pfp_dfl_plm; 
		reg.pmc_oi     = 1; /* overflow interrupt */
		reg.pmc_pm     = evt->pfp_pm; 
		reg.pmc_thres  = param ? param->pfp_ita2_counters[j].thres: 0;
		reg.pmc_ism    = param ? 
			(param->pfp_ita2_counters[j].ism ? param->pfp_ita2_counters[j].ism : param->pfp_ita2_ism) : PFMLIB_ITA2_ISM_BOTH;
		reg.pmc_umask  = is_ear(cnt_list[j]) ? 0x0 : itanium2_pe[cnt_list[j]].pme_umask;
		reg.pmc_es     = itanium2_pe[cnt_list[j]].pme_code;

		/*
		 * Note that we don't force PMC4.pmc_ena = 1 because the kernel takes care of this for us.
		 * This way we don't have to program something in PMC4 even when we don't use it
		 */
		pc[j].reg_num   = assign[j]; 
		pc[j].reg_value = reg.reg_val;

		pfm_vbprintf("[pmc%d=0x%06lx thres=%d es=0x%02x plm=%d umask=0x%x pm=%d ism=0x%x oi=%d] %s\n", 
					assign[j], reg.reg_val, 
					reg.pmc_thres,
					reg.pmc_es,reg.pmc_plm, 
					reg.pmc_umask, reg.pmc_pm,
					reg.pmc_ism,
					reg.pmc_oi,
					itanium2_pe[cnt_list[j]].pme_name);
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_iear(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita2_reg_t reg;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfmlib_ita2_param_t fake_param;
	int pos = *idx;
	int found_iear=0;
	int i, t;

	for (i=0; i < evt->pfp_count; i++) {
		t = is_iear(evt->pfp_evt[i]);

		if (found_iear && t) return PFMLIB_ERR_EVTMANY; /* cannot have multiple I-EAR events */

		if (t) found_iear = 1;
	}

	if (found_iear == 0) return PFMLIB_SUCCESS;

	if (param == NULL || param->pfp_ita2_iear.ear_used == 0) {
		memset(&fake_param, 0, sizeof(fake_param));
		param = &fake_param;

		pfm_ita2_get_ear_mode(evt->pfp_evt[i], &param->pfp_ita2_iear.ear_mode);
		param->pfp_ita2_iear.ear_umask = evt_umask(evt->pfp_evt[i]);
		param->pfp_ita2_iear.ear_ism   = PFMLIB_ITA2_ISM_BOTH; /* force both instruction sets */

		DPRINT(("i-ear event with no info\n"));
	}

	/* sanity check on the mode */
	if (param->pfp_ita2_iear.ear_mode < 0 || param->pfp_ita2_iear.ear_mode > 2) return PFMLIB_ERR_INVAL;

	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	reg.reg_val = 0;

	reg.pmc10_ita2_reg.iear_plm   = param->pfp_ita2_iear.ear_plm ? param->pfp_ita2_iear.ear_plm: evt->pfp_dfl_plm;
	reg.pmc10_ita2_reg.iear_pm    = evt->pfp_pm;
	reg.pmc10_ita2_reg.iear_ct    = param->pfp_ita2_iear.ear_mode == PFMLIB_ITA2_EAR_TLB_MODE ? 0x0 : 0x2;
	reg.pmc10_ita2_reg.iear_umask = param->pfp_ita2_iear.ear_umask;
	reg.pmc10_ita2_reg.iear_ism   = param->pfp_ita2_iear.ear_ism;

	pc[pos].reg_num   = 10; /* PMC10 is I-EAR config register */
	pc[pos].reg_value = reg.reg_val;

	pos++;

	pfm_vbprintf("[pmc10=0x%lx: ctb=%s plm=%d pm=%d ism=0x%x umask=0x%x]\n", 
			reg.reg_val,
			reg.pmc10_ita2_reg.iear_ct ? "Cache" : "TLB",
			reg.pmc10_ita2_reg.iear_plm,
			reg.pmc10_ita2_reg.iear_pm,
			reg.pmc10_ita2_reg.iear_ism,
			reg.pmc10_ita2_reg.iear_umask);

	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_dear(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita2_reg_t reg;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfmlib_ita2_param_t fake_param;
	int pos = *idx;
	int found_dear=0;
	int i, t;

	for (i=0; i < evt->pfp_count; i++) {
		t = is_dear(evt->pfp_evt[i]);

		if (found_dear && t) return PFMLIB_ERR_EVTMANY; /* cannot have multiple D-EAR events */

		if (t) found_dear = 1;
	}

	if (found_dear == 0) return PFMLIB_SUCCESS;


	if (param == NULL || param->pfp_ita2_dear.ear_used == 0) {
		memset(&fake_param, 0, sizeof(fake_param));
		param = &fake_param;

		pfm_ita2_get_ear_mode(evt->pfp_evt[i], &param->pfp_ita2_dear.ear_mode);
		param->pfp_ita2_dear.ear_umask = evt_umask(evt->pfp_evt[i]);
		param->pfp_ita2_dear.ear_ism   = PFMLIB_ITA2_ISM_BOTH; /* force both instruction sets */

		DPRINT(("d-ear event with no info\n"));
	}

	/* sanity check on the mode */
	if (param->pfp_ita2_dear.ear_mode > 2) return PFMLIB_ERR_INVAL;

	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;


	reg.reg_val = 0;

	reg.pmc11_ita2_reg.dear_plm   = param->pfp_ita2_dear.ear_plm ? param->pfp_ita2_dear.ear_plm: evt->pfp_dfl_plm;
	reg.pmc11_ita2_reg.dear_pm    = evt->pfp_pm;
	reg.pmc11_ita2_reg.dear_mode  = param->pfp_ita2_dear.ear_mode;
	reg.pmc11_ita2_reg.dear_umask = param->pfp_ita2_dear.ear_umask;
	reg.pmc11_ita2_reg.dear_ism   = param->pfp_ita2_dear.ear_ism;

	pc[pos].reg_num    = 11;  /* PMC11 is D-EAR config register */
	pc[pos].reg_value  = reg.reg_val;

	pos++;

	pfm_vbprintf("[pmc11=0x%lx: mode=%s plm=%d pm=%d ism=0x%x umask=0x%x]\n", 
			reg.reg_val,
			reg.pmc11_ita2_reg.dear_mode == 0 ? "L1D" : 
			(reg.pmc11_ita2_reg.dear_mode == 1 ? "L1DTLB" : "ALAT"),
			reg.pmc11_ita2_reg.dear_plm,	
			reg.pmc11_ita2_reg.dear_pm,
			reg.pmc11_ita2_reg.dear_ism,
			reg.pmc11_ita2_reg.dear_umask);


	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_opcm(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfm_ita2_reg_t reg, pmc15;
	int pos = *idx;

	if (param == NULL) return PFMLIB_SUCCESS;

	/* not constrained by PMC8 or PMC9 */
	pmc15.reg_val = 0xffffffff; /* XXX: use PAL instead. PAL value is 0xfffffff0 */

	if (   param->pfp_ita2_pmc8.opcm_used 
	    || (param->pfp_ita2_irange.rr_used && param->pfp_ita2_irange.rr_nbr_used!=0) ) {

		if (pos == max_count) return PFMLIB_ERR_FULL;

		reg.reg_val = param->pfp_ita2_pmc8.opcm_used ? param->pfp_ita2_pmc8.pmc_val : 0xffffffff3fffffff;

		if (param->pfp_ita2_irange.rr_used) {
			reg.pmc8_9_ita2_reg.opcm_ig_ad = 0;
			reg.pmc8_9_ita2_reg.opcm_inv   = param->pfp_ita2_irange.rr_flags & PFMLIB_ITA2_RR_INV ? 1 : 0;
		}

		memset(pc+pos, 0, sizeof(pfarg_reg_t));

		/* force bit 2 to 1 */
		reg.pmc8_9_ita2_reg.opcm_bit2 = 1;

		pc[pos].reg_num     = 8;
		pc[pos++].reg_value = reg.reg_val;

		/*
		 * will be constrained by PMC8
		 */
		if (param->pfp_ita2_pmc8.opcm_used) {
			pmc15.pmc15_ita2_reg.opcmc_ibrp0_pmc8 = 0;
			pmc15.pmc15_ita2_reg.opcmc_ibrp2_pmc8 = 1;
		}

		pfm_vbprintf("[pmc8=0x%lx m=%d i=%d f=%d b=%d match=0x%x mask=0x%x inv=%d ig_ad=%d]\n",
				reg.reg_val,
				reg.pmc8_9_ita2_reg.opcm_m,
				reg.pmc8_9_ita2_reg.opcm_i,
				reg.pmc8_9_ita2_reg.opcm_f,
				reg.pmc8_9_ita2_reg.opcm_b,
				reg.pmc8_9_ita2_reg.opcm_match,
				reg.pmc8_9_ita2_reg.opcm_mask,
				reg.pmc8_9_ita2_reg.opcm_inv,
				reg.pmc8_9_ita2_reg.opcm_ig_ad);
	}

	if (param->pfp_ita2_pmc9.opcm_used) {
		/*
		 * PMC9 can only be used to qualify IA64_INST_RETIRED_* events
		 */
		if (check_inst_retired_events(evt) != evt->pfp_count) return PFMLIB_ERR_FEATCOMB;

		if (pos == max_count) return PFMLIB_ERR_FULL;

		memset(pc+pos, 0, sizeof(pfarg_reg_t));

		reg.reg_val = param->pfp_ita2_pmc9.pmc_val;

		/* force bit 2 to 1 */
		reg.pmc8_9_ita2_reg.opcm_bit2 = 1;

		pc[pos].reg_num     = 9;
		pc[pos++].reg_value = reg.reg_val;

		/*
		 * will be constrained by PMC9
		 */
		pmc15.pmc15_ita2_reg.opcmc_ibrp1_pmc9 = 0;
		pmc15.pmc15_ita2_reg.opcmc_ibrp3_pmc9 = 0;

		pfm_vbprintf("[pmc9=0x%lx m=%d i=%d f=%d b=%d match=0x%x mask=0x%x]\n",
				reg.reg_val,
				reg.pmc8_9_ita2_reg.opcm_m,
				reg.pmc8_9_ita2_reg.opcm_i,
				reg.pmc8_9_ita2_reg.opcm_f,
				reg.pmc8_9_ita2_reg.opcm_b,
				reg.pmc8_9_ita2_reg.opcm_match,
				reg.pmc8_9_ita2_reg.opcm_mask);

	}
	/*
	 * setup opcode match configuration register
	 */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	memset(pc+pos, 0, sizeof(pfarg_reg_t));

	pc[pos].reg_num     = 15;
	pc[pos++].reg_value = pmc15.reg_val;

	pfm_vbprintf("[pmc15=0x%lx ibrp0_pmc8=%d ibrp1_pmc9=%d ibrp2_pmc8=%d ibrp3_pmc9=%d]\n",
			pmc15.reg_val,
			pmc15.pmc15_ita2_reg.opcmc_ibrp0_pmc8,
			pmc15.pmc15_ita2_reg.opcmc_ibrp1_pmc9,
			pmc15.pmc15_ita2_reg.opcmc_ibrp2_pmc8,
			pmc15.pmc15_ita2_reg.opcmc_ibrp3_pmc9);

	*idx = pos;

	return PFMLIB_SUCCESS;
}


static int
pfm_dispatch_btb(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	int i, pos = *idx;
	int *cnt_list = evt->pfp_evt;
	pfm_ita2_reg_t reg;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfmlib_ita2_param_t fake_param;
	int found_btb=0, found_alat=0, found_dear = 0;
	int t;

	reg.reg_val = 0;

	for (i=0; i < evt->pfp_count; i++) {
		t = is_btb(cnt_list[i]);
		/* 
		 * more than one BTB event defined is invalid
		 */
		if (found_btb && t) return PFMLIB_ERR_EVTMANY; 

		if (t) found_btb = 1;

		if (is_ear_alat(cnt_list[i])) found_alat = 1;

		if (is_dear(cnt_list[i]) && is_ear_tlb(cnt_list[i])) found_dear = 1;
	}
	DPRINT(("found_btb=%d found_alat=%d found_dear_tlb=%d\n", found_btb, found_alat, found_dear));
	/* 
	 * nothing found
	 */
	if ((found_btb | found_alat | found_dear) == 0 && (param == NULL || param->pfp_ita2_btb.btb_used == 0)) 
		return PFMLIB_SUCCESS;

	/*
	 * D-EAR ALAT/TLB and BTB cannot be used at the same time
	 */
	if (found_btb && (found_alat || found_dear)) return PFMLIB_ERR_EVTINCOMP;

	/*
	 * in case of ALAT or D-EAR tlb with simply clear pmc12
	 */
	if (found_alat || found_dear) goto assign_value;

	if (param == NULL || param->pfp_ita2_btb.btb_used == 0 ) {
		memset(&fake_param, 0, sizeof(fake_param));
		param = &fake_param;

		param->pfp_ita2_btb.btb_ds  = 0; 	/* capture branch targets */
		param->pfp_ita2_btb.btb_tm  = 0x3; 	/* all branches */
		param->pfp_ita2_btb.btb_ptm = 0x3; 	/* all branches */
		param->pfp_ita2_btb.btb_ppm = 0x3; 	/* all branches */
		param->pfp_ita2_btb.btb_brt = 0x0; 	/* all branches */

		DPRINT(("btb event with no info\n"));
	}

	reg.pmc12_ita2_reg.btbc_plm = param->pfp_ita2_btb.btb_plm ? param->pfp_ita2_btb.btb_plm: evt->pfp_dfl_plm;
	reg.pmc12_ita2_reg.btbc_pm  = evt->pfp_pm;
	reg.pmc12_ita2_reg.btbc_ds  = param->pfp_ita2_btb.btb_ds & 0x1;
	reg.pmc12_ita2_reg.btbc_tm  = param->pfp_ita2_btb.btb_tm & 0x3;
	reg.pmc12_ita2_reg.btbc_ptm = param->pfp_ita2_btb.btb_ptm & 0x3;
	reg.pmc12_ita2_reg.btbc_ppm = param->pfp_ita2_btb.btb_ppm & 0x3;
	reg.pmc12_ita2_reg.btbc_brt = param->pfp_ita2_btb.btb_brt & 0x3;

assign_value:
	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	pc[pos].reg_num     = 12;
	pc[pos++].reg_value = reg.reg_val;

	pfm_vbprintf("[pmc12=0x%lx plm=%d pm=%d ds=%d tm=%d ptm=%d ppm=%d brt=%d]\n",
				reg.reg_val,
				reg.pmc12_ita2_reg.btbc_plm,
				reg.pmc12_ita2_reg.btbc_pm,
				reg.pmc12_ita2_reg.btbc_ds,
				reg.pmc12_ita2_reg.btbc_tm,
				reg.pmc12_ita2_reg.btbc_ptm,
				reg.pmc12_ita2_reg.btbc_ppm,
				reg.pmc12_ita2_reg.btbc_brt);

	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

static void 
do_normal_rr(pfmlib_param_t *evt, unsigned long start, unsigned long end, 
			     pfarg_dbreg_t *br, int nbr, int dir, int *idx, int *reg_idx)
{
	unsigned long size, l_addr, c;
	unsigned long l_offs = 0, r_offs = 0;
	unsigned long l_size, r_size;
	dbreg_t db;
	int p2;

	if (nbr < 1 || end <= start) return;

	size = end - start;

	DPRINT(("start=0x%016lx end=0x%016lx size=0x%lx bytes (%lu bundles) nbr=%d dir=%d\n", 
			start, end, size, size >> 4, nbr, dir));

	p2 = ia64_fls(size);

	c = ALIGN_DOWN(end, p2);

	DPRINT(("largest power of two possible: 2^%d=0x%lx, crossing=0x%016lx\n", 
				p2, 
				1UL << p2, c));

	if ((c - (1UL<<p2)) >= start) {
		l_addr = c - (1UL << p2);
	} else {
		p2--;

		if ((c + (1UL<<p2)) <= end)  {
			l_addr = c;
		} else {
			l_addr = c - (1UL << p2);
		}
	}
	l_size = l_addr - start;
	r_size = end - l_addr-(1UL<<p2);

	if (PFMLIB_DEBUG()) {
		printf("largest chunk: 2^%d=0x%lx @0x%016lx-0x%016lx\n", p2, 1UL<<p2, l_addr, l_addr+(1UL<<p2));
		if (l_size) printf("before: 0x%016lx-0x%016lx\n", start, l_addr);
		if (r_size) printf("after : 0x%016lx-0x%016lx\n", l_addr+(1UL<<p2), end);
	}

	if (dir == 0 && l_size != 0 && nbr == 1) {
		p2++;
		l_addr = end - (1UL << p2);
		if (PFMLIB_DEBUG()) {
			l_offs = start - l_addr;
			printf(">>l_offs: 0x%lx\n", l_offs);
		}
	} else if (dir == 1 && r_size != 0 && nbr == 1) {
		p2++;
		l_addr = start;
		if (PFMLIB_DEBUG()) {
			r_offs = l_addr+(1UL<<p2) - end;
			printf(">>r_offs: 0x%lx\n", r_offs);
		}
	}
	l_size = l_addr - start;
	r_size = end - l_addr-(1UL<<p2);
	
	if (PFMLIB_DEBUG()) {
		printf(">>largest chunk: 2^%d @0x%016lx-0x%016lx\n", p2, l_addr, l_addr+(1UL<<p2));
		if (l_size && !l_offs) printf(">>before: 0x%016lx-0x%016lx\n", start, l_addr);
		if (r_size && !r_offs) printf(">>after : 0x%016lx-0x%016lx\n", l_addr+(1UL<<p2), end);
	}

	/*
	 * we initialize the mask to full 0 and
	 * only update the mask field. the rest is left
	 * to zero, except for the plm.
	 * in the case of ibr, the x-field must be 0. For dbr
	 * the value of r-field and w-field is ignored.
	 */

	db.val        = 0;
	db.db.db_mask = ~((1UL << p2)-1);
	/* 
	 * we always use default privilege level.
	 * plm is ignored for DBRs.
	 */
	db.db.db_plm  = evt->pfp_dfl_plm; 


	br[*idx].dbreg_num     = *reg_idx;
	br[*idx].dbreg_value   = l_addr;

	br[*idx+1].dbreg_num   = *reg_idx+1;
	br[*idx+1].dbreg_value = db.val;

	*idx     += 2;
	*reg_idx += 2;

	nbr--;
	if (nbr) {
		int r_nbr, l_nbr;

		r_nbr = l_nbr = nbr >>1;

		if (nbr & 0x1) {
			/*
			 * our simple heuristic is:
			 * we assign the largest number of registers to the largest
			 * of the two chunks
			 */
			if (l_size > r_size) {
				l_nbr++;
			} else {
				r_nbr++;
			}

		}
		do_normal_rr(evt, start, l_addr, br, l_nbr, 0, idx, reg_idx);
		do_normal_rr(evt, l_addr+(1UL<<p2), end, br, r_nbr, 1, idx, reg_idx);
	}
}


static void
print_one_range(pfmlib_ita2_rr_t *rr, pfmlib_ita2_rr_desc_t *lim, pfarg_dbreg_t *dbr, int n_pairs, int fine_mode)
{
	int i, j;
	dbreg_t *d;
	unsigned long r_end;

		printf("[0x%lx-0x%lx): %d register pair(s)%s%s\n", 
				lim->rr_start, lim->rr_end,
				n_pairs,
				fine_mode ? ", fine_mode" : "",
				rr->rr_flags & PFMLIB_ITA2_RR_INV ? ", inversed" : "");
		printf("start offset: -0x%lx end_offset: +0x%lx\n", lim->rr_soff, lim->rr_eoff);

		for (j=0; j < n_pairs; j++) {

			i = j<<1;
			d     = (dbreg_t *)&dbr[i+1].dbreg_value;
			r_end = dbr[i].dbreg_value+((~(d->db.db_mask)) & ~(0xffUL << 56));

			if (fine_mode)
				printf("brp%u:  db%u: 0x%016lx db%u: plm=0x%x mask=0x%016lx\n", 
					dbr[i].dbreg_num>>1, 
					dbr[i].dbreg_num, 
					dbr[i].dbreg_value, 
					dbr[i+1].dbreg_num, 
					d->db.db_plm, d->db.db_mask);
			else
				printf("brp%u:  db%u: 0x%016lx db%u: plm=0x%x mask=0x%016lx end=0x%016lx\n", 
					dbr[i].dbreg_num>>1, 
					dbr[i].dbreg_num, 
					dbr[i].dbreg_value, 
					dbr[i+1].dbreg_num, 
					d->db.db_plm, d->db.db_mask,
					r_end);
		}
}


/*
 * reg_idx = base register index to use (for IBRP1, reg_idx = 2)
 */
static int
compute_fine_rr(pfmlib_param_t *evt, pfmlib_ita2_rr_t *rr, int n, int reg_idx)
{
	int i;
	pfarg_dbreg_t *br = rr->rr_br;
	pfmlib_ita2_rr_desc_t *lim;
	unsigned long addr;
	dbreg_t db;

	lim = rr->rr_limits;

	db.val        = 0;
	db.db.db_mask = FINE_MODE_MASK;
	db.db.db_plm  = evt->pfp_dfl_plm; 

	if (n > 2) return PFMLIB_ERR_IRRTOOMANY;

	for (i=0; i < n; i++, reg_idx += 2, lim++, br+= 4) {
		/*
		 * setup lower limit pair 
		 *
		 * because of the PMU bug, we must align down to the closest bundle-pair
		 * aligned address. 5 => 32-byte aligned address
		 */
		addr = has_fine_mode_bug ? ALIGN_DOWN(lim->rr_start, 5) : lim->rr_start;
		lim->rr_soff = lim->rr_start - addr;

		br[0].dbreg_num   = reg_idx;
		br[0].dbreg_value = addr;
		br[1].dbreg_num   = reg_idx+1;
		br[1].dbreg_value = db.val; 

		/*
		 * setup upper limit pair
		 *
		 *
		 * In fine mode, the bundle address stored in the upper limit debug
		 * registers is included in the count, so we substract 0x10 to exclude it.
		 *
		 * because of the PMU bug, we align the (corrected) end to the nearest
		 * 32-byte aligned address + 0x10. With this correction and depending
		 * on the correction, we may count one 
		 *
		 * 
		 */
		
		addr = lim->rr_end - 0x10;
		if (has_fine_mode_bug && (addr & 0x1f) == 0) addr += 0x10;
		lim->rr_eoff = addr - lim->rr_end + 0x10;

		br[2].dbreg_num   = reg_idx+4;
		br[2].dbreg_value = addr;

		br[3].dbreg_num   = reg_idx+5;
		br[3].dbreg_value = db.val; 


		if (PFMLIB_VERBOSE()) print_one_range(rr, lim, br, 2, 1);
	}
	rr->rr_nbr_used = i<<2;

	return PFMLIB_SUCCESS;
}

/*
 * reg_idx = base register index to use (for IBRP1, reg_idx = 2)
 */
static int
compute_single_rr(pfmlib_param_t *evt, pfmlib_ita2_rr_t *rr, int reg_idx)
{
	unsigned long size, end, start;
	unsigned long p_start, p_end;
	pfarg_dbreg_t *br = rr->rr_br;
	pfmlib_ita2_rr_desc_t *lim;
	dbreg_t db;
	int l, m;

	lim   = rr->rr_limits;
	end   = lim->rr_end;
	start = lim->rr_start;
	size  = end - start;

	l = ia64_fls(size);

	m = l;
	if (size & ((1UL << l)-1)) {
		if (l>62) {
			printf("range: [0x%lx-0x%lx] too big\n", start, end);
			return PFMLIB_ERR_IRRTOOBIG;
		}
		m++;
	}

	DPRINT(("size=%ld, l=%d m=%d, internal: 0x%lx full: 0x%lx\n",
				size,
				l, m,
				1UL << l, 
				1UL << m));

	for (; m < 64; m++) {
		p_start = ALIGN_DOWN(start, m);
		p_end   = p_start+(1UL<<m); 
		if (p_end >= end) goto found;
	} 
	return PFMLIB_ERR_IRRINVAL;
found:
	DPRINT(("m=%d p_start=0x%lx p_end=0x%lx\n", m, p_start,p_end));

	/* when the event is not IA64_INST_RETIRED, then we MUST use ibrp0 */
	br[0].dbreg_num   = reg_idx;
	br[0].dbreg_value = p_start;

	db.val        = 0;
	db.db.db_mask = ~((1UL << m)-1);
	db.db.db_plm  = evt->pfp_dfl_plm; 


	br[1].dbreg_num   = reg_idx + 1;
	br[1].dbreg_value = db.val; 

	lim->rr_soff = start - p_start;
	lim->rr_eoff = p_end - end;

	if (PFMLIB_VERBOSE()) print_one_range(rr, lim, br, 1, 0);

	rr->rr_nbr_used = 2;

	return PFMLIB_SUCCESS;
}

static int
compute_normal_rr(pfmlib_param_t *evt, pfmlib_ita2_rr_t *rr, int n, int base_idx)
{
	int i, j, br_index, reg_index, prev_index;
	pfmlib_ita2_rr_desc_t *lim;
	unsigned long r_end;
	pfarg_dbreg_t *br = rr->rr_br;
	dbreg_t *d;

	lim       = rr->rr_limits;
	br        = rr->rr_br;
	reg_index = base_idx;
	br_index  = 0;

	for (i=0; i < n; i++, lim++) {
		/* 
		 * running out of registers
		 */
		if (br_index == 8) break;

		prev_index = br_index;

		do_normal_rr(evt, lim->rr_start, 
				  lim->rr_end, 
				  br, 
				  4 - (reg_index>>1), /* how many pairs available */
				  0,
				  &br_index,
				  &reg_index);

		DPRINT(("br_index=%d reg_index=%d\n", br_index, reg_index));
		/*
		 * compute offsets
		 */
		lim->rr_soff = lim->rr_eoff = 0;

		for(j=prev_index; j < br_index; j+=2) {

			d     = (dbreg_t *)&br[j+1].dbreg_value;
			r_end = br[j].dbreg_value+((~(d->db.db_mask)+1) & ~(0xffUL << 56));

			if (br[j].dbreg_value <= lim->rr_start)
				lim->rr_soff = lim->rr_start - br[j].dbreg_value; 

			if (r_end >= lim->rr_end)
				lim->rr_eoff = r_end - lim->rr_end; 
		}

		if (PFMLIB_VERBOSE()) print_one_range(rr, lim, br, (br_index-prev_index)>>1, 0);


	}

	/* do not have enough registers to cover all the ranges */
	if (br_index == 8 && i < n) return PFMLIB_ERR_TOOMANY;

	rr->rr_nbr_used = br_index;

	return PFMLIB_SUCCESS;
}


static int
pfm_dispatch_irange(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita2_reg_t reg;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfmlib_ita2_rr_t *rr;
	int i, pos = *idx, ret;
	int retired_only, retired_count, fine_mode, prefetch_count;
	int n_intervals;

	if (param == NULL) return PFMLIB_SUCCESS;

	if (param->pfp_ita2_irange.rr_used == 0) return PFMLIB_SUCCESS;

	rr = &param->pfp_ita2_irange;

	ret = check_intervals(rr, 0, &n_intervals);
	if (ret != PFMLIB_SUCCESS) return ret;

	if (n_intervals < 1) return PFMLIB_ERR_IRRINVAL;
	
	retired_count  = check_inst_retired_events(evt);
	retired_only   = retired_count == evt->pfp_count;
	prefetch_count = check_prefetch_events(evt);
	fine_mode      = rr->rr_flags & PFMLIB_ITA2_RR_NO_FINE_MODE ? 
		         0 : check_fine_mode_possible(rr, n_intervals);

	DPRINT(("n_intervals=%d retired_only=%d retired_count=%d prefetch_count=%d fine_mode=%d\n", 
		n_intervals, retired_only, retired_count, prefetch_count, fine_mode));

	/*
	 * On Itanium2, there are more constraints on what can be measured with irange.
	 *
	 * - The fine mode is the best because you directly set the lower and upper limits of
	 *   the range. This uses 2 ibr pairs for range (ibrp0/ibrp2 and ibp1/ibrp3). Therefore
	 *   at most 2 fine mode ranges can be defined. There is a limit on the size and alignment
	 *   of the range to allow fine mode: the range must be less than 4KB in size AND the lower
	 *   and upper limits must NOT cross a 4KB page boundary. The fine mode works will all events.
	 *
	 * - if the fine mode fails, then for all events, except IA64_TAGGED_INST_RETIRED_*, only 
	 *   the first pair of ibr is available: ibrp0. This imposes some severe restrictions on the
	 *   size and alignement of the range. It can be bigger than 4KB and must be properly aligned
	 *   on its size. The library relaxes these constraints by allowing the covered areas to be 
	 *   larger than the expected range. It may start before and end after. You can determine how
	 *   far off the range is in either direction for each range by looking at the rr_soff (start
	 *   offset) and rr_eoff (end offset).
	 *
	 * - if the events include certain prefetch events then only IBRP1 can be used in fine mode
	 *   See 10.3.5.1 Exception 1. 
	 *
	 * - Finally, when the events are ONLY IA64_TAGGED_INST_RETIRED_* then all IBR pairs can be used 
	 *   to cover the range giving us more flexibility to approximate the range when it is not 
	 *   properly aligned on its size (see 10.3.5.2 Exception 2).
	 */

	if (fine_mode == 0 && retired_only == 0 && n_intervals > 1) return PFMLIB_ERR_IRRTOOMANY;

	/* we do not default to non-fine mode to support more ranges */
	if (n_intervals > 2 && fine_mode == 1) return PFMLIB_ERR_IRRTOOMANY;

	if (fine_mode == 0) {
		if (retired_only) {
			ret = compute_normal_rr(evt, rr, n_intervals, 0);
		} else {
			/* unless we have only prefetch and instruction retired events, 
			 * we cannot satisfy the request because the other events cannot
			 * be measured on anything but IBRP0.
			 */
			if (prefetch_count && (prefetch_count+retired_count) != evt->pfp_count) 
				return PFMLIB_ERR_FEATCOMB;

			ret = compute_single_rr(evt, rr, prefetch_count ? 2 : 0);
		}
	} else {
		if (prefetch_count && n_intervals != 1) return PFMLIB_ERR_IRRTOOMANY;

		ret = compute_fine_rr(evt, rr, n_intervals, prefetch_count ? 2 : 0);
	}
	if (ret != PFMLIB_SUCCESS) {
		return ret == PFMLIB_ERR_TOOMANY ? PFMLIB_ERR_IRRTOOMANY : ret;
	}

	if (pos == max_count) return PFMLIB_ERR_FULL;

	reg.reg_val = 0xdb6; /* default value */

	for (i=0; i < rr->rr_nbr_used; i++) {
		if (rr->rr_br[i].dbreg_num == 0) reg.pmc14_ita2_reg.iarc_ibrp0 = 0;
		if (rr->rr_br[i].dbreg_num == 2) reg.pmc14_ita2_reg.iarc_ibrp1 = 0;
		if (rr->rr_br[i].dbreg_num == 4) reg.pmc14_ita2_reg.iarc_ibrp2 = 0;
		if (rr->rr_br[i].dbreg_num == 6) reg.pmc14_ita2_reg.iarc_ibrp3 = 0;
	}

	if (retired_only && (param->pfp_ita2_pmc8.opcm_used ||param->pfp_ita2_pmc9.opcm_used)) {
		/*
		 * PMC8 + IA64_INST_RETIRED only works if irange on IBRP0 and/or IBRP2
		 * PMC9 + IA64_INST_RETIRED only works if irange on IBRP1 and/or IBRP3
		 */
		for (i=0; i < rr->rr_nbr_used; i++) {
			if (rr->rr_br[i].dbreg_num == 0 && param->pfp_ita2_pmc9.opcm_used)  return PFMLIB_ERR_FEATCOMB;
			if (rr->rr_br[i].dbreg_num == 2 && param->pfp_ita2_pmc8.opcm_used)  return PFMLIB_ERR_FEATCOMB;
			if (rr->rr_br[i].dbreg_num == 4 && param->pfp_ita2_pmc9.opcm_used)  return PFMLIB_ERR_FEATCOMB;
			if (rr->rr_br[i].dbreg_num == 6 && param->pfp_ita2_pmc8.opcm_used)  return PFMLIB_ERR_FEATCOMB;
		}
	}


	if (fine_mode) {
		reg.pmc14_ita2_reg.iarc_fine = 1;
	}

	/* initialize pmc request slot */
	memset(pc+pos, 0, sizeof(pfarg_reg_t));

	pc[pos].reg_num     = 14;
	pc[pos++].reg_value = reg.reg_val;

	pfm_vbprintf("[pmc14=0x%lx ibrp0=%d ibrp1=%d ibrp2=%d ibrp3=%d fine=%d]\n",
			reg.reg_val,
			reg.pmc14_ita2_reg.iarc_ibrp0,
			reg.pmc14_ita2_reg.iarc_ibrp1,
			reg.pmc14_ita2_reg.iarc_ibrp2,
			reg.pmc14_ita2_reg.iarc_ibrp3,
			reg.pmc14_ita2_reg.iarc_fine);

	*idx = pos;

	return PFMLIB_SUCCESS;
}

static const int iod_tab[8]={
	/* --- */	3,
	/* --D */	2,
	/* -O- */	-1,/* NOT REACHED, nothing to do if simply using OPC */
	/* -OD */	0, /* =IOD safe because default IBR is harmless */
	/* I-- */	1, /* =IO safe because by defaut OPC is turned off */
	/* I-D */	0, /* =IOD safe because by default opc is turned off */
	/* IO- */	1,
	/* IOD */	0
};

/*
 * IMPORTANT: MUST BE CALLED *AFTER* pfm_dispatch_irange() to make sure we see 
 * the irange programming to adjust pmc13.
 */
static int
pfm_dispatch_drange(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	pfmlib_ita2_rr_t *rr;
	pfm_ita2_reg_t pmc13;
	int i, pos = *idx;
	int iod_codes[4], dfl_val;
	int n_intervals, ret;
#define DR_USED	0x1
#define OP_USED	0x2
#define IR_USED	0x4

	if (param == NULL) return PFMLIB_SUCCESS;
	/*
	 * nothing to do here, the default value for pmc13 is good enough
	 */
	if (  param->pfp_ita2_pmc8.opcm_used == 0
	   && param->pfp_ita2_pmc9.opcm_used == 0
	   && param->pfp_ita2_drange.rr_used == 0
	   && param->pfp_ita2_irange.rr_used == 0) return PFMLIB_SUCCESS;

	if (pos == max_count) return PFMLIB_ERR_FULL;

	/*
	 * it seems like the ignored bits need to have special values
	 * otherwise this does not work.
	 */
	pmc13.reg_val = 0x2078fefefefe; /* XXX: use PAL instead */

	/*
	 * initialize iod codes
	 */
	dfl_val      = param->pfp_ita2_pmc8.opcm_used || param->pfp_ita2_pmc9.opcm_used ? OP_USED : 0;
	iod_codes[0] = iod_codes[1] = iod_codes[2] = iod_codes[3] = 0;
	if (param->pfp_ita2_drange.rr_used == 1) {

		rr = &param->pfp_ita2_drange;

		ret = check_intervals(rr, 1, &n_intervals);
		if (ret != PFMLIB_SUCCESS) return ret;

		if (n_intervals < 1) return PFMLIB_ERR_DRRINVAL;

		ret = compute_normal_rr(evt, rr, n_intervals,0);
		if (ret != PFMLIB_SUCCESS) {
			return ret == PFMLIB_ERR_TOOMANY ? PFMLIB_ERR_DRRTOOMANY : ret;
		}

		/*
		 * Update iod_codes are update to reflect the use of the DBR constraint.
		 */
		for (i=0; i < rr->rr_nbr_used; i++) {
			if (rr->rr_br[i].dbreg_num == 0) iod_codes[0] |= DR_USED | dfl_val;
			if (rr->rr_br[i].dbreg_num == 2) iod_codes[1] |= DR_USED | dfl_val;
			if (rr->rr_br[i].dbreg_num == 4) iod_codes[2] |= DR_USED | dfl_val;
			if (rr->rr_br[i].dbreg_num == 6) iod_codes[3] |= DR_USED | dfl_val;
		}

	} 

	/*
	 * XXX: assume dispatch_irange executed before calling this function
	 */
	if (param->pfp_ita2_irange.rr_used == 1) {
		pfmlib_ita2_rr_t *rr2;
		int fine_mode = 0;

		rr2 = &param->pfp_ita2_irange;

		/*
		 * we need to find out whether or not the irange is using
		 * fine mode. If this is the case, then we only need to
		 * program pmc13 for the ibr pairs which designate the lower
		 * bounds of a range. For instance, if IBRP0/IBRP2 are used, 
		 * then we only need to program pmc13.cfg_ibrp0 and pmc13.ena_dbrp0,
		 * the PMU will automatically use IBRP2, even though pmc13.ena_dbrp2=0.
		 */
		for(i=0; i <= pos; i++) {
			if (pc[i].reg_num == 14) {
				pfm_ita2_reg_t pmc14;
				pmc14.reg_val = pc[i].reg_value;
				if (pmc14.pmc14_ita2_reg.iarc_fine == 1) fine_mode = 1; 
				break;
			}
		}

		/* 
		 * Update to reflect the use of the IBR constraint
		 */
		for (i=0; i < rr2->rr_nbr_used; i++) {
			if (rr2->rr_br[i].dbreg_num == 0) iod_codes[0] |= IR_USED | dfl_val;
			if (rr2->rr_br[i].dbreg_num == 2) iod_codes[1] |= IR_USED | dfl_val;
			if (fine_mode == 0 && rr2->rr_br[i].dbreg_num == 4) iod_codes[2] |= IR_USED | dfl_val;
			if (fine_mode == 0 && rr2->rr_br[i].dbreg_num == 6) iod_codes[3] |= IR_USED | dfl_val;
		}
	}

	/*
	 * update the cfg dbrpX field. If we put a constraint on a cfg dbrp, then
	 * we must enable it in the corresponding ena_dbrpX
	 */
	pmc13.pmc13_ita2_reg.darc_ena_dbrp0 = iod_codes[0] ? 1 : 0;
	pmc13.pmc13_ita2_reg.darc_cfg_dbrp0 = iod_tab[iod_codes[0]];

	pmc13.pmc13_ita2_reg.darc_ena_dbrp1 = iod_codes[1] ? 1 : 0;
	pmc13.pmc13_ita2_reg.darc_cfg_dbrp1 = iod_tab[iod_codes[1]];

	pmc13.pmc13_ita2_reg.darc_ena_dbrp2 = iod_codes[2] ? 1 : 0;
	pmc13.pmc13_ita2_reg.darc_cfg_dbrp2 = iod_tab[iod_codes[2]];

	pmc13.pmc13_ita2_reg.darc_ena_dbrp3 = iod_codes[3] ? 1 : 0;
	pmc13.pmc13_ita2_reg.darc_cfg_dbrp3 = iod_tab[iod_codes[3]];


	pc[pos].reg_num     = 13;
	pc[pos++].reg_value = pmc13.reg_val;

	pfm_vbprintf("[pmc13=0x%lx cfg_dbrp0=%d cfg_dbrp1=%d cfg_dbrp2=%d cfg_dbrp3=%d ena_dbrp0=%d ena_dbrp1=%d ena_dbrp2=%d ena_dbrp3=%d]\n",
			pmc13.reg_val,
			pmc13.pmc13_ita2_reg.darc_cfg_dbrp0,
			pmc13.pmc13_ita2_reg.darc_cfg_dbrp1,
			pmc13.pmc13_ita2_reg.darc_cfg_dbrp2,
			pmc13.pmc13_ita2_reg.darc_cfg_dbrp3,
			pmc13.pmc13_ita2_reg.darc_ena_dbrp0,
			pmc13.pmc13_ita2_reg.darc_ena_dbrp1,
			pmc13.pmc13_ita2_reg.darc_ena_dbrp2,
			pmc13.pmc13_ita2_reg.darc_ena_dbrp3);
	*idx = pos;

	return PFMLIB_SUCCESS;
}





static int
check_qualifier_constraints(pfmlib_param_t *evt)
{
	int i;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	int *cnt_list = evt->pfp_evt;

	for(i=0; i < evt->pfp_count; i++) {
		if (evt_use_irange(param) && has_iarr(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
		if (evt_use_drange(param) && has_darr(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
		if (evt_use_opcm(param) && has_opcm(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_ita2_dispatch_events(pfmlib_param_t *evt, pfarg_reg_t *pc, int *count)
{
	int idx, ret;
	int max_count;
	pfmlib_ita2_param_t *p;

	if (evt == NULL || pc == NULL || count == NULL) return PFMLIB_ERR_INVAL;

	p = ITA2_PARAM(evt);

	/* simple  sanity check */
	if (p && p->pfp_magic != PFMLIB_ITA2_PARAM_MAGIC) return PFMLIB_ERR_MAGIC;

	/*
	 * evt->pfp_count can be zero, in which case, the caller is only
	 * interested in setting up non counter registers.
	 */

	/* not enough room in pc[] for requested number of counters */
	if (*count < evt->pfp_count) return PFMLIB_ERR_INVAL;

	/* check opcode match, range restriction qualifiers */
	if (p && check_qualifier_constraints(evt) == -1) return PFMLIB_ERR_FEATCOMB;

	ret = pfm_ita2_dispatch_counters(evt, pc);
	if (ret != PFMLIB_SUCCESS) return ret;

	max_count = *count;

	/* index of first free entry */
	idx       = evt->pfp_count;

	if (p == NULL) goto no_special_features;

	/* now check for I-EAR */
	ret = pfm_dispatch_iear(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	/* now check for D-EAR */
	ret = pfm_dispatch_dear(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	/* XXX: must be done before dispatch_opcm()  and dispatch_drange() */
	ret = pfm_dispatch_irange(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = pfm_dispatch_drange(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	/* now check for Opcode matchers */
	ret = pfm_dispatch_opcm(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = pfm_dispatch_btb(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

no_special_features:
	/* how many entries are used for the setup */
	*count = idx;

	return PFMLIB_SUCCESS;
}


/* XXX: return value is also error code */
int
pfm_ita2_get_event_maxincr(int i, unsigned long *maxincr)
{
	if (i<0 || i >= PME_ITA2_EVENT_COUNT || maxincr == NULL) return PFMLIB_ERR_INVAL;
	*maxincr = itanium2_pe[i].pme_maxincr;
	return PFMLIB_SUCCESS;
}

int
pfm_ita2_is_ear(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_ear(i);
}

int
pfm_ita2_is_dear(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_dear(i);
}

int
pfm_ita2_is_dear_tlb(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_dear(i) && is_ear_tlb(i);
}
	
int
pfm_ita2_is_dear_cache(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_dear(i) && is_ear_cache(i);
}

int
pfm_ita2_is_dear_alat(int i)
{
	return i>= 0 && i < PME_ITA2_EVENT_COUNT && is_ear_alat(i);
}
	
int
pfm_ita2_is_iear(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_iear(i);
}

int
pfm_ita2_is_iear_tlb(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_iear(i) && is_ear_tlb(i);
}
	
int
pfm_ita2_is_iear_cache(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && is_iear(i) && is_ear_cache(i);
}
	
int
pfm_ita2_is_btb(int i)
{
	return i >=0 && i < PME_ITA2_EVENT_COUNT && is_btb(i);
}

int
pfm_ita2_support_iarr(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && has_iarr(i);
}


int
pfm_ita2_support_darr(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT  && has_darr(i);
}


int
pfm_ita2_support_opcm(int i)
{
	return i >= 0 && i < PME_ITA2_EVENT_COUNT && has_opcm(i);
}

int
pfm_ita2_get_ear_mode(int i, pfmlib_ita2_ear_mode_t *m)
{
	int r;

	if (!is_ear(i) || m == NULL) return PFMLIB_ERR_INVAL;

	r = PFMLIB_ITA2_EAR_TLB_MODE;
	if (is_ear_tlb(i))  goto done;

	r = PFMLIB_ITA2_EAR_CACHE_MODE;
	if (is_ear_cache(i))  goto done;

	r = PFMLIB_ITA2_EAR_ALAT_MODE;
	if (is_ear_alat(i)) goto done;

	return PFMLIB_ERR_INVAL;
done:
	*m = r;
	return PFMLIB_SUCCESS;
}

/*
 * Function used to print information about a specific events. More than
 * one event can be printed in case an event code is given rather than
 * a specific name. A callback function is used for printing.
 */
static int 
pfm_ita2_print_info(int v, int (*pf)(const char *fmt,...)) 
{
	pme_ita2_entry_t *e;
        static const char *quals[]={ "[Instruction Address Range]", "[OpCode Match]", "[Data Address Range]" };
	static const char *groups[]= {"???", "L1D Cache", "L2 Cache"};
	long c;
        int i;

	if (v < 0 || v >= PME_ITA2_EVENT_COUNT || pf == NULL) return PFMLIB_ERR_INVAL;
	e = itanium2_pe+v;

	if (is_ear(e-itanium2_pe))
		i=8;
	else
		i=3;
	(*pf)("Umask  : ");
	c = e->pme_umask;
	for (; i >=0; i--) {
		(*pf)("%d", c & 1<<i ? 1 : 0);
	}
	(*pf)("\n");

	(*pf)( "EAR    : %s (%s)\n",
		event_is_ear(e) ? (event_is_dear(e) ? "Data" : "Inst") : "No",
		event_is_ear(e) ? (event_is_ear_tlb(e) ? "TLB Mode": (event_is_ear_alat(e) ? "ALAT Mode": "Cache Mode")): "N/A");
	

	(*pf)("BTB    : %s\n", event_is_btb(e) ? "Yes" : "No");
	if (e->pme_maxincr > 1) 
		(*pf)("MaxIncr: %u  (Threshold [0-%u])\n", e->pme_maxincr,  e->pme_maxincr-1);
 	else 
		(*pf)("MaxIncr: %u  (Threshold 0)\n", e->pme_maxincr);

	(*pf)("Qual   : ");

	c = e->pme_qualifiers.qual;
	if ((c&0x7) == 0) {
		(*pf)("None");
	} else {
		for (i=0; i < 3; i++ ) {
                	if (c & 0x1) (*pf)("%s ", quals[i]);
                	c >>= 1;
        	}
	}
	(*pf)("\n");

	if (e->pme_qualifiers.pme_qual.pme_group == PFMLIB_ITA2_EVT_NO_GRP)
		(*pf)("Group  : None\n");
	else {
		unsigned long g = e->pme_qualifiers.pme_qual.pme_group;
		const char *str =  g < 3 ? groups[g]: 0;
		(*pf)("Group  : %s\n", str);
	}
	if (e->pme_qualifiers.pme_qual.pme_set == 0xf)
		(*pf)("Set    : None\n");
	else
		(*pf)("Set    : %ld\n", e->pme_qualifiers.pme_qual.pme_set);

	if (e->pme_desc) (*pf)("Desc   : %s\n", e->pme_desc);

	return PFMLIB_SUCCESS;
}

static unsigned long
pfm_ita2_get_event_code(int i)
{
	return itanium2_pe[i].pme_code;
}

static unsigned long
pfm_ita2_get_event_vcode(int i)
{
	return itanium2_pe[i].pme_entry_code.pme_vcode;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_ita2_get_event_umask(int i, unsigned long *umask)
{
	if (i<0 || i >= PME_ITA2_EVENT_COUNT || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = evt_umask(i);
	return PFMLIB_SUCCESS;
}
	
int
pfm_ita2_get_event_group(int i, int *grp)
{
	if (i<0 || i >= PME_ITA2_EVENT_COUNT || grp == NULL) return PFMLIB_ERR_INVAL;
	*grp = evt_grp(i);
	return PFMLIB_SUCCESS;
}

int
pfm_ita2_get_event_set(int i, int *set)
{
	if (i<0 || i >= PME_ITA2_EVENT_COUNT || set == NULL) return PFMLIB_ERR_INVAL;
	*set = evt_set(i);
	return PFMLIB_SUCCESS;
}

static char *
pfm_ita2_get_event_name(int i)
{
	return itanium2_pe[i].pme_name;
}

static unsigned long
pfm_ita2_get_event_counters(int i)
{
	return itanium2_pe[i].pme_counters;
}

static int
pfm_ita2_num_counters(void)
{
	return  PMU_ITA2_MAX_COUNTERS;
}


static unsigned long pfm_ita2_impl_pmds[4]={
	0x1ffffUL,
	0x0,
	0x0
};

static unsigned long pfm_ita2_impl_pmcs[4]={
	0xffffUL,
	0x0,
	0x0
};


static int
pfm_ita2_get_impl_pmcs(unsigned long impl_pmcs[4])
{
	impl_pmcs[0] = pfm_ita2_impl_pmcs[0];
	impl_pmcs[1] = pfm_ita2_impl_pmcs[1];
	impl_pmcs[2] = pfm_ita2_impl_pmcs[2];
	impl_pmcs[3] = pfm_ita2_impl_pmcs[3];

	return PFMLIB_SUCCESS;
}

static int
pfm_ita2_get_impl_pmds(unsigned long impl_pmds[4])
{
	impl_pmds[0] = pfm_ita2_impl_pmds[0];
	impl_pmds[1] = pfm_ita2_impl_pmds[1];
	impl_pmds[2] = pfm_ita2_impl_pmds[2];
	impl_pmds[3] = pfm_ita2_impl_pmds[3];

	return PFMLIB_SUCCESS;
}


pfm_pmu_support_t itanium2_support={
		"McKinley",
		PFMLIB_ITANIUM2_PMU,
		PME_ITA2_EVENT_COUNT,
		pfm_ita2_get_event_code,
		pfm_ita2_get_event_vcode,
		pfm_ita2_get_event_name,
		pfm_ita2_get_event_counters,
		pfm_ita2_print_info,
		pfm_ita2_dispatch_events,
		pfm_ita2_num_counters,
		pfm_ita2_detect,
		pfm_ita2_get_impl_pmcs,
		pfm_ita2_get_impl_pmds
};
