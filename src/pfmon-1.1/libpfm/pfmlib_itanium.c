/*
 * pfmlib_itanium.c : support for Itanium-family PMU 
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
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* public headers */
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_itanium_priv.h"
#include "itanium_events.h"

#define is_ear(i)	event_is_ear(itanium_pe+(i))
#define is_tlb_ear(i)	event_is_tlb_ear(itanium_pe+(i))
#define is_iear(i)	event_is_iear(itanium_pe+(i))
#define is_dear(i)	event_is_dear(itanium_pe+(i))
#define is_btb(i)	event_is_btb(itanium_pe+(i))
#define has_opcm(i)	event_opcm_ok(itanium_pe+(i))
#define has_iarr(i)	event_iarr_ok(itanium_pe+(i))
#define has_darr(i)	event_darr_ok(itanium_pe+(i))

#define evt_use_opcm(e)		((e)->pfp_ita_pmc8.opcm_used != 0 || (e)->pfp_ita_pmc9.opcm_used !=0)
#define evt_use_irange(e)	((e)->pfp_ita_irange.rr_used)
#define evt_use_drange(e)	((e)->pfp_ita_drange.rr_used)

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
pfm_ita_detect(void)
{
	/*
	 * we support all chips (there is only one!) in the Itanium family
	 */
	return pfm_get_cpu_family() == 0x07 ? PFMLIB_SUCCESS : PFMLIB_ERR_NOTSUPP;
}

/*
 * Part of the following code will eventually go into a perfmon library
 */
static int
valid_assign(int *as, int cnt)
{
	int i;
	for(i=0; i < cnt; i++) if (as[i]==0) return PFMLIB_ERR_NOASSIGN;
	return PFMLIB_SUCCESS;
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 * Upon return the pfarg_reg_t structure is ready to be submitted to kernel
 */
static int
pfm_ita_dispatch_counters(pfmlib_param_t *evt, pfarg_reg_t *pc)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	pfm_ita_reg_t reg;
	int *cnt_list = evt->pfp_evt;
	int i,j,k,l, m;
	unsigned int max_l0, max_l1, max_l2, max_l3;
	int assign[PMU_ITA_MAX_COUNTERS];
	unsigned int cnt = evt->pfp_count;

#define	has_counter(e,b)	(itanium_pe[e].pme_counters & (1 << (b)) ? (b) : 0)

	if (PFMLIB_DEBUG()) {
		for (m=0; m < cnt; m++) {
			printf("ev[%d]=%s counters=0x%lx\n", m, itanium_pe[cnt_list[m]].pme_name, 
				itanium_pe[cnt_list[m]].pme_counters);
		}
	}
	if (cnt > PMU_ITA_MAX_COUNTERS) return PFMLIB_ERR_TOOMANY;

	max_l0 = PMU_FIRST_COUNTER + PMU_ITA_MAX_COUNTERS;
	max_l1 = PMU_FIRST_COUNTER + PMU_ITA_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_FIRST_COUNTER + PMU_ITA_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_FIRST_COUNTER + PMU_ITA_MAX_COUNTERS*(cnt>3);

	if (PFMLIB_DEBUG())
		printf("max_l0=%d max_l1=%d max_l2=%d max_l3=%d\n", max_l0, max_l1, max_l2, max_l3);
	/*
	 *  This code needs fixing. It is not very pretty and 
	 *  won't handle more than 4 counters if more become
	 *  available !
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=PMU_FIRST_COUNTER; i < max_l0; i++) {

		assign[0]= has_counter(cnt_list[0],i);

		if (max_l1 == PMU_FIRST_COUNTER && valid_assign(assign,cnt) == PFMLIB_SUCCESS) goto done;

		for (j=PMU_FIRST_COUNTER; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(cnt_list[1],j);

			if (max_l2 == PMU_FIRST_COUNTER && valid_assign(assign,cnt) == PFMLIB_SUCCESS) goto done;

			for (k=PMU_FIRST_COUNTER; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(cnt_list[2],k);

				if (max_l3 == PMU_FIRST_COUNTER && valid_assign(assign,cnt) == PFMLIB_SUCCESS) goto done;
				for (l=PMU_FIRST_COUNTER; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(cnt_list[3],l);

					if (valid_assign(assign,cnt) == PFMLIB_SUCCESS) goto done;
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
		reg.pmu_reg = 0; /* clear all */
		/* if not specified per event, then use default (could be zero: measure nothing) */
		reg.pmc_plm    = evt->pfp_plm[j] ? evt->pfp_plm[j]: evt->pfp_dfl_plm; 
		reg.pmc_oi     = 1; /* overflow interrupt */
		reg.pmc_pm     = evt->pfp_pm; 
		reg.pmc_thres  = param ? param->pfp_ita_counters[j].thres: 0;
		reg.pmc_ism    = param ? 
			(param->pfp_ita_counters[j].ism ? param->pfp_ita_counters[j].ism : param->pfp_ita_ism) : 0;
		reg.pmc_umask  = is_ear(cnt_list[j]) ? 0x0 : itanium_pe[cnt_list[j]].pme_umask;
		reg.pmc_es     = itanium_pe[cnt_list[j]].pme_code;

		pc[j].reg_num   = assign[j]; 
		pc[j].reg_value = reg.pmu_reg;

		pfm_vbprintf("[pmc%d=0x%06lx thres=%d es=0x%02x plm=%d umask=0x%x pm=%d ism=0x%x oi=%d] %s\n", 
					assign[j], reg.pmu_reg, 
					reg.pmc_thres,
					reg.pmc_es,reg.pmc_plm, 
					reg.pmc_umask, reg.pmc_pm,
					reg.pmc_ism,
					reg.pmc_oi,
					itanium_pe[cnt_list[j]].pme_name);
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_iear(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita_reg_t reg;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	pfmlib_ita_param_t fake_param;
	int pos = *idx;
	int i;

	if (param == NULL || param->pfp_ita_iear.ear_used == 0) {
		for (i=0; i < evt->pfp_count; i++) {
			if (is_iear(evt->pfp_evt[i])) goto found;
		}
		/* nothing to do */
		return PFMLIB_SUCCESS;
found:
		memset(&fake_param, 0, sizeof(fake_param));
		param = &fake_param;

		param->pfp_ita_iear.ear_is_tlb = pfm_ita_is_iear_tlb(evt->pfp_evt[i]);
		param->pfp_ita_iear.ear_umask  = pfm_ita_get_event_umask(evt->pfp_evt[i]);
		param->pfp_ita_iear.ear_ism    = 0x0; /* force both instruction sets */

		DPRINT(("i-ear event with no info\n"));
	}


	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	reg.pmu_reg = 0;

	/* if not specified, then use default (could be zero: measure nothing) */
	reg.pmc10_ita_reg.iear_plm   = param->pfp_ita_iear.ear_plm ? param->pfp_ita_iear.ear_plm: evt->pfp_dfl_plm;
	reg.pmc10_ita_reg.iear_pm    = evt->pfp_pm;
	reg.pmc10_ita_reg.iear_tlb   = param->pfp_ita_iear.ear_is_tlb ? 1 : 0;
	reg.pmc10_ita_reg.iear_umask = param->pfp_ita_iear.ear_umask;
	reg.pmc10_ita_reg.iear_ism   = param->pfp_ita_iear.ear_ism;

	pc[pos].reg_num   = 10;  /* PMC10 is I-EAR config register */
	pc[pos].reg_value = reg.pmu_reg;

	pos++;

	pfm_vbprintf("[pmc10=0x%lx tlb=%s plm=%d pm=%d ism=0x%x umask=0x%x]\n", 
			reg.pmu_reg,
			reg.pmc10_ita_reg.iear_tlb ? "Yes" : "No",
			reg.pmc10_ita_reg.iear_plm,
			reg.pmc10_ita_reg.iear_pm,
			reg.pmc10_ita_reg.iear_ism,
			reg.pmc10_ita_reg.iear_umask);

	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_dear(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita_reg_t reg;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	int pos = *idx;

	if (param->pfp_ita_dear.ear_used == 0) return PFMLIB_SUCCESS;

	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	reg.pmu_reg = 0;

	/* if not specified, then use default (could be zero: measure nothing) */
	reg.pmc11_ita_reg.dear_plm   = param->pfp_ita_dear.ear_plm ? param->pfp_ita_dear.ear_plm : evt->pfp_dfl_plm;
	reg.pmc11_ita_reg.dear_pm    = evt->pfp_pm;
	reg.pmc11_ita_reg.dear_tlb   = param->pfp_ita_dear.ear_is_tlb ? 1 : 0;
	reg.pmc11_ita_reg.dear_ism   = param->pfp_ita_dear.ear_ism;
	reg.pmc11_ita_reg.dear_umask = param->pfp_ita_dear.ear_umask;
	reg.pmc11_ita_reg.dear_pt    = param->pfp_ita_drange.rr_used ? 0: 1;

	pc[pos].reg_num    = 11;  /* PMC11 is D-EAR config register */
	pc[pos].reg_value  = reg.pmu_reg;

	pos++;

	pfm_vbprintf("[pmc11=0x%lx tlb=%s plm=%d pm=%d ism=0x%x umask=0x%x pt=%d]\n", 
			reg.pmu_reg,
			reg.pmc11_ita_reg.dear_tlb ? "Yes" : "No",
			reg.pmc11_ita_reg.dear_plm,	
			reg.pmc11_ita_reg.dear_pm,
			reg.pmc11_ita_reg.dear_ism,
			reg.pmc11_ita_reg.dear_umask,
			reg.pmc11_ita_reg.dear_pt);

	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

static int
pfm_dispatch_opcm(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	pfm_ita_reg_t reg;
	int pos = *idx;

	if (param->pfp_ita_pmc8.opcm_used) {

		if (pos == max_count) return PFMLIB_ERR_FULL;

		reg.pmu_reg = param->pfp_ita_pmc8.pmc_val;

		pc[pos].reg_num     = 8;
		pc[pos++].reg_value = reg.pmu_reg;


		pfm_vbprintf("[pmc8=0x%lx m=%d i=%d f=%d b=%d match=0x%x mask=0x%x]\n",
				reg.pmu_reg,
				reg.pmc8_9_ita_reg.m,
				reg.pmc8_9_ita_reg.i,
				reg.pmc8_9_ita_reg.f,
				reg.pmc8_9_ita_reg.b,
				reg.pmc8_9_ita_reg.match,
				reg.pmc8_9_ita_reg.mask);
	}

	if (param->pfp_ita_pmc9.opcm_used) {

		if (pos == max_count) return PFMLIB_ERR_FULL;

		reg.pmu_reg = param->pfp_ita_pmc9.pmc_val;

		pc[pos].reg_num     = 9;
		pc[pos++].reg_value = reg.pmu_reg;


		pfm_vbprintf("[pmc9 m=%d i=%d f=%d b=%d match=0x%x mask=0x%x]\n",
				reg.pmc8_9_ita_reg.m,
				reg.pmc8_9_ita_reg.i,
				reg.pmc8_9_ita_reg.f,
				reg.pmc8_9_ita_reg.b,
				reg.pmc8_9_ita_reg.match,
				reg.pmc8_9_ita_reg.mask);
	}
	*idx = pos;
	return PFMLIB_SUCCESS;
}


static int
pfm_dispatch_btb(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	int i, pos = *idx;
	pfm_ita_reg_t reg;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	pfmlib_ita_param_t fake_param;
	int found_btb=0;

	reg.pmu_reg = 0;

	if (param == NULL || param->pfp_ita_btb.btb_used == 0) {
		for (i=0; i < evt->pfp_count; i++) {
			/* 
			 * more than one BTB event defined is invalid
			 */
			if (found_btb == 1 && is_btb(evt->pfp_evt[i])) return PFMLIB_ERR_EVTMANY; 

			if (is_btb(evt->pfp_evt[i])) found_btb = 1;
		}
		/* nothing found */
		if (found_btb  == 0) return PFMLIB_SUCCESS;

		memset(&fake_param, 0, sizeof(fake_param));
		param = &fake_param;

		param->pfp_ita_btb.btb_tar = 0x1; 	/* capture TAR  */
		param->pfp_ita_btb.btb_tm  = 0x3; 	/* all branches */
		param->pfp_ita_btb.btb_ptm = 0x3; 	/* all branches */
		param->pfp_ita_btb.btb_ppm = 0x3; 	/* all branches */
		param->pfp_ita_btb.btb_tac = 0x1; 	/* capture TAC  */
		param->pfp_ita_btb.btb_bac = 0x1; 	/* capture BAC  */

		DPRINT(("btb event with no info\n"));
	}

	/* not enough space for this */
	if (pos == max_count) return PFMLIB_ERR_FULL;

	/* if not specified, then use default (could be zero: measure nothing) */
	reg.pmc12_ita_reg.btbc_plm = param->pfp_ita_btb.btb_plm ? param->pfp_ita_btb.btb_plm : evt->pfp_dfl_plm;
	reg.pmc12_ita_reg.btbc_pm  = evt->pfp_pm;
	reg.pmc12_ita_reg.btbc_tar = param->pfp_ita_btb.btb_tar & 0x1;
	reg.pmc12_ita_reg.btbc_tm  = param->pfp_ita_btb.btb_tm  & 0x3;
	reg.pmc12_ita_reg.btbc_ptm = param->pfp_ita_btb.btb_ptm & 0x3;
	reg.pmc12_ita_reg.btbc_ppm = param->pfp_ita_btb.btb_ppm & 0x3;
	reg.pmc12_ita_reg.btbc_bpt = param->pfp_ita_btb.btb_tac & 0x1;
	reg.pmc12_ita_reg.btbc_bac = param->pfp_ita_btb.btb_bac & 0x1;

	memset(pc+pos, 0, sizeof(pfarg_reg_t));
	pc[pos].reg_num     = 12;
	pc[pos++].reg_value = reg.pmu_reg;


	pfm_vbprintf("[pmc12=0x%lx plm=%d pm=%d tar=%d tm=%d ptm=%d ppm=%d bpt=%d bac=%d]\n",
			reg.pmu_reg,
			reg.pmc12_ita_reg.btbc_plm,
			reg.pmc12_ita_reg.btbc_pm,
			reg.pmc12_ita_reg.btbc_tar,
			reg.pmc12_ita_reg.btbc_tm,
			reg.pmc12_ita_reg.btbc_ptm,
			reg.pmc12_ita_reg.btbc_ppm,
			reg.pmc12_ita_reg.btbc_bpt,
			reg.pmc12_ita_reg.btbc_bac);

	/* update final number of entries used */
	*idx = pos;

	return PFMLIB_SUCCESS;
}

/*
 * mode = 0 -> check code (enforce bundle alignment)
 * mode = 1 -> check data
 */
static int
check_intervals(pfmlib_ita_rr_t *rr, int mode, int *n_intervals)
{
	int i;
	pfmlib_ita_rr_desc_t *lim = rr->rr_limits;

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

static void 
do_normal_rr(pfmlib_param_t *evt, unsigned long start, unsigned long end, 
			     pfarg_dbreg_t *br, int nbr, int dir, int *idx, int *reg_idx)
{
#	define ALIGN_DOWN(a,p)	((a) & ~((1UL<<p)-1))
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
print_one_range(pfmlib_ita_rr_t *rr, pfmlib_ita_rr_desc_t *lim, pfarg_dbreg_t *dbr, int n_pairs)
{
	int i, j;
	dbreg_t *d;
	unsigned long r_end;

		printf("[0x%lx-0x%lx): %d register pair(s)\n", 
				lim->rr_start, lim->rr_end,
				n_pairs);
		printf("start offset: -0x%lx end_offset: +0x%lx\n", lim->rr_soff, lim->rr_eoff);

		for (j=0; j < n_pairs; j++) {

			i = j<<1;
			d     = (dbreg_t *)&dbr[i+1].dbreg_value;
			r_end = dbr[i].dbreg_value+((~(d->db.db_mask)) & ~(0xffUL << 56));

			printf("brp%u:  db%u: 0x%016lx db%u: plm=0x%x mask=0x%016lx end=0x%016lx\n", 
				dbr[i].dbreg_num>>1, 
				dbr[i].dbreg_num, 
				dbr[i].dbreg_value, 
				dbr[i+1].dbreg_num, 
				d->db.db_plm, d->db.db_mask,
				r_end);
		}
}

static int
compute_normal_rr(pfmlib_param_t *evt, pfmlib_ita_rr_t *rr, int n)
{
	int i, j, br_index, reg_index, prev_index;
	pfmlib_ita_rr_desc_t *lim;
	unsigned long r_end;
	pfarg_dbreg_t *br = rr->rr_br;
	dbreg_t *d;

	lim       = rr->rr_limits;
	br        = rr->rr_br;
	reg_index = 0;
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

		if (PFMLIB_VERBOSE()) print_one_range(rr, lim, br, (br_index-prev_index)>>1);


	}

	/* do not have enough registers to cover all the ranges */
	if (br_index == 8 && i < n) return PFMLIB_ERR_TOOMANY;

	rr->rr_nbr_used = br_index;

	return PFMLIB_SUCCESS;
}


static int
pfm_dispatch_irange(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfm_ita_reg_t reg;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	pfmlib_ita_rr_t *rr;
	int pos = *idx, ret;
	int n_intervals;

	if (param->pfp_ita_irange.rr_used == 0) return PFMLIB_SUCCESS;

	rr = &param->pfp_ita_irange;

	ret = check_intervals(rr, 0, &n_intervals);
	if (ret != PFMLIB_SUCCESS) return ret;

	if (n_intervals < 1) return PFMLIB_ERR_IRRINVAL;
	
	DPRINT(("n_intervals=%d\n", n_intervals));

	ret = compute_normal_rr(evt, rr, n_intervals);
	if (ret != PFMLIB_SUCCESS) {
		return ret == PFMLIB_ERR_TOOMANY ? PFMLIB_ERR_IRRTOOMANY : ret;
	}
	if (pos == max_count) return PFMLIB_ERR_FULL;

	reg.pmu_reg = 0;

	reg.pmc13_ita_reg.irange_ta = 0x0;

	memset(pc+pos, 0, sizeof(pfarg_reg_t));

	pc[pos].reg_num     = 13;
	pc[pos++].reg_value = reg.pmu_reg;

	pfm_vbprintf("[pmc13=0x%lx ta=%d]\n", reg.pmu_reg, reg.pmc13_ita_reg.irange_ta);
	
	*idx = pos;

	return PFMLIB_SUCCESS;
}
	
static int
pfm_dispatch_drange(pfmlib_param_t *evt, pfarg_reg_t *pc, int *idx, int max_count)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	int *cnt_list = evt->pfp_evt;
	pfm_ita_reg_t reg;
	pfmlib_ita_rr_t *rr;
	int i, pos = *idx, ret;
	int n_intervals;

	if (param->pfp_ita_drange.rr_used == 0) return PFMLIB_SUCCESS;

	rr = &param->pfp_ita_drange;

	ret = check_intervals(rr, 1 , &n_intervals);
	if (ret != PFMLIB_SUCCESS) return ret;

	if (n_intervals < 1) return PFMLIB_ERR_IRRINVAL;
	
	DPRINT(("n_intervals=%d\n", n_intervals));

	ret = compute_normal_rr(evt, rr, n_intervals);
	if (ret != PFMLIB_SUCCESS) {
		return ret == PFMLIB_ERR_TOOMANY ? PFMLIB_ERR_DRRTOOMANY : ret;
	}

	for (i=0; i < evt->pfp_count; i++) {
		if (is_dear(cnt_list[i])) return PFMLIB_SUCCESS; /* will be done there */
	}

	if (pos == max_count) return PFMLIB_ERR_FULL;

	/*
	 * this will clear the pt field which is what we want for drange
	 * (reg.pmc11_ita_reg.dear_pt = 0)
	 *
	 * XXX: could even get rid of reg altogether here
	 */
	memset(pc+pos, 0, sizeof(pfarg_reg_t));

	reg.pmu_reg = 0UL;

	pc[pos].reg_num     = 11;
	pc[pos++].reg_value = reg.pmu_reg;

	pfm_vbprintf("[pmc11=0x%lx tlb=%s plm=%d pm=%d ism=0x%x umask=0x%x pt=%d]\n", 
			reg.pmu_reg,
			reg.pmc11_ita_reg.dear_tlb ? "Yes" : "No",
			reg.pmc11_ita_reg.dear_plm,	
			reg.pmc11_ita_reg.dear_pm,
			reg.pmc11_ita_reg.dear_ism,
			reg.pmc11_ita_reg.dear_umask,
			reg.pmc11_ita_reg.dear_pt);


	*idx = pos;

	return PFMLIB_SUCCESS;
}

static int
check_qualifier_constraints(pfmlib_param_t *evt)
{
	int i;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	int *cnt_list = evt->pfp_evt;

	for(i=0; i < evt->pfp_count; i++) {
		if (evt_use_irange(param) && has_iarr(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
		if (evt_use_drange(param) && has_darr(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
		if (evt_use_opcm(param) && has_opcm(cnt_list[i]) == 0) return PFMLIB_ERR_FEATCOMB;
	}
	return PFMLIB_SUCCESS;
}

static int
pfm_ita_dispatch_events(pfmlib_param_t *evt, pfarg_reg_t *pc, int *count)
{
	int idx, ret;
	int max_count;
	pfmlib_ita_param_t *p;

	p = ITA_PARAM(evt);

	/* sanity check */
	if (p && p->pfp_magic != PFMLIB_ITA_PARAM_MAGIC) return PFMLIB_ERR_MAGIC;

	/*
	 * evt->pfp_count can be zero, in which case, the caller is only
	 * interested in setting up non counter registers.
	 */

	/* not enough room in pc[] for requested number of counters */
	if (*count < evt->pfp_count) return PFMLIB_ERR_FULL;

	/* check opcode match, range restriction qualifiers */
	if (p && check_qualifier_constraints(evt) != PFMLIB_SUCCESS) return PFMLIB_ERR_FEATCOMB;

	ret = pfm_ita_dispatch_counters(evt, pc);
	if (ret != PFMLIB_SUCCESS) return ret;

	max_count = *count;

	/* index of first free entry */
	idx       = evt->pfp_count;


	/* now check for I-EAR */
	ret = pfm_dispatch_iear(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	if (p == NULL) goto no_special_features;

	/* now check for D-EAR */
	ret = pfm_dispatch_dear(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	/* now check for Opcode matchers */
	ret = pfm_dispatch_opcm(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = pfm_dispatch_btb(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = pfm_dispatch_irange(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

	ret = pfm_dispatch_drange(evt, pc, &idx, max_count);
	if (ret != PFMLIB_SUCCESS) return ret;

no_special_features:
	/* how many entries are used for the setup */
	*count = idx;

	return PFMLIB_SUCCESS;
}


/* XXX: return value is also error code */
int
pfm_ita_event_maxincr(int i, unsigned long *maxincr)
{
	if (i<0 || i >= PME_ITA_COUNT || maxincr == NULL) return PFMLIB_ERR_INVAL;
	*maxincr = itanium_pe[i].pme_maxincr;
	return PFMLIB_SUCCESS;
}

int
pfm_ita_is_ear(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! is_ear(i) ? 0 : 1;
}

int
pfm_ita_is_dear(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! is_dear(i) ? 0 : 1;
}

int
pfm_ita_is_dear_tlb(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! (is_dear(i) && is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_ita_is_dear_cache(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! (is_dear(i) && !is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_ita_is_iear(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! is_iear(i) ? 0 : 1;
}

int
pfm_ita_is_iear_tlb(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! (is_iear(i) && is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_ita_is_iear_cache(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! (is_iear(i) && !is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_ita_is_btb(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! is_btb(i) ? 0 : 1;
}

int
pfm_ita_support_iarr(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! has_iarr(i) ? 0 : 1;
}


int
pfm_ita_support_darr(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! has_darr(i) ? 0 : 1;
}


int
pfm_ita_support_opcm(int i)
{
	return i < 0 || i >= PME_ITA_COUNT || ! has_opcm(i) ? 0 : 1;
}

/*
 * Function used to print information about a specific events. More than
 * one event can be printed in case an event code is given rather than
 * a specific name. A callback function is used for printing.
 */
static int 
pfm_ita_print_info(int v, int (*pf)(const char *fmt,...)) 
{
	pme_ita_entry_t *e;
        const char *quals[]={ "[Instruction Address Range]", "[OpCode Match]", "[Data Address Range]" };
	long c;
        int i;

	if (v < 0 || v >= PME_ITA_COUNT || pf == NULL) return PFMLIB_ERR_INVAL;
	e = itanium_pe+v;

	(*pf)("Umask  : ");
	c = e->pme_umask;
	for (i=3; i >=0; i--) {
		(*pf)("%d", c & 1<<i ? 1 : 0);
	}
	(*pf)("\n");

	(*pf)( "EAR    : %s (%s)\n",
		e->pme_ear ? (e->pme_dear ? "Data" : "Inst") : "No",
		e->pme_ear ? (e->pme_tlb ? "TLB Mode": "Cache Mode"): "N/A");
	

	(*pf)("BTB    : %s\n", e->pme_btb ? "Yes" : "No");

	if (e->pme_maxincr > 1) 
		(*pf)("MaxIncr: %u  (Threshold [0-%u])\n", e->pme_maxincr,  e->pme_maxincr-1);
 	else 
		(*pf)("MaxIncr: %u  (Threshold 0)\n", e->pme_maxincr);

	(*pf)("Qual   : ");

	c = e->pme_qualifiers.qual;
	if ((c & 0x7) == 0) {
		(*pf)("None");
	} else {
		for (i=0; i < 3; i++ ) {
                	if (c & 0x1) (*pf)("%s ", quals[i]);
                	c >>= 1;
        	}
	}
	(*pf)("\n");
	return PFMLIB_SUCCESS;
}

static unsigned long
pfm_ita_get_event_code(int i)
{
	return itanium_pe[i].pme_code;
}

static unsigned long
pfm_ita_get_event_vcode(int i)
{
	return itanium_pe[i].pme_entry_code.pme_vcode;
}

unsigned long
pfm_ita_get_event_umask(int i)
{
	return itanium_pe[i].pme_umask;
}

static char *
pfm_ita_get_event_name(int i)
{
	return itanium_pe[i].pme_name;
}

static unsigned long
pfm_ita_get_event_counters(int i)
{
	return itanium_pe[i].pme_counters;
}

static int
pfm_ita_num_counters(void)
{
	return  PMU_ITA_MAX_COUNTERS;
}

pfm_pmu_support_t itanium_support={
		"Itanium",
		PFMLIB_ITANIUM_PMU,
		PME_ITA_COUNT,
		pfm_ita_get_event_code,
		pfm_ita_get_event_vcode,
		pfm_ita_get_event_name,
		pfm_ita_get_event_counters,
		pfm_ita_print_info,
		pfm_ita_dispatch_events,
		pfm_ita_num_counters,
		pfm_ita_detect
};
