/*
 * 
 *
 * Copyright (C) 2001 Hewlett-Packard Co
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
#include <errno.h>
#include <syscall.h>
#include <signal.h>

#include "perfmon.h"
#include "pfmlib.h"

#include "pfm_private.h"

#include "pme_list.h"

#define is_ear(i)	event_is_ear(pe+(i))
#define is_tlb_ear(i)	event_is_tlb_ear(pe+(i))
#define is_iear(i)	event_is_iear(pe+(i))
#define is_dear(i)	event_is_dear(pe+(i))
#define is_btb(i)	event_is_btb(pe+(i))

/*
 * contains runtime configuration options for the library.
 * mostly for debug purposes.
 */
static pfmlib_options_t pfm_options;

int
pfmlib_config(pfmlib_options_t *opt)
{
	/* probably needs checking */
	pfm_options = *opt;
	return 0;
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
 * Upon return the perfmon_req_t structure is ready to be submitted to kernel
 */
int
pfm_dispatch_counters(pfm_event_config_t *evt, perfmon_req_t *pc)
{
	int i,j,k,l, m;
	unsigned int max_l1, max_l2, max_l3;
	int assign[PMU_MAX_COUNTERS];
	perfmon_reg_t reg;
	int *cnt_list = evt->pec_evt;
	unsigned int *thres_list = evt->pec_thres;
	unsigned int cnt = evt->pec_count;

#define	has_counter(e,b)	(pe[e].pme_counters & (1 << b) ? PMU_FIRST_COUNTER+b : 0)

	if (pfm_options.pfm_debug)
		for (m=0; m < cnt; m++) {
			printf("ev[%d]=%s counters=0x%lx\n", m, pe[cnt_list[m]].pme_name, pe[cnt_list[m]].pme_counters);
		}

	max_l1 = PMU_MAX_COUNTERS*(cnt>1);
	max_l2 = PMU_MAX_COUNTERS*(cnt>2);
	max_l3 = PMU_MAX_COUNTERS*(cnt>3);

	/*
	 *  This code needs fixing. It is not very pretty and 
	 *  won't handle more than 4 counters if more become
	 *  available !
	 *  For now, worst case in the loop nest: 4! (factorial)
	 */
	for (i=0; i < PMU_MAX_COUNTERS; i++) {

		assign[0]= has_counter(cnt_list[0],i);

		if (!max_l1 && valid_assign(assign,cnt)) goto done;

		for (j=0; j < max_l1; j++) {

			if (j == i) continue;

			assign[1] = has_counter(cnt_list[1],j);

			if (!max_l2 && valid_assign(assign,cnt)) goto done;

			for (k=0; k < max_l2; k++) {

				if(k == i || k == j) continue;

				assign[2] = has_counter(cnt_list[2],k);

				if (!max_l3 && valid_assign(assign,cnt)) goto done;
				for (l=0; l < max_l3; l++) {

					if(l == i || l == j || l == k) continue;

					assign[3] = has_counter(cnt_list[3],l);

					if (valid_assign(assign,cnt)) goto done;
				}
			}
		}
	}
	/* we cannot satisfy the constraints */
	return -1;
done:
	/* cleanup the array */
	memset(pc, 0, cnt*sizeof(perfmon_req_t));

	for (j=0; j < cnt ; j++ ) {
		reg.pmu_reg = 0; /* clear all */

		if ((evt->pec_pmc8 !=  PFM_OPC_MATCH_ALL || evt->pec_pmc9 != PFM_OPC_MATCH_ALL)
		   && event_opcm_ok(pe+cnt_list[j]) == 0) {
			printf("Event %s does not support opcode matching\n", pe[cnt_list[j]].pme_name);
			return -1;
		}
		
		reg.pmc_plm    = evt->pec_plm; /* XXX fixme: per counter */
		reg.pmc_oi     = 1; /* overflow interrupt */
		reg.pmc_pm     = 0; /* not a privileged monitor */
		reg.pmc_thres  = thres_list[j];
		reg.pmc_ism    = 0; /* ia-64 instruction set */
		reg.pmc_umask  = is_ear(cnt_list[j]) ? 0x0 : pe[cnt_list[j]].pme_umask;
		reg.pmc_es     = pe[cnt_list[j]].pme_code;

		pc[j].pfr_reg.reg_num   = assign[j]; 
		pc[j].pfr_reg.reg_value = reg.pmu_reg;

		/* must be an option for non EARS */
		if (is_ear(cnt_list[j]) || is_btb(cnt_list[j])) pc[j].pfr_reg.reg_flags = PFM_REGFL_OVFL_NOTIFY;

		if (pfm_options.pfm_debug) {
			printf("[cnt=0x%x,thres=%d,es=0x%x,val=0x%lx flags=0x%x] %s\n", 
					assign[j], thres_list[j], reg.pmc_es, 
					reg.pmu_reg, 
					pc[j].pfr_reg.reg_flags,
					pe[cnt_list[j]].pme_name);
		}
	}
	return 0;
}

static int
pfm_dispatch_ears(pfm_event_config_t *evt, perfmon_req_t *pc, int pos, int *count)
{
	int i;
	perfmon_reg_t reg;
	int *cnt_list = evt->pec_evt;
	char done_dear=0, done_iear=0;

	for (i=0; i < evt->pec_count; i++) {

		if (!is_ear(cnt_list[i])) continue;

		/* not enough space for this */
		if (pos == *count) return -1;

		reg.pmu_reg = 0;
		/*
		 * does not really make sense to have separate pid for IEAR and DEAR
		 */
		if (is_iear(evt->pec_evt[i])) {
			/* cannot measure 2 I-EAR at the same time */
			if (done_iear) return -1;

			reg.pmc10_reg.iear_plm   = evt->pec_plm; /* XXX fixme: per counter */
			reg.pmc10_reg.iear_pm    = 0; /* not a privileged monitor */
			reg.pmc10_reg.iear_tlb   = is_tlb_ear(cnt_list[i]) ? 1 : 0;
			reg.pmc10_reg.iear_umask = pe[cnt_list[i]].pme_umask;
			reg.pmc10_reg.iear_ism   = 0; /* ia-64 instruction set */

			pc[pos].pfr_reg.reg_num   = 10;  /* PMC10 is I-EAR config register */
			pc[pos].pfr_reg.reg_value = reg.pmu_reg;

			done_iear = 1;

			if (pfm_options.pfm_debug) {
				printf(__FUNCTION__" pmc10=0x%lx\nI-EAR:  TLB: %s, PLM: %d, UMASK: 0x%x\n", 
							reg.pmu_reg,
							reg.pmc10_reg.iear_tlb ? "Yes" : "No",
							reg.pmc10_reg.iear_plm,
							reg.pmc10_reg.iear_umask);
			}
		} else {
			/* cannot measure 2 D-EAR at the same time */
			if (done_dear) return -1;

			reg.pmc11_reg.dear_plm   = evt->pec_plm; /* XXX fixme: per counter */
			reg.pmc11_reg.dear_pm    = 0; /* not a privileged monitor */
			reg.pmc11_reg.dear_tlb   = is_tlb_ear(cnt_list[i]) ? 1 : 0;
			reg.pmc11_reg.dear_ism   = 0; /* ia-64 instruction set */
			reg.pmc11_reg.dear_umask = pe[cnt_list[i]].pme_umask;
			reg.pmc11_reg.dear_pt    = 1; /* XXX fixme: coordinate with Data Range Check */

			pc[pos].pfr_reg.reg_num    = 11;  /* PMC11 is D-EAR config register */
			pc[pos].pfr_reg.reg_value  = reg.pmu_reg;

			done_dear = 1;	

			if (pfm_options.pfm_debug) {
				printf(__FUNCTION__" pmc11=0x%lx\npos=%d D-EAR:  TLB: %s, PLM: %d, UMASK: 0x%x\n", 
							reg.pmu_reg,
							pos,
							reg.pmc11_reg.dear_tlb ? "Yes" : "No",
							reg.pmc11_reg.dear_plm,
							reg.pmc11_reg.dear_umask);
			}

		}
		pos++;
	}
	/* update final number of entries used */
	*count = pos;
	return 0;
}

static int
pfm_dispatch_opcm(pfm_event_config_t *evt, perfmon_req_t *pc, int pos, int *count)
{
	if (evt->pec_pmc8 != PFM_OPC_MATCH_ALL) {
		if (pos == *count) return -1;
		pc[pos].pfr_reg.reg_num     = 8;
		pc[pos++].pfr_reg.reg_value = evt->pec_pmc8; 

		if (pfm_options.pfm_debug) {
			perfmon_reg_t reg;

			reg.pmu_reg = evt->pec_pmc8;

			printf("PMC8: pmc8=0x%lx\n",evt->pec_pmc8);
			printf("PMC8: m=%d i=%d f=%d b=%d match=0x%x mask=0x%x\n",
				reg.pmc8_9_reg.m,
				reg.pmc8_9_reg.i,
				reg.pmc8_9_reg.f,
				reg.pmc8_9_reg.b,
				reg.pmc8_9_reg.match,
				reg.pmc8_9_reg.mask);
		}

	}

	if (evt->pec_pmc9 != PFM_OPC_MATCH_ALL) {
		if (pos == *count) return -1;
		pc[pos].pfr_reg.reg_num     = 9;
		pc[pos++].pfr_reg.reg_value = evt->pec_pmc9; 

		if (pfm_options.pfm_debug) {
			perfmon_reg_t reg;

			reg.pmu_reg = evt->pec_pmc9;

			printf("PMC9: m=%d i=%d f=%d b=%d match=0x%x mask=0x%x\n",
				reg.pmc8_9_reg.m,
				reg.pmc8_9_reg.i,
				reg.pmc8_9_reg.f,
				reg.pmc8_9_reg.b,
				reg.pmc8_9_reg.match,
				reg.pmc8_9_reg.mask);
		}

	}
	*count = pos;
	return 0;
}


static int
pfm_dispatch_btb(pfm_event_config_t *evt, perfmon_req_t *pc, int pos, int *count)
{
	int i;
	perfmon_reg_t reg;
	int *cnt_list = evt->pec_evt;
	char done_btb=0;

	reg.pmu_reg = 0;

	for (i=0; i < evt->pec_count; i++) {
		if (!is_btb(cnt_list[i])) continue;

		/* we can only have one BTB event */
		if (done_btb) return -1;

		/* not enough space for this */
		if (pos == *count) return -1;

		reg.pmc12_reg.btbc_plm = evt->pec_plm; /* XXX: should be per counter */
		reg.pmc12_reg.btbc_pm  = 0;		/* XXX: should clarify this */
		reg.pmc12_reg.btbc_tar = evt->pec_btb_tar & 0x1;
		reg.pmc12_reg.btbc_tm  = evt->pec_btb_tm & 0x3;
		reg.pmc12_reg.btbc_ptm = evt->pec_btb_ptm & 0x3;
		reg.pmc12_reg.btbc_ppm = evt->pec_btb_ppm & 0x3;
		reg.pmc12_reg.btbc_bpt = evt->pec_btb_tac & 0x1;
		reg.pmc12_reg.btbc_bac = evt->pec_btb_bac & 0x1;

		pc[pos].pfr_reg.reg_num     = 12;
		pc[pos++].pfr_reg.reg_value = reg.pmu_reg;

		done_btb = 1;

		if (pfm_options.pfm_debug) {
			printf("PMC12=0x%lx plm=%d tar=%d tm=%d ptm=%d ppm=%d bpt=%d bac=%d\n",
					reg.pmu_reg,
					reg.pmc12_reg.btbc_plm,
					reg.pmc12_reg.btbc_tar,
					reg.pmc12_reg.btbc_tm,
					reg.pmc12_reg.btbc_ptm,
					reg.pmc12_reg.btbc_ppm,
					reg.pmc12_reg.btbc_bpt,
					reg.pmc12_reg.btbc_bac);
		}

	}
	/* update final number of entries used */
	*count = pos;
	return 0;
}

int
pfm_dispatch_events(pfm_event_config_t *evt, perfmon_req_t *pc, int *count)
{
	int ret;
	int max_count, tmp;
	if (evt == NULL || pc == NULL || count == NULL) return -1;

	/* check for nothing to do */
	if (evt->pec_count == 0) {
		*count = 0;
		return 0;
	}
	/* not enough slots in command, will return minimum required */
	if (*count < evt->pec_count) {
		*count = evt->pec_count;
		return -1;
	}

	if (pfm_dispatch_counters(evt, pc) == -1) {
		*count = 0;
		return -1;
	}
	tmp = max_count = *count;

	/* now check for EARS */
	if (pfm_dispatch_ears(evt, pc, evt->pec_count, &tmp) == -1) return -1;

	ret = tmp;
	tmp = max_count;

	/* now check for Opcode matchers */
	if (pfm_dispatch_opcm(evt, pc, ret, &tmp) == -1) return -1;

	ret = tmp;

	if (pfm_dispatch_btb(evt, pc, ret, &max_count) == -1) return -1;

	/* how many entries are used for the setup */
	*count = max_count;

	return 0;
}

static int
gen_event_code(char *str)
{
	int base;
	long ret;
	char *endptr = NULL;

	base = strlen(str) > 1 && str[1] == 'x' ? 16 : 10;

	ret = strtol(str,&endptr,base);

	/* check for errors */
	if (*endptr!='\0') return -1;

	return (int)ret;
}


int
pfm_findeventbyname(char *n)
{
	int i;

	/* we do case insensitive comparisons */
	for(i=0; i < PME_COUNT; i++ ) {
		if (!strcasecmp(pe[i].pme_name, n)) return i;
	}
	return -1;
}

int
pfm_findeventbyvcode(int code)
{
	int i;

	for(i=0; i < PME_COUNT; i++ ) {
		if (pe[i].pme_vcode == code) return i;
	}
	return -1;
}

int
pfm_findeventbycode(int code)
{
	int i;

	for(i=0; i < PME_COUNT; i++ ) {
		if (pe[i].pme_code == code) return i;
	}
	return -1;
}

int
pfm_findevent(char *v, int retry)
{
	int num;
	int ev;

	if (isdigit(*v)) {
		if ((num = gen_event_code(v)) == -1) return -1;
		ev = pfm_findeventbyvcode(num);
		if (ev == -1 && retry) ev = pfm_findeventbycode(num);
		
	} else 
		ev = pfm_findeventbyname(v);

	return ev;
}

int
pfm_findeventbycode_next(int code, int i)
{
	for(++i; i < PME_COUNT; i++ ) {
		if (pe[i].pme_code == code) return i;
	}
	return -1;
}

int
pfm_findeventbyvcode_next(int code, int i)
{
	for(++i; i < PME_COUNT; i++ ) {
		if (pe[i].pme_vcode == code) return i;
	}
	return -1;
}


/* XXX: return value is also error code */
int
pfm_event_threshold(int i)
{
	return (i<0 || i >= PME_COUNT) ? -1 : pe[i].pme_thres;
}

char *
pfm_event_name(int i)
{
	return (i<0 || i >= PME_COUNT) ? "Unknown" : pe[i].pme_name;
}

int
pfm_get_firstevent(void)
{
	return 0; /* could be different */
}

int
pfm_get_nextevent(int i)
{
	/* may extend to check validity on particular CPU */
	return i >= PME_COUNT-1 ? -1 : i+1;
}

int
pfm_is_ear(int i)
{
	return i < 0 || i >= PME_COUNT || ! is_ear(i) ? 0 : 1;
}

int
pfm_is_dear(int i)
{
	return i < 0 || i >= PME_COUNT || ! is_dear(i) ? 0 : 1;
}

int
pfm_is_dear_tlb(int i)
{
	return i < 0 || i >= PME_COUNT || ! (is_dear(i) && is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_is_dear_cache(int i)
{
	return i < 0 || i >= PME_COUNT || ! (is_dear(i) && !is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_is_iear(int i)
{
	return i < 0 || i >= PME_COUNT || ! is_iear(i) ? 0 : 1;
}

int
pfm_is_iear_tlb(int i)
{
	return i < 0 || i >= PME_COUNT || ! (is_iear(i) && is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_is_iear_cache(int i)
{
	return i < 0 || i >= PME_COUNT || ! (is_iear(i) && !is_tlb_ear(i)) ? 0 : 1;
}
	
int
pfm_is_btb(int i)
{
	return i < 0 || i >= PME_COUNT || ! is_btb(i) ? 0 : 1;
}


/*
 * Function used to print information about a specific events. More than
 * one event can be printed in case an event code is given rather than
 * a specific name. A callback function is used for printing.
 */
int
pfm_print_event_info(char *name, int (*pf)(const char *fmt,...))
{
	pme_entry_t *e;
        const char *quals[]={ "[Instruction Address Range]", "[OpCode match]", "[Data Address Range]" };
	int (*find_next)(int code, int i);
	long c;
        int i, code;
	int full_name_used, code_is_used = 0;
	int v, vbis, num;


	/* we can't quite use pfm_findevent() because we need to try
	 * both ways systematically.
	 */
	if (isdigit(*name)) {
		if ((num = gen_event_code(name)) == -1) return -1;
		v    = pfm_findeventbyvcode(num);
		vbis = pfm_findeventbycode(num);
		
	} else {
		v = pfm_findeventbyname(name);
		vbis = -1;
	}
	if (v == -1 && vbis == -1) return -1;

	/*
	 * This is code is to work around a tricky case
	 * where code == vcode (when umask==0)
	 */
	if (v != -1 && v != vbis) {
		find_next = pfm_findeventbyvcode_next;
	} else {
		v = vbis;
		code_is_used =1;
		find_next = pfm_findeventbycode_next;
	}

	full_name_used = !isdigit(*name) ? 1: 0;

	e = pe+v;
	
	code = code_is_used ? e->pme_code : e->pme_vcode;

	do {	
		e = pe+v;

		(*pf)(	"Name   : %s\n" 
			"VCode  : 0x%x\n"
			"Code   : 0x%x\n"
			"EAR    : %s (%s) ",
			e->pme_name,
			e->pme_vcode,
			e->pme_code,
			e->pme_ear ? (e->pme_dear ? "Data" : "Inst") : "No",
			e->pme_ear ? (e->pme_tlb ? "TLB Mode": "Cache Mode"): "N/A");
		
		(*pf)("Umask: ");
		c = e->pme_umask;
		for (i=3; i >=0; i--) {
			(*pf)("%d", c & 1<<i ? 1 : 0);
		}
		(*pf)("\n");

		if (e->pme_umask==PME_UMASK_NONE || is_ear(e-pe))
			(*pf)("Umask  : None\n");
		else {
			(*pf)("Umask  : ");
			c = e->pme_umask;
			for (i=3; i >=0; i--) {
				(*pf)("%d", c & 1<<i ? 1 : 0);
			}
			putchar('\n');
		}

		(*pf)(	"PMD/PMC: [");

		c = e->pme_counters;
		for (i=0; i < PMU_MAX_COUNTERS; i++ ) {
			if (c & 0x1) (*pf)("%d ", PMU_FIRST_COUNTER+i);
			c>>=1;
		}

		/* \b is not very pretty ! */
		(*pf)(	"\b]\n"
			"Incr   : %u\n"
			"Qual   : ",
			e->pme_thres);

		c = e->pme_qualifiers.qual;
		if (c == 0)
			(*pf)("None");
	        else
			for (i=0; i < 8*sizeof(int); i++ ) {
                		if (c & 0x1) (*pf)("%s ", quals[i]);
                		c >>= 1;
        		}
		putchar('\n');
		putchar('\n');
	} while (!full_name_used && (v=find_next(code, (int)(e-pe))) != -1);

	return 0;
}

/*
 * once this API is finalized, we should implement this in GNU libc
 */
int
perfmonctl(int pid, int cmd, int flags, perfmon_req_t *ops, int count)
{
	return syscall(__NR_perfmonctl, pid, cmd, flags, ops, count,0);
}
