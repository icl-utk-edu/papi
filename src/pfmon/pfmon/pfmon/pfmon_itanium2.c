/*
 * pfmon_itanium2.c - Itanium2 PMU support for pfmon
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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium2.h>

#include "pfmon.h"
#include "pfmon_itanium2.h"

#define PFMON_SUPPORT_NAME	"Itanium2"

#define M_PMD(x)		(1UL<<(x))
#define DEAR_REGS_MASK		(M_PMD(2)|M_PMD(3)|M_PMD(17))
#define DEAR_ALAT_REGS_MASK	(M_PMD(3)|M_PMD(17))
#define IEAR_REGS_MASK		(M_PMD(0)|M_PMD(1))
#define BTB_REGS_MASK		(M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))

static pfmon_ita2_options_t pfmon_ita2_opt;	/* keep track of global program options */
static pfmlib_ita2_param_t pfmlib_ita2_param;

static void
gen_thresholds(char *arg, pfmlib_param_t *evt)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	char *p;
	int cnt=0;
	unsigned long thres, maxincr;

	/*
	 * the default value for the threshold is 0: this means at least once 
	 * per cycle.
	 */
	if (arg == NULL) {
		int i;
		for (i=0; i < evt->pfp_event_count; i++) param->pfp_ita2_counters[i].thres = 0;
		return;
	}
	while (arg) {

		if (cnt == options.max_counters || cnt == evt->pfp_event_count) goto too_many;

		p = strchr(arg,',');

		if ( p ) *p++ = '\0';

		thres = atoi(arg);

		/*
		 *  threshold = multi-occurence -1
		 * this is because by setting threshold to n, one counts only
		 * when n+1 or more events occurs per cycle.
	 	 */

		pfm_ita2_get_event_maxincr(evt->pfp_events[cnt].event, &maxincr);
		if (thres > (maxincr-1)) goto too_big;

		param->pfp_ita2_counters[cnt++].thres = thres;

		arg = p;
	}
	return;
too_big:
	fatal_error("event %d: threshold must be in [0-%d)\n", cnt, maxincr);
too_many:
	fatal_error("too many thresholds specified\n");
}

static char *retired_events[]={
	"IA64_TAGGED_INST_RETIRED_IBRP0_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP1_PMC9",
	"IA64_TAGGED_INST_RETIRED_IBRP2_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP3_PMC9",
	NULL
};

static void
check_ibrp_events(pfmlib_param_t *evt)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	unsigned long umasks_retired[4];
	unsigned long umask;
	int j, i, seen_retired, ibrp, code, idx;
	int retired_code, incr;
	
	/*
	 * in fine mode, it is enough to use the event
	 * which only monitors the first debug register
	 * pair. The two pairs making up the range
	 * are guaranteed to be consecutive in rr_br[].
	 */
	incr = pfm_ita2_irange_is_fine(evt) ? 4 : 2;

	for (i=0; retired_events[i]; i++) {
		pfm_find_event(retired_events[i], &idx);
		pfm_ita2_get_event_umask(idx, umasks_retired+i);
	}

	pfm_get_event_code(idx, &retired_code);

	/*
	 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
	 * not completely cover the all the debug register pairs used to make up the range.
	 * This could otherwise lead to misinterpretation of the results.
	 */
	for (i=0; i < param->pfp_ita2_irange.rr_nbr_used; i+= incr) {

		ibrp = param->pfp_ita2_irange.rr_br[i].dbreg_num >>1;

		seen_retired = 0;
		for(j=0; j < evt->pfp_event_count; j++) {
			pfm_get_event_code(evt->pfp_events[j].event, &code);
			if (code != retired_code) continue;
			seen_retired = 1;
			pfm_ita2_get_event_umask(evt->pfp_events[j].event, &umask);
			if (umask == umasks_retired[ibrp]) break;
		}
		if (seen_retired && j == evt->pfp_event_count)
			warning("warning: code range uses IBR pair %d which is not monitored using %s\n", ibrp, retired_events[ibrp]);
	}
}

static void
install_irange(int pid, pfmlib_param_t *evt)
{
	int  r;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	check_ibrp_events(evt);

	r = perfmonctl(pid, PFM_WRITE_IBRS, param->pfp_ita2_irange.rr_br, param->pfp_ita2_irange.rr_nbr_used);
	if (r == -1) fatal_error("cannot install code range restriction: %s\n", strerror(errno));
}

static void
install_drange(int pid, pfmlib_param_t *evt)
{
	int r;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	r = perfmonctl(pid, PFM_WRITE_DBRS, param->pfp_ita2_drange.rr_br, param->pfp_ita2_drange.rr_nbr_used);
	if (r == -1) fatal_error("cannot install data range restriction: %s\n", strerror(errno));
}

static void
prepare_pmd16(pid_t pid, pfmlib_param_t *evt)
{
	pfarg_reg_t pdx[1];
	int i, r;

	/*
	 * First of all we reset pmd16
	 */
	memset(pdx, 0, sizeof(pdx));

	pdx[0].reg_num = 16;

	r = perfmonctl(pid, PFM_WRITE_PMDS, pdx, 1);
	if (r == -1) fatal_error("cannot reset pmd16: %s\n", strerror(errno));

	/*
	 * Next, we search all occurences of BRANCH_EVENT and add PMD16 to the
	 * list of other registers to reset on overflow.
	 */
	for(i=0; i < evt->pfp_event_count; i++) {
		if (pfm_ita2_is_btb(evt->pfp_events[i].event)) {
			/*
			 * add pmd16 to the set of pmds to reset when pd[i].reg_num 
			 * overflows
			 */
			evt->pfp_pc[i].reg_reset_pmds[0] |= 1UL << 16;

			/*
			 * pfmon uses the BTB for sampling, so we must make sure we get the overflow
			 * notification on BRANCH_EVENT. This is not done by the library (anymore).
			 */
			evt->pfp_pc[i].reg_flags |= PFM_REGFL_OVFL_NOTIFY;

			DPRINT(("added pmd16 to pmd%u\n", evt->pfp_pc[i].reg_num));
		}
	}
}

static void
fixup_ears(pid_t pid, pfmlib_param_t *evt)
{
	int i;

	/*
	 * search all events for an EAR event
	 */
	for(i=0; i < evt->pfp_event_count; i++) {
		if (pfm_ita2_is_ear(evt->pfp_events[i].event)) {
			/*
			 * pfmon uses the EAR for sampling, so we must make sure we get the overflow
			 * notification on DATA_EAR_* or INSTRUCTION_EAR_*. This is not done by the 
			 * library (anymore).
			 */
			evt->pfp_pc[i].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
			/*
			 * for D-EAR, we must clear PMD3.stat and PMD17.vl to make
			 * sure we do not interpret the register in the wrong manner.
			 *
			 * This is only relevant when the D-EAR is not used as the
			 * sampling period, i.e. in free running mode.
			 *
			 * XXX: do this only when not used for sampling period
			 */
			if (pfm_ita2_is_dear(evt->pfp_events[i].event)) 
				evt->pfp_pc[i].reg_reset_pmds[0] |= 1UL << 3 | 1UL << 17;
		}
	}
}

/*
 * Executed in the context of the child, this is the last chance to modify programming
 * before the PMC and PMD register are written.
 */
static int
pfmon_ita2_install_counters(int pid, pfmlib_param_t *evt, pfarg_reg_t *pd)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	if (param->pfp_ita2_irange.rr_used) install_irange(pid, evt);

	if (param->pfp_ita2_drange.rr_used) install_drange(pid, evt);

	if (param->pfp_ita2_btb.btb_used) prepare_pmd16(pid, evt);

	fixup_ears(pid, evt);

	return 0;
}

static void
pfmon_ita2_usage(void)
{
	printf(
		"--event-thresholds=thr1,thr2,...\tset event thresholds (no space)\n"
		"--opc-match8=val\t\t\tset opcode match for pmc8\n"
		"--opc-match9=val\t\t\tset opcode match for pmc9\n"
		"--btb-tm-tk\t\t\t\tcapture taken IA-64 branches only\n"
		"--btb-tm-ntk\t\t\t\tcapture not taken IA-64 branches only\n"
		"--btb-ptm-correct\t\t\tcapture branch if target predicted correctly\n"
		"--btb-ptm-incorrect\t\t\tcapture branch if target is mispredicted\n"
		"--btb-ppm-correct\t\t\tcapture branch if path is predicted correctly\n"
		"--btb-ppm-incorrect\t\t\tcapture branch if path is mispredicted\n"
		"--btb-ds-pred\t\t\t\tcapture info about branch predictions\n"
		"--btb-brt-iprel\t\t\t\tcapture IP-relative branches only\n"
		"--btb-brt-ret\t\t\t\tcapture return branches only\n"
		"--btb-brt-ind\t\t\t\tcapture non-return indirect branches only\n"
		"--btb-all-mispredicted\t\t\tcapture all mispredicted branches\n"
		"--irange=start-end\t\t\tspecify an instruction address range constraint\n"
		"--drange=start-end\t\t\tspecify a data address range constraint\n"
		"--checkpoint-func=addr\t\t\ta bundle address to use as checkpoint\n"
		"--ia32\t\t\t\t\tmonitor IA-32 execution only\n"
		"--ia64\t\t\t\t\tmonitor IA-64 execution only\n"
		"--insn-sets=set1,set2,...\t\tset per event instruction set (setX=[ia32|ia64|both])\n"
		"--inverse-irange\t\t\tinverse instruction range restriction\n"
	);
}

static void
setup_ear(pfmlib_param_t *evt)
{
	int i, done_iear = 0, done_dear = 0;
	int ev;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	for (i=0; i < evt->pfp_event_count; i++) {

		ev = evt->pfp_events[i].event;

		if (pfm_ita2_is_ear(ev) == 0) continue;

		if (pfm_ita2_is_dear(ev)) {
			if (done_dear) {
				fatal_error("cannot specify more than one D-EAR event at the same time\n");
			}

			pfm_ita2_get_ear_mode(ev, &pfmon_ita2_opt.dear_mode);

			param->pfp_ita2_dear.ear_used   = 1;
			param->pfp_ita2_dear.ear_mode   = pfmon_ita2_opt.dear_mode;
			param->pfp_ita2_dear.ear_plm    = evt->pfp_events[i].plm; /* use plm from event */
			param->pfp_ita2_dear.ear_ism    = param->pfp_ita2_counters[i].ism;
			pfm_ita2_get_event_umask(ev, &param->pfp_ita2_dear.ear_umask);
			
			options.smpl_regs |=  pfm_ita2_is_dear_alat(ev) ? DEAR_ALAT_REGS_MASK : DEAR_REGS_MASK;

			done_dear = 1;
		}

		if (pfm_ita2_is_iear(ev)) {
			if (done_iear) {
				fatal_error("cannot specify more than one D-EAR event at the same time\n");
			}
			pfm_ita2_get_ear_mode(ev, &pfmon_ita2_opt.iear_mode);

			param->pfp_ita2_iear.ear_used   = 1;
			param->pfp_ita2_iear.ear_mode   = pfmon_ita2_opt.iear_mode;
			param->pfp_ita2_iear.ear_plm    = evt->pfp_events[i].plm; /* use plm from event */
			param->pfp_ita2_iear.ear_ism    = param->pfp_ita2_counters[i].ism;
			pfm_ita2_get_event_umask(ev, &param->pfp_ita2_iear.ear_umask);

			options.smpl_regs |=  IEAR_REGS_MASK;

			done_iear = 1;
		}
	}	
}

static int
setup_btb(pfmlib_param_t *evt)
{
	int i, ev;
	int found_alat = 0, found_btb = 0, found_dear = 0;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	/*
	 * For pfmon, we do not activate the BTB registers unless a BRANCH_EVENT
	 * is specified in the event list. The libpfm library does not have this restriction.
	 *
	 * XXX: must make sure BRANCH_EVENT shows up only once
	 */
	for (i=0; i < evt->pfp_event_count; i++) {
		ev = evt->pfp_events[i].event;
		if (pfm_ita2_is_btb(ev)) found_btb = 1;
		if (pfm_ita2_is_dear_alat(ev)) found_alat = 1;
		if (pfm_ita2_is_dear_tlb(ev)) found_dear = 1;
	}
       if (!found_btb &&
           !pfmon_ita2_opt.opt_btb_ds &&
           !pfmon_ita2_opt.opt_btb_tm &&
           !pfmon_ita2_opt.opt_btb_ptm &&
           !pfmon_ita2_opt.opt_btb_ppm &&
           !pfmon_ita2_opt.opt_btb_brt)
         return 0;

	/*
	 * PMC12 must be zero when D-EAR ALAT is configured
	 * The library does the check but here we can print a more detailed error message
	 */
	if (found_btb && found_alat) fatal_error("cannot use BTB and D-EAR ALAT at the same time\n");
	if (found_btb && found_dear) fatal_error("cannot use BTB and D-EAR TLB at the same time\n");

	/*
	 * set the use bit, such that the library will program PMC12
	 */
	param->pfp_ita2_btb.btb_used = 1;

	/* by default, the registers are setup to 
	 * record every possible branch.
	 *
	 * The data selector is set to capture branch target rather than prediction.
	 *
	 * The record nothing is not available because it simply means
	 * don't use a BTB event.
	 *
	 * So the only thing the user can do is:
	 * 	- narrow down the type of branches to record. 
	 * 	  This simplifies the number of cases quite substantially.
	 * 	- change the data selector
	 */
	param->pfp_ita2_btb.btb_ds  = 0;
	param->pfp_ita2_btb.btb_tm  = 0x3;
	param->pfp_ita2_btb.btb_ptm = 0x3;
	param->pfp_ita2_btb.btb_ppm = 0x3;
	param->pfp_ita2_btb.btb_brt = 0x0;
	param->pfp_ita2_btb.btb_plm = evt->pfp_events[i].plm; /* use the plm from the BTB event */

	if (pfmon_ita2_opt.opt_btb_ds)  param->pfp_ita2_btb.btb_ds  = 1;
	if (pfmon_ita2_opt.opt_btb_tm)  param->pfp_ita2_btb.btb_tm  = pfmon_ita2_opt.opt_btb_tm & 0x3;
	if (pfmon_ita2_opt.opt_btb_ptm) param->pfp_ita2_btb.btb_ptm = pfmon_ita2_opt.opt_btb_ptm & 0x3;
	if (pfmon_ita2_opt.opt_btb_ppm) param->pfp_ita2_btb.btb_ppm = pfmon_ita2_opt.opt_btb_ppm & 0x3;
	if (pfmon_ita2_opt.opt_btb_brt) param->pfp_ita2_btb.btb_brt = pfmon_ita2_opt.opt_btb_brt & 0x3;

	vbprintf("btb options: ds=%d tm=%d ptm=%d ppm=%d brt=%d\n",
		param->pfp_ita2_btb.btb_ds,
		param->pfp_ita2_btb.btb_tm,
		param->pfp_ita2_btb.btb_ptm,
		param->pfp_ita2_btb.btb_ppm,
		param->pfp_ita2_btb.btb_brt);

	options.smpl_regs |=  BTB_REGS_MASK;

	DPRINT(("smpl_regs=0x%lx\n", options.smpl_regs));

	return 0;
}

/*
 * Itanium2-specific options
 * For options with indexes, they must be > 256
 */
static struct option cmd_ita2_options[]={
	{ "event-thresholds", 1, 0, 400 },
	{ "opc-match8", 1, 0, 401},
	{ "opc-match9", 1, 0, 402},
	{ "btb-all-mispredicted", 0, 0, 403},
	{ "checkpoint-func", 1, 0, 404},
	{ "irange", 1, 0, 405},
	{ "drange", 1, 0, 406},
	{ "insn-sets", 1, 0, 407},
	{ "btb-ds-pred", 0, &pfmon_ita2_opt.opt_btb_ds, 1},
	{ "btb-tm-tk", 0, &pfmon_ita2_opt.opt_btb_tm, 0x2},
	{ "btb-tm-ntk", 0, &pfmon_ita2_opt.opt_btb_tm, 0x1},
	{ "btb-ptm-correct", 0, &pfmon_ita2_opt.opt_btb_ptm, 0x2},
	{ "btb-ptm-incorrect", 0, &pfmon_ita2_opt.opt_btb_ptm, 0x1},
	{ "btb-ppm-correct", 0, &pfmon_ita2_opt.opt_btb_ppm, 0x2},
	{ "btb-ppm-incorrect", 0, &pfmon_ita2_opt.opt_btb_ppm, 0x1},
	{ "btb-brt-iprel", 0, &pfmon_ita2_opt.opt_btb_brt, 0x1},
	{ "btb-brt-ret", 0, &pfmon_ita2_opt.opt_btb_brt, 0x2},
	{ "btb-brt-ind", 0, &pfmon_ita2_opt.opt_btb_brt, 0x3},
	{ "ia32", 0, &pfmon_ita2_opt.opt_ia32, 0x1},
	{ "ia64", 0, &pfmon_ita2_opt.opt_ia64, 0x1},
	{ "inverse-irange", 0, &pfmon_ita2_opt.opt_inv_rr, 0x1},
	{ 0, 0, 0, 0}
};

static int
pfmon_ita2_initialize(pfmlib_param_t *evt)
{
	int r;

	r = pfmon_register_options(cmd_ita2_options, sizeof(cmd_ita2_options));
	if (r == -1) return -1;

	memset(&pfmlib_ita2_param, 0, sizeof(pfmlib_ita2_param));

	pfmlib_ita2_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;

	/* connect model specific library parameters */
	evt->pfp_model = &pfmlib_ita2_param;

	/* connect pfmon model specific options */
	options.model_options = &pfmon_ita2_opt;

	return 0;
}

static void
setup_insn(pfmlib_param_t *evt)
{
	static const struct {
		char *name;
		pfmlib_ita2_ism_t val;
	} insn_sets[]={
		{ "", 0  }, /* empty element: indicate use default value set by pfmon */
		{ "ia32", PFMLIB_ITA2_ISM_IA32 },
		{ "ia64", PFMLIB_ITA2_ISM_IA64 },
		{ "both", PFMLIB_ITA2_ISM_BOTH },
		{ NULL, 0}
	};
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	char *p, *arg;
	pfmlib_ita2_ism_t dfl_ism;
	int i, cnt=0;

	/* 
	 * set default instruction set 
	 */
	if (pfmon_ita2_opt.opt_ia32  && pfmon_ita2_opt.opt_ia64)
		dfl_ism = PFMLIB_ITA2_ISM_BOTH;
	else if (pfmon_ita2_opt.opt_ia64)
		dfl_ism = PFMLIB_ITA2_ISM_IA64;
	else if (pfmon_ita2_opt.opt_ia32)
		dfl_ism = PFMLIB_ITA2_ISM_IA32;
	else
		dfl_ism = PFMLIB_ITA2_ISM_BOTH;

	/*
	 * propagate default instruction set to all events
	 */
	for(i=0; i < evt->pfp_event_count; i++) param->pfp_ita2_counters[i].ism = dfl_ism;

	/*
	 * apply correction for per-event instruction set
	 */
	for (arg = pfmon_ita2_opt.insn_str; arg; arg = p) {
		if (cnt == evt->pfp_event_count) goto too_many;

		p = strchr(arg,',');
			
		if (p) *p = '\0';

		if (*arg) {
			for (i=0 ; insn_sets[i].name; i++) {
				if (!strcmp(insn_sets[i].name, arg)) goto found;
			}
			goto error;
found:
			param->pfp_ita2_counters[cnt++].ism = insn_sets[i].val;
		}
		/* place the comma back so that we preserve the argument list */
		if (p) *p++ = ',';
	}
	return;
error:
	fatal_error("unknown per-event instruction set %s (choices are ia32, ia64, or both)\n", arg);
	/* no return */
too_many:
	fatal_error("too many per-event instruction sets specified, max=%d\n", evt->pfp_event_count);
}

static int
pfmon_ita2_parse_options(int code, char *optarg, pfmlib_param_t *evt)
{
	switch(code) {
		case  400:
			if (pfmon_ita2_opt.thres_arg) fatal_error("thresholds already defined\n");
			pfmon_ita2_opt.thres_arg = optarg;
			break;
		case  401:
			if (pfmon_ita2_opt.opcm8_str) fatal_error("opcode matcher pmc8 is specified twice\n");
			pfmon_ita2_opt.opcm8_str = optarg;
			break;
		case  402:
			if (pfmon_ita2_opt.opcm9_str) fatal_error("opcode matcher pmc9 is specified twice\n");
			pfmon_ita2_opt.opcm9_str = optarg;
			break;
		case  403: /* all mispredicted branches */
			/* shortcut to the following options
			 * must not be used with other btb options
			 */
			pfmon_ita2_opt.opt_btb_ds    = 0;
			pfmon_ita2_opt.opt_btb_tm    = 0x3;
			pfmon_ita2_opt.opt_btb_ptm   = 0x1;
			pfmon_ita2_opt.opt_btb_ppm   = 0x1;
			pfmon_ita2_opt.opt_btb_brt   = 0x3;
			break;
		case  404:
			if (pfmon_ita2_opt.irange_str) {
				fatal_error("cannot use checkpoints and instruction range at the same time\n");
			}
			if (pfmon_ita2_opt.chkp_func_str) {
				fatal_error("checkpoint already  defined for %s\n", pfmon_ita2_opt.chkp_func_str);
			}
			pfmon_ita2_opt.chkp_func_str = optarg;
			break;

		case  405:
			if (pfmon_ita2_opt.chkp_func_str) {
				fatal_error("cannot use instruction range and checkpoints at the same time\n");
			}
			if (pfmon_ita2_opt.irange_str) {
				fatal_error("cannot specify more than one instruction range\n");
			}
			pfmon_ita2_opt.irange_str = optarg;
			break;
		case  406:
			if (pfmon_ita2_opt.drange_str) {
				fatal_error("cannot specify more than one data range\n");
			}
			pfmon_ita2_opt.drange_str = optarg;
			break;
		case  407:
			if (pfmon_ita2_opt.insn_str) fatal_error("per-event instruction sets already defined");
			pfmon_ita2_opt.insn_str = optarg;
			break;
		default:
			return -1;
	}
	return 0;
}

static void
setup_opcm(pfmlib_param_t *evt)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	char *endptr = NULL;

	if (pfmon_ita2_opt.opcm8_str) {
		if (isdigit(pfmon_ita2_opt.opcm8_str[0])) {
			param->pfp_ita2_pmc8.pmc_val = strtoul(pfmon_ita2_opt.opcm8_str, &endptr, 0);
			if (*endptr != '\0') 
				fatal_error("invalid value for opcode match pmc8: %s\n", pfmon_ita2_opt.opcm8_str);
		} else if (find_opcode_matcher(pfmon_ita2_opt.opcm8_str, &param->pfp_ita2_pmc8.pmc_val) == 0) 
				fatal_error("invalid opcode matcher value: %s\n", pfmon_ita2_opt.opcm8_str);

		param->pfp_ita2_pmc8.opcm_used = 1;

		vbprintf("[pmc8=0x%lx]\n", param->pfp_ita2_pmc8.pmc_val); 
	}

	if (pfmon_ita2_opt.opcm9_str) {
		if (isdigit(pfmon_ita2_opt.opcm9_str[0])) {
			param->pfp_ita2_pmc9.pmc_val = strtoul(pfmon_ita2_opt.opcm9_str, &endptr, 0);
			if (*endptr != '\0') 
				fatal_error("invalid value for opcode match pmc9: %s\n", pfmon_ita2_opt.opcm8_str);
		} else if (find_opcode_matcher(pfmon_ita2_opt.opcm9_str, &param->pfp_ita2_pmc9.pmc_val) == 0) 
				fatal_error("invalid opcode matcher value: %s\n", pfmon_ita2_opt.opcm9_str);

		param->pfp_ita2_pmc9.opcm_used = 1;

		vbprintf("opcode matcher pmc9=0x%lx\n", param->pfp_ita2_pmc9.pmc_val); 
	}
}

static void
setup_rr(pfmlib_param_t *evt)
{
	unsigned long start, end;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);

	/*
	 * we cannot have function checkpoint and irange
	 */
	if (pfmon_ita2_opt.chkp_func_str) {
		if (options.priv_lvl_str)
			fatal_error("cannot use both a checkpoint function and per-event privilege level masks\n");

		gen_code_range(pfmon_ita2_opt.chkp_func_str, &start, &end);

		/* just one bundle for this one */
		end = start + 0x10;

		vbprintf("checkpoint function at 0x%lx\n", start);

	} else if (pfmon_ita2_opt.irange_str) {

		if (options.priv_lvl_str)
			fatal_error("cannot use both a code range function and per-event privilege level masks\n");

		gen_code_range( pfmon_ita2_opt.irange_str, &start, &end); 

		if (start & 0xf) fatal_error("code range does not start on bundle boundary : 0x%lx\n", start);
		if (end & 0xf) fatal_error("code range does not end on bundle boundary : 0x%lx\n", end);

		vbprintf("irange is [0x%lx-0x%lx)=%ld bytes\n", start, end, end-start);
	}

	
	/*
	 * now finalize irange/chkp programming of the range
	 */
	if (pfmon_ita2_opt.irange_str || pfmon_ita2_opt.chkp_func_str) { 

		param->pfp_ita2_irange.rr_used   = 1;
		param->pfp_ita2_irange.rr_flags |= pfmon_ita2_opt.opt_inv_rr ? PFMLIB_ITA2_RR_INV : 0;

		/*
		 * due to a bug in the PMU, fine mode does not work for small ranges (less than
		 * 2 bundles), so we force non-fine mode to work around the problem
		 */
		if (pfmon_ita2_opt.chkp_func_str)
			param->pfp_ita2_irange.rr_flags |= PFMLIB_ITA2_RR_NO_FINE_MODE;

		param->pfp_ita2_irange.rr_limits[0].rr_start = start;
		param->pfp_ita2_irange.rr_limits[0].rr_end   = end;
		param->pfp_ita2_irange.rr_limits[0].rr_plm   = evt->pfp_dfl_plm; /* use default */
	}
	
	if (pfmon_ita2_opt.drange_str) {
		if (options.priv_lvl_str)
			fatal_error("cannot use both a data range and  per-event privilege level masks\n");

		gen_data_range( pfmon_ita2_opt.drange_str, &start, &end);

		vbprintf("drange is [0x%lx-0x%lx)=%ld bytes\n", start, end, end-start);
		
		param->pfp_ita2_drange.rr_used = 1;

		param->pfp_ita2_drange.rr_limits[0].rr_start = start;
		param->pfp_ita2_drange.rr_limits[0].rr_end   = end;
		param->pfp_ita2_drange.rr_limits[0].rr_plm   = evt->pfp_dfl_plm; /* use default */
	}

}

/*
 * It is not possible to measure more than one of the
 * L2_OZQ_CANCELS0, L2_OZQ_CANCELS1, L2_OZQ_CANCELS2 at the
 * same time.
 */
static char *cancel_events[]=
{
	"L2_OZQ_CANCELS0_ANY",
	"L2_OZQ_CANCELS1_REL",
	"L2_OZQ_CANCELS2_ACQ"
};
#define NCANCEL_EVENTS	sizeof(cancel_events)/sizeof(char *)

static void
check_cancel_events(pfmlib_param_t *evt)
{
	int i, j, tmp, code;
	int cancel_codes[NCANCEL_EVENTS];
	int idx = -1;

	for(i=0; i < NCANCEL_EVENTS; i++) {
		pfm_find_event_byname(cancel_events[i], &tmp);
		pfm_get_event_code(tmp, &code);
		cancel_codes[i] = code;
	}
	for(i=0; i < evt->pfp_event_count; i++) {
		for (j=0; j < NCANCEL_EVENTS; j++) {
			pfm_get_event_code(evt->pfp_events[i].event, &code);
			if (code == cancel_codes[j]) {
				if (idx != -1) {
					char *name, *name2;
					pfm_get_event_name(idx, &name);
					pfm_get_event_name(evt->pfp_events[i].event, &name2);
					fatal_error("%s and %s cannot be measured at the same time\n", name, name2);
				}
				idx = evt->pfp_events[i].event;
			}
		}
	}
}

static void
check_cross_groups_and_set_umask(pfmlib_param_t *evt)
{
	unsigned long ref_umask, umask;
	int g, g2, s, s2;
	unsigned int cnt = evt->pfp_event_count;
	pfmlib_event_t *e= evt->pfp_events;
	char *name1, *name2;
	int i, j;

	for (i=0; i < cnt; i++) {

		pfm_ita2_get_event_group(e[i].event, &g);
		pfm_ita2_get_event_set(e[i].event, &s);

		if (g == PFMLIB_ITA2_EVT_NO_GRP) continue;

		pfm_ita2_get_event_umask(e[i].event, &ref_umask);

		for (j=i+1; j < cnt; j++) {
			pfm_ita2_get_event_group(e[j].event, &g2);
			if (g2 != g) continue;

			pfm_ita2_get_event_set(e[j].event, &s2);
			if (s2 != s) goto error;

			/* only care about L2 cache group */
			if (g != PFMLIB_ITA2_EVT_L2_CACHE_GRP || (s == 1 || s == 2)) continue;

			pfm_ita2_get_event_umask(e[j].event, &umask);
			/*
			 * there is no assignment valid if more than one event of 
			 * the set has a umask
			 */
			if (umask && ref_umask != umask) goto error;
		}
	}
	return;
error:
	pfm_get_event_name(e[i].event, &name1);
	pfm_get_event_name(e[j].event, &name2);
	fatal_error("event %s and %s cannot be measured at the same time\n", name1, name2);
}

static void
check_ita2_event_combinations(pfmlib_param_t *evt)
{
	char *name;
	int i, use_opcm, inst_retired_idx;
	int ev, code, inst_retired_code;
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);


	/*
	 * here we repeat some of the tests done by the library
	 * to provide more detailed feedback (printf()) to the user.
	 *
	 * XXX: not all tests are duplicated, so we will not get detailed
	 * error reporting for all possible cases.
	 */
	check_counter_conflict(evt, 0xf0UL);
	check_cancel_events(evt);
	check_cross_groups_and_set_umask(evt);

	use_opcm = param->pfp_ita2_pmc8.opcm_used || param->pfp_ita2_pmc9.opcm_used; 

	pfm_find_event_byname("IA64_INST_RETIRED", &inst_retired_idx);
	pfm_get_event_code(inst_retired_idx, &inst_retired_code);

	for (i=0; i < evt->pfp_event_count; i++) {

		ev = evt->pfp_events[i].event;

		pfm_get_event_name(ev, &name);
		pfm_get_event_code(ev, &code);

		if (use_opcm && pfm_ita2_support_opcm(ev) == 0)
			fatal_error("event %s does not support opcode matching\n", name);

		if (param->pfp_ita2_pmc9.opcm_used && code != inst_retired_code) 
			fatal_error("pmc9 can only be used to qualify the IA64_INST_RETIRED events\n");

		if (param->pfp_ita2_irange.rr_used && pfm_ita2_support_iarr(ev) == 0)
			fatal_error("event %s does not support instruction address range restrictions\n", name);

		if (param->pfp_ita2_drange.rr_used && pfm_ita2_support_darr(ev) == 0)
			fatal_error("event %s does not support data address range restrictions\n", name);
	}
}


static int
pfmon_ita2_post_options(pfmlib_param_t *evt)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);


	if (options.trigger_saddr_str || options.trigger_eaddr_str) {
		if (pfmon_ita2_opt.irange_str)
			fatal_error("cannot use a trigger address with instruction range restrictions\n");
		if (pfmon_ita2_opt.drange_str)
			fatal_error("cannot use a trigger address with data range restrictions\n");
		if (pfmon_ita2_opt.chkp_func_str)
			fatal_error("cannot use a trigger address with function checkpoint\n");
	}
	/*
	 * XXX: link pfmon option to library CPU-model specific configuration
	 */
	pfmon_ita2_opt.params = param;

	/*
	 * setup the instruction set support
	 *
	 * and reject any invalid feature combination for IA-32 only monitoring
	 *
	 * We do not warn of the fact that IA-32 execution will be ignored
	 * when used with incompatible features unless the user requested IA-32
	 * ONLY monitoring. 
	 */
	if (pfmon_ita2_opt.opt_ia32 == 1 && pfmon_ita2_opt.opt_ia64 == 0) {

		/*
		 * Code & Data range restrictions are ignored for IA-32
		 */
		if (pfmon_ita2_opt.irange_str|| pfmon_ita2_opt.drange_str) 
			fatal_error("you cannot use range restrictions when monitoring IA-32 execution only\n");

		/*
		 * Code range restriction (used by checkpoint) is ignored for IA-32
		 */
		if (pfmon_ita2_opt.chkp_func_str) 
			fatal_error("you cannot use function checkpoint when monitoring IA-32 execution only\n");
	}

	setup_insn(evt);

	setup_rr(evt);

	setup_btb(evt);

	setup_opcm(evt);

	/*
	 * BTB is only valid in IA-64 mode
	 */
	if (param->pfp_ita2_btb.btb_used && pfmon_ita2_opt.opt_ia32) {
		fatal_error("cannot use the BTB when monitoring IA-32 execution\n");
	}

	setup_ear(evt);

	/* 
	 * we systematically initialize thresholds to their minimal value
	 * or requested value
	 */
	gen_thresholds(pfmon_ita2_opt.thres_arg, evt);

	check_ita2_event_combinations(evt);

	return 0;
}

static int
pfmon_ita2_print_header(FILE *fp)
{
	char *name;
	int i, isn;
	static const char *insn_str[]={
		"ia32/ia64",
		"ia32", 
		"ia64"
	};


	fprintf(fp, "#\n#\n# instruction sets:\n");

	for(i=0; i < PMU_MAX_PMDS; i++) {
		if (options.rev_pc[i] == -1) continue;

		pfm_get_event_name(options.events[options.rev_pc[i]].event, &name);

		isn = pfmon_ita2_opt.params->pfp_ita2_counters[options.rev_pc[i]].ism;
		fprintf(fp, "#\tPMD%d: %s, %s\n", 
			i,
			name,
			insn_str[isn]);
	} 
	fprintf(fp, "#\n");

	return 0;
}

static void
pfmon_ita2_detailed_event_name(int evt)
{
	unsigned long umask;
	unsigned long maxincr;
	char *grp_str;
	int grp, set;

	pfm_ita2_get_event_umask(evt, &umask);
	pfm_ita2_get_event_maxincr(evt, &maxincr);
	pfm_ita2_get_event_group(evt, &grp);

	printf("umask=0x%02lx incr=%ld iarr=%c darr=%c opcm=%c ", 
		umask, 
		maxincr,
		pfm_ita2_support_iarr(evt) ? 'Y' : 'N',
		pfm_ita2_support_darr(evt) ? 'Y' : 'N',
		pfm_ita2_support_opcm(evt) ? 'Y' : 'N');

	if (grp != PFMLIB_ITA2_EVT_NO_GRP) {
		pfm_ita2_get_event_set(evt, &set);
		grp_str = grp == PFMLIB_ITA2_EVT_L1_CACHE_GRP ? "l1_cache" : "l2_cache";

		printf("grp=%s set=%d", grp_str, set);
	}
}

pfmon_support_t pfmon_itanium2={
	"Itanium2",
	PFMLIB_ITANIUM2_PMU,
	pfmon_ita2_initialize,		/* initialize */
	pfmon_ita2_usage,		/* usage */
	pfmon_ita2_parse_options,	/* parse */
	pfmon_ita2_post_options,	/* post */
	NULL,				/* overflow */
	pfmon_ita2_install_counters,	/* install counters */
	pfmon_ita2_print_header,	/* print header */
	pfmon_ita2_detailed_event_name	/* detailed event name */
};

