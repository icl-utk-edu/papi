/*
 * pfmon_itanium.c - Itanium PMU support for pfmon
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
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium.h>

#include "pfmon.h"
#include "pfmon_itanium.h"

#define PFMON_SUPPORT_NAME	"Itanium"

#define M_PMD(x)		(1UL<<(x))
#define DEAR_REGS_MASK		(M_PMD(2)|M_PMD(3)|M_PMD(17))
#define IEAR_REGS_MASK		(M_PMD(0)|M_PMD(1))
#define BTB_REGS_MASK		(M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))



static pfmon_ita_options_t pfmon_ita_opt;	/* keep track of global program options */
static pfmlib_ita_param_t pfmlib_ita_param;

static void
gen_thresholds(char *arg, pfmlib_param_t *evt)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	char *p;
	int cnt=0;
	unsigned long thres, maxincr;

	/*
	 * the default value for the threshold is 0: this means at least once 
	 * per cycle.
	 */
	if (arg == NULL) {
		int i;
		for (i=0; i < evt->pfp_event_count; i++) param->pfp_ita_counters[i].thres = 0;
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
		pfm_ita_get_event_maxincr(evt->pfp_events[cnt].event, &maxincr);
		if (thres > (maxincr-1)) goto too_big;

		param->pfp_ita_counters[cnt++].thres = thres;

		arg = p;
	}
	return;
too_big:
	fatal_error("event %d: threshold must be in [0-%d)\n", cnt, maxincr);
too_many:
	fatal_error("too many thresholds specified\n");
}

static void
install_irange(int pid, pfmlib_param_t *evt)
{
	int  r;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	r = perfmonctl(pid, PFM_WRITE_IBRS, param->pfp_ita_irange.rr_br, param->pfp_ita_irange.rr_nbr_used);
	if (r == -1) fatal_error("cannot install code range restriction: %s\n", strerror(errno));
}

static void
install_drange(int pid, pfmlib_param_t *evt)
{
	int r;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	r = perfmonctl(pid, PFM_WRITE_DBRS, param->pfp_ita_drange.rr_br, param->pfp_ita_drange.rr_nbr_used);
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
		if (pfm_ita_is_btb(evt->pfp_events[i].event)) {
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
	pfmlib_event_t *e = evt->pfp_events;
	int i;

	/*
	 * search all events for an EAR event
	 */
	for(i=0; i < evt->pfp_event_count; i++) {
		if (pfm_ita_is_ear(e[i].event)) {
			/*
			 * pfmon uses the EAR for sampling, so we must make sure we get the overflow
			 * notification on DATA_EAR_* or INSTRUCTION_EAR_*. This is not done by the 
			 * library (anymore).
			 */
			evt->pfp_pc[i].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
		}
	}
}

/*
 * Executed in the context of the child, this is the last chance to modify programming
 * before the PMC and PMD register are written.
 */
static int
pfmon_ita_install_counters(int pid, pfmlib_param_t *evt, pfarg_reg_t *pd)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	if (param->pfp_ita_irange.rr_used) install_irange(pid, evt);

	if (param->pfp_ita_drange.rr_used) install_drange(pid, evt);

	if (param->pfp_ita_btb.btb_used) prepare_pmd16(pid, evt);

	fixup_ears(pid, evt);

	return 0;
}

static void
pfmon_ita_usage(void)
{
	printf(
		"--event-thresholds=thr1,thr2,...\tset event thresholds (no space)\n"
		"--opc-match8=val\t\t\tset opcode match for PMC8\n"
		"--opc-match9=val\t\t\tset opcode match for PMC9\n"
		"--btb-no-tar\t\t\t\tdon't capture TAR predictions\n"
		"--btb-no-bac\t\t\t\tdon't capture BAC predictions\n"
		"--btb-no-tac\t\t\t\tdon't capture TAC predictions\n"
		"--btb-tm-tk\t\t\t\tcapture taken IA-64 branches only\n"
		"--btb-tm-ntk\t\t\t\tcapture not taken IA-64 branches only\n"
		"--btb-ptm-correct\t\t\tcapture branch if target predicted correctly\n"
		"--btb-ptm-incorrect\t\t\tcapture branch if target is mispredicted\n"
		"--btb-ppm-correct\t\t\tcapture branch if path is predicted correctly\n"
		"--btb-ppm-incorrect\t\t\tcapture branch if path is mispredicted\n"
		"--btb-all-mispredicted\t\t\tcapture all mispredicted branches\n"
		"--irange=start-end\t\t\tspecify an instruction address range constraint\n"
		"--drange=start-end\t\t\tspecify a data address range constraint\n"
		"--checkpoint-func=addr\t\t\ta bundle address to use as checkpoint\n"
		"--ia32\t\t\t\t\tmonitor IA-32 execution only\n"
		"--ia64\t\t\t\t\tmonitor IA-64 execution only\n"
		"--insn-sets=set1,set2,...\t\tset per event instruction set (setX=[ia32|ia64|both])\n"
	);
}

static void
setup_ear(pfmlib_param_t *evt)
{
	int i, done_iear = 0, done_dear = 0;
	int ev;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	for (i=0; i < evt->pfp_event_count; i++) {
		ev = evt->pfp_events[i].event;
		if (pfm_ita_is_ear(ev) == 0) continue;

		if (pfm_ita_is_dear(ev)) {
			if (done_dear) {
				fatal_error("cannot specify more than one D-EAR event at the same time\n");
			}
			pfmon_ita_opt.opt_use_dear_tlb = pfm_ita_is_dear_tlb(ev) ? 1 : 0;

			param->pfp_ita_dear.ear_used   = 1;
			param->pfp_ita_dear.ear_is_tlb = pfmon_ita_opt.opt_use_dear_tlb;
			param->pfp_ita_dear.ear_plm    = evt->pfp_events[i].plm; /* use plm from event */
			param->pfp_ita_dear.ear_ism    = param->pfp_ita_counters[i].ism;
			pfm_ita_get_event_umask(ev, &param->pfp_ita_dear.ear_umask);

			options.smpl_regs |=  DEAR_REGS_MASK;

			done_dear = 1;
		}

		if (pfm_ita_is_iear(ev)) {
			if (done_iear) {
				fatal_error("cannot specify more than one D-EAR event at the same time\n");
			}
			pfmon_ita_opt.opt_use_iear_tlb = pfm_ita_is_iear_tlb(ev) ? 1 : 0;

			param->pfp_ita_iear.ear_used   = 1;
			param->pfp_ita_iear.ear_is_tlb = pfmon_ita_opt.opt_use_iear_tlb;
			param->pfp_ita_iear.ear_plm    = evt->pfp_events[i].plm; /* use plm from event */
			param->pfp_ita_iear.ear_ism    = param->pfp_ita_counters[i].ism;

			pfm_ita_get_event_umask(ev, &param->pfp_ita_iear.ear_umask);

			options.smpl_regs |=  IEAR_REGS_MASK;

			done_iear = 1;
		}
	}	
}

static int
setup_btb(pfmlib_param_t *evt)
{
	int i;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	/*
	 * For pfmon, we do not activate the BTB registers unless a BRANCH_EVENT
	 * is specified in the event list. The libpfm library does not have this restriction.
	 *
	 * XXX: must make sure BRANCH_EVENT shows up only once
	 */
	for (i=0; i < evt->pfp_event_count; i++) {
		if (pfm_ita_is_btb(evt->pfp_events[i].event)) goto found;
	}
	/*
	 * if the user specified an BTB option (but not the event) 
	 * then we program the BTB as a free running config.
	 *
	 * XXX: cannot record ALL branches
	 */
	if (  pfmon_ita_opt.opt_btb_notar
	   || pfmon_ita_opt.opt_btb_notac
	   || pfmon_ita_opt.opt_btb_nobac
	   || pfmon_ita_opt.opt_btb_tm
	   || pfmon_ita_opt.opt_btb_ptm
	   || pfmon_ita_opt.opt_btb_ppm) goto found;
	return 0;

found:
	/*
	 * set the use bit, such that the library will program PMC12
	 */
	param->pfp_ita_btb.btb_used = 1;

	/* by default, the registers are setup to 
	 * record every possible branch.
	 * The record nothing is not available because it simply means
	 * don't use a BTB event.
	 * So the only thing the user can do is narrow down the type of
	 * branches to record. This simplifies the number of cases quite
	 * substantially.
	 */
	param->pfp_ita_btb.btb_tar = 1;
	param->pfp_ita_btb.btb_tac = 1;
	param->pfp_ita_btb.btb_bac = 1;
	param->pfp_ita_btb.btb_tm  = 0x3;
	param->pfp_ita_btb.btb_ptm = 0x3;
	param->pfp_ita_btb.btb_ppm = 0x3;
	param->pfp_ita_btb.btb_plm = evt->pfp_events[i].plm; /* use the plm from the BTB event */

	if (pfmon_ita_opt.opt_btb_notar) param->pfp_ita_btb.btb_tar = 0;
	if (pfmon_ita_opt.opt_btb_notac) param->pfp_ita_btb.btb_tac = 0;
	if (pfmon_ita_opt.opt_btb_nobac) param->pfp_ita_btb.btb_bac = 0;
	if (pfmon_ita_opt.opt_btb_tm)    param->pfp_ita_btb.btb_tm  = pfmon_ita_opt.opt_btb_tm & 0x3;
	if (pfmon_ita_opt.opt_btb_ptm)   param->pfp_ita_btb.btb_ptm = pfmon_ita_opt.opt_btb_ptm & 0x3;
	if (pfmon_ita_opt.opt_btb_ppm)   param->pfp_ita_btb.btb_ppm = pfmon_ita_opt.opt_btb_ppm & 0x3;

	vbprintf("Branch Trace Buffer Options:\n\tplm=%d tar=%c tac=%c bac=%c tm=%d ptm=%d ppm=%d\n",
		param->pfp_ita_btb.btb_plm,
		param->pfp_ita_btb.btb_tar ? 'Y' : 'N',
		param->pfp_ita_btb.btb_tac ? 'Y' : 'N',
		param->pfp_ita_btb.btb_bac ? 'Y' : 'N',
		param->pfp_ita_btb.btb_tm,
		param->pfp_ita_btb.btb_ptm,
		param->pfp_ita_btb.btb_ppm);

	options.smpl_regs |=  BTB_REGS_MASK;

	DPRINT(("pfmon_itanium: smpl_regs=0x%lx\n", options.smpl_regs));

	return 0;
}

/*
 * Itanium-specific options
 * For options with indexes, they must be > 256
 */
static struct option cmd_ita_options[]={
	{ "event-thresholds", 1, 0, 400 },
	{ "opc-match8", 1, 0, 401},
	{ "opc-match9", 1, 0, 402},
	{ "btb-all-mispredicted", 0, 0, 403},
	{ "checkpoint-func", 1, 0, 404},
	{ "irange", 1, 0, 405},
	{ "drange", 1, 0, 406},
	{ "insn-sets", 1, 0, 407},

	{ "btb-no-tar", 0, &pfmon_ita_opt.opt_btb_notar, 1},
	{ "btb-no-bac", 0, &pfmon_ita_opt.opt_btb_nobac, 1},
	{ "btb-no-tac", 0, &pfmon_ita_opt.opt_btb_notac, 1},
	{ "btb-tm-tk", 0, &pfmon_ita_opt.opt_btb_tm, 0x2},
	{ "btb-tm-ntk", 0, &pfmon_ita_opt.opt_btb_tm, 0x1},
	{ "btb-ptm-correct", 0, &pfmon_ita_opt.opt_btb_ptm, 0x2},
	{ "btb-ptm-incorrect", 0, &pfmon_ita_opt.opt_btb_ptm, 0x1},
	{ "btb-ppm-correct", 0, &pfmon_ita_opt.opt_btb_ppm, 0x2},
	{ "btb-ppm-incorrect", 0, &pfmon_ita_opt.opt_btb_ppm, 0x1},
	{ "ia32", 0, &pfmon_ita_opt.opt_ia32, 0x1},
	{ "ia64", 0, &pfmon_ita_opt.opt_ia64, 0x1},
	{ 0, 0, 0, 0}
};

static int
pfmon_ita_initialize(pfmlib_param_t *evt)
{
	int r;

	r = pfmon_register_options(cmd_ita_options, sizeof(cmd_ita_options));
	if (r == -1) return -1;

	memset(&pfmlib_ita_param, 0, sizeof(pfmlib_ita_param));

	pfmlib_ita_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;

	/* connect model specific library parameters */
	evt->pfp_model = &pfmlib_ita_param;

	/* connect pfmon model specific options */
	options.model_options = &pfmon_ita_opt;

	return 0;
}

static int
pfmon_ita_parse_options(int code, char *optarg, pfmlib_param_t *evt)
{
	switch(code) {
		case  400:
			if (pfmon_ita_opt.thres_arg) fatal_error("thresholds already defined\n");
			pfmon_ita_opt.thres_arg = optarg;
			break;
		case  401:
			if (pfmon_ita_opt.opcm8_str) fatal_error("opcode matcher pmc8 is specified twice\n");
			pfmon_ita_opt.opcm8_str = optarg;
			break;
		case  402:
			if (pfmon_ita_opt.opcm9_str) fatal_error("opcode matcher pmc9 is specified twice\n");
			pfmon_ita_opt.opcm9_str = optarg;
			break;
		case  403:
			/* shortcut to the following options
			 * must not be used with other btb options
			 */
			pfmon_ita_opt.opt_btb_notar = 0;
			pfmon_ita_opt.opt_btb_nobac = 0;
			pfmon_ita_opt.opt_btb_notac = 0;
			pfmon_ita_opt.opt_btb_tm    = 0x3;
			pfmon_ita_opt.opt_btb_ptm   = 0x1;
			pfmon_ita_opt.opt_btb_ppm   = 0x1;
			break;
		case  404:
			if (pfmon_ita_opt.irange_str) {
				fatal_error("cannot use checkpoints and instruction range at the same time\n");
			}
			if (pfmon_ita_opt.chkp_func_str) {
				fatal_error("checkpoint already  defined for %s\n", pfmon_ita_opt.chkp_func_str);
			}
			pfmon_ita_opt.chkp_func_str = optarg;
			break;

		case  405:
			if (pfmon_ita_opt.chkp_func_str) {
				fatal_error("cannot use instruction range and checkpoints at the same time\n");
			}
			if (pfmon_ita_opt.irange_str) {
				fatal_error("cannot specify more than one instruction range\n");
			}
			pfmon_ita_opt.irange_str = optarg;
			break;

		case  406:
			if (pfmon_ita_opt.drange_str) {
				fatal_error("cannot specify more than one data range\n");
			}
			pfmon_ita_opt.drange_str = optarg;
			break;
		case  407:
			if (pfmon_ita_opt.insn_str) fatal_error("per event instruction sets already defined");
			pfmon_ita_opt.insn_str = optarg;
			break;
		default:
			return -1;
	}
	return 0;
}

static void
setup_opcm(pfmlib_param_t *evt)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	char *endptr = NULL;

	if (pfmon_ita_opt.opcm8_str) {
		if (isdigit(pfmon_ita_opt.opcm8_str[0])) {
			param->pfp_ita_pmc8.pmc_val = strtoul(pfmon_ita_opt.opcm8_str, &endptr, 0);
			if (*endptr != '\0') 
				fatal_error("invalid value for opcode match pmc8: %s\n", pfmon_ita_opt.opcm8_str);
		} else if (find_opcode_matcher(pfmon_ita_opt.opcm8_str, &param->pfp_ita_pmc8.pmc_val) == 0) 
				fatal_error("invalid opcode matcher value: %s\n", pfmon_ita_opt.opcm8_str);

		param->pfp_ita_pmc8.opcm_used = 1;

		vbprintf("[pmc8=0x%lx]\n", param->pfp_ita_pmc8.pmc_val); 
	}

	if (pfmon_ita_opt.opcm9_str) {
		if (isdigit(pfmon_ita_opt.opcm9_str[0])) {
			param->pfp_ita_pmc9.pmc_val = strtoul(pfmon_ita_opt.opcm9_str, &endptr, 0);
			if (*endptr != '\0') 
				fatal_error("invalid value for opcode match pmc9: %s\n", pfmon_ita_opt.opcm8_str);
		} else if (find_opcode_matcher(pfmon_ita_opt.opcm9_str, &param->pfp_ita_pmc9.pmc_val) == 0) 
				fatal_error("invalid opcode matcher value: %s\n", pfmon_ita_opt.opcm9_str);

		param->pfp_ita_pmc9.opcm_used = 1;

		vbprintf("opcode matcher pmc9=0x%lx\n", param->pfp_ita_pmc9.pmc_val); 
	}
}


static void
setup_rr(pfmlib_param_t *evt)
{
	unsigned long start, end;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	if (pfmon_ita_opt.chkp_func_str) {
		if (options.priv_lvl_str)
			fatal_error("cannot use both a checkpoint function and per-event privilege level masks\n");

		gen_code_range(pfmon_ita_opt.chkp_func_str, &start, &end);
		
		/* just one bundle for this one */
		end = start + 0x10;

		vbprintf("checkpoint function at 0x%lx\n", start);
	} else if (pfmon_ita_opt.irange_str) {

		if (options.priv_lvl_str)
			fatal_error("cannot use both a code range function and per-event privilege level masks\n");

		gen_code_range(pfmon_ita_opt.irange_str, &start, &end); 

		if (start & 0xf) fatal_error("code range does not start on bundle boundary : 0x%lx\n", start);
		if (end & 0xf) fatal_error("code range does not end on bundle boundary : 0x%lx\n", end);

		vbprintf("irange is [0x%lx-0x%lx)=%ld bytes\n", start, end, end-start);
	}

	/*
	 * now finalize irange/chkp programming of the range
	 */
	if (pfmon_ita_opt.irange_str || pfmon_ita_opt.chkp_func_str) { 

		param->pfp_ita_irange.rr_used = 1;

		param->pfp_ita_irange.rr_limits[0].rr_start = start;
		param->pfp_ita_irange.rr_limits[0].rr_end   = end;
		param->pfp_ita_irange.rr_limits[0].rr_plm   = evt->pfp_dfl_plm; /* use default */
	}

	if (pfmon_ita_opt.drange_str) {
		if (options.priv_lvl_str)
			fatal_error("cannot use both a data range and  per-event privilege level masks\n");

		gen_data_range(pfmon_ita_opt.drange_str, &start, &end);

		vbprintf("drange is [0x%lx-0x%lx)=%ld bytes\n", start, end, end-start);
		
		param->pfp_ita_drange.rr_used = 1;

		param->pfp_ita_drange.rr_limits[0].rr_start = start;
		param->pfp_ita_drange.rr_limits[0].rr_end   = end;
		param->pfp_ita_drange.rr_limits[0].rr_plm   = evt->pfp_dfl_plm; /* use default */
	}

}


/*
 * This function checks the configuration to verify
 * that the user does not try to combine features with
 * events that are incompatible.The library does this also
 * but it's hard to then detail the cause of the error.
 */
static void
check_ita_event_combinations(pfmlib_param_t *evt)
{
	char *name;
	int i, use_opcm, ev;
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	use_opcm = param->pfp_ita_pmc8.opcm_used || param->pfp_ita_pmc9.opcm_used; 
	for (i=0; i < evt->pfp_event_count; i++) {

		ev = evt->pfp_events[i].event;

		pfm_get_event_name(ev, &name);

		if (use_opcm && pfm_ita_support_opcm(ev) == 0)
			fatal_error("event %s does not support opcode matching\n", name);

		if (param->pfp_ita_irange.rr_used && pfm_ita_support_iarr(ev) == 0)
			fatal_error("event %s does not support instruction address range restrictions\n", name);

		if (param->pfp_ita_drange.rr_used && pfm_ita_support_darr(ev) == 0)
			fatal_error("event %s does not support data address range restrictions\n", name);
	}
	/*
	 * we do not call check_counter_conflict() because Itanium does not have events
	 * which can only be measured on one counter, therefore this routine would not
	 * catch anything at all.
	 */
}

static void
setup_insn(pfmlib_param_t *evt)
{
	static const struct {
		char *name;
		pfmlib_ita_ism_t val;
	} insn_sets[]={
		{ "", 0  }, /* empty element: indicate use default value set by pfmon */
		{ "ia32", PFMLIB_ITA_ISM_IA32 },
		{ "ia64", PFMLIB_ITA_ISM_IA64 },
		{ "both", PFMLIB_ITA_ISM_BOTH },
		{ NULL, 0}
	};
	pfmlib_ita_param_t *param = ITA_PARAM(evt);
	char *p, *arg;
	pfmlib_ita_ism_t dfl_ism;
	int i, cnt=0;

	/* 
	 * set default instruction set 
	 */
	if (pfmon_ita_opt.opt_ia32  && pfmon_ita_opt.opt_ia64)
		dfl_ism = PFMLIB_ITA_ISM_BOTH;
	else if (pfmon_ita_opt.opt_ia64)
		dfl_ism = PFMLIB_ITA_ISM_IA64;
	else if (pfmon_ita_opt.opt_ia32)
		dfl_ism = PFMLIB_ITA_ISM_IA32;
	else
		dfl_ism = PFMLIB_ITA_ISM_BOTH;

	/*
	 * propagate default instruction set to all events
	 */
	for(i=0; i < evt->pfp_event_count; i++) param->pfp_ita_counters[i].ism = dfl_ism;

	/*
	 * apply correction for per-event instruction set
	 */
	for (arg = pfmon_ita_opt.insn_str; arg; arg = p) {
		if (cnt == evt->pfp_event_count) goto too_many;

		p = strchr(arg,',');
			
		if (p) *p = '\0';

		if (*arg) {
			for (i=0 ; insn_sets[i].name; i++) {
				if (!strcmp(insn_sets[i].name, arg)) goto found;
			}
			goto error;
found:
			param->pfp_ita_counters[cnt++].ism = insn_sets[i].val;
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
pfmon_ita_post_options(pfmlib_param_t *evt)
{
	pfmlib_ita_param_t *param = ITA_PARAM(evt);

	if (options.trigger_saddr_str || options.trigger_eaddr_str) {
		if (pfmon_ita_opt.irange_str)
			fatal_error("cannot use a trigger address with instruction range restrictions\n");
		if (pfmon_ita_opt.drange_str)
			fatal_error("cannot use a trigger address with data range restrictions\n");
		if (pfmon_ita_opt.chkp_func_str)
			fatal_error("cannot use a trigger address with function checkpoint\n");
	}
	/*
	 * XXX: link pfmon option to library CPU-model specific configuration
	 */
	pfmon_ita_opt.params = param;

	/*
	 * setup the instruction set support
	 *
	 * and reject any invalid combination for IA-32 only monitoring
	 *
	 * We do not warn of the fact that IA-32 execution will be ignored
	 * when used with incompatible features unless the user requested IA-32
	 * ONLY monitoring. 
	 */
	if (pfmon_ita_opt.opt_ia32 == 1 && pfmon_ita_opt.opt_ia64 == 0) {

		/*
		 * Code & Data range restrictions are ignored for IA-32
		 */
		if (pfmon_ita_opt.irange_str|| pfmon_ita_opt.drange_str) 
			fatal_error("you cannot use range restrictions when monitoring IA-32 execution only\n");

		/*
		 * Code range restriction (used by checkpoint) is ignored for IA-32
		 */
		if (pfmon_ita_opt.chkp_func_str) 
			fatal_error("you cannot use function checkpoint when monitoring IA-32 execution only\n");

		/*
		 * opcode matcher are ignored for IA-32
		 */
		if (param->pfp_ita_pmc8.opcm_used || param->pfp_ita_pmc9.opcm_used) 
			fatal_error("you cannot use the opcode matcher(s) when monitoring IA-32 execution only\n");

	}
	setup_insn(evt);

	setup_rr(evt);

	setup_btb(evt);

	setup_opcm(evt);

	/*
	 * BTB is only valid in IA-64 mode
	 */
	if (param->pfp_ita_btb.btb_used && pfmon_ita_opt.opt_ia32) {
			fatal_error("cannot use the BTB when monitoring IA-32 execution\n");
	}

	setup_ear(evt);

	/* 
	 * we systematically initialize thresholds to their minimal value
	 * or requested value
	 */
	gen_thresholds(pfmon_ita_opt.thres_arg, evt);

	check_ita_event_combinations(evt);

	return 0;
}

static int
pfmon_ita_print_header(FILE *fp)
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

		isn = pfmon_ita_opt.params->pfp_ita_counters[options.rev_pc[i]].ism;
		fprintf(fp, "#\tPMD%d: %s, %s\n", 
			i,
			name,
			insn_str[isn]);
	} 
	fprintf(fp, "#\n");

	return 0;
}

static void
pfmon_ita_detailed_event_name(int evt)
{
	unsigned long umask;
	unsigned long maxincr;

	pfm_ita_get_event_umask(evt, &umask);
	pfm_ita_get_event_maxincr(evt, &maxincr);

	printf("umask=0x%02lx incr=%ld iarr=%c darr=%c opcm=%c ", 
			umask, 
			maxincr,
			pfm_ita_support_iarr(evt) ? 'Y' : 'N',
			pfm_ita_support_darr(evt) ? 'Y' : 'N',
			pfm_ita_support_opcm(evt) ? 'Y' : 'N');
}


pfmon_support_t pfmon_itanium={
	"Itanium",
	PFMLIB_ITANIUM_PMU,
	pfmon_ita_initialize,		/* initialize */
	pfmon_ita_usage,		/* usage */
	pfmon_ita_parse_options,	/* parse */
	pfmon_ita_post_options,		/* post */
	NULL,				/* overflow */
	pfmon_ita_install_counters,	/* install counters */
	pfmon_ita_print_header,		/* print header */
	pfmon_ita_detailed_event_name	/* detailed event name */
};
