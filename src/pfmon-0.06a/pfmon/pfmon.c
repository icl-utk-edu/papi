/*
 * pfmon - a sample tool to measure performance on Linux/ia64
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
#include <stdarg.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/wait.h>
#include <signal.h>
#include <setjmp.h>
#include <getopt.h>

#include "perfmon.h"
#include "pfmlib.h"
#include "mysiginfo.h"

#define PFMON_VERSION	"0.06a"
#define PFMON_DFL_EVENT	"CPU_CYCLES"
#define PFMON_DFL_PLM	PFM_PLM3

#define PFMON_DFL_DEAR_SMPL_RATE	1
#define PFMON_DFL_DEAR_OVFL_RATE	1
#define PFMON_DFL_IEAR_SMPL_RATE	1
#define PFMON_DFL_IEAR_OVFL_RATE	1
#define PFMON_DFL_BTB_SMPL_RATE		4 /* assume 4 pairs (source/target) */
#define PFMON_DFL_BTB_OVFL_RATE		4

#define PFMON_DFL_SMPL_ENTRIES	2048

#define PFMON_DFL_IEAR_FP	stdout
#define PFMON_DFL_DEAR_FP	stdout
#define PFMON_DFL_BTB_FP	stdout

#define OVERFLOW_SIG	SIGPROF

#define R_PMD(x)		(1<<(x))
#define DEAR_REGS_MASK		(R_PMD(2)|R_PMD(3)|R_PMD(17))
#define IEAR_REGS_MASK		(R_PMD(0)|R_PMD(1))
#define BTB_REGS_MASK		(R_PMD(8)|R_PMD(9)|R_PMD(10)|R_PMD(11)|R_PMD(12)|R_PMD(13)|R_PMD(14)|R_PMD(15)|R_PMD(16))

typedef struct {
	struct {
		unsigned int opt_plm;	/* which privilege level to monitor (more than one possible) */
		unsigned int opt_debug;	/* print debug information */
		unsigned int opt_verbose;	/* verbose output */
		unsigned int opt_append;	/* append to output file */
		unsigned int opt_raw_trace;	/* raw sampling trace files */
		unsigned int opt_noblock;	/* do not block child on overflow */
		unsigned int opt_ffork;	/* follow across fork */

		unsigned int opt_btb_notar;	
		unsigned int opt_btb_notac;	
		unsigned int opt_btb_tm;	
		unsigned int opt_btb_ptm;	
		unsigned int opt_btb_ppm;	
		unsigned int opt_btb_bpt;	
		unsigned int opt_btb_nobac;	
	} program_opt_flags;

	char *opt_outfile;		/* output file name for counters */

	char *smpl_file;
	FILE *smpl_fp;

	/* XXX may be able to collapse into single data structure */

	unsigned long smpl_entries;	/* number of entries in sampling buffer */
	unsigned long smpl_regs;	/* which PMDs are record in sampling buffer */

	unsigned long dear_smpl_rate;	/* D-EAR reload value after notification */
	unsigned long dear_ovfl_rate;	/* D-EAR reload value after overflow */

	unsigned long iear_smpl_rate;	/* I-EAR reload value after notification */
	unsigned long iear_ovfl_rate;	/* I-EAR reload value after overflow */

	unsigned long btb_smpl_rate;	/* BTB reload value after notification */
	unsigned long btb_ovfl_rate;	/* BTB reload value after overflow */

	perfmon_smpl_hdr_t 	*smpl_hdr;	/* virtual address of sampling buffer header */
	perfmon_smpl_entry_t	*smpl_addr;	/* virtual address of first sampling entry */

	pid_t	      notify_pid;	/* who to notify on overflow */
} program_opt_t;

#define opt_plm		program_opt_flags.opt_plm
#define opt_debug	program_opt_flags.opt_debug
#define opt_verbose	program_opt_flags.opt_verbose
#define opt_append	program_opt_flags.opt_append
#define opt_raw_trace	program_opt_flags.opt_raw_trace
#define opt_noblock	program_opt_flags.opt_noblock
#define opt_ffork	program_opt_flags.opt_ffork
#define opt_btb_notar	program_opt_flags.opt_btb_notar
#define opt_btb_notac	program_opt_flags.opt_btb_notac
#define opt_btb_nobac	program_opt_flags.opt_btb_nobac
#define opt_btb_tm	program_opt_flags.opt_btb_tm
#define opt_btb_ptm	program_opt_flags.opt_btb_ptm
#define opt_btb_ppm	program_opt_flags.opt_btb_ppm

static program_opt_t options;	/* keep track of global program options */

/*
 * This table is used to ease the overflow notification processing
 * It contains a reverse index of the events being monitored.
 * For every hardware counter it gives the corresponding programmed event.
 * This is useful when you get the raw bitvector from the kernel and need
 * to figure out which event it correspond to.
 *
 * This needs to be global because access from the overflow signal
 * handler.
 */
static int rev_pc[PMU_MAX_COUNTERS];	

/*
 * global variables needed during measurements
 */
static jmp_buf jbuf;	/* setjmp buffer */
static int child_pid;	/* process id of signaling child */

static void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

int
gen_events(char *arg, pfm_event_config_t *evt)
{
	char *v = arg, *p;
	int *opt_lst = evt->pec_evt;
	int ev;
	int cnt=0;

	while (v) {

		if (cnt == PMU_MAX_COUNTERS) goto too_many;

		p = strchr(v,',');

		if ( p ) *p++ = '\0';

		/* must match vcode only */
		if ((ev = pfm_findevent(v,0)) == -1) goto error;

		opt_lst[cnt++] = ev;

		v = p;
	}
	return cnt;
error:
	warning("unknown event %s\n", v);
	return -1;
too_many:
	warning("too many events specified, max=%d\n", PMU_MAX_COUNTERS);
	return -1;
}

static void
gen_thresholds(char *arg, pfm_event_config_t *evt)
{
	char *p;
	int cnt=0, thres;

	/*
	 * the default value for the threshold is 0: this means at least once 
	 * per cycle.
	 */
	if (arg == NULL) {
		int i;
		for (i=0; i < evt->pec_count; i++) evt->pec_thres[i] = 0;
		return;
	}
	while (arg) {

		if (cnt == PMU_MAX_COUNTERS || cnt == evt->pec_count) goto too_many;

		p = strchr(arg,',');

		if ( p ) *p++ = '\0';

		thres = atoi(arg);

		if (thres < 0) goto bad_value;

		/*
		 *  threshold = multi-occurence -1
		 * this is because by setting threshold to n, one counts only
		 * when n+1 or more events occurs per cycle.
	 	 */
		if (thres > pfm_event_threshold(evt->pec_evt[cnt])-1 ) goto too_big;

		evt->pec_thres[cnt++] = thres;

		arg = p;
	}
	return;
bad_value:
	fatal_error("Invalid threshold: %d\n", thres);
too_big:
	fatal_error("Event %d: threshold must be in [0-%d[\n", cnt, pfm_event_threshold(evt->pec_evt[cnt]));
too_many:
	fatal_error("Too many thresholds specified\n");
}

static int
dump_raw_smpl_buffer(void)
{
	perfmon_smpl_hdr_t *hdr = options.smpl_hdr;
	static unsigned int been_there;

	if (been_there == 0) {
		if (fwrite(hdr, sizeof(*hdr), 1, options.smpl_fp) != 1) goto error;
		been_there = 1;
	}

	if (fwrite((hdr+1), hdr->hdr_entry_size, hdr->hdr_count, options.smpl_fp) != 1) goto error;

	return 0;
error:
	warning("Can't write to raw sampling file: %s\n", strerror(errno));
	return -1;
}

static int
show_btb_reg(int j, perfmon_reg_t *reg)
{
	int ret;
	int is_valid = reg->pmd8_15_reg.btb_b == 0 && reg->pmd8_15_reg.btb_mp == 0 ? 0 :1; 

	ret = fprintf(options.smpl_fp, "\tPMD%-2d: 0x%016lx b=%d mp=%d valid=%c\n",
			j,
			reg->pmu_reg,
			 reg->pmd8_15_reg.btb_b,
			 reg->pmd8_15_reg.btb_mp,
			is_valid ? 'Y' : 'N');

	if (!is_valid) return ret;

	if (reg->pmd8_15_reg.btb_b) {
		ret = fprintf(options.smpl_fp, "\t\tSource Address: 0x%016lx (slot %d)\n"
						"\t\tPrediction: %s\n\n",
			 reg->pmd8_15_reg.btb_addr<<4,
			 reg->pmd8_15_reg.btb_slot,
			 reg->pmd8_15_reg.btb_mp ? "Failure" : "Success");
	} else {
		ret = fprintf(options.smpl_fp, "\t\tTarget Address: 0x%016lx\n\n",
			 reg->pmd8_15_reg.btb_addr<<4);
	}

	return ret;
}

static int
show_btb_trace(perfmon_reg_t *reg, perfmon_reg_t **btb_regs)
{
	int i, last, ret=1; /* ret=0 means error */


	i    = (reg->pmd16_reg.btbi_full) ? reg->pmd16_reg.btbi_bbi : 0;
	last = reg->pmd16_reg.btbi_bbi;

	if (options.opt_debug) {
		printf("btb_trace: i=%d last=%d bbi=%d full=%d\n", i, last,reg->pmd16_reg.btbi_bbi,reg->pmd16_reg.btbi_full );
	}
	do {
		if (btb_regs[i]) ret = show_btb_reg(i+8, btb_regs[i]);
		i = (i+1) % 8;
	} while (i != last);
	return ret;
}

static int
process_smpl_buffer(void)
{
	static const char *tlb_levels[]={"not captured", "L2 Data TLB", "VHPT", "OS Handler"};
	static const char *tlb_hdls[]={"VHPT", "OS"};
	static unsigned long entry_number;

	perfmon_smpl_hdr_t *hdr = options.smpl_hdr;
	perfmon_smpl_entry_t *ent = options.smpl_addr;
	unsigned long pos, msk;
	perfmon_reg_t *pmd16;
	perfmon_reg_t *btb_regs[PMU_MAX_BTB];
	perfmon_reg_t *reg;
	int has_btb_trace;
	int i, j, ret;


	if (options.smpl_hdr == NULL) return 0;	/* nothing to do */

	if (options.opt_raw_trace) return dump_raw_smpl_buffer();

	if (entry_number == 0) {
		/*fprintf(options.smpl_fp, "D-EAR rate=%ld entries=%ld\n", options.dear_rate, options.dear_entries);*/
		fprintf(options.smpl_fp, "Format=v%d entry_size=%ld bytes\n", hdr->hdr_version, hdr->hdr_entry_size);
		fprintf(options.smpl_fp, "Recorded PMDs=0x%016lx\n", hdr->hdr_pmds);

		if (options.smpl_file) printf("Sampling results are in file \"%s\"\n", options.smpl_file); 
	}

	if (hdr->hdr_version != 0x1) {
		warning("Sampling format %d is not supported\n", hdr->hdr_version);
		return -1;
	}

	pos = (unsigned long)ent;
	has_btb_trace = options.smpl_regs & (1<<16);

	if (options.opt_debug) printf("hdr_count=%ld smpl_regs=0x%lx has_btb_trace=%c\n",hdr->hdr_count, options.smpl_regs, has_btb_trace ? 'Y':'N');

	for(i=0; i < hdr->hdr_count; i++) {

		ret = 0;

		ret += fprintf(options.smpl_fp, 
			"\nEntry %ld PID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			entry_number++,
			ent->pid,
			ent->cpu,
			ent->stamp,
			ent->ip);

		ret += fprintf(options.smpl_fp, "\tPMD OVFL: ");

		for(j=PMU_FIRST_COUNTER, msk = ent->regs; msk; msk >>=1, j++) {	
			if (msk & 0x1) fprintf(options.smpl_fp, "%s(%d) ", pfm_event_name(rev_pc[j-PMU_FIRST_COUNTER]), j);
		}

		ret += fputc('\n', options.smpl_fp);

		reg = (perfmon_reg_t *)(ent+1);

		pmd16 = NULL;
		/* if contains btb */
		if (has_btb_trace) memset(btb_regs, 0, sizeof(btb_regs));

		for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			switch(j) {
				case 0:
					ret += fprintf(options.smpl_fp, "\tPMD0  : 0x%016lx, valid=%c cache line 0x%lx",
							reg->pmu_reg,
							reg->pmd0_reg.iear_v ? 'Y': 'N',
							reg->pmd0_reg.iear_icla<<5L);

					if (pfm_is_iear_tlb(rev_pc[j]) || pfm_is_dear_tlb(rev_pc[j]))
						ret += fprintf(options.smpl_fp, " (TLB %s)\n", tlb_hdls[reg->pmd0_reg.iear_tlb]);
					else
						ret += fprintf(options.smpl_fp, "\n");
					break;
				case 1:
					ret += fprintf(options.smpl_fp, "\tPMD1  : 0x%016lx, (Latency %d)\n",
							reg->pmu_reg,
							reg->pmd1_reg.iear_lat);
					break;
				case 3:
					ret += fprintf(options.smpl_fp, "\tPMD3  : 0x%016lx ", reg->pmu_reg);

					/* it's just a guess here ! */
					if (reg->pmd3_reg.dear_level == 0)
						ret += fprintf(options.smpl_fp, ", Latency %d\n", reg->pmd3_reg.dear_lat);
					else
						ret += fprintf(options.smpl_fp, ", TLB %s\n", tlb_levels[reg->pmd3_reg.dear_level]);
					break;
				case 16:
#if 0
					ret += fprintf(options.smpl_fp, "\tPMD16 : 0x%016lx bbi=%d full=%c last_written=PMD%d\n",
							reg->pmu_reg,
							reg->pmd16_reg.btbi_bbi,
							reg->pmd16_reg.btbi_full ? 'Y': 'N',
							8+(((8*	reg->pmd16_reg.btbi_full)+(reg->pmd16_reg.btbi_bbi-1))%8));
#endif
					/*
					 * keep track of what the BTB index is saying
					 */
					pmd16 = reg;
					break;
				case 17:
					ret += fprintf(options.smpl_fp, "\tPMD17 : 0x%016lx (slot %d) valid=%c\n",
							reg->pmd17_reg.dear_iaddr << 4,
							reg->pmd17_reg.dear_slot,
							reg->pmd17_reg.dear_v ? 'Y': 'N');
					break;
				default:
					/*
					 * If we find a BTB then record it for later
					 */
					if (j>7 && j < 16) 
						btb_regs[j-8] = reg;
					else
						ret += fprintf(options.smpl_fp, "\tPMD%-2d : 0x%016lx\n", j, reg->pmu_reg);
			}
			reg++;
			/* Lazily detect output error now */
			if (ret == 0) goto error;
		}
		/*
		 *  we display BTB at the end because that's the only way
		 *  to reflect the timeline in which they have been captured
		 *  due to the BTB wraparound characteristic
		 */
		if (pmd16) show_btb_trace(pmd16, btb_regs);

		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	
	}
	return 0;
error:
	warning("Can't write to sampling file: %s\n", strerror(errno));
	return -1;
}

/* XXX: should be done by kernel on restart */
static void
reset_btb_index(int pid)
{
	perfmon_req_t pd[1];

	memset(pd, 0, sizeof(pd));

	pd[0].pfr_reg.reg_num = 16;

	if (perfmonctl(pid, PFM_WRITE_PMDS, 0, pd, 1) == -1) {
		warning("Cannot reset BTB index for process %d", pid);
	}
	if (options.opt_debug)
		printf("PMD16 is reset for process %d\n", pid);
}

static int
check_overflow(int pid, int i)
{
	if (options.opt_verbose) 
		printf("Overflow on PMD%d %s\n", i, pfm_event_name(rev_pc[i - PMU_FIRST_COUNTER]));

	/*
	 * does not work since 2.4.3
	 * if (pfm_is_btb(rev_pc[i - PMU_FIRST_COUNTER])) reset_btb_index(pid);
	 */

	/* if we are sampling, then dump the buffer */
	if (options.smpl_hdr) return process_smpl_buffer();

	return 0;
}

void
overflow_handler(int n, struct mysiginfo *info, struct sigcontext *sc)
{
	unsigned long mask =info->sy_pfm_ovfl;
	int i;

	if (options.opt_debug) 
		printf("Overflow notification: pid=%d bv=0x%lx\n", info->sy_pid, info->sy_pfm_ovfl);

	for(i= PMU_FIRST_COUNTER; mask; mask >>=1, i++) {

		if (options.opt_debug) printf("mask=0x%lx i=%d\n", mask, i);

		/*
		 * On error in overflow processing (no disk space) we stop processing overflows
		 * but we need to restart child process to run to completion or at least get a
		 * chance to kill it. We send the signal first and then wakeup the process.
		 */
		if ((mask & 0x1) != 0  && check_overflow(info->sy_pid, i) != 0) {

			printf("Overflow problem: killing process %d\n", info->sy_pid);

			kill(info->sy_pid, SIGKILL); /* could ne nicer ? */

			break;
		}
	}

	if (perfmonctl(info->sy_pid, PFM_RESTART, 0, 0, 0) == -1) {
		warning("overflow cannot restart process %d: %d\n", info->sy_pid, errno);
	}
}

void
child_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	if (options.opt_debug) printf("SIGCHLD handler for %d\n", info->si_pid);
	/*
	 * we need to record the child pid here because we need to avoid
	 * a race condition with the parent returning from fork().
	 * In some cases, the pid=fork() instruction is not completed before
	 * we come to the SIGCHILD handler. the pid variable still has its
	 * default (zero) value. That's because the signal was received on
	 * return from fork() by the parent.
	 * So here we keept track of who just died and use a global variable
	 * to pass it back to the parent.
	 */
	child_pid = info->si_pid;

	/*
	 * That's not very pretty but that's one way of avoiding a race
	 * condition with the pause() system call. You may deadlock if the 
	 * signal is delivered before the parent reaches the pause() call.
	 * Using a variable and test reduces the window but it still exists.
	 * longjmp/setjmp avoids it completely.
	 */
	longjmp(jbuf,1);
}

/*
 * Does the pretty printing of results
 */
void
print_results(pfm_event_config_t *evt, perfmon_req_t *pc, perfmon_req_t *pd)
{
	int i;
	FILE *fp = NULL;

	if (options.opt_outfile) {
		fp = fopen(options.opt_outfile, options.opt_append ? "a" : "w");
		if (fp == NULL) {
			warning("cannot open %s for writing, defaulting to stdout\n", options.opt_outfile);
		}
	}

	if (fp == NULL)	{
		fp = stdout;
	} else {
		printf("Results are in file \"%s\"\n",options.opt_outfile);
	}	

	if (options.smpl_hdr) 
		process_smpl_buffer();
	else
		for (i=0; i < evt->pec_count; i++) {
			if (options.opt_verbose)
				fprintf(fp,"counter=0x%lx thres=%d count=%-16lu event=%s\n",
					pc[i].pfr_reg.reg_num,
					evt->pec_thres[i],
					pd[i].pfr_reg.reg_value,
					pfm_event_name(evt->pec_evt[i]));
			else
				fprintf(fp, "%-16lu %s\n",
					pd[i].pfr_reg.reg_value,
					pfm_event_name(evt->pec_evt[i]));
		}
	if (options.opt_outfile) fclose(fp);
}

static void
gen_reverse_table(pfm_event_config_t *evt, perfmon_req_t *pc)
{
	int i;

	/*
	 * evt can only contain counter events, therefore it is
	 * safe to assume the following code.
	 *
	 * The only assumption made by this code is that the
	 * counter configurations always come first, then
	 * the EAR, BTB, Opcode....
	 */
	for (i=0; i < evt->pec_count; i++) {
		rev_pc[pc[i].pfr_reg.reg_num- PMU_FIRST_COUNTER] = evt->pec_evt[i];
		if (options.opt_debug) 
			printf("rev_pc[%ld]=%s\n",pc[i].pfr_reg.reg_num-PMU_FIRST_COUNTER, pfm_event_name(evt->pec_evt[i]));

	}
}

/*
 * Executed in the context of the child
 */
void
install_counters(int pid, pfm_event_config_t *evt, perfmon_req_t *pc, int count)
{
	perfmon_req_t pd[PMU_MAX_PMDS];
	int i;

	/* 
	 * reset PMU (guarantee not active on return) and unfreeze
	 * must be done before writing to any PMC/PMD
	 */ 
	if (perfmonctl(pid, PFM_ENABLE, 0, 0, 0) == -1) {
		if (errno == ENOSYS) 
			fatal_error("Your kernel does not have performance monitoring support !\n");

		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/* we start monitoring as close as possible from exec() */
	if (perfmonctl(pid, PFM_WRITE_PMCS, 0, pc, count) == -1) {

		fatal_error("child: perfmonctl error WRITE_PMCS errno %d\n",errno);
	}
	memset(pd, 0, sizeof(pd));

	/*
	 * For now, EARs pid and signals are setup on the PMCs
	 * However initial value and reset values are setup as part of PMDS
	 */
	for(i=0; i < evt->pec_count; i++) {

		pd[i].pfr_reg.reg_num = pc[i].pfr_reg.reg_num;

		if (pfm_is_dear(evt->pec_evt[i])) {
			pd[i].pfr_reg.reg_value = options.dear_smpl_rate;
			pd[i].pfr_reg.reg_ovfl_reset = options.dear_ovfl_rate;
			pd[i].pfr_reg.reg_smpl_reset = options.dear_smpl_rate; /* XXX: adds extra notification cost */
		} else if (pfm_is_iear(evt->pec_evt[i])) {
			pd[i].pfr_reg.reg_value = options.iear_smpl_rate;
			pd[i].pfr_reg.reg_ovfl_reset = options.iear_ovfl_rate;
			pd[i].pfr_reg.reg_smpl_reset = options.iear_smpl_rate; /* XXX: adds extra notification cost */
		} else if (pfm_is_btb(evt->pec_evt[i])) {
			pd[i].pfr_reg.reg_value = options.btb_smpl_rate;
			pd[i].pfr_reg.reg_ovfl_reset = options.btb_ovfl_rate;
			pd[i].pfr_reg.reg_smpl_reset = options.btb_smpl_rate; /* XXX: adds extra notification cost */
		} else {
			pd[i].pfr_reg.reg_value = 0;
		}

		/* we always notify on overflow in this program */
		pd[i].pfr_reg.reg_flags = PFM_REGFL_OVFL_NOTIFY;

	}

	if (perfmonctl(pid, PFM_WRITE_PMDS, 0, pd, evt->pec_count) == -1) {
		fatal_error( "child: perfmonctl error WRITE_PMDS errno %d\n",errno);
	}

}


/*
 * Does the actual measurements
 */
void
do_measures(pfm_event_config_t *evt, char **argv)
{
	perfmon_req_t pd[PMU_MAX_PMDS];
	perfmon_req_t pc[PMU_MAX_PMCS];	/* configuration */
	perfmon_req_t ctx[1];
	int ret, i, status, pid, mypid=getpid(); /* keep the compiler happy ! */
	int count = sizeof(pc)/sizeof(perfmon_req_t);

	/*
	 * assign events to counters, configure additional PMCs`
	 * count may be greater than pec_count when non trivial features are used
	 */
	if (pfm_dispatch_events(evt, pc, &count) == -1) {
		fatal_error("Can't satisfy constraints on counters\n");
	}
		
	if (options.opt_debug) {
		int j;
		printf("do_measures: count=%d pec=%d\n", count, evt->pec_count);
		for(j=0; j < count; j++) {
			printf("pc[%d].reg_num=%ld\n", j, pc[j].pfr_reg.reg_num);
		}
	}
	gen_reverse_table(evt, pc);

	/* get a clean context request */
	memset(ctx, 0, sizeof(ctx));

	/* 
	 * check for EAR or BTB counters, if so request a sampling buffer in context
	 * XXX: may extend to adjust buffer size based on type 
	 */
	for(i=0; i < evt->pec_count; i++) {
		if (pfm_is_dear(evt->pec_evt[i])) {
			ctx[0].pfr_ctx.smpl_entries = options.smpl_entries;
			options.smpl_regs |= DEAR_REGS_MASK;
		} else if (pfm_is_iear(evt->pec_evt[i])) {
			ctx[0].pfr_ctx.smpl_entries = options.smpl_entries;
			options.smpl_regs |= IEAR_REGS_MASK;
		} else if (pfm_is_btb(evt->pec_evt[i])) {
			ctx[0].pfr_ctx.smpl_entries = options.smpl_entries;
			options.smpl_regs |= BTB_REGS_MASK;
		} 
	}
	ctx[0].pfr_ctx.smpl_regs = options.smpl_regs;

	if (options.opt_debug) {
		printf("context.smpl_regs=0x%lx\n",ctx[0].pfr_ctx.smpl_regs);
		printf("context.smpl_entries=%ld\n",ctx[0].pfr_ctx.smpl_entries);
	}

	ctx[0].pfr_ctx.notify_pid = mypid;
	ctx[0].pfr_ctx.notify_sig = SIGPROF;

	ctx[0].pfr_ctx.flags     |= options.opt_ffork ? PFM_FL_INHERIT_ALL : PFM_FL_INHERIT_ONCE;
	ctx[0].pfr_ctx.flags     |= options.opt_noblock ? PFM_FL_SMPL_OVFL_NOBLOCK: 0;
	if (options.opt_debug) {
		printf("context_flags=0x%x\n", ctx[0].pfr_ctx.flags); 
	}

	/* XXX: need to add noblock here */
	if (perfmonctl(mypid, PFM_CREATE_CONTEXT, 0 , ctx, 1) == -1 ) {
		fatal_error("Can't create PFM context %d\n", errno);
	}

	/*
	 * extract virtual address of buffers in our address space
	 * Some values may be null if not requested
	 */
	if (options.smpl_regs) {
		options.smpl_hdr  = ctx[0].pfr_ctx.smpl_vaddr;
		options.smpl_addr = (perfmon_smpl_entry_t *)(options.smpl_hdr+1);
	}

	if (options.opt_debug) {
		printf("smpl_hdr=%p smpl_addr=%p\n", (void *)options.smpl_hdr, (void *)options.smpl_addr);
	}

	/* back from signal handler ? */
	if (setjmp(jbuf) == 1) goto extract_results;

	if ((pid=fork()) == -1) fatal_error("Cannot fork process\n");

	if (pid == 0) {		 
		/* child */
		 pid = getpid();

		if (options.opt_verbose) printf("Starting %s [%d]\n", argv[0], pid);

		install_counters(pid, evt, pc, count);

		pfm_start();

		ret = execvp(argv[0], argv);

		fatal_error("child: cannot exec %s: %s\n", argv[0], strerror(errno));
		/* NOT REACHED */
	}
	/* 
	 * Block for child to finish
	 * The child process may already be done by the time we get here.
	 * The parent will skip to the next statement directly in this case.
	 */
	do { 
		for(;;) pause();

extract_results:
		pid = child_pid; /* make sure we get the right child */

		memset(pd, 0, sizeof(pd));

		/* XXX: must only pick up counters, not other configuration registers */
		for(i=0; i < evt->pec_count; i++) {

			pd[i].pfr_reg.reg_num = pc[i].pfr_reg.reg_num;

			if (options.opt_debug)
				printf("reading: pd[%d].reg_num=%ld\n", i, pd[i].pfr_reg.reg_num);
		}

		/*
		 * read the PMDS now using our context (collector)
		 */
		if (perfmonctl(pid, PFM_READ_PMDS, 0, pd, evt->pec_count) == -1) {
			fatal_error("perfmonctl error READ_PMDS for process %d errno %d\n", pid, errno);
		}

		/* cleanup child now */
		waitpid(pid, &status, 0);

		/* we may or may not want to check child exit status here */
		if (WEXITSTATUS(status) != 0) {
			printf("Warning: process %d exited with non zero value (%d): results may be incorrect\n", pid, WEXITSTATUS(status));
		}

		if (options.opt_verbose) 
			printf("process %d exited with status %d\n", pid, WEXITSTATUS(status));


		if (options.opt_debug) {
			for (i=0; i < PMU_MAX_COUNTERS; i++) {
				printf("pmd[%d]=%lx\n", i, pd[i].pfr_reg.reg_value);
			}
		}

		/* dump results */
		print_results(evt, pc, pd);

	} while (0);
}

void
list_all_events(void)
{
	int i;

	for(i=pfm_get_firstevent(); i != -1; i = pfm_get_nextevent(i)) {
		printf("\t%s\n", pfm_event_name(i));
	}
}

static struct option cmd_options[]={
	{ "event-info", 1, 0, 1},
	{ "show-event-list", 0, 0, 2 },
	{ "debug", 0, &options.opt_debug, 1 },
	{ "kernel-level", 0, 0, 4 },
	{ "user-level", 0, 0, 5 },
	{ "events", 1, 0, 6 },
	{ "help", 0, 0, 7 },
	{ "verbose", 0, &options.opt_verbose, 1 },
	{ "version", 0, 0, 9 },
	{ "event-thresholds", 1, 0, 10 },
	{ "print", 0, 0, 11 },
	{ "outfile", 1, 0, 12 },
	{ "append", 0, &options.opt_append, 1},
	{ "dear-smpl-rate", 1, 0, 14},
	{ "iear-smpl-rate", 1, 0, 15},
	{ "btb-smpl-rate", 1, 0, 16},
	{ "smpl-entries", 1, 0, 17},
	{ "dear-ovfl-rate", 1, 0, 18},
	{ "iear-ovfl-rate", 1, 0, 19},
	{ "opc-match8", 1, 0, 20},
	{ "opc-match9", 1, 0, 21},
	{ "smpl-file", 1, 0, 22},
	{ "follow-fork", 0, 0, 23},
	{ "btb-all-mispredicted", 0, 0, 24},
	{ "raw-trace-file",0, &options.opt_raw_trace, 1},
	{ "overflow-noblock",0, &options.opt_noblock, 1},
	{ "btb-ovfl-rate", 1, 0, 27},
	{ "btb-no-tar", 0, &options.opt_btb_notar, 1},
	{ "btb-no-bac", 0, &options.opt_btb_nobac, 1},
	{ "btb-no-tac", 0, &options.opt_btb_notac, 1},
	{ "btb-tm-tk", 0, &options.opt_btb_tm, 0x2},
	{ "btb-tm-ntk", 0, &options.opt_btb_tm, 0x1},
	{ "btb-ptm-correct", 0, &options.opt_btb_ptm, 0x2},
	{ "btb-ptm-incorrect", 0, &options.opt_btb_ptm, 0x1},
	{ "btb-ppm-correct", 0, &options.opt_btb_ppm, 0x2},
	{ "btb-ppm-incorrect", 0, &options.opt_btb_ppm, 0x1},
	{ 0, 0, 0, 0}
};

static void
usage(char **argv)
{
	printf("Usage: %s [OPTIONS]... COMMAND\n", argv[0]);

	printf(	"-h, --help\t\t\t\tdisplay this help and exit\n"
		"--version\t\t\t\toutput version information and exit\n"
		"-l, --show-event-list\t\t\tdisplay list of supported events by name\n"
		"-i <event>, --event-info=event\t\tdisplay information about an event\n"
		"-u, --user-level\t\t\tmonitor at the user level for all events\n"
		"-k, --kernel-level\t\t\tmonitor at the kernel level for all events\n"
		"-e, --events=ev1,ev2,...\t\tselect events to monitor (no space)\n"
		"--event-thresholds=thr1,thr2,...\tset event thresholds (no space)\n"
		"--debug\t\t\t\t\tenable debug prints\n"
		"--verbose\t\t\t\tprint more information during execution\n"
		"--outfile=filename\t\t\tprint results in a file\n"
		"--append\t\t\t\tappend results to outfile\n"
		"--dear-smpl-rate=val\t\t\tset data EAR sampling rate after notification\n"
		"--dear-ovfl-rate=val\t\t\tset data EAR sampling rate\n"
		"--iear-smpl-rate=val\t\t\tset instruction EAR sampling rate after notification\n"
		"--iear-ovfl-rate=val\t\t\tset instruction EAR sampling rate\n"
		"--btb-smpl-rate=val\t\t\tset BTB sampling rate after notification\n"
		"--btb-ovfl-rate=val\t\t\tset BTB sampling rate\n"
		"--smpl-entries=val\t\t\tset number of entries for sampling buffer\n"
		"--opc-match8=val\t\t\tset opcode match for PMC8\n"
		"--opc-match9=val\t\t\tset opcode match for PMC9\n"
		"--smpl-file=filename\t\t\tfile to save the sampling results\n"
		"--raw-trace-file\t\t\tsampling results will be dumped in binary form\n"
		"--overflow-noblock\t\t\tDon't block process on overflow\n"
		"--follow-fork\t\t\t\tmonitor performance on all children\n"
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

	);
}

static void
setup_signals(void)
{
	struct sigaction act;

	/* Install SIGCHLD handler */
	memset(&act,0,sizeof(act));

	act.sa_handler = (sig_t)child_handler;
	act.sa_flags   = SA_NOCLDSTOP;
	sigaction (SIGCHLD, &act, 0);

	memset(&act,0,sizeof(act));
	act.sa_handler = (sig_t)overflow_handler;
	sigaction (OVERFLOW_SIG, &act, 0);
}

static void
setup_ears_btb_rates(void)
{

	if (options.dear_smpl_rate == 0) {
		options.dear_smpl_rate = PFMON_DFL_DEAR_SMPL_RATE;
	}
	if (options.dear_ovfl_rate == 0) {
		options.dear_ovfl_rate = PFMON_DFL_DEAR_OVFL_RATE;
	}

	if (options.iear_smpl_rate == 0) {
		options.iear_smpl_rate = PFMON_DFL_IEAR_SMPL_RATE;
	}
	if (options.iear_ovfl_rate == 0) {
		options.iear_ovfl_rate = PFMON_DFL_IEAR_OVFL_RATE;
	}

	if (options.btb_smpl_rate == 0) {
		options.btb_smpl_rate = PFMON_DFL_BTB_SMPL_RATE;
	}
	if (options.btb_ovfl_rate == 0) {
		options.btb_ovfl_rate = PFMON_DFL_BTB_OVFL_RATE;
	}

	if (options.smpl_entries == 0) {
		options.smpl_entries = PFMON_DFL_SMPL_ENTRIES;
	}

	if (options.smpl_file) {
		options.smpl_fp = fopen(options.smpl_file, "w");
		if (options.smpl_fp==NULL)
			fatal_error("Cannot create sampling file %s : %d\n", options.smpl_file, errno); 
	} else 
		options.smpl_fp = PFMON_DFL_DEAR_FP;

	options.dear_smpl_rate *= -1;
	options.iear_smpl_rate *= -1;
	options.btb_smpl_rate  *= -1;

	options.dear_ovfl_rate *= -1;
	options.iear_ovfl_rate *= -1;
	options.btb_ovfl_rate  *= -1;

	if (options.opt_verbose) {
		printf("Sampling entries: %ld Sampling file: %s\n", options.smpl_entries, options.smpl_file ? options.smpl_file : "standard output");

		printf("Sampling Rates:\n\tD-EAR rate: 0x%016lx (%ld)\n\tI-EAR rate: 0x%016lx (%ld)\n\t  BTB rate: 0x%016lx (%ld)\n",
			options.dear_smpl_rate,
			options.dear_smpl_rate,
			options.iear_smpl_rate,
			options.iear_smpl_rate,
			options.btb_smpl_rate,
			options.btb_smpl_rate);
		printf("Overflow Rates:\n\tD-EAR rate: 0x%016lx (%ld)\n\tI-EAR rate: 0x%016lx (%ld)\n\t  BTB rate: 0x%016lx (%ld)\n",
			options.dear_ovfl_rate,
			options.dear_ovfl_rate,
			options.iear_ovfl_rate,
			options.iear_ovfl_rate,
			options.btb_ovfl_rate,
			options.btb_ovfl_rate);
	}

}

static void
close_ears_btb(void)
{
	/* when we use stdout, then file is NULL , so this is safe*/
	if (options.smpl_file) fclose(options.smpl_fp);

	/* the virtual mappings and kernel buffers are automatically
	 * destroyed by kernel on exit when no other user exists.
	 */
}

static void
setup_btb_flags(pfm_event_config_t *evt)
{
	/* by default, the registers are setup to 
	 * record every possible branch.
	 * The record nothing is not available becuase it simply means
	 * don't use a BTB event.
	 * So the only thing the user can do is narrow down the type of
	 * branches to record. This simplifies the number of cases quite
	 * substancially.
	 */
	evt->pec_btb_tar = 1;
	evt->pec_btb_tac = 1;
	evt->pec_btb_bac = 1;
	evt->pec_btb_tm  = 0x3;
	evt->pec_btb_ptm = 0x3;
	evt->pec_btb_ppm = 0x3;

	if (options.opt_btb_notar) evt->pec_btb_tar = 0;
	if (options.opt_btb_notac) evt->pec_btb_tac = 0;
	if (options.opt_btb_nobac) evt->pec_btb_bac = 0;
	if (options.opt_btb_tm) evt->pec_btb_tm = options.opt_btb_tm & 0x3;
	if (options.opt_btb_ptm) evt->pec_btb_ptm = options.opt_btb_ptm & 0x3;
	if (options.opt_btb_ppm) evt->pec_btb_ppm = options.opt_btb_ppm & 0x3;

	if (options.opt_verbose) {
		printf("BTB options: TAR=%c TAC=%c BAC=%c TM=%d PTM=%d PPM=%d\n",
			evt->pec_btb_tar ? 'Y' : 'N',
			evt->pec_btb_tac ? 'Y' : 'N',
			evt->pec_btb_bac ? 'Y' : 'N',
			evt->pec_btb_tm,
			evt->pec_btb_ptm,
			evt->pec_btb_ppm);
	}
}

int
main(int argc, char **argv)
{
	pfm_event_config_t evt;	/* hold most configuration data */
	char *thres_arg = NULL;
	char *endptr = NULL;
	pfmlib_options_t pfmlib_options;
	int c;


	memset(&evt, 0, sizeof(evt));
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));

	while ((c=getopt_long(argc, argv,"kue:li:", cmd_options, 0)) != -1) {
		switch(c) {
			case   0: continue; /* fast path for options */
			case   1:
			case 'i':
				if (pfm_print_event_info(optarg, printf) == -1)
					fatal_error("Event %s is unknown\n", optarg);
				exit(0);
			case   2:
			case 'l':
				list_all_events();
				exit(0);

			case   4:
			case 'k':
				options.opt_plm |= PFM_PLM0;
				break;

			case   5:
			case 'u':
				options.opt_plm |= PFM_PLM3;
				break;

			case   6:
			case 'e':
				evt.pec_count = gen_events(optarg,&evt);
			       	if (evt.pec_count == -1) fatal_error("invalid events\n");
				break;

			case   7:
			case 'h':
				usage(argv);
				exit(0);
			case   9:
				printf("pfmon version %s Date: %s\n"
					"Copyright (C) 2001 Hewlett-Packard Company\n"
					"Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>\n",
					PFMON_VERSION, __DATE__);
				exit(0);
			case  10:
				thres_arg = optarg;
				break;
			case  12:
				options.opt_outfile = optarg;
				break;
			case  14:
				options.dear_smpl_rate = strtoul(optarg, &endptr, 10);
				break;
			case  15:
				options.iear_smpl_rate = strtoul(optarg, &endptr, 10);
				break;
			case  16:
				options.btb_smpl_rate = strtoul(optarg, &endptr, 10);
				break;
			case  17:
				options.smpl_entries = strtoul(optarg, &endptr, 10);
				break;

			case  18:
				options.dear_ovfl_rate = strtoul(optarg, &endptr, 10);
				break;
			case  19:
				options.iear_ovfl_rate = strtoul(optarg, &endptr, 10);
				break;
			case  20:
				evt.pec_pmc8 = strtoul(optarg, &endptr, 16);
				break;
			case  21:
				evt.pec_pmc9 = strtoul(optarg, &endptr, 16);
				break;
			case  22:
				options.smpl_file = optarg;
				break;
			case  23:
				warning("Ignoring the --follow-fork option: not yet available");
				break;
			case  24:
				/* shortcut to the following options
				 * must not be used with other btb options
				 */
				options.opt_btb_notar = 0;
				options.opt_btb_nobac = 0;
				options.opt_btb_notac = 0;
				options.opt_btb_tm    = 0x3;
				options.opt_btb_ptm   = 0x1;
				options.opt_btb_ppm   = 0x1;
				break;
			case  27:
				options.btb_ovfl_rate = strtoul(optarg, &endptr, 10);
				break;
			default:
				fatal_error("Unknown option\n");
		}
	}

	if (options.opt_debug) pfmlib_options.pfm_debug = 1;

	if (optind == argc) {
		fatal_error("You need to specify a command to measure\n");
	}

	/*
	 * make sure we do at least one measure
	 */
	if (evt.pec_count == 0) {
		evt.pec_evt[0] = pfm_findeventbyname(PFMON_DFL_EVENT);

		if (evt.pec_evt[0] == -1)
			fatal_error("Default event %s not present\n",PFMON_DFL_EVENT); 

		evt.pec_count  = 1;

		warning("Defaulting to event %s only\n", PFMON_DFL_EVENT);
	}

	setup_ears_btb_rates();

	setup_btb_flags(&evt);

	atexit(close_ears_btb);

	setup_signals();

	options.notify_pid = getpid();

	if (options.opt_debug) {
		printf("%s process id is %d\n", argv[0], getpid());
	}
	/* 
	 * we systematically initialize thresholds to their minimal value
	 * or requested value
	 */
	gen_thresholds(thres_arg, &evt);

	if (options.opt_plm == 0) {
		options.opt_plm = PFMON_DFL_PLM;
		if (options.opt_verbose) 
			printf("Measuring at %s priviledge level ONLY\n", PFMON_DFL_PLM == PFM_PLM3 ? "user" : "kernel");
	}
	evt.pec_plm = options.opt_plm;

	/* enable some debugging in the library as well */
	pfmlib_config(&pfmlib_options);

	do_measures(&evt, argv+optind);

	return 0;
}
