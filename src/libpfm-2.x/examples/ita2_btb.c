/*
 * ita_btb.c - example of how use the BTB with the Itanium 2 PMU
 *
 * Copyright (C) 2002-2005 Hewlett-Packard Co
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium2.h>

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

/*
 * The BRANCH_EVENT is increment by 1 for each branch event. Such event is composed of
 * two entries in the BTB: a source and a target entry. The BTB is full after 4 branch
 * events.
 */
#define SMPL_PERIOD	(4UL*256)

/*
 * We use a small buffer size to exercise the overflow handler
 */
#define SMPL_BUF_NENTRIES	64

#define M_PMD(x)		(1UL<<(x))
#define BTB_REGS_MASK		(M_PMD(8)|M_PMD(9)|M_PMD(10)|M_PMD(11)|M_PMD(12)|M_PMD(13)|M_PMD(14)|M_PMD(15)|M_PMD(16))

static void *smpl_vaddr;
static unsigned long smpl_regs; /* just 64 registers for now */
static pfmlib_param_t evt;

/* 
 * we don't use static to make sure the compiler does not inline the function
 */
long func1(void) { return 0;}

long
do_test(unsigned long loop)
{
	long sum  = 0;

	pfm_start();
	while(loop--) {
		if (loop & 0x1) 
			sum += func1();
		else
			sum += loop;
	}
	pfm_stop();
	return sum;
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

/*
 * print content of sampling buffer
 *
 * XXX: using stdio to print from a signal handler is not safe with multi-threaded
 * applications
 */
#define safe_printf	printf

static int
show_btb_reg(int j, pfm_ita2_reg_t reg)
{
	int ret;
	int is_valid = reg.pmd8_15_ita2_reg.btb_b == 0 && reg.pmd8_15_ita2_reg.btb_mp == 0 ? 0 :1; 

	ret = safe_printf("\tPMD%-2d: 0x%016lx b=%d mp=%d valid=%c\n",
			j,
			reg.reg_val,
			 reg.pmd8_15_ita2_reg.btb_b,
			 reg.pmd8_15_ita2_reg.btb_mp,
			is_valid ? 'Y' : 'N');

	if (!is_valid) return ret;

	if (reg.pmd8_15_ita2_reg.btb_b) {
		unsigned long addr;

		addr = 	reg.pmd8_15_ita2_reg.btb_addr<<4;
		addr |= reg.pmd8_15_ita2_reg.btb_slot < 3 ?  reg.pmd8_15_ita2_reg.btb_slot : 0;

		ret = safe_printf("\t       Source Address: 0x%016lx\n"
				  "\t       Taken=%c Prediction: %s\n\n",
			 addr,
			 reg.pmd8_15_ita2_reg.btb_slot < 3 ? 'Y' : 'N',
			 reg.pmd8_15_ita2_reg.btb_mp ? "Failure" : "Success");
	} else {
		ret = safe_printf("\t       Target Address: 0x%016lx\n\n",
			 ((unsigned long)reg.pmd8_15_ita2_reg.btb_addr<<4));
	}
	return ret;
}

static void
show_btb(pfm_ita2_reg_t *btb, pfm_ita2_reg_t *pmd16)
{
	int i, last;


	i    = (pmd16->pmd16_ita2_reg.btbi_full) ? pmd16->pmd16_ita2_reg.btbi_bbi : 0;
	last = pmd16->pmd16_ita2_reg.btbi_bbi;

	do {
		show_btb_reg(i+8, btb[i]);
		i = (i+1) % 8;
	} while (i != last);
}


int
process_smpl_buffer(void)
{
	perfmon_smpl_hdr_t *hdr = (perfmon_smpl_hdr_t *)smpl_vaddr;
	perfmon_smpl_entry_t *ent;
	unsigned long pos;
	unsigned long smpl_entry = 0;
	pfm_ita2_reg_t *reg, *pmd16;
	int i, ret;

	/*
	 * Make sure the kernel uses the format we understand
	 */
	if (PFM_VERSION_MAJOR(hdr->hdr_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
		fatal_error("Perfmon v%u.%u sampling format is not supported\n", 
				PFM_VERSION_MAJOR(hdr->hdr_version),
				PFM_VERSION_MINOR(hdr->hdr_version));
	}

	pos = (unsigned long)(hdr+1);
	/*
	 * walk through all the entries recored in the buffer
	 */
	for(i=0; i < hdr->hdr_count; i++) {

		ret = 0;

		ent = (perfmon_smpl_entry_t *)pos;
		/*
		 * print entry header
		 */
		safe_printf("Entry %ld PID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			smpl_entry++,
			ent->pid,
			ent->cpu,
			ent->stamp,
			ent->ip);


		/*
		 * point to first recorded register (always contiguous with entry header)
		 */
		reg = (pfm_ita2_reg_t*)(ent+1);

		/*
		 * in this particular example, we have pmd8-pmd15 has the BTB. We have also
		 * included pmd16 (BTB index) has part of the registers to record. This trick
		 * allows us to get the index to decode the sequential order of the BTB.
		 *
		 * Recorded registers are always recorded in increasing order. So we know
		 * that pmd16 is at a fixed offset (+8*sizeof(unsigned long)) from pmd8.
		 */
		pmd16 = reg+8;
		show_btb(reg, pmd16);

		/*
		 * move to next entry
		 */
		pos += hdr->hdr_entry_size;

	}
	return 0;
}

static void
overflow_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	unsigned long mask =info->sy_pfm_ovfl[0];
	pfarg_reg_t pd[1];

	/*
	 * Check to see if we received a spurious SIGPROF, i.e., one not
	 * generated by the perfmon subsystem.
	 */
	if (info->sy_code != PROF_OVFL) {
		printf("Received spurious SIGPROF si_code=%d\n", info->sy_code);
		return;
	} 
	/*
	 * Each bit set in the overflow mask represents an overflowed counter.
	 *
	 * Here we check that the overflow was caused by our first counter.
	 */
	if ((mask & (1UL<< evt.pfp_pc[0].reg_num)) == 0) {
		printf("Something is wrong, unexpected mask 0x%lx\n", mask);
		exit(1);
	}

	/*
	 * Read the value of the second counter
	 */
	pd[0].reg_num = evt.pfp_pc[1].reg_num;

	if (perfmonctl(getpid(), PFM_READ_PMDS, pd, 1) == -1) {
		perror("PFM_READ_PMDS");
		exit(1);
	}

	printf("Notification received\n");
	process_smpl_buffer();
	/*
	 * And resume monitoring
	 */
	if (perfmonctl(getpid(), PFM_RESTART,NULL, 0) == -1) {
		perror("PFM_RESTART");
		exit(1);
	}
	/* Here we have the PMU enabled and are capturing events */
}


int
main(void)
{
	int ret;
	int type = 0;
	pid_t pid = getpid();
	pfmlib_ita2_param_t ita_param;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_context_t ctx[1];
	pfmlib_options_t pfmlib_options;
	struct sigaction act;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("Can't initialize library\n");
	}

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM2_PMU) {
		char *model; 
		pfm_get_pmu_name(&model);
		fatal_error("this program does not work with %s PMU\n", model);
	}

	/*
	 * Install the overflow handler (SIGPROF)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)overflow_handler;
	sigaction (SIGPROF, &act, 0);


	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);



	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&evt,0, sizeof(evt));
	memset(&ita_param,0, sizeof(ita_param));


	/*
	 * because we use a model specific feature, we must initialize the
	 * model specific pfmlib parameter structure and link it to the
	 * common structure.
	 * The magic number is a simple mechanism used by the library to check
	 * that the model specific data structure is decent. You must set it manually
	 * otherwise the model specific feature won't work.
	 */
	ita_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
	evt.pfp_model       = &ita_param;

	/*
	 * Before calling pfm_find_dispatch(), we must specify what kind
	 * of branches we want to capture. We are interesteed in all the mispredicted branches, 
	 * therefore we program we set the various fields of the BTB config to:
	 */
	ita_param.pfp_ita2_btb.btb_used = 1;

	ita_param.pfp_ita2_btb.btb_ds  = 0;
	ita_param.pfp_ita2_btb.btb_tm  = 0x3;
	ita_param.pfp_ita2_btb.btb_ptm = 0x3;
	ita_param.pfp_ita2_btb.btb_ppm = 0x3;
	ita_param.pfp_ita2_btb.btb_brt = 0x0;
	ita_param.pfp_ita2_btb.btb_plm = PFM_PLM3;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_event_byname("BRANCH_EVENT", &evt.pfp_events[0].event) != PFMLIB_SUCCESS) {
		fatal_error("cannot find event BRANCH_EVENT\n");
	}

	/*
	 * set the (global) privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	evt.pfp_dfl_plm   = PFM_PLM3; 
	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * for this example, we will get notified ONLY when the sampling
	 * buffer is full. The monitoring is not to be inherited
	 * in derived tasks
	 */
	ctx[0].ctx_flags        = PFM_FL_INHERIT_NONE;
	ctx[0].ctx_notify_pid   = getpid();
	ctx[0].ctx_smpl_entries = SMPL_BUF_NENTRIES;
	ctx[0].ctx_smpl_regs[0] = smpl_regs = BTB_REGS_MASK;


	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(pid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	printf("Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr);

	smpl_vaddr = ctx[0].ctx_smpl_vaddr;

	/* 
	 * Must be done before any PMD/PMD calls (unfreeze PMU). Initialize
	 * PMC/PMD to safe values. psr.up is cleared.
	 */
	if (perfmonctl(pid, PFM_ENABLE, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * indicate we want notification when buffer is full
	 */
	evt.pfp_pc[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY;

	/*
	 * Now prepare the argument to initialize the PMD and the sampling period
	 */
	pd[0].reg_num         = evt.pfp_pc[0].reg_num;
	pd[0].reg_value       = (~0UL) - SMPL_PERIOD +1;
	pd[0].reg_long_reset  = (~0UL) - SMPL_PERIOD +1;
	pd[0].reg_short_reset = (~0UL) - SMPL_PERIOD +1;

	/*
	 * When our counter overflows, we want to BTB index to be reset, so that we keep
	 * in sync. This is required to make it possible to interpret pmd16 on overflow
	 * to avoid repeating the same branch several times.
	 */
	evt.pfp_pc[0].reg_reset_pmds[0] = M_PMD(16);

	/*
	 * reset pmd16, short and long reset value are set to zero as well
	 */
	pd[1].reg_num         = 16;
	pd[1].reg_value       = 0UL;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt.pfp_pc, evt.pfp_pc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, 2) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * Let's roll now.
	 */

	do_test(100000);

	/*
	 * We must call the processing routine to cover the last entries recorded
	 * in the sampling buffer, i.e. which may not be full
	 */
	process_smpl_buffer();

	/* 
	 * let's stop this now
	 */
	if (perfmonctl(pid, PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_DESTROY errno %d\n",errno);
	}
	return 0;
}
