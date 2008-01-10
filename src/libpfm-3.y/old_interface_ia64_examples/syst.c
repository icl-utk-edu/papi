/*
 * syst.c - example of a simple system wide monitoring program
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <syscall.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128

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

#ifndef __NR_sched_setaffinity
#ifdef __ia64__
#define __NR_sched_setaffinity 1231
#endif
#endif
/*
 * Hack to get this to work without libc support
 */
int
my_setaffinity(pid_t pid, unsigned int len, unsigned long *mask)
{
	return syscall(__NR_sched_setaffinity, pid, len, mask);
}


int
main(int argc, char **argv)
{
	char **p;
	unsigned long my_mask;
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_context_t ctx[1];
	pfarg_load_t load_args;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_options_t pfmlib_options;
	unsigned int which_cpu;
	int ret, ctx_fd;
	unsigned int i;
	unsigned int num_counters;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}
	pfm_get_num_counters(&num_counters);
	
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);


	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	/*
	 * be nice to user!
	 */
	if (argc > 1) {
		p = argv+1;
		for (i=0; *p ; i++, p++) {
			if (pfm_find_event(*p, &inp.pfp_events[i].event) != PFMLIB_SUCCESS) {
				fatal_error("Cannot find %s event\n", *p);
			}
		}
	} else {
		if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS) {
			fatal_error("cannot find cycle event\n");
		}
		if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS) {
			fatal_error("cannot find inst retired event\n");
		}
		i = 2;
	}
	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user   level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3|PFM_PLM0;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * indicate we are using the monitors for a system-wide session.
	 * This may impact the way the library sets up the PMC values.
	 */
	inp.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * In system wide mode, the perfmon context cannot be inherited.
	 * Also in this mode, we cannot use the blocking form of user level notification.
	 */
	ctx[0].ctx_flags = PFM_FL_SYSTEM_WIDE;

	/*
	 * pick a random CPU. Assumes CPU are numbered with no holes
	 */
	srandom(getpid());

	which_cpu = random() % sysconf(_SC_NPROCESSORS_ONLN);

	/*
	 * perfmon relies on the application to have the task pinned
	 * on one CPU by the time the PFM_CONTEXT_LOAD command is issued.
	 * The perfmon context will record the active CPU at the time of PFM_CONTEXT_LOAD
	 * and will reject any access coming from another CPU. Therefore it
	 * is advisable to pin the task ASAP before doing any perfmon calls.
	 *
	 * On RHAS and 2.5/2.6, this can be easily achieved using the
	 * sched_setaffinity() system call.
	 */
	my_mask = 1UL << which_cpu;

	ret = my_setaffinity(getpid(), sizeof(unsigned long), &my_mask);
	if (ret == -1) {
		fatal_error("cannot set affinity to 0x%lx: %s\n", my_mask, strerror(errno));
	}
	/*
	 * after the call the task is pinned to which_cpu
	 */

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/*
	 * extact our file descriptor
	 */
	ctx_fd = ctx->ctx_fd;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We must pfp_pmc_count to determine the number of PMC to intialize.
	 * We must use pfp_event_count to determine the number of PMD to initialize.
	 * Some events causes extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
	 *
	 * This step is new compared to libpfm-2.x. It is necessary because the library no
	 * longer knows about the kernel data structures.
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
	 * the PMC controlling the event ALWAYS come first, that's why this loop
	 * is safe even when extra PMC are needed to support a particular event.
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pd[i].reg_num = outp.pfp_pmcs[i].reg_num;
	}

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(ctx_fd, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(ctx_fd, PFM_WRITE_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * for system wide session, we can only attached to ourself
	 */
	load_args.load_pid = getpid();

	if (perfmonctl(ctx_fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
		fatal_error("perfmonctl error PFM_LOAD_CONTEXT errno %d\n",errno);
	}

	/*
	 * start monitoring. We must go to the kernel because psr.pp cannot be
	 * changed at the user level.
	 */
	if (perfmonctl(ctx_fd, PFM_START, 0, 0) == -1) {
		fatal_error("perfmonctl error PFM_START errno %d\n",errno);
	}
	printf("<monitoring started on CPU%d>\n", which_cpu);

	printf("<press a key to stop monitoring>\n");
	getchar();

	/*
	 * stop monitoring. We must go to the kernel because psr.pp cannot be
	 * changed at the user level.
	 */
	if (perfmonctl(ctx_fd, PFM_STOP, 0, 0) == -1) {
		fatal_error("perfmonctl error PFM_STOP errno %d\n",errno);
	}

	printf("<monitoring stopped on CPU%d>\n\n", which_cpu);

	/*
	 * now read the results
	 */
	if (perfmonctl(ctx_fd, PFM_READ_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error( "perfmonctl error READ_PMDS errno %d\n",errno);
		return -1;
	}

	/*
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 *
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);
		printf("CPU%-2d PMD%u %20"PRIu64" %s\n",
			which_cpu,
			pd[i].reg_num,
			pd[i].reg_value,
			name);
	}

	/*
	 * let's stop this now
	 */
	close(ctx_fd);

	return 0;
}
