/*
 * syst.c - example of a simple system wide monitoring program
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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

#include <perfmon/pfmlib.h>

#define WHICH_CPU	0

static char *event_list[]={
	"cpu_cycles",
	"IA64_INST_RETIRED",
	NULL
};

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

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
main(int argc, char **argv)
{
	char **p;
	int i, ret;
	pid_t pid = getpid();
	pfmlib_param_t evt;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_context_t ctx[1];
	pfmlib_options_t pfmlib_options;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}
	
	/* 
	 * check that the user did not specify too many events
	 */
	if (argc-1 > pfm_get_num_counters()) {
		printf("Too many events specified\n");
		exit(1);
	}


	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&evt,0, sizeof(evt));

	/*
	 * be nice to user!
	 */
	p = argc > 1 ? argv+1 : event_list;

	for (i=0; *p; i++, p++) {
		if (pfm_find_event(*p, &evt.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find %s event\n", *p);
		}
	}

	/*
	 * set the privilege mode:
	 * 	PFM_PLM0 : kernel level
	 */
	evt.pfp_dfl_plm   = PFM_PLM0; 
	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = i;

	/*
	 * for system wide monitoring, we must use privileged monitors
	 */
	evt.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * In system wide mode, the perfmon context cannot be inherited. 
	 * Also in this mode, we cannot use the blocking form of user level notification.
	 */
	ctx[0].ctx_flags = PFM_FL_INHERIT_NONE | PFM_FL_SYSTEM_WIDE;

	/*
	 * pick the CPU we will run on. System wide mode applies only to
	 * one CPU at a time. You need to run several instance on different
	 * CPU to get full coverage (see pfmon_system.c for an example).
	 * As a consequence ctx_cpu_mask must have ONLY one bit set. 
	 *
	 * Until Linux has an interface to explicitely pin a task on a CPU, we rely
	 * on perfmon to do this inside the kernel. This is accomplished by the 
	 * PFM_CREATE_CONTEXT call. When returning from this call, the thread is
	 * guaranteed to run on the specified CPU if it is online.
	 *
	 */
	ctx[0].ctx_cpu_mask = 1UL << WHICH_CPU;

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(pid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/* 
	 * Must be done before any PMD/PMD calls (unfreeze PMU). Initialize
	 * PMC/PMD to safe values. psr.up is cleared.
	 */
	if (perfmonctl(pid, PFM_ENABLE, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * Now prepare the argument to initialize the PMDs.
	 * the memset(pd) initialized the entire array to zero already, so
	 * we just have to fill in the register numbers from the pc[] array.
	 */
	for (i=0; i < evt.pfp_event_count; i++) {
		pd[i].reg_num = evt.pfp_pc[i].reg_num;
	}

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
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * start monitoring. We must go to the kernel because psr.pp cannot be
	 * changed at the user level.
	 */
	if (perfmonctl(pid, PFM_START, 0, 0) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	printf("<Press a key to stop monitoring>\n");
	getchar();

	/*
	 * stop monitoring. We must go to the kernel because psr.pp cannot be
	 * changed at the user level.
	 */
	if (perfmonctl(pid, PFM_STOP, 0, 0) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	printf("<Monitoring stopped on CPU%d>\n\n", WHICH_CPU);

	/* 
	 * now read the results
	 */
	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt.pfp_event_count) == -1) {
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
	for (i=0; i < evt.pfp_event_count; i++) {
		char *name;
		pfm_get_event_name(evt.pfp_events[i].event, &name);
		printf("PMD%u %20lu %s\n", 
			pd[i].reg_num, 
			pd[i].reg_value, 
			name);
	}

	/* 
	 * let's stop this now
	 */
	if (perfmonctl(pid, PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_DESTROY errno %d\n",errno);
	}

	return 0;
}
