/*
 * self.c - example of a simple self monitoring task
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
#include <signal.h>

#include <perfmon/pfmlib.h>

static char *event_list[]={
	"cpu_cycles",
	"IA64_INST_RETIRED",
	NULL
};

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

/*
 * our test code (function cannot be made static otherwise it is optimized away)
 */
unsigned long
noploop(unsigned long loop)
{

	while (loop--) {}
	return loop;
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
	for (i=0; *p ; i++, p++) {
		if (pfm_find_event(*p, &evt.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find %s event\n", *p);
		}
	}

	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
	 */
	evt.pfp_dfl_plm   = PFM_PLM3; 

	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = i;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * for this example, we have decided not to get notified
	 * on counter overflows and the monitoring is not to be inherited
	 * in derived tasks.
	 */
	ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

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
		{int i; for(i=0; i < evt.pfp_event_count; i++) printf("pmd%d: 0x%x\n", i, pd[i].reg_flags);}
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * Let's roll now
	 */
	pfm_start();

	noploop(10000000);

	pfm_stop();

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
		fatal_error( "child: perfmonctl error PFM_DESTROY errno %d\n",errno);
	}

	return 0;
}
