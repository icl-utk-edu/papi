/*
 * self.c - example of a simple self monitoring task
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
	int cnt, i, ret;
	pid_t pid = getpid();
	pfmlib_param_t evt;
	pfarg_reg_t pc[NUM_PMCS];
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

	memset(pc, 0, sizeof(pc));
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
		if (pfm_find_event(*p, 0, evt.pfp_evt+i) != PFMLIB_SUCCESS) {
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
	evt.pfp_count = i;

	/*
	 * indicate how many entries can pc[] support
	 */
	cnt = NUM_PMCS;

	/*
	 * use the library to find the monitors to use
	 *
	 * upon return, cnt contains the number of entries
	 * used in pc[].
	 */
	if ((ret=pfm_dispatch_events(&evt, pc, &cnt)) != PFMLIB_SUCCESS) {
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
	if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
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
		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * Now prepare the argument to initialize the PMDs.
	 * the memset(pd) initialized the entire array to zero already, so
	 * we just have to fill in the register numbers from the pc[] array.
	 */
	for (i=0; i < evt.pfp_count; i++) {
		pd[i].reg_num = pc[i].reg_num;
	}
	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, pc, cnt) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt.pfp_count) == -1) {
		{int i; for(i=0; i < cnt; i++) printf("pmd%d: 0x%x\n", i, pd[i].reg_flags);}
		fatal_error( "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
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
	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt.pfp_count) == -1) {
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
	for (i=0; i < evt.pfp_count; i++) {
		char *name;
		pfm_get_event_name(evt.pfp_evt[i], &name);
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
