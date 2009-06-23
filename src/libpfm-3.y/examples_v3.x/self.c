/*
 * self.c - example of a simple self monitoring task
 *
 * Copyright (c) 2002-2007 Hewlett-Packard Development Company, L.P.
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
 * applications on Linux.
 */

#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static volatile int quit;
void sig_handler(int n)
{
	quit = 1;
}

/*
 * our test code (function cannot be made static otherwise it is optimized away)
 */
void
noploop(void)
{
	for(;quit == 0;);
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
	unsigned int i;
	int ret, ctx_fd;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmr_t pd[NUM_PMDS];
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_sinfo_t sif;
	pfmlib_options_t pfmlib_options;
	unsigned int num_counters;
	size_t len;
	char *name;

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	pfm_get_max_event_name_len(&len);
	name = malloc(len+1);
	if (!name)
		fatal_error("cannot allocate event name buffer\n");

	pfm_get_num_counters(&num_counters);

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&sif, 0, sizeof(sif));

	/*
	 * prepare parameters to library.
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	/*
	 * be nice to user!
	 */
	if (argc > 1) {
		p = argv+1;
		for (i=0; *p ; i++, p++) {
			ret = pfm_find_full_event(*p, &inp.pfp_events[i]);
			if (ret != PFMLIB_SUCCESS)
				fatal_error("event %s: %s\n", *p, pfm_strerror(ret));
		}
	} else {
		if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
			fatal_error("cannot find cycle event\n");

		if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
			fatal_error("cannot find inst retired event\n");
		i = 2;
	}

	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * now create a new per-thread session
	 * This just creates a new session with some initial state, it is not
	 * active nor attached to any thread yet.
	 */
	ctx_fd = pfm_create(0, &sif);
	if (ctx_fd == -1)  {
		if (errno == ENOSYS)
			fatal_error("Your kernel does not have performance monitoring support!\n");
		fatal_error("cannot create session %s\n", strerror(errno));
	}

	/*
	 * build the pfp_unavail_pmcs bitmask by looking
	 * at what perfmon has available. It is not always
	 * the case that all PMU registers are actually available
	 * to applications. For instance, on IA-32 platforms, some
	 * registers may be reserved for the NMI watchdog timer.
	 *
	 * With this bitmap, the library knows which registers NOT to
	 * use. Of source, it is possible that no valid assignement may
	 * be possible if certain PMU registers  are not available.
	 */
	detect_unavail_pmu_regs(&sif, &inp.pfp_unavail_pmcs, NULL);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of PMC to intialize.
	 * We use pfp_pmd_count to determine the number of PMD to initialize.
	 * Some events/features may cause extra PMCs to be used, leading to:
	 * 	- pfp_pmc_count may be >= pfp_event_count
	 * 	- pfp_pmd_count may be >= pfp_event_count
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	for (i=0; i < outp.pfp_pmd_count; i++) {
		pd[i].reg_num   = outp.pfp_pmds[i].reg_num;
	}
	/*
	 * Now program the registers
	 */
	if (pfm_write(ctx_fd, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)))
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(ctx_fd, 0, PFM_RW_PMD, pd, outp.pfp_pmd_count * sizeof(*pd)))
		fatal_error("pfm_write(PMD) error errno %d\n",errno);
	/*
	 * now we attach the session to ourself
	 */
	if (pfm_attach(ctx_fd, 0, getpid()))
		fatal_error("pfm_attacherror errno %d\n",errno);

	/*
	 * Let's roll now
	 */
	if (pfm_set_state(ctx_fd, 0, PFM_ST_START))
		fatal_error("pfm_set_state(start) error errno %d\n",errno);

	signal(SIGALRM, sig_handler);
	alarm(10);
	noploop();

	if (pfm_set_state(ctx_fd, 0, PFM_ST_STOP))
		fatal_error("pfm_set_state(stop) error errno %d\n",errno);

	/*
	 * now read the results. We use pfp_event_count because
	 * libpfm guarantees that counters for the events always
	 * come first.
	 */
	if (pfm_read(ctx_fd, 0, PFM_RW_PMD, pd, inp.pfp_event_count * sizeof(*pd)))
		fatal_error( "pfm_read error errno %d\n",errno);

	/*
	 * print the results
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pfm_get_full_event_name(&inp.pfp_events[i], name, len+1);
		printf("PMD%-3u %20"PRIu64" %s\n",
			pd[i].reg_num,
			pd[i].reg_value,
			name);
	}
	free(name);
	/*
	 * and destroy our session
	 */
	close(ctx_fd);

	return 0;
}
