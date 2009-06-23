/*
 * check_events.c - check if event assignment is possible
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#include <inttypes.h>
#include <perfmon/pfmlib.h>

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_PMU_NAME_LEN 32

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
 * The goal of this program is to exercise the event assignment
 * code for a specific PMU model. This program is independent of
 * the kernel API.
 */
int
main(int argc, char **argv)
{
	char **p;
	unsigned int i;
	int ret;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_options_t pfmlib_options;
	char model[MAX_PMU_NAME_LEN];
	unsigned int num_counters;

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
	printf("PMU model: %s\n", model);

	pfm_get_num_counters(&num_counters);
	printf("%u counters available\n", num_counters);

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
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	for (i=0; i < outp.pfp_pmc_count; i++)
		printf("PMC%u=0x%llx\n",
			outp.pfp_pmcs[i].reg_num,
			outp.pfp_pmcs[i].reg_value);

	for (i=0; i < outp.pfp_pmd_count; i++)
		printf("PMD%u\n", outp.pfp_pmds[i].reg_num);
	return 0;
}
