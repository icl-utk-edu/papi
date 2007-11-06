/*
 * whichpmu.c - example of how to figure out the host PMU model detected by pfmlib
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
#include <stdio.h>
#include <string.h>

#include <perfmon/pfmlib.h>

#define MAX_PMU_NAME_LEN 32
int
main(void)
{
	pfmlib_regmask_t impl_pmds;
	pfmlib_regmask_t impl_pmcs;
	pfmlib_regmask_t impl_counters;
	unsigned int num_pmds, num_pmcs, num_counters, num_events;
	unsigned int width = 0;
	unsigned int i;
	char model[MAX_PMU_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		return 1;
	}
	memset(&impl_pmcs, 0, sizeof(impl_pmcs));
	memset(&impl_pmds, 0, sizeof(impl_pmds));
	memset(&impl_counters, 0, sizeof(impl_counters));

	/*
	 * Now simply print the CPU model detected by pfmlib
	 *
	 * When the CPU model is not directly supported AND the generic support
	 * is compiled into the library, the detected will yield "Generic" which
	 * mean that only the architected features will be supported.
	 *
	 * This call can be used to tune applications based on the detected host
	 * CPU model. This is useful because some features are CPU model specific,
	 * such as address range restriction which is an Itanium feature.
	 *
	 */
	pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
	pfm_get_hw_counter_width(&width);
	pfm_get_impl_pmds(&impl_pmds);
	pfm_get_impl_pmcs(&impl_pmcs);
	pfm_get_impl_counters(&impl_counters);
	pfm_get_num_events(&num_events);
	pfm_get_num_pmds(&num_pmds);
	pfm_get_num_pmcs(&num_pmcs);
	pfm_get_num_counters(&num_counters);

	printf("PMU model detected by pfmlib: %s\n", model);

	printf("]\nnumber of PMD registers     : %u\n", num_pmds);
	printf("implemented PMD registers   : [ ");
	for (i=0; num_pmds; i++) {
		if (pfm_regmask_isset(&impl_pmds, i) == 0) continue;
		printf("%-3u", i);
		num_pmds--;
	}

	printf("]\nnumber of PMC registers     : %u\n", num_pmcs);
	printf("implemented PMC registers   : [ ");
	for (i=0; num_pmcs; i++) {
		if (pfm_regmask_isset(&impl_pmcs, i) == 0) continue;
		printf("%-3u", i);
		num_pmcs--;
	}

	printf("]\nnumber of counters          : %u\n", num_counters);
	printf("implemented counters        : [ ");
	for (i=0; num_counters; i++) {
		if (pfm_regmask_isset(&impl_counters, i) == 0) continue;
		printf("%-3u", i);
		num_counters--;
	}
	printf("]\nhardware counter width      : %u\n", width);
	printf("number of events supported  : %u\n", num_events);
	return 0;
}
