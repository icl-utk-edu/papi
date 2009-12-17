/*
 * check_events.c - show event encoding
 *
 * Copyright (c) 2009 Google, Inc
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
#include <string.h>
#include <err.h>

#include <perfmon/pfmlib.h>

int
main(int argc, const char **argv)
{
	const char *arg[3];
	const char **p, *name;
	pfm_event_info_t info;
	uint64_t *codes;
	int count;
	int i, j;
	int ret, idx;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFM_SUCCESS)
		errx(1, "cannot initialize library: %s\n", pfm_strerror(ret));

	printf("Supported PMU models:\n");
	for(i=0; i < PFM_PMU_MAX; i++) {
		name = pfm_get_pmu_name(i);
		if (!name)
			continue;
		
		printf("\t[%d, %s, \"%s\"]\n", i, pfm_get_pmu_name(i), pfm_get_pmu_desc(i));
	}

	printf("Detected PMU models:\n");
	for(i=0; i < PFM_PMU_MAX; i++) {
		if (pfm_pmu_present(i))
			printf("\t[%d, %s, \"%s\"]\n", i, pfm_get_pmu_name(i), pfm_get_pmu_desc(i));
	}

	printf("Total events: %d\n", pfm_get_nevents());

	/*
	 * be nice to user!
	 */
	if (argc < 2  && pfm_pmu_present(PFM_PMU_PERF_EVENT)) {
		arg[0] = "PERF_COUNT_HW_CPU_CYCLES";
		arg[1] = "PERF_COUNT_HW_INSTRUCTIONS";
		arg[2] = NULL;
		p = arg;
	} else {
		p = argv+1;
	}

	if (!*p)
		errx(1, "you must pass at least one event");

	codes = NULL;
	count = 0;
	while(*p) {
		/*
		 * extract raw event encoding
		 *
		 * For perf_event encoding, use
		 * #include <perfmon/pfmlib_perf_event.h>
		 * and the function:
		 * pfm_get_perf_event_encoding()
		 */
		ret = pfm_get_event_encoding(*p, PFM_PLM3, NULL, &idx, &codes, &count);
		if (ret != PFM_SUCCESS) {
			/*
			 * codes is too small for this event
			 * free and let the library resize
			 */
			if (ret == PFM_ERR_TOOSMALL) {
				free(codes);
				codes = NULL;
				count = 0;
				continue;
			}
			errx(1, "cannot encode event %s: %s", *p, pfm_strerror(ret));
		}
		ret = pfm_get_event_info(idx, &info);
		if (ret != PFM_SUCCESS)
			errx(1, "cannot get event info: %s", pfm_strerror(ret));

		printf("Event %s:\n", *p);
		printf("\tPMU: %s\n", pfm_get_pmu_desc(info.pmu));
		printf("\tIDX: %d\n", idx);
		for(j=0; j < count; j++)
			printf("\tcodes[%d]=0x%"PRIx64"\n", j, codes[j]);

		p++;
	}
	if (codes)
		free(codes);
	return 0;
}
