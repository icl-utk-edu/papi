/*
 * showevtinfo.c - show event information
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <regex.h>

#include <perfmon/pfmlib.h>

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static size_t max_len;

static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

static void
show_event_info(char *name, unsigned int idx)
{
	pfmlib_regmask_t cnt, impl_cnt;
	char *desc;
	unsigned int n1, n2, i, c;
	int code, prev_code = 0, first = 1;
	int ret;

	pfm_get_event_counters(idx, &cnt);
	pfm_get_num_counters(&n2);
	pfm_get_impl_counters(&impl_cnt);

	n1 = n2;
	printf("#-----------------------------\n"
	       "Name     : %s\n",
	       name);

	pfm_get_event_description(idx, &desc);
 	printf("Desc     : %s\n", desc);
	free(desc);

	printf("Code     :");
	for (i=0; n1; i++) {
		if (pfm_regmask_isset(&impl_cnt, i))
			n1--;
		if (pfm_regmask_isset(&cnt, i)) {
		    pfm_get_event_code_counter(idx,i,&code);
		    if (first == 1 || code != prev_code) {
			    printf(" 0x%x", code);
		    	    first = 0;
		    }
		    prev_code = code;
		}
	}
	putchar('\n');

	n1 = n2;
	printf("Counters : [ ");
	for (i=0; n1; i++) {
		if (pfm_regmask_isset(&impl_cnt, i))
			n1--;
		if (pfm_regmask_isset(&cnt, i))
			printf("%d ", i);
	}
	puts("]");
	pfm_get_num_event_masks(idx, &n1);
	for (i = 0; i < n1; i++) {
		ret = pfm_get_event_mask_name(idx, i, name, max_len+1);
		if (ret != PFMLIB_SUCCESS)
			continue;
		pfm_get_event_mask_description(idx, i, &desc);
		pfm_get_event_mask_code(idx, i, &c);
		printf("Umask-%02u : 0x%02x : [%s] : %s\n", i, c, name, desc);
		free(desc);
	}
}

#define MAX_PMU_NAME_LEN 32
int
main(int argc, char **argv)
{
	unsigned int i, count, match;
	int ret;
	char *name;
	regex_t preg;
	char model[MAX_PMU_NAME_LEN];

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("PMU model not supported by library\n");

	pfm_get_max_event_name_len(&max_len);
	name = malloc(max_len+1);
	if (name == NULL)
		fatal_error("cannot allocate name buffer\n");

	pfm_get_num_events(&count);

	if (argc == 1)
		*argv = ".*"; /* match everything */
	else
		++argv;

	pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
	printf("PMU model: %s\n", model);
	while(*argv) {
		if (regcomp(&preg, *argv, REG_ICASE|REG_NOSUB))
			fatal_error("error in regular expression for event \"%s\"\n", *argv);

		match = 0;

		for(i=0; i < count; i++) {
			ret = pfm_get_event_name(i, name, max_len+1);
			/* skip unsupported events */
			if (ret != PFMLIB_SUCCESS)
				continue;

			if (regexec(&preg, name, 0, NULL, 0) == 0) {
				show_event_info(name, i);
				match++;
			}
		}
		if (match == 0)
			fatal_error("event %s not found\n", *argv);

		argv++;
	}
	free(name);
	return 0;
}
