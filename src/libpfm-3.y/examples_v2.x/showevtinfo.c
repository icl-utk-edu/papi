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

static struct {
	int sort;
	uint64_t mask;
} options;

typedef struct {
	uint64_t code;
	int idx;
} code_info_t;

static char *name;

static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

int compare_codes(const void *a, const void *b)
{
	const code_info_t *aa = a;
	const code_info_t *bb = b;
	uint64_t m = options.mask;

	if ((aa->code & m) < (bb->code &m))
		return -1;
	if ((aa->code & m) == (bb->code & m))
		return 0;
	return 1;
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

static int
show_info(regex_t *preg)
{
	unsigned int i, count = 0, match = 0;
	int ret;

	pfm_get_num_events(&count);

	for(i=0; i < count; i++) {
		ret = pfm_get_event_name(i, name, max_len+1);
		/* skip unsupported events */
		if (ret != PFMLIB_SUCCESS)
			continue;

		if (regexec(preg, name, 0, NULL, 0) == 0) {
			show_event_info(name, i);
			match++;
		}
	}
	return match;
}

static int
show_info_sorted(regex_t *preg)
{
	unsigned int i, n, count = 0, match = 0;
	int code, ret;
	code_info_t *codes = NULL;

	pfm_get_num_events(&count);

	codes = malloc(count * sizeof(*codes));
	if (!codes)
		fatal_error("cannot allocate memory\n");

	for(i=0, n = 0; i < count; i++, n++) {
		ret = pfm_get_event_code(i, &code);
		/* skip unsupported events */
		if (ret != PFMLIB_SUCCESS)
			continue;

		codes[n].idx = i;
		codes[n].code = code;
	}

	qsort(codes, n, sizeof(*codes), compare_codes);

	for(i=0; i < n; i++) {
		ret = pfm_get_event_name(codes[i].idx, name, max_len+1);
		/* skip unsupported events */
		if (ret != PFMLIB_SUCCESS)
			continue;

		if (regexec(preg, name, 0, NULL, 0) == 0) {
			show_event_info(name, codes[i].idx);
			match++;
		}
	}
	free(codes);
	return match;
}

static void
usage(void)
{
	printf("showevtinfo [-h] [-s] [-m mask]\n"
		"-L\t\tlist one event per line\n"
		"-h\t\tget help\n"
		"-s\t\tsort event by PMU and by code based on -m mask\n"
		"-m mask\t\thexadecimal event code mask, bits to match when sorting\n");
}



#define MAX_PMU_NAME_LEN 32
int
main(int argc, char **argv)
{
	static char *argv_all[2] = { ".*", NULL };
	char *endptr = NULL;
	char **args;
	int c, match;
	regex_t preg;
	char model[MAX_PMU_NAME_LEN];

	while ((c=getopt(argc, argv,"hsm:")) != -1) {
		switch(c) {
			case 's':
				options.sort = 1;
				break;
			case 'm':
				options.mask = strtoull(optarg, &endptr, 16);
				if (*endptr)
					fatal_error("mask must be in hexadecimal\n");
				break;
			case 'h':
				usage();
				exit(0);
			default:
				fatal_error("unknown error");
		}
	}

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("PMU model not supported by library\n");

	if (options.mask == 0)
		options.mask = ~0;

	if (optind == argc) {
		args = argv_all;
	} else {
		args = argv + optind;
	}

	pfm_get_max_event_name_len(&max_len);
	name = malloc(max_len+1);
	if (name == NULL)
		fatal_error("cannot allocate name buffer\n");

	if (argc == 1)
		*argv = ".*"; /* match everything */
	else
		++argv;

	pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
	printf("PMU model: %s\n", model);

	while(*args) {
		if (regcomp(&preg, *args, REG_ICASE|REG_NOSUB))
			fatal_error("error in regular expression for event \"%s\"", *argv);

		if (options.sort)
			match = show_info_sorted(&preg);
		else
			match = show_info(&preg);

		if (match == 0)
			fatal_error("event %s not found", *args);

		args++;
	}

	regfree(&preg);
	free(name);

	return 0;
}
