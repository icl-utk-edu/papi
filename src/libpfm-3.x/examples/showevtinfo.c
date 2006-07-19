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
show_event_info(unsigned int idx)
{
	pfmlib_regmask_t cnt, impl_cnt;
	char *desc;
	unsigned int n, i;
	int code;
	size_t len;
	char *name;

	/*
	 * figure out maximum size for event and umask names
	 */
	pfm_get_max_event_name_len(&len);
	name = malloc(len+1);
	if (name == NULL)
		fatal_error("cannot allocate name buffer\n");

	pfm_get_event_name(idx, name, len);
	pfm_get_event_code(idx, &code);
	pfm_get_event_counters(idx, &cnt);
	pfm_get_num_counters(&n);
	pfm_get_impl_counters(&impl_cnt);

	printf("Name     : %s\n"
			"Code     : 0x%x\n"
			"Counters : [ "
			,
			name,
			code);

	for (i=0; n; i++) {
		if (pfm_regmask_isset(&impl_cnt, i))
			n--;
		if (pfm_regmask_isset(&cnt, i))
			printf("%d ", i);
	}
	puts("]");

	pfm_get_num_event_masks(idx, &n);
	printf("Umasks   :\n");
	for (i = 0; n; n--) {
		pfm_get_event_mask_description(idx, i, &desc);
		pfm_get_event_mask_name(idx, i, name, len);

		printf("\t [%s] : %s\n", name, desc);

		free(desc);
	}
	pfm_get_event_description(idx, &desc);
 	printf("Desc     : %s\n", desc);

	free(desc);
}

int
main(int argc, char **argv)
{
	unsigned int i, count, match;
	size_t len;
	char *name;
	regex_t preg;

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("PMU model not supported by library\n");

	pfm_get_max_event_name_len(&len);
	name = malloc(len+1);
	if (name == NULL)
		fatal_error("cannot allocate name buffer\n");


	pfm_get_num_events(&count);


	if (argc == 1)
		*argv = ".*"; /* match everything */
	else
		++argv;

	while(*argv) {
		if (regcomp(&preg, *argv, REG_ICASE|REG_NOSUB))
			fatal_error("error in regular expression for event \"%s\"\n", *argv);

		match = 0;

		for(i=0; i < count; i++) {
			pfm_get_event_name(i, name, len);
			if (regexec(&preg, name, 0, NULL, 0) == 0) {
				show_event_info(i);
				match++;
			}
		}
		if (match == 0)
			fatal_error("event %s not found\n", *argv);

		argv++;
	}
	return 0;
}
