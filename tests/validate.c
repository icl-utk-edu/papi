/*
 * validate.c - validate event tables + encodings
 *
 * Copyright (c) 2010 Google, Inc
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
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <err.h>

#include <perfmon/pfmlib.h>

#define __weak_func	__attribute__((weak))

__weak_func int validate_arch(FILE *fp)
{
	return 0;
}

static struct {
	int valid_intern;
	int valid_arch;
} options;

static void
usage(void)
{
	printf("validate [-c] [-a] [-A]\n"
		"-c\trun the library validate events\n"
		"-a\trun architecture specific event tests\n"
		"-A\trun all tests\n"
		"-h\tget help\n");
}

static int
validate_event_tables(void)
{
	pfm_pmu_info_t pinfo;
	int i, ret, retval = 0;

	memset(&pinfo, 0, sizeof(pinfo));

	pinfo.size = sizeof(pinfo);

	pfm_for_all_pmus(i) {
		ret = pfm_get_pmu_info(i, &pinfo);
		if (ret != PFM_SUCCESS)
			continue;

		fprintf(stderr, "Checking %s:\n", pinfo.name);
		ret = pfm_pmu_validate(i, stderr);
		if (ret != PFM_SUCCESS && ret != PFM_ERR_NOTSUPP)
			retval = 1;
	}
	return retval;
}

int
main(int argc, char **argv)
{
	int ret, c;
	int retval1 = 0;
	int retval2 = 0;


	while ((c=getopt(argc, argv,"hcaA")) != -1) {
		switch(c) {
			case 'c':
				options.valid_intern = 1;
				break;
			case 'a':
				options.valid_arch = 1;
				break;
			case 'A':
				options.valid_arch = 1;
				options.valid_intern = 1;
				break;
			case 'h':
				usage();
				exit(0);
			default:
				errx(1, "unknown option error");
		}
	}
	/* to allow encoding of events from non detected PMU models */
	setenv("LIBPFM_ENCODE_INACTIVE", "1", 1);

	ret = pfm_initialize();
	if (ret != PFM_SUCCESS)
		errx(1, "cannot initialize libpfm: %s", pfm_strerror(ret));

	/* run everything by default */
	if (!(options.valid_intern || options.valid_arch)) {
		options.valid_intern = 1;
		options.valid_arch = 1;
	}

	if (options.valid_intern) {
		printf("Libpfm internal table tests:"); fflush(stdout);
		retval1 = validate_event_tables();
		if (retval1)
			printf(" Failed (%d errors)\n", retval1);
		else
			printf(" Passed\n");
	}

	if (options.valid_arch) {
		printf("Architecture specific tests:"); fflush(stdout);
		retval2 = validate_arch(stderr);
		if (retval2)
			printf(" Failed (%d errors)\n", retval2);
		else
			printf(" Passed\n");
	}

	pfm_terminate();

	return retval1 || retval2;
}
