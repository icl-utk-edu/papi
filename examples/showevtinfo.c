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
#include <err.h>

#include <perfmon/pfmlib.h>

static struct {
	int compact;
} options;

extern int pfm_pmu_validate_events(pfm_pmu_t pmu, FILE *fp);

static void
show_event_info_compact(const char *event_name, int idx)
{
	const char *pmu_name, *attr_name;
	pfm_pmu_t pmu;
	int i;

	pmu = pfm_get_event_pmu(idx);
	if (pmu < 0)
		errx(1, "cannot determine event PMU");

	pmu_name = pfm_get_pmu_name(pmu);

	pfm_for_each_event_attr(i, idx) {
		attr_name = pfm_get_event_attr_name(idx, i);
		printf("%s::%s:%s\n", pmu_name, event_name, attr_name);
		
	}
}

static void
show_event_info(const char *name, int idx)
{
	const char *desc;
	uint64_t code;
	pfm_pmu_t pmu;
	pfm_attr_t atype;
	int mod = 0, um = 0;
	int i, ret;

	pmu = pfm_get_event_pmu(idx);
	if (pmu < 0)
		errx(1, "cannot determine event PMU");


	printf("#-----------------------------\n"
	       "IDX	 : %d\n"
	       "PMU name : %s (%s)\n"
	       "Name     : %s\n",
		idx,
		pfm_get_pmu_name(pmu),
		pfm_get_pmu_desc(pmu),
		name);

	desc = pfm_get_event_desc(idx);
	if (!desc)
		desc = "no description available";

 	printf("Desc     : %s\n", desc);

	ret = pfm_get_event_code(idx, &code);
	if (ret == PFM_SUCCESS)
		printf("Code     : 0x%"PRIx64"\n", code);
	else
		printf("Code     : NA\n");

	pfm_for_each_event_attr(i, idx) {
		pfm_get_event_attr_code(idx, i, &code);
		name = pfm_get_event_attr_name(idx, i);
		desc = pfm_get_event_attr_desc(idx, i);
		atype = pfm_get_event_attr_type(idx, i);
		switch(atype) {
		case PFM_ATTR_UMASK:
			printf("Umask-%02u : 0x%02"PRIx64" : [%s] : %s\n", um, code, name, desc);
			um++;
			break;
		case PFM_ATTR_MOD_BOOL:
			printf("Modif-%02u : 0x%02"PRIx64" : [%s] : %s (boolean)\n", mod, code, name, desc);
			mod++;
			break;
		case PFM_ATTR_MOD_INTEGER:
			printf("Modif-%02u : 0x%02"PRIx64" : [%s] : %s (integer)\n", mod, code, name, desc);
			mod++;
			break;
		default:
			printf("Attr-%02u  : 0x%02"PRIx64" : [%s] : %s\n", i, code, name, desc);
		}
	}
}

static void
usage(void)
{
	printf("showevtinfo [-L] [-h]\n");
}

static int
validate_event_tables(void)
{
	const char *name;
	int i, ret, retval = 0;

	for(i=0; i < PFM_PMU_MAX; i++) {
		name = pfm_get_pmu_name(i);
		if (!name)
			continue;

		ret = pfm_pmu_validate_events(i, stdout);
		if (ret != PFM_SUCCESS)
			retval = 1;
		printf("Checked %s: %s\n", name, ret == PFM_SUCCESS ? "OK" : "ERROR");
	}
	return retval;
}


int
main(int argc, char **argv)
{
	const char *name, *pname;
	pfm_pmu_t pmu;
	int i, match;
	regex_t preg;
	char *fullname = NULL;
	size_t len, l = 0;
	int ret, c, validate = 0;

	while ((c=getopt(argc, argv,"hCL")) != -1) {
		switch(c) {
			case 'L':
				options.compact = 1;
				break;
			case 'C':
				validate = 1;
				break;			
			case 'h':
				usage();
				exit(0);
			default:
				errx(1, "unknown error");
		}
	}
	ret = pfm_initialize();
	if (ret != PFM_SUCCESS)
		errx(1, "cannot initialize libpfm: %s", pfm_strerror(ret));

	if (validate)
		exit(validate_event_tables());

	if (optind == argc) {
		argv[0] = ".*"; /* match everything */
		argv[1] = NULL;
	} else
		argv++;

	if (!options.compact) {
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
	}

	while(*argv) {
		if (regcomp(&preg, *argv, REG_ICASE|REG_NOSUB))
			errx(1, "error in regular expression for event \"%s\"", *argv);

		match = 0;

		pfm_for_each_event(i) {

			name = pfm_get_event_name(i);
			pmu = pfm_get_event_pmu(i);
			pname = pfm_get_pmu_name(pmu);	

			len = strlen(name) + strlen(pname) + 1 + 2;
			if (len > l) {
				l = len;
				fullname = realloc(fullname, l);
				if (!fullname)
					err(1, "cannot allocate memory");
			}
			sprintf(fullname, "%s::%s", pname, name);

			if (regexec(&preg, fullname, 0, NULL, 0) == 0) {
				if (options.compact)
					show_event_info_compact(name, i);
				else
					show_event_info(name, i);
				match++;
			}
		}
		if (match == 0)
			errx(1, "event %s not found", *argv);

		argv++;
	}

	regfree(&preg);

	if (fullname)
		free(fullname);

	return 0;
}
