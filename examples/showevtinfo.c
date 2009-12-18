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
show_event_info_compact(pfm_event_info_t *info)
{
	pfm_event_attr_info_t ainfo;
	pfm_pmu_info_t pinfo;
	int i, ret;

	memset(&ainfo, 0, sizeof(ainfo));
	memset(&pinfo, 0, sizeof(pinfo));

	pfm_get_pmu_info(info->pmu, &pinfo);

	pfm_for_each_event_attr(i, info) {
		ret = pfm_get_event_attr_info(info->idx, i, &ainfo);
		if (ret != PFM_SUCCESS)
			err(1, "cannot get attribute info: %s", pfm_strerror(ret));

		printf("%s::%s:%s\n", pinfo.name, info->name, ainfo.name);
	}
}

static void
show_event_info(pfm_event_info_t *info)
{
	pfm_event_attr_info_t ainfo;
	pfm_pmu_info_t pinfo;
	int mod = 0, um = 0;
	int i, ret;

	memset(&ainfo, 0, sizeof(ainfo));
	memset(&pinfo, 0, sizeof(pinfo));

	ret = pfm_get_pmu_info(info->pmu, &pinfo);
	if (ret)
		errx(1, "cannot get pmu info: %s", pfm_strerror(ret));

	printf("#-----------------------------\n"
	       "IDX	 : %d\n"
	       "PMU name : %s (%s)\n"
	       "Name     : %s\n",
		info->idx,
		pinfo.name,
		pinfo.desc,
		info->name);

	printf("Desc     : %s\n", info->desc ? info->desc : "no description available");
	printf("Code     : 0x%"PRIx64"\n", info->code);

	pfm_for_each_event_attr(i, info) {
		ret = pfm_get_event_attr_info(info->idx, i, &ainfo);
		if (ret != PFM_SUCCESS)
			errx(1, "cannot retrieve event %s attribute info: %s\n", info->name, pfm_strerror(ret));

		switch(ainfo.type) {
		case PFM_ATTR_UMASK:
			printf("Umask-%02u : 0x%02"PRIx64" : [%s] : %s%s\n",
				um,
				ainfo.code,
				ainfo.name,
				ainfo.desc,
				ainfo.is_dfl ? " (DEFAULT)" : "");
			um++;
			break;
		case PFM_ATTR_MOD_BOOL:
			printf("Modif-%02u : 0x%02"PRIx64" : [%s] : %s (boolean)\n", mod, ainfo.code, ainfo.name, ainfo.desc);
			mod++;
			break;
		case PFM_ATTR_MOD_INTEGER:
			printf("Modif-%02u : 0x%02"PRIx64" : [%s] : %s (integer)\n", mod, ainfo.code, ainfo.name, ainfo.desc);
			mod++;
			break;
		default:
			printf("Attr-%02u  : 0x%02"PRIx64" : [%s] : %s\n", i, ainfo.code, ainfo.name, ainfo.desc);
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
	pfm_pmu_info_t pinfo;
	int i, ret, retval = 0;

	memset(&pinfo, 0, sizeof(pinfo));

	for(i=0; i < PFM_PMU_MAX; i++) {
		ret = pfm_get_pmu_info(i, &pinfo);
		if (ret != PFM_SUCCESS)
			continue;

		printf("Checking %s: ", pinfo.name); fflush(stdout);
		ret = pfm_pmu_validate_events(i, stdout);
		if (ret != PFM_SUCCESS && ret != PFM_ERR_NOTSUPP)
			retval = 1;
		printf("%s\n", ret == PFM_SUCCESS ? "OK" : pfm_strerror(ret));
	}
	return retval;
}


int
main(int argc, char **argv)
{
	pfm_pmu_info_t pinfo;
	pfm_event_info_t info;
	int i, match;
	regex_t preg;
	char *fullname = NULL;
	size_t len, l = 0;
	int ret, c, validate = 0;

	memset(&info, 0, sizeof(info));
	memset(&pinfo, 0, sizeof(pinfo));

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
			ret = pfm_get_pmu_info(i, &pinfo);
			if (ret != PFM_SUCCESS)
				continue;

			printf("\t[%d, %s, \"%s\"]\n", i, pinfo.name,  pinfo.desc);
		}

		printf("Detected PMU models:\n");
		for(i=0; i < PFM_PMU_MAX; i++) {
			ret = pfm_get_pmu_info(i, &pinfo);
			if (ret != PFM_SUCCESS)
				continue;

			if (pinfo.is_present)
				printf("\t[%d, %s, \"%s\", %d events]\n", i, pinfo.name, pinfo.desc, pinfo.nevents);
		}
		printf("Total events: %d\n", pfm_get_nevents());
	}

	while(*argv) {
		if (regcomp(&preg, *argv, REG_ICASE|REG_NOSUB))
			errx(1, "error in regular expression for event \"%s\"", *argv);

		match = 0;

		pfm_for_each_event(i) {

			ret = pfm_get_event_info(i, &info);
			if (ret != PFM_SUCCESS)
				errx(1, "cannot get event info: %s", pfm_strerror(ret));

			ret = pfm_get_pmu_info(info.pmu, &pinfo);
			if (ret != PFM_SUCCESS)
				errx(1, "cannot get PMU name: %s", pfm_strerror(ret));

			len = strlen(info.name) + strlen(pinfo.name) + 1 + 2;
			if (len > l) {
				l = len;
				fullname = realloc(fullname, l);
				if (!fullname)
					err(1, "cannot allocate memory");
			}
			sprintf(fullname, "%s::%s", pinfo.name, info.name);

			if (regexec(&preg, fullname, 0, NULL, 0) == 0) {
				if (options.compact)
					show_event_info_compact(&info);
				else
					show_event_info(&info);
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
