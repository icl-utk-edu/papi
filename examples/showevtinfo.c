/*
 * showevtinfo.c - show event information
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
#include <regex.h>
#include <err.h>

#include <perfmon/pfmlib.h>

#define MAXBUF	1024

static struct {
	int compact;
	int sort;
	int encode;
	int combo;
	uint64_t mask;
} options;

typedef struct {
	uint64_t code;
	int idx;
} code_info_t;

static int
event_has_pname(char *s)
{
	char *p;
	return (p = strchr(s, ':')) && *(p+1) == ':';
}

static int
print_codes(char *buf, int plm)
{
	uint64_t *codes = NULL;
	int j, ret, count = 0;

	ret = pfm_get_event_encoding(buf, PFM_PLM0|PFM_PLM3, NULL, NULL, &codes, &count);
	if (ret != PFM_SUCCESS)
		return -1;

	if (count)
		printf("0x%"PRIx64, codes[0]);
	for (j=1; j < count; j++)
		printf(" 0x%"PRIx64, codes[j]);
	free(codes);
	return 0;
}

static int
check_valid(char *buf, int plm)
{
	uint64_t *codes = NULL;
	int ret, count = 0;

	ret = pfm_get_event_encoding(buf, PFM_PLM0|PFM_PLM3, NULL, NULL, &codes, &count);
	if (ret != PFM_SUCCESS)
		return -1;
	free(codes);
	return 0;
}

static void
show_event_info_combo(pfm_event_info_t *info)
{
	pfm_event_attr_info_t *ainfo;
	pfm_pmu_info_t pinfo;
	char buf[MAXBUF];
	int len, numasks = 0;
	int i, j, ret;
	uint64_t total, m;

	memset(&pinfo, 0, sizeof(pinfo));

	pinfo.size = sizeof(pinfo);

	pfm_get_pmu_info(info->pmu, &pinfo);

	ainfo = malloc(info->nattrs * sizeof(*ainfo));
	if (!ainfo)
		err(1, "event %s : ", info->name);

	/*
	 * extract attribute information and count number
	 * of umasks
	 *
	 * we cannot just drop non umasks because we need
	 * to keep attributes in order for the enumeration
	 * of 2^n
	 */
	pfm_for_each_event_attr(i, info) {
		ainfo[i].size = sizeof(*ainfo);

		ret = pfm_get_event_attr_info(info->idx, i, &ainfo[i]);
		if (ret != PFM_SUCCESS)
			err(1, "cannot get attribute info: %s", pfm_strerror(ret));

		if (ainfo[i].type == PFM_ATTR_UMASK)
			numasks++;
	}
	if (numasks) {
		if (info->nattrs > ((sizeof(total)<<3))) {
			warnx("too many umasks, cannot show all combinations for event %s", info->name);
			goto end;
		}
		total = 1ULL << info->nattrs;

		for (i = 1; i < total; i++) {
			len = sizeof(buf);
			len -= snprintf(buf, len, "%s::%s", pinfo.name, info->name);
			if (len <= 0) {
				warnx("event name too long%s", info->name);
				goto end;
			}
			for(m = i, j= 0; m; m >>=1, j++) {
				if (m & 0x1ULL) {
					/* we have hit a non umasks attribute, skip */
					if (ainfo[j].type != PFM_ATTR_UMASK)
						break;

					if (len < (1 + strlen(ainfo[j].name))) {
						warnx("umasks combination too long for event %s", buf);
						break;
					}
					strncat(buf, ":", len); len--;
					strncat(buf, ainfo[j].name, len);
					len -= strlen(ainfo[j].name);
				}
			}
			/* if found a valid umask combination, check encoding */
			if (m == 0) {
				if (options.encode)
					ret = print_codes(buf, PFM_PLM0|PFM_PLM3);
				else
					ret = check_valid(buf, PFM_PLM0|PFM_PLM3);
				if (!ret)
					printf("%s%s\n", options.encode ? "\t":"", buf);
			}
		}
	} else {
		snprintf(buf, sizeof(buf)-1, "%s::%s", pinfo.name, info->name);
		buf[sizeof(buf)-1] = '\0';

		ret = options.encode ? print_codes(buf, PFM_PLM0|PFM_PLM3) : 0;
		if (!ret)
			printf("%s%s\n", options.encode ? "\t":"", buf);
	}
end:
	free(ainfo);
}

static void
show_event_info_compact(pfm_event_info_t *info)
{
	pfm_event_attr_info_t ainfo;
	pfm_pmu_info_t pinfo;
	char buf[MAXBUF];
	int i, ret, um = 0;

	memset(&ainfo, 0, sizeof(ainfo));
	memset(&pinfo, 0, sizeof(pinfo));

	pinfo.size = sizeof(pinfo);
	ainfo.size = sizeof(ainfo);

	pfm_get_pmu_info(info->pmu, &pinfo);

	pfm_for_each_event_attr(i, info) {
		ret = pfm_get_event_attr_info(info->idx, i, &ainfo);
		if (ret != PFM_SUCCESS)
			err(1, "cannot get attribute info: %s", pfm_strerror(ret));

		if (ainfo.type != PFM_ATTR_UMASK)
			continue;

		snprintf(buf, sizeof(buf)-1, "%s::%s:%s", pinfo.name, info->name, ainfo.name);
		buf[sizeof(buf)-1] = '\0';

		ret = 0;
		if (options.encode) {
			ret = print_codes(buf, PFM_PLM0|PFM_PLM3);
			if (!ret)
				putchar('\t');
		}
		if (!ret)
			printf("%s\n", buf);
		um++;
	}
	if (um == 0) {
		snprintf(buf, sizeof(buf)-1, "%s::%s", pinfo.name, info->name);
		buf[sizeof(buf)-1] = '\0';
		if (options.encode) {
			print_codes(buf, PFM_PLM0|PFM_PLM3);
			putchar('\t');
		}
		printf("%s", buf);
		putchar('\n');
	}
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
show_event_info(pfm_event_info_t *info)
{
	pfm_event_attr_info_t ainfo;
	pfm_pmu_info_t pinfo;
	int mod = 0, um = 0;
	int i, ret;

	memset(&ainfo, 0, sizeof(ainfo));
	memset(&pinfo, 0, sizeof(pinfo));

	pinfo.size = sizeof(pinfo);
	ainfo.size = sizeof(ainfo);

	ret = pfm_get_pmu_info(info->pmu, &pinfo);
	if (ret)
		errx(1, "cannot get pmu info: %s", pfm_strerror(ret));

	printf("#-----------------------------\n"
	       "IDX	 : %d\n"
	       "PMU name : %s (%s)\n"
	       "Name     : %s\n"
	       "Equiv	 : %s\n",
		info->idx,
		pinfo.name,
		pinfo.desc,
		info->name,
		info->equiv ? info->equiv : "None");

	printf("Desc     : %s\n", info->desc ? info->desc : "no description available");
	printf("Code     : 0x%"PRIx64"\n", info->code);

	pfm_for_each_event_attr(i, info) {
		ret = pfm_get_event_attr_info(info->idx, i, &ainfo);
		if (ret != PFM_SUCCESS)
			errx(1, "cannot retrieve event %s attribute info: %s\n", info->name, pfm_strerror(ret));

		switch(ainfo.type) {
		case PFM_ATTR_UMASK:
			printf("Umask-%02u : 0x%02"PRIx64" : [%s] : ",
				um,
				ainfo.code,
				ainfo.name);

			if (ainfo.equiv)
				printf("Alias to %s", ainfo.equiv);
			else
				printf("%s", ainfo.desc);

			if (ainfo.is_dfl)
				printf(" (DEFAULT)");
				putchar('\n');
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


static int
show_info(char *event, regex_t *preg)
{
	pfm_pmu_info_t pinfo;
	pfm_event_info_t info;
	pfm_pmu_t last_pmu = PFM_PMU_NONE;
	int i, ret, match = 0, pname;
	size_t len, l = 0;
	char *fullname = NULL;

	memset(&pinfo, 0, sizeof(pinfo));
	memset(&info, 0, sizeof(info));

	pinfo.size = sizeof(pinfo);
	info.size = sizeof(info);

	pname = event_has_pname(event);

	/*
	 * scan all supported events, incl. those
	 * from undetected PMU models
	 */
	pfm_for_each_event(i) {
		ret = pfm_get_event_info(i, &info);
		if (ret != PFM_SUCCESS)
			errx(1, "cannot get event info: %s", pfm_strerror(ret));


		if (info.pmu != last_pmu) {
			ret = pfm_get_pmu_info(info.pmu, &pinfo);
			if (ret != PFM_SUCCESS)
				errx(1, "cannot get pmu info: %s", pfm_strerror(ret));

			last_pmu = pinfo.pmu;
		}

		/* no pmu prefix, just look for detected PMU models */
		if (!pname && !pinfo.is_present)
			continue;

		len = strlen(info.name) + strlen(pinfo.name) + 1 + 2;
		if (len > l) {
			l = len;
			fullname = realloc(fullname, l);
			if (!fullname)
				err(1, "cannot allocate memory");
		}
		sprintf(fullname, "%s::%s", pinfo.name, info.name);

		if (regexec(preg, fullname, 0, NULL, 0) == 0) {
			if (options.compact)
				if (options.combo)
					show_event_info_combo(&info);
				else
					show_event_info_compact(&info);
			else
				show_event_info(&info);
			match++;
		}
	}
	if (fullname)
		free(fullname);

	return match;
}

static int
show_info_sorted(char *event, regex_t *preg)
{
	pfm_pmu_info_t pinfo;
	pfm_event_info_t info;
	int i, j, ret, n, match = 0;
	size_t len, l = 0;
	char *fullname = NULL;
	code_info_t *codes;

	memset(&pinfo, 0, sizeof(pinfo));
	memset(&info, 0, sizeof(info));

	pinfo.size = sizeof(pinfo);
	info.size = sizeof(info);

	pfm_for_all_pmus(j) {

		ret = pfm_get_pmu_info(j, &pinfo);
		if (ret != PFM_SUCCESS)
			continue;

		codes = malloc(pinfo.nevents * sizeof(*codes));
		if (!codes)
			err(1, "cannot allocate memory\n");

		/* scans all supported events */
		n = 0;
		pfm_for_each_event(i) {

			ret = pfm_get_event_info(i, &info);
			if (ret != PFM_SUCCESS)
				errx(1, "cannot get event info: %s", pfm_strerror(ret));

			if (info.pmu != j)
				continue;

			codes[n].idx = info.idx;
			codes[n].code = info.code;
			n++;
		}
		qsort(codes, n, sizeof(*codes), compare_codes);
		for(i=0; i < n; i++) {
			ret = pfm_get_event_info(codes[i].idx, &info);
			if (ret != PFM_SUCCESS)
				errx(1, "cannot get event info: %s", pfm_strerror(ret));

			len = strlen(info.name) + strlen(pinfo.name) + 1 + 2;
			if (len > l) {
				l = len;
				fullname = realloc(fullname, l);
				if (!fullname)
					err(1, "cannot allocate memory");
			}
			sprintf(fullname, "%s::%s", pinfo.name, info.name);

			if (regexec(preg, fullname, 0, NULL, 0) == 0) {
				if (options.compact)
					show_event_info_compact(&info);
				else
					show_event_info(&info);
				match++;
			}
		}
		free(codes);
	}
	if (fullname)
		free(fullname);

	return match;
}

static void
usage(void)
{
	printf("showevtinfo [-L] [-E] [-h] [-s] [-C] [-m mask]\n"
		"-L\t\tlist one event per line\n"
		"-E\t\tlist one event per line with encoding\n"
		"-M\t\tdisplay all valid unit masks combination (use with -L or -E)\n"
		"-h\t\tget help\n"
		"-s\t\tsort event by PMU and by code based on -m mask\n"
		"-m mask\t\thexadecimal event code mask, bits to match when sorting\n");
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

		printf("Checking %s:\n", pinfo.name);
		ret = pfm_pmu_validate(i, stdout);
		if (ret != PFM_SUCCESS && ret != PFM_ERR_NOTSUPP)
			retval = 1;
	}
	return retval;
}

/*
 * keep: [pmu::]event
 * drop everything else
 */
static void
drop_event_attributes(char *str)
{
	char *p;

	p = strchr(str, ':');
	if (!p)
		return;

	str = p+1;
	/* keep PMU name */
	if (*str == ':')
		str++;

	/* stop string at 1st attribute */
	p = strchr(str, ':');
	if (p)
		*p = '\0';
}

int
main(int argc, char **argv)
{
	static char *argv_all[2] = { ".*", NULL };
	pfm_pmu_info_t pinfo;
	char *endptr = NULL;
	char **args;
	int i, match;
	regex_t preg;
	int ret, c, validate = 0;

	memset(&pinfo, 0, sizeof(pinfo));

	pinfo.size = sizeof(pinfo);

	while ((c=getopt(argc, argv,"hCELsm:M")) != -1) {
		switch(c) {
			case 'L':
				options.compact = 1;
				break;
			case 'E':
				options.compact = 1;
				options.encode = 1;
				break;
			case 'M':
				options.combo = 1;
				break;
			case 's':
				options.sort = 1;
				break;
			case 'C':
				validate = 1;
				break;
			case 'm':
				options.mask = strtoull(optarg, &endptr, 16);
				if (*endptr)
					errx(1, "mask must be in hexadecimal\n");
				break;
			case 'h':
				usage();
				exit(0);
			default:
				errx(1, "unknown option error");
		}
	}
	ret = pfm_initialize();
	if (ret != PFM_SUCCESS)
		errx(1, "cannot initialize libpfm: %s", pfm_strerror(ret));

	if (validate)
		exit(validate_event_tables());

	if (options.mask == 0)
		options.mask = ~0;

	if (optind == argc) {
		args = argv_all;
	} else {
		args = argv + optind;
	}

	if (!options.compact) {
		int total_events = 0;

		printf("Supported PMU models:\n");
		pfm_for_all_pmus(i) {
			ret = pfm_get_pmu_info(i, &pinfo);
			if (ret != PFM_SUCCESS)
				continue;

			printf("\t[%d, %s, \"%s\"]\n", i, pinfo.name,  pinfo.desc);
		}

		printf("Detected PMU models:\n");
		pfm_for_all_pmus(i) {
			ret = pfm_get_pmu_info(i, &pinfo);
			if (ret != PFM_SUCCESS)
				continue;

			if (pinfo.is_present) {
				printf("\t[%d, %s, \"%s\", %d events]\n", i, pinfo.name, pinfo.desc, pinfo.nevents);
				total_events += pinfo.nevents;
			}
		}
		printf("Total events: %d available, %d supported\n", pfm_get_nevents(), total_events);
	}

	while(*args) {
		/* drop umasks and modifiers */
		drop_event_attributes(*args);
		if (regcomp(&preg, *args, REG_ICASE))
			errx(1, "error in regular expression for event \"%s\"", *argv);

		if (options.sort)
			match = show_info_sorted(*args, &preg);
		else
			match = show_info(*args, &preg);

		if (match == 0)
			errx(1, "event %s not found", *args);

		args++;
	}

	regfree(&preg);

	pfm_terminate();

	return 0;
}
