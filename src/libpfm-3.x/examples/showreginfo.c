/*
 * showreginfo.c - show PMU register information
 *
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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
#include <fcntl.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

#include <perfmon/perfmon.h>

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

static int
get_value(char *fn, char *buffer, size_t maxlen)
{
	int fd;
	ssize_t ret;

	fd = open(fn, O_RDONLY);
	if (fd == -1)
		return -1;

	ret = read(fd, buffer, maxlen-1);
	if (ret == -1)
		fatal_error("cannot read from %s\n", fn);
	buffer[ret-1] = '\0';
	close(fd);
	return 0;
}

/*
 * This example shows how to retrieve the PMU register mapping information.
 * It does not use the libpfm library. 
 * The mapping gives the translation between the logical register names,
 * as exposed by the perfmon interface, and the actual hardware registers.
 * Depending on the PMU and perfmon implementation, not all registers are
 * necessarily PMU registers, some may correspond to software resources.
 */
int
main(int argc, char **argv)
{
	unsigned long long dfl, rsvd;
	unsigned long hw_addr;
	char pname[64];
	char name[64], buffer[32];
	unsigned int i, num_pmcs = 0, num_pmds = 0;
	int ret;

	ret = get_value("/sys/kernel/perfmon/pmu_desc/model", buffer, sizeof(buffer));
	if (ret == -1)
		fatal_error("invalid or missing perfmon support for your CPU (need at least v2.3)\n");

	printf("model  : %s\n", buffer);
	puts(  "-------------------------------------------------------------------------------------\n"
	       "name   |   default  value   |   reserved  mask   | hw address or index | description\n"
	       "-------+--------------------+--------------------+---------------------+-------------");

	for(i=0; i < PFM_MAX_PMCS; i++) {

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmc%d/name", i);
		ret = get_value(pname, name, sizeof(name));
		if (ret)
			continue;

		num_pmcs++;

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmc%d/dfl_val", i);
		get_value(pname, buffer, sizeof(buffer));
		dfl = strtoull(buffer, NULL, 16);

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmc%d/rsvd_msk", i);
		get_value(pname, buffer, sizeof(buffer));
		rsvd = strtoull(buffer, NULL, 16);

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmc%d/addr", i);
		get_value(pname, buffer, sizeof(buffer));
		hw_addr = strtoul(buffer, NULL, 0);

		printf("pmc%-3d | 0x%016llx | 0x%016llx | 0x%-17lx | %s\n",
			i,
			dfl,
			rsvd,
			hw_addr,
			name);

	}
	puts("-------+--------------------+--------------------+---------------------+-------------");

	for(i=0; i < PFM_MAX_PMDS; i++) {
		
		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmd%d/name", i);
		ret = get_value(pname, name, sizeof(name));
		if (ret)
			continue;

		num_pmds++;
		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmd%d/dfl_val", i);
		get_value(pname, buffer, sizeof(buffer));
		dfl = strtoull(buffer, NULL, 16);

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmd%d/rsvd_msk", i);
		get_value(pname, buffer, sizeof(buffer));
		rsvd = strtoull(buffer, NULL, 16);

		sprintf(pname, "/sys/kernel/perfmon/pmu_desc/pmd%d/addr", i);
		get_value(pname, buffer, sizeof(buffer));
		hw_addr = strtoul(buffer, NULL, 0);

		printf("pmd%-3d | 0x%016llx | 0x%016llx | 0x%-17lx | %s\n",
			i,
			dfl,
			rsvd,
			hw_addr,
			name);
	}
	puts("-------------------------------------------------------------------------------------");
	printf("%u PMC registers, %u PMD registers\n", num_pmcs, num_pmds);
	return 0;
}
