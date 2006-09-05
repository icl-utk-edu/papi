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
	int fd;
	unsigned int num_pmcs;
	unsigned int num_pmds;
	char *lname, *p, *s, *buffer, *ptr;
	unsigned long long def, reset;
	size_t pgsz;
	ssize_t n;

	num_pmcs = num_pmds = 0;

	fd = open("/sys/kernel/perfmon/pmu_desc/mappings", O_RDONLY);
	if (fd == -1)
		fatal_error("invalid or missing perfmon support for your CPU (need at least v2.2)\n");

	pgsz = getpagesize();

	buffer = ptr = calloc(1, pgsz);
	if (buffer == NULL)
		fatal_error("cannot allocate read buffer\n");

	/*
	 * sysfs file cannot exceed the size of a page.
	 */
	n =read(fd, buffer, pgsz);

	close(fd);

	if (n < 1)
		fatal_error("cannot read PMU mappings\n");

	puts( "--------------------------------------------------------------\n"
	       "name   |   default  value   |   reserved  mask(*)| description\n"
	       "-------+--------------------+--------------------+------------");

	for(;;) {
		lname = ptr;
		p = strchr(lname, ':');
		if (p == NULL) goto error;
		*p++ = '\0';

		s = p;
		p = strchr(s,':');
		if (p == NULL) goto error;

		*p++ = '\0';
		def = strtoull(s, NULL, 0);

		s = p;
		p = strchr(s,':');
		if (p == NULL) goto error;
		*p++ = '\0';
		reset = strtoull(s, NULL, 0);


		if (lname[2] == 'C') 
			num_pmcs++;
		else
			num_pmds++;

	       if (num_pmds == 1)
		       puts("-------+--------------------+--------------------+------------");

		/*
		 * for the perfmon subsystem, the reserved bits are zero
		 * in the reserved mask. To make it more natural to users
		 * we inverse the mask, i.e., reserved bits are 1
		 */
		ptr = strchr(p, '\n');
		*ptr++ = '\0';

		printf("%-6s | 0x%016llx | 0x%016llx | %s\n",
			lname,
			def, ~reset, p);

		if (*ptr != 'P')
			break;
	}
	free(buffer);

	printf("(*) reserved mask: when a bit is set, it means the bit is reserved in the register\n");
	if (num_pmds == 0 && num_pmcs == 0)
		printf("No PMU description is installed\n");
	else 
		printf("%u PMC registers, %u PMD registers\n", num_pmcs, num_pmds);

	return 0;
error:
	fatal_error("invalid format in /proc/perfmon_map\n");
}
