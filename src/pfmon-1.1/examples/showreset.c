/*
 * showreset.c - getting the PAL reset values for the PMCs
 *
 * Copyright (C) 2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

#include <perfmon/pfmlib.h>

#define NUM_PMCS PMU_MAX_PMCS

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


int
main(int argc, char **argv)
{
	int i, cnum = 0;
	unsigned long m;
	pfarg_reg_t pc[NUM_PMCS];
	unsigned long impl_pmcs[4];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}
	memset(impl_pmcs, 0, sizeof(impl_pmcs));
	memset(pc, 0, sizeof(pc));
	
	pfm_get_impl_pmcs(impl_pmcs);

	m = impl_pmcs[0];
	for(i=0; m; i++, m>>=1) {
		if ((m & 0x1) == 0) continue;
		pc[cnum++].reg_num = i;
	}

	if (perfmonctl(0, PFM_GET_PMC_RESET_VAL, pc, cnum) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't get reset values: %s\n", strerror(errno));
	}

	for (i=0; i < cnum; i++) {
		printf("PMC%u 0x%016lx\n", pc[i].reg_num, pc[i].reg_value);
	}
	return 0;
}
