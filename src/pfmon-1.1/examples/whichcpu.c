/*
 * whichcpu.c - example of how to figure out the host CPU model detected by pfmlib
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

#include <perfmon/pfmlib.h>

int
main(void)
{
	char *model;
	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		return 1;
	}
	/*
	 * Now simply print the CPU model detected by pfmlib
	 *
	 * When the CPU model is not directly supported AND the generic support
	 * is compiled into the library, the detected will yield "Generic" which
	 * mean that only the architected features will be supported.
	 *
	 * This call can be used to tune applications based on the detected host
	 * CPU model. This is useful because some features are CPU model specific,
	 * such as address range restriction which is an Itanium feature.
	 *
	 */
	pfm_get_pmu_name(&model);
	printf("PMU model detected by pfmlib: %s\n", model); 
	return 0;
}
