/*
 * 
 *
 * Copyright (C) 2001 Hewlett-Packard Co
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

#include "pfmlib.h"

/* test function */
 
int
noploop(unsigned long loop)
{

	while (loop--) { }
	return 0;
}

int
main(int argc, char **argv)
{
	unsigned long loop;
	pfm_event_desc_t desc;
	char *events[]={
		"nops_retired",
		"ia64_inst_retired",
		NULL
	};
	unsigned long *vals;
	int ret = 1, count, i;

	count = sizeof(events)/sizeof(char *)-1;

	vals = malloc(count*sizeof(unsigned long));
	if (vals == NULL) {
		fprintf(stderr, "Cannot allocate vals\n");
		return 1;
	}

	desc = pfm_desc_alloc();
	if (desc == PFM_DESC_NULL) {
		fprintf(stderr, "Cannot allocate PFM desc\n");
		return 1;
	}

	/* configure and install counters */
	if (pfm_install_counters(desc, events, NULL, PFM_PLM3) == -1) {
		fprintf(stderr, "Invalid events configuration\n");
		goto end;
	}

	loop = argc > 1 ? atol(argv[1]) : 1;

	/* start monitoring */
	pfm_start();

	/* our test function */
	ret = noploop(loop);

	/* stop monitoring */
	pfm_stop();

	/* read results */
	if (pfm_read_counters(desc, count, vals) == -1) {
		fprintf(stderr, "Can't read counters\n");
		goto end;
	}

	/* print results */
	for (i=0; i < count; i++) {
		printf("%-16lu %s\n", vals[i], events[i]);
	}

	ret = 0;
end:		
	pfm_desc_free(desc);
	free(vals);

	return ret;
}
