/*
 * btb_smpl.c - shows one branch per line on Itanium & Itanium2 BTB
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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/*
 * include pfmon main header file. In this file we find the definition
 * of what a sampling output format must provide
 */
#include "pfmon.h"
#include <perfmon/pfmlib_itanium.h>

/*
 * the name of the output format.
 *
 * You must make sure it is unique and hints the user as to which the format does
 */
#define SMPL_OUTPUT_NAME	"btb"

/*
 * The routine which processes the sampling buffer when an overflow if notified to pfmon
 *
 * Argument: a point to a pfmon_smpl_ctx_t structure. 
 * 	     This structures contains:
 * 	     	- the file descriptor to use for printing
 * 	     	- a counter that can be increment to keep track of how many entries have been processed so far
 * 	     	- the user level virtual base address of the kernel sampling buffer
 * 	     	- the cpu mask indicating the CPU from which we are now processing data (only 1 bit is set)
 * Return:
 * 	 0: success
 * 	-1: error
 */

static int pmu_is_itanium2;	/* true if host PMU is of type Itanium2 */

static int
btb_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	perfmon_smpl_entry_t *ent = (perfmon_smpl_entry_t *)(hdr+1);
	FILE *fp = csmpl->smpl_fp;
	unsigned long pos, val, pmd16, entry;
	unsigned long *btb;
	int i, ret = 0;
	int l,f, last_type;
	static unsigned long entries_saved;	/* total number of entries saved */

	/*
	 * make sure PMD8-PMD16 are recorded
	 */
	if ((options.smpl_regs & 0x1ff00UL) != 0x1ff00UL) {
		warning("you are not recording the PMDS representing the BTB: 0x%lx\n", options.smpl_regs);
		return -1;
	}

	/* sanity check */
	if (hdr->hdr_pmds[0] != options.smpl_regs) {
		fatal_error("kernel did not record PMDs we were expecting 0x%lx(kernel) != 0x%lx\n", hdr->hdr_pmds, options.smpl_regs);
	}

	pos   = (unsigned long)ent;
	entry = options.opt_aggregate_res ? entries_saved : csmpl->entry_count;

	/* 
	 * print the raw value of each PMD
	 */
	for(i=0; i < hdr->hdr_count; i++) {

		/*
		 * overlay btb array
		 */
		btb   = (unsigned long *)(ent+1);
		pmd16 = btb[8];

		/* btb[8]=pmd16, compute first and last element */
		f = pmd16 & 0x8 ? pmd16 & 0x7 : 0;
		l = pmd16 & 0x7;

	DPRINT(("btb_trace: pmd16=0x%lx i=%d last=%d bbi=%d full=%d ita: bbi=%d full=%d\n", 
			btb[8],
			f,
			l, 
			btb[8] & 0x7,
			(btb[8]>>3) & 0x1));

		last_type = 2; /* no branch type, force 0 to be printed */
		do {
			if ((btb[f] & 0x1)) { /* branch instruction */
				if (last_type < 2) {
					if (last_type == 1) fprintf(fp, "0x0000000000000000");
					fprintf(fp, "\n");
				}
			} else { /* branch target */
				if (last_type == 2) fprintf(fp, "0x0000000000000000 ");
			}

			val = btb[f];

			/* correct bundle address on Itanium2 */
			if (pmu_is_itanium2) {
				unsigned long b;
				b       = (pmd16 >> (4 + 4*f)) & 0x1;
				val 	+= b<<4;
			}
			ret = fprintf(fp, "0x%016lx ", val); /* it is enough to capture write error here only */

			last_type = btb[f] & 0x1; /* 1= last PMD contained a branch insn, 0=a branch target */

			f = (f+1) % 8;

		} while (f != l && ret >0);

		if (last_type == 1) fprintf(fp, "0x0000000000000000");
		fprintf(fp, "\n");

		/*
		 * move to the next sampling entry using the entry_size field.
		 * You should not rely on sizeof() for this as there may be a 
		 * gap between entries to get proper alignement
		 */
		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	
		entry++;
		/*
		 * increment number of processed sampling entries (optional)
		 */
	}
	/* update counts */
	if (options.opt_aggregate_res) {
		csmpl->entry_count += entry - entries_saved;
		entries_saved      = entry;
	} else {
		entries_saved       += entry - csmpl->entry_count; 
		csmpl->entry_count = entry;
	}

	return ret > 0 ? 0 : -1; /* if we could not save all entries, return error */
}

/*
 * Print explanation about the format of sampling output. This function is optional.
 * The output of this function becomes visible only when the --with-header option
 * is specified. The output is placed after the standard pfmon header.
 *
 * Argument: 
 * 	a pointer to the pfmon sampling context  which contains:
 * 	     	- the file descriptor to use for printing
 * 	     	- a counter that can be increment to keep track of how many entries have been processed so far
 * 	     	- the user level virtual base address of the kernel sampling buffer
 * 	     	- the cpu mask indicating the CPU from which we are now processing data (only 1 bit is set)
 *
 * Return:
 * 	 0 if successful
 * 	-1 otherwise
 */
static int
btb_print_header(pfmon_smpl_ctx_t *csmpl)
{
	FILE *fp = csmpl->smpl_fp;

	fprintf(fp, "# btb only sampling output format (1 branch event/line)\n"
			 "# line format: source target\n"
			 "# a value of 0x0000000000000000 indicates source or target not captured\n"
		    );
	return 0;
}

/*
 * Function invoked before monitoring is started to verify that pfmon is configured
 * (invoked) in a way that is compatible with the use of this format. Note that the 
 * CPU model is already checked by then.
 *
 * Argument:
 * 	a pointer to the pfmlib_param-t structure which will be passed to libpfm.
 * 	At this point the structure is fully initialized, so the user is free to peek 
 * 	values out of it, modifications are NOT recommended.
 * Return:
 * 	 0 if validation is successful
 * 	-1 otherwise
 *
 * IMPORTANT: This is the place to check if the format is compatible with the kernel
 * sampling buffer format. 
 */
static int
btb_validate_smpl(pfmlib_param_t *evt)
{
	int idx;
	int pmu_type;

	/*
	 * check that the kernel uses the same sampling buffer format as we do
	 *
	 * the pfm_smpl_version field is initialized with the kernel sampling buffer format
	 * before coming here.
	 */
	if (PFM_VERSION_MAJOR(options.pfm_smpl_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
		warning("perfmon v%u.%u sampling format is not supported by the %s sampling output module\n", 
				PFM_VERSION_MAJOR(options.pfm_smpl_version),
				PFM_VERSION_MINOR(options.pfm_smpl_version),
				SMPL_OUTPUT_NAME);
		return -1;
	}
	/*
	 * must be measuring BTB only
	 */
	if (evt->pfp_event_count > 1) {
		warning("the btb sampling output format with the BRANCH_EVENT only\n");
		return -1;
	}
	/*
	 * find event for host CPU
	 */
	if (pfm_find_event_byname("BRANCH_EVENT", &idx) != PFMLIB_SUCCESS) {
		warning("cannot find BRANCH_EVENT in the event table for this PMU\n");
		return -1;
	}
	/*
	 * verify we have the right event
	 */
	if (evt->pfp_events[0].event != idx) {
		warning("you must use the BRANCH_EVENT event\n");
		return -1;
	}
	/* need to know if Itanium or Itanium2 */
	pfm_get_pmu_type(&pmu_type);

	pmu_is_itanium2 = pmu_type == PFMLIB_ITANIUM2_PMU ? 1 : 0;

	return 0;
}


/*
 * structure describing the format which is visible to pfmon_smpl.c
 *
 * The structure MUST be manually added to the smpl_outputs[] table in
 * pfmon_smpl.c
 */
pfmon_smpl_output_t btb_smpl_output={
		SMPL_OUTPUT_NAME,		
		PFMON_PMU_MASK(PFMLIB_ITANIUM_PMU)|PFMON_PMU_MASK(PFMLIB_ITANIUM2_PMU),		
		"Column-style BTB raw values",
/* validate */	btb_validate_smpl,
/* open     */	NULL,
/* close    */	NULL,
/* process  */	btb_process_smpl_buffer,
/* header   */	btb_print_header
};
