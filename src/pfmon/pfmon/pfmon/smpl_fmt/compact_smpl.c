/*
 * compact_smpl.c - compact output for sampling buffer
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
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <perfmon/pfmlib_generic.h>

#include "pfmon.h"
#define SMPL_OUTPUT_NAME	"compact"


static int
compact_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	perfmon_smpl_entry_t *ent;
	FILE *fp = csmpl->smpl_fp;
	unsigned long pos, entry;
	pfm_gen_reg_t *reg;
	int i, ret, npmds, n;
	static unsigned long entries_saved;	/* total number of entries saved */



	npmds = hweight64(hdr->hdr_pmds[0]);
	ent   = (perfmon_smpl_entry_t *)(hdr+1);
	entry = options.opt_aggregate_res ? entries_saved : csmpl->entry_count;
	pos   = (unsigned long)ent;

	DPRINT(("hdr_count=%lu smpl_regs=0x%lx npmds%lu\n",hdr->hdr_count, hdr->hdr_pmds[0], npmds));

	for(i=0; i < hdr->hdr_count; i++) {
		n = npmds;
		ret = fprintf(fp, "%-8lu %-8d %-2d 0x%016lx 0x%016lx 0x%04lx %lu ",
				entry,
				ent->pid,
				ent->cpu,
				ent->ip,
				ent->stamp,
				ent->regs,
				-1*ent->last_reset_val);

		reg = (pfm_gen_reg_t *)(ent+1);

		while(n--) {
			ret = fprintf(fp, "0x%016lx ", reg->reg_val);
			reg++;
		}
		ret = fputc('\n', fp);

		/* Lazily detect output error now */
		if (ret == 0) goto error;
		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	
		entry++;
	}

	/* update counts */
	if (options.opt_aggregate_res) {
		csmpl->entry_count += entry - entries_saved;
		entries_saved      = entry;
	} else {
		entries_saved       += entry - csmpl->entry_count; 
		csmpl->entry_count = entry;
	}

	return 0;
error:
	fatal_error("cannot write to sampling file: %s\n", strerror(errno));
	/* not reached */
	return -1;
}

static int
compact_print_header(pfmon_smpl_ctx_t *csmpl)
{
	unsigned long msk;
	int j, col;
	FILE *fp = csmpl->smpl_fp;

	fprintf(fp, "#\n#\n# column  1: entry number\n"
	 	 	  "# column  2: process id\n"
		 	  "# column  3: cpu number\n"
		 	  "# column  4: instruction pointer\n"
		 	  "# column  5: unique timestamp\n"
		 	  "# column  6: bitmask of PMDs which overflowed\n");
	/*
	 * we keep the same number of columns to avoid confusion and print a warning 
	 * message
	 */
	if (options.opt_has_random)
		fprintf(fp, "# column  7: initial value of first overflowed PMD\n");
	else
		fprintf(fp, "# column  7: UNSUPPORTED FEATURE (initial value of first overflowed PMD)\n");

	col = 8;

	for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	

		if ((msk & 0x1) == 0) continue;

		fprintf(csmpl->smpl_fp, "# column %2u: PMD%d\n", col, j);
		col++;
	}

	return 0;
}

static int
validate_compact_smpl(pfmlib_param_t *evt)
{
	if (PFM_VERSION_MAJOR(options.pfm_smpl_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
		warning("perfmon v%u.%u sampling format is not supported by the %s sampling output module\n", 
				PFM_VERSION_MAJOR(options.pfm_smpl_version),
				PFM_VERSION_MINOR(options.pfm_smpl_version),
				SMPL_OUTPUT_NAME);
		return -1;
	}

	return 0;
}


pfmon_smpl_output_t compact_smpl_output={
		SMPL_OUTPUT_NAME,
		PFMON_PMU_MASK(PFMLIB_GENERIC_PMU),
		"Column-style raw values",
/* validate */	validate_compact_smpl,
/* open     */	NULL,
/* close    */	NULL,
/* process  */	compact_process_smpl_buffer,
/* header   */	compact_print_header
};
