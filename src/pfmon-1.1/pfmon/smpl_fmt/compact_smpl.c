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

#include "pfmon.h"

#define SMPL_OUTPUT_NAME	"compact"

#if 0
static int
explain_ita_reg(pfmon_smpl_ctx_t *csmpl, int num, int *column)
{
	int col = *column;
	int fd = csmpl->smpl_fd;
	pfmon_ita_options_t *opt = (pfmon_ita_options_t *)options.model_options;

	switch(num) {
		case 0:
			safe_fprintf(fd, "# column %u: PMD0 I-EAR instruction cache line address (0x%016lx if invalid)\n", col++, PMD0_INVALID_VALUE);

			if (opt->opt_use_iear_tlb)
				safe_fprintf(fd, "# column %u: PMD0 I-EAR instruction TLB type (\"-\" if invalid)\n", col++);
			break;
		case 1:
			if (opt->opt_use_iear_tlb) break;
			safe_fprintf(fd, "# column %u: PMD1 I-EAR latency in CPU cycles (%u if invalid)\n", col++, PMD1_INVALID_VALUE);
			break;
		case 2:
			safe_fprintf(fd, "# column %u: PMD2 D-EAR Data address (0x%016lx if invalid)\n", col++, PMD2_INVALID_VALUE);
			break;
		case 3:
			if (opt->opt_use_dear_tlb)
				safe_fprintf(fd, "# column %u: PMD3 D-EAR TLB level (\"-\" if invalid)\n", col++);
			else
				safe_fprintf(fd, "# column %u: PMD3 D-EAR Latency in CPU cycles (%u if invalid)\n", col++, PMD3_INVALID_VALUE);
			break;
		case 16:
			safe_fprintf(fd, "# column %u: PMD16 Branch Trace Buffer Index\n", col++);
			break;
		case 17:
			safe_fprintf(fd, "# column %u: PMD17 D-EAR instruction address (0x%16lx if invalid)\n", col++, PMD17_INVALID_VALUE);
			break;
		default: 
			/*
			* If we find a BTB then record it for later
			 */
			if (num>7 && num < 16) 
				safe_fprintf(fd, "# column %u: PMD%u branch history\n",  col++, num);
			else
				return -1;
	}
	*column = col;
	return 0;
}
#endif



static int
compact_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	perfmon_smpl_entry_t *ent = (perfmon_smpl_entry_t *)(hdr+1);
	int fd = csmpl->smpl_fd;
	unsigned long pos, msk;
	pmu_reg_t *reg;
	int i, j, ret;

	if (hdr->hdr_version != PFM_SMPL_VERSION) {
		fatal_error("perfmon v%u.%u sampling format is not supported\n", 
				PFM_VERSION_MAJOR(hdr->hdr_version),
				PFM_VERSION_MINOR(hdr->hdr_version));
	}

	/* sanity check */
	if (hdr->hdr_pmds[0] != options.smpl_regs) {
		fatal_error("kernel did not record PMDs we were expecting 0x%lx(kernel) != 0x%lx\n", hdr->hdr_pmds, options.smpl_regs);
	}

	pos = (unsigned long)ent;

	DPRINT(("hdr_count=%lu smpl_regs=0x%lx\n",hdr->hdr_count, hdr->hdr_pmds[0]));

	for(i=0; i < hdr->hdr_count; i++) {

		ret = 0;

		if (options.opt_no_ent_header == 0) 
			ret += safe_fprintf(fd, "%-8lu %-8d %-2d 0x%016lx 0x%04lx ",
				*csmpl->smpl_entry,
				ent->pid,
				ent->cpu,
				ent->stamp,
				ent->regs);

		reg = (pmu_reg_t *)(ent+1);

		for(j=0, msk = hdr->hdr_pmds[0]; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			ret += safe_fprintf(fd, "0x%016lx ", reg->pmu_reg);
			reg++;

			/* Lazily detect output error now */
			if (ret == 0) goto error;

		}

		ret += safe_fprintf(fd, "\n");

		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	

		(*csmpl->smpl_entry)++;
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
	int j, column = 6;
	int fd = csmpl->smpl_fd;

	safe_fprintf(fd, "#\n#\n# column 1: entry number\n"
	 	 	 "# column 2: process id\n"
		 	 "# column 3: CPU number\n"
		 	 "# column 4: unique timestamp\n"
		 	 "# column 5: bitmask of PMDs which overflowed\n");

	for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	

		if ((msk & 0x1) == 0) continue;

		safe_fprintf(csmpl->smpl_fd, "# column %u: PMD%d\n", column++, j);
	}

	return 0;
}

static int
validate_compact_smpl(pfmlib_param_t *evt)
{
	if (options.pfm_smpl_version != PFM_SMPL_VERSION) {
		warning("perfmon v%u.%u sampling format is not supported by the %s sampling output module\n", 
				SMPL_OUTPUT_NAME,
				PFM_VERSION_MAJOR(options.pfm_smpl_version),
				PFM_VERSION_MINOR(options.pfm_smpl_version));
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
