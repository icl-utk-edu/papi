/*
 * detailed_ita_smpl.c - detailed sampling output format for Itanium PMU
 *
 * Copyright (C) 2001-2002 Hewlett-Packard Co
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
#include <unistd.h>
#include <errno.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"
#include "pfmon_itanium.h"

#define PMD0_INVALID_VALUE	(~0UL)
#define PMD1_INVALID_VALUE	(0U)
#define PMD2_INVALID_VALUE	(~0UL)
#define PMD3_INVALID_VALUE	(0U)	/* when doing cache measurements */
#define PMD17_INVALID_VALUE	(~0UL)


#define SMPL_OUTPUT_NAME	"detailed-itanium"

static int
show_ita_btb_reg(int fd, int j, pfm_ita_reg_t reg)
{
	int ret;
	int is_valid = reg.pmd8_15_ita_reg.btb_b == 0 && reg.pmd8_15_ita_reg.btb_mp == 0 ? 0 :1; 

	ret = safe_fprintf(fd, "\tPMD%-2d: 0x%016lx b=%d mp=%d valid=%c\n",
			j,
			reg.pmu_reg,
			 reg.pmd8_15_ita_reg.btb_b,
			 reg.pmd8_15_ita_reg.btb_mp,
			is_valid ? 'Y' : 'N');

	if (!is_valid) return ret;

	if (reg.pmd8_15_ita_reg.btb_b) {
		ret = safe_fprintf(fd, "\t       Source Address: 0x%016lx (slot %d)\n"
						"\t       Prediction: %s\n\n",
			 (reg.pmd8_15_ita_reg.btb_addr<<4), 
			 reg.pmd8_15_ita_reg.btb_slot,
			 reg.pmd8_15_ita_reg.btb_mp ? "Failure" : "Success");
	} else {
		ret = safe_fprintf(fd, "\t       Target Address: 0x%016lx\n\n",
			 (reg.pmd8_15_ita_reg.btb_addr<<4));
	}

	return ret;
}

static int
show_ita_btb_trace(int fd, pfm_ita_reg_t reg, pfm_ita_reg_t *btb_regs)
{
	int i, last, ret;

	i    = (reg.pmd16_ita_reg.btbi_full) ? reg.pmd16_ita_reg.btbi_bbi : 0;
	last = reg.pmd16_ita_reg.btbi_bbi;

	DPRINT(("btb_trace: i=%d last=%d bbi=%d full=%d\n", 
			i,
			last, 
			reg.pmd16_ita_reg.btbi_bbi,
			reg.pmd16_ita_reg.btbi_full));

	do {
		ret = show_ita_btb_reg(fd, i+8, btb_regs[i]);
		i = (i+1) % 8;
	} while (i != last);

	return ret;
}

static int
print_ita_reg(pfmon_smpl_ctx_t *csmpl, int rnum, unsigned long rval)
{
	static const char *tlb_levels[]={"N/A", "L2DTLB", "VHPT", "SW"};
	static const char *tlb_hdls[]={"VHPT", "SW"};
	static pfm_ita_reg_t btb_regs[PMU_ITA_MAX_BTB];

	pfm_ita_reg_t reg;
	pfm_ita_reg_t pmd16;
	pfmon_ita_options_t *opt = (pfmon_ita_options_t *)options.model_options;
	int fd = csmpl->smpl_fd;
	int ret = 0;
	int found_pmd16 = 0;

	reg.pmu_reg = rval;

	switch(rnum) {
		case 0:
			safe_fprintf(fd, "\tPMD0 : 0x%016lx, valid %c, cache line 0x%lx",
				reg.pmu_reg,
				reg.pmd0_ita_reg.iear_v ? 'Y': 'N',
				reg.pmd0_ita_reg.iear_icla<<5L);

			if (opt->opt_use_iear_tlb)
				ret = safe_fprintf(fd, ", TLB %s\n", tlb_hdls[reg.pmd0_ita_reg.iear_tlb]);
			else
				ret = safe_fprintf(fd, "\n");
			break;
		case 1:
			if (opt->opt_use_iear_tlb == 0)
				ret = safe_fprintf(fd, "\tPMD1 : 0x%016lx, latency %u\n",
						reg.pmu_reg,
						reg.pmd1_ita_reg.iear_lat);
			break;
		case 3:
			safe_fprintf(fd, "\tPMD3 : 0x%016lx ", reg.pmu_reg);

			if (opt->opt_use_dear_tlb)
				ret = safe_fprintf(fd, ", TLB %s\n", tlb_levels[reg.pmd3_ita_reg.dear_level]);
			else
				ret = safe_fprintf(fd, ", latency %u\n", reg.pmd3_ita_reg.dear_lat);
			break;
		case 16:
			/*
			 * keep track of what the BTB index is saying
			 */
			pmd16 = reg;
			found_pmd16 = 1;
			break;
		case 17:

			ret = safe_fprintf(fd, "\tPMD17: 0x%016lx, valid %c, address 0x%016lx\n",
					reg.pmu_reg,
					reg.pmd17_ita_reg.dear_v ? 'Y': 'N',
					(reg.pmd17_ita_reg.dear_iaddr << 4) | reg.pmd17_ita_reg.dear_slot);
			break;
		default:
			/*
			* If we find a BTB then record it for later
			 */
			if (rnum>7 && rnum < 16)
				btb_regs[rnum-8] = reg;
			else
				ret = safe_fprintf(fd, "\tPMD%-2d: 0x%016lx\n", rnum, reg.pmu_reg);
	}

	if (found_pmd16) ret = show_ita_btb_trace(fd, pmd16, btb_regs);

	return ret;
}

static int
validate_ita_smpl(pfmlib_param_t *evt)
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


static int
detailed_ita_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr;
	perfmon_smpl_entry_t *ent;
	int fd = csmpl->smpl_fd;
	unsigned long pos, msk;
	pmu_reg_t *reg;
	int i, j, ret;


	hdr = csmpl->smpl_hdr;
	ent = (perfmon_smpl_entry_t *)(hdr+1);
	pos = (unsigned long)ent;

	DPRINT(("hdr_count=%ld smpl_regs=0x%lx\n",hdr->hdr_count, options.smpl_regs));
	safe_fprintf(fd, "hdr_count=%ld smpl_regs=0x%lx\n",hdr->hdr_count, options.smpl_regs);

	for(i=0; i < hdr->hdr_count; i++) {
		ret =  safe_fprintf(fd, 
			"entry %ld PID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			*csmpl->smpl_entry,
			ent->pid,
			ent->cpu,
			ent->stamp,
			ent->ip);

		(*csmpl->smpl_entry)++;

		ret += safe_fprintf(fd, "\tPMD OVFL: ");
		msk = ent->regs >> PMU_FIRST_COUNTER;
		for(j=PMU_FIRST_COUNTER ; msk; msk >>=1, j++) {	
			//pfm_get_event_name(options.monitor_events[options.rev_pc[j]], &name);
			//if (msk & 0x1) safe_fprintf(fd, "%s(%d) ", name, j);
			if (msk & 0x1) safe_fprintf(fd, "%d ", j);
		}

		ret += safe_fprintf(fd, "\n");

		reg = (pmu_reg_t*)(ent+1);

		for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			ret = print_ita_reg(csmpl, j, reg->pmu_reg);
			if (ret == -1) goto error;
			reg++;
		}
		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	
	}
	return 0;
error:
	fatal_error("cannot write to sampling file: %s\n", strerror(errno));
	/* not reached */
	return -1;
}

pfmon_smpl_output_t detailed_itanium_smpl_output={
		SMPL_OUTPUT_NAME,
		PFMON_PMU_MASK(PFMLIB_ITANIUM_PMU),
		"Details each event in clear text",
/* validate */	validate_ita_smpl,
/* open     */	NULL,
/* close    */	NULL,
/* process  */	detailed_ita_process_smpl_buffer,
/* header   */	NULL
};
