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


#include "pfmon.h"
#include "pfmon_itanium.h"

#include <perfmon/pfmlib_itanium.h>

#define PMD0_INVALID_VALUE	(~0UL)
#define PMD1_INVALID_VALUE	(0U)
#define PMD2_INVALID_VALUE	(~0UL)
#define PMD3_INVALID_VALUE	(0U)	/* when doing cache measurements */
#define PMD17_INVALID_VALUE	(~0UL)


#define SMPL_OUTPUT_NAME	"detailed-itanium"

static int
show_ita_btb_reg(FILE *fp, int j, pfm_ita_reg_t reg)
{
	int ret;
	int is_valid = reg.pmd8_15_ita_reg.btb_b == 0 && reg.pmd8_15_ita_reg.btb_mp == 0 ? 0 :1; 

	ret = fprintf(fp, "\tPMD%-2d: 0x%016lx b=%d mp=%d valid=%c\n",
			j,
			reg.reg_val,
			 reg.pmd8_15_ita_reg.btb_b,
			 reg.pmd8_15_ita_reg.btb_mp,
			is_valid ? 'Y' : 'N');

	if (!is_valid) return ret;

	if (reg.pmd8_15_ita_reg.btb_b) {
		unsigned long addr;

		addr = 	reg.pmd8_15_ita_reg.btb_addr<<4;
		addr |= reg.pmd8_15_ita_reg.btb_slot < 3 ?  reg.pmd8_15_ita_reg.btb_slot : 0;

		ret = fprintf(fp, "\t       Source Address: 0x%016lx\n"
				  "\t       Taken=%c Prediction: %s\n\n",
			 addr,
			 reg.pmd8_15_ita_reg.btb_slot < 3 ? 'Y' : 'N',
			 reg.pmd8_15_ita_reg.btb_mp ? "Failure" : "Success");
	} else {
		ret = fprintf(fp, "\t       Target Address: 0x%016lx\n\n",
			 (reg.pmd8_15_ita_reg.btb_addr<<4));
	}

	return ret;
}

static int
show_ita_btb_trace(FILE *fp, pfm_ita_reg_t reg, pfm_ita_reg_t *btb_regs)
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
		ret = show_ita_btb_reg(fp, i+8, btb_regs[i]);
		i = (i+1) % 8;
	} while (i != last);

	return ret;
}

static int
print_ita_reg(pfmon_smpl_ctx_t *csmpl, int rnum, unsigned long rval)
{
	static const char *tlb_levels[]={"N/A", "L2DTLB", "VHPT", "SW"};
	static const char *tlb_hdls[]={"VHPT", "SW"};
	static pfm_ita_reg_t btb_regs[PMU_ITA_NUM_BTB];

	pfm_ita_reg_t reg;
	pfm_ita_reg_t pmd16;
	pfmon_ita_options_t *opt = (pfmon_ita_options_t *)options.model_options;
	FILE *fp = csmpl->smpl_fp;
	int ret = 0;
	int found_pmd16 = 0;

	reg.reg_val = rval;

	switch(rnum) {
		case 0:
			fprintf(fp, "\tPMD0 : 0x%016lx, valid %c, cache line 0x%lx",
				reg.reg_val,
				reg.pmd0_ita_reg.iear_v ? 'Y': 'N',
				reg.pmd0_ita_reg.iear_icla<<5);

			if (opt->opt_use_iear_tlb)
				ret = fprintf(fp, ", TLB %s\n", tlb_hdls[reg.pmd0_ita_reg.iear_tlb]);
			else
				ret = fprintf(fp, "\n");
			break;
		case 1:
			if (opt->opt_use_iear_tlb == 0)
				ret = fprintf(fp, "\tPMD1 : 0x%016lx, latency %u\n",
						reg.reg_val,
						reg.pmd1_ita_reg.iear_lat);
			break;
		case 3:
			fprintf(fp, "\tPMD3 : 0x%016lx ", reg.reg_val);

			if (opt->opt_use_dear_tlb)
				ret = fprintf(fp, ", TLB %s\n", tlb_levels[reg.pmd3_ita_reg.dear_level]);
			else
				ret = fprintf(fp, ", latency %u\n", reg.pmd3_ita_reg.dear_lat);
			break;
		case 16:
			/*
			 * keep track of what the BTB index is saying
			 */
			pmd16 = reg;
			found_pmd16 = 1;
			break;
		case 17:

			ret = fprintf(fp, "\tPMD17: 0x%016lx, valid %c, address 0x%016lx\n",
					reg.reg_val,
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
				ret = fprintf(fp, "\tPMD%-2d: 0x%016lx\n", rnum, reg.reg_val);
	}

	if (found_pmd16) ret = show_ita_btb_trace(fp, pmd16, btb_regs);

	return ret;
}

static int
validate_ita_smpl(pfmlib_param_t *evt)
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


static int
detailed_ita_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr;
	perfmon_smpl_entry_t *ent;
	FILE *fp = csmpl->smpl_fp;
	unsigned long pos, msk, entry;
	pfm_ita_reg_t *reg;
	int i, j, ret;
	static unsigned long entries_saved;	/* total number of entries saved */


	hdr   = csmpl->smpl_hdr;
	ent   = (perfmon_smpl_entry_t *)(hdr+1);
	pos   = (unsigned long)ent;
	entry = options.opt_aggregate_res ? entries_saved : csmpl->entry_count;

	DPRINT(("hdr_count=%ld smpl_regs=0x%lx\n",hdr->hdr_count, options.smpl_regs));

	for(i=0; i < hdr->hdr_count; i++) {
		ret =  fprintf(fp, 
			"entry %ld PID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			entry,
			ent->pid,
			ent->cpu,
			ent->stamp,
			ent->ip);

		ret += fprintf(fp, "\tOVFL: ");
		msk = ent->regs >> PMU_FIRST_COUNTER;
		for(j=PMU_FIRST_COUNTER ; msk; msk >>=1, j++) {	
			if (msk & 0x1) fprintf(fp, "%d ", j);
		}

		/* when randomization is not supported this value is 0 */
		ret += fprintf(fp, " LAST_VAL: %lu\n", -ent->last_reset_val);

		reg = (pfm_ita_reg_t*)(ent+1);

		for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			ret = print_ita_reg(csmpl, j, reg->reg_val);
			if (ret == -1) goto error;
			reg++;
		}
		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	
		entry++;
	}
	/* update counts */
	if (options.opt_aggregate_res) {
		csmpl->entry_count += entry - entries_saved;
		entries_saved       = entry;
	} else {
		entries_saved     += entry - csmpl->entry_count; 
		csmpl->entry_count = entry;
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
