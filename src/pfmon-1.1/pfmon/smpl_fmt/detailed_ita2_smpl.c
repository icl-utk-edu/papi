/*
 * detailed_ita2_smpl.c - detailed sampling output format for the Itanium2 PMU family
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

#include "pfmon.h"
#include "pfmon_itanium2.h"

#define	SMPL_OUTPUT_FORMAT "detailed-itanium2"

#define PMD0_INVALID_VALUE	(~0UL)
#define PMD1_INVALID_VALUE	(0U)
#define PMD2_INVALID_VALUE	(~0UL)
#define PMD3_INVALID_VALUE	(0U)	/* when doing cache measurements */
#define PMD17_INVALID_VALUE	(~0UL)


static int
show_ita2_btb_reg(FILE *fp, int j, pfm_ita2_reg_t reg, pfm_ita2_reg_t pmd16)
{
	int ret;
	unsigned long bruflush, b1;
	int is_valid = reg.pmd8_15_ita2_reg.btb_b == 0 && reg.pmd8_15_ita2_reg.btb_mp == 0 ? 0 :1; 

	b1       = (pmd16.reg_val >> (4 + 4*(j-8))) & 0x1;
	bruflush = (pmd16.reg_val >> (5 + 4*(j-8))) & 0x1;

	ret = fprintf(fp, "\tPMD%-2d: 0x%016lx b=%d mp=%d bru=%ld b1=%ld valid=%c\n",
			j,
			reg.reg_val,
			 reg.pmd8_15_ita2_reg.btb_b,
			 reg.pmd8_15_ita2_reg.btb_mp,
			 bruflush, b1,
			is_valid ? 'Y' : 'N');

	if (!is_valid) return ret;

	if (reg.pmd8_15_ita2_reg.btb_b) {
		unsigned long addr;

		
		addr = (reg.pmd8_15_ita2_reg.btb_addr+b1)<<4;

		addr |= reg.pmd8_15_ita2_reg.btb_slot < 3 ?  reg.pmd8_15_ita2_reg.btb_slot : 0;

		ret = fprintf(fp, "\t       Source Address: 0x%016lx\n"
				  "\t       Taken=%c Prediction: %s\n\n",
			 addr,
			 reg.pmd8_15_ita2_reg.btb_slot < 3 ? 'Y' : 'N',
			 reg.pmd8_15_ita2_reg.btb_mp ? "FE Failure" : 
			 bruflush ? "BE Failure" : "Success");
	} else {
		ret = fprintf(fp, "\t       Target Address: 0x%016lx\n\n",
			 (reg.pmd8_15_ita2_reg.btb_addr<<4));
	}

	return ret;
}

static int
show_ita2_btb_trace(FILE *fp, pfm_ita2_reg_t reg, pfm_ita2_reg_t *btb_regs)
{
	int i, last, ret;

	i    = (reg.pmd16_ita2_reg.btbi_full) ? reg.pmd16_ita2_reg.btbi_bbi : 0;
	last = reg.pmd16_ita2_reg.btbi_bbi;

	DPRINT(("btb_trace: i=%d last=%d bbi=%d full=%d\n", 
			i,
			last, 
			reg.pmd16_ita2_reg.btbi_bbi,
			reg.pmd16_ita2_reg.btbi_full));

	do {
		ret = show_ita2_btb_reg(fp, i+8, btb_regs[i], reg);
		i = (i+1) % 8;
	} while (i != last);

	return ret;
}

static int
print_ita2_reg(pfmon_smpl_ctx_t *csmpl, int rnum, unsigned long rval)
{
	static const char *tlb_levels[]={"N/A", "L2DTLB", "VHPT", "FAULT", "ALL"};
	static const char *tlb_hdls[]={"N/A", "L2TLB", "VHPT", "SW"};
	static pfm_ita2_reg_t btb_regs[PMU_ITA2_MAX_BTB];

	pfm_ita2_reg_t reg;
	pfm_ita2_reg_t pmd16;
	pfmon_ita2_options_t *opt = (pfmon_ita2_options_t *)options.model_options;
	FILE *fp = csmpl->smpl_fp;
	int ret = 0;
	int found_pmd16 = 0;

	reg.reg_val = rval;

	switch(rnum) {
		case 0:
			fprintf(fp, "\tPMD0 : 0x%016lx, valid=%c cache line 0x%lx",
				reg.reg_val,
				reg.pmd0_ita2_reg.iear_stat ? 'Y': 'N',
				reg.pmd0_ita2_reg.iear_iaddr<<5); /* cache line address */

			/* show which level the hit was handled */
			if (opt->iear_mode == PFMLIB_ITA2_EAR_TLB_MODE)
				ret = fprintf(fp, ", TLB %s\n", tlb_hdls[reg.pmd0_ita2_reg.iear_stat]);
			else
				ret = fprintf(fp, "\n");
			break;
		case 1:
			if (opt->iear_mode  != PFMLIB_ITA2_EAR_TLB_MODE)
				ret = fprintf(fp, "\tPMD1 : 0x%016lx, latency %u, overflow %c\n",
						reg.reg_val,
						reg.pmd1_ita2_reg.iear_latency, 
						reg.pmd1_ita2_reg.iear_overflow ? 'Y' : 'N');
			break;
		case 3:
			fprintf(fp, "\tPMD3 : 0x%016lx, valid %c", 
					reg.reg_val,
					reg.pmd3_ita2_reg.dear_stat ? 'Y' : 'N');


			if (opt->dear_mode == PFMLIB_ITA2_EAR_TLB_MODE) {
				ret = fprintf(fp, ", TLB %s\n", tlb_levels[reg.pmd3_ita2_reg.dear_stat]);
			} else if (opt->dear_mode == PFMLIB_ITA2_EAR_CACHE_MODE) {
				ret = fprintf(fp, ", latency %u, overflow %c\n", 
						reg.pmd3_ita2_reg.dear_latency,
						reg.pmd3_ita2_reg.dear_overflow ? 'Y' : 'N');
			} else {
				fputc('\n', fp);
			}
			break;
		case 16:
			/*
			 * keep track of what the BTB index is saying
			 */
			pmd16 = reg;
			found_pmd16 = 1;
			break;
		case 17:
			/*
			 * iaddr is the address of the 2-bundle group (size of dispersal window)
			 * therefore we adjust it with the pdm17.bn field to get which of the 2 bundles
			 * caused the miss.
			 */
			ret = fprintf(fp, "\tPMD17: 0x%016lx, valid %c, bundle %d, address 0x%016lx\n",
					reg.reg_val,
					reg.pmd17_ita2_reg.dear_vl ? 'Y': 'N',
					reg.pmd17_ita2_reg.dear_bn,
					((reg.pmd17_ita2_reg.dear_iaddr+reg.pmd17_ita2_reg.dear_bn) << 4) | reg.pmd17_ita2_reg.dear_slot);
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
	if (found_pmd16) ret = show_ita2_btb_trace(fp, pmd16, btb_regs);

	return ret;
}

static int
detailed_ita2_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr;
	perfmon_smpl_entry_t *ent;
	FILE *fp = csmpl->smpl_fp;
	unsigned long pos, msk;
	pmu_reg_t *reg;
	int i, j, ret;


	hdr = csmpl->smpl_hdr;
	ent = (perfmon_smpl_entry_t *)(hdr+1);
	pos = (unsigned long)ent;

	DPRINT(("hdr_count=%ld smpl_regs=0x%lx\n",hdr->hdr_count, options.smpl_regs));

	for(i=0; i < hdr->hdr_count; i++) {
		ret =  fprintf(fp, 
			"entry %ld PID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			*csmpl->smpl_entry,
			ent->pid,
			ent->cpu,
			ent->stamp,
			ent->ip);

		(*csmpl->smpl_entry)++;

		ret += fprintf(fp, "\tPMD OVFL: ");
		msk = ent->regs >> PMU_FIRST_COUNTER;
		for(j=PMU_FIRST_COUNTER ; msk; msk >>=1, j++) {	
			if (msk & 0x1) fprintf(fp, "%d ", j);
		}

		ret += fprintf(fp, "\n");

		reg = (pmu_reg_t*)(ent+1);

		for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			ret = print_ita2_reg(csmpl, j, reg->reg_val);
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

static int
validate_ita2_smpl(pfmlib_param_t *evt)
{
	if (options.pfm_smpl_version != PFM_SMPL_VERSION) {
		warning("perfmon v%u.%u sampling format is not supported by the %s sampling output module\n", 
				SMPL_OUTPUT_FORMAT,
				PFM_VERSION_MAJOR(options.pfm_smpl_version),
				PFM_VERSION_MINOR(options.pfm_smpl_version));
		return -1;
	}
	return 0;
}


pfmon_smpl_output_t detailed_itanium2_smpl_output={
		SMPL_OUTPUT_FORMAT,
		PFMON_PMU_MASK(PFMLIB_ITANIUM2_PMU),
		"Details each event in clear text",
/* validate */	validate_ita2_smpl,
/* open     */	NULL,
/* close    */	NULL,
/* process  */	detailed_ita2_process_smpl_buffer,
/* header   */	NULL
};
