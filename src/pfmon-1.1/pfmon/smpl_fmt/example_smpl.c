/*
 * example_smpl.c - example of a sampling output format
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
/*
 * if the format is specific to a CPU model, then you can include the pfmon
 * CPU model specific header file:
 * #include "pfmon_itanium.h"
 */


/*
 * the name of the output format.
 *
 * You must make sure it is unique and hints the user as to which the format does
 */
#define SMPL_OUTPUT_NAME	"example"

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
 *
 * Important:
 * 	this routine should not use the stdio routine to print the results to the screen or file, use safe_fprintf() instead.
 * 	This routine avoids some locking problem in case pfmon is running system-wide SMP mode. This routine is called from
 * 	a SIGPROF signal handler and the ptread standard apparently stipulates that you cannot use STDIO routines from
 * 	a signal handler because of potential deadlock.
 */
static int
example_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	perfmon_smpl_entry_t *ent = (perfmon_smpl_entry_t *)(hdr+1);
	int fd = csmpl->smpl_fd;
	unsigned long pos, msk;
	pmu_reg_t *reg;
	int i, j, ret;

	/* sanity check */
	if (hdr->hdr_pmds[0] != options.smpl_regs) {
		fatal_error("kernel did not record PMDs we were expecting 0x%lx(kernel) != 0x%lx\n", hdr->hdr_pmds, options.smpl_regs);
	}

	pos = (unsigned long)ent;

	safe_fprintf(fd, "entries recorded=%lu smpl_regs=0x%lx\n", hdr->hdr_count, hdr->hdr_pmds[0]);

	/* 
	 * print the raw value of each PMD
	 */
	for(i=0; i < hdr->hdr_count; i++) {

		/*
		 * position just after the entry header
		 */
		reg = (pmu_reg_t *)(ent+1);

		/*
		 * there is one register saved er register indicated in the smpl_regs field
		 * of the pfarg_ctx_t. This field is duplicated in the hdr_pmds[] field of
		 * the sampling buffer header
		 */
		for(j=0, msk = hdr->hdr_pmds[0]; msk; msk >>=1, j++) {	

			if ((msk & 0x1) == 0) continue;

			/*
			 * safe_fprintf() returns:
			 * 	 - number of character written if successful
			 * 	 - 0 or -1 otherwise
			 */
			ret = safe_fprintf(fd, "0x%016lx ", reg->pmu_reg);
			if (ret <= 0) goto error;

			/*
			 * move to next register for this entry
			 */
			reg++;
		}
		/*
		 * complete the line
		 */
		ret += safe_fprintf(fd, "\n");

		/*
		 * move to the next sampling entry using the entry_size field.
		 * You should not rely on sizeof() for this as there may be a 
		 * gap between entries to get proper alignement
		 */
		pos += hdr->hdr_entry_size;
		ent = (perfmon_smpl_entry_t *)pos;	

		/*
		 * increment number of processed sampling entries (optional)
		 */
		(*csmpl->smpl_entry)++;
	}
	return 0;
error:
	fatal_error("cannot write to sampling file: %s\n", strerror(errno));
	/* not reached */
	return -1;
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
example_print_header(pfmon_smpl_ctx_t *csmpl)
{
	unsigned long msk;
	int j, column = 1;
	int fd = csmpl->smpl_fd;

	safe_fprintf(fd, "# using example sampling output format\n");

	for(j=0, msk = options.smpl_regs; msk; msk >>=1, j++) {	

		if ((msk & 0x1) == 0) continue;

		safe_fprintf(csmpl->smpl_fd, "# column %u: PMD%d\n", column++, j);
	}
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
example_validate_smpl(pfmlib_param_t *evt)
{
	/*
	 * check that the kernel uses the same sampling buffer format as we do
	 *
	 * the pfm_smpl_version field is initialized with the kernel sampling buffer format
	 * before coming here.
	 */
	if (options.pfm_smpl_version != PFM_SMPL_VERSION) {
		warning("perfmon v%u.%u sampling format is not supported by the %s sampling output module\n", 
				SMPL_OUTPUT_NAME,
				PFM_VERSION_MAJOR(options.pfm_smpl_version),
				PFM_VERSION_MINOR(options.pfm_smpl_version));
		return -1;
	}
	return 0;
}


/*
 * structure describing the format which is visible to pfmon_smpl.c
 *
 * The structure MUST be manually added to the smpl_outputs[] table in
 * pfmon_smpl.c
 */
pfmon_smpl_output_t example_smpl_output={
		/* 
		 * name of the format
		 */
		SMPL_OUTPUT_NAME,		
		/* 
		 * Because some formats can be used with more than one PMU models, this
		 * pmu_mask field is a bitfield. Each bit represents a PMU model. At least
		 * one bit must be set. use PFMON_PMU_MASK(t) where t is the PFMLIB_t_PMU
		 * from pfmlib.h
		 */
		PFMON_PMU_MASK(PFMLIB_GENERIC_PMU),		
		/*
		 * a small description
		 */
		"Sampling output example. Any CPU models",
		/*
		 * what is the validate function. NULL is also a valid choice here
		 */
/* validate */	example_validate_smpl,
		/*
		 * not yet used
		 */
/* open     */	NULL,
/* close    */	NULL,
		/*
		 * Routine to process sampling buffer on overflow. NULL is NOT an option here!
		 */
/* process  */	example_process_smpl_buffer,
		/*
		 * print a format-specific header. NULL is also a valid choice here
		 */
/* header   */	example_print_header
};
