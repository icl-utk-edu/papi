/*
 * raw_smpl.c - raw output for sampling buffer
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

#define SMPL_OUTPUT_NAME	"raw"

static int
raw_process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	size_t sz;

	if (csmpl->entry_count == 0) {
		if (write(fileno(csmpl->smpl_fp), hdr, sizeof(*hdr)) < sizeof(*hdr)) goto error;
	}

	sz = hdr->hdr_entry_size*hdr->hdr_count;
	csmpl->entry_count += hdr->hdr_count;

	if (write(fileno(csmpl->smpl_fp), (hdr+1), sz) != sz) goto error;

	return 0;
error:
	fatal_error("cannot write to raw sampling file: %s\n", strerror(errno));
	/* not reached */
	return -1;
}

static int
validate_raw_smpl(pfmlib_param_t *evt)
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


pfmon_smpl_output_t raw_smpl_output={
	SMPL_OUTPUT_NAME,
	PFMON_PMU_MASK(PFMLIB_GENERIC_PMU),
	"raw values in binary format",
	validate_raw_smpl,
	NULL,
	NULL,
	raw_process_smpl_buffer,
	NULL
};
