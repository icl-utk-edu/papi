/*
 * pfmon_smpl.c - sampling support for pfmon
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
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

#define PFMON_DFL_SMPL_ENTRIES	2048

extern pfmon_smpl_output_t raw_smpl_output;
extern pfmon_smpl_output_t compact_smpl_output;
extern pfmon_smpl_output_t detailed_itanium_smpl_output;
//extern pfmon_smpl_output_t example_smpl_output;

static pfmon_smpl_output_t *smpl_outputs[]={
	&detailed_itanium_smpl_output,
	&raw_smpl_output,
	&compact_smpl_output,
//	&example_smpl_output,
	NULL
};


static void
print_smpl_output_header(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	char *name;
	int fd = csmpl->smpl_fd;
	unsigned long msk;
	int i;

	print_standard_header(fd, options.opt_aggregate_res ? 
				  options.cpu_mask : csmpl->cpu_mask);

	if (hdr) {
		safe_fprintf(fd, "#\n# kernel sampling format: %d.%d\n# sampling entry size: %lu\n", 
			PFM_VERSION_MAJOR(hdr->hdr_version), 
			PFM_VERSION_MINOR(hdr->hdr_version), 
			hdr->hdr_entry_size);
	}

	safe_fprintf(fd, "#\n# recorded PMDs: ");

	for(i=0, msk =options.smpl_regs; msk; msk>>=1, i++) 
		if (msk & 0x1) safe_fprintf(fd, "PMD%d ", i);

	safe_fprintf(fd, "\n# sampling entries count: %lu\n", options.smpl_entries);

	safe_fprintf(fd, "#\n# sampling rates (short/long): ");

	for(i=0; i < options.monitor_count-1; i++) {
		pfm_get_event_name(options.monitor_events[i], &name);
		if (options.short_rates[i] == 0 && options.long_rates[i] == 0) {
			safe_fprintf(fd, "%s(none), ", name); 
		} else {
			safe_fprintf(fd, "%s(%lu/%lu), ", 
				name,
				options.short_rates[i]*-1,
				options.long_rates[i]*-1);
		}
	}
	pfm_get_event_name(options.monitor_events[i], &name);
	if (options.short_rates[i] == 0 && options.long_rates[i] == 0) {
		safe_fprintf(fd, "%s(none)\n#\n",  name);
	} else {
		safe_fprintf(fd, "%s(%lu/%lu)\n#\n", 
			name,
			options.short_rates[i]*-1,
			options.long_rates[i]*-1);
	}

	/* 
	 * invoke additional header printing routine if defined
	 */
	if (options.smpl_output->print_header)
		(*options.smpl_output->print_header)(csmpl);

	safe_fprintf(fd, "#\n#\n");
}

/*
 * id indicates which "context" to use. This is mostly used for system wide 
 * when more than one CPUs are being monitored.
 */
int
process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	return (*options.smpl_output->process_smpl)(csmpl);
}

int
setup_sampling_output(pfmon_smpl_ctx_t *csmpl)
{
        char filename[PFMON_MAX_FILENAME_LEN];
        int fd = fileno(stdout);

	if (options.opt_use_smpl == 0) return 0;

        if (options.smpl_file) {
                if (options.opt_syst_wide && options.opt_aggregate_res == 0) {
                        sprintf(filename, "%s.cpu%d", options.smpl_file, find_cpu(getpid()));
                } else {
                        strcpy(filename, options.smpl_file);
                }

                fd = open(filename, O_CREAT|O_TRUNC|O_WRONLY, 0666);
                if (fd == -1) {
                        warning("cannot create sampling output file %s: %s\n", options.smpl_file, strerror(errno));
                        return -1;
                }
                vbprintf("results are in file \"%s\"\n",filename);

        }

        csmpl->smpl_fd = fd;

	if (options.opt_with_header) print_smpl_output_header(csmpl);

        return 0;
}

void
close_sampling_output(pfmon_smpl_ctx_t *csmpl)
{
	if (options.opt_use_smpl == 0) return;

	if (csmpl->smpl_fd != fileno(stdout)) close(csmpl->smpl_fd);
}

void 
setup_sampling_rates(pfmlib_param_t *evt, char *smpl_args, char *ovfl_args)
{
	int cnt = 0;

	if (smpl_args) {
		/*
		 * in case not all rates are specified, they will default to zero, i.e. no sampling
		 * on this counter
		 */
		cnt = gen_event_smpl_rates(smpl_args, options.monitor_count, options.long_rates);
		if (cnt == -1) fatal_error("cannot set long periods\n");

		/*
		 * in case the short period rates were not specified, we copy them from the long period rates
		 */
		if (ovfl_args == 0) {
			memcpy(options.short_rates, options.long_rates, cnt*sizeof(unsigned long));
		}
		if (cnt) options.opt_use_smpl = 1;
	}

	if (ovfl_args) {
		/*
		 * in case not all rates are specified, they will default to zero, i.e. no sampling
		 * on this counter
		 */
		cnt = gen_event_smpl_rates(ovfl_args, options.monitor_count, options.short_rates);
		if (cnt == -1) fatal_error("cannot set short periods\n");
		/*
		 * in case the long period rates were not specified, we copy them from the short period rates
		 */
		if (smpl_args == 0) {
			memcpy(options.long_rates, options.short_rates, cnt*sizeof(unsigned long));
		}
		if (cnt) options.opt_use_smpl = 1;
	}

	if (options.opt_use_smpl) {
		if (options.smpl_output == NULL) 
			fatal_error("you must choose a sampling output format\n");

		if (options.smpl_output->validate) {
			if ((*options.smpl_output->validate)(evt) == -1) 
				fatal_error("cannot use sampling output format\n");
		}
		if (options.smpl_output->process_smpl == NULL)
			fatal_error("sampling output format %s has no process sampling buffer entry point\n", 
					options.smpl_output->name);

	}

	if (options.smpl_entries == 0) {
		options.smpl_entries = PFMON_DFL_SMPL_ENTRIES;
	}


#ifdef PFMON_DEBUG
	if (options.opt_debug) {
		int i;

		debug_print("sampling buffer size: %ld entries\n", options.smpl_entries);
		debug_print("long periods: ");
		for(i=0; i < options.monitor_count; i++) {
			debug_print("%lu ", -options.long_rates[i]);
		}
		debug_print("\nshort periods: ");
		for(i=0; i < options.monitor_count; i++) {
			debug_print("%lu ", -options.short_rates[i]);
		}
		debug_print("\n");
	}
#endif
}

/*
 * look for a matching sampling format.
 * The name and CPU model must match.
 *
 * if ignore_cpu is true, then we don't check if the host CPU matches
 *
 * XXX: we abuse libpfm error codes!
 */
int
pfmon_find_smpl_output(char *name, pfmon_smpl_output_t **fmt, int ignore_cpu)
{
	pfmon_smpl_output_t **p = smpl_outputs;
	unsigned long mask;
	int type;

	pfm_get_pmu_type(&type);
	mask = PFMON_PMU_MASK(type);
	while (*p) {	
		if (!strcmp(name, (*p)->name)) {
			if (ignore_cpu == 0 && ((*p)->pmu_mask & mask) == 0) return PFMLIB_ERR_BADHOST;
			*fmt = *p;
			return PFMLIB_SUCCESS;
		}
		p++;
	}
	return PFMLIB_ERR_NOTFOUND;
}

void
pfmon_list_smpl_outputs(void)
{
	pfmon_smpl_output_t **p = smpl_outputs;

	printf("supported sampling outputs: ");
	while (*p) {	
		printf("[%s] ", (*p)->name);
		p++;
	}
	printf("\n");
}

void
pfmon_smpl_output_info(pfmon_smpl_output_t *fmt)
{
	unsigned long m;
	char *name;
	int i;

	printf("Name        : %s\n"
	       "Description : %s\n",
		fmt->name,
		fmt->description);

	printf("PMU models  : ");
	m = fmt->pmu_mask;
	for(i=0; m; i++, m >>=1) {
		if (m & 0x1) {
			pfm_get_pmu_name_bytype(i, &name);
			printf("[%s] ", name);
		}
	}
	printf("\n");
}

