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
#include <signal.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

#define PFMON_DFL_SMPL_ENTRIES	2048

#ifdef CONFIG_PFMON_SMPL_FMT_RAW
extern pfmon_smpl_output_t raw_smpl_output;
#endif

#ifdef CONFIG_PFMON_SMPL_FMT_COMPACT
extern pfmon_smpl_output_t compact_smpl_output;
#endif

#ifdef CONFIG_PFMON_SMPL_FMT_DET_ITA
extern pfmon_smpl_output_t detailed_itanium_smpl_output;
#endif

#ifdef CONFIG_PFMON_SMPL_FMT_DET_ITA2
extern pfmon_smpl_output_t detailed_itanium2_smpl_output;
#endif

#ifdef CONFIG_PFMON_SMPL_FMT_BTB
extern pfmon_smpl_output_t btb_smpl_output;
#endif

#ifdef CONFIG_PFMON_SMPL_FMT_EXAMPLE
extern pfmon_smpl_output_t example_smpl_output;
#endif

static pfmon_smpl_output_t *smpl_outputs[]={

#ifdef CONFIG_PFMON_SMPL_FMT_DET_ITA
	&detailed_itanium_smpl_output,	/* must be first for Itanium */
#endif
#ifdef CONFIG_PFMON_SMPL_FMT_DET_ITA2
	&detailed_itanium2_smpl_output,	/* must be first for Itanium2 */
#endif
#ifdef CONFIG_PFMON_SMPL_FMT_RAW
	&raw_smpl_output,
#endif
#ifdef CONFIG_PFMON_SMPL_FMT_COMPACT
	&compact_smpl_output,
#endif
#ifdef CONFIG_PFMON_SMPL_FMT_BTB
	&btb_smpl_output,
#endif
#ifdef CONFIG_PFMON_SMPL_FMT_EXAMPLE
	&example_smpl_output,
#endif
	NULL
};


static void
print_smpl_output_header(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	char *name;
	FILE *fp = csmpl->smpl_fp;
	unsigned long msk;
	int i;

	print_standard_header(fp, options.opt_aggregate_res ? 
				  options.cpu_mask : csmpl->cpu_mask);

	if (hdr) {
		fprintf(fp, "#\n# kernel sampling format: %d.%d\n# sampling entry size: %lu\n", 
			PFM_VERSION_MAJOR(hdr->hdr_version), 
			PFM_VERSION_MINOR(hdr->hdr_version), 
			hdr->hdr_entry_size);
	}

	fprintf(fp, "#\n# recorded PMDs: ");

	for(i=0, msk =options.smpl_regs; msk; msk>>=1, i++) 
		if (msk & 0x1) fprintf(fp, "PMD%d ", i);

	fprintf(fp, "\n# sampling entries count: %lu\n", options.smpl_entries);

	fprintf(fp, "#\n# sampling rates (short/long): ");

	for(i=0; i < options.monitor_count-1; i++) {
		pfm_get_event_name(options.monitor_events[i], &name);
		if (options.short_rates[i] == 0 && options.long_rates[i] == 0) {
			fprintf(fp, "%s(none), ", name); 
		} else {
			fprintf(fp, "%s(%lu/%lu), ", 
				name,
				options.short_rates[i]*-1,
				options.long_rates[i]*-1);
		}
	}
	pfm_get_event_name(options.monitor_events[i], &name);
	if (options.short_rates[i] == 0 && options.long_rates[i] == 0) {
		fprintf(fp, "%s(none)\n#\n",  name);
	} else {
		fprintf(fp, "%s(%lu/%lu)\n#\n", 
			name,
			options.short_rates[i]*-1,
			options.long_rates[i]*-1);
	}

	/* 
	 * invoke additional header printing routine if defined
	 */
	if (options.smpl_output->print_header)
		(*options.smpl_output->print_header)(csmpl);

	fprintf(fp, "#\n#\n");
}

/*
 * id indicates which "context" to use. This is mostly used for system wide 
 * when more than one CPUs are being monitored.
 */
int
process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	if (csmpl == NULL || csmpl->smpl_hdr == NULL) return -1;

	return (*options.smpl_output->process_smpl)(csmpl);
}

int
setup_sampling_output(pfmon_smpl_ctx_t *csmpl)
{
        char filename[PFMON_MAX_FILENAME_LEN];
        FILE *fp = stdout;

	if (options.opt_use_smpl == 0) return 0;

        if (options.smpl_file) {
                if (options.opt_syst_wide && options.opt_aggregate_res == 0 && is_regular_file(options.smpl_file)) {
                        sprintf(filename, "%s.cpu%d", options.smpl_file, find_cpu(getpid()));
                } else {
                        strcpy(filename, options.smpl_file);
                }

                fp = fopen(filename, "w");
                if (fp == NULL) {
                        warning("cannot create sampling output file %s: %s\n", options.smpl_file, strerror(errno));
                        return -1;
                }
                vbprintf("results are in file \"%s\"\n",filename);

        }

        csmpl->smpl_fp = fp;

	if (options.opt_with_header) print_smpl_output_header(csmpl);

        return 0;
}

void
close_sampling_output(pfmon_smpl_ctx_t *csmpl)
{
	if (options.opt_use_smpl == 0) return;

	if (csmpl->smpl_fp && csmpl->smpl_fp != stdout) fclose(csmpl->smpl_fp);
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
		/*
		 * try to pick a default output format, if none specified by the user
		 */
		if (options.smpl_output == NULL) {
			if (pfmon_find_smpl_output(NULL, &options.smpl_output, 0) != PFMLIB_SUCCESS) 
				fatal_error("no sampling output format available for this PMU model\n");
		}

		vbprintf("using %s sampling output format\n", options.smpl_output->name);

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
	pfmon_smpl_output_t **p;
	unsigned long mask, gen_mask;
	int type;

	pfm_get_pmu_type(&type);
	mask     = PFMON_PMU_MASK(type);
	gen_mask = PFMON_PMU_MASK(PFMLIB_GENERIC_PMU);

	for(p = smpl_outputs; *p ; p++) {

		if (name == NULL || !strcmp(name, (*p)->name)) {
			if (ignore_cpu == 0 && (*p)->pmu_mask != gen_mask && ((*p)->pmu_mask & mask) == 0) {
				if (name) return PFMLIB_ERR_BADHOST;
				continue;
			}
			*fmt = *p;
			return PFMLIB_SUCCESS;
		}
	}
	return PFMLIB_ERR_NOTFOUND;
}

void
pfmon_list_smpl_outputs(void)
{
	pfmon_smpl_output_t **p = smpl_outputs;
	unsigned long mask, gen_mask;
	int type;

	pfm_get_pmu_type(&type);
	mask     = PFMON_PMU_MASK(type);
	gen_mask = PFMON_PMU_MASK(PFMLIB_GENERIC_PMU);

	printf("supported sampling outputs: ");
	while (*p) {	
		if ((*p)->pmu_mask == gen_mask || ((*p)->pmu_mask & mask)) printf("[%s] ", (*p)->name);
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

static void
kill_task(pid_t pid)
{
	/*
	 * Not very nice but works
	 */
	kill(pid, SIGKILL);
}

void
pfmon_process_smpl_buf(pfmon_smpl_ctx_t *csmpl, pid_t pid)
{
	sigset_t mask;

	sigemptyset(&mask);
	sigaddset(&mask, SIGCHLD);
	sigaddset(&mask, SIGALRM);
	
	sigprocmask(SIG_BLOCK, &mask, NULL);

	process_smpl_buffer(csmpl);

	if (perfmonctl(pid, PFM_RESTART, 0, 0) == -1) {
		if (options.opt_syst_wide == 0) kill_task(pid);
		fatal_error("overflow cannot restart monitoring, aborting: %s\n", strerror(errno));
	}

	sigprocmask(SIG_UNBLOCK, &mask, NULL);
}

