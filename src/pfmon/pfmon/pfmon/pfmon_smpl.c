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
	if (options.smpl_regs == 0UL) 
		fprintf(fp, "none");
	else
		for(i=0, msk =options.smpl_regs; msk; msk>>=1, i++) 
			if (msk & 0x1) fprintf(fp, "PMD%d ", i);

	fprintf(fp, "\n# sampling buffer entries: %lu\n", options.smpl_entries);

	fprintf(fp, "#\n# short sampling rates (base/mask/seed):\n");
	for(i=0; i < options.monitor_count; i++) {
		pfm_get_event_name(options.events[i].event, &name);

		if (options.short_rates[i].flags & PFMON_RATE_VAL_SET) {
			fprintf(fp, "#\t%s %lu",
				name,
				-options.short_rates[i].value);

			if (options.short_rates[i].flags & PFMON_RATE_MASK_SET) {
				fprintf(fp, "/0x%lx/%lu", 
					options.short_rates[i].mask,
					options.short_rates[i].seed);
			}
			fputc('\n', fp);
		} else {
			fprintf(fp, "#\t%s none\n", name); 
		}
	}

	fprintf(fp, "#\n# long sampling rates (base/mask/seed):\n");
	for(i=0; i < options.monitor_count; i++) {
		pfm_get_event_name(options.events[i].event, &name);

		if (options.long_rates[i].flags & PFMON_RATE_VAL_SET) {
			fprintf(fp, "#\t%s %lu",
				name,
				-options.long_rates[i].value);

			if (options.long_rates[i].flags & PFMON_RATE_MASK_SET) {
				fprintf(fp, "/0x%lx/%lu", 
					options.long_rates[i].mask,
					options.long_rates[i].seed);
			}
			fputc('\n', fp);
		} else {
			fprintf(fp, "#\t%s none\n", name); 
		}
	}

	/* 
	 * invoke additional header printing routine if defined
	 */
	if (options.smpl_output->print_header)
		(*options.smpl_output->print_header)(csmpl);

	fprintf(fp, "#\n#\n");
}

static int
pfmon_check_smpl_buf_bug(pfmon_smpl_ctx_t *csmpl)
{
	perfmon_smpl_hdr_t *hdr = csmpl->smpl_hdr;
	perfmon_smpl_entry_t *ent = (perfmon_smpl_entry_t *)(hdr+1);
	static unsigned long prev_stamp;
	static unsigned int  prev_cpu;

	if (PFM_VERSION_MAJOR(hdr->hdr_version) != PFM_VERSION_MAJOR(PFM_SMPL_VERSION)) {
		fatal_error("perfmon v%u.%u sampling format is not supported\n", 
				PFM_VERSION_MAJOR(hdr->hdr_version),
				PFM_VERSION_MINOR(hdr->hdr_version));
	}

	/* sanity check */
	if (hdr->hdr_pmds[0] != options.smpl_regs) {
		fatal_error("kernel did not record PMDs we were expecting 0x%lx(kernel) != 0x%lx\n", hdr->hdr_pmds, options.smpl_regs);
	}

	if (hdr->hdr_count == 0UL) return 0;

	/*
	 * 2.4.18 and 2.4.19 (as of 11/11/2002) do have a bug in the 
	 * sampling buffer which can cause the same set of entries to be
	 * reported twice. This only affects the last set of collected
	 * samples, just before the process terminates (goes into zombie state).
	 *
	 * The problem is that the last restart by pfmon did not take effect
	 * because the task was dying. As such when pfmon comes one last time
	 * to check for samples, it finds the last set again, because the buffer 
	 * index was not reset. This is the case for *non-blocking* sampling sessions.
	 * In such case, the reset of the index must be done by monitored task but
	 * here it never happens.
	 *
	 * There will be a fix in the next kernel release to add a field to
	 * help identify sets of samples. For now, we rely on the timestamp to 
	 * differentiate between new and old set of samples.
	 */
	if (ent->cpu == prev_cpu && ent->stamp == prev_stamp) {
		vbprintf("skipping identical set of samples\n"); 
		return -1;
	}
	prev_cpu   = ent->cpu;
	prev_stamp = ent->stamp;
	return 0;
}

/*
 * id indicates which "context" to use. This is mostly used for system wide 
 * when more than one CPUs are being monitored.
 */
int
process_smpl_buffer(pfmon_smpl_ctx_t *csmpl)
{
	if (csmpl == NULL || csmpl->smpl_hdr == NULL) return -1;

	if (pfmon_check_smpl_buf_bug(csmpl)) return 0;
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
setup_sampling_rates(pfmlib_param_t *evt, char *long_smpl_rates, char *short_smpl_rates, char *smpl_periods_random)
{
	int cnt = 0;

	if (smpl_periods_random && long_smpl_rates == NULL && short_smpl_rates == NULL)
		fatal_error("no short or long periods provided to apply randomization\n");

	if (smpl_periods_random && options.opt_has_random == 0)
		fatal_error("host kernel does not support randomization, you need 2.4.20 or 2.5.39 or higher\n");

	if (long_smpl_rates) {
		/*
		 * in case not all rates are specified, they will default to zero, i.e. no sampling
		 * on this counter
		 */
		cnt = gen_smpl_rates(long_smpl_rates, options.monitor_count, options.long_rates);
		if (cnt == -1) fatal_error("cannot set long sampling rates\n");

		/*
		 * in case the short period rates were not specified, we copy them from the long period rates
		 */
		if (short_smpl_rates == NULL) {
			memcpy(options.short_rates, options.long_rates, cnt*sizeof(pfmon_smpl_rate_t));
		}
		if (cnt) options.opt_use_smpl = 1;
	}

	if (short_smpl_rates) {
		/*
		 * in case not all rates are specified, they will default to zero, i.e. no sampling
		 * on this counter
		 */
		cnt = gen_smpl_rates(short_smpl_rates, options.monitor_count, options.short_rates);
		if (cnt == -1) fatal_error("cannot set short sampling rates\n");


		/*
		 * in case the long period rates were not specified, we copy them from the short period rates
		 */
		if (long_smpl_rates == NULL) {
			memcpy(options.long_rates, options.short_rates, cnt*sizeof(pfmon_smpl_rate_t));
		}
		if (cnt) options.opt_use_smpl = 1;
	}
	if (smpl_periods_random) {
		/*
		 * we place the masks/seeds into the long rates table. It is always defined
		 */
		gen_smpl_randomization(smpl_periods_random, options.monitor_count, options.long_rates);
		
		/* propagate  mask/seed to short rates */
		for(cnt = 0; cnt < options.monitor_count; cnt++) {
			options.short_rates[cnt].mask  = options.long_rates[cnt].mask;
			options.short_rates[cnt].seed  = options.long_rates[cnt].seed;
			options.short_rates[cnt].flags = options.long_rates[cnt].flags;

			if (options.long_rates[cnt].flags & PFMON_RATE_MASK_SET) {
				options.events[cnt].flags |= PFMON_MONITOR_RANDOMIZE;
			}
		}

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


	if (options.opt_verbose) {
		int i;

		vbprintf("sampling buffer size: %ld entries\n", options.smpl_entries);
		vbprintf("long sampling periods(val/mask/seed): ");
		for(i=0; i < options.monitor_count; i++) {
			vbprintf("%lu/0x%lx/%lu/0x%x ", 
				-options.long_rates[i].value,
				options.long_rates[i].mask,
				options.long_rates[i].seed,
				options.long_rates[i].flags);
		}
		vbprintf("\nshort sampling periods(val/mask/seed): ");
		for(i=0; i < options.monitor_count; i++) {
			vbprintf("%lu/0x%lx/%lu/0x%x ", 
				-options.short_rates[i].value,
				options.short_rates[i].mask,
				options.short_rates[i].seed,
				options.long_rates[i].flags);
		}
		vbprintf("\n");
	}
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

