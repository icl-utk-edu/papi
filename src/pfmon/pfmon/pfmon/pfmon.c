/*
 * pfmon.c 
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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <regex.h>
#include <errno.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"


#define PFMON_DFL_EVENT		"CPU_CYCLES"
#define PFMON_DFL_PLM		PFM_PLM3

#define PFMON_FORCED_GEN	"pfmon_gen"


extern pfmon_support_t pfmon_itanium2;
extern pfmon_support_t pfmon_itanium;
extern pfmon_support_t pfmon_generic;

static pfmon_support_t *pfmon_cpus[]={
#ifdef CONFIG_PFMON_ITANIUM2
	&pfmon_itanium2,
#endif
#ifdef CONFIG_PFMON_ITANIUM
	&pfmon_itanium,
#endif
#ifdef CONFIG_PFMON_GENERIC
	&pfmon_generic,		/* must always be last because matches any CPU */
#endif
	NULL
};

pfmon_support_t	*pfmon_current;		/* current pfmon support */

program_options_t options;		/* keep track of global program options */

/*
 * Does the pretty printing of results
 *
 * In the case where results are aggregated, the routine expect the results
 * in pd[] and will not generate CPU (or pid) specific filenames
 */
int
print_results(pfarg_reg_t *pd, pfmon_smpl_ctx_t *csmpl)
{
	int i, ret = 0;
	FILE *fp = NULL;
	char *name;
	char prefix[16];
	char counter_str[32];
	char filename[PFMON_MAX_FILENAME_LEN];

	if (options.opt_outfile) {
		if (options.opt_syst_wide && options.opt_aggregate_res == 0 && is_regular_file(options.opt_outfile)) {
			sprintf(filename, "%s.cpu%d", options.opt_outfile, find_cpu(getpid()));
		} else {
			strcpy(filename, options.opt_outfile);
		}
		fp = fopen(filename, options.opt_append ? "a" : "w");
		if (fp == NULL) {
			/*
			 * XXX: may be to revisit this one
			 */
			warning("cannot open %s for writing, defaulting to stdout\n", options.opt_outfile);
		}
	}

	if (fp == NULL)	{
		fp = stdout;
	} else {
		vbprintf("results are in file \"%s\"\n",filename);
	}	

	if (options.opt_use_smpl) {
		ret = process_smpl_buffer(csmpl);
		/*
		 * we do not print counts when sampling. Most of the time
		 * they are useless. The interesting data is in the sampling
		 * output
		 */
		if (ret || options.opt_smpl_print_counts == 0) return ret;
	}	

	if (options.opt_with_header) {
		print_standard_header(fp, options.opt_aggregate_res ? 
					  options.cpu_mask : csmpl->cpu_mask);
	}

	prefix[0] = '\0';
	if (options.opt_syst_wide) {
		if (fp == stdout && options.opt_aggregate_res == 0) {
			sprintf(prefix, "CPU%-2d", find_cpu(getpid()));
		} 
	} else {
		/* nothing for now */
	}

	for (i=0; i < options.monitor_count; i++) {

		pfm_get_event_name(options.events[i].event, &name);

		counter2str(pd[i].reg_value, counter_str);

		fprintf(fp, "%s %26s %s\n", prefix, counter_str, name);
	}

	if (fp && fp != stdout) fclose(fp);

	return ret;
}

/*
 * Executed in the context of the child
 */
int
install_counters(int pid, pfmlib_param_t *evt)
{
	pfarg_reg_t pd[PMU_MAX_PMDS];
	int i;

	memset(pd, 0, sizeof(pd));

	/*
	 * install initial value, long and short overflow rates
	 */
	for(i=0; i < evt->pfp_event_count; i++) {

		pd[i].reg_num         = evt->pfp_pc[i].reg_num;
		pd[i].reg_value       = options.long_rates[i].value;
		pd[i].reg_short_reset = options.short_rates[i].value;
		pd[i].reg_long_reset  = options.long_rates[i].value;
		pd[i].reg_random_mask = options.long_rates[i].mask;	/* mask is per monitor */
		pd[i].reg_random_seed = options.long_rates[i].seed;	/* seed is per monitor */

		vbprintf("[pmd%ld ival=0x%lx long_rate=0x%lx short_rate=0x%lx mask=0x%lx seed=%lu rand=%c]\n",
					pd[i].reg_num,
					pd[i].reg_value,
					pd[i].reg_long_reset,
					pd[i].reg_short_reset,
					pd[i].reg_random_mask,
					pd[i].reg_random_seed,
					evt->pfp_pc[i].reg_flags & PFM_REGFL_RANDOM ? 'y' : 'n');

	}

	/*
	 * Give the CPU model specific code a chance to review programming
	 */
	if (pfmon_current->pfmon_install_counters) {
		if (pfmon_current->pfmon_install_counters(pid, evt, pd) == -1) {
			warning("child: cannot install implementation specific counters\n");
			return -1;
		}

	}

	if (options.opt_debug) {
		for(i=0; i < evt->pfp_event_count; i++) {
			DPRINT(("install_counters: pmc%lu reset_pmds=0x%lx\n",
					evt->pfp_pc[i].reg_num,
					evt->pfp_pc[i].reg_reset_pmds[0]));
		}
	}

	/*
	 * now program the PMC registers
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt->pfp_pc, evt->pfp_pc_count) == -1) {
		warning("install_counters: perfmonctl error WRITE_PMCS %s\n",strerror(errno));
		return -1;
	}
	/*
	 * and the PMD registers
	 */
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt->pfp_event_count) == -1) {
		warning( "install_counters: perfmonctl error WRITE_PMDS %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

/*
 * Does the actual measurements
 */
static int
do_measurements(pfmlib_param_t *evt, char **argv)
{
	pfarg_context_t ctx[1];
	int i, ret;

	/*
	 * assign events to counters, configure additional PMCs
	 * count may be greater than pfp_count when non trivial features are used
	 * We are guaranteed that the first n entries contains the n counters
	 * specified on the command line. The optional PMCs always come after.
	 */
	ret = pfm_dispatch_events(evt);
	if (ret != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * in case we just want to check for a valid event combination, we
	 * exit here
	 */
	if (options.opt_check_evt_only) exit(0);

	if (options.opt_debug) {
		int j;
		DPRINT(("do_measurements: pc_count=%d event_count=%d\n", evt->pfp_pc_count, evt->pfp_event_count));
		for(j=0; j < evt->pfp_pc_count; j++) {
			DPRINT(("pc[%d].reg_num=%ld\n", j, evt->pfp_pc[j].reg_num));
		}
	}

	gen_reverse_table(evt, options.rev_pc);

	/* get a clean context request */
	memset(ctx, 0, sizeof(ctx));

	/*
	 * XXX: need to cleanup this
	 */
	if (options.opt_use_smpl) {
		for(i=0; i < evt->pfp_event_count; i++) {
			if (options.long_rates[i].flags & PFMON_RATE_VAL_SET) {

				evt->pfp_pc[i].reg_flags |= PFM_REGFL_OVFL_NOTIFY;

				if (options.events[i].flags & PFMON_MONITOR_RANDOMIZE)
					evt->pfp_pc[i].reg_flags |= PFM_REGFL_RANDOM;

			} else {
				options.smpl_regs |= 1UL << evt->pfp_pc[i].reg_num;
			}
		}
		ctx[0].ctx_smpl_entries = options.smpl_entries;
	}

	/*
	 * default setting for notification (can be overriden later)
	 */
	ctx[0].ctx_notify_pid   = getpid();

	ctx[0].ctx_smpl_regs[0] = options.smpl_regs;

	if (options.opt_syst_wide) {
		ctx[0].ctx_flags  = PFM_FL_SYSTEM_WIDE; 
		ctx[0].ctx_flags |= options.opt_excl_idle ? PFM_FL_EXCL_IDLE: 0;
	} else {
		ctx[0].ctx_flags |= options.opt_fclone ? PFM_FL_INHERIT_ALL : PFM_FL_INHERIT_ONCE;
	}
	ctx[0].ctx_flags |= options.opt_block ? PFM_FL_NOTIFY_BLOCK : 0;

	DPRINT(("ctx_smpl_entries=%lu ctx_smpl_regs=0x%lx ctx_flags=0x%x\n", 
				ctx[0].ctx_smpl_entries,
				ctx[0].ctx_smpl_regs[0],
				ctx[0].ctx_flags)); 

	return options.opt_syst_wide ? measure_system_wide(evt, ctx, argv) 
				     : measure_per_task(evt, ctx, argv);
}

static void
show_detailed_event_name(int i, char *name)
{
	unsigned long counters;
	int code;

	pfm_get_event_code(i, &code);
	pfm_get_event_counters(i, &counters);

	printf("%s code=0x%02x counters=0x%lx ", name, code, counters); 

	if (pfmon_current->pfmon_show_detailed_event_name) {
		(*pfmon_current->pfmon_show_detailed_event_name)(i);
	}
	putchar('\n');
}

/*
 * mode=0 : just print event name
 * mode=1 : print name + other information (some of which maybe model specific)
 */
static void
pfmon_list_all_events(char *pattern, int mode)
{
	char *name;
	regex_t preg;
	int i;

	if (pattern) {
		int done = 0;

		if (regcomp(&preg, pattern, REG_ICASE|REG_NOSUB)) {
			fatal_error("error in regular expression for event \"%s\"\n", pattern);
		}

		for(i=pfm_get_first_event(); i != -1; i = pfm_get_next_event(i)) {
			pfm_get_event_name(i, &name);
			if (regexec(&preg, name, 0, NULL, 0) == 0) {
				if (mode == 0) 
					printf("%s\n", name);
				else
					show_detailed_event_name(i, name);
				done = 1;
			}
		}
		if (done == 0) fatal_error("event not supported\n");
	} else {
		for(i=pfm_get_first_event(); i != -1; i = pfm_get_next_event(i)) {
			pfm_get_event_name(i, &name);
			if (mode == 0) 
				printf("%s\n", name);
			else
				show_detailed_event_name(i, name);

		}
	}
}

static struct option pfmon_common_options[]={
	{ "event-info", 1, 0, 1},
	{ "show-events", 2, 0, 2 },
	{ "kernel-level", 0, 0, 3 },
	{ "user-level", 0, 0, 4 },
	{ "events", 1, 0, 5 },
	{ "help", 0, 0, 6 },
	{ "version", 0, 0, 7 },
	{ "outfile", 1, 0, 8 },
	{ "long-show-events", 2, 0, 9 },
	{ "info", 0, 0, 10},
	{ "smpl-entries", 1, 0, 11},
	{ "smpl-outfile", 1, 0, 12},
	{ "long-smpl-periods", 1, 0, 13},
	{ "short-smpl-periods", 1, 0, 14},
	{ "cpu-mask", 1, 0, 15},
	{ "session-timeout", 1, 0, 16},
	{ "trigger-address", 1, 0, 17},
	{ "priv-levels", 1, 0, 18},
	{ "symbol-file", 1, 0, 19},
	{ "smpl-output-format", 1, 0, 20},
	{ "smpl-output-info", 1, 0, 21},
	{ "sysmap-file", 1, 0, 22},
	{ "smpl-periods-random", 1, 0, 23},
	{ "trigger-start-address", 1, 0, 24},
	{ "trigger-start-delay", 1, 0, 25},

	{ "verbose", 0, &options.opt_verbose, 1 },
	{ "append", 0, &options.opt_append, 1},
	{ "overflow-block",0, &options.opt_block, 1},
	{ "system-wide", 0, &options.opt_syst_wide, 1},
	{ "debug", 0, &options.opt_debug, 1 },
	{ "aggregate-results", 0, &options.opt_aggregate_res, 1 },

	{ "with-header", 0, &options.opt_with_header, 1},
	{ "us-counter-format",0, &options.opt_print_cnt_mode, 1},
	{ "eu-counter-format",0, &options.opt_print_cnt_mode, 2},
	{ "hex-counter-format",0, &options.opt_print_cnt_mode, 3},
	{ "show-time",0, &options.opt_show_rusage, 1},
	{ "check-events-only",0, &options.opt_check_evt_only, 1},
	{ "smpl-print-counts",0, &options.opt_smpl_print_counts, 1},
	{ "exclude-idle",0, &options.opt_excl_idle, 1},
	{ 0, 0, 0, 0}
};

static struct option *pfmon_cmd_options = pfmon_common_options;

static void
usage(char **argv)
{
	printf("usage: %s [OPTIONS]... COMMAND\n", argv[0]);

	printf(	"-h, --help\t\t\t\tdisplay this help and exit\n"
		"-V, --version\t\t\t\toutput version information and exit\n"
		"-l[regex], --show-events[=regex]\tdisplay all or a matching subset of the events\n"
		"--long-show-events[=regex]\t\tdisplay all or a matching subset of the events with info\n"
		"-i event, --event-info=event\t\tdisplay information about an event (numeric code or regex)\n"
		"-u, -3 --user-level\t\t\tmonitor at the user level for all events (default: on)\n"
		"-k, -0 --kernel-level\t\t\tmonitor at the kernel level for all events (default: off)\n"
		"-1\t\t\t\t\tmonitor execution at privilege level 1 (default: off)\n"
		"-2\t\t\t\t\tmonitor execution at privilege level 2 (default: off)\n"
		"-e, --events=ev1,ev2,...\t\tselect events to monitor (no space)\n"
		"-I,--info\t\t\t\tlist supported PMU models and compiled in sampling output formats\n"
		"-t secs, --session-timeout=secs\t\tduration of the system wide session in seconds\n"
		"-S format, --smpl-output-info=format\tdisplay information about a sampling output format\n"
		"--debug\t\t\t\t\tenable debug prints\n"
		"--verbose\t\t\t\tprint more information during execution\n"
		"--outfile=filename\t\t\tprint results in a file\n"
		"--append\t\t\t\tappend results to outfile\n"
		"--overflow-block\t\t\tblock the task when sampling buffer is full (default: off)\n"
		"--system-wide\t\t\t\tcreate a system wide monitoring session (default: off)\n"
		"--smpl-outfile=filename\t\t\tfile to save the sampling results\n"
		"--smpl-entries=val\t\t\tset number of entries for sampling buffer (default: %lu)\n"
		"--long-smpl-periods=val1,val2,...\tset sampling period after user notification\n"
		"--short-smpl-periods=val1,val2,...\tset sampling period\n"
		"--with-header\t\t\t\tgenerate a header for results\n"
		"--cpu-mask=0xn\t\t\t\tbitmask indicating on which CPU to start system wide monitoring\n"
		"--aggregate-results\t\t\taggregate counts and sampling buffer outputs for multi CPU monitoring\n"
		"--trigger-start-address=addr\t\tstart monitoring only when addr (code) is reached\n"
		"--priv-levels=lvl1,lvl2,...\t\tset privilege level per event (any combination of [0123uk]))\n"
		"--us-counter-format\t\t\tprint counters using commas (1,024)\n"
		"--eu-counter-format\t\t\tprint counters using points (1.024)\n"
		"--hex-counter-format\t\t\tprint counters in hexadecimal (0x400)\n"
		"--smpl-output-format=fmt\t\tselect fmt as sampling output format, use -L to list formats\n"
		"--show-time\t\t\t\tshow real,user, and system time for the command executed\n"
		"--symbol-file=filename\t\t\tELF image containing a symbol table\n"
		"--sysmap-file=filename\t\t\tSystem.map-format file containing a symbol table\n"
		"--check-events-only\t\t\tverify combination of events and exit (no measurement)\n"
		"--smpl-periods-random=mask1:seed1,...\tapply randomization to long and short periods\n"
		"--smpl-print-counts\t\t\tprint counters values when sampling session ends (default: no)\n"
		"--trigger-start-delay=secs\t\tnumber of seconds before activating monitoring\n"
		"--exclude-idle\t\t\t\texclude idle task from system wide sessions\n",
		PFMON_DFL_SMPL_ENTRIES
	);
}

int
pfmon_register_options(struct option *cmd, size_t sz)
{
	size_t dsz;
	char *o;
	static int options_done;

	/* let's make sure we do ont do it twice */
	if (options_done) return -1;

	if (cmd == NULL || sz == 0) {
		pfmon_cmd_options = pfmon_common_options;
		options_done = 1;
		return 0;
	}

	dsz = sizeof(pfmon_common_options)-sizeof(struct option);

	o = (char *)malloc(dsz+sz);
	if (o == NULL) return -1;

	memcpy(o, pfmon_common_options, dsz);
	memcpy(o+dsz, cmd, sz);

	pfmon_cmd_options = (struct option *)o;

	options_done = 1;

	return 0;
}


static void
pfmon_detect(void)
{
	pfmon_support_t **p = pfmon_cpus;
	int type;

	pfm_get_pmu_type(&type);

	while (*p) {
		if ((*p)->pmu_type == type) break;
		p++;
	}

	if (*p == NULL) fatal_error("no detected PMU support\n");

	pfmon_current = *p;

	vbprintf("pfmon will use %s PMU support\n", (*p)->name);
}

/*
 * We use the command name as the hint for forced generic
 * mode. We cannot use an option because, the command line 
 * options depends on the detected support.
 */
static int
check_forced_generic(char *cmd)
{
	char *p;

	p = strrchr(cmd, '/');
	if (p) cmd = p + 1;

	return strcmp(cmd, PFMON_FORCED_GEN) ? 0 : 1;
}

static void
pfmon_check_perfmon(void)
{
	pfarg_features_t ft;

	if (perfmonctl(0, PFM_GET_FEATURES, &ft, 1) == -1) {
		if (errno == ENOSYS) {
			fatal_error("host kernel does not have perfmon support\n");
		}
		fatal_error("you need at least kernel 2.4.18 or 2.5.3 to run this version of pfmon\n");
	}

	if (PFM_VERSION_MAJOR(ft.ft_version) !=  PFM_VERSION_MAJOR(PFM_VERSION)) {
		fatal_error("perfmon version mistmatch, must have at least %u.x\n", 
			    PFM_VERSION_MAJOR(PFM_VERSION));
	}

	options.pfm_version      = ft.ft_version;
	options.pfm_smpl_version = ft.ft_smpl_version;

	/*
	 * check support for randomization
	 */
	if (PFM_VERSION_MAJOR(options.pfm_version) > 1 ||   PFM_VERSION_MINOR(options.pfm_version) > 1) {
		options.opt_has_random = 1;
	}
	/*
	 * check support for idle exclusion 
	 */
	if (PFM_VERSION_MAJOR(options.pfm_version) > 1 ||   PFM_VERSION_MINOR(options.pfm_version) > 2) {
		options.opt_has_exclude = 1;
	}

}

static void
setup_trigger_address(void)
{
	unsigned long region;

	gen_code_range(options.trigger_saddr_str, &options.trigger_start_addr, NULL); 

	if (options.trigger_start_addr & 0xf) 
		fatal_error("trigger start address does not start on bundle boundary : 0x%lx\n", options.trigger_start_addr);

	region = options.trigger_start_addr >> 61;
	if (region > 4) 
		fatal_error("cannot specify a trigger start address inside the kernel : 0x%lx\n", options.trigger_start_addr);
}

static int
pfmon_print_event_info(char *event)
{
	char *name;
	regex_t preg;
	int done = PFMLIB_ERR_NOTFOUND;
	int i;

	if (isdigit(*event)) {
		done = pfm_print_event_info(event, printf);
		goto skip_regex;
	}
	if (regcomp(&preg, event, REG_ICASE|REG_NOSUB)) {
		fatal_error("error in regular expression for event \"%s\"\n", event);
	}

	for(i=pfm_get_first_event(); i != -1; i = pfm_get_next_event(i)) {
		pfm_get_event_name(i, &name);
		if (regexec(&preg, name, 0, NULL, 0) == 0) {
			pfm_print_event_info_byindex(i, printf);
			done = PFMLIB_SUCCESS;
		}
	}
skip_regex:
	if (done != PFMLIB_SUCCESS)
		fatal_error("event \"%s\" not found\n", event);

	return 0;
}


static void
pfmon_show_info(void)
{
	unsigned int version;

	print_simple_cpuinfo(stdout, "detected host CPUs: ");
	pfm_list_supported_pmus(printf);
	pfmon_list_smpl_outputs();
	pfm_get_version(&version);
	printf("pfmlib version: %u.%u\n", PFMLIB_MAJ_VERSION(version), PFMLIB_MIN_VERSION(version));
	printf("kernel perfmon version: %u.%u\n", PFM_VERSION_MAJOR(options.pfm_version), PFM_VERSION_MINOR(options.pfm_version));
}

static void
pfmon_propagate2libary(pfmlib_param_t *evt, int count, pfmon_monitor_t *events)
{
	int i;

	evt->pfp_event_count = count;
	for(i=0; i < count; i++) {
		evt->pfp_events[i].event = events[i].event;
		evt->pfp_events[i].plm   = events[i].plm;
	}
}

static void
pfmon_initialize(char **argv, pfmlib_param_t *evt)
{

	pfmon_check_perfmon();

	if (pfm_initialize() != PFMLIB_SUCCESS) fatal_error("cannot initialize library\n");

	if (check_forced_generic(argv[0]) && pfm_force_pmu(PFMLIB_GENERIC_PMU) != PFMLIB_SUCCESS)
		fatal_error("failed to force  generic mode (support may not be available).\n");

	pfmon_detect();
	pfmon_check_cpus();

	extract_pal_info(&options);

	if (pfmon_current->pfmon_initialize) 
		pfmon_current->pfmon_initialize(evt);

	load_config_file();
}

static void
setup_plm(pfmlib_param_t *evt)
{
	char *arg;
	int cnt=0, val;
	/*
	 * if not specified, force defaut priv level to PLM3 (user)
	 */
	if (options.opt_plm == 0) {
		options.opt_plm = PFMON_DFL_PLM;
		vbprintf("measuring at %s privilege level ONLY\n", PFMON_DFL_PLM == PFM_PLM3 ? "user" : "kernel");
	}

	/* set default privilege level: used when not explicitly specified for an event */
	evt->pfp_dfl_plm = options.opt_plm;

	/*
	 * set default priv level for all events
	 */
	for(cnt=0; cnt < evt->pfp_event_count; cnt++) {
		evt->pfp_events[cnt].plm = options.opt_plm;
		options.events[cnt].plm  = options.opt_plm;
	}
	if (options.priv_lvl_str == NULL) return;
	/*
	 * adjust using per-event settings
	 */
	for (cnt=0, arg = options.priv_lvl_str ; *arg; ) {
		if (cnt == evt->pfp_event_count) goto too_many;
		val = 0;
		while (*arg && *arg != ',') {
			switch (*arg) {
				case 'k':
				case '0': val |= 1; break;
				case '1': val |= 2; break;
				case '2': val |= 4; break;
				case '3': 
				case 'u': val |= 8; break;
				default: goto error;
			}
			arg++;
		}
		if (*arg) arg++;

		if (val) {
			evt->pfp_events[cnt].plm = val;
			options.events[cnt].plm  = val;
		}
		cnt++;
	}
	return;
error:
	fatal_error("unknown per-event privilege level %c ([ku0123])\n", *arg);
	/* no return */
too_many:
	fatal_error("too many per-event privilege levels specified, max=%d\n", evt->pfp_event_count);
	/* no return */
}

int
main(int argc, char **argv)
{
	pfmlib_param_t evt;	/* hold most configuration data */
	pfmlib_options_t pfmlib_options;
	char *endptr = NULL;
	char *long_smpl_args = NULL, *short_smpl_args = NULL, *smpl_random_args = NULL;
	pfmon_smpl_output_t *smpl_output;
	int c, r;

	memset(&evt, 0, sizeof(evt));
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));

	pfmon_initialize(argv, &evt);

	while ((c=getopt_long(argc, argv,"+0123kuvhe:Il::i:Vt:S:", pfmon_cmd_options, 0)) != -1) {
		switch(c) {
			case   0: continue; /* fast path for options */

			case 'v': options.opt_verbose = 1;
				  break;
			case   1:
			case 'i':
				exit(pfmon_print_event_info(optarg));
			case   2:
			case 'l':
				pfmon_list_all_events(optarg, 0);
				exit(0);
			case '1':
				options.opt_plm |= PFM_PLM1;
				break;
			case '2':
				options.opt_plm |= PFM_PLM2;
				break;
			case '3':
 			case   4:
 			case 'u':
 				options.opt_plm |= PFM_PLM3;
				break;
			case '0':
			case   3:
			case 'k':
				options.opt_plm |= PFM_PLM0;
				break;
			case   5:
			case 'e':
				if (evt.pfp_event_count) fatal_error("events already defined\n");
				options.monitor_count = gen_event_list(optarg, options.events);
				break;
			case   6:
			case 'h':
				usage(argv);
				if (pfmon_current->pfmon_usage) pfmon_current->pfmon_usage();
				exit(0);
			case 'V':
			case   7:
				printf("pfmon version " PFMON_VERSION " Date: " __DATE__ "\n"
					"Copyright (C) 2001-2002 Hewlett-Packard Company\n");
				exit(0);
			case   8:
				options.opt_outfile = optarg;
				break;
			case   9:
				pfmon_list_all_events(optarg, 1);
				exit(0);
			case  10:
			case 'I':
				pfmon_show_info();
				exit(0);
			case  11:
				if (options.smpl_entries) fatal_error("sampling entries specificed twice\n");
				options.smpl_entries = strtoul(optarg, &endptr, 0);
				if (*endptr != '\0') 
					fatal_error("invalid number of entries: %s\n", optarg);
				break;
			case  12:
				if (options.smpl_file) fatal_error("sampling output file specificed twice\n");
				options.smpl_file = optarg;
				break;
			case  13:
				/* need to wait until we have the events */
				if (long_smpl_args) fatal_error("long sampling rates specificed twice\n");
				long_smpl_args = optarg;
				break;
			case  14:
				/* need to wait until we have the events */
				if (short_smpl_args) fatal_error("short sampling rates specificed twice\n");
				short_smpl_args = optarg;
				break;
			case 15:
				options.cpu_mask = strtoul(optarg, &endptr, 16);
				if (*endptr != '\0') 
					fatal_error("invalid mask %s\n", optarg);
				break;
			case 't':
			case 16 :
				if (options.session_timeout) fatal_error("too many timeouts\n");
			  	options.session_timeout = strtoul(optarg,&endptr, 10);
				if (*endptr != '\0') 
					fatal_error("invalid number of seconds for timeout: %s\n", optarg);
				break;
			case 17 :
				fatal_error("--trigger-address is obsolete, use --trigger-start-address instead\n");
				break;

			case 18 :
				if (options.priv_lvl_str) fatal_error("per event privilege levels already defined");
				options.priv_lvl_str = optarg;
				break;
			case 19 :
				if (options.symbol_file) {
					if (options.opt_sysmap_syms)
						fatal_error("Cannot use --sysmap-file and --symbol-file at the same time\n");
					fatal_error("symbol file already defined\n");
				}
				if (*optarg == '\0') 
					fatal_error("you must provide a filename for --symbol-file\n");
				options.symbol_file = optarg;
				break;
			case 20:
				if (options.smpl_output)
					fatal_error("sampling output format already defined\n");

				r = pfmon_find_smpl_output(optarg, &options.smpl_output, 0);
				if (r != PFMLIB_SUCCESS)
					fatal_error("invalid sampling output format %s: %s\n", optarg, pfm_strerror(r));
				break;
			case 'S':
			case 21 : r = pfmon_find_smpl_output(optarg, &smpl_output, 1);
				  if (r != PFMLIB_SUCCESS)
					fatal_error("invalid sampling output format %s: %s\n", optarg, pfm_strerror(r));
				  pfmon_smpl_output_info(smpl_output);
				  exit(0);
			case 22 :
				if (options.symbol_file) {
					if (options.opt_sysmap_syms == 0)
						fatal_error("Cannot use --sysmap-file and --symbol-file at the same time\n");
					fatal_error("sysmap file already defined\n");
				}
				if (*optarg == '\0') 
					fatal_error("you must provide a filename for --sysmap-file\n");
				options.opt_sysmap_syms = 1;
				options.symbol_file = optarg;
				break;
			case 23 :
				/* need to wait until we have the events */
				if (smpl_random_args) fatal_error("randomization parameters specified twice\n");
				smpl_random_args = optarg;
				break;
			case 24 : 
				if (options.trigger_delay) fatal_error("cannot use a trigger address with a trigger delay\n");
				if (options.trigger_saddr_str) fatal_error("trigger start address specificed twice\n");
				options.trigger_saddr_str = optarg;
				break;
			case 25 :
				if (options.trigger_saddr_str) fatal_error("cannot use a trigger delay with a trigger address\n");
				if (options.trigger_delay) fatal_error("trigger start address specificed twice\n");
				options.trigger_delay = strtoul(optarg,&endptr, 10);
				if (*endptr != '\0') 
					fatal_error("invalid trigger delay : %s\n", optarg);
				break;

			default:
				if (pfmon_current->pfmon_parse_options == NULL ||
				     pfmon_current->pfmon_parse_options(c, optarg, &evt) == -1)
					fatal_error(""); /* just quit silently now */
		}
	}


	/*
	 * propagate debug option to library
	 */
	if (options.opt_debug) pfmlib_options.pfm_debug = 1;

	if (optind == argc && options.opt_syst_wide == 0 && options.opt_check_evt_only == 0)
		fatal_error("you need to specify a command to measure\n");

	if (options.opt_syst_wide == 0 && options.cpu_mask)
		warning("cpu-mask is currently ignored in this mode\n");

	if (options.opt_syst_wide) {
		unsigned long new_mask;
		unsigned long max_cpu_mask;

		max_cpu_mask = (1UL << options.online_cpus) - 1;

		new_mask = options.cpu_mask & max_cpu_mask;

		if (options.cpu_mask != new_mask && options.opt_verbose) 
			warning("cpu-mask truncated to 0x%lx\n", new_mask);

		if (options.cpu_mask != 0UL && new_mask == 0UL)
			fatal_error("invalid cpu-mask specified\n");

		options.cpu_mask = new_mask;

		if (options.cpu_mask == 0UL) options.cpu_mask = max_cpu_mask;

		if (optind != argc && options.session_timeout > 0)
			fatal_error("you cannot use both a timeout and command in system-wide mode\n");

		if (options.opt_excl_idle && options.opt_has_exclude == 0)
			fatal_error("your kernel does not have support for idle task exclusion, you need at least perfmon v1.3\n");
	} 
	/*
	 * try to use the command to get the symbols
	 * XXX: require absolute path
	 */
	if (options.symbol_file == NULL) options.symbol_file = argv[optind];

	/*
	 * make sure we do at least one measure
	 */
	if (options.monitor_count == 0) {

		if (pfm_find_event_byname(PFMON_DFL_EVENT, &options.events[0].event) != PFMLIB_SUCCESS)
			fatal_error("default event %s does not exist\n",PFMON_DFL_EVENT); 

		options.monitor_count  = 1;

		vbprintf("defaulting to event: %s\n", PFMON_DFL_EVENT);
	}

	/*
	 * propagate monitor parameters to library
	 */
	pfmon_propagate2libary(&evt, options.monitor_count, options.events);

	/*
	 * for system wide session, we force privileged monitors
	 */
	if (options.opt_syst_wide) {
		evt.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

		if (options.opt_block == 1) 
			fatal_error("cannot use blocking mode in system wide monitoring\n");

		if (options.trigger_saddr_str)
			fatal_error("cannot use a trigger start address in system wide mode\n");

	}

	DPRINT(("%s process id is %d\n", argv[0], getpid()));

	setup_plm(&evt);

		setup_sampling_rates(&evt, long_smpl_args, short_smpl_args, smpl_random_args);

	/* propagate verbosity to library */
	if (options.opt_verbose) {
		pfmlib_options.pfm_verbose = 1;
	}

	/* enable some debugging in the library as well */
	pfm_set_options(&pfmlib_options);

	/*
	 * check trigger address validity
	 */
	if (options.trigger_saddr_str) setup_trigger_address();

	/* used in sampling output header */
	options.argv    = argv;
	options.command = argv+optind;

	if (pfmon_current->pfmon_post_options && pfmon_current->pfmon_post_options(&evt) == -1) exit(1);

	do_measurements(&evt, argv+optind);

	return 0;
}
