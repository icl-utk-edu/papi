/*
 * multiplex.c - example of user-level event multiplexing
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/poll.h>
#include <sys/wait.h>
#include <sys/ptrace.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#define MULTIPLEX_VERSION	"0.1"

#define MIN_FULL_PERIODS 100

#define SMPL_FREQ_IN_HZ	100

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

#define MAX_NUM_COUNTERS	32
#define MAX_PMU_NAME_LEN	32
typedef struct {
	struct {
		int opt_plm;	/* which privilege level to monitor (more than one possible) */
		int opt_debug;	/* print debug information */
		int opt_verbose;	/* verbose output */
		int opt_us_format;	/* print large numbers with comma for thousands */
	} program_opt_flags;

	unsigned long max_counters;	/* maximum number of counter for the platform */
	unsigned long smpl_freq;
	unsigned long smpl_period;

	unsigned long cpu_mhz;
	unsigned long full_periods;
} program_options_t;

#define opt_plm			program_opt_flags.opt_plm
#define opt_debug		program_opt_flags.opt_debug
#define opt_verbose		program_opt_flags.opt_verbose
#define opt_us_format		program_opt_flags.opt_us_format

typedef struct {
	char			*event_names[MAX_NUM_COUNTERS];
	pfmlib_input_param_t	pfm_inp;
	pfmlib_output_param_t	pfm_outp;
	pfarg_reg_t		pmcs[MAX_NUM_COUNTERS];
	pfarg_reg_t		pmds[MAX_NUM_COUNTERS];
	unsigned long		values[MAX_NUM_COUNTERS];
	unsigned long		n_runs;
	unsigned int		n_counters;
	unsigned int		n_pmcs;
} event_set_t;

typedef int	pfm_ctxid_t;

static pfm_ctxid_t ctxid;
static int current_set;
static program_options_t options;

/*
 * NO MORE THAN MAX_COUNTERS-1 (3) EVENTS PER SET
 */
static event_set_t events[]={
	{ {"BACK_END_BUBBLE_ALL","BACK_END_BUBBLE_L1D_FPU_RSE","BE_EXE_BUBBLE_ALL", },},
	{ {"BACK_END_BUBBLE_FE", "BACK_END_BUBBLE_L1D_FPU_RSE", "BE_RSE_BUBBLE_ALL",},},
	{ {"BE_L1D_FPU_BUBBLE_ALL", "BE_L1D_FPU_BUBBLE_L1D", "BE_EXE_BUBBLE_FRALL",},},
	{ {"BE_EXE_BUBBLE_GRALL", "BE_EXE_BUBBLE_GRGR", },},
	{ {"NOPS_RETIRED", "CPU_CYCLES", },}
};
#define N_SETS	(sizeof(events)/sizeof(event_set_t))

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
vbprintf(char *fmt, ...)
{
	va_list ap;

	if (options.opt_verbose == 0) return;

	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}

static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}


static unsigned long
get_cpu_speed(void)
{
	FILE *fp1;	
	unsigned long f = 0;
	char buffer[128], *p, *value;

	memset(buffer, 0, sizeof(buffer));

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return 0;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL) goto end;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL) goto end;	

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp("cpu MHz", buffer, 7)) {
			sscanf(value, "%lu", &f);
			goto end;
		}
	}
end:
	fclose(fp1);
	return f;
}


static void
update_set(pfm_ctxid_t ctxid, int set_idx)
{
	event_set_t *cset = events + set_idx;
	int count;
	int ret;
	int i;


	/*
	 * we do not read the last counter (cpu_cycles) to avoid overwriting
	 * the reg_value field which will be used for next round
	 *
	 * We need to retry the read in case we get EBUSY because it means that
	 * the child task context is not yet available from inspection by PFM_READ_PMDS.
	 *
	 */
	count = cset->n_counters - 1;

	ret = perfmonctl(ctxid, PFM_READ_PMDS, cset->pmds, count);
	if (ret == -1) {
		fatal_error("update_set reading set %d: %s\n", set_idx, strerror(errno));
	}

	/* update counts for this set */
	for (i=0; i < count; i++) {
		cset->values[i]        += cset->pmds[i].reg_value;
		cset->pmds[i].reg_value = 0UL; /* reset for next round */
	}
}


#if 0
static void
update_last_set(pfm_ctxid_t ctxid, int set_idx)
{
	event_set_t *cset = events + set_idx;
	unsigned long cycles;
	int i;

	/*
	 * this time we read ALL the counters (including CPU_CYCLES) because we
	 * need it to scale the last period
	 */
	if (perfmonctl(ctxid, PFM_READ_PMDS, cset->pmds, cset->n_counters) == -1) {
		fatal_error("update_last_set reading set %d\n", set_idx);
	}
	
	cycles = ~0UL - cset->pmds[cset->n_counters-1].reg_value;

	printf("last period = %4.1f%% of full period\n", (cycles*100.0)/options.smpl_period);

	/* this time we scale the value to the length of this last period */
	for (i=0; i < cset->n_counters-1; i++) {
		cset->values[i] += (cset->pmds[i].reg_value*cycles)/options.smpl_period;
	}
}
#endif

int
child(char **arg)
{
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

	execvp(arg[0], arg);
	/* not reached */

	exit(1);
}

static void
dec2sep(char *str2, char *str, char sep)
{
	int i, l, b, j, c=0;

	l = strlen(str2);
	if (l <= 3) {
		strcpy(str, str2);
		return;
	}
	b = l +  l /3 - (l%3 == 0); /* l%3=correction to avoid extraneous comma at the end */
	for(i=l, j=0; i >= 0; i--, j++) {
		if (j) c++;
		str[b-j] = str2[i];
		if (c == 3) {
			str[b-++j] = sep;
			c = 0;
		}
	}
}

static void
print_results(void)
{
	unsigned int i, j;
	event_set_t *e;
	char tmp1[32], tmp2[32];
	char mtotal_str[32], *mtotal;
	char stotal_str[32], *stotal;

       /*
 	* print the results
 	*
 	* It is important to realize, that the first event we specified may not
 	* be in PMD4. Not all events can be measured by any monitor. That's why
 	* we need to use the pc[] array to figure out where event i was allocated.
 	*
 	*/
	printf("%lu Hz period = %lu cycles @ %lu Mhz\n", options.smpl_freq, options.smpl_period, options.cpu_mhz);
	printf("%lu full periods\n", options.full_periods);
	printf("%lu event sets\n", N_SETS);
	printf("set        measured total     #runs         scaled total event name\n");
	printf("-------------------------------------------------------------------\n");

	for (i=0; i < N_SETS; i++) {
		e = events + i;
		for(j=0; j < e->n_counters-1; j++) {

			sprintf(tmp1, "%"PRIu64, e->values[j]);

			if (options.opt_us_format) {
				dec2sep(tmp1, mtotal_str, ',');
				mtotal = mtotal_str;
			} else {
				mtotal  = tmp1;
			}
			sprintf(tmp2, "%"PRIu64, (e->values[j]*options.full_periods)/e->n_runs);  /* stupid scaling */

			if (options.opt_us_format) {
				dec2sep(tmp2, stotal_str, ',');
				stotal = stotal_str;
			} else {
				stotal  = tmp2;
			}

			printf("%03d: %20s  %8"PRIu64" %20s %s\n",
				i,
				mtotal,
				e->n_runs,
				stotal,
				e->event_names[j]);
		}
	}
}

static void
switch_sets(void)
{
	event_set_t *cset;

	update_set(ctxid, current_set);
	current_set = (current_set+1) % N_SETS;


	cset        = events+current_set;
	cset->n_runs++;

	vbprintf("starting run %lu for set %d n_pmcs=%d pmd=%"PRIu64"\n",
		cset->n_runs, current_set, cset->n_pmcs,
		cset->pmds[cset->n_counters-1].reg_value);

	/*
	 * if one set as less events than another one, the left-over events will continue
	 * to count for nothing. That's fine because we will restore their values when
	 * the correspinding set is reloaded
	 */
	if (perfmonctl(ctxid, PFM_WRITE_PMCS, cset->pmcs, cset->n_pmcs) == -1) {
		fatal_error("overflow handler writing pmcs set %d : %d\n", current_set, errno);
	}

	if (perfmonctl(ctxid, PFM_WRITE_PMDS, cset->pmds, cset->n_counters) == -1) {
		fatal_error("overflow handler writing pmds set %d\n", current_set);
	}

	options.full_periods++;

	if (perfmonctl(ctxid, PFM_RESTART,NULL, 0) == -1) {
		perror("PFM_RESTART");
		exit(1);
	}

}

int
parent(char **arg)
{
	event_set_t *e;
	pfarg_context_t ctx[1];
	pfarg_load_t load_arg;
	event_set_t *cset;
	pfm_msg_t msg;
	struct pollfd ctx_pollfd;
	pfmlib_regmask_t impl_counters, used_pmcs;
	pfmlib_event_t cycle_event;
	unsigned int i, j, k, l,idx;
	int r, status, ret;
	unsigned int max_counters, allowed_counters;
	pid_t pid;

	pfm_get_num_counters(&max_counters);
	if (max_counters < 2) 
		fatal_error("not enough counter to do anything meaningful\n");

	allowed_counters = max_counters-1; /* reserve one slot for our sampling period */

	memset(&used_pmcs, 0, sizeof(used_pmcs));
	memset(&impl_counters, 0, sizeof(impl_counters));

	pfm_get_impl_counters(&impl_counters);

	memset(ctx, 0, sizeof(ctx));
	memset(&load_arg, 0, sizeof(load_arg));

	if (pfm_get_cycle_event(&cycle_event) != PFMLIB_SUCCESS) {
		fatal_error("Cannot find cycle event\n");
	}

	options.smpl_period = (options.cpu_mhz*1000000)/options.smpl_freq;

	vbprintf("%lu Hz period = %lu cycles @ %lu Mhz\n", options.smpl_freq, options.smpl_period, options.cpu_mhz);

	for (i=0; i < N_SETS; i++) {

		e = events+i;

		memset(&e->pfm_inp,0, sizeof(pfmlib_input_param_t));
		memset(&e->pfm_outp,0, sizeof(pfmlib_output_param_t));

		for(j=0; e->event_names[j] && j < allowed_counters; j++) {

			if (pfm_find_event(e->event_names[j], &idx) != PFMLIB_SUCCESS) {
				fatal_error("Cannot find %s event\n", e->event_names[j]);
			}
			e->pfm_inp.pfp_events[j].event = idx;
		}

		if (e->event_names[j]) {
			fatal_error("cannot have more than %d events per set (CPU_CYCLES uses 1 slot)\n", allowed_counters);
		}
		e->pfm_inp.pfp_events[j]   = cycle_event;
		e->pfm_inp.pfp_event_count = j+1;
		e->pfm_inp.pfp_dfl_plm     = options.opt_plm;

		e->n_pmcs     = j+1; 		/* used pmcs +1=sampling period */
		e->n_counters = j+1;		/* used pmd/pmc counter pairs  +1=sampling period */

		vbprintf("PMU programming for set %d\n", i);

		if ((ret=pfm_dispatch_events(&e->pfm_inp, NULL, &e->pfm_outp, NULL)) != PFMLIB_SUCCESS) {
			fatal_error("cannot configure events for set %d: %s\n", i, pfm_strerror(ret));
		}
		/*
		 * propagate from libpfm to kernel data structures
		 */
		for (j=0; j < e->n_counters; j++) {
			e->pmcs[j].reg_num   = e->pfm_outp.pfp_pmcs[j].reg_num;
			e->pmcs[j].reg_value = e->pfm_outp.pfp_pmcs[j].reg_value;

			e->pmds[j].reg_num   = e->pmcs[j].reg_num;

			pfm_regmask_set(&used_pmcs, e->pmcs[j].reg_num);
		}

		/* last counter contains our sampling counter */
		e->pmcs[j-1].reg_flags 	    |= PFM_REGFL_OVFL_NOTIFY;
		e->pmds[j-1].reg_value       = (~0) - options.smpl_period + 1;
		e->pmds[j-1].reg_short_reset = (~0) - options.smpl_period + 1;
		e->pmds[j-1].reg_long_reset  = (~0) - options.smpl_period + 1;

		for (j=0; j < e->n_counters-1; j++) {
			vbprintf("[pmd[%u]=0x%"PRIx64"/0x%"PRIx64"/0x%"PRIx64"]\n",
				e->pmds[j].reg_num,
				e->pmds[j].reg_value,
				e->pmds[j].reg_short_reset,
				e->pmds[j].reg_long_reset);
		}
		vbprintf("[pmd[%u]=0x%"PRIx64"/0x%"PRIx64"/0x%"PRIx64"]\n",
				e->pmds[j].reg_num,
				e->pmds[j].reg_value,
				e->pmds[j].reg_short_reset,
				e->pmds[j].reg_long_reset);

		/*
		 * we blank the unused pmcs to make sure every set uses all the counters, i.e.,
		 * cannot overflow due to some previous sampling periods that uses a counter
		 * beyond the number used by the current set
		 */
		for(j=0, k=e->n_pmcs, l=0; l < max_counters; j++) {
			if (pfm_regmask_isset(&impl_counters, j) == 0) continue;
			l++;
			if (pfm_regmask_isset(&used_pmcs, j)) continue;
			e->pmcs[k].reg_num   = j;
			e->pmcs[k].reg_value = 0UL;
			k++;
		}
		e->n_pmcs= k;
	}
	/*
	 * point to first set of counters
	 */
	current_set = 0;

	/*
	 * we block on counter overflow
	 */
	ctx[0].ctx_flags = PFM_FL_NOTIFY_BLOCK;

	/*
	 * attach the context to the task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * extract context id
	 */
	ctxid = ctx[0].ctx_fd;

	/*
	 * set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
	 * fd not visible to child.
	 */
	if (fcntl(ctxid, F_SETFD, FD_CLOEXEC))
		fatal_error("cannot set CLOEXEC: %s\n", strerror(errno));

	ctx_pollfd.fd     = ctxid;
	ctx_pollfd.events = POLLIN;

	cset  = events + current_set;
	cset->n_runs++;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(ctxid, PFM_WRITE_PMCS, cset->pmcs, cset->n_pmcs) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}

	/*
	 * initialize the PMDs
	 */
	if (perfmonctl(ctxid, PFM_WRITE_PMDS, cset->pmds, cset->n_counters) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * now launch the child code
	 */
	if ((pid= fork()) == -1) fatal_error("Cannot fork process\n");
	if (pid == 0) exit(child(arg));

	/*
	 * wait for the child to exec
	 */
	r = waitpid(pid, &status, WUNTRACED);

	if (r < 0 || WIFEXITED(status))
		fatal_error("error command already terminated, exit code %d\n", WEXITSTATUS(status));

	vbprintf("child created and stopped\n");

	/*
	 * the child is stopped, load context
	 */
	load_arg.load_pid = pid;
	if (perfmonctl(ctxid, PFM_LOAD_CONTEXT, &load_arg, 1) == -1) {
		fatal_error("perfmonctl error PFM_LOAD_CONTEXT errno %d\n",errno);
	}

	/*
	 * make sure monitoring will be activated when the execution is resumed
	 */
	if (perfmonctl(ctxid, PFM_START, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_START errno %d\n",errno);
	}

	/*
	 * resume execution
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	/*
	 * mainloop
	 */
	for(;;) {
		ret = read(ctxid, &msg, sizeof(msg));
		if (ret < 0) break;
		switch(msg.type) {
			case PFM_MSG_OVFL:
				switch_sets();
				break;
			case PFM_MSG_END:
				goto finish_line;
			default: printf("unknown message type %d\n", msg.type);
		}
	}
finish_line:

	if (options.full_periods < MIN_FULL_PERIODS) {
		fatal_error("Not enough periods (%lu) to print results\n", options.full_periods);
	}

	//update_last_set(pid, current_set);

	waitpid(pid, &status, 0);

	print_results();

	if (ctxid) close(ctxid);

	return 0;
}



static struct option multiplex_options[]={
	{ "help", 0, 0, 1},
	{ "freq", 1, 0, 2 },
	{ "kernel-level", 0, 0, 3 },
	{ "user-level", 0, 0, 4 },
	{ "version", 0, 0, 5 },

	{ "verbose", 0, &options.opt_verbose, 1 },
	{ "debug", 0, &options.opt_debug, 1 },
	{ "us-counter-format", 0, &options.opt_us_format, 1},
	{ 0, 0, 0, 0}
};

static void
print_usage(char **argv)
{
	printf("usage: %s [OPTIONS]... COMMAND\n", argv[0]);

	printf(	"-h, --help\t\t\t\tdisplay this help and exit\n"
		"-V, --version\t\t\t\toutput version information and exit\n"
		"-u, --user-level\t\t\tmonitor at the user level for all events\n"
		"-k, --kernel-level\t\t\tmonitor at the kernel level for all events\n"
		"-c, --us-counter-format\tprint large counts with comma for thousands\n"
		"--freq=number\t\t\t\tset sampling frequency in Hz\n"
		"--verbose\t\t\t\tprint more information during execution\n"
	);
}


int
main(int argc, char **argv)
{
	char *endptr = NULL;
	pfmlib_options_t pfmlib_options;
	int c, type;


	while ((c=getopt_long(argc, argv,"+vhkuVc", multiplex_options, 0)) != -1) {
		switch(c) {
			case   0: continue; /* fast path for options */

			case   1:
				  print_usage(argv);
				  exit(0);

			case 'v': options.opt_verbose = 1;
				  break;
			case  'c':
				  options.opt_us_format = 1;
				  break;
			case   2:
			case 'V':
				if (options.smpl_freq) fatal_error("sampling frequency set twice\n");
				options.smpl_freq = strtoul(optarg, &endptr, 10);
				if (*endptr != '\0')
					fatal_error("invalid freqyency: %s\n", optarg);
				break;
			case   3:
			case 'k':
				options.opt_plm |= PFM_PLM0;
				break;
			case   4:
			case 'u':
				options.opt_plm |= PFM_PLM3;
				break;
			case   5:
				printf("multiplex version " MULTIPLEX_VERSION " Date: " __DATE__ "\n"
					"Copyright (C) 2002 Hewlett-Packard Company\n");
				exit(0);
			default:
				fatal_error(""); /* just quit silently now */
		}
	}

	if (optind == argc) fatal_error("you need to specify a command to measure\n");

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("can't initialize library\n");
	}
	/*
	 * Let's make sure we run this on the right CPU family
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM2_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with %s PMU\n", model);
	}

	if ((options.cpu_mhz = get_cpu_speed()) == 0) {
		fatal_error("can't get CPU speed\n");
	}
	if (options.smpl_freq == 0UL) options.smpl_freq = SMPL_FREQ_IN_HZ;
	if (options.opt_plm == 0) options.opt_plm = PFM_PLM3;
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = options.opt_verbose; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	return parent(argv+optind);
}
