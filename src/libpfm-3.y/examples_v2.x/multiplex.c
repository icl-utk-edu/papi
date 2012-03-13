/*
 * multiplex2.c - example of kernel-level time-based or overflow-based event multiplexing
 *
 * Copyright (c) 2004-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
#include <syscall.h>
#include <getopt.h>
#include <signal.h>
#include <math.h>
#include <limits.h>
#include <fcntl.h>
#include <setjmp.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <sys/poll.h>
#include <sys/ptrace.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define MIN_FULL_PERIODS	2

#define MAX_EVT_NAME_LEN	128

#define MULTIPLEX_VERSION	"0.2"

#define SMPL_FREQ_IN_HZ	300

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

#define MAX_NUM_COUNTERS	NUM_PMDS
#define MAX_PMU_NAME_LEN	32

typedef struct {
	struct {
		int opt_plm;	/* which privilege level to monitor (more than one possible) */
		int opt_debug;	/* print debug information */
		int opt_verbose;	/* verbose output */
		int opt_us_format;	/* print large numbers with comma for thousands */
		int opt_ovfl_switch;	/* overflow-based switching */
		int opt_is_system;	/* use system-wide */
		int opt_intr_only;	/* interrupts only*/
		int opt_no_cmd_out;	/* redirect cmd output to /dev/null */
		int opt_no_header;	/* no header */
	} program_opt_flags;

	unsigned long	max_counters;	/* maximum number of counter for the platform */
	unsigned long	session_timeout;
	uint64_t	smpl_period;
	uint32_t	smpl_freq;

	unsigned long	cpu_mhz;

	pid_t		attach_pid;
	int		pin_cmd_cpu;
	int		pin_cpu;
	struct timespec switch_timeout;
} program_options_t;

#define opt_plm			program_opt_flags.opt_plm
#define opt_debug		program_opt_flags.opt_debug
#define opt_verbose		program_opt_flags.opt_verbose
#define opt_us_format		program_opt_flags.opt_us_format
#define opt_ovfl_switch		program_opt_flags.opt_ovfl_switch
#define opt_is_system		program_opt_flags.opt_is_system
#define opt_intr_only		program_opt_flags.opt_intr_only
#define opt_no_cmd_out		program_opt_flags.opt_no_cmd_out
#define opt_no_header		program_opt_flags.opt_no_header

typedef struct _event_set_t {
	struct _event_set_t	*next;
	unsigned short		id;
	unsigned int		n_events;
	unsigned int		pmcs_base;
	unsigned int		pmds_base;
	int			npmcs;
	int			npmds;
	unsigned long		set_runs;
	char			*event_str;
} event_set_t;

static program_options_t options;

static pfarg_pmc_t	*all_pmcs;
static pfarg_pmd_t	*all_pmds;
static uint64_t		*all_values;
static event_set_t	*current_set, *all_sets;

static unsigned int 	num_pmds, num_pmcs, num_sets, total_events;
static unsigned long	full_periods;
static volatile int	time_to_quit;
static jmp_buf jbuf;

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

/*
 * unreliable for CPU with variable clock speed
 */
static unsigned long
get_cpu_speed(void)
{
	FILE *fp1;	
	unsigned long f1 = 0, f2 = 0;
	char buffer[128], *p, *value;

	memset(buffer, 0, sizeof(buffer));

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return 0;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL)
			break;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			break;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp("cpu MHz", buffer, 7)) {
			float fl;
			sscanf(value, "%f", &fl);
			f1 = lroundf(fl);
			break;
		}
		if (!strncmp("BogoMIPS", buffer, 8)) {
			float fl;
			sscanf(value, "%f", &fl);
			f2 = lroundf(fl);
		}
	}
	fclose(fp1);
	return f1 == 0 ? f2 : f1;
}

/*
 * pin task to CPU
 */
#ifndef __NR_sched_setaffinity
#error "you need to define __NR_sched_setaffinity"
#endif

#define MAX_CPUS	2048
#define NR_CPU_BITS	(MAX_CPUS>>3)
int
pin_cpu(pid_t pid, unsigned int cpu)
{
	uint64_t my_mask[NR_CPU_BITS];

	if (cpu >= MAX_CPUS)
		fatal_error("this program supports only up to %d CPUs\n", MAX_CPUS);

	my_mask[cpu>>6] = 1ULL << (cpu&63);

	return syscall(__NR_sched_setaffinity, pid, sizeof(my_mask), &my_mask);
}

int
child(char **arg)
{
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

	if (options.pin_cmd_cpu != -1) {
		pin_cpu(getpid(), options.pin_cmd_cpu);
		vbprintf("command running on CPU core %d\n", options.pin_cmd_cpu);
	}

	if (options.opt_no_cmd_out) {
		close(1);
		close(2);
	}
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
		if (c == 3 && i>0) {
			str[b-++j] = sep;
			c = 0;
		}
	}
}

static void
print_results(void)
{
	unsigned int i, j, cnt;
	int ovfl_adj;
	uint64_t value, set_runs;
	event_set_t *e;
	char *p;
	char tmp1[32], tmp2[32], *str;
	char mtotal_str[32], *mtotal;
	char stotal_str[32], *stotal;

	if (full_periods < num_sets)
		fatal_error("not all sets have been activated, need to run longer %lu\n", full_periods);

	/*
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 *
	 */
	if (options.opt_no_header == 0) {
		printf("# %u Hz period = %u usecs\n# %"PRIu64" cycles @ %lu MHz\n", 
			options.smpl_freq, 
			1000000 / options.smpl_freq, 
			options.smpl_period,
			options.cpu_mhz);

		if (options.opt_ovfl_switch == 0)
			printf("# using time-based multiplexing\n"
				"# %uus effective switch timeout\n", 
				1000000 / options.smpl_freq);
		else
			printf("# using overflow-based multiplexing\n");

		if (options.opt_is_system)
			printf("# system-wide mode on CPU core %d\n",options.pin_cpu);

		printf("# %d sets\n", num_sets);
		printf("# %.2f average run per set\n", (double)full_periods/num_sets);
		printf("# set       measured total     #runs         scaled total event name\n");
		printf("# ------------------------------------------------------------------\n");
	}
	ovfl_adj= options.opt_ovfl_switch ? 1 : 0;

	for (i=0, e = all_sets, cnt = 0; i < num_sets; i++, e = e->next) {

		set_runs = e->set_runs;

		str = e->event_str;

		for(j=0; j < e->npmds-ovfl_adj; j++, cnt++) {

			value = all_values[j+e->pmds_base];

			sprintf(tmp1, "%"PRIu64, value);

			if (options.opt_us_format) {
				dec2sep(tmp1, mtotal_str, ',');
			} else {
				strcpy(mtotal_str, tmp1);
			}
			mtotal = mtotal_str;

			/* 
			 * scaling
			 */
			sprintf(tmp2, "%"PRIu64, ((value*full_periods)/set_runs));  

			if (options.opt_us_format) {
				dec2sep(tmp2, stotal_str, ',');
			} else {
				strcpy(stotal_str, tmp2);
			}
			stotal  = stotal_str;

			printf("  %03d %20s  %8"PRIu64" %20s %s\n",
				i,
				mtotal,
				set_runs,
				stotal,
				str);
			p = strchr(str, '\0');
			if (p)
				str = p+1;
		}
		/*
		 * skip first event
		 */
		if (options.opt_ovfl_switch) cnt++;
	}
}

static void
update_set(int ctxid)
{
	int count;
	int base;
	int ret;
	int i;

	base = current_set->pmds_base;

	/*
	 * we do not read the last counter (cpu_cycles) to avoid overwriting
	 * the reg_value field which will be used for next round
	 *
	 * We need to retry the read in case we get EBUSY because it means that
	 * the child task context is not yet available from inspection by PFM_READ_PMDS2.
	 *
	 */
	count = current_set->npmds;

	if (options.opt_ovfl_switch)
		count--;

	ret = pfm_read_pmds(ctxid, all_pmds + base, count);
	if (ret == -1)
		fatal_error("error reading set: %s\n", strerror(errno));

	/* update counts for this set */
	for (i=0; i < count; i++) {
		all_values[base+i] += all_pmds[base+i].reg_value;
		/* reset for next round */
		all_pmds[base+i].reg_value = 0UL;
	}
}
static void
switch_sets(int ctxid)
{
	update_set(ctxid);

	current_set = current_set->next;

	if (current_set == NULL)
		current_set = all_sets;

	current_set->set_runs++;

	vbprintf("starting set %d run %lu\n",
		current_set->id,
		current_set->set_runs);

	/*
	 * we must reprogram all avaibale PMCs (or PMDS) to ensure that no
	 * state is left over from the previous set and which could conflict
	 * on restart
	 */
	if (pfm_write_pmcs(ctxid, all_pmcs+current_set->pmcs_base, current_set->npmcs) == -1) {
		fatal_error("error writing pmcs: %s\n", strerror(errno));
	}

	if (pfm_write_pmds(ctxid, all_pmds+current_set->pmds_base, current_set->npmds) == -1) {
		fatal_error("error writing pmds: %s\n", strerror(errno));
	}

	full_periods++;

	if (options.opt_ovfl_switch && pfm_restart(ctxid) == -1) {
		if (errno != EBUSY)
			fatal_error("error pfm_restart: %s\n", strerror(errno));
		/*
		 * in case of EBUSY, it probably means the task has exited now
		 */
	}
}

static void
sigintr_handler(int sig)
{
	if (sig == SIGALRM) 
		time_to_quit = 1;
	else
		time_to_quit = 2;
	longjmp(jbuf, 1);
}

static void
sigchld_handler(int sig)
{
	time_to_quit = 1;
}

static int
measure_one_task(char **argv)
{
	int ctxid;
	pfarg_ctx_t ctx[1];
	pfarg_load_t load_arg;
	struct pollfd pollfd;
	pid_t pid;
	int status, ret;

	memset(ctx, 0, sizeof(ctx));
	memset(&load_arg, 0, sizeof(load_arg));

	/*
	 * create the context
	 */

	ctxid = pfm_create_context(ctx, NULL, NULL, 0);
	if (ctxid == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/*
	 * set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
	 * fd not visible to child.
	 */
	if (fcntl(ctxid, F_SETFD, FD_CLOEXEC))
		fatal_error("cannot set CLOEXEC: %s\n", strerror(errno));

	/*
	 * write registers for first set
	 */
	if (pfm_write_pmcs(ctxid, all_pmcs+current_set->pmcs_base, current_set->npmcs) == -1) {
		fatal_error("error pfm_write_pmcs: %s\n", strerror(errno));
	}

	if (pfm_write_pmds(ctxid, all_pmds+current_set->pmds_base, current_set->npmds) == -1) {
		fatal_error("error pfm_write_pmds: %s\n", strerror(errno));
	}
	/*
	 * now launch the child code
	 */
	if (options.attach_pid == 0) {
		if ((pid= fork()) == -1) fatal_error("Cannot fork process\n");
		if (pid == 0) exit(child(argv));
	} else {
		pid = options.attach_pid;
		ret = ptrace(PTRACE_ATTACH, pid, NULL, 0);
		if (ret) {
			fatal_error("cannot attach to task %d: %s\n",options.attach_pid, strerror(errno));
		}
	}

	ret = waitpid(pid, &status, WUNTRACED);
	if (ret < 0 || WIFEXITED(status))
		fatal_error("error command already terminated, exit code %d\n", WEXITSTATUS(status));

	vbprintf("child created and stopped\n");

	/*
	 * now attach the context
	 */
	load_arg.load_pid = pid;
	if (pfm_load_context(ctxid, &load_arg) == -1) {
		fatal_error("pfm_load_context error errno %d\n",errno);
	}

	current_set->set_runs = 1;

	/*
	 * start monitoring
	 */
	if (pfm_start(ctxid, NULL) == -1) {
		fatal_error("pfm_start error errno %d\n",errno);
	}

	ptrace(PTRACE_DETACH, pid, NULL, 0);

	if (setjmp(jbuf) == 1) {
		if (time_to_quit == 1) {
			printf("timeout expired\n");
		}
		if (time_to_quit == 2)
			printf("session interrupted\n");
		goto finish_line;
	}


	if (options.session_timeout) {
		printf("<monitoring for %lu seconds>\n", options.session_timeout);
		alarm(options.session_timeout);
	}
	pollfd.fd = ctxid;
	pollfd.events = POLLIN;
	pollfd.revents = 0;

	while(time_to_quit == 0) {
		/*
		 * mainloop. poll timeout is in msecs
		 */
		ret = poll(&pollfd, 1, 1000 / options.smpl_freq);
		switch(ret) {
			case  0:
				ret = ptrace(PTRACE_ATTACH, pid, NULL, 0);
				if (ret) {
					time_to_quit = 1;
					break;
				}

				ret = waitpid(pid, &status, WUNTRACED);
				/*
				 * exit with time_to_quit = 0
				 * to avoid unloading from dead thread
				 */
				if (WIFEXITED(status))
					goto finish_line;

				switch_sets(ctxid);

				ptrace(PTRACE_DETACH, pid, NULL, 0);
				break;

			case -1: fatal_error("poll error: %s\n", strerror(errno));

			default: /* we don't even read END_MSG */
				 time_to_quit = 1;
		}
	}
finish_line:
	/*
	 * cleanup after an alarm timeout
	 */
	if (time_to_quit) {
		/* stop monitored task */
		ptrace(PTRACE_ATTACH, pid, NULL, 0);
		waitpid(pid, NULL, WUNTRACED);

		/* detach context */
		pfm_unload_context(ctxid);
	}

	if (options.attach_pid == 0) {
		kill(pid, SIGKILL);
		waitpid(pid, &status, 0);
	} else {
		ptrace(PTRACE_DETACH, pid, NULL, 0);
	}

	if (time_to_quit < 2) print_results();

	close(ctxid);

	return 0;
}

static int
measure_one_cpu(char **argv)
{
	int ctxid, status;
	pfarg_ctx_t ctx[1];
	pfarg_load_t load_arg;
	struct pollfd pollfd;
	pid_t pid = 0;
	int ret, timeout;

	memset(ctx, 0, sizeof(ctx));
	memset(&load_arg, 0, sizeof(load_arg));

	if (options.pin_cpu == -1) {
		options.pin_cpu = 0;
		printf("forcing monitoring onto CPU core 0\n");
		pin_cpu(getpid(), 0);
	}

	ctx[0].ctx_flags = PFM_FL_SYSTEM_WIDE;
	/*
	 * create the context
	 */
	ctxid = pfm_create_context(ctx, NULL, NULL, 0);
	if (ctxid == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
	 * fd not visible to child.
	 */
	if (fcntl(ctxid, F_SETFD, FD_CLOEXEC))
		fatal_error("cannot set CLOEXEC: %s\n", strerror(errno));

	/*
	 * Now program the all the registers in one call
	 *
	 * Note that there is a limitation on the size of the argument vector
	 * that can be passed. It is usually set to a page size (16KB).
	 */
	if (pfm_write_pmcs(ctxid, all_pmcs+current_set->pmcs_base, current_set->npmcs) == -1)
		fatal_error("error: pfm_write_pmcs errno: %s\n", strerror(errno));

	/*
	 * initialize the PMD registers.
	 *
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(ctxid, all_pmds+current_set->pmds_base, current_set->npmds) == -1)
		fatal_error("pfm_write_pmds error errno %d\n", strerror(errno));

	/*
	 * now launch the child code
	 */
	if (*argv) {
		if ((pid = fork()) == -1) fatal_error("Cannot fork process\n");
		if (pid == 0) exit(child(argv));
	} 

	/*
	 * wait for the child to exec or be stopped
	 * We do this even in system-wide mode to ensure
	 * that the task does not start until we are ready
	 * to monitor.
	 */
	if (pid) {
		ret = waitpid(pid, &status, WUNTRACED);
		if (ret < 0 || WIFEXITED(status))
			fatal_error("error command already terminated, exit code %d\n", WEXITSTATUS(status));

		vbprintf("child created and stopped\n");
	}

	/*
	 * now attach the context
	 */
	load_arg.load_pid = options.pin_cpu;
	if (pfm_load_context(ctxid, &load_arg) == -1)
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * start monitoring
	 */
	if (pfm_start(ctxid, NULL) == -1)
		fatal_error("pfm_start error errno %d\n",errno);

	if (pid) {
		signal(SIGCHLD, sigchld_handler);
		ptrace(PTRACE_DETACH, pid, NULL, 0);
	}

	/*
	 * mainloop
	 */
	pollfd.fd = ctxid;
	pollfd.events = POLLIN;
	pollfd.revents = 0;

	timeout = options.opt_ovfl_switch ?  -1 : (1000 / options.smpl_freq);

	while (time_to_quit == 0) {
		ret = poll(&pollfd, 1, timeout);
		switch(ret) {
			case 1:
			case 0:
				/*
 				 *we are consuming the message.
				 * to avoid this phase we could use PFM_FL_OVFL_NO_MSG
				 * and use signal based notification
				 */
				if (options.opt_ovfl_switch) {
					ssize_t r;
					pfarg_msg_t msg;
					r = read(ctxid, &msg, sizeof(msg));
					(void) r;
				}
				switch_sets(ctxid);
				break;
			default: 
				if (errno != EINTR)
					fatal_error("poll fails\n");
		}
	}
	if (full_periods < MIN_FULL_PERIODS)
		fatal_error("Not enough periods (%lu) to print results\n", full_periods);

	if (pid)
		waitpid(pid, &status, 0);

	print_results();

	close(ctxid);

	return 0;
}


int
mainloop(char **argv)
{
	event_set_t *e;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_regmask_t impl_counters, used_pmcs;
	pfmlib_event_t cycle_event;
	unsigned int i, j;
	char *p, *str;
	unsigned int max_counters, allowed_counters;
	int ret;

	pfm_get_num_counters(&max_counters);

	if (max_counters < 2 && options.opt_ovfl_switch) 
		fatal_error("not enough counter to get overflow switching to work\n");

	allowed_counters = max_counters;

	/*
	 * account for overflow counter (cpu cycles)
	 */
	if (options.opt_ovfl_switch) allowed_counters--;

	memset(&used_pmcs, 0, sizeof(used_pmcs));
	memset(&impl_counters, 0, sizeof(impl_counters));

	pfm_get_impl_counters(&impl_counters);

	options.smpl_period = (options.cpu_mhz*1000000)/options.smpl_freq;

	vbprintf("%lu Hz period = %"PRIu64" cycles @ %lu Mhz\n", options.smpl_freq, options.smpl_period, options.cpu_mhz);

	for (e = all_sets; e; e = e->next) {
		for (p = str = e->event_str; p ; ) {
			p = strchr(str, ',');
			if (p) str = p +1;
			total_events++;
		}
	}

	/*
	 * account for extra event per set (cycle event)
	 */
	if (options.opt_ovfl_switch) {
		total_events += num_sets;
		/*
		 * look for our trigger event
		 */
		if (pfm_get_cycle_event(&cycle_event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find cycle event\n");
		}
	}

	vbprintf("total_events=%u\n", total_events);

	all_pmcs   = calloc(1, sizeof(pfarg_pmc_t)*total_events);
	all_pmds   = calloc(1, sizeof(pfarg_pmd_t)*total_events);
	all_values = calloc(1, sizeof(uint64_t)*total_events);

	if (all_pmcs == NULL || all_pmds == NULL || all_values == NULL)
		fatal_error("cannot allocate event tables\n");

	/*
	 * use the library to figure out assignments for all events of all sets
	 */
	for (i=0, e = all_sets; i < num_sets; i++, e = e->next) {

		memset(&inp,0, sizeof(inp));
		memset(&outp,0, sizeof(outp));

		/*
	 	 * build the pfp_unavail_pmcs bitmask by looking
	 	 * at what perfmon has available. It is not always
	 	 * the case that all PMU registers are actually available
	 	 * to applications. For instance, on IA-32 platforms, some
	 	 * registers may be reserved for the NMI watchdog timer.
	 	 *
	 	 * With this bitmap, the library knows which registers NOT to
	 	 * use. Of source, it is possible that no valid assignement may
	 	 * be possible if certina PMU registers  are not available.
	 	 */
		detect_unavail_pmcs(-1, &inp.pfp_unavail_pmcs);


		str = e->event_str;
		for(j=0, p = str; p && j < allowed_counters; j++) {

			p = strchr(str, ',');
			if (p) *p = '\0';

			if (pfm_find_full_event(str, &inp.pfp_events[j]) != PFMLIB_SUCCESS) {
				fatal_error("Cannot find %s event for set %d event %d\n", str, i, j);
			}
			if (p) {
				*p = ',';
				str = p + 1;
			}
		}
		if (p) {
			fatal_error("error in set %d: cannot have more than %d event(s) per set %s\n",
				    i,
				    allowed_counters,
				    options.opt_ovfl_switch ? "(overflow switch mode)": "(hardware limit)");
		}
		/*
		 * add the cycle event as the last event when we switch on overflow
		 */
		if (options.opt_ovfl_switch) {
			inp.pfp_events[j]   = cycle_event;
			inp.pfp_event_count = j+1;
			e->n_events	    = j+1;
		} else {
			e->n_events         = j;
			inp.pfp_event_count = j;
		}

		inp.pfp_dfl_plm = options.opt_plm;

		if (options.opt_is_system) 
			inp.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

		vbprintf("PMU programming for set %d\n", i);

		/*
		 * let the library do the hard work
		 */
		if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS) {
			fatal_error("cannot configure events for set %d: %s\n", i, pfm_strerror(ret));
		}
		e->id = i;

		e->pmcs_base = num_pmcs;
		e->pmds_base = num_pmds;

		/*
		 * propagate from libpfm to kernel data structures
		 */
		for (j=0; j < outp.pfp_pmc_count; j++, num_pmcs++) {
			all_pmcs[num_pmcs].reg_num   = outp.pfp_pmcs[j].reg_num;
			all_pmcs[num_pmcs].reg_value = outp.pfp_pmcs[j].reg_value;
		}
		for (j=0; j < outp.pfp_pmd_count; j++, num_pmds++)
			all_pmds[num_pmds].reg_num = outp.pfp_pmds[j].reg_num;

		e->npmcs  = num_pmcs - e->pmcs_base;
		e->npmds  = num_pmds - e->pmds_base;

		if (options.opt_ovfl_switch) {
			/*
			 * We do this even in system-wide mode to ensure
			 * that the task does not start until we are ready
			 * to monitor.
			 * setup the sampling period
			 */
			all_pmds[num_pmds-1].reg_value       = - options.smpl_period;
			all_pmds[num_pmds-1].reg_short_reset = - options.smpl_period;
			all_pmds[num_pmds-1].reg_long_reset  = - options.smpl_period;
			all_pmds[num_pmds-1].reg_flags = PFM_REGFL_OVFL_NOTIFY;
		}
		vbprintf("set%d pmc_base=%d pmd_base=%d npmcs=%d npmds=%d\n",
			e->id,
			e->pmcs_base,
			e->pmds_base,
			e->npmcs,
			e->npmds);
	}

	current_set = all_sets;

	signal(SIGALRM, sigintr_handler);
	signal(SIGINT, sigintr_handler);

	if (options.opt_is_system)
		return measure_one_cpu(argv);

	return measure_one_task(argv);
}

static struct option multiplex_options[]={
	{ "help", 0, 0, 1},
	{ "freq", 1, 0, 2 },
	{ "kernel-level", 0, 0, 3 },
	{ "user-level", 0, 0, 4 },
	{ "version", 0, 0, 5 },
	{ "set", 1, 0, 6 },
	{ "session-timeout", 1, 0, 7 },
	{ "attach-task", 1, 0, 8 },
	{ "pin-cmd", 1, 0, 9 },
	{ "cpu", 1, 0, 10 },

	{ "verbose", 0, &options.opt_verbose, 1 },
	{ "debug", 0, &options.opt_debug, 1 },
	{ "us-counter-format", 0, &options.opt_us_format, 1},
	{ "ovfl-switch", 0, &options.opt_ovfl_switch, 1},
	{ "system-wide", 0, &options.opt_is_system, 1},
	{ "no-cmd-output", 0, &options.opt_no_cmd_out, 1},
	{ "no-header", 0, &options.opt_no_header, 1},
	{ 0, 0, 0, 0}
};

static void
generate_default_sets(void)
{
	event_set_t *es, *tail = NULL;
	pfmlib_event_t events[2];
	size_t len;
	char *name;
	unsigned int i;
	int ret;
	
	ret = pfm_get_cycle_event(&events[0]);
	if (ret != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	ret = pfm_get_inst_retired_event(&events[1]);
	if (ret != PFMLIB_SUCCESS)
		fatal_error("cannot find instruction retired event\n");

	pfm_get_max_event_name_len(&len);

	for (i=0; i < 2; i++) {
		name = malloc(len+1);
		if (name == NULL) {
			fatal_error("cannot allocate space for event name\n");
		}
		pfm_get_full_event_name(&events[i], name, len+1);

		es = (event_set_t *)malloc(sizeof(event_set_t));
		if (es == NULL)
			fatal_error("cannot allocate new event set\n");

		memset(es, 0, sizeof(*es));

		es->event_str = name;
		es->next      = NULL;
		es->n_events  = 0;

		if (all_sets == NULL)
			all_sets = es;
		else
			tail->next = es;
		tail = es;
	}
	num_sets = i;
}

static void
print_usage(char **argv)
{
	printf("usage: %s [OPTIONS]... COMMAND\n", argv[0]);

	printf(	"-h, --help\t\t\t\tdisplay this help and exit\n"
		"-V, --version\t\t\t\toutput version information and exit\n"
		"-u, --user-level\t\t\tmonitor at the user level for all events\n"
		"-k, --kernel-level\t\t\tmonitor at the kernel level for all events\n"
		"-c, --us-counter-format\tprint large counts with comma for thousands\n"
		"-p pid, --attach-task pid\tattach to a running task\n"
		"--set=ev1[,ev2,ev3,ev4,...]\t\tdescribe one set\n"
		"--freq=number\t\t\t\tset set switching frequency in Hz\n"
		"-c cpu, --cpu=cpu\t\t\tCPU to use for system-wide [default current]\n"
		"--ovfl-switch\t\t\t\t\tuse overflow based multiplexing (default: time-based)\n"
		"--verbose\t\t\t\tprint more information during execution\n"
		"--system-wide\t\t\t\tuse system-wide (only one CPU at a time)\n"
		"--excl-idle\t\t\texclude idle task(system-wide only)\n"
		"--excl-intr\t\t\texclude interrupt triggered execution(system-wide only)\n"
		"--intr-only\t\t\tinclude only interrupt triggered execution(system-wide only)\n"
		"--session-timeout=sec\t\t\tsession timeout in seconds (system-wide only)\n"
		"--no-cmd-output\t\t\t\toutput of executed command redirected to /dev/null\n"
		"--pin-cmd=cpu\t\t\t\tpin executed command onto a specific cpu\n"
	);
}

int
main(int argc, char **argv)
{
	char *endptr = NULL;
	pfmlib_options_t pfmlib_options;
	event_set_t *tail = NULL, *es;
	unsigned long long_val;
	int c, ret;

	options.pin_cmd_cpu = options.pin_cpu = -1;

	while ((c=getopt_long(argc, argv,"+vhkuVct:p:", multiplex_options, 0)) != -1) {
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
			case 'V':
			case   5:
				printf("multiplex version " MULTIPLEX_VERSION " Date: " __DATE__ "\n"
					"Copyright (C) 2004 Hewlett-Packard Company\n");
				exit(0);
			case   6:
				es = (event_set_t *)malloc(sizeof(event_set_t));
				if (es == NULL) fatal_error("cannot allocate new event set\n");

				es->event_str = optarg;
				es->next      = NULL;
				es->n_events  = 0;

				if (all_sets == NULL)
					all_sets = es;
				else
					tail->next = es;
				tail = es;
				num_sets++;
				break;
			case 't':
			case   7:
				if (options.session_timeout) fatal_error("too many timeouts\n");
				if (*optarg == '\0') fatal_error("--session-timeout needs an argument\n");
			  	long_val = strtoul(optarg,&endptr, 10);
				if (*endptr != '\0') 
					fatal_error("invalid number of seconds for timeout: %s\n", optarg);

				if (long_val >= UINT_MAX) 
					fatal_error("timeout is too big, must be < %u\n", UINT_MAX);

				options.session_timeout = (unsigned int)long_val;
				break;
			case 'p':
			case   8:
				if (options.attach_pid) fatal_error("process to attach specified twice\n");
				options.attach_pid = (pid_t)atoi(optarg);
				break;
			case  9:
				if (options.pin_cmd_cpu != -1) fatal_error("cannot pin command twice\n");
				options.pin_cmd_cpu  = atoi(optarg);
				break;

			case  10:
				if (options.pin_cpu != -1) fatal_error("cannot pin to more than one cpu\n");
				options.pin_cpu  = atoi(optarg);
				break;
			default:
				fatal_error(""); /* just quit silently now */
		}
	}

	if (optind == argc && options.opt_is_system == 0 && options.attach_pid == 0) 
		fatal_error("you need to specify a command to measure\n");

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = options.opt_verbose; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);


	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	if ((options.cpu_mhz = get_cpu_speed()) == 0)
		fatal_error("can't get CPU speed\n");

	if (options.smpl_freq == 0UL)
		options.smpl_freq = SMPL_FREQ_IN_HZ;

	if (options.opt_plm == 0)
		options.opt_plm = PFM_PLM3;

	if (num_sets == 0)
		generate_default_sets();

	return mainloop(argv+optind);
}
