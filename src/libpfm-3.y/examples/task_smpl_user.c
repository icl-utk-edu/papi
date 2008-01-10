/*
 * task_smpl_user.c - example of a task collecting a profile from user level
 *
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <stdarg.h>
#include <getopt.h>
#include <time.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <syscall.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#include "detect_pmcs.h"

#define SAMPLING_PERIOD	100000
#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

typedef struct {
	int opt_no_show;
	int opt_block;
	int opt_sys;
} options_t;

static uint64_t collected_samples;
static pfarg_pmd_t pd[NUM_PMDS];
static unsigned int num_pmds;
static options_t options;
static volatile int terminate;

static struct option the_options[]={
	{ "help", 0, 0,  1},
	{ "ovfl-block", 0, &options.opt_block, 1},
	{ "no-show", 0, &options.opt_no_show, 1},
	{ "system-wide", 0, &options.opt_sys, 1},
	{ 0, 0, 0, 0}
};

static void fatal_error(char *fmt,...) __attribute__((noreturn));

#define BPL (sizeof(uint64_t )<<3)
#define LBPL	6

static inline void pfm_bv_set(uint64_t *bv, uint16_t rnum)
{
	bv[rnum>>LBPL] |= 1UL << (rnum&(BPL-1));
}

static inline int pfm_bv_isset(uint64_t *bv, uint16_t rnum)
{
	return bv[rnum>>LBPL] & (1UL <<(rnum&(BPL-1))) ? 1 : 0;
}

static inline void pfm_bv_copy(uint64_t *d, uint64_t *j, uint16_t n)
{
	if (n <= BPL)
		*d = *j;
	else {
		memcpy(d, j, (n>>LBPL)*sizeof(uint64_t));
	}
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

static void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
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

int
child(char **arg)
{
	if (options.opt_sys) {
		printf("child pinned on CPU0\n");
		pin_cpu(getpid(), 0);
	}

	/*
	 * force the task to stop before executing the first
	 * user level instruction
	 */
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

	execvp(arg[0], arg);
	/* not reached */
	exit(1);
}

void
show_task_rusage(const struct timeval *start, const struct timeval *end, const struct rusage *ru)
{
	long secs, suseconds, end_usec;

	 secs     =  end->tv_sec - start->tv_sec;
	 end_usec = end->tv_usec;

	if (end_usec < start->tv_usec) {
      		end_usec += 1000000;
      		secs--;
    	}

  	suseconds = end_usec - start->tv_usec;

	printf ("real %ldh%02ldm%02ld.%03lds user %ldh%02ldm%02ld.%03lds sys %ldh%02ldm%02ld.%03lds\n", 
		secs / 3600, 
		(secs % 3600) / 60, 
		secs % 60,
		suseconds / 1000,

		ru->ru_utime.tv_sec / 3600, 
		(ru->ru_utime.tv_sec % 3600) / 60, 
		ru->ru_utime.tv_sec% 60,
		(long)(ru->ru_utime.tv_usec / 1000),

		ru->ru_stime.tv_sec / 3600, 
		(ru->ru_stime.tv_sec % 3600) / 60, 
		ru->ru_stime.tv_sec% 60,
		(long)(ru->ru_stime.tv_usec / 1000)
		);
}
static void
process_sample(int fd,  unsigned long ip, pid_t pid, pid_t tid, uint16_t cpu)
{
	unsigned int j;

	if (pfm_read_pmds(fd, pd, num_pmds))
		fatal_error("pfm_read_pmds error errno %d\n",errno);

	if (options.opt_no_show) goto done;

	printf("entry %"PRIu64" PID:%d TID: %d CPU:%u LAST_VAL: %"PRIu64" IIP:0x%lx\n",
		collected_samples,
		pid,
		tid,
		cpu,
		- pd[0].reg_last_reset_val,
		ip);

	for(j=1; j < num_pmds; j++) {	
		printf("PMD%-2d = %"PRIu64"\n", pd[j].reg_num, pd[j].reg_value);
	}
done:
	collected_samples++;

}

static void
cld_handler(int n)
{
	terminate = 1;
}

int
mainloop(char **arg)
{
	pfarg_ctx_t ctx;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_load_t load_args;
	struct timeval start_time, end_time;
	struct rusage rusage;
	pfarg_msg_t msg;
	uint64_t ovfl_count = 0;
	pid_t pid;
	int status, ret, fd;
	unsigned int i, num_counters;

	/*
	 * intialize all locals
	 */
	memset(&ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(pc, 0, sizeof(pc));
	memset(&load_args, 0, sizeof(load_args));


	pfm_get_num_counters(&num_counters);

	/*
	 * locate events
	 */
	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");

	i = 2;

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	printf("measuring at plm=0x%x\n",  inp.pfp_dfl_plm);

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;
	inp.pfp_flags       = options.opt_sys ? PFMLIB_PFP_SYSTEMWIDE : 0;

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

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of PMC to intialize.
	 * We use pfp_pmd_count to determine the number of PMD to initialize.
	 * Some events/features may cause extra PMCs to be used, leading to:
	 * 	- pfp_pmc_count may be >= pfp_event_count
	 * 	- pfp_pmd_count may be >= pfp_event_count
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i=0; i < outp.pfp_pmd_count; i++) {
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
		/*
		 * we also want to reset the other PMDs on
		 * every overflow. If we do not set
		 * this, the non-overflowed counters
		 * will be untouched.
		 */
		if (i)
			pfm_bv_set(pd[0].reg_reset_pmds, pd[i].reg_num);
	}

	/*
	 * we our sampling counter overflow, we want to be notified.
	 * The notification will come ONLY when the sampling buffer
	 * becomes full.
	 *
	 * We also activate randomization of the sampling period.
	 */
	pd[0].reg_flags	|= PFM_REGFL_OVFL_NOTIFY | PFM_REGFL_RANDOM;

	pd[0].reg_value       = - SAMPLING_PERIOD;
	pd[0].reg_short_reset = - SAMPLING_PERIOD;
	pd[0].reg_long_reset  = - SAMPLING_PERIOD;

	/*
	 * setup randomization parameters, we allow a range of up to +256 here.
	 */
	pd[0].reg_random_seed = 5;
	pd[0].reg_random_mask = 0xff;

	printf("programming %u PMCS and %u PMDS\n", outp.pfp_pmc_count, inp.pfp_event_count);

	/*
	 * prepare context structure.
	 */
	if (options.opt_sys) {
		if (options.opt_block)
			fatal_error("blocking mode not supported in system-wide\n");

		printf("system-wide monitoring on CPU0\n");

		pin_cpu(getpid(), 0);

		ctx.ctx_flags |= PFM_FL_SYSTEM_WIDE;
	}

	if (options.opt_block)
		ctx.ctx_flags  |= PFM_FL_NOTIFY_BLOCK;

	/*
	 * now create our perfmon context.
	 */
	fd = pfm_create_context(&ctx, NULL, NULL, 0);
	if (fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);
	/*
	 * initialize the PMDs
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(fd, pd, outp.pfp_pmd_count))
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	num_pmds = outp.pfp_pmd_count;

	signal(SIGCHLD, SIG_IGN);

	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1)
		fatal_error("Cannot fork process\n");

	/*
	 * In order to get the PFM_END_MSG message, it is important
	 * to ensure that the child task does not inherit the file
	 * descriptor of the context. By default, file descriptor
	 * are inherited during exec(). We explicitely close it
	 * here. We could have set it up through fcntl(FD_CLOEXEC)
	 * to achieve the same thing.
	 */
	if (pid == 0) {
		close(fd);
		child(arg);
	}

	/*
	 * wait for the child to exec
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * process is stopped at this point
	 */
	if (WIFEXITED(status)) {
		warning("task %s [%d] exited already status %d\n", arg[0], pid, WEXITSTATUS(status));
		goto terminate_session;
	}

	/*
	 * attach context to stopped task
	 */
	load_args.load_pid = options.opt_sys ? getpid() : pid;
	if (pfm_load_context(fd, &load_args))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * activate monitoring for stopped task.
	 * (nothing will be measured at this point
	 */
	if (pfm_start(fd, NULL))
		fatal_error("pfm_start error errno %d\n",errno);

	if (options.opt_sys)
		signal(SIGCHLD, cld_handler);
	/*
	 * detach child. Side effect includes
	 * activation of monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);
	gettimeofday(&start_time, NULL);

	/*
	 * core loop
	 */
	while(terminate == 0) {
		/*
		 * wait for overflow/end notification messages
		 */
		ret = read(fd, &msg, sizeof(msg));
		if (ret == -1) {
			if (errno != EINTR) fatal_error("cannot read perfmon msg: %s\n", strerror(errno));
			continue;
		}
		switch(msg.type) {
			case PFM_MSG_OVFL: /* one sample to process */
				process_sample(fd, msg.pfm_ovfl_msg.msg_ovfl_ip,
						   msg.pfm_ovfl_msg.msg_ovfl_pid,
						   msg.pfm_ovfl_msg.msg_ovfl_tid,
						   msg.pfm_ovfl_msg.msg_ovfl_cpu);
				ovfl_count++;
				if (pfm_restart(fd) == -1) {
					if (errno != EBUSY)
						fatal_error("pfm_restart error errno %d\n",errno);
				}
				break;
			case PFM_MSG_END: /* monitored task terminated (not for system-wide) */
				printf("task terminated\n");
				terminate = 1;
				break;
			default: fatal_error("unknown message type %d\n", msg.type);
		}
	}
terminate_session:
	/*
	 * cleanup child
	 */
	wait4(pid, &status, 0, &rusage);
	gettimeofday(&end_time, NULL);

	/*
	 * destroy perfmon context
	 */
	close(fd);

	printf("%"PRIu64" samples collected in %"PRIu64" buffer overflows\n", collected_samples, ovfl_count);
	show_task_rusage(&start_time, &end_time, &rusage);

	return 0;
}

static void
usage(void)
{
	printf("usage: task_smpl [-h] [--help] [--no-show] [--ovfl-block] cmd\n");
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	int c;

	while ((c=getopt_long(argc, argv,"h", the_options, 0)) != -1) {
		switch(c) {
			case 0: continue;

			case 1:
			case 'h':
				usage();
				exit(0);
			default:
				fatal_error("");
		}
	}
	if (argv[optind] == NULL) {
		fatal_error("You must specify a command to execute\n");
	}
	
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("Can't initialize library\n");
	}

	return mainloop(argv+optind);
}
