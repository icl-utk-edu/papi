/*
 * pfmon_task.c 
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
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/time.h>
#include <asm/ptrace_offsets.h>
#include <semaphore.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

#define PSR_UP_BIT	2
#define PSR_DB_BIT	24

static int child_pid;	/* process id of signaling child */
static int time_to_quit;
static sem_t ovfl_sem;
static unsigned long ovfl_cnt;

/*
 * XXX: we MUST NOT have any DPRINT(), warning, fatal_error() in ANY signal handler
 * as this is not threadsafe.
 */ 
static void
alarm_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	/*
	 * XXX: should do something more gentle here
	 */
	kill(child_pid, SIGKILL);
}

/*
 * pfmon only uses the overflow handler when sampling.
 */
static void
overflow_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	/* ignore spurious overflow interrupts */
	if (info->sy_code != PROF_OVFL) {
		return;
	}
	/* keep some statistics */
	ovfl_cnt++;

	/* 
	 * force processing of the sampling buffer upon return from the handler
	 */
	sem_post(&ovfl_sem);
}

static void
child_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	/*
	 * We are only interested in SIGCHLD indicating that the process is
	 * dead
	 */
	if (info->si_code != CLD_EXITED && info->si_code != CLD_KILLED) return;

	/*
	 * stop the alarm, if any
	 */
	if (options.session_timeout) alarm(0);

	time_to_quit = 1;

	sem_post(&ovfl_sem);
}

static void
setup_child_handler(void)
{
	struct sigaction act;

	memset(&act,0,sizeof(act));


	act.sa_handler = (sig_t)child_handler;
	sigaction (SIGCHLD, &act, 0);
}


static void
setup_overflow_handler(void)
{
	struct sigaction act;
	sigset_t my_set;

	memset(&act,0,sizeof(act));

	sigemptyset(&my_set);
	sigaddset(&my_set, SIGCHLD);
	sigaddset(&my_set, SIGALRM);

	act.sa_handler = (sig_t)overflow_handler;
	act.sa_mask    = my_set;
	sigaction (SIGPROF, &act, 0);
}

static void
setup_alarm_handler(void)
{
	struct sigaction act;

	memset(&act,0,sizeof(act));

	act.sa_handler = (sig_t)alarm_handler;
	sigaction (SIGALRM, &act, 0);
}


static void
install_trigger_address(pid_t pid)
{
	int r;

	vbprintf("trigger address is 0x%016lx\n", options.trigger_addr);

	r = set_code_breakpoint(pid, 0, options.trigger_addr);
	if (r == -1) 
		fatal_error("cannot set start address at 0x%lx for process [%d]\n", 
			    options.trigger_addr,
			    pid);

	/*
	 * set psr.db to enable breakpoints
	 */
	set_psr_bit(pid, PSR_DB_BIT, PSR_MODE_SET);
}

static void
clear_trigger_address(pid_t pid)
{
	/*
	 * clear psr.db to disable breakpoints
	 */
	set_psr_bit(pid, PSR_DB_BIT, PSR_MODE_CLEAR);
}

static int
do_measure_one_task(pfmlib_param_t *evt, pfarg_context_t *ctx, pfarg_reg_t *pc, int count, char **argv)
{
	pfarg_reg_t pd[PMU_MAX_PMDS];
	pfmon_smpl_ctx_t *csmpl = options.smpl_ctx;
	struct timeval time_start, time_end;
	struct rusage ru;
	pid_t mypid = getpid(), pid;
	unsigned long private_smpl_entry = 0UL;
	int trigger_mode = 0;
	int i, status;

	if (perfmonctl(mypid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == EBUSY) {
			fatal_error("concurrent conflicting monitoring session is present in your system\n");
		} else
			fatal_error("can't create PFM context: %s\n", strerror(errno));
	}

	sem_init(&ovfl_sem, 0, 0);

	if (options.opt_use_smpl) {
		csmpl->smpl_hdr = ctx->ctx_smpl_vaddr;
		DPRINT(("sampling buffer at %p\n", csmpl->smpl_hdr));
		csmpl->smpl_entry = &private_smpl_entry;
	}

	if (setup_sampling_output(csmpl) == -1) return -1;

	/* 
	 * back from signal handler?
	 */

	gettimeofday(&time_start, NULL);

	if ((pid= child_pid = fork()) == -1) fatal_error("cannot fork process\n");

	if (pid == 0) {		 
		/* child */
		pid = getpid();

		if (options.opt_verbose) {
			char **p = argv;
			printf("starting process [%d]: ", pid);
			while (*p) printf("%s ", *p++);
			printf("\n");
		}

		enable_pmu(pid);

		install_counters(pid, evt, pc, count);

		/*
		 * The use of ptrace() allows us to actually start monitoring after the exec()
		 * is done, i.e., when the new program is ready to go back to user mode for the
		 * "first time". Using this technique we ensure that the overhead of 
		 * setting up the protection + execvp() is not captured in the results. This
		 * can be important for short running programs.
		 */
		ptrace(PTRACE_TRACEME, 0, NULL, NULL);

		/*
		 * after this call, only the creator of the context, i.e. our parent here,
		 * can access the context. This ensures that the monitored program cannot
		 * mess up our session.
		 */
		protect_context(pid);

		execvp(argv[0], argv);

		fatal_error("child: cannot exec %s: %s\n", argv[0], strerror(errno));
		/* NOT REACHED */
	} 
trigger_restart:
	/* 
	 * wait for the child to exec 
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * the child exited: execvp() failed
	 */
	if (WIFSTOPPED(status) == 0) goto end_of_exec;

	if (options.trigger_addr_str) {
		if (trigger_mode == 0) {
			install_trigger_address(pid);
			trigger_mode = 1;
			ptrace(PTRACE_CONT, pid, NULL, NULL);
			goto trigger_restart;
		} 
		clear_trigger_address(pid);

		vbprintf("reached trigger address at 0x%016lx, enabling monitoring\n", options.trigger_addr);
	} 
	/*
	 * START the child process when it resumes
	 *
	 * set psr.up in the psr of the child
	 */
	set_psr_bit(pid, PSR_UP_BIT, PSR_MODE_SET);

	/*
	 * Now install the SIGCHLD handler to make sure we catch the end of the execution
	 * and collect the PMDS before a waitpid().
	 */
	setup_child_handler();

	/*
	 * detach the process, let it run free of ptrace()
	 */
	ptrace(PTRACE_DETACH, pid, NULL, NULL);

	if (options.session_timeout) {
		alarm(options.session_timeout);
	}

	for(;;) {
		sem_wait(&ovfl_sem);
		if (time_to_quit == 1) break;
		pfmon_process_smpl_buf(options.smpl_ctx, pid);
	}

	if (options.session_timeout) {
		alarm(0);
	}
	pid = child_pid; /* make sure we get the right child */

	memset(pd, 0, sizeof(pd));

	for(i=0; i < evt->pfp_count; i++) {
		pd[i].reg_num = pc[i].reg_num;
	}

	/*
	 * read the PMDS in the child's context. This is allowed because we are the creator.
	 * Also at this point we know the child is in zombie state, i.e. stable state.
	 */
	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt->pfp_count) == -1) {
		fatal_error("perfmonctl error READ_PMDS for process %d %s\n", pid, strerror(errno));
	}


	/* 
	 * We cannot issue this call BEFORE we read the PMD registers.
	 *
	 * Cleanup child now 
	 */
	wait4(pid, &status, 0, &ru);

	gettimeofday(&time_end, NULL);

end_of_exec:
	/* we may or may not want to check child exit status here */
	if (WEXITSTATUS(status) != 0) {
		warning("process %d exited with non zero value (%d): results may be incorrect\n", pid, WEXITSTATUS(status));
	}

	vbprintf("process %d exited with status %d\n", pid, WEXITSTATUS(status));

	if (options.opt_show_rusage) show_task_rusage(&time_start, &time_end, &ru);

	vbprintf("%lu sampling buffer overflows\n", ovfl_cnt);

	print_results(pd, csmpl);

	return 0;
}

int
measure_per_task(pfmlib_param_t *evt, pfarg_context_t *ctx, pfarg_reg_t *pc, int count, char **argv)
{
	setup_overflow_handler();
	setup_alarm_handler();
	return do_measure_one_task(evt, ctx, pc, count, argv);
}

