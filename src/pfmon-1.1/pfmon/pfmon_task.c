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
#include <setjmp.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <asm/ptrace_offsets.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

#define PSR_UP_BIT	2
#define PSR_DB_BIT	24

static jmp_buf jbuf;	/* setjmp buffer */
static int child_pid;	/* process id of signaling child */

static void
kill_task(pid_t pid)
{
	/*
	 * Not very nice but works
	 */
	kill(pid, SIGKILL);
}

static void
alarm_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	vbprintf("%lu second(s) timeout expired: killing command\n", options.session_timeout);

	/*
	 * XXX: should do something more gentle here
	 */
	kill(child_pid, SIGKILL);
}

static void
overflow_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	unsigned long mask =info->sy_pfm_ovfl[0];
	pfmon_smpl_ctx_t *csmpl = options.smpl_ctx;

	if (info->sy_code != PROF_OVFL) {
		printf("received spurious SIGPROF si_code=%d\n", info->sy_code);
		return;
	}

	if (csmpl->smpl_hdr == NULL) {
		warning("overflow handler but not sampling\n");
		return;
	}

	DPRINT(("overflow notification: pid=%d bv=0x%lx\n", info->sy_pid, info->sy_pfm_ovfl[0]));

	if ((mask>> PMU_FIRST_COUNTER) == 0UL) {
		warning("system wide overflow handler: empty mask\n");
		return;
	}

	process_smpl_buffer(csmpl);

#if 0
	for(i= PMU_FIRST_COUNTER; mask; mask >>=1, i++) {

		DPRINT(("mask=0x%lx i=%d\n", mask, i));

		/* 
		 * if we are sampling, process the buffer 
		 *
		 * pfmon does not use notification, unless it is sampling
		 */
		if ((mask & 0x1) != 0  && csmpl->smpl_hdr) {
			process_smpl_buffer(csmpl);
		}
	}
#endif
	if (perfmonctl(info->sy_pid, PFM_RESTART, 0, 0) == -1) {
		kill_task(info->sy_pid);
		fatal_error("overflow cannot restart process %d, aborting: %s\n", info->sy_pid, strerror(errno));
	}
}

static void
child_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	DPRINT(("SIGCHLD handler for %d code=%d\n", info->si_pid, info->si_code));

	/*
	 * stop the alarm, if any
	 */
	if (options.session_timeout) alarm(0);

	/*
	 * we need to record the child pid here because we need to avoid
	 * a race condition with the parent returning from fork().
	 * In some cases, the pid=fork() instruction is not completed before
	 * we come to the SIGCHILD handler. the pid variable still has its
	 * default (zero) value. That's because the signal was received on
	 * return from fork() by the parent.
	 * So here we keept track of who just died and use a global variable
	 * to pass it back to the parent.
	child_pid = info->si_pid;
	 */

	/*
	 * That's not very pretty but that's one way of avoiding a race
	 * condition with the pause() system call. You may deadlock if the 
	 * signal is delivered before the parent reaches the pause() call.
	 * Using a variable and test reduces the window but it still exists.
	 * longjmp/setjmp avoids it completely.
	 */
	siglongjmp(jbuf,1);
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
	if (options.opt_use_smpl) {
		csmpl->smpl_hdr = ctx->ctx_smpl_vaddr;
		DPRINT(("sampling buffer at %p\n", csmpl->smpl_hdr));
		csmpl->smpl_entry = &private_smpl_entry;
	}

	if (setup_sampling_output(csmpl) == -1) return -1;

	/* 
	 * back from signal handler?
	 */

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
	if (sigsetjmp(jbuf, 1) == 1) goto extract_results;

	setup_child_handler();

	/*
	 * detach the process, let it run free of ptrace()
	 */
	ptrace(PTRACE_DETACH, pid, NULL, NULL);

	/* 
	 * Block for child to finish
	 * The child process may already be done by the time we get here.
	 * The parent will skip to the next statement directly in this case.
	 */

	if (options.session_timeout) {
		alarm(options.session_timeout);
	}

	for(;;) pause();

extract_results:
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
	waitpid(pid, &status, 0);
end_of_exec:
	/* we may or may not want to check child exit status here */
	if (WEXITSTATUS(status) != 0) {
		warning("process %d exited with non zero value (%d): results may be incorrect\n", pid, WEXITSTATUS(status));
	}

	vbprintf("process %d exited with status %d\n", pid, WEXITSTATUS(status));

	/* dump results */
	if (pfmon_current->pfmon_print_results)
		pfmon_current->pfmon_print_results(pd, csmpl);
	else
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

