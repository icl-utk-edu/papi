/*
 * pfmon_system.c 
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
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/wait.h>

#include <perfmon/pfmlib.h>

#include "pfmon.h"

/*
 * argument passsed to each thread
 * pointer arguments are ALL read-only as they are shared
 * between all threads. To make modification, we need to make a copy first.
 */
typedef struct {
	int id;			/* thread index */
	int cpu;		/* which CPU to pin it on */
	pfarg_context_t ctx;	/* only private copy of context */
	pfmlib_param_t *evt;	/* read-only pfmlib_param */
	pfarg_reg_t *pc;	/* read-only copy of PMC to be written */
	int count;		/* how many PMC to write in pc */
} thread_arg_t;

#define SESSION_RUN   	0
#define SESSION_STOP  	1
#define SESSION_ABORTED	2

typedef struct _barrier {
	pthread_mutex_t mutex;
	pthread_cond_t	cond;
	unsigned long	counter;
	unsigned long	max;
} barrier_t;

#define THREAD_STARTED	0
#define THREAD_RUN	1
#define THREAD_DONE	2
#define THREAD_ERROR	3

static barrier_t barrier;
static int session_state;
static int who_must_print;

static pthread_mutex_t session_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  session_cond  = PTHREAD_COND_INITIALIZER;

static pthread_mutex_t results_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  results_cond  = PTHREAD_COND_INITIALIZER;
static pthread_key_t   param_key;

static int thread_state[PFMON_MAX_CTX];

static	pfarg_reg_t sys_pd[PMU_MAX_PMDS];

static unsigned long common_smpl_entry; /* next sampling entry to output (aggregate mode) */

static int
barrier_init(barrier_t *b, unsigned long count)
{
	int r;

	r = pthread_mutex_init(&b->mutex, NULL);
	if (r == -1) return -1;
	r = pthread_cond_init(&b->cond, NULL);
	if (r == -1) return -1;
	b->max = b->counter = count;

	return 0;
}

static void
cleanup_barrier(void *arg)
{
	int r;
	barrier_t *b = (barrier_t *)arg;
	r = pthread_mutex_unlock(&b->mutex);
	DPRINT(("free barrier mutex r=%d\n", r));
}


static int
barrier_wait(barrier_t *b)
{
	int r, oldstate;

	pthread_cleanup_push(cleanup_barrier, b);

	pthread_mutex_lock(&b->mutex);

	pthread_testcancel();

	if (--b->counter == 0) {
		DPRINT(("last thread entered\n"));

		/* reset barrier */
		b->counter = b->max;
		r = pthread_cond_broadcast(&b->cond);
	} else {
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldstate);

		while (b->counter != b->max) {
			DPRINT(("thread %d waiting\n", (int)pthread_self()));
			pthread_cond_wait(&b->cond, &b->mutex);
		}

		pthread_setcancelstate(oldstate, NULL);
	}
	pthread_mutex_unlock(&b->mutex);

	pthread_cleanup_pop(0);

	return 0;
}

static void
syswide_aggregate_results(pfarg_reg_t *pd)
{
	int i;

	for (i=0; i < options.monitor_count; i++) {
		sys_pd[i].reg_value += pd[i].reg_value;
	}
}

static int
do_measure_one_cpu(void *data)
{
	thread_arg_t *arg = (thread_arg_t *)data;
	pfarg_reg_t pd[PMU_MAX_PMDS];
	pfarg_context_t *ctx = &arg->ctx;
	pfmon_smpl_ctx_t *csmpl;
	unsigned long private_smpl_entry = 0UL;
	pid_t mypid = getpid();
	int i;

	pthread_setspecific(param_key, arg);

	/* locate private sampling context */
	csmpl = options.smpl_ctx+arg->id;

	/*
	 * provide the pinning information
	 *
	 * XXX: this should really go into a separate system call (maybe prctl(2))
	 *
	 * Here we modify on private copy of the context.
	 */
	ctx->ctx_cpu_mask   = csmpl->cpu_mask = 1UL << arg->cpu;
	ctx->ctx_notify_pid = mypid;

	//if (arg->cpu == 1) sleep(2);

	DPRINT(("[%d] before CPU%d must be on CPU%d mask=0x%lx\n", 
		getpid(), find_cpu(getpid()), arg->cpu, ctx->ctx_cpu_mask));

	if (perfmonctl(mypid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == EBUSY) {
			warning("Concurrent conflicting monitoring session is present in your system\n");
		} else {
			warning("Can't create PFM context: %s\n", strerror(errno));
		}
		goto error;
	}

	DPRINT(("[%d] after CPU%d\n", getpid(), find_cpu(getpid())));

	if (find_cpu(getpid()) != arg->cpu) {
		warning("Thread does not run on correct CPU: %d instead of %d\n",
		       find_cpu(getpid()), arg->cpu);
		goto error;
	}

	if (options.opt_use_smpl) {
		if (options.opt_aggregate_res) {
			csmpl->smpl_entry = &common_smpl_entry;
		} else {
			csmpl->smpl_entry = &private_smpl_entry;
			if (setup_sampling_output(csmpl) == -1) goto error;
		}
		csmpl->smpl_hdr = ctx->ctx_smpl_vaddr;

		DPRINT(("sampling buffer at %p\n", csmpl->smpl_hdr));
	}


	if (enable_pmu(mypid) == -1) goto error;

	if (install_counters(mypid, arg->evt, arg->pc, arg->count) == -1) goto error;

	thread_state[arg->id] = THREAD_RUN;

	barrier_wait(&barrier);

	DPRINT(("[%d] after barrier\n", mypid));

	if (session_state == SESSION_ABORTED) goto error;

	vbprintf("starting system wide session on CPU%d\n", arg->cpu);

	if (session_start(mypid) == -1) goto error;

	pthread_mutex_lock(&session_mutex);
	while (session_state != SESSION_STOP) {
		if (session_state == SESSION_ABORTED) goto error;
		pthread_cond_wait(&session_cond, &session_mutex);
	}
	pthread_mutex_unlock(&session_mutex);

	DPRINT(("[%d] seen SESSION_STOP\n", mypid));

	if (session_stop(mypid) == -1) goto error;

	vbprintf("stopped system wide session on CPU%d\n", arg->cpu);

	memset(pd, 0, sizeof(pd));

	/*
	 * we use pfp_count and not count to capture only the counters
	 * and not any other PMC. The counters are GUARANTEED to be at the beginning
	 */
	for(i=0; i < arg->evt->pfp_count; i++) {

		pd[i].reg_num = arg->pc[i].reg_num;

			DPRINT(("will read: pd[%d].reg_num=%ld\n", i, pd[i].reg_num));
	}

	/*
	 * read the PMDS now using our context
	 */
	if (perfmonctl(mypid, PFM_READ_PMDS, pd, arg->evt->pfp_count) == -1) {
		warning("perfmonctl error READ_PMDS for process %d %s\n", mypid, strerror(errno));
		goto error;
	}
	DPRINT(("[%d] read PMDS\n", mypid));

	/* 
	 * dump results 
	 */
	if (options.opt_aggregate_res) {
		pthread_mutex_lock(&results_mutex);
		syswide_aggregate_results(pd);
		pthread_mutex_unlock(&results_mutex);
	} else {
		/*
		 * the use of a conditional variable ensures
		 * that results are printed in the natural
		 * order: cpu0, cpu1, ....
		 */
		pthread_mutex_lock(&results_mutex);
		while (who_must_print != arg->id) {
			pthread_cond_wait(&results_cond, &results_mutex);
		}
		print_results(pd, csmpl);
		who_must_print++;
		pthread_cond_broadcast(&results_cond);
		pthread_mutex_unlock(&results_mutex);
	}

	if (options.opt_use_smpl && options.opt_aggregate_res == 0) 
		close_sampling_output(csmpl);

	thread_state[arg->id] = THREAD_DONE;
	pthread_exit((void *)(0UL));
error:
	close_sampling_output(csmpl);
	thread_state[arg->id] = THREAD_ERROR;
	pthread_exit((void *)(~0UL));
}

/*
 * We cannot use any of the stdio routine during the execution of the handler because
 * they are not lock-safe with regards to ASYNC signals.
 */
static void
syswide_overflow_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	unsigned long mask =info->sy_pfm_ovfl[0];
	thread_arg_t *arg = (thread_arg_t *)pthread_getspecific(param_key);
	pfmon_smpl_ctx_t *csmpl = options.smpl_ctx+arg->id;

	if (info->sy_code != PROF_OVFL) {
		warning("Received spurious SIGPROF si_code=%d\n", info->sy_code);
		return;
	} 

	if (csmpl->smpl_hdr == NULL) {
		warning("overflow handler but not sampling on CPU%d\n", arg->cpu);
		return;
	}

	safe_fprintf(1, "Overflow in thread %d [%d] from CPU%d on CPU%d mask=0x%lx\n", 
		info->sy_pid, 
		getpid(),
		arg->cpu, 
		find_cpu(info->sy_pid),
		info->sy_pfm_ovfl[0]);

	if ((mask>> PMU_FIRST_COUNTER) == 0UL) {
		warning("system wide overflow handler: empty mask on CPU%d\n", arg->cpu);
		return;
	}

	if (options.opt_aggregate_res) pthread_mutex_lock(&results_mutex);
	process_smpl_buffer(csmpl);
	if (options.opt_aggregate_res) pthread_mutex_unlock(&results_mutex);
#if 0	
	for(i= PMU_FIRST_COUNTER; mask; mask >>=1, i++) {

		if (options.opt_verbose) {
			char *name;
			pfm_get_event_name(options.monitor_events[options.rev_pc[i]], &name);
			printf("Overflow on PMD%d %s\n", i,  name);
		}

		/* 
		 * if we are sampling, process the buffer 
		 *
		 * pfmon does not use notification, unless it is sampling
		 */
		if ((mask & 0x1) != 0  && csmpl->smpl_hdr) {
			pthread_mutex_lock(&results_mutex);
			process_smpl_buffer(csmpl);
			pthread_mutex_unlock(&results_mutex);
		}
	}
#endif

	if (perfmonctl(info->sy_pid, PFM_RESTART, 0, 0) == -1) {
		fatal_error("overflow cannot restart process %d: %d\n", info->sy_pid, errno);
	}
}

static void
setup_signals(void)
{
	struct sigaction act;
	sigset_t my_set;

	/* Install SIGCHLD handler */
	memset(&act,0,sizeof(act));

	sigemptyset(&my_set);
	sigaddset(&my_set, SIGCHLD);

	act.sa_handler = (sig_t)syswide_overflow_handler;
	act.sa_mask    = my_set;
	sigaction (SIGPROF, &act, 0);
}

static __inline__ int
hweight64 (unsigned long x)
{
	unsigned long result;
#ifdef __GNUC__
	__asm__ ("popcnt %0=%1" : "=r" (result) : "r" (x));
#elif defined(INTEL_ECC_COMPILER)
	result = _m64_popcnt(x);
#else
#error "you need to provide inline assembly from your compiler"
#endif
	return (int)result;
}

static void
exit_system_wide(int i)
{
	pthread_exit((void *)((unsigned long)i));
}

static int
delimit_session(char **argv)
{
	pid_t pid;
	int status;

	/*
	 * take care of the easy case first: no command to start
	 */
	if (argv == NULL || *argv == NULL) {
		/*
		 * this will start the session in each "worker" thread
		 */
		barrier_wait(&barrier);

		if (options.session_timeout) {
			printf("<Session to end in %lu seconds>\n", options.session_timeout);
			sleep(options.session_timeout);
		} else {
			printf("<Press ENTER to stop session>\n");
			getchar();
		}
		return 0;
	}
	/*
	 * we fork+exec the command to run during our system wide monitoring
	 * session. When the command ends, we stop the session and print
	 * the results.
	 */
	if ((pid=fork()) == -1) {
		warning("Cannot fork new process\n");
		return -1;
	}

	if (pid == 0) {		 
		pid = getpid();

		if (options.opt_verbose) {
			char **p = argv;
			printf("starting process [%d]: ", pid);
			while (*p) printf("%s ", *p++);
			printf("\n");
		}

		execvp(argv[0], argv);

		warning("child: cannot exec %s: %s\n", argv[0], strerror(errno));
		exit(-1);
	} 
	barrier_wait(&barrier);

	waitpid(pid, &status, 0);

	/* we may or may not want to check child exit status here */
	if (WEXITSTATUS(status) != 0) {
		warning("process %d exited with non zero value (%d): results may be incorrect\n", pid, WEXITSTATUS(status));
	}

	return 0;
}

int
measure_system_wide(pfmlib_param_t *evt, pfarg_context_t *ctx, pfarg_reg_t *pc, int count, char **argv)
{
	int i, ret, n;
	unsigned long mask;
	void *retval;
	pthread_t thread_list[PFMON_MAX_CPUS];
	thread_arg_t arg[PFMON_MAX_CPUS];
	int nready;

	setup_signals();

	if (options.opt_aggregate_res && options.opt_use_smpl) {
		if (setup_sampling_output(options.smpl_ctx) == -1) return -1;
	}

	session_state = SESSION_RUN;

	n = hweight64(options.cpu_mask);

	DPRINT(("system wide session on %d CPU\n", n));

	barrier_init(&barrier, n+1);

	register_exit_function(exit_system_wide);

	pthread_key_create(&param_key, NULL);


	mask = options.cpu_mask;

	for(i=0, n = 0; mask; mask>>=1, i++) {

		if ((mask & 0x1) == 0) continue;

		if (n > 0 && options.opt_aggregate_res && options.opt_use_smpl) {
			options.smpl_ctx[n].smpl_fd  = options.smpl_ctx[0].smpl_fd;
		}
		arg[n].id    = n;
		arg[n].cpu   = i;
		arg[n].ctx   = *ctx;	/* copy the context, because we modify it */
		arg[n].evt   = evt;
		arg[n].pc    = pc;
		arg[n].count = count;

		thread_state[n] = THREAD_STARTED;

		ret = pthread_create(&thread_list[n], NULL, (void *(*)(void *))do_measure_one_cpu, arg+n);
		if (ret != 0) goto abort;

		DPRINT(("created thread[%d], %d\n", n, (int)thread_list[n]));
		n++;
	}
	/*
	 * must make sure that the installation went ok
	 */
	do {
		nready = 0;
		for(i=0; i < n ; i++) {
			if (thread_state[i] == THREAD_ERROR) {
				DPRINT(("the thread on CPU%d had a problem, aborting\n", i));
				goto abort;
			}
			if (thread_state[i] == THREAD_RUN) nready++;
		}
		DPRINT(("n=%d nready=%d\n", n, nready));
		usleep(100000);
	} while (nready < n);

	if (delimit_session(argv) == -1) goto abort;

	/*
	 * set end of session and unblock all threads
	 */
	pthread_mutex_lock(&session_mutex);
	session_state = SESSION_STOP;
	pthread_cond_broadcast(&session_cond);
	pthread_mutex_unlock(&session_mutex);

	DPRINT(("[%d] after session stop\n"));
	/*
	 * collect all resources
	 */
	for(i=0; i< n; i++) {
		ret = pthread_join(thread_list[i], &retval);
		if (ret !=0) warning("cannot joing thread %d\n", i);
		DPRINT(("thread %d exited with ret=%ld\n", i, (unsigned long)retval));
	}

	if (options.opt_aggregate_res) {
		print_results(sys_pd, options.smpl_ctx);
		if (options.opt_use_smpl) close_sampling_output(options.smpl_ctx);
	}

	pthread_key_delete(param_key);

	register_exit_function(NULL);


	return 0;

abort:
	session_state = SESSION_ABORTED;


	DPRINT(("aborting %d threads\n", n));
	for(i=0; i < n; i++) {
		DPRINT(("cancel %d\n", i));
		pthread_cancel(thread_list[i]);
	}
	for(i=0; i < n; i++) {
		ret = pthread_join(thread_list[i], &retval);
		if (ret != 0) warning("cannot join thread %i\n", i);
		DPRINT(("thread %d exited with ret=%ld\n", i, (unsigned long)retval));
	}

	register_exit_function(NULL);

	return -1;
}
