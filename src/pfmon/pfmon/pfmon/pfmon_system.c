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
#include <sys/time.h>
#include <semaphore.h>

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
} thread_arg_t;

#define SESSION_RUN   	0
#define SESSION_STOP  	1
#define SESSION_ABORTED	2

typedef struct _barrier {
	pthread_mutex_t mutex;
	pthread_cond_t	cond;
	unsigned long	counter;
	unsigned long	max;
	unsigned long   generation; /* avoid race condition on wake-up */
} barrier_t;

#define THREAD_STARTED	0
#define THREAD_RUN	1
#define THREAD_DONE	2
#define THREAD_ERROR	3

static barrier_t barrier;
static int session_state;
static int who_must_print;
static int syswide_alarm_expired;
static int syswide_time_to_quit;

static pthread_mutex_t results_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  results_cond  = PTHREAD_COND_INITIALIZER;
static pthread_key_t   param_key;

static int thread_state[PFMON_MAX_CTX];
static sem_t ovfl_sem[PFMON_MAX_CTX];
static sem_t syswide_alarm_sem;
static unsigned long ovfl_cnts[PFMON_MAX_CTX];

static	pfarg_reg_t sys_pd[PMU_MAX_PMDS];

static int
barrier_init(barrier_t *b, unsigned long count)
{
	int r;

	r = pthread_mutex_init(&b->mutex, NULL);
	if (r == -1) return -1;
	r = pthread_cond_init(&b->cond, NULL);
	if (r == -1) return -1;
	b->max = b->counter = count;

	b->generation = 0;

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
	unsigned long generation;
	int r, oldstate;

	pthread_cleanup_push(cleanup_barrier, b);

	pthread_mutex_lock(&b->mutex);

	pthread_testcancel();


	if (--b->counter == 0) {
		DPRINT(("last thread entered\n"));

		/* reset barrier */
		b->counter = b->max;
		/*
		 * bump generation number, this avoids thread getting stuck in the
		 * wake up loop below in case a thread just out of the barrier goes
		 * back in right away before all the thread from the previous "round"
		 * have "escaped".
		 */
		b->generation++;

		r = pthread_cond_broadcast(&b->cond);
	} else {

		generation = b->generation;

		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldstate);

		while (b->counter != b->max && generation == b->generation) {
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
	pid_t mypid = getpid();
	int mycpu, cur_cpu;
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
	mycpu 		    = arg->cpu;

	DPRINT(("[%d] before CPU%d must be on CPU%d mask=0x%lx\n", 
		mypid, 
		find_cpu(mypid), 
		mycpu, 
		ctx->ctx_cpu_mask));

	if (perfmonctl(mypid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == EBUSY) {
			warning("CPU%d concurrent conflicting monitoring session is present in your system\n", mycpu);
		} else {
			warning("CPU%d Can't create PFM context: %s\n", mycpu, strerror(errno));
		}
		goto error;
	}
	cur_cpu = find_cpu(mypid);
	DPRINT(("[%d] CPU%d thread now on CPU%d\n", mypid, mycpu, cur_cpu));

	if (cur_cpu != mycpu) {
		warning("Thread does not run on correct CPU: %d instead of %d\n",
		       cur_cpu, mycpu);
		goto error;
	}

	if (options.opt_use_smpl) {
		if (options.opt_aggregate_res == 0 && setup_sampling_output(csmpl) == -1) goto error;
		csmpl->smpl_hdr = ctx->ctx_smpl_vaddr;

		DPRINT(("CPU%d sampling buffer at %p\n", mycpu, csmpl->smpl_hdr));
	}

	if (enable_pmu(mypid) == -1) goto error;

	if (install_counters(mypid, arg->evt) == -1) goto error;

	thread_state[arg->id] = THREAD_RUN;

	barrier_wait(&barrier);

	DPRINT(("CPU%d after barrier state=%d\n", mycpu, session_state));

	if (session_state == SESSION_ABORTED) goto error;


	if (session_start(mypid) == -1) goto error;

	vbprintf("CPU%d started monitoring\n", mycpu);

	for(;;) {
		sem_wait(ovfl_sem+arg->id);

		if (session_state == SESSION_STOP) break;

		pthread_testcancel();

		if (options.opt_aggregate_res) pthread_mutex_lock(&results_mutex);

		pfmon_process_smpl_buf(csmpl, mypid);

		if (options.opt_aggregate_res) pthread_mutex_unlock(&results_mutex);

		pthread_testcancel();
	}


	if (session_stop(mypid) == -1) goto error;

	vbprintf("CPU%d stopped monitoring\n", mycpu);

	memset(pd, 0, sizeof(pd));

	/*
	 * we use pfp_count and not count to capture only the counters
	 * and not any other PMC. The counters are GUARANTEED to be at the beginning
	 */
	for(i=0; i < arg->evt->pfp_event_count; i++) {
		pd[i].reg_num = arg->evt->pfp_pc[i].reg_num;
		DPRINT(("CPU%d will read pmd%ld\n", mycpu, pd[i].reg_num));
	}

	/*
	 * read the PMDS now using our context
	 */
	if (perfmonctl(mypid, PFM_READ_PMDS, pd, arg->evt->pfp_event_count) == -1) {
		warning("CPU%d perfmonctl error READ_PMDS: %s\n", mycpu, strerror(errno));
		goto error;
	}
	DPRINT(("CPU%d has read PMDS\n", mycpu));

	/* 
	 * dump results 
	 */
	if (options.opt_aggregate_res) {
		pthread_mutex_lock(&results_mutex);

		syswide_aggregate_results(pd);

		if (options.opt_use_smpl) pfmon_process_smpl_buf(csmpl,  getpid());

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
	/* NO RETURN */
error:
	vbprintf("CPU%d session aborted\n", mycpu);
	close_sampling_output(csmpl);
	thread_state[arg->id] = THREAD_ERROR;
	pthread_exit((void *)(~0UL));
	/* NO RETURN */
}

/*
 * We cannot use any of the stdio routine during the execution of the handler because
 * they are not lock-safe with regards to ASYNC signals.
 *
 * NO DEBUG(), warning() or fatal_error() in the handler!
 */
static void
syswide_overflow_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	thread_arg_t *arg = (thread_arg_t *)pthread_getspecific(param_key);

	/* ignore spurious overflow interrupts */
	if (info->sy_code != PROF_OVFL) {
		return;
	} 

	/* keep some statistics */
	ovfl_cnts[arg->cpu]++;

	/* 
	 * force processing of the sampling buffer upon return from the handler
	 */
	sem_post(ovfl_sem+arg->id);
}

static void
syswide_alarm_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	/*
	 * XXX: should do something more gentle here
	 */
	syswide_alarm_expired = 1;
	sem_post(&syswide_alarm_sem);
}

static void
syswide_child_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	/*
	 * We are only interested in SIGCHLD indicating that the process is
	 * dead
	 */
	if (info->si_code != CLD_EXITED && info->si_code != CLD_KILLED) return;

	/*
	 * stop the alarm, if any
	 */
	if (options.session_timeout || options.trigger_delay) alarm(0);

	syswide_time_to_quit = 1;
	sem_post(&syswide_alarm_sem);
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
	sigaddset(&my_set, SIGALRM);

	act.sa_handler = (sig_t)syswide_overflow_handler;
	act.sa_mask    = my_set;
	sigaction (SIGPROF, &act, 0);
	
	memset(&act,0,sizeof(act));

	sigemptyset(&my_set);
	sigaddset(&my_set, SIGCHLD);

	act.sa_mask    = my_set;
	act.sa_handler = (sig_t)syswide_alarm_handler;

	sigaction (SIGALRM, &act, 0);
	
	memset(&act,0,sizeof(act));

	act.sa_handler = (sig_t)syswide_child_handler;
	sigaction (SIGCHLD, &act, 0);
}

static void
exit_system_wide(int i)
{
	thread_arg_t *arg = (thread_arg_t *)pthread_getspecific(param_key);

	DPRINT(("thread on CPU%d aborting\n", arg->cpu));

	thread_state[arg->id] = THREAD_ERROR;

	pthread_exit((void *)((unsigned long)i));
}

static int
delay_start(void)
{
	sem_init(&syswide_alarm_sem, 0, 0);

	alarm(options.trigger_delay);

	sem_wait(&syswide_alarm_sem);

	DPRINT(("delay_start: expire=%d quit=%d\n", syswide_alarm_expired, syswide_time_to_quit));

	return syswide_alarm_expired ? 0 : -1;
}

static int
delimit_session(char **argv)
{
	struct timeval time_start, time_end;
	struct rusage ru;
	pid_t pid;
	int status;
	int ret = 0;

	/*
	 * take care of the easy case first: no command to start
	 */
	if (argv == NULL || *argv == NULL) {

		if (options.trigger_delay) delay_start();

		/*
		 * this will start the session in each "worker" thread
		 */
		barrier_wait(&barrier);

		if (options.session_timeout) {
			printf("<session to end in %lu seconds>\n", options.session_timeout);
			sleep(options.session_timeout);
		} else {
			printf("<press ENTER to stop session>\n");
			getchar();
		}
		return 0;
	}
	gettimeofday(&time_start, NULL);
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

	if (options.trigger_delay && delay_start() == -1) {
		printf("process %d terminated before session was activated, nothing measured\n", pid);
		session_state = SESSION_ABORTED;
		ret = -1;
	}

	/*
	 * this will start the session in each "worker" thread
	 */
	barrier_wait(&barrier);

	wait4(pid, &status, 0, &ru);

	gettimeofday(&time_end, NULL);

	/* we may or may not want to check child exit status here */
	if (WEXITSTATUS(status) != 0) {
		warning("process %d exited with non zero value (%d): results may be incorrect\n", pid, WEXITSTATUS(status));
	}

	if (options.opt_show_rusage) show_task_rusage(&time_start, &time_end, &ru);
	return ret;
}

int
measure_system_wide(pfmlib_param_t *evt, pfarg_context_t *ctx, char **argv)
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
			options.smpl_ctx[n].smpl_fp  = options.smpl_ctx[0].smpl_fp;
		}
		arg[n].id    = n;
		arg[n].cpu   = i;
		arg[n].ctx   = *ctx;	/* copy the context, because we modify it */
		arg[n].evt   = evt;

		thread_state[n] = THREAD_STARTED;
		sem_init(ovfl_sem+n, 0, 0);

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
				DPRINT(("CPU%d thread aborted\n", i));
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
	session_state = SESSION_STOP;

	for (i=0; i < n; i++) sem_post(ovfl_sem+i);

	DPRINT(("main thread after session stop\n"));

	for(i=0; i< n; i++) {
		ret = pthread_join(thread_list[i], &retval);
		if (ret !=0) warning("cannot join thread %d\n", i);
		DPRINT(("CPU%d thread exited with value %ld\n", arg[i].cpu, (unsigned long)retval));
	}

	if (options.opt_aggregate_res) {
		print_results(sys_pd, options.smpl_ctx);
		if (options.opt_use_smpl) close_sampling_output(options.smpl_ctx);
	}

	pthread_key_delete(param_key);

	register_exit_function(NULL);

	if (options.opt_verbose) {
		for(i=0; i < n; i++) vbprintf("CPU%d : %lu sampling buffer overflows\n", i, ovfl_cnts[i]);
	}
	return 0;
abort:
	session_state = SESSION_ABORTED;

	vbprintf("aborting %d threads\n", n);

	for(i=0; i < n; i++) {
		DPRINT(("cancelling on CPU%d\n", i));
		pthread_cancel(thread_list[i]);
	}
	for(i=0; i < n; i++) {
		ret = pthread_join(thread_list[i], &retval);
		if (ret != 0) warning("cannot join thread %i\n", i);
		DPRINT(("CPU%d thread exited with value %ld\n", arg[i].cpu, (unsigned long)retval));
	}

	pthread_key_delete(param_key);

	register_exit_function(NULL);

	return -1;
}
