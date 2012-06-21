/*
 *
 * self_smpl_multi.c - multi-thread self-sampling program
 *
 * Copyright (c) 2008 Mark W. Krentel
 * Contributed by Mark W. Krentel <krentel@cs.rice.edu>
 * Modified by Stephane Eranian <eranian@gmail.com>
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 *
 *  Test perfmon overflow without PAPI.
 *
 *  Create a new thread, launch perfmon overflow counters in both
 *  threads, print the number of interrupts per thread and per second,
 *  and look for anomalous interrupts.  Look for mismatched thread
 *  ids, bad message type, or failed pfm_restart().
 *
 *  self_smpl_multi is a test program to stress signal delivery in the context
 *  of a multi-threaded self-sampling program which is common with PAPI and HPC.
 *  There is an issue with existing (as of 2.6.30) kernel which do not provide
 *  a reliable way of having the signal delivered to the thread in which the
 *  counter overflow occurred. This is problematic for many self-monitoring
 *  program.
 *
 *  This program demonstrates the issue by tracking the number of times
 *  the signal goes to the wrong thread. The bad behavior is exacerbated
 *  if the monitored threads, themselves, already use signals. Here we
 *  use SIGLARM.
 *
 *  Note that kernel developers have been made aware of this problem and
 *  a fix has been proposed. It introduces a new F_SETOWN_TID parameter to
 *  fcntl().
 */
#include <sys/time.h>
#include <sys/types.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include <linux/unistd.h>
#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>
#include "detect_pmcs.h"

#define PROGRAM_TIME  8
#define THRESHOLD  20000000

int program_time = PROGRAM_TIME;
int threshold = THRESHOLD;
int signum = SIGIO;
pthread_barrier_t barrier;

#define MAX_THR  128

/*
 *  the following definitions come
 *  from the F_SETOWN_EX patch from Peter Zijlstra
 * Check out: http://lkml.org/lkml/2009/8/4/128 
 */
#ifndef F_SETOWN_EX
#define F_SETOWN_EX	15
#define F_GETOWN_EX	16

#define F_OWNER_TID	0
#define F_OWNER_PID	1
#define F_OWNER_PGRP	2

struct f_owner_ex {
	int	type;
	pid_t	pid;
};
#endif

struct over_args {
	int fd;
	pid_t tid;
	int id;
};

struct over_args fd2ov[MAX_THR];

long count[MAX_THR];
long total[MAX_THR];
long iter[MAX_THR];
long mismatch[MAX_THR];
long bad_msg[MAX_THR];
long bad_restart[MAX_THR];
int fown;

int __thread myid; /* TLS */

pid_t
gettid(void)
{
	return (pid_t)syscall(__NR_gettid);
}

void
user_callback(int m)
{
	count[m]++;
	total[m]++;
}

void
do_cycles(void)
{
	struct timeval start, last, now;
	unsigned long x, sum;

	gettimeofday(&start, NULL);
	last = start;
	count[myid] = 0;
	total[myid] = 0;
	iter[myid] = 0;

	do {

		sum = 1;
		for (x = 1; x < 250000; x++) {
			/* signal pending to private queue because of
			 * pthread_kill(), i.e., tkill()
			 */
			if ((x % 5000) == 0)
				pthread_kill(pthread_self(), SIGUSR1);
			sum += x;
		}
/* 
		if (sum < 0)
			printf("==>>  SUM IS NEGATIVE !!  <<==\n");
*** SUM IS UNSIGNED! ***
*/
		iter[myid]++;

		gettimeofday(&now, NULL);
		if (now.tv_sec > last.tv_sec) {
			printf("%ld: myid = %3d, fd = %3d, count = %4ld, iter = %4ld, rate = %ld/Kiter\n",
				now.tv_sec - start.tv_sec,
				myid,
				fd2ov[myid].fd,
				count[myid], iter[myid],
				(1000 * count[myid])/iter[myid]);

			count[myid] = 0;
			iter[myid] = 0;
			last = now;
		}
	} while (now.tv_sec < start.tv_sec + program_time);
}

#define DPRINT(str)   \
printf("(%s) si->fd = %d, ov->self = 0x%lx, self = 0x%lx\n",   \
       str, fd, (unsigned long)ov->self, (unsigned long)self)
void
sigusr1_handler(int sig, siginfo_t *info, void *context)
{
}

/*
 * a signal handler cannot safely invoke printf()
 */
void
sigio_handler(int sig, siginfo_t *info, void *context)
{
	int fd, i;
	pid_t tid;
	struct over_args *ov;
	pfarg_msg_t msg;

	/*
	 * file descripor is the only reliable source of information
	 * to identify context from which the notification originated
	 *
	 * Depending on scheduling, the signal may not be processed by the
	 * thread which posted it, i.e., the thread which had the notification
	 *
	 * POSIX aysnchronous signals cannot be targeted to specific threads
	 */
	fd = info->si_fd;
 	tid = gettid();


	for(i=0; i < MAX_THR; i++)
		if (fd2ov[i].fd == fd)
			break;

	if (i == MAX_THR)
		errx(1, "bad info.si_fd: %d", fd);

 	ov = &fd2ov[i];

	/*
 	 * current thread id may not always match the id associated with
 	 * the file descriptor
 	 */
	if (tid != ov->tid)
		mismatch[myid]++;

	/*
 	 * extract notification message
 	 */
	if (read(fd, &msg, sizeof(msg)) != sizeof(msg))
		errx(1, "read from sigio fd failed");

	if (msg.type != PFM_MSG_OVFL)
		bad_msg[myid]++;

	user_callback(myid);

	/*
 	 * when context is not that of the current thread, pfm_restart() does
 	 * not guarante that upon return monitoring will be resumed. There
 	 * may be a delay due to scheduling.
 	 */
	if (pfm_restart(fd) != 0)
		bad_restart[myid]++;
}

void
overflow_start(char *name)
{
	pfmlib_input_param_t  inp;
	pfmlib_output_param_t outp;
	pfarg_pmc_t pc[PFMLIB_MAX_PMCS];
	pfarg_pmd_t pd[PFMLIB_MAX_PMDS];
	struct f_owner_ex fown_ex;
	pfarg_ctx_t ctx;
	pfarg_load_t load_arg;
	struct over_args *ov;
	int i, fd, flags;
	int ret;

        memset(&ctx, 0, sizeof(ctx));
        memset(&inp, 0, sizeof(inp));
        memset(&outp, 0, sizeof(outp));
        memset(pc, 0, sizeof(pc));
        memset(pd, 0, sizeof(pd));
        memset(&load_arg, 0, sizeof(load_arg));

	ov = &fd2ov[myid];

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		errx(1, "pfm_get_cycle_event failed");

	inp.pfp_event_count = 1;
	inp.pfp_dfl_plm = PFM_PLM3;
	inp.pfp_flags = 0;

	fd = pfm_create_context(&ctx, NULL, NULL, 0);
	if (fd < 0)
		err(1, "pfm_create_context failed");

	ov->fd = fd;
	ov->tid = gettid();
	ov->id = myid;

	detect_unavail_pmcs(fd, &inp.pfp_unavail_pmcs);

	ret = pfm_dispatch_events(&inp, NULL, &outp, NULL);
	if (ret != PFMLIB_SUCCESS)
		errx(1, "pfm_dispatch_events failed: %s", pfm_strerror(ret));

	for (i = 0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num =   outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i = 0; i < outp.pfp_pmd_count; i++) {
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
	}

	pd[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
	pd[0].reg_value = - threshold;
	pd[0].reg_long_reset = - threshold;
	pd[0].reg_short_reset = - threshold;

	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count))
		err(1, "pfm_write_pmcs failed");

	if (pfm_write_pmds(fd, pd, outp.pfp_pmd_count))
		err(1, "pfm_write_pmds failed");

	load_arg.load_pid = gettid();
	if (pfm_load_context(fd, &load_arg) != 0)
		err(1, "pfm_load_context failed");

	flags = fcntl(fd, F_GETFL, 0);
	if (fcntl(fd, F_SETFL, flags | O_ASYNC) < 0)
		err(1, "fcntl SETFL failed");

	fown_ex.type = F_OWNER_TID;
	fown_ex.pid  = gettid();
	ret = fcntl(fd,
		    (fown ? F_SETOWN_EX : F_SETOWN), 
		    (fown ? (unsigned long)&fown_ex: gettid()));
	if (ret)
		err(1, "fcntl SETOWN failed");

	if (fcntl(fd, F_SETSIG, signum) < 0)
		err(1, "fcntl SETSIG failed");

	if (pfm_start(fd, NULL))
		err(1, "pfm_start failed");
		
	printf("launch %s: fd: %d, tid: %d\n", name, fd, ov->tid);
}

void
overflow_stop(void)
{
	pfm_self_stop(fd2ov[myid].fd);
}

void *
my_thread(void *v)
{
	int retval = 0;
	
	myid = (unsigned long)v;

	pthread_barrier_wait(&barrier);

	overflow_start("side");
	do_cycles();
	overflow_stop();

	pthread_exit((void *)&retval);
}

static void
usage(void)
{
	printf("self_smpl_multi [-t secs] [-p period] [-s signal] [-f] [-n threads]\n"
		"-t secs: duration of the run in seconds\n"
		"-p period: sampling period in CPU cycles\n"
		"-s signal: signal to use (default: SIGIO)\n"
		"-n thread: number of threads to create (default: 1)\n"
		"-f : use F_SETOWN_EX for correct delivery of signal to thread (default: off)\n");
}

/*
 *  Program args: program_time, threshold, signum.
 */
int
main(int argc, char **argv)
{
	struct sigaction sa;
	pthread_t allthr[MAX_THR];
	sigset_t set;
	int i, ret, max_thr = 1;

	while((i=getopt(argc, argv, "t:p:s:fhn:")) != EOF) {
		switch(i) {
		case 'h':
			usage();
			return 0;
		case 't':
			program_time = atoi(optarg);
			break;
		case 'p':
			threshold = atoi(optarg);
			break;
		case 's':
			signum = atoi(optarg);
			break;
		case 'f':
			fown = 1;
			break;
		case 'n':
			max_thr = atoi(optarg);
			if (max_thr >= MAX_THR)
				errx(1, "no more than %d threads", MAX_THR);
			break;
		default:
			errx(1, "invalid option");
		}
	}
	printf("program_time = %d, threshold = %d, signum = %d fcntl(%s), threads = %d\n",
		program_time, threshold, signum,
		fown ? "F_SETOWN_EX" : "F_SETOWN",
		max_thr);

	for (i = 0; i < MAX_THR; i++) {
		mismatch[i] = 0;
		bad_msg[i] = 0;
		bad_restart[i] = 0;
	}

	memset(&sa, 0, sizeof(sa));
	sigemptyset(&set);

	sa.sa_sigaction = sigusr1_handler;
	sa.sa_mask = set;
	sa.sa_flags = SA_SIGINFO;

	if (sigaction(SIGUSR1, &sa, NULL) != 0)
		errx(1, "sigaction failed");

	memset(&sa, 0, sizeof(sa));
	sigemptyset(&set);

	sa.sa_sigaction = sigio_handler;
	sa.sa_mask = set;
	sa.sa_flags = SA_SIGINFO;

	if (sigaction(signum, &sa, NULL) != 0)
		errx(1, "sigaction failed");

	if (pfm_initialize() != PFMLIB_SUCCESS)
		errx(1, "pfm_initialize failed");

	/*
 	 * +1 because main thread is also using the barrier
 	 */
	pthread_barrier_init(&barrier, 0, max_thr+1);

	for(i=0; i < max_thr; i++) {
		ret = pthread_create(allthr+i, NULL, my_thread, (void *)(unsigned long)i);
		if (ret)
			err(1, "pthread_create failed");
	}
	myid = i;
	sigemptyset(&set);
	sigaddset(&set, SIGIO);
	if (pthread_sigmask(SIG_BLOCK, &set, NULL))
		err(1, "cannot mask SIGIO in main thread");

	pthread_barrier_wait(&barrier);
	printf("\n\n");

	for (i = 0; i < max_thr; i++) {
		pthread_join(allthr[i], NULL);
	}
	printf("\n\n");
	for (i = 0; i < max_thr; i++) {
		printf("myid = %3d, fd = %3d, total = %4ld, mismatch = %ld, "
			"bad_msg = %ld, bad_restart = %ld\n",
			fd2ov[i].id, fd2ov[i].fd, total[i], mismatch[i],
			bad_msg[i], bad_restart[i]);
	}
	return (0);
}
