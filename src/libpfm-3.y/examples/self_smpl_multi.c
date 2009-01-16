/*
 *
 * self_smpl_multi.c - multi-thread self-sampling program
 *
 * Copyright (c) 2008 Mark W. Krentel
 * Contributed by Mark W. Krentel <krentel@cs.rice.edu>
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

#include <linux/unistd.h>
#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>
#include "detect_pmcs.h"

#define PROGRAM_TIME  8
#define THRESHOLD  20000000

int program_time = PROGRAM_TIME;
int threshold = THRESHOLD;
int signum = SIGIO;

#define MAX_FD  20

struct over_args {
	pfmlib_event_t ev;
	pfmlib_input_param_t  inp;
	pfmlib_output_param_t outp;
	pfarg_pmr_t pc[PFMLIB_MAX_PMCS];
	pfarg_pmd_attr_t pd[PFMLIB_MAX_PMDS];
	int fd;
	pid_t tid;
	pthread_t self;
};

struct over_args *fd2ov[MAX_FD];

long count[MAX_FD];
long total[MAX_FD];
long iter[MAX_FD];
long mismatch[MAX_FD];
long bad_msg[MAX_FD];
long bad_restart[MAX_FD];
long ser_no = 0;

pid_t
gettid(void)
{
#ifdef SYS_gettid
	return (pid_t)syscall(SYS_gettid);
#elif defined(__NR_gettid)
	return (pid_t)syscall(__NR_gettid);
#else
#error "Unable to implement gettid."
#endif
}

void
user_callback(int fd)
{
	count[fd]++;
	total[fd]++;
	ser_no++;
}

void
do_cycles(void)
{
	struct timeval start, last, now;
	double x, sum;
	int fd;

	for (fd = 0; fd < MAX_FD; fd++) {
		if (fd2ov[fd] != NULL &&
				pthread_equal(fd2ov[fd]->self, pthread_self())) {
			break;
		}
	}

	gettimeofday(&start, NULL);
	last = start;
	count[fd] = 0;
	total[fd] = 0;
	iter[fd] = 0;

	do {
		sum = 1.0;
		for (x = 1.0; x < 250000.0; x += 1.0)
			sum += x;
		if (sum < 0.0)
			printf("==>>  SUM IS NEGATIVE !!  <<==\n");
		iter[fd]++;

		gettimeofday(&now, NULL);
		if (now.tv_sec > last.tv_sec) {
			printf("%ld: fd = %d, count = %4ld, iter = %4ld, rate = %ld/Kiter\n",
					now.tv_sec - start.tv_sec, fd, count[fd], iter[fd],
					(1000 * count[fd])/iter[fd]);
			count[fd] = 0;
			iter[fd] = 0;
			last = now;
		}
	} while (now.tv_sec < start.tv_sec + program_time);
}

#define DPRINT(str)   \
printf("(%s) ser = %ld, fd = %d, tid = %d, self = %p\n",   \
       str, ser_no, fd, tid, (void *)self)

void
sigio_handler(int sig, siginfo_t *info, void *context)
{
	int fd;
	pid_t tid;
	pthread_t self;
	struct over_args *ov;
	pfarg_msg_t msg;

	/*
	 * file descripor is the only reliable source of information
	 * to identify session from which the notification originated
	 *
	 * Depending on scheduling, the signal may not be processed by the
	 * thread which posted it, i.e., the thread which had the nortification
	 *
	 * POSIX aysnchronous signals cannot be targeted to specific threads
	 */
	fd = info->si_fd;
 	self = pthread_self();
 	tid = gettid();
 	ov = fd2ov[fd];

	if (fd < 0 || fd >= MAX_FD)
		errx(1, "bad info.si_fd: %d", fd);

	/*
 	 * current thread id may not always match the id associated with
 	 * the dfile descriptor
 	 */
	if (tid != ov->tid || !pthread_equal(self, ov->self)) {
		mismatch[fd]++;
		DPRINT("bad thread");
	}

	pfm_read(fd, 0, PFM_RW_PMD, &ov->pd[1], sizeof(pfarg_pmd_attr_t));
	/*
 	 * extract notification message
 	 */
	if (read(fd, &msg, sizeof(msg)) != sizeof(msg))
		errx(1, "read from sigio fd failed");

	/*
 	 * cannot be PFM_END_MSG starting with perfmon v2.8
 	 */
	if (msg.type == PFM_MSG_END) {
		DPRINT("pfm_msg_end");
	} else if (msg.type != PFM_MSG_OVFL) {
		bad_msg[fd]++;
		DPRINT("bad msg type");
	}

	user_callback(fd);

	/*
 	 * when session is not that of the current thread, pfm_restart() does
 	 * not guarante that upon return monitoring will be resumed. There
 	 * may be a delay due to scheduling.
 	 */
	if (pfm_set_state(fd, 0, PFM_ST_RESTART) != 0) {
		bad_restart[fd]++;
		DPRINT("bad restart");
	}
}

void
overflow_start(struct over_args *ov, char *name)
{
	int i, fd, flags;
	pfarg_sinfo_t sif;

	memset(ov, 0, sizeof(struct over_args));
	memset(&sif, 0, sizeof(sif));

	if (pfm_get_cycle_event(&ov->ev) != PFMLIB_SUCCESS)
		errx(1, "pfm_get_cycle_event failed");

	ov->inp.pfp_event_count = 1;
	ov->inp.pfp_dfl_plm = PFM_PLM3;
	ov->inp.pfp_flags = 0;
	ov->inp.pfp_events[0] = ov->ev;

	fd = pfm_create(0, &sif);
	if (fd < 0)
		errx(1, "pfm_create_session failed");

	fd2ov[fd] = ov;
	ov->fd = fd;
	ov->tid = gettid();
	ov->self = pthread_self();

	detect_unavail_pmu_regs(&sif, &ov->inp.pfp_unavail_pmcs, NULL);

	if (pfm_dispatch_events(&ov->inp, NULL, &ov->outp, NULL) != PFMLIB_SUCCESS)
		errx(1, "pfm_dispatch_events failed");

	for (i = 0; i < ov->outp.pfp_pmc_count; i++) {
		ov->pc[i].reg_num =   ov->outp.pfp_pmcs[i].reg_num;
		ov->pc[i].reg_value = ov->outp.pfp_pmcs[i].reg_value;
	}
	for (i = 0; i < ov->outp.pfp_pmd_count; i++) {
		ov->pd[i].reg_num = ov->outp.pfp_pmds[i].reg_num;
	}

	ov->pd[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
	ov->pd[0].reg_value = - threshold;
	ov->pd[0].reg_long_reset = - threshold;
	ov->pd[0].reg_short_reset = - threshold;

	if (pfm_write(fd, 0, PFM_RW_PMC, ov->pc, ov->outp.pfp_pmc_count * sizeof(pfarg_pmr_t)))
		errx(1, "pfm_write failed");

	if (pfm_write(fd, 0, PFM_RW_PMD_ATTR, ov->pd, ov->outp.pfp_pmd_count * sizeof(pfarg_pmd_attr_t)))
		errx(1, "pfm_write(PMD) failed");

	if (pfm_attach(fd, 0, gettid()) != 0)
		errx(1, "pfm_attach failed");

	flags = fcntl(fd, F_GETFL, 0);
	if (fcntl(fd, F_SETFL, flags | O_ASYNC) < 0)
		errx(1, "fcntl SETFL failed");

	if (fcntl(fd, F_SETOWN, gettid()) < 0)
		errx(1, "fcntl SETOWN failed");

	if (fcntl(fd, F_SETSIG, signum) < 0)
		errx(1, "fcntl SETSIG failed");

	if (pfm_set_state(fd, 0, PFM_ST_START))
		errx(1, "pfm_set_state(start) failed");

	printf("launch %s: fd: %d, tid: %d, self: %p\n",
		name, fd, ov->tid, (void *)ov->self);
}

void
overflow_stop(struct over_args *ov)
{
	if (pfm_set_state(ov->fd, 0, PFM_ST_STOP))
		errx(1, "pfm_set_state(stop) failed");
}

void *
my_thread(void *v)
{
	struct over_args ov;

	overflow_start(&ov, "side");
	do_cycles();
	overflow_stop(&ov);

	return (NULL);
}

/*
 *  Program args: program_time, threshold, signum.
 */
int
main(int argc, char **argv)
{
	pthread_t thr;
	struct over_args ov;
	struct sigaction sa;
	sigset_t set;
	int i;

	if (argc < 2 || sscanf(argv[1], "%d", &program_time) < 1)
		program_time = PROGRAM_TIME;
	if (argc < 3 || sscanf(argv[2], "%d", &threshold) < 1)
		threshold = THRESHOLD;
	if (argc < 4 || sscanf(argv[3], "%d", &signum) < 1)
		signum = SIGIO;

	printf("program_time = %d, threshold = %d, signum = %d\n",
		program_time, threshold, signum);

	for (i = 0; i < MAX_FD; i++) {
		fd2ov[i] = NULL;
		mismatch[i] = 0;
		bad_msg[i] = 0;
		bad_restart[i] = 0;
	}

	memset(&sa, 0, sizeof(sa));
	sigemptyset(&set);
	sa.sa_sigaction = sigio_handler;
	sa.sa_mask = set;
	sa.sa_flags = SA_SIGINFO;

	if (sigaction(signum, &sa, NULL) != 0)
		errx(1, "sigaction failed");

	if (pfm_initialize() != PFMLIB_SUCCESS)
		errx(1, "pfm_initialize failed");

	printf("\n");
	if (pthread_create(&thr, NULL, my_thread, NULL) != 0)
		errx(1, "pthread_create failed");

	overflow_start(&ov, "main");
	do_cycles();
	overflow_stop(&ov);

	printf("\n");
	for (i = 0; i < MAX_FD; i++) {
		if (fd2ov[i] != NULL) {
			printf("total[%d] = %ld, mismatch[%d] = %ld, "
				"bad_msg[%d] = %ld, bad_restart[%d] = %ld\n",
				i, total[i], i, mismatch[i],
				i, bad_msg[i], i, bad_restart[i]);
		}
	}

	return (0);
}
