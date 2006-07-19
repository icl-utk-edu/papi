/*
 * notify_standalone.c - self-contained (no libpfm) overflow notification program.
 *
 * The *_standalone.c examples are to be used with PMU model not supported
 * directly by libpfm as simple test programs to exercise the kernel API.
 *
 * Copyright (c) 20052-2006 Hewlett-Packard Development Company, L.P.
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <syscall.h>
#include <unistd.h>
#include <perfmon/perfmon.h>

#include "standalone.h"

#define NUM_PMCS	32
#define NUM_PMDS	32

#ifdef __mips__
#define SMPL_PERIOD	300000ULL
#else
#define SMPL_PERIOD	3000000000ULL
#endif
 
static int ctx_fd;
static volatile unsigned long notification_received;

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

static void
sigio_handler(int n, struct siginfo *info, struct sigcontext *sc)
{

	/*
	 * printf safe because busyloop() is not using printf()
	 */
	printf("Notification %lu\n", notification_received);

	/*
	 * At this point, the counter used for the sampling period has already
	 * be reset by the kernel because we are in non-blocking mode, self-monitoring.
	 */

	/*
	 * increment our notification counter
	 */
	notification_received++;

	/*
	 * And resume monitoring
	 */
	if (pfm_restart(ctx_fd) == -1) {
		fatal_error("pfm_restart: %s", strerror(errno));
	}
}

void
busyloop(void)
{
	double a = 0.0;
	/*
	 * busy loop to burn CPU cycles
	 */
	for(;notification_received < 10;) {
  		a = a + (double)1.0;
	}
	printf("sum=%g\n", a);
}

static void
program_pmu_xeon(int fd, int cpu, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	unsigned int npmcs = 0;
	unsigned int npmds = 0;

	printf("program set up for P4/Xeon/EM64T\n");
	/*
	 * measuring instr_retired with CRU_ESCR0, IQ_CCCR0, IQ_CTR0
	 * CRU_ESCR0.event_select=2
	 * CRU_ESCR0.usr=1 (at the user level)
	 * CRU_ESCR0.tag_enable=1
	 * CRU_ESCR0.event_mask=NBOGUSTAG
	 * IQ_CCCR0.cccr_select=4
	 * IQ_CCCR0.enable=1
	 * IQ_CCCR0.active_thread=3 (either logical CPU is active)
	 */
	pc[npmcs].reg_num   = 20; /* CRU_ESCR0 */
	pc[npmcs].reg_value = (2ULL <<25) | (1ULL<<9) |(1ULL<<2) | (1ULL<<4);
	npmcs++;
	pc[npmcs].reg_num   = 29; /* IQ_CCCR0 */
	pc[npmcs].reg_value = (4ULL<<13) | (1ULL<<12) | (3ULL <<16);
	npmcs++;

	*num_pmcs = npmcs;

	/*
	 * IQ_CTR0
	 */
	pd[npmds].reg_num   = 6;
	pd[npmds].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	pd[npmds].reg_value       = - SMPL_PERIOD;
	pd[npmds].reg_long_reset  = - SMPL_PERIOD;
	pd[npmds].reg_short_reset = - SMPL_PERIOD;
	npmds++;

	*num_pmds = npmds;
}

static void
program_pmu_mips(int fd, int cpu, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	unsigned int npmcs = 0;
	unsigned int npmds = 0;

	if (cpu == STANDALONE_MIPS20K) {
	    pc[npmcs].reg_num = 0;
	    pc[npmcs].reg_value = 0x3ULL << 5 | 0x8ULL; /* FP_INST, user mode*/
	    npmcs++;
	    pd[npmds].reg_num = 0;
	} else if (cpu == STANDALONE_MIPS5K) {
	    pc[npmcs].reg_num = 1;
	    pc[npmcs].reg_value = 0x5ULL << 5 | 0x8ULL; /* FP_INST, user mode*/
	    npmcs++;
	    pd[npmds].reg_num   = 1;
	} 

	pd[npmds].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	pd[npmds].reg_value       = - SMPL_PERIOD;
	pd[npmds].reg_long_reset  = - SMPL_PERIOD;
	pd[npmds].reg_short_reset = - SMPL_PERIOD;
	npmds++;

	printf("Measuring floating point instr. user mode: reg. %u value 0x%"PRIx64"\n",pc[0].reg_num,pc[0].reg_value);

	*num_pmcs = npmcs;
	*num_pmds = npmds;
}

static void
program_pmu(int fd, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	int cpu;

	cpu = cpu_detect();
	switch(cpu) {
		case STANDALONE_MIPS5K:
		case STANDALONE_MIPS20K:
			program_pmu_mips(fd, cpu, pc, num_pmcs, pd, num_pmds);
			break;
		case STANDALONE_P4:
			program_pmu_xeon(fd, cpu, pc, num_pmcs, pd, num_pmds);
			break;
		default:
			fatal_error("PMU model not supported by this program. It may be directly supported by libpfm, try the other examples\n");
	}

}

int
main(int argc, char **argv)
{
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_ctx_t ctx;
	unsigned int num_pmcs, num_pmds;
	pfarg_load_t load_args;
	struct sigaction act;
	int ret;

	num_pmcs = num_pmds = 0;

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));

	/*
	 * Install the signal handler (SIGIO)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)sigio_handler;
	sigaction (SIGIO, &act, 0);
	
	/*
	 * create a new context, per-thread context.
	 * This just creates a new context with some initial state, it is not
	 * active nor attached to any thread.
	 *
	 * we do not need the overflow notification message. This avoids
	 * having to read the overflow message in the notification handler.
	 */
	ctx.ctx_flags = PFM_FL_OVFL_NO_MSG;

	if (pfm_create_context(&ctx, NULL, 0) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * extract the unique identifier for our context, a regular file descriptor
	 */
	ctx_fd = ctx.ctx_fd;

	program_pmu(ctx_fd, pc, &num_pmcs, pd, &num_pmds);

	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(ctx_fd, pc, num_pmcs) == -1)
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(ctx_fd, pd, num_pmds) == -1)
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 *  attach the context to self
	 */
	load_args.load_pid = getpid();
	if (pfm_load_context(ctx_fd, &load_args) == -1) {
		fatal_error("pfm_load_context error errno %d\n",errno);
	}

	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL, 0) | O_ASYNC);
	if (ret == -1) {
		fatal_error("cannot set ASYNC: %s\n", strerror(errno));
	}

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(ctx_fd, F_SETOWN, getpid());
	if (ret == -1) {
		fatal_error("cannot setown: %s\n", strerror(errno));
	}

	/*
	 * start monitoring
	 */
	if (pfm_start(ctx_fd, NULL) == -1) {
		fatal_error("pfm_start error errno %d\n",errno);
	}

	/*
	 * do some work
	 */
	busyloop();

	/*
	 * stop monitoring
	 */
	if (pfm_stop(ctx_fd) == -1) {
		fatal_error("pfm_stop error errno %d\n",errno);
	}
	/*
	 * destroy context
	 */
	close(ctx_fd);

	return 0;
}
