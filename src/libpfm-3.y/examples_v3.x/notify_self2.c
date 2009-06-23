/*
 * notify_self2.c - example of how you can use overflow notifications with F_SETSIG
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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
#include <fcntl.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#include "detect_pmcs.h"

#define SMPL_PERIOD	1000000000ULL

static volatile unsigned long notification_received;

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static pfarg_pmr_t pdx[1];
static int ctx_fd;
static char *event1_name;

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
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

static void
sigio_handler(int n, struct siginfo *info, void *data)
{
	pfarg_msg_t msg;
	int fd;
	int r =0;

	if (info == NULL)
		fatal_error("info is NULL\n");

	fd = info->si_fd;
	if (fd != ctx_fd)
		fatal_error("handler does not get valid file descriptor\n");

	if (event1_name && pfm_read(fd, 0, PFM_RW_PMD, pdx, sizeof(pdx)))
		fatal_error("pfm_read: %s", strerror(errno));
retry:
	r = read(fd, &msg, sizeof(msg));
	if (r != sizeof(msg)) {
		if(r == -1 && errno == EINTR) {
			warning("read interrupted, retrying\n");
			goto retry;
		}
		fatal_error("cannot read overflow message: %s\n", strerror(errno));
	}

	if (msg.type != PFM_MSG_OVFL)
		fatal_error("unexpected msg type: %d\n",msg.type);

	/*
	 * increment our notification counter
	 */
	notification_received++;

	/*
	 * XXX: risky to do printf() in signal handler!
	 */
	if (event1_name)
		printf("Notification %lu: %"PRIu64" %s\n", notification_received, pdx[0].reg_value, event1_name);
	else
		printf("Notification %lu\n", notification_received);

	/*
	 * And resume monitoring
	 */
	if (pfm_set_state(fd, 0, PFM_ST_RESTART))
		fatal_error("pfm_set_state(restart): %d\n", errno);
}

/*
 * infinite loop waiting for notification to get out
 */
void
busyloop(void)
{
	/*
	 * busy loop to burn CPU cycles
	 */
	for(;notification_received < 20;) ;
}

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6

static inline void pfm_bv_set(uint64_t *bv, uint16_t rnum)
{
	bv[rnum>>LBPL] |= 1UL << (rnum&(BPL-1));
}

int
main(int argc, char **argv)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_pmd_attr_t pd[NUM_PMDS];
	pfarg_sinfo_t sif;
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	unsigned int i, num_counters;
	size_t len;
	int ret;

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	/*
	 * Install the signal handler (SIGIO)
	 *
	 * SA_SIGINFO required on some platforms
	 * to get siginfo passed to handler.
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)sigio_handler;
	act.sa_flags   = SA_SIGINFO;
	sigaction (SIGIO, &act, 0);

	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&sif,0, sizeof(sif));

	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");

	i = 2;

	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm = PFM_PLM3;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}

	inp.pfp_event_count = i;
	/*
	 * how many counters we use
	 */
	if (i > 1) {
		pfm_get_max_event_name_len(&len);
		event1_name = malloc(len+1);
		if (event1_name == NULL)
			fatal_error("cannot allocate event name\n");

		pfm_get_full_event_name(&inp.pfp_events[1], event1_name, len+1);
	}

	/*
	 * now create the session for self monitoring/per-task
	 */
	ctx_fd = pfm_create(0, &sif);
	if (ctx_fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("cannot create session %s\n", strerror(errno));
	}

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
	detect_unavail_pmu_regs(&sif, &inp.pfp_unavail_pmcs, NULL);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("Cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
	/*
	 * We want to get notified when the counter used for our first
	 * event overflows
	 */
	pd[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY;

	if (inp.pfp_event_count > 1) {
		pfm_bv_set(pd[0].reg_reset_pmds, pd[1].reg_num);
		pdx[0].reg_num = pd[1].reg_num;
	}

	/*
	 * we arm the first counter, such that it will overflow
	 * after SMPL_PERIOD events have been observed
	 */
	pd[0].reg_value       = - SMPL_PERIOD;
	pd[0].reg_long_reset  = - SMPL_PERIOD;
	pd[0].reg_short_reset = - SMPL_PERIOD;

	/*
	 * Now program the registers
	 */
	if (pfm_write(ctx_fd, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)))
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(ctx_fd, 0, PFM_RW_PMD_ATTR, pd, outp.pfp_pmd_count * sizeof(*pd)))
		fatal_error("pfm_write(PMD) error errno %d\n",errno);

	/*
	 * we want to monitor ourself
	 */
	if (pfm_attach(ctx_fd, 0, getpid()))
		fatal_error("pfm_attach error errno %d\n",errno);

	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL, 0) | O_ASYNC);
	if (ret == -1)
		fatal_error("cannot set ASYNC: %s\n", strerror(errno));

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(ctx_fd, F_SETOWN, getpid());
	if (ret == -1)
		fatal_error("cannot setown: %s\n", strerror(errno));
#ifndef _GNU_SOURCE
#error "this program must be compiled with -D_GNU_SOURCE"
#else
	/*
	 * when you explicitely declare that you want a particular signal,
	 * even with you use the default signal, the kernel will send more
	 * information concerning the event to the signal handler.
	 *
	 * In particular, it will send the file descriptor from which the
	 * event is originating which can be quite useful when monitoring
	 * multiple tasks from a single thread.
	 */
	ret = fcntl(ctx_fd, F_SETSIG, SIGIO);
	if (ret == -1)
		fatal_error("cannot setsig: %s\n", strerror(errno));
#endif
	/*
	 * Let's roll now
	 */
	if (pfm_set_state(ctx_fd, 0, PFM_ST_START))
		fatal_error("pfm_set_state(start) error errno %d\n", errno);

	busyloop();

	if (pfm_set_state(ctx_fd, 0, PFM_ST_STOP))
		fatal_error("pfm_set_state(stop) error errno %d\n", errno);

	/*
	 * destroy our context
	 */
	close(ctx_fd);

	if (event1_name)
		free(event1_name);

	return 0;
}
