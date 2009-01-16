/*
 * notify_self.c - example of how you can use overflow notifications
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 * Modified by Phil Mucci <mucci@cs.utk.edu> to add the fork()
 * Adapted to v2.0 interface by Stephane Eranian <eranian@gmail.com>
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
 * applications on Linux/ia64.
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


#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#define SMPL_PERIOD	1000000000UL

static volatile unsigned long notification_received;

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static pfarg_reg_t pd[NUM_PMDS];
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
sigio_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	pfm_msg_t msg;
	int fd = ctx_fd;
	int r;

	if (fd != ctx_fd) {
		fatal_error("handler does not get valid file descriptor\n");
	}

	if (event1_name && perfmonctl(fd, PFM_READ_PMDS, pd+1, 1) == -1) {
		fatal_error("PFM_READ_PMDS: %s", strerror(errno));
	}

	r = read(fd, &msg, sizeof(msg));
	if (r != sizeof(msg)) {
		fatal_error("cannot read overflow message: %s\n", strerror(errno));
	}

	if (msg.type != PFM_MSG_OVFL) {
		fatal_error("unexpected msg type: %d\n",msg.type);
	}

	/*
	 * XXX: risky to do printf() in signal handler!
	 */
	if (event1_name)
		printf("Notification %lu: %"PRIu64" %s\n", notification_received, pd[1].reg_value, event1_name);
	else
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
	if (perfmonctl(fd, PFM_RESTART,NULL, 0) == -1) {
		fatal_error("PFM_RESTART: %s", strerror(errno));
	}
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
	for(;notification_received < 3;) ;

	/*
	 * forking causes the context to be shared with the child
	 * When the child terminates, it closes its descriptor.
	 * The parent's remains and notification keep on coming.
	 */
	if (fork() == 0) {
		printf("child terminates\n");
		fflush(stdout);
		exit(0);
	}
	printf("after fork\n");
	fflush(stdout);
	for(;notification_received < 6;) ;
}

int
main(int argc, char **argv)
{
	pfarg_context_t ctx[1];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	unsigned int i, num_counters;
	size_t len;
	int ret;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}

	/*
	 * Install the signal handler (SIGIO)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)sigio_handler;
	sigaction (SIGIO, &act, 0);

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	memset(pc, 0, sizeof(pc));
	memset(ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

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

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	pfm_get_max_event_name_len(&len);

	event1_name = malloc(len+1);
	if (event1_name == NULL)
		fatal_error("cannot allocate event name\n");

	pfm_get_full_event_name(&inp.pfp_events[1], event1_name, len+1);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS) {
		fatal_error("Cannot configure events: %s\n", pfm_strerror(ret));
	}

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	ctx_fd = ctx->ctx_fd;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of registers to
	 * setup. Note that this field can be >= pfp_event_count.
	 */

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	for (i=0; i < inp.pfp_event_count; i++) {
		pd[i].reg_num   = pc[i].reg_num;
	}
	/*
	 * We want to get notified when the counter used for our first
	 * event overflows
	 */
	pc[0].reg_flags 	|= PFM_REGFL_OVFL_NOTIFY;
	pc[0].reg_reset_pmds[0] |= 1UL << outp.pfp_pmcs[1].reg_num;

	/*
	 * we arm the first counter, such that it will overflow
	 * after SMPL_PERIOD events have been observed
	 */
	pd[0].reg_value       = (~0UL) - SMPL_PERIOD + 1;
	pd[0].reg_long_reset  = (~0UL) - SMPL_PERIOD + 1;
	pd[0].reg_short_reset = (~0UL) - SMPL_PERIOD + 1;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more than counting monitors.
	 */
	if (perfmonctl(ctx_fd, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}

	if (perfmonctl(ctx_fd, PFM_WRITE_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * we want to monitor ourself
	 */
	load_args.load_pid = getpid();

	if (perfmonctl(ctx_fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
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
	 * Let's roll now
	 */
	pfm_self_start(ctx_fd);

	busyloop();

	pfm_self_stop(ctx_fd);

	/*
	 * free our context
	 */
	close(ctx_fd);

	return 0;
}
