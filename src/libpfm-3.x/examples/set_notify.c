/*
 * set_notify.c - example of how to get notification at the end of a set chain
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

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#include "detect_pmcs.h"

#define NUM_SETS	3
#define THE_TIMEOUT	1

static volatile unsigned long notification_received;

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static int ctx_fd;
static char *event1_name;
static pfarg_setinfo_t setinfo[NUM_SETS];
static pfarg_pmd_t pd[2];

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
	unsigned int k;
	int r;

retry:
	r = read(ctx_fd, &msg, sizeof(msg));
	if (r != sizeof(msg)) {
		if(r == -1 && errno == EINTR) {
			warning("read interrupted, retrying\n");
			goto retry;
		}
		fatal_error("cannot read overflow message: %s\n", strerror(errno));
	}

	if (msg.type != PFM_MSG_OVFL) {
		fatal_error("unexpected msg type: %d\n",msg.type);
	}

	if (pfm_getinfo_evtsets(ctx_fd, setinfo, NUM_SETS) == -1) {
		fatal_error("pfm_getinfo_evtsets: %s", strerror(errno));
	}
	if (pfm_read_pmds(ctx_fd, pd, 2) == -1) {
		fatal_error("pfm_read_pmds: %s", strerror(errno));
	}
	/*
	 * XXX: risky to do printf() in signal handler!
	 */
	printf("Notification %lu: set%u pd[0]=%"PRIx64" pd[1]=%"PRIx64"\n",
		notification_received,
		pd[0].reg_set,
		pd[0].reg_value,
		pd[1].reg_value);

	for(k=0; k < NUM_SETS; k++)
		printf("set%u %"PRIu64" runs\n", setinfo[k].set_id, setinfo[k].set_runs);

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
}

#ifdef __ia64__
#define FUDGE 1
#else
#define FUDGE 0x100
#endif

/*
 * build end marker set
 */
void
setup_end_marker(int fd, unsigned int set_id, uint64_t num_ovfls, int plm_mask)
{
	pfarg_setdesc_t setdesc;
	pfarg_pmc_t pc[8];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	unsigned int i;
	int ret;

	memset(&setdesc, 0, sizeof(setdesc));
	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	/*
	 * we use the cycle event twice:
	 *   - first as sampling period to force switch to set 0
	 *   - second as sampling period to force notification
	 */
	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	inp.pfp_events[1] = inp.pfp_events[0];

	inp.pfp_dfl_plm     = plm_mask;
	inp.pfp_event_count = 2;

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
	detect_unavail_pmcs(fd, &inp.pfp_unavail_pmcs);

	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("Cannot configure events: %s\n", pfm_strerror(ret));

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
		pc[i].reg_set   = set_id;
	}
	for (i=0; i < outp.pfp_pmd_count; i++) {
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
		pd[i].reg_set = set_id;
	}

	/*
	 * first cycle overflow: no notification, simply trigger a switch
	 */
	pd[0].reg_flags		  = 0;
	pd[0].reg_value           = -1;
	pd[0].reg_long_reset      = -1;
	pd[0].reg_short_reset     = -1;
	pd[0].reg_ovfl_switch_cnt = 1;

	/*
	 * second cycle overflow: generate notification, switch on restart
	 */
	pd[1].reg_flags		  = PFM_REGFL_OVFL_NOTIFY;
	pd[1].reg_value           = -num_ovfls*FUDGE;
	pd[1].reg_long_reset      = -num_ovfls*FUDGE;
	pd[1].reg_short_reset     = -num_ovfls*FUDGE;
	pd[1].reg_ovfl_switch_cnt =  1;

	/*
	 * set uses overflow switch
	 */
	setdesc.set_id              = set_id;
	setdesc.set_flags           = PFM_SETFL_OVFL_SWITCH;
	setdesc.set_timeout	    = 0;

	if (pfm_create_evtsets(fd, &setdesc, 1) == -1)
		fatal_error("pfm_create_evtsets error errno %d\n",errno);

	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count) == -1)
		fatal_error("pfm_write_pmcs error errno %d\n",errno);
	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(fd, pd, outp.pfp_pmd_count) == -1)
		fatal_error("pfm_write_pmds error errno %d\n",errno);
}

int
main(int argc, char **argv)
{
	pfarg_ctx_t ctx[1];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_load_t load_args;
	pfarg_setdesc_t setdesc;
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	uint64_t num_ovfls;
	unsigned int i, k;
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

	num_ovfls = argc > 1 ? strtoull(argv[1], NULL, 10) : 3;

	printf("chain contains %d sets, time switching every %u seconds\n"
	       "notification every %"PRIu64" times the end of the chain is reached\n",
		NUM_SETS,
		THE_TIMEOUT,
		num_ovfls);

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
	memset(ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&setdesc,0, sizeof(setdesc));

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	pfm_get_max_event_name_len(&len);
	event1_name = malloc(len+1);
	if (event1_name == NULL) {
		fatal_error("cannot allocate event name\n");
	}
	pfm_get_full_event_name(&inp.pfp_events[1], event1_name, len+1);

	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = 1;

	/*
	 * now create the context for self monitoring/per-task
	 */
	ctx_fd = pfm_create_context(ctx, NULL, NULL, 0);
	if (ctx_fd == -1) {
		if (errno == ENOSYS)
			fatal_error("Your kernel does not have performance monitoring support!\n");
		fatal_error("Can't create PFM context %s\n", strerror(errno));
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
	detect_unavail_pmcs(ctx_fd, &inp.pfp_unavail_pmcs);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("Cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of registers to
	 * setup. Note that this field can be >= pfp_event_count.
	 */

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;

	pd[0].reg_value           = 0;
	pd[0].reg_long_reset      = 0;
	pd[0].reg_short_reset     = 0;
	pd[0].reg_ovfl_switch_cnt = 0;

	for(k=0; k < NUM_SETS; k++) {

		setdesc.set_id              = setinfo[k].set_id = k;
		setdesc.set_flags           = PFM_SETFL_TIME_SWITCH;
		setdesc.set_timeout	    = THE_TIMEOUT * 1000000000; /* in nsecs */

		for (i=0; i < outp.pfp_pmc_count; i++)
			pc[i].reg_set = k;

		for (i=0; i < outp.pfp_pmd_count; i++)
			pd[i].reg_set = k;

		if (pfm_create_evtsets(ctx_fd, &setdesc, 1) == -1)
			fatal_error("pfm_create_evtsets error errno %d\n",errno);

		if (pfm_write_pmcs(ctx_fd, pc, outp.pfp_pmc_count) == -1)
			fatal_error("pfm_write_pmcs error errno %d\n",errno);

		/*
	 	 * To be read, each PMD must be either written or declared
	 	 * as being part of a sample (reg_smpl_pmds)
	 	 */
		if (pfm_write_pmds(ctx_fd, pd, outp.pfp_pmd_count) == -1)
			fatal_error("pfm_write_pmds error errno %d\n",errno);
	}
	setup_end_marker(ctx_fd, k, num_ovfls, inp.pfp_dfl_plm);

	/*
	 * we want to monitor ourself
	 */
	load_args.load_pid = getpid();
	load_args.load_set = 0;

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
	 * Let's roll now
	 */

	pfm_start(ctx_fd, NULL);

	busyloop();

	pfm_stop(ctx_fd);

	close(ctx_fd);

	free(event1_name);

	return 0;
}
