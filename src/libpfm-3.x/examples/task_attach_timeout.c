/*
 * task_attach_timeout.c - attach to another task for monitoring for a short while
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
 */
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/poll.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128

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

int
parent(pid_t pid, unsigned long delay)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_ctx_t ctx[1];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_load_t load_args;
	struct pollfd pollfd;
	pfarg_msg_t msg;
	unsigned int i, num_counters;
	int status, ret;
	int ctx_fd;
	char name[MAX_EVT_NAME_LEN];

	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&load_args,0, sizeof(load_args));

	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS) {
		fatal_error("cannot find cycle event\n");
	}

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS) {
		fatal_error("cannot find inst retired event\n");
	}
	i = 2;

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}
	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * now create a context. we will later attach it to the task we are creating.
	 */
	ctx_fd = pfm_create_context(ctx, NULL, NULL, 0);
	if (ctx_fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
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
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * use our file descriptor for the poll.
	 * we are interested in read events only.
	 */
	pollfd.fd     = ctx_fd;
	pollfd.events = POLLIN;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We must pfp_pmc_count to determine the number of PMC to intialize.
	 * We must use pfp_event_count to determine the number of PMD to initialize.
	 * Some events causes extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
	 *
	 * This step is new compared to libpfm-2.x. It is necessary because the library no
	 * longer knows about the kernel data structures.
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann counting monitors.
	 */

	if (pfm_write_pmcs(ctx_fd, pc, outp.pfp_pmc_count) == -1)
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(ctx_fd, pd, outp.pfp_pmd_count) == -1)
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	ret = ptrace(PTRACE_ATTACH, pid, NULL, 0);
	if (ret == -1) {
		fatal_error("cannot attach to %d: %s\n", pid, strerror(errno));
	}

	/*
	 * wait for the child to be actually stopped
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * check if process exited early
	 */
	if (WIFEXITED(status)) {
		fatal_error("command process %d exited too early with status %d\n", pid, WEXITSTATUS(status));
	}

	/*
	 * the task is stopped at this point
	 */
	
	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = pid;

	if (pfm_load_context(ctx_fd, &load_args) == -1) {
		fatal_error("pfm_load_context error errno %d\n",errno);
	}

	/*
	 * activate monitoring. The task is still STOPPED at this point. Monitoring
	 * will not take effect until the execution of the task is resumed.
	 */
	if (pfm_start(ctx_fd, NULL) == -1) {
		fatal_error("pfm_start error errno %d\n",errno);
	}

	/*
	 * now resume execution of the task, effectively activating
	 * monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);
	printf("attached to [%d], timeout set to %lu seconds\n", pid, delay);

	/*
	 * now the task is running
	 */

	/*
	 * We cannot simply do a waitpid() because we may be attaching to a process
	 * totally unrelated to our program. Instead we use a perfmon facility that
	 * notifies us when the monitoring task is exiting.
	 *
	 * When a task with a monitoring context attached to it exits, a PFM_MSG_END
	 * is generated. It can be retrieve with a simple read() on the context's descriptor.
	 *
	 * Another reason why you might return from the read is if there was a counter
	 * overflow, unlikely in this example.
	 *
	 * To measure only for short period of time, use select or poll with a timeout,
	 * see task_attach_timeout.c
	 *
	 */
	ret = poll(&pollfd, 1, delay*1000);
	switch( ret ) {
		case -1:
			fatal_error("cannot read from descriptor: %s\n", strerror(errno));
			/* no return */
		case  1:
			/*
	 		 * there is a message, i.e., the program exited before our timeout
	 		 */
			if (ret == 1) {
				/*
		 		* extract message
		 		*/
				ret = read(ctx_fd, &msg, sizeof(msg));

				if (msg.type != PFM_MSG_END) {
					fatal_error("unexpected msg type : %d\n", msg.type);
				}
			}
			break;
		case   0:
			/*
			 * we timed out, we need to stop the task to unload
			 */
			ret = ptrace(PTRACE_ATTACH, pid, NULL, 0);
			if (ret == -1) {
				fatal_error("cannot attach to %d: %s\n", pid, strerror(errno));
			}
			/*
			 * wait for task to be actually stopped
			 */
			waitpid(pid, &status, WUNTRACED);

			/*
	 		 * check if process exited, then no need to unload
	 		 */
			if (WIFEXITED(status)) goto read_results;

			if (pfm_unload_context(ctx_fd) == -1) {
				fatal_error("pfm_unload_context error errno %d\n",errno);
			}

			/*
			 * let it run free again
			 */
			ptrace(PTRACE_DETACH, pid, NULL, 0);
			break;
		default:
			fatal_error("unexpected return from poll: %d\n", ret);
	}

read_results:
	/*
	 * now simply read the results.
	 */
	if (pfm_read_pmds(ctx_fd, pd, inp.pfp_event_count) == -1) {
		fatal_error("pfm_read_pmds error errno %d\n",errno);
		return -1;
	}

	/*
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 *
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);
		printf("PMD%-3u %20"PRIu64" %s\n",
			pd[i].reg_num,
			pd[i].reg_value,
			name);
	}
	/*
	 * free the context
	 */
	close(ctx_fd);

	return 0;
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	unsigned long delay;
	pid_t pid;
	int ret;

	if (argc < 2)
		fatal_error("usage: %s pid [timeout]\n", argv[0]);

	pid   = atoi(argv[1]);
	delay = argc > 2 ? strtoul(argv[2], NULL, 10) : 10;

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	return parent(pid, delay);
}
