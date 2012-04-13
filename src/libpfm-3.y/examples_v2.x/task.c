/*
 * task.c - example of a task monitoring another one
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
child(char **arg)
{
	/*
	 * will cause the program to stop before executing the first
	 * user level instruction. We can only attach (load) a context
	 * if the task is in the STOPPED state.
	 */
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

	/*
	 * execute the requested command
	 */
	execvp(arg[0], arg);

	fatal_error("cannot exec: %s\n", arg[0]);
	/* not reached */
}

int
parent(char **arg)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_ctx_t ctx[1];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_load_t load_args;
	unsigned int i, num_counters;
	int status, ret;
	int ctx_fd;
	pid_t pid;
	char name[MAX_EVT_NAME_LEN];

	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&load_args,0, sizeof(load_args));

	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");
	i = 2;

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	if (num_counters < i) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}
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
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of PMC to intialize.
	 * We use pfp_pmd_count to determine the number of PMD to initialize.
	 * Some events/features may cause extra PMCs to be used, leading to:
	 * 	- pfp_pmc_count may be >= pfp_event_count
	 * 	- pfp_pmd_count may be >= pfp_event_count
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for(i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;

	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(ctx_fd, pc, outp.pfp_pmc_count) == -1)
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	if (pfm_write_pmds(ctx_fd, pd, outp.pfp_pmd_count) == -1)
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1) fatal_error("Cannot fork process\n");

	/*
	 * and launch the child code
	 */
	if (pid == 0) {
		close(ctx_fd);
		exit(child(arg));
	}

	/*
	 * wait for the child to exec
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * check if process exited early
	 */
	if (WIFEXITED(status))
		fatal_error("command %s exited too early with status %d\n", arg[0], WEXITSTATUS(status));

	/*
	 * the task is stopped at this point
	 */
	
	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = pid;
	if (pfm_load_context(ctx_fd, &load_args) == -1)
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * activate monitoring. The task is still STOPPED at this point. Monitoring
	 * will not take effect until the execution of the task is resumed.
	 */
	if (pfm_start(ctx_fd, NULL) == -1)
		fatal_error("pfm_start error errno %d\n",errno);

	/*
	 * now resume execution of the task, effectively activating
	 * monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	/*
	 * now the task is running
	 */

	/*
	 * simply wait for completion
	 */
	waitpid(pid, &status, 0);

	/*
	 * the task has disappeared at this point but our context is still
	 * present and contains all the latest counts.
	 */

	/*
	 * now simply read the results.
	 */
	if (pfm_read_pmds(ctx_fd, pd, inp.pfp_event_count) == -1)
		fatal_error("pfm_read_pmds error errno %d\n",errno);

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
	int ret;

	if (argc < 2) {
		fatal_error("You must specify a command to execute\n");
	}
	
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose= 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	return parent(argv+1);
}
