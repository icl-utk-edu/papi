/*
 * task_attach_timeout_np.c - attach to another task without ptrace()
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
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
#include <fcntl.h>
#include <sys/wait.h>
#include <err.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128

int
parent(pid_t pid, unsigned long delay)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_ctx_t ctx[1];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_pmd_t pd[NUM_PMDS];
	uint64_t prev_pd[NUM_PMDS];
	pfarg_load_t load_args;
	pfarg_msg_t msg;
	unsigned int i, num_counters;
	int ret;
	int ctx_fd;
	char name[MAX_EVT_NAME_LEN];

	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&load_args,0, sizeof(load_args));
	memset(prev_pd,0, sizeof(prev_pd));

	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		errx(1, "cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		errx(1, "cannot find inst retired event\n");

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
		if (errno == ENOSYS)
			errx(1, "your kernel does not have performance monitoring support!\n");
		err(1, "cannot create PFM context");
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
		errx(1, "cannot configure events: %s\n", pfm_strerror(ret));

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

	if (pfm_write_pmcs(ctx_fd, pc, outp.pfp_pmc_count))
		err(1, "pfm_write_pmcs error");

	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(ctx_fd, pd, outp.pfp_pmd_count))
		err(1, "pfm_write_pmds error");

	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = pid;
	if (pfm_load_context(ctx_fd, &load_args))
		err(1, "pfm_load_context");

	/*
	 * activate monitoring. The task is still STOPPED at this point. Monitoring
	 * will not take effect until the execution of the task is resumed.
	 */
	if (pfm_start(ctx_fd, NULL))
		err(1, "pfm_start");

	/*
	 * now resume execution of the task, effectively activating
	 * monitoring.
	 */
	printf("attached to [%d], timeout set to %lu seconds\n", pid, delay);

	/*
 	 * we wil be polling on the context fd, so enable non-blocking mode
 	 */
	ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL) | O_NONBLOCK);
	if (ret)
		errx(1, "fcntl");

	for(;delay--;) {
		sleep(1);

		/*
		 * read the results, no stopping necessary
		 */
		if (pfm_read_pmds(ctx_fd, pd, inp.pfp_event_count))
			err(1, "pfm_read_pmds");

		/*
		 * print the results
		 */
		for (i=0; i < inp.pfp_event_count; i++) {

			pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);

			printf("PMD%-3u %20"PRIu64" %s\n",
					pd[i].reg_num,
					pd[i].reg_value - prev_pd[i],
					name);

			prev_pd[i] = pd[i].reg_value;
		}

		/*
		 * check if task has exited
		 */
		ret = read(ctx_fd, &msg, sizeof(msg));
		if (ret == sizeof(msg)) {
			if (msg.type != PFM_MSG_END)
				errx(1, "unexpected msg type : %d\n", msg.type);
			printf("[%d] terminated\n", pid);
			goto done;
		}
	}

	if (pfm_unload_context(ctx_fd))
		err(1, "pfm_unload_context");

	printf("detached from [%d]\n", pid);
done:
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
		errx(1, "usage: %s pid [timeout]\n", argv[0]);

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
		errx(1, "cannot initialize library: %s\n", pfm_strerror(ret));

	return parent(pid, delay);
}
