/*
 * self_smpl.c - example of self sampling using a kernel samplig buffer
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on mont_dear.c from:
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <err.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

#define SMPL_PERIOD	(2400000)

typedef pfm_dfl_smpl_hdr_t	smpl_hdr_t;
typedef pfm_dfl_smpl_entry_t	smpl_entry_t;
typedef pfm_dfl_smpl_arg_t	smpl_arg_t;

static int fd;
static void *smpl_vaddr;
static size_t entry_size;

long
do_test(unsigned long size)
{
    unsigned long i, sum  = 0;
    int *array;

    array = (int *)malloc(size * sizeof(int));
    if (array == NULL ) {
        printf("line = %d No memory available!\n", __LINE__);
        exit(1);
    }
    for(i=0; i<size; i++) {
        array[i]=1;
    }
    free(array);
    return sum;
}

static void
process_smpl_buffer(void)
{
	static uint64_t last_ovfl = ~0UL;
	static uint64_t smpl_entry = 0;
	smpl_hdr_t *hdr;
	smpl_entry_t *ent;
	unsigned long pos;
	uint64_t count;

	hdr = (smpl_hdr_t *)smpl_vaddr;

	/*
	 * check that we are not diplaying the previous set of samples again.
	 * Required to take care of the last batch of samples.
	 */
	if (hdr->hdr_overflows <= last_ovfl && last_ovfl != ~0UL) {
		printf("skipping identical set of samples %"PRIu64" <= %"PRIu64"\n", hdr->hdr_overflows, last_ovfl);
		return;
	}

	pos = (unsigned long)(hdr+1);
	count = hdr->hdr_count;
	/*
	 * walk through all the entries recored in the buffer
	 */
	while(count--) {

		ent = (smpl_entry_t *)pos;
		/*
		 * print entry header
		 */
		printf("Entry %"PRIu64" PID:%d TID:%d CPU:%d STAMP:0x%"PRIx64" IIP:0x%016"PRIx64"\n",
			smpl_entry++,
			ent->tgid,
			ent->pid,
			ent->cpu,
			ent->tstamp,
			ent->ip);

		/*
		 * move to next entry
		 */
		pos += entry_size;
	}
}

static void
overflow_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	process_smpl_buffer();
	/*
	 * And resume monitoring
	 */
	if (pfm_restart(fd))
		errx(1, "pfm_restart");
}

int
main(void)
{
	pfarg_pmd_t pd[8];
	pfarg_pmc_t pc[8];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_ctx_t ctx;
	pfarg_load_t load_args;
	smpl_arg_t buf_arg;
	pfmlib_options_t pfmlib_options;
	unsigned long nloop = 10000;
	struct sigaction act;
	unsigned int i;
	int ret;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS)
		errx(1, "cannot initialize library\n");

	/*
	 * Install the overflow handler (SIGIO)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)overflow_handler;
	sigaction (SIGIO, &act, 0);

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);



	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(pc, 0, sizeof(pc));
	memset(&ctx, 0, sizeof(ctx));
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&load_args, 0, sizeof(load_args));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		errx(1, "cannot find cycle event\n");

	/*
	 * set the (global) privilege mode:
	 * 	PFM_PLM0 : kernel level only
	 */
	inp.pfp_dfl_plm   = PFM_PLM3|PFM_PLM0;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 *
	 * We use all global settings for this EAR.
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		errx(1, "cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 *
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	buf_arg.buf_size = getpagesize();

	/*
	 * do not generate overflow notification messages
	 */
	ctx.ctx_flags = PFM_FL_OVFL_NO_MSG;

	/*
	 * now create the context for self monitoring/per-task
	 */
	fd = pfm_create_context(&ctx, PFM_DFL_SMPL_NAME, &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS)
			errx(1, "kernel does not have performance monitoring support!\n");
		errx(1, "cannot create PFM context %s\n", strerror(errno));
	}

	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	smpl_vaddr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (smpl_vaddr == MAP_FAILED)
		errx(1, "cannot mmap sampling buffer errno %d\n", errno);

	printf("Sampling buffer mapped at %p\n", smpl_vaddr);

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

	/*
	 * figure out pmd mapping from output pmc
	 */
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num   = outp.pfp_pmds[i].reg_num;

	/*
	 * indicate we want notification when buffer is full
	 */
	pd[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
	entry_size = sizeof(smpl_entry_t);

	/*
	 * initialize the PMD and the sampling period
	 */
	pd[0].reg_value       = - SMPL_PERIOD;
	pd[0].reg_long_reset  = - SMPL_PERIOD;
	pd[0].reg_short_reset = - SMPL_PERIOD;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count))
		errx(1, "pfm_write_pmcs error errno %d\n",errno);

	if (pfm_write_pmds(fd, pd, outp.pfp_pmd_count))
		errx(1, "pfm_write_pmds error errno %d\n",errno);

	/*
	 * attach context to stopped task
	 */
	load_args.load_pid = getpid();
	if (pfm_load_context(fd, &load_args))
		errx(1, "pfm_load_context error errno %d\n",errno);

	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_ASYNC);
	if (ret == -1)
		errx(1, "cannot set ASYNC: %s\n", strerror(errno));

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(fd, F_SETOWN, getpid());
	if (ret == -1)
		errx(1, "cannot setown: %s\n", strerror(errno));

	/*
	 * Let's roll now.
	 */
	ret = pfm_start(fd, NULL);
	if (ret == -1)
		errx(1, "cannot pfm_start: %s\n", strerror(errno));

	while(nloop--)
		do_test(100000);

	ret = pfm_stop(fd);
	if (ret == -1)
		errx(1, "cannot pfm_stop: %s\n", strerror(errno));

	/*
	 * We must call the processing routine to cover the last entries recorded
	 * in the sampling buffer, i.e. which may not be full
	 */
	process_smpl_buffer();

	/*
	 * let's stop this now
	 */
	munmap(smpl_vaddr, (size_t)buf_arg.buf_size);
	close(fd);
	return 0;
}
