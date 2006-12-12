/*
 * mont_opcode.c - example of how to use the opcode matcher with the Dual-Core Itanium 2 PMU
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib_montecito.h>

#define OPCM_EVENT "IA64_TAGGED_INST_RETIRED_IBRP0_PMC32_33"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128
#define MAX_PMU_NAME_LEN	32

#define NLOOP 			200UL

/*
 * we don't use static to make sure the compiler does not inline the function
 */
int
do_test(unsigned long loop)
{
	unsigned long sum = 0;
	while(loop--) sum += loop;
	return sum;
}


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
main(void)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_mont_input_param_t mont_inp;
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_ctx_t ctx;
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	int ret;
	int type = 0;
	int id;
	unsigned int i;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("Can't initialize library\n");

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_MONTECITO_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with the %s PMU\n", model);
	}

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));

	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&mont_inp,0, sizeof(mont_inp));

	/*
	 * We indicate that we are using the first opcode matcher (PMC32/PMC33).
	 */
	mont_inp.pfp_mont_opcm1.opcm_used = 1;

	/*
	 * We want to match all the br.cloop in our test function.
	 * This branch is an IP-relative branch for which the major
	 * opcode (bits [40-37]) is 4 and the btype field (bits[6-8]) is 5.
	 * We ignore all the other fields in the opcode.
	 *
	 * On Montecito, the opcode matcher covers the full 41 bits of each
	 * instruction but we'll ignore them in this example. Hence the
	 * match value is:
	 *
	 * 	match = (4<<37)| (5<<6) = 0x8000000140
	 *
	 * On Montecito, the match field covers the full 41 bits of each instruction.
	 * But for this example, we only care about the major and btype field,
	 * and we ignore all other bits. When a bit is set in the mask it means
	 * that the corresponding match bit value is a "don't care". A bit
	 * with value of zero indicates that the corresponding match bit
	 * must match. Hence we build the following mask:
	 *
	 *	mask = ~((0xf<<37) | (0x3<<6)) = 0x1fffffff3f;
	 *
	 * The 0xf comes from the fact that major opcode is 4-bit wide.
	 * The 0x3 comes from the fact that btype is 3-bit wide.
	 */
	mont_inp.pfp_mont_opcm1.opcm_b     = 1;
	mont_inp.pfp_mont_opcm1.opcm_match = 0x8000000140;
	mont_inp.pfp_mont_opcm1.opcm_mask  = 0x1fffffff3f;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_full_event(OPCM_EVENT, &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find event %s\n", OPCM_EVENT);

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, &mont_inp, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * now create the context for self monitoring/per-task
	 */
	id = pfm_create_context(&ctx, NULL, NULL, 0);
	if (id == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
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
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (pfm_write_pmcs(id, pc, outp.pfp_pmc_count))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	if (pfm_write_pmds(id, pd, outp.pfp_pmd_count))
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = getpid();

	if (pfm_load_context(id, &load_args))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * Let's roll now.
	 */
	pfm_self_start(id);

	do_test(NLOOP);

	pfm_self_stop(id);

	/*
	 * now read the results
	 */
	if (pfm_read_pmds(id, pd, inp.pfp_event_count))
		fatal_error("pfm_read_pmds error errno %d\n",errno);

	/*
	 * print the results
	 */
	pfm_get_full_event_name(&inp.pfp_events[0], name, MAX_EVT_NAME_LEN);

	printf("PMD%-3u %20lu %s (expected %lu)\n",
		pd[0].reg_num,
		pd[0].reg_value,
		name, NLOOP);

	if (pd[0].reg_value != 0)
		printf("compiler used br.cloop\n");
	else
		printf("compiler did not use br.cloop\n");

	/*
	 * let's stop this now
	 */
	close(id);
	return 0;
}
