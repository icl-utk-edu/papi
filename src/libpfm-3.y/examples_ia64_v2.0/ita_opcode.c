/*
 * ita_opcode.c - example of how to use the opcode matcher with the Itanium PMU
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
 * applications on Linux/ia64.
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
#include <perfmon/pfmlib_itanium.h>

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128
#define MAX_PMU_NAME_LEN	32

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
	int ret;
	int type = 0;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_ita_input_param_t ita_inp;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_context_t ctx[1];
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	unsigned int i;
	int id;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("Can't initialize library\n");
	}

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with the %s PMU\n", model);
	}

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);



	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&ita_inp,0, sizeof(ita_inp));
	memset(&load_args,0, sizeof(load_args));

	/*
	 * We indicate that we are using the PMC8 opcode matcher. This is required
	 * otherwise the library add PMC8 to the list of PMC to pogram during
	 * pfm_dispatch_events().
	 */
	ita_inp.pfp_ita_pmc8.opcm_used = 1;

	/*
	 * We want to match all the br.cloop in our test function.
	 * This branch is an IP-relative branch for which the major
	 * opcode (bits [40-37]=4) and the btype field is 5 (which represents
	 * bits[6-8]) so it is included in the match/mask fields of PMC8.
	 * It is necessarily in a B slot.
	 *
	 * We don't care which operands are used with br.cloop therefore
	 * the mask field of pmc8 is set such that only the 4 bits of the
	 * opcode and 3 bits of btype must match exactly. This is accomplished by
	 * clearing the top 4 bits and bits [6-8] of the mask field and setting the
	 * remaining bits.  Similarly, the match field only has the opcode value  and btype
	 * set according to the encoding of br.cloop, the
	 * remaining bits are zero. Bit 60 of PMC8 is set to indicate
	 * that we look only in B slots  (this is the only possibility for
	 * this instruction anyway).
	 *
	 * So the binary representation of the value for PMC8 is as follows:
	 *
	 * 6666555555555544444444443333333333222222222211111111110000000000
	 * 3210987654321098765432109876543210987654321098765432109876543210
	 * ----------------------------------------------------------------
	 * 0001010000000000000000101000000000000011111111111111000111111000
	 *
	 * which yields a value of 0x1400028003fff1f8.
	 *
	 * Depending on the level of optimization to compile this code, it may
	 * be that the count reported could be zero, if the compiler uses a br.cond
	 * instead of br.cloop.
	 */
	ita_inp.pfp_ita_pmc8.pmc_val = 0x1400028003fff1f8;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_full_event("IA64_TAGGED_INST_RETIRED_PMC8", &inp.pfp_events[0]) != PFMLIB_SUCCESS) {
		fatal_error("Cannot find event IA64_TAGGED_INST_RETIRED_PMC8\n");
	}

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
	if ((ret=pfm_dispatch_events(&inp, &ita_inp, &outp, NULL)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
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
	/*
	 * extract our file descriptor
	 */
	id = ctx[0].ctx_fd;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We must pfp_pmc_count to determine the number of PMC to intialize.
	 * We must use pfp_event_count to determine the number of PMD to initialize.
	 * Some events cause extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
	 *
	 * This step is new compared to libpfm-2.x. It is necessary because the library no
	 * longer knows about the kernel data structures.
	 */

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
	 * the PMC controlling the event ALWAYS come first, that's why this loop
	 * is safe even when extra PMC are needed to support a particular event.
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pd[i].reg_num   = pc[i].reg_num;
	}

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(id, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(id, PFM_WRITE_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}
	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = getpid();

	if (perfmonctl(id, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
		fatal_error("perfmonctl error PFM_LOAD_CONTEXT errno %d\n",errno);
	}

	/*
	 * Let's roll now.
	 */
	pfm_self_start(id);

	do_test(100UL);

	pfm_self_stop(id);

	/*
	 * now read the results
	 */
	if (perfmonctl(id, PFM_READ_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error( "perfmonctl error READ_PMDS errno %d\n",errno);
	}

	/*
	 * print the results
	 */
	pfm_get_full_event_name(&inp.pfp_events[0], name, MAX_EVT_NAME_LEN);
	printf("PMD%u %20lu %s\n",
			pd[0].reg_num,
			pd[0].reg_value,
			name);

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
