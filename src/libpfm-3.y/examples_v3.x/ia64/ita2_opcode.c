/*
 * ita2_opcode.c - example of how to use the opcode matcher with the Itanium2 PMU
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
#include <perfmon/pfmlib_itanium2.h>

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
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_ita2_input_param_t ita2_inp;
	pfarg_pmr_t pd[NUM_PMDS];
	pfarg_pmr_t pc[NUM_PMCS];
	pfmlib_options_t pfmlib_options;
	int ret;
	int type = 0;
	int id;
	unsigned int i;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM2_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with the %s PMU\n", model);
	}

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));

	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&ita2_inp,0, sizeof(ita2_inp));

	/*
	 * We indicate that we are using the PMC8 opcode matcher. This is required
	 * otherwise the library add PMC8 to the list of PMC to pogram during
	 * pfm_dispatch_events().
	 */
	ita2_inp.pfp_ita2_pmc8.opcm_used = 1;

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
	 *
	 *
	 * The 0x1 sets the ig_ad field to make sure we ignore any range restriction.
	 * Also bit 2 must always be set
	 */
	ita2_inp.pfp_ita2_pmc8.pmc_val = 0x1400028003fff1fa;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_full_event("IA64_TAGGED_INST_RETIRED_IBRP0_PMC8", &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find event IA64_TAGGED_INST_RETIRED_IBRP0_PMC8\n");

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
	if ((ret=pfm_dispatch_events(&inp, &ita2_inp, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	/*
	 * now create the session
	 */
	id = pfm_create(0, NULL);
	if (id == -1) {
		if (errno == ENOSYS)
			fatal_error("Your kernel does not have performance monitoring support!\n");
		fatal_error("cannot create session %s\n", strerror(errno));
	}
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
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (pfm_write(id, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)) == -1)
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(id, 0, PFM_RW_PMD, pd, outp.pfp_pmd_count * sizeof(*pd)) == -1)
		fatal_error("pfm_write(PMD) error errno %d\n",errno);

	/*
	 * now we attach
	 */
	if (pfm_attach(id, 0, getpid()) == -1)
		fatal_error("pfm_attach error errno %d\n",errno);

	/*
	 * Let's roll now.
	 */
	if (pfm_set_state(id, 0, PFM_ST_START))
		fatal_error("pfm_set_state error errno %d\n",errno);

	do_test(100UL);

	if (pfm_set_state(id, 0, PFM_ST_STOP))
		fatal_error("pfm_set_state error errno %d\n",errno);

	/*
	 * now read the results
	 */
	if (pfm_read(id, 0, PFM_RW_PMD, pd, inp.pfp_event_count * sizeof(*pd)) == -1)
		fatal_error("pfm_read error errno %d\n",errno);

	/*
	 * print the results
	 */
	pfm_get_full_event_name(&inp.pfp_events[0], name, MAX_EVT_NAME_LEN);
	printf("PMD%-3u %20lu %s\n",
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
