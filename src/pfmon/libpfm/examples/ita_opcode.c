/*
 * ita_opcode.c - example of how to use the opcode matcher with the Itanium PMU
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_itanium.h>

#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

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
	char *name;
	pid_t pid = getpid();
	pfmlib_param_t evt;
	pfmlib_ita_param_t ita_param;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_context_t ctx[1];
	pfmlib_options_t pfmlib_options;

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
		char *model; 
		pfm_get_pmu_name(&model);
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

	memset(&evt,0, sizeof(evt));
	memset(&ita_param,0, sizeof(ita_param));

	/*
	 * because we use a model specific feature, we must initialize the
	 * model specific pfmlib parameter structure and link it to the
	 * common structure.
	 * The magic number is a simple mechanism used by the library to check
	 * that the model specific data structure is decent. You must set it manually
	 * otherwise the model specific feature won't work.
	 */
	ita_param.pfp_magic = PFMLIB_ITA_PARAM_MAGIC;
	evt.pfp_model       = &ita_param;

	/*
	 * We indicate that we are using the PMC8 opcode matcher. This is required
	 * otherwise the library add PMC8 to the list of PMC to pogram during
	 * pfm_dispatch_events().
	 */
	ita_param.pfp_ita_pmc8.opcm_used = 1;

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
	ita_param.pfp_ita_pmc8.pmc_val = 0x1400028003fff1f8;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_event_byname("IA64_TAGGED_INST_RETIRED_PMC8", &evt.pfp_events[0].event) != PFMLIB_SUCCESS) {
		fatal_error("Cannot find event IA64_TAGGED_INST_RETIRED_PMC8\n");
	}

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	evt.pfp_dfl_plm   = PFM_PLM3; 
	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}
	/*
	 * for this example, we have decided not to get notified
	 * on counter overflows and the monitoring is not to be inherited
	 * in derived tasks
	 */
	ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(pid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/* 
	 * Must be done before any PMD/PMD calls (unfreeze PMU). Initialize
	 * PMC/PMD to safe values. psr.up is cleared.
	 */
	if (perfmonctl(pid, PFM_ENABLE, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * Now prepare the argument to initialize the PMD.
	 */
	pd[0].reg_num = evt.pfp_pc[0].reg_num;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt.pfp_pc, evt.pfp_pc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * Let's roll now.
	 */
	pfm_start();

	do_test(100UL);

	pfm_stop();

	/* 
	 * now read the results
	 */
	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error( "perfmonctl error READ_PMDS errno %d\n",errno);
	}

	/* 
	 * print the results
	 */
	pfm_get_event_name(evt.pfp_events[0].event, &name);
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
	if (perfmonctl(pid, PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_DESTROY errno %d\n",errno);
	}
	return 0;
}
