/*
 * ita2_rr.c - example of how to use data range restriction with the Itanium2 PMU
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
#include <perfmon/pfmlib_itanium2.h>

#ifdef __GNUC__
extern inline void
clear_psr_ac(void)
{
	__asm__ __volatile__("rum psr.ac;;" ::: "memory" );
}
#elif defined(INTEL_ECC_COMPILER)
#define clear_psr_ac()	__rum(1UL<<3)
#else
#error "You need to define clear_psr_ac() for your compiler"
#endif

#define TEST_DATA_COUNT	16
#define N_LOOP	100000000UL
#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

/*
 * here we capture only misaligned_loads because it cannot
 * be measured with misaligned_stores_retired at the same time
 */
static char *event_list[]={
	"misaligned_loads_retired",
	NULL
};


typedef union {
	unsigned long   l_tab[2];
	unsigned int    i_tab[4];
	unsigned short  s_tab[8];
	unsigned char   c_tab[16];
} test_data_t;

static int
do_test(test_data_t *data)
{
	unsigned int *l, v;

	l = (unsigned int *)(data->c_tab+1);

	if (((unsigned long)l & 0x1) == 0) {
		printf("Data is not unaligned, can't run test\n");
		return  -1;
	}

	v = *l;
	v++;
	*l = v;

	return 0;
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
main(int argc, char **argv)
{
	char **p;
	test_data_t *test_data, *test_data_fake;
	unsigned long range_start, range_end;
	int ret, i, type = 0;
	pid_t pid = getpid();
	pfmlib_param_t evt;
	pfmlib_ita2_param_t ita2_param;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_dbreg_t dbr[8];
	pfarg_context_t ctx[1];
	pfmlib_options_t pfmlib_options;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("Can't initialize library\n");
	}

	/*
	 * Let's make sure we run this on the right CPU family
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM2_PMU) {
		char *model; 
		pfm_get_pmu_name(&model);
		fatal_error("this program does not work with %s PMU\n", model);
	}
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	/*
	 * now let's allocate the data structure we will be monitoring
	 */
	test_data = (test_data_t *)malloc(sizeof(test_data_t)*TEST_DATA_COUNT);
	if (test_data == NULL) {
		fatal_error("cannot allocate test data structure");
	}
	test_data_fake = (test_data_t *)malloc(sizeof(test_data_t)*TEST_DATA_COUNT);
	if (test_data_fake == NULL) {
		fatal_error("cannot allocate test data structure");
	}
	/*
	 * Compute the range we are interested in
	 */
	range_start = (unsigned long)test_data;
	range_end   = range_start + sizeof(test_data_t)*TEST_DATA_COUNT;
	
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(dbr,0, sizeof(dbr));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&evt,0, sizeof(evt));
	memset(&ita2_param,0, sizeof(ita2_param));


	/*
	 * because we use a model specific feature, we must initialize the
	 * model specific pfmlib parameter structure and link it to the
	 * common structure.
	 * The magic number is a simple mechanism used by the library to check
	 * that the model specific data structure is decent. You must set it manually
	 * otherwise the model specific feature won't work.
	 */
	ita2_param.pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;
	evt.pfp_model       = &ita2_param;

	/*
	 * find requested event
	 */
	p = event_list;
	for (i=0; *p ; i++, p++) {
		if (pfm_find_event(*p, &evt.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find %s event\n", *p);
		}
	}


	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	evt.pfp_dfl_plm   = PFM_PLM3; 
	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = i;

	/*
	 * We use the library to figure out how to program the debug registers
	 * to cover the data range we are interested in. The rr_end parameter
	 * must point to the byte after the last element of the range (C-style range).
	 *
	 * Because of the masking mechanism and therefore alignment constraints used to implement 
	 * this feature, it may not be possible to exactly cover a given range. It may be that
	 * the coverage exceeds the desired range. So it is possible to capture noise if
	 * the surrounding addresses are also heavily used. You can figure out by how much the
	 * actual range is off compared to the requested range by checking the rr_soff and rr_eoff 
	 * fields on return from the library call.
	 *
	 * Upon return, the rr_dbr array is programmed and the number of debug registers (not pairs)
	 * used to cover the range is in rr_nbr_used. 
	 */

	ita2_param.pfp_ita2_drange.rr_used = 1;
	ita2_param.pfp_ita2_drange.rr_limits[0].rr_start = range_start;
	ita2_param.pfp_ita2_drange.rr_limits[0].rr_end   = range_end;


	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}

	printf("data range  : [0x%016lx-0x%016lx): %d pair of debug registers used\n"
	       "start_offset:-0x%lx end_offset:+0x%lx\n", 
			range_start, 
			range_end, 
			ita2_param.pfp_ita2_drange.rr_nbr_used >> 1, 
			ita2_param.pfp_ita2_drange.rr_limits[0].rr_soff, 
			ita2_param.pfp_ita2_drange.rr_limits[0].rr_eoff);

	printf("fake data range: [0x%016lx-0x%016lx)\n", 
			(unsigned long)test_data_fake,
			(unsigned long)test_data_fake+sizeof(test_data_t)*TEST_DATA_COUNT);

	/*
	 * for this example, we have decided not to get notified
	 * on counter overflows and the monitoring is not to be inherited
	 * in derived tasks
	 */
	ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(pid, PFM_CREATE_CONTEXT, ctx, 1) == -1) {
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
		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * Now prepare the argument to initialize the PMDs.
	 * the memset(pd) initialized the entire array to zero already, so
	 * we just have to fill in the register numbers from the pc[] array.
	 */
	for (i=0; i < evt.pfp_event_count; i++) {
		pd[i].reg_num = evt.pfp_pc[i].reg_num;
	}

	/*
	 * Program the data debug registers. 
	 *
	 * IMPORTANT: programming the debug register MUST always be done before the PMCs
	 * otherwise the kernel will fail on PFM_WRITE_PMCS. This is for security reasons.
	 */
	if (perfmonctl(pid, PFM_WRITE_DBRS, ita2_param.pfp_ita2_drange.rr_br, ita2_param.pfp_ita2_drange.rr_nbr_used) == -1) {
		fatal_error( "child: perfmonctl error PFM_WRITE_DBRS errno %d\n",errno);
	}

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more than coutning monitors.
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt.pfp_pc, evt.pfp_pc_count) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}

	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error( "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/* 
	 * Let's make sure that the hardware does the unaligned accesses (do not use the
	 * kernel software handler otherwise the PMU won't see the unaligned fault).
	 */
	clear_psr_ac();

	/*
	 * Let's roll now.
	 *
	 * The idea behind this test is to have two dynamically allocated data structures
	 * which are access in a unaligned fashion. But we want to capture only the unaligned
	 * accesses on one of the two. So the debug registers are programmed to cover the
	 * first one ONLY. Then we activate monotoring and access the two data structures.
	 * This is an artificial example just to demonstrate how to use data address range
	 * restrictions.
	 */
	pfm_start();

	for(i=0; i < N_LOOP; i++) {
		do_test(test_data);
		do_test(test_data_fake);
	}

	pfm_stop();

	/* 
	 * now read the results
	 */
	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error( "perfmonctl error READ_PMDS errno %d\n",errno);
	}

	/* 
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 *
	 * For this example, we expect to see a value of 1 for misaligned loads.
	 * But it can be two when the test_data and test_data_fake
	 * are allocated very close from each other and the range created with the debug
	 * registers is larger then test_data.
	 *
	 */
	for (i=0; i < evt.pfp_event_count; i++) {
		char *name;
		pfm_get_event_name(evt.pfp_events[i].event, &name);
		printf("PMD%u %20lu %s (expected %lu)\n", 
			pd[i].reg_num, 
			pd[i].reg_value, 
			name, N_LOOP);

		if (pd[i].reg_value != N_LOOP) {
			printf("error: Result should be 1 for %s\n", name);
			break;
		}
	}
	/* 
	 * let's stop this now
	 */
	if (perfmonctl(pid, PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_DESTROY errno %d\n",errno);
	}
	free(test_data);
	free(test_data_fake);

	return 0;
}
