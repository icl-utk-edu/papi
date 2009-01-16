/*
 * ita_rr.c - example of how to use data range restriction with the Itanium PMU
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

#define N_LOOP 100000000U

#if defined(__ECC) && defined(__INTEL_COMPILER)

/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>

#define clear_psr_ac()	__rum(1UL<<3)

#elif defined(__GNUC__)

static inline void
clear_psr_ac(void)
{
	__asm__ __volatile__("rum psr.ac;;" ::: "memory" );
}
#else
#error "You need to define clear_psr_ac() for your compiler"
#endif

#define TEST_DATA_COUNT	16

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_PMU_NAME_LEN	32
#define MAX_EVT_NAME_LEN	128

typedef struct {
	char *event_name;
	unsigned long expected_value;
}  event_desc_t;


static event_desc_t event_list[]={
	{ "misaligned_loads_retired", N_LOOP },
	{ "misaligned_stores_retired", N_LOOP },
	{ NULL, 0UL}
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
	event_desc_t *p;
	test_data_t *test_data, *test_data_fake;
	unsigned long range_start, range_end;
	int ret, type = 0;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_ita_input_param_t ita_inp;
	pfmlib_ita_output_param_t ita_outp;
	pfarg_pmr_t pd[NUM_PMDS];
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_pmr_t dbrs[8];
	pfmlib_options_t pfmlib_options;
	unsigned int i;
	int id;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	/*
	 * Let's make sure we run this on the right CPU family
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_ITANIUM_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
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
	memset(pc, 0, sizeof(pc));
	memset(dbrs,0, sizeof(dbrs));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&ita_inp,0, sizeof(ita_inp));
	memset(&ita_outp,0, sizeof(ita_outp));


	/*
	 * find requested event
	 */
	p = event_list;
	for (i=0; p->event_name ; i++, p++) {
		if (pfm_find_event(p->event_name, &inp.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find %s event\n", p->event_name);
		}
	}


	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;
	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * We use the library to figure out how to program the debug registers
	 * to cover the data range we are interested in. The rr_end parameter
	 * must point to the byte after the last of the range (C-style range).
	 *
	 * Because of the masking mechanism and therefore alignment constraints used to implement
	 * this feature, it may not be possible to exactly cover a given range. It may be that
	 * the coverage exceeds the desired range. So it is possible to capture noise if
	 * the surrounding addresses are also heavily used. You can figure out, the actual
	 * start and end offsets of the generated range by checking the rr_soff and rr_eoff fields
	 * in the pfmlib_ita_output_param_t structure when coming back from the library call.
	 *
	 * Upon return, the pfmlib_ita_output_param_t.pfp_ita_drange.rr_dbr array is programmed and
	 * the number of entries used to cover the range is in rr_nbr_used.
	 */

	/*
	 * We indicate that we are using a Data Range Restriction feature.
	 * In this particular case this will cause, pfm_dispatch_events() to
	 * add pmc13 to the list of PMC registers to initialize and the
	 */

	ita_inp.pfp_ita_drange.rr_used = 1;
	ita_inp.pfp_ita_drange.rr_limits[0].rr_start = range_start;
	ita_inp.pfp_ita_drange.rr_limits[0].rr_end   = range_end;


	/*
	 * use the library to find the monitors to use
	 *
	 * upon return, cnt contains the number of entries
	 * used in pc[].
	 */
	if ((ret=pfm_dispatch_events(&inp, &ita_inp, &outp, &ita_outp)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	printf("data range  : [0x%016lx-0x%016lx): %d pair of debug registers used\n"
	       "start_offset:-0x%lx end_offset:+0x%lx\n",
			range_start,
			range_end,
			ita_outp.pfp_ita_drange.rr_nbr_used >> 1,
			ita_outp.pfp_ita_drange.rr_infos[0].rr_soff,
			ita_outp.pfp_ita_drange.rr_infos[0].rr_eoff);

	printf("fake data range: [0x%016lx-0x%016lx)\n",
			(unsigned long)test_data_fake,
			(unsigned long)test_data_fake+sizeof(test_data_t)*TEST_DATA_COUNT);

	/*
	 * now create the session
	 */
	id =pfm_create(0, NULL);
	if (id == -1) {
		if (errno == ENOSYS)
			fatal_error("Your kernel does not have performance monitoring support!\n");
		fatal_error("cannot create session %s\n", strerror(errno));
	}
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
	 * propagate the setup for the debug registers from the library to the arguments
	 * to the syscall. The library does not know the type of the syscall
	 * anymore. DBRS are ampped at PMC264+PMC271
	 */
	for (i=0; i < ita_outp.pfp_ita_drange.rr_nbr_used; i++) {
		dbrs[i].reg_num   = 264+ita_outp.pfp_ita_drange.rr_br[i].reg_num;
		dbrs[i].reg_value = ita_outp.pfp_ita_drange.rr_br[i].reg_value;
	}

	/*
	 * Program the data debug registers.
	 */
	if (pfm_write(id, 0, PFM_RW_PMC, dbrs, ita_outp.pfp_ita_drange.rr_nbr_used * sizeof(*dbrs)) == -1)
		fatal_error("pfm_write_pmrs error errno %d\n",errno);

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more than coutning monitors.
	 */
	if (pfm_write(id, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)) == -1)
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(id, 0, PFM_RW_PMD, pd, inp.pfp_event_count * sizeof(*pd)) == -1)
		fatal_error("pfm_write(PMD) error errno %d\n",errno);

	/*
	 * now we attach session
	 */
	if (pfm_attach(id, 0, getpid()) == -1)
		fatal_error("pfm_attach error errno %d\n",errno);

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
	if (pfm_set_state(id, 0, PFM_ST_START))
		fatal_error("pfm_set_state error errno %d\n",errno);

	for (i=0; i < N_LOOP; i++) {
		do_test(test_data);
		do_test(test_data_fake);
	}
	if (pfm_set_state(id, 0, PFM_ST_STOP))
		fatal_error("pfm_set_state error errno %d\n",errno);

	/*
	 * now read the results
	 */
	if (pfm_read(id, 0, PFM_RW_PMD, pd, inp.pfp_event_count * sizeof(*pd)) == -1)
		fatal_error( "pfm_read error errno %d\n",errno);

	/*
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 *
	 * For this example, we expect to see a value of 1 for both misaligned loads
	 * and misaligned stores. But it can be two when the test_data and test_data_fake
	 * are allocate very close from each other and the range created with the debug
	 * registers is larger then test_data.
	 *
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);
		printf("PMD%u %20lu %s (expected %lu)\n",
			pd[i].reg_num,
			pd[i].reg_value,
			name, event_list[i].expected_value);

		if (pd[i].reg_value != event_list[i].expected_value) {
			printf("error: Result should be %lu for %s\n", event_list[i].expected_value, name);
			break;
		}
	}
	/*
	 * let's stop this now
	 */
	close(id);

	free(test_data);
	free(test_data_fake);

	return 0;
}
