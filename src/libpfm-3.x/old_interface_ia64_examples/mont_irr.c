/*
 * mont_irr.c - example of how to use code range restriction with the Dual-Core Itanium 2 PMU
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

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128
#define MAX_PMU_NAME_LEN	32

#define VECTOR_SIZE	1000000UL

typedef struct {
	char *event_name;
	unsigned long expected_value;
}  event_desc_t;

static event_desc_t event_list[]={
	{ "fp_ops_retired", VECTOR_SIZE<<1 },
	{ NULL, 0UL }
};


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


void
saxpy(double *a, double *b, double *c, unsigned long size)
{
	unsigned long i;

	for(i=0; i < size; i++) {
		c[i] = 2*a[i] + b[i];
	}
}

void
saxpy2(double *a, double *b, double *c, unsigned long size)
{
	unsigned long i;

	for(i=0; i < size; i++) {
		c[i] = 2*a[i] + b[i];
	}
}



static int
do_test(void)
{
	unsigned long size;
	double *a, *b, *c;

	size = VECTOR_SIZE;

	a = malloc(size*sizeof(double));
	b = malloc(size*sizeof(double));
	c = malloc(size*sizeof(double));

	if (a == NULL || b == NULL || c == NULL) fatal_error("Cannot allocate vectors\n");

	memset(a, 0, size*sizeof(double));
	memset(b, 0, size*sizeof(double));
	memset(c, 0, size*sizeof(double));

	saxpy(a,b,c, size);
	saxpy2(a,b,c, size);

	return 0;
}

int
main(int argc, char **argv)
{
	event_desc_t *p;
	unsigned long range_start, range_end;
	int ret, type = 0;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_mont_input_param_t mont_inp;
	pfmlib_mont_output_param_t mont_outp;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_dbreg_t ibrs[8];
	pfarg_context_t ctx;
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	struct fd {			/* function descriptor */
		unsigned long addr;
		unsigned long gp;
	} *fd;
	unsigned int i;
	int id;
	char name[MAX_EVT_NAME_LEN];

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("Can't initialize library\n");

	/*
	 * Let's make sure we run this on the right CPU family
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_MONTECITO_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with %s PMU\n", model);
	}
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 1; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Compute the range we are interested in
	 *
	 * On IA-64, the function pointer does not point directly
	 * to the function but to a descriptor which contains two
	 * unsigned long: the first one is the actual start address
	 * of the function, the second is the gp (global pointer)
	 * to load into r1 before jumping into the function. Unlesss
	 * we're jumping into a shared library the gp is the same as
	 * the current gp.
	 *
	 * In the artificial example, we also rely on the compiler/linker
	 * NOT reordering code layout. We depend on saxpy2() being just
	 * after saxpy().
	 *
	 */
	fd = (struct fd *)saxpy;
	range_start = fd->addr;

	fd = (struct fd *)saxpy2;
	range_end   =  fd->addr;
	
	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(&ctx, 0, sizeof(ctx));
	memset(ibrs,0, sizeof(ibrs));
	memset(&load_args,0, sizeof(load_args));

	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&mont_inp,0, sizeof(mont_inp));
	memset(&mont_outp,0, sizeof(mont_outp));

	/*
	 * find requested event
	 */
	p = event_list;
	for (i=0; p->event_name ; i++, p++) {
		if (pfm_find_event(p->event_name, &inp.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("cannot find %s event\n", p->event_name);
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
	 *
	 * In the case of code range restriction on Itanium 2, the library will try to use the fine
	 * mode first and then it will default to using multiple pairs to cover the range.
	 */

	mont_inp.pfp_mont_irange.rr_used = 1;	/* indicate we use code range restriction */
	mont_inp.pfp_mont_irange.rr_limits[0].rr_start = range_start;
	mont_inp.pfp_mont_irange.rr_limits[0].rr_end   = range_end;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, &mont_inp, &outp, &mont_outp)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * print offsets
	 */
	printf("code range  : [0x%016lx-0x%016lx)\n"
	       "start_offset:-0x%lx end_offset:+0x%lx\n"
		"%d pairs of debug registers used\n",
			range_start,
			range_end,
			mont_outp.pfp_mont_irange.rr_infos[0].rr_soff,
			mont_outp.pfp_mont_irange.rr_infos[0].rr_eoff,
			mont_outp.pfp_mont_irange.rr_nbr_used >> 1);

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, &ctx, 1) == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/*
	 * extract our file descriptor
	 */
	id = ctx.ctx_fd;

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We must pfp_pmc_count to determine the number of PMC to intialize.
	 * We must use pfp_event_count to determine the number of PMD to initialize.
	 * Some events cause extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
	 */

	for (i=0;  i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
	 * figure out pmd mapping from output pmc
	 */
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num   = outp.pfp_pmds[i].reg_num;
	
	/*
	 * propagate IBR settings. IBRS are mapped to PMC256-PMC263
	 */
	for (i=0; i < mont_outp.pfp_mont_irange.rr_nbr_used; i++) {
		ibrs[i].dbreg_num   = mont_outp.pfp_mont_irange.rr_br[i].reg_num;
		ibrs[i].dbreg_value = mont_outp.pfp_mont_irange.rr_br[i].reg_value;
	}
	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more than coutning monitors.
	 */
	if (perfmonctl(id, PFM_WRITE_IBRS, ibrs, mont_outp.pfp_mont_irange.rr_nbr_used) == -1)
		fatal_error("child: perfmonctl error PFM_WRITE_IBRS errno %d\n",errno);

	if (perfmonctl(id, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count))
		fatal_error("child: pfm_write_pmcs error errno %d\n",errno);

	if (perfmonctl(id, PFM_WRITE_PMDS, pd, outp.pfp_pmd_count) == -1)
		fatal_error("child: pfm_write_pmds error errno %d\n",errno);
	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = getpid();
	if (perfmonctl(id, PFM_LOAD_CONTEXT, &load_args, 1))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * Let's roll now.
	 *
	 * We run two distinct copies of the same function but we restrict measurement
	 * to the first one (saxpy). Therefore the expected count is half what you would
	 * get if code range restriction was not used. The core loop in both case uses
	 * two floating point operation per iteration.
	 */
	pfm_self_start(id);

	do_test();

	pfm_self_stop(id);

	/*
	 * now read the results
	 */
	if (perfmonctl(id, PFM_READ_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error( "pfm_read_pmds error errno %d\n",errno);
	}

	/*
	 * print the results
	 *
	 * It is important to realize, that the first event we specified may not
	 * be in PMD4. Not all events can be measured by any monitor. That's why
	 * we need to use the pc[] array to figure out where event i was allocated.
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);
		printf("PMD%-3u %20lu %s (expected %lu)\n",
			pd[i].reg_num,
			pd[i].reg_value,
			name, event_list[i].expected_value);
	}
	/*
	 * let's stop this now
	 */
	close(id);

	return 0;
}
