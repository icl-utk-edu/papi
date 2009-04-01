/*
 * self_view.c - example of self-monitoring with PMD access via remapping
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
 */
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128

typedef struct {
	uint64_t	val;
	unsigned int	reg_num;
	char		name[MAX_EVT_NAME_LEN];
} pmd_val_t;


static pfm_set_view_t	*view0;
static pmd_val_t	pmdv[NUM_PMDS];
static unsigned int	num_pmdv;
static uint64_t		ovfl_mask;

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

#ifdef __ia64__
#define get_pmd(n)	ia64_get_pmd(n)
#elif defined(__x86_64__) || defined(__i386__)
/*
 * IMPORTANT:
 *  there is an implicit assumption here that the perfmon-2 
 *  PMD mapping and the rdpmc mapping correspond. As such
 *  pmd0 -> rdpmc(0)
 *  pmd1 -> rdpmc(1)
 *  ...
 */
#define rdpmc(counter,low,high) \
	__asm__ __volatile__("rdpmc" \
		: "=a" (low), "=d" (high) \
		: "c" (counter))

uint64_t
get_pmd(unsigned int n)
{
	uint32_t high, low;
	uint64_t tmp;
	rdpmc(n, low, high);
	tmp = (uint64_t)high <<32 | low;
	return tmp;
}
#elif defined(__mips__) || defined(__powerpc__)
/*
 * XXX: MIPS does not have an instruction to read a counter at the user level
 */
uint64_t
get_pmd(unsigned int n)
{
  return(0ULL);
}
#else
#error "you need to define get_pmd for your architecture"
#endif

static void
show_view_self(int is_loaded)
{
	unsigned int i, is_active;
	uint64_t val, hw_val;
	uint64_t mask = ovfl_mask;
	unsigned long retries = 0;
	unsigned long start_seq, end_seq;

	retries = -1;
	/*
	 * print the results
	 */
retry:
	retries++;
	/*
	 * get initial sequence number, if the number changes
	 * while we are scanning, it means the view was updated
	 * and we need to retry
	 */
	start_seq = view0->set_seq;
	/*
	 * active is true if the set is the active set
	 */
	is_active = view0->set_status & PFM_SETVFL_ACTIVE;

	printf("retries=%lu active=%d view_seq=%lu set_runs=%"PRIu64"\n",
		retries,
		view0->set_status,
		view0->set_seq,
		view0->set_runs);

	/*
	 * extract information directly from view
	 */
	for (i=0; i < num_pmdv; i++) {
		val = view0->set_pmds[pmdv[i].reg_num];
		/*
		 * is context is attached and set is active then
		 * we need to complement the software value with 
		 * the current hardware value. For self-monitoring
		 * with simply need to read the PMD
		 */
		if (is_loaded && is_active) {
			hw_val = get_pmd(pmdv[i].reg_num);
			/*
			 * update the lower portion of the 64-bit
			 * value with the hardware value
			 */
			val    = (val & ~mask) | (hw_val & mask);
		}
		pmdv[i].val  = val;

	}
	/*
	 * check if sequence number is still the same
	 */
	end_seq = view0->set_seq;
	if (end_seq != start_seq) goto retry;

	/*
	 * print results
	 */
	for (i=0; i < num_pmdv; i++) {
		printf("%20"PRIu64" %s\n", pmdv[i].val, pmdv[i].name);
	}
}

uint64_t
noploop(uint64_t loop)
{

	while (loop--) {
		if ((loop % 10000) == 0) show_view_self(1);
	}
	return loop;
}



int
main(int argc, char **argv)
{
	char **p;
	unsigned int i;
	int ret, ctx_fd;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_ctx_t ctx[1];
	pfarg_setinfo_t setinfo;
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	unsigned int num_counters, width;

#ifdef __mips__
	printf("<<WARNING: MIPS does not have an instruction to read a counter at the user level. Results are wrong>>\n");
#endif
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	pfm_get_num_counters(&num_counters);

	pfm_get_hw_counter_width(&width);

	ovfl_mask = (1ULL << width)-1;
	printf("width=%u ovfl_mask=0x%"PRIx64"\n", width, ovfl_mask);	

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(ctx, 0, sizeof(ctx));
	memset(&load_args, 0, sizeof(load_args));
	memset(&setinfo, 0, sizeof(setinfo));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));

	/*
	 * be nice to user!
	 */
	if (argc > 1) {
		p = argv+1;
		for (i=0; *p ; i++, p++) {
			if (pfm_find_full_event(*p, &inp.pfp_events[i]) != PFMLIB_SUCCESS) {
				fatal_error("Cannot find %s event\n", *p);
			}
		}
	} else {
		if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS) {
			fatal_error("cannot find cycle event\n");
		}
		if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS) {
			fatal_error("cannot find inst retired event\n");
		}
		i = 2;
	}
	/*
	 * set the default privilege mode for all counters:
	 * 	PFM_PLM3 : user level only
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

	ctx[0].ctx_flags = PFM_FL_MAP_SETS;
	/*
	 * now create a new context, per process context.
	 * This just creates a new context with some initial state, it is not
	 * active nor attached to any process.
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
	for(i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
	/*
	 * Now program the registers
	 *
	 * We don't use the same variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events (pmd) we specified, i.e., contains more than counting
	 * monitors.
	 */
	if (pfm_write_pmcs(ctx_fd, pc, outp.pfp_pmc_count))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(ctx_fd, pd, outp.pfp_pmd_count))
		fatal_error("pfm)write_pmds error errno %d\n",errno);

	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = getpid();

	if (pfm_load_context(ctx_fd, &load_args))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * retrieve set0 information
	 */
	if (pfm_getinfo_evtsets(ctx_fd, &setinfo, 1))
		fatal_error("pfm_getinfo_evtsets error errno %d\n",errno);

	printf("set_id=0 mmap_offset=%llu\n", (unsigned long long)setinfo.set_mmap_offset);

	view0 = (pfm_set_view_t *)mmap(NULL, getpagesize(), PROT_READ, MAP_PRIVATE, ctx_fd, setinfo.set_mmap_offset);
	if (view0 == MAP_FAILED) {
		fatal_error("cannot mmap set view errno %d\n", errno);
	}
	printf("view=%p set_id=0 set_status=%u\n", view0, view0->set_status);

	/*
	 * Let's roll now
	 */
	pfm_self_start(ctx_fd);

	noploop(10000000);

	pfm_self_stop(ctx_fd);

	/*
	 * unload context (stop + unload)
	 */
	if (pfm_unload_context(ctx_fd))
		fatal_error( "pfm_unload_context error errno %d\n",errno);

	/*
	 * now read the results
	 */
	if (pfm_read_pmds(ctx_fd, pd, outp.pfp_pmd_count))
		fatal_error( "pfm_read_pmds error errno %d\n",errno);
	
	printf("results using pfm_read_pmds:\n");
	for (i=0; i < num_pmdv; i++) {
		printf("%20"PRIu64" %s\n", pd[i].reg_value, pmdv[i].name);
	}

	show_view_self(0);

	/*
	 * and destroy our context
	 */
	munmap(view0, getpagesize());
	close(ctx_fd);

	return 0;
}
