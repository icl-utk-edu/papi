/*
 * syst.c - example of a simple system wide monitoring program
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
 */
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <syscall.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>


#include "detect_pmcs.h"

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128

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

/*
 * pin task to CPU
 */
#ifndef __NR_sched_setaffinity
#error "you need to define __NR_sched_setaffinity"
#endif

#define MAX_CPUS	2048
#define NR_CPU_BITS	(MAX_CPUS>>3)
int
pin_cpu(pid_t pid, unsigned int cpu)
{
	uint64_t my_mask[NR_CPU_BITS];

	if (cpu >= MAX_CPUS)
		fatal_error("this program supports only up to %d CPUs\n", MAX_CPUS);

	my_mask[cpu>>6] = 1ULL << (cpu&63);

	return syscall(__NR_sched_setaffinity, pid, sizeof(my_mask), &my_mask);
}

int
main(int argc, char **argv)
{
	char **p;
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_pmr_t pd[NUM_PMDS];
	pfarg_sinfo_t sif;
	uint64_t pdo[NUM_PMDS];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_options_t pfmlib_options;
	unsigned int which_cpu;
	int ret, ctx_fd;
	unsigned int i, l;
	unsigned int num_counters;
	char name[MAX_EVT_NAME_LEN];

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
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	pfm_get_num_counters(&num_counters);


	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(pdo, 0, sizeof(pdo));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&sif,0, sizeof(sif));

	/*
	 * be nice to user!
	 */
	if (argc > 1) {
		p = argv+1;
		for (i=0; *p ; i++, p++) {
			if (pfm_find_full_event(*p, &inp.pfp_events[i]) != PFMLIB_SUCCESS)
				fatal_error("Cannot find %s event\n", *p);
		}
	} else {
		if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
			fatal_error("cannot find cycle event\n");
		if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
			fatal_error("cannot find inst retired event\n");
		i = 2;
	}
	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user   level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3|PFM_PLM0;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}
	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

	/*
	 * indicate we are using the monitors for a system-wide session.
	 * This may impact the way the library sets up the PMC values.
	 */
	inp.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;

	/*
	 * pick a random CPU. Assumes CPU are numbered with no holes
	 */
	srandom(getpid());

	which_cpu = random() % sysconf(_SC_NPROCESSORS_ONLN);

	/*
	 * The monitored CPU is determined by the processor core
	 * executing the PFM_LOAD_CONTEXT command. To ensure, we
	 * measure the right core, we pin the thread before making
	 * the call.
	 */
	ret = pin_cpu(getpid(), which_cpu);
	if (ret == -1)
		fatal_error("cannot set affinity to CPU%d: %s\n", which_cpu, strerror(errno));
	/*
	 * after the call the task is pinned to which_cpu
	 */

	/*
	 * now create the system-wide session 
	 */
	ctx_fd = pfm_create(PFM_FL_SYSTEM_WIDE, &sif);
	if (ctx_fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("cannot create session %s\n", strerror(errno));
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
	detect_unavail_pmu_regs(&sif, &inp.pfp_unavail_pmcs, NULL);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, NULL, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We use pfp_pmc_count to determine the number of PMC to intialize.
	 * We use pfp_pmd_count to determine the number of PMD to initialize.
	 * Some events/features may cause extra PMCs to be used, leading to:
	 * 	- pfp_pmc_count may be >= pfp_event_count
	 * 	- pfp_pmd_count may be >= pfp_event_count
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
	/*
	 * Now program the registers
	 */
	if (pfm_write(ctx_fd, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)) == -1)
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(ctx_fd, 0, PFM_RW_PMD, pd, outp.pfp_pmd_count * sizeof(*pd)) == -1)
		fatal_error("pfm_write(PMDS) error errno %d\n",errno);

	/*
 	 * attach the session to the CPU
	 */
	if (pfm_attach(ctx_fd, 0, which_cpu) == -1)
		fatal_error("pfm_attach error errno %d\n",errno);

	printf("<monitoring started on CPU%d, press CTRL-C to quit before 20s time limit>\n", which_cpu);

	for(l=0; l < 10; l++) {
		/*
		 * start monitoring
		 */
		if (pfm_set_state(ctx_fd, 0, PFM_ST_START) == -1)
			fatal_error("pfm_set_state(start) error errno %d\n",errno);

		sleep(2);

		/*
		 * stop monitoring. 
		 * changed at the user level.
		 */
		if (pfm_set_state(ctx_fd, 0, PFM_ST_STOP) == -1)
			fatal_error("pfm_set_state(stop) error errno %d\n",errno);

		/*
		 * read the results
		 */
		if (pfm_read(ctx_fd, 0, PFM_RW_PMD, pd, inp.pfp_event_count * sizeof(*pd)) == -1)
			fatal_error( "pfm_read error errno %d\n",errno);

		/*
		 * print the results
		 */
		puts("------------------------");
		for (i=0; i < inp.pfp_event_count; i++) {
			pfm_get_full_event_name(&inp.pfp_events[i], name, MAX_EVT_NAME_LEN);
			printf("CPU%-2d PMD%-3u raw=%-20"PRIu64" delta=%-20"PRIu64" %s\n",
					which_cpu,
					pd[i].reg_num,
					pd[i].reg_value,
					pd[i].reg_value - pdo[i],
					name);
			pdo[i] = pd[i].reg_value;
		}
	}
	/*
	 * destroy everything
	 */
	close(ctx_fd);

	return 0;
}
