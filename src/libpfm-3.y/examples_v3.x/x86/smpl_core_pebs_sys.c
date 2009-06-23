/*
 * smpl_core_pebs_sys.c - Intel Core processor PEBS system-wide example 
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on code:
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <syscall.h>
#include <unistd.h>
#include <poll.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_pebs_core_smpl.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_core.h>
#include <perfmon/pfmlib_intel_atom.h>

#include "../detect_pmcs.h"

#define SMPL_EVENT	"INSTRUCTIONS_RETIRED" /* not all event support PEBS */

#define NUM_PMCS	16
#define NUM_PMDS	16

#define SMPL_PERIOD	100000ULL	 /* must not use more bits than actual HW counter width */

typedef pfm_pebs_core_smpl_hdr_t	smpl_hdr_t;
typedef pfm_pebs_core_smpl_entry_t	smpl_entry_t;
typedef pfm_pebs_core_smpl_arg_t	smpl_arg_t;
#define FMT_NAME			PFM_PEBS_CORE_SMPL_NAME

static uint64_t collected_samples;

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

static void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

int
child(char **arg)
{
	/*
	 * force the task to stop before executing the first
	 * user level instruction
	 */

	execvp(arg[0], arg);
	/* not reached */
	exit(1);
}

static void
process_smpl_buf(smpl_hdr_t *hdr)
{
	static uint64_t last_overflow = ~0; /* initialize to biggest value possible */
	static uint64_t last_count;
	smpl_entry_t *ent;
	uint64_t entry;
	unsigned long count;

	count = (hdr->ds.pebs_index - hdr->ds.pebs_buf_base)/sizeof(*ent);

	if (hdr->overflows == last_overflow && last_count == count) {
		warning("skipping identical set of samples %"PRIu64" = %"PRIu64"\n",
			hdr->overflows, last_overflow);
		return;	
	}
	last_count = count;
	last_overflow = hdr->overflows;

	/*
	 * the beginning of the buffer does not necessarily follow the header
	 * due to alignement.
	 */
	ent   = (smpl_entry_t *)((unsigned long)(hdr+1)+ hdr->start_offs);
	entry = collected_samples;

	while(count--) {
		/*
		 * print some of the machine registers of each sample
		 */
		printf("entry %06"PRIu64" eflags:0x%08llx EAX:0x%08llx ESP:0x%08llx IP:0x%08llx\n",
			entry,
			(unsigned long long)ent->eflags,
			(unsigned long long)ent->eax,
			(unsigned long long)ent->esp,
			(unsigned long long)ent->ip);
		ent++;
		entry++;
	}
	collected_samples = entry;
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

static volatile int done;

static void handler(int n)
{
	done = 1;
}

int
main(int argc, char **argv)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_core_input_param_t mod_inp;
	pfmlib_options_t pfmlib_options;
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_pmd_attr_t pd[NUM_PMDS];
	pfarg_sinfo_t sif;
	struct pollfd fds;
	smpl_arg_t buf_arg;
	pfarg_msg_t msg;
	smpl_hdr_t *hdr;
	void *buf_addr;
	uint64_t pebs_size;
	pid_t pid;
	int ret, fd, type;
	unsigned int i;
	uint32_t ctx_flags;

	if (argc < 2)
		fatal_error("you need to pass a program to sample\n");

	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("libpfm intialization failed\n");

	/*
	 * check we are on an Intel Core PMU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_INTEL_CORE_PMU && type != PFMLIB_INTEL_ATOM_PMU)
		fatal_error("This program only works with an Intel Core processor\n");

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&inp, 0, sizeof(inp));
	memset(&outp, 0, sizeof(outp));
	memset(&mod_inp, 0, sizeof(mod_inp));
	memset(&sif, 0, sizeof(sif));

	memset(&buf_arg, 0, sizeof(buf_arg));

	memset(&fds, 0, sizeof(fds));

	/*
	 * search for our sampling event
	 */
	if (pfm_find_full_event(SMPL_EVENT, &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find sampling event %s\n", SMPL_EVENT);

	inp.pfp_event_count = 1;
	inp.pfp_dfl_plm = PFM_PLM3;

	/*
	 * important: inform libpfm we do use PEBS
	 */
	mod_inp.pfp_core_pebs.pebs_used = 1;

	/*
	 * sampling buffer parameters
	 */
	pebs_size = 3 * getpagesize();
	buf_arg.buf_size = pebs_size;

	/*
	 * sampling period cannot use more bits than HW counter can supoprt
	 */
	buf_arg.cnt_reset = -SMPL_PERIOD;

	/*
	 * We want a system-wide context for sampling
	 */
	ctx_flags = PFM_FL_SYSTEM_WIDE | PFM_FL_SMPL_FMT;

	/*
	 * trigger notification (interrupt) when reaching the very end of
	 * the buffer
	 */
	buf_arg.intr_thres = (pebs_size/sizeof(smpl_entry_t))*90/100;

	/*
 	 * we want to measure CPU0, thus we pin ourself to the CPU before invoking
 	 * perfmon. This ensures that the sampling buffer will be allocated on the
 	 * same NUMA node.
 	 */
	ret = pin_cpu(getpid(), 0);
	if (ret)
		fatal_error("cannot pin on CPU0");

	/*
	 * create session and sampling buffer
	 */
	fd = pfm_create(ctx_flags, &sif, FMT_NAME, &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("cannot create session %s, maybe you do not have the PEBS sampling format in the kernel.\nCheck /sys/kernel/perfmon/formats\n", strerror(errno));
	}

	/*
	 * map buffer into our address space
	 */
	buf_addr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	printf("session [%d] buffer mapped @%p\n", fd, buf_addr);
	if (buf_addr == MAP_FAILED)
		fatal_error("cannot mmap sampling buffer errno %d\n", errno);

	hdr = (smpl_hdr_t *)buf_addr;

	printf("pebs_base=0x%llx pebs_end=0x%llx index=0x%llx\n"
	       "intr=0x%llx version=%u.%u\n"
	       "entry_size=%zu ds_size=%zu\n",
			(unsigned long long)hdr->ds.pebs_buf_base,
			(unsigned long long)hdr->ds.pebs_abs_max,
			(unsigned long long)hdr->ds.pebs_index,
			(unsigned long long)hdr->ds.pebs_intr_thres,
			PFM_VERSION_MAJOR(hdr->version),
			PFM_VERSION_MINOR(hdr->version),
			sizeof(smpl_entry_t),
			sizeof(hdr->ds));

	if (PFM_VERSION_MAJOR(hdr->version) < 1)
		fatal_error("invalid buffer format version\n");

	/*
	 * get which PMC registers are available
	 */
	detect_unavail_pmu_regs(&sif, &inp.pfp_unavail_pmcs, NULL);

	/*
	 * let libpfm figure out how to assign event onto PMU registers
	 */
	if (pfm_dispatch_events(&inp, &mod_inp, &outp, NULL) != PFMLIB_SUCCESS)
		fatal_error("cannot assign event %s\n", SMPL_EVENT);


	/*
	 * propagate PMC setup from libpfm to perfmon
	 */
	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;

		/*
		 * must disable 64-bit emulation on the PMC0 counter.
		 * PMC0 is the only counter useable with PEBS. We must disable
		 * 64-bit emulation to avoid getting interrupts for each
		 * sampling period, PEBS takes care of this part.
		 */
		if (pc[i].reg_num == 0)
			pc[i].reg_flags = PFM_REGFL_NO_EMUL64;
	}

	/*
	 * propagate PMD set from libpfm to perfmon
	 */
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;

	/*
	 * setup sampling period for first counter
	 * we want notification on overflow, i.e., when buffer is full
	 */
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	pd[0].reg_value = -SMPL_PERIOD;

	pd[0].reg_long_reset = -SMPL_PERIOD;
	pd[0].reg_short_reset = -SMPL_PERIOD;
	
	/*
	 * Now program the registers
	 */
	if (pfm_write(fd, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)) == -1)
		fatal_error("pfm_write error errno %d\n",errno);

	if (pfm_write(fd, 0, PFM_RW_PMD_ATTR, pd, outp.pfp_pmd_count * sizeof(*pd)) == -1)
		fatal_error("pfm_write(PMD) error errno %d\n",errno);

	/*
	 *  attach the session to CPU0
	 */
	if (pfm_attach(fd, 0, 0) == -1)
		fatal_error("pfm_attach error errno %d\n",errno);

	/*
	 * Create the child task
	 */
	signal(SIGCHLD, handler);

	if ((pid=fork()) == -1)
		fatal_error("Cannot fork process\n");

	if (pid == 0) {
		/* child does not inherit context file descriptor */
		close(fd);

		/* if child is too short-lived we may not measure it */
		child(argv+1);
	}

	/*
	 * start monitoring
	 */
	if (pfm_set_state(fd, 0, PFM_ST_START) == -1)
		fatal_error("pfm_set_state(start) error errno %d\n",errno);

	fds.fd = fd;
	fds.events = POLLIN;
	/*
	 * core loop
	 */
	for(;done == 0;) {
		/*
		 * Must use a timeout to avoid a race condition
		 * with the SIGCHLD signal
		 */
		ret = poll(&fds, 1, 500);

		/*
		 * if timeout expired, then check done
		 */
		if (ret == 0)
			continue;

		if (ret == -1) {
			if(ret == -1 && errno == EINTR) {
				warning("read interrupted, retrying\n");
				continue;
			}
			fatal_error("poll failed: %s\n", strerror(errno));
		}

		ret = read(fd, &msg, sizeof(msg));
		if (ret == -1)
			fatal_error("cannot read perfmon msg: %s\n", strerror(errno));

		switch(msg.type) {
			case PFM_MSG_OVFL: /* the sampling buffer is full */
				process_smpl_buf(hdr);
				/*
				 * reactivate monitoring once we are done with the samples
				 * in syste-wide, interface guarantees monitoring is active
				 * upon return from the pfm_restart() syscall
				 */
				if (pfm_set_state(fd, 0, PFM_ST_RESTART) == -1)
					fatal_error("pfm_set_state(restart) error errno %d\n",errno);
				break;
			default: fatal_error("unknown message type %d\n", msg.type);
		}
	}
	/*
	 * cleanup child
	 */
	waitpid(pid, NULL, 0);

	/*
	 * stop monitoring, this is required in order to guarantee that the PEBS buffer
	 * header is updated with the latest position, such that we see see the final
	 * samples
	 */
	if (pfm_set_state(fd, 0, PFM_ST_STOP) == -1)
		fatal_error("pfm_set_state(stop) error errno %d\n",errno);

	/*
	 * check for any leftover samples. Must have monitoring stopped
	 * for this operation to have guarantee it is up to date
	 */
	process_smpl_buf(hdr);

	/*
	 * close session
	 */
	close(fd);

	/*
	 * unmap sampling buffer and actually free the perfmon session
	 */
	munmap(buf_addr, (size_t)buf_arg.buf_size);

	return 0;
}
