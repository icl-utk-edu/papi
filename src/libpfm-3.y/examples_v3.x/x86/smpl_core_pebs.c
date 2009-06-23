/*
 * smpl_core_pebs.c - Intel Core processor PEBS example
 *
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
#include <sys/wait.h>
#include <sys/ptrace.h>
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
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

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
	uint64_t count;

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
	smpl_arg_t buf_arg;
	pfarg_msg_t msg;
	smpl_hdr_t *hdr;
	void *buf_addr;
	uint64_t pebs_size;
	pid_t pid;
	int ret, fd, status, type;
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

	/*
	 * search for our sampling event
	 */
	if (pfm_find_full_event(SMPL_EVENT, &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find sampling event %s\n", SMPL_EVENT);

	inp.pfp_event_count = 1;
	inp.pfp_dfl_plm = PFM_PLM3|PFM_PLM0;

	/*
	 * important: inform libpfm we do use PEBS
	 */
	mod_inp.pfp_core_pebs.pebs_used = 1;

	/*
	 * sampling buffer size
	 *
	 * requested size includes space for:
	 * 	- buffer header
	 * 	- alignment padding (up to 1ULL<3 -1)
	 * 	- actual PEBS buffer
	 */
	pebs_size = 3 * getpagesize();
	buf_arg.buf_size = pebs_size;

	/*
	 * sampling period cannot use more bits than HW counter can supoprt
	 */
	buf_arg.cnt_reset = -SMPL_PERIOD;

	/*
	 * we want to block the monitored thread when the buffer becomes full
	 */
	ctx_flags = PFM_FL_NOTIFY_BLOCK | PFM_FL_SMPL_FMT;

	/*
	 * trigger notification (interrupt) when reached 90% of entries
	 * are recorded.
	 */
	buf_arg.intr_thres = (pebs_size/sizeof(smpl_entry_t))*90/100;

	printf("ent=%zu pebs_sz=%"PRIu64" max=%"PRIu64" thr=%"PRIu64"\n",
		sizeof(smpl_entry_t),
		pebs_size,
		pebs_size/sizeof(smpl_entry_t),
		(pebs_size*90/100)/sizeof(smpl_entry_t));

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
		 * must disable 64-bit emulation on the PMC0 counter
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

	signal(SIGCHLD, SIG_IGN);
	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1) fatal_error("Cannot fork process\n");

	/*
	 * In order to get the PFM_END_MSG message, it is important
	 * to ensure that the child task does not inherit the file
	 * descriptor of the session. By default, file descriptor
	 * are inherited during exec(). We explicitely close it
	 * here. We could have set it up through fcntl(FD_CLOEXEC)
	 * to achieve the same thing.
	 */
	if (pid == 0) {
		close(fd);
		child(argv+1);
	}

	/*
	 * wait for the child to exec
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * process is stopped at this point
	 */
	if (WIFEXITED(status)) {
		warning("task %s [%d] exited already status %d\n", argv[1], pid, WEXITSTATUS(status));
		goto terminate_session;
	}

	/*
	 *  attach the session to child
	 */
	if (pfm_attach(fd, 0, pid) == -1)
		fatal_error("pfm_attach error errno %d\n",errno);

	/*
	 * start monitoring
	 */
	if (pfm_set_state(fd, 0, PFM_ST_START) == -1)
		fatal_error("pfm_set_state(start) error errno %d\n",errno);

	/*
	 * detach child. Side effect includes
	 * activation of monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	/*
	 * core loop
	 */
	for(;;) {
		/*
		 * wait for overflow/end notification messages
		 */
		ret = read(fd, &msg, sizeof(msg));
		if (ret == -1) {
			if(ret == -1 && errno == EINTR) {
				warning("read interrupted, retrying\n");
				continue;
			}
			fatal_error("cannot read perfmon msg: %s\n", strerror(errno));
		}
		switch(msg.type) {
			case PFM_MSG_OVFL: /* the sampling buffer is full */
				process_smpl_buf(hdr);
				/*
				 * reactivate monitoring once we are done with the samples
				 *
				 * Note that this call can fail with EBUSY in non-blocking mode
				 * as the task may have disappeared while we were processing
				 * the samples.
				 */
				if (pfm_set_state(fd, 0, PFM_ST_RESTART) == -1) {
					if (errno != EBUSY)
						fatal_error("pfm_set_state(restart) error errno %d\n",errno);
					else
						warning("pfm_set_state(restart): task has probably terminated \n");
				}
				break;
			case PFM_MSG_END: /* monitored task terminated */
				warning("task terminated\n");
				goto terminate_session;
			default: fatal_error("unknown message type %d\n", msg.type);
		}
	}
terminate_session:
	/*
	 * cleanup child
	 */
	wait4(pid, &status, 0, NULL);

	/*
	 * check for any leftover samples
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
