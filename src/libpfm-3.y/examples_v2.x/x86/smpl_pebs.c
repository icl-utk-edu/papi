/*
 * smpl_pebs.c - Unified Intel PEBS sampling example
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#include <err.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_pebs_smpl.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_core.h>
#include <perfmon/pfmlib_intel_nhm.h>

#include "../detect_pmcs.h"

#define SMPL_EVENT	"INST_RETIRED:ANY_P" /* PEBS event on all processors */

#define NUM_PMCS	16
#define NUM_PMDS	16

#define SMPL_PERIOD	240000ULL /* must not use more bits than actual HW counter width */

typedef pfm_pebs_smpl_hdr_t	smpl_hdr_t;
typedef pfm_pebs_smpl_arg_t	smpl_arg_t;
#define FMT_NAME		PFM_PEBS_SMPL_NAME

static uint64_t collected_samples;
static uint64_t last_overflow = ~0; /* initialize to biggest value possible */

static void (*print_entry)(uint64_t entry, void *addr);
static int maxpebs = 1; /* 1=Atom/Core, up to 4 on Nehalem */

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
print_p4_entry(uint64_t entry, void *addr)
{
	pfm_pebs_p4_smpl_entry_t *ent = addr;

	printf("entry %06"PRIu64" eflags:0x%08llx EAX:0x%08llx ESP:0x%08llx IP:0x%08llx\n",
		entry,
		(unsigned long long)ent->eflags,
		(unsigned long long)ent->eax,
		(unsigned long long)ent->esp,
		(unsigned long long)ent->ip);
}


static void
print_core_entry(uint64_t entry, void *addr)
{
	pfm_pebs_core_smpl_entry_t *ent = addr;

	printf("entry %06"PRIu64" eflags:0x%08llx EAX:0x%08llx ESP:0x%08llx IP:0x%08llx\n",
		entry,
		(unsigned long long)ent->eflags,
		(unsigned long long)ent->eax,
		(unsigned long long)ent->esp,
		(unsigned long long)ent->ip);
}

static void
print_nhm_entry(uint64_t entry, void *addr)
{
	pfm_pebs_nhm_smpl_entry_t *ent = addr;

	printf("entry %06"PRIu64" eflags:0x%08llx EAX:0x%08llx ESP:0x%08llx IP:0x%08llx OVFL:0x%08llx\n",
		entry,
		(unsigned long long)ent->eflags,
		(unsigned long long)ent->eax,
		(unsigned long long)ent->esp,
		(unsigned long long)ent->ip,
		(unsigned long long)ent->ia32_perf_global_status);
}

static void
process_smpl_buf(smpl_hdr_t *hdr)
{
	static uint64_t last_count;
	void *ent;
	uint64_t entry;
	unsigned long count;

	count = hdr->count;

	if (hdr->overflows == last_overflow && last_count == count) {
		warnx("skipping identical set of samples %"PRIu64" = %"PRIu64"\n",
			hdr->overflows, last_overflow);
		return;	
	}
	last_count = count;
	last_overflow = hdr->overflows;

	/*
	 * the beginning of the buffer does not necessarily follow the header
	 * due to alignement.
	 */
	ent   = (hdr+1);
	entry = collected_samples;

	while(count--) {
		(*print_entry)(entry, ent);
		ent += hdr->entry_size;
		entry++;
	}
	collected_samples = entry;
}


int
main(int argc, char **argv)
{
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_core_input_param_t core_inp;
	pfmlib_nhm_input_param_t nhm_inp;
	void *mod_inp = NULL;
	pfmlib_options_t pfmlib_options;
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_ctx_t ctx;
	smpl_arg_t buf_arg;
	pfarg_load_t load_args;
	pfarg_msg_t msg;
	smpl_hdr_t *hdr;
	void *buf_addr;
	pid_t pid;
	int ret, fd, status, type;
	unsigned int i;

	if (argc < 2)
		errx(1, "you need to pass a program to sample");

	if (pfm_initialize() != PFMLIB_SUCCESS)
		errx(1, "libpfm intialization failed");

	memset(&core_inp, 0, sizeof(core_inp));
	memset(&nhm_inp, 0, sizeof(nhm_inp));

	/*
	 * check we are on an Intel Core PMU
	 */
	pfm_get_pmu_type(&type);
	switch(type) {
		case PFMLIB_INTEL_CORE_PMU:
		case PFMLIB_INTEL_ATOM_PMU:
			print_entry = print_core_entry;
			core_inp.pfp_core_pebs.pebs_used = 1;
			mod_inp = &core_inp;
			break;
		case PFMLIB_INTEL_NHM_PMU:
			print_entry = print_nhm_entry;
			nhm_inp.pfp_nhm_pebs.pebs_used = 1;
			mod_inp = &nhm_inp;
			break;
		case PFMLIB_PENTIUM4_PMU:
			print_entry = print_p4_entry;
			break;
		default:
			errx(1, "PMU model does not have PEBS support");
	}

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

	memset(&ctx, 0, sizeof(ctx));
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&load_args, 0, sizeof(load_args));

	/*
	 * search for our sampling event
	 */
	if (pfm_find_full_event(SMPL_EVENT, &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		errx(1, "cannot find sampling event %s", SMPL_EVENT);

	for(i=1; i < maxpebs; i++)
		inp.pfp_events[i] = inp.pfp_events[0];

	inp.pfp_event_count = i;
	inp.pfp_dfl_plm = PFM_PLM3|PFM_PLM0;

	/*
	 * sampling buffer parameters
	 */
	buf_arg.buf_size = 2 * getpagesize();

	for(i=0; i < maxpebs; i++)
		buf_arg.cnt_reset[i] = -SMPL_PERIOD;

	/*
	 * create context and sampling buffer
	 */
	fd = pfm_create_context(&ctx, FMT_NAME, &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS) {
			errx(1, "Your kernel does not have performance monitoring support!\n");
		}
		err(1, "cannot create session, maybe you do not have the PEBS"
		       " sampling format in the kernel. You need perfmon_pebs_smpl."
		       "\nCheck /sys/kernel/perfmon/formats");
	}

	/*
	 * map buffer into our address space
	 */
	buf_addr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (buf_addr == MAP_FAILED)
		err(1, "cannot mmap sampling buffer");

	printf("context [%d] buffer mapped @%p\n", fd, buf_addr);

	hdr = (smpl_hdr_t *)buf_addr;

	printf("pebs_start=%p pebs_end=%p version=%u.%u.%u entry_size=%u\n",
		hdr+1,
		hdr+1,
		(hdr->version >> 16) & 0xff,
		(hdr->version >> 8) & 0xff, 
		hdr->version & 0xff, 
		hdr->entry_size);

	printf("max PEBS entries: %zu\n", (size_t)hdr->pebs_size / hdr->entry_size);

	if (((hdr->version >> 16) & 0xff) < 1)
		errx(1, "invalid buffer format version");

	/*
	 * get which PMC registers are available
	 */
	detect_unavail_pmcs(fd, &inp.pfp_unavail_pmcs);

	/*
	 * let libpfm figure out how to assign event onto PMU registers
	 */
	if (pfm_dispatch_events(&inp, mod_inp, &outp, NULL) != PFMLIB_SUCCESS)
		errx(1, "cannot assign event %s\n", SMPL_EVENT);


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
	for(i=0; i < maxpebs; i++) {
		pd[i].reg_flags = PFM_REGFL_OVFL_NOTIFY;
		pd[i].reg_value = -SMPL_PERIOD;
		pd[i].reg_long_reset = -SMPL_PERIOD;
		pd[i].reg_short_reset = -SMPL_PERIOD;
	}
	
	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count) == -1)
		err(1, "pfm_write_pmcs error");

	if (pfm_write_pmds(fd, pd, outp.pfp_pmd_count) == -1)
		err(1, "pfm_write_pmds error");

	signal(SIGCHLD, SIG_IGN);
	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1)
		err(1, "cannot fork process");

	/*
	 * In order to get the PFM_END_MSG message, it is important
	 * to ensure that the child task does not inherit the file
	 * descriptor of the context. By default, file descriptor
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
		warnx("task %s [%d] exited already status %d\n", argv[1], pid, WEXITSTATUS(status));
		goto terminate_session;
	}

	/*
	 *  attach the context to child
	 */
	load_args.load_pid = pid;
	if (pfm_load_context(fd, &load_args) == -1)
		err(1, "pfm_load_context error");

	/*
	 * start monitoring
	 */
	if (pfm_start(fd, NULL) == -1)
		err(1, "pfm_start error");

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
				warnx("read interrupted, retrying");
				continue;
			}
			err(1, "cannot read perfmon msg");
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
				if (pfm_restart(fd) == -1) {
					if (errno != EBUSY)
						err(1, "pfm_restart error");
					else
						warnx("pfm_restart: task has probably terminated \n");
				}
				break;
			case PFM_MSG_END: /* monitored task terminated */
				warnx("TASK terminated");
				goto terminate_session;
			default:
				errx(1, "unknown message type %d", msg.type);
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

	printf("collected samples %"PRIu64", %"PRIu64" overflows\n",
		collected_samples, last_overflow);

	/*
	 * close context
	 */
	close(fd);

	/*
	 * unmap sampling buffer and actually free the perfmon context
	 */
	munmap(buf_addr, (size_t)buf_arg.buf_size);

	return 0;
}
