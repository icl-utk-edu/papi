/*
 * smpl_nhm_lbr.c - Intel Nehalem LBR sampling
 *
 * Copyright (c) 2008 Google, Inc
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
 */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <setjmp.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_intel_nhm.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

#include "../detect_pmcs.h"

#define SAMPLING_PERIOD	100000
#define EVENT_NAME	"br_inst_retired:all_branches"

typedef pfm_dfl_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_dfl_smpl_hdr_t		smpl_hdr_t;
typedef pfm_dfl_smpl_entry_t		smpl_entry_t;
typedef pfm_dfl_smpl_arg_t		smpl_arg_t;
#define FMT_NAME			PFM_DFL_SMPL_NAME

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static jmp_buf jbuf;

static uint64_t collected_samples, collected_partial, ovfl_count;

static void fatal_error(char *fmt,...) __attribute__((noreturn));

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6

static void handler (int n)
{
	longjmp(jbuf, 1);
}

static inline void pfm_bv_set(uint64_t *bv, uint16_t rnum)
{
	bv[rnum>>LBPL] |= 1UL << (rnum&(BPL-1));
}

static void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

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
	size_t pos, count, entry_size;
	uint64_t entry, *reg;
	uint64_t tos, i;
	
	if (hdr->hdr_overflows == last_overflow && hdr->hdr_count == last_count) {
		warning("skipping identical set of samples %"PRIu64" = %"PRIu64"\n",
			hdr->hdr_overflows, last_overflow);
		return;	
	}
	/*
 	 * 33 = 32 LBR registers + LBR_TOS
 	 */
	entry_size = sizeof(smpl_entry_t) + 33 * sizeof(uint64_t);

	count = hdr->hdr_count;

	ent   = (smpl_entry_t *)(hdr+1);
	pos   = (unsigned long)ent;
	entry = collected_samples;

	while(count--) {
		printf("entry %"PRIu64" PID:%d TID:%d CPU:%d LAST_VAL:%"PRIu64" OVFL:%u IIP:0x%llx\n",
			entry,
			ent->tgid,
			ent->pid,
			ent->cpu,
			-ent->last_reset_val,
			ent->ovfl_pmd,
			(unsigned long long)ent->ip);
		/*
		 * TOS is pmd31 and comes first
		 * TOs points to most recent entry
		 */
		reg = (uint64_t *)(ent+1);
		tos = reg[0] * 2;

		/*
 		 * i points to oldest entry, the one to print first
 		 */
		i = (tos + 2) % 32;
		/*
 		 * iterate over the 16 branches printing src -> dst
 		 */
		while (i != tos) {
			printf("0x%016"PRIx64" -> 0x%016"PRIx64"\n",
				reg[1+i],
				reg[1+i+1]);
			i = (i + 2) % 32;
		}
		pos += entry_size;
		ent = (smpl_entry_t *)pos;
		entry++;
	}
	collected_samples = entry;
	last_overflow = hdr->hdr_overflows;
	if (last_count != hdr->hdr_count && (last_count || last_overflow == 0))
		collected_partial += hdr->hdr_count;
	last_count = hdr->hdr_count;
}

int
mainloop(char **arg)
{
	smpl_hdr_t *hdr;
	pfarg_ctx_t ctx;
	smpl_arg_t buf_arg;
	pfmlib_input_param_t inp;
	pfmlib_nhm_input_param_t mod_inp;
	pfmlib_output_param_t outp;
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_load_t load_args;
	pfarg_msg_t msg;
	void *buf_addr;
	pid_t pid;
	int i, status, ret, fd;

	/*
	 * intialize all locals
	 */
	memset(&ctx, 0, sizeof(ctx));
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&load_args, 0, sizeof(load_args));
	memset(&mod_inp, 0, sizeof(mod_inp));

	ret = pfm_find_full_event(EVENT_NAME, &inp.pfp_events[0]);
	if (ret != PFMLIB_SUCCESS)
		fatal_error("cannot find event %s\n", EVENT_NAME);
	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = 1;


	mod_inp.pfp_nhm_lbr.lbr_used = 1;
	mod_inp.pfp_nhm_lbr.lbr_plm = 0; /* inherit from pfp_dfl_plm */

	/*
	 * setup LBR filter
	 *
	 * By default all types of branches are captured
	 *
	 * it is possible to filter out some types of branches using
	 * the macros in pfmlib_intel_nhm.c.
	 *
	 * for instance, to only capture nears calls, you do:
	 *
	 * lbr_filter = PFM_NHM_LBR_NEAR_REL_CALL
	 */
	mod_inp.pfp_nhm_lbr.lbr_filter = 0;
	
	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 *
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	buf_arg.buf_size = 4 * getpagesize();

	/*
	 * now create our perfmon context.
	 */
	fd = pfm_create_context(&ctx, FMT_NAME, &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS)
			fatal_error("Your kernel does not have performance monitoring support!\n");

		fatal_error("cannot create PFM context %s\n", strerror(errno));
	}

	detect_unavail_pmcs(fd, &inp.pfp_unavail_pmcs);

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, &mod_inp, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
 	 * we use only one counter
 	 */
	pd[0].reg_num = outp.pfp_pmds[0].reg_num;
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	/*
 	 * add 2 x 16 LBR entries + LBR_TOS to smpl_pmds
 	 */
	for(i=31; i < 64; i++)
		pfm_bv_set(pd[0].reg_smpl_pmds, i);

	/*
 	 * we need to reset LBR after each sample to be able to determine
 	 * whether or not we get new data
 	 *
 	 * LBR_TOS(PMD31) is read-only, it is not included in reset_pmds
	 */
	for(i=32; i < 64; i++)
		pfm_bv_set(pd[0].reg_reset_pmds, i);
	/*
	 * set sampling periods
	 */
	pd[0].reg_value       = - SAMPLING_PERIOD;
	pd[0].reg_short_reset = - SAMPLING_PERIOD;
	pd[0].reg_long_reset  = - SAMPLING_PERIOD;

	/*
	 * prepare context structure.
	 */
	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	buf_addr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (buf_addr == MAP_FAILED)
		fatal_error("cannot mmap sampling buffer: %s\n", strerror(errno));

	printf("buffer mapped @%p\n", buf_addr);

	hdr = (smpl_hdr_t *)buf_addr;

	printf("hdr_cur_offs=%llu version=%u.%u\n",
		(unsigned long long)hdr->hdr_cur_offs,
		PFM_VERSION_MAJOR(hdr->hdr_version),
		PFM_VERSION_MINOR(hdr->hdr_version));

	if (PFM_VERSION_MAJOR(hdr->hdr_version) < 1)
		fatal_error("invalid buffer format version\n");

	/*
	 * program the PMCs
	 */
	if (pfm_write_pmcs(fd, pc, outp.pfp_pmc_count))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	/*
	 *  program our counter
	 */
	if (pfm_write_pmds(fd, pd, 1))
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 * create the child task
	 */
	if ((pid=fork()) == -1)
		fatal_error("Cannot fork process\n");

	/*
	 * create child
	 * make sure child does not inherit the file descriptor
	 */
	if (pid == 0) {
		close(fd);
		child(arg);
	}

	/*
	 * wait for the child to exec
	 */
	waitpid(pid, &status, WUNTRACED);

	/*
	 * process is stopped at this point
	 */
	if (WIFEXITED(status)) {
		warning("task %s [%d] exited already status %d\n", arg[0], pid, WEXITSTATUS(status));
		goto terminate_session;
	}

	/*
	 * attach context to stopped task
	 */
	load_args.load_pid = pid;
	if (pfm_load_context (fd, &load_args))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * activate monitoring for stopped task.
	 * (nothing will be measured at this point
	 */
	if (pfm_start(fd, NULL))
		fatal_error("pfm_start error errno %d\n",errno);
	/*
	 * detach child. Side effect includes
	 * activation of monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	signal(SIGCHLD, handler);

	if (setjmp(jbuf) == 1)
		goto terminate_session;
	/*
	 * core loop
	 */
	for(;;) {
		ret = read(fd, &msg, sizeof(msg));
		if (ret == -1) {
			if(ret == -1 && errno == EINTR) {
				warning("read interrupted, retrying\n");
				continue;
			}
			fatal_error("cannot read perfmon msg: %s\n", strerror(errno));
		}
		if (msg.type == PFM_MSG_OVFL) {
			process_smpl_buf(hdr);
			ovfl_count++;
			/*
			 * reactivate monitoring once we are done with the samples
			 *
			 * Note that this call can fail with EBUSY in non-blocking mode
			 * as the task may have disappeared while we were processing
			 * the samples.
			 */
			if (pfm_restart(fd)) {
				if (errno != EBUSY)
					fatal_error("pfm_restart error errno %d\n",errno);
			}
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

	close(fd);

	ret = munmap(hdr, (size_t)buf_arg.buf_size);
	if (ret)
		fatal_error("cannot unmap buffer: %s\n", strerror(errno));

	printf("%"PRIu64" samples (%"PRIu64" in partial buffer) collected in %"PRIu64" buffer overflows\n",
		collected_samples,
		collected_partial,
		ovfl_count);

	return 0;
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	int ret;

	if (argc == 1)
		fatal_error("You must specify a command to execute\n");
	
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 1; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	return mainloop(argv+1);
}
