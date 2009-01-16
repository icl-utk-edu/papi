/*
 * task_smpl.c - example of a task sampling another one using a randomized sampling period
 *
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
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
#include <stdarg.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

#include <perfmon/perfmon.h>
#include <perfmon/perfmon_default_smpl.h>
#include <perfmon/pfmlib.h>

typedef pfm_default_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_default_smpl_hdr_t		smpl_hdr_t;
typedef pfm_default_smpl_entry_t	smpl_entry_t;
typedef pfm_default_smpl_ctx_arg_t	ctx_arg_t;
typedef int			 	ctxid_t;
#define FMT_UUID		 	PFM_DEFAULT_SMPL_UUID

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define FIRST_COUNTER	4

static unsigned long collect_samples;
static void *buf_addr;
static pfm_uuid_t buf_fmt_id = FMT_UUID;


static void fatal_error(char *fmt,...) __attribute__((noreturn));

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

static __inline__ int
bit_weight(unsigned long x)
{
	int sum = 0;
	for (; x ; x>>=1) {
		if (x & 0x1UL) sum++;
	}
	return sum;

}

static void
process_smpl_buf(int id, unsigned long smpl_pmd_mask, int need_restart)
{
	static unsigned long last_overflow = ~0UL; /* initialize to biggest value possible */
	smpl_hdr_t *hdr = (smpl_hdr_t *)buf_addr;
	smpl_entry_t *ent;
	unsigned long count, entry, *reg, pos, msk;
	unsigned long entry_size;
	int j;
	


	printf("processing %s buffer at %p\n", need_restart==0 ? "leftover" : "", hdr);
	if (hdr->hdr_overflows <= last_overflow && last_overflow != ~0UL) {
		warning("skipping identical set of samples %lu <= %lu\n",
			hdr->hdr_overflows, last_overflow);
		return;	
	}
	last_overflow = hdr->hdr_overflows;

	count = hdr->hdr_count;

	ent   = (smpl_entry_t *)(hdr+1);
	pos   = (unsigned long)ent;
	entry = collect_samples;

	/*
	 * in this example program, we use fixed-size entries, therefore we
	 * can compute the entry size in advance. Perfmon-2 supports variable
	 * size entries.
	 */
	entry_size = sizeof(smpl_entry_t)+(bit_weight(smpl_pmd_mask)<<3);

	while(count--) {
		printf("entry %ld PID:%d CPU:%d IIP:0x%016lx\n",
			entry,
			ent->pid,
			ent->cpu,
			ent->ip);

		printf("\tOVFL: %d LAST_VAL: %lu\n", ent->ovfl_pmd, -ent->last_reset_val);

		/*
		 * print body: additional PMDs recorded
		 * PMD are recorded in increasing index order
		 */
		reg = (unsigned long *)(ent+1);

		for(j=0, msk = smpl_pmd_mask; msk; msk >>=1, j++) {	
			if ((msk & 0x1) == 0) continue;
			printf("PMD%-2d = 0x%016lx\n", j, *reg);
			reg++;
		}
		/*
		 * we could have removed this and used:
		 * ent = (smpl_entry_t *)reg
		 * instead.
		 */
		pos += entry_size;
		ent = (smpl_entry_t *)pos;
		entry++;
	}
	collect_samples = entry;

	/*
	 * reactivate monitoring once we are done with the samples
	 *
	 * Note that this call can fail with EBUSY in non-blocking mode
	 * as the task may have disappeared while we were processing
	 * the samples.
	 */
	if (need_restart && perfmonctl(id, PFM_RESTART, 0, 0) == -1) {
		if (errno != EBUSY)
			fatal_error("perfmonctl error PFM_RESTART errno %d\n",errno);
		else
			warning("PFM_RESTART: task has probably terminated \n");
	}
}

int
mainloop(char **arg)
{
	ctx_arg_t ctx;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_reg_t pc[NUM_PMCS];
	pfarg_load_t load_args;
	pfm_msg_t msg;
	unsigned long ovfl_count = 0UL;
	unsigned long sample_period;
	unsigned long smpl_pmd_mask = 0UL;
	pid_t pid;
	int status, ret, fd;
	unsigned int i, num_counters;

	/*
	 * intialize all locals
	 */
	memset(&ctx, 0, sizeof(ctx));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));

	/*
	 * locate events
	 */
	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");

	i = 2;

	if (i > num_counters) {
		i = num_counters;
		printf("too many events provided (max=%d events), using first %d event(s)\n", num_counters, i);
	}
	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;
	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = i;

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

	/*
	 * the PMC controlling the event ALWAYS come first, that's why this loop
	 * is safe even when extra PMC are needed to support a particular event.
	 */
	for (i=0; i < inp.pfp_event_count; i++) {
		pd[i].reg_num   = pc[i].reg_num;
		/* build sampling mask */
		smpl_pmd_mask  |= 1UL << pc[i].reg_num;
	}

	printf("smpl_pmd_mask=0x%lx\n", smpl_pmd_mask);

	/*
	 * now we indicate what to record when each counter overflows.
	 * In our case, we only have one sampling period and it is set for the
	 * first event. Here we indicate that when the sampling period expires
	 * then we want to record the value of all the other counters.
	 *
	 * We exclude the first counter in this case.
	 */
	smpl_pmd_mask  &= ~(1UL << pc[0].reg_num);

	pc[0].reg_smpl_pmds[0] = smpl_pmd_mask;

	/*
	 * we our sampling counter overflow, we want to be notified.
	 * The notification will come ONLY when the sampling buffer
	 * becomes full.
	 *
	 * We also activate randomization of the sampling period.
	 */
	pc[0].reg_flags	|= PFM_REGFL_OVFL_NOTIFY | PFM_REGFL_RANDOM;

	/*
	 * we also want to reset the other PMDs on
	 * every overflow. If we do not set
	 * this, the non-overflowed counters
	 * will be untouched.
	 */
	pc[0].reg_reset_pmds[0] |= smpl_pmd_mask;

	sample_period = 1000000UL;

	pd[0].reg_value       = (~0) - sample_period + 1;
	pd[0].reg_short_reset = (~0) - sample_period + 1;
	pd[0].reg_long_reset  = (~0) - sample_period + 1;
	/*
	 * setup randomization parameters, we allow a range of up to +256 here.
	 */
	pd[0].reg_random_seed = 5;
	pd[0].reg_random_mask = 0xff;


	printf("programming %u PMCS and %u PMDS\n", outp.pfp_pmc_count, inp.pfp_event_count);

	/*
	 * prepare context structure.
	 *
	 * format specific parameters MUST be concatenated to the regular
	 * pfarg_context_t structure. For convenience, the default sampling
	 * format provides a data structure that already combines the pfarg_context_t
	 * with what is needed fot this format.
	 */

	 /*
	  * We initialize the format specific information.
	  * The format is identified by its UUID which must be copied
	  * into the ctx_buf_fmt_id field.
	  */
	memcpy(ctx.ctx_arg.ctx_smpl_buf_id, buf_fmt_id, sizeof(pfm_uuid_t));

	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 *
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	ctx.buf_arg.buf_size = 8192;

	/*
	 * now create our perfmon context.
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, &ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * extract the file descriptor we will use to
	 * identify this newly created context
	 */
	fd = ctx.ctx_arg.ctx_fd;

	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	buf_addr = ctx.ctx_arg.ctx_smpl_vaddr;

	printf("context [%d] buffer mapped @%p\n", fd, buf_addr);

	/*
	 * Now program the registers
	 */
	if (perfmonctl(fd, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	/*
	 * initialize the PMDs
	 */
	if (perfmonctl(fd, PFM_WRITE_PMDS, pd, inp.pfp_event_count) == -1) {
		fatal_error("perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1) fatal_error("Cannot fork process\n");

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
	if (perfmonctl(fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
		fatal_error("perfmonctl error PFM_LOAD_CONTEXT errno %d\n",errno);
	}
	/*
	 * activate monitoring for stopped task.
	 * (nothing will be measured at this point
	 */
	if (perfmonctl(fd, PFM_START, NULL, 0) == -1) {
		fatal_error(" perfmonctl error PFM_START errno %d\n",errno);
	}
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
			fatal_error("cannot read perfmon msg: %s\n", strerror(errno));
		}
		switch(msg.type) {
			case PFM_MSG_OVFL: /* the sampling buffer is full */
				process_smpl_buf(fd, smpl_pmd_mask, 1);
				ovfl_count++;
				break;
			case PFM_MSG_END: /* monitored task terminated */
				printf("task terminated\n");
				goto terminate_session;
			default: fatal_error("unknown message type %d\n", msg.type);
		}
	}
terminate_session:
	/*
	 * cleanup child
	 */
	waitpid(pid, &status, 0);

	/*
	 * check for any leftover samples
	 */
	process_smpl_buf(fd, smpl_pmd_mask, 0);

	/*
	 * destroy perfmon context
	 */
	close(fd);

	printf("%lu samples collected in %lu buffer overflows\n", collect_samples, ovfl_count);

	return 0;
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;

	if (argc < 2)
		fatal_error("You must specify a command to execute\n");

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fatal_error("Can't initialize library\n");
	}

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */
	pfm_set_options(&pfmlib_options);

	return mainloop(argv+1);
}
