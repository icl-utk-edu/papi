/*
 * smpl_standalone.c - self-contained (no libpfm) sampling example
 *
 * The *_standalone.c examples are to be used with PMU model not supported
 * directly by libpfm as simple test programs to exercise the kernel API.
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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <syscall.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/mman.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

#include "standalone.h"

#define NUM_PMCS	32
#define NUM_PMDS	32

#define SMPL_PERIOD	1000000ULL

typedef pfm_dfl_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_dfl_smpl_hdr_t		smpl_hdr_t;
typedef pfm_dfl_smpl_entry_t		smpl_entry_t;
typedef pfm_dfl_smpl_arg_t		smpl_arg_t;
#define FMT_UUID		 	PFM_DFL_SMPL_UUID

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

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6

static inline void pfm_bv_set(uint64_t *bv, uint16_t rnum)
{
	bv[rnum>>LBPL] |= 1UL << (rnum&(BPL-1));
}

static inline int pfm_bv_isset(uint64_t *bv, uint16_t rnum)
{
	return bv[rnum>>LBPL] & (1UL <<(rnum&(BPL-1))) ? 1 : 0;
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
process_smpl_buf(smpl_hdr_t *hdr, uint64_t *smpl_pmds, unsigned int num_smpl_pmds, size_t entry_size)
{
	static uint64_t last_overflow = ~0; /* initialize to biggest value possible */
	static uint64_t last_count;
	smpl_entry_t *ent;
	size_t pos, count;
	uint64_t entry, *reg;
	unsigned int j, n;
	
	if (hdr->hdr_overflows == last_overflow && last_count == hdr->hdr_count) {
		warning("skipping identical set of samples %"PRIu64" = %"PRIu64"\n",
			hdr->hdr_overflows, last_overflow);
		return;
	}
	last_overflow = hdr->hdr_overflows;
	count = last_count = hdr->hdr_count;

	ent   = (smpl_entry_t *)(hdr+1);
	pos   = (unsigned long)ent;
	entry = collected_samples;

	while(count--) {
		printf("entry %"PRIu64" PID:%d TID:%d CPU:%d LAST_VAL:%"PRIu64" IIP:0x%llx\n",
			entry,
			ent->tgid,
			ent->pid,
			ent->cpu,
			-ent->last_reset_val,
			(unsigned long long)ent->ip);

		/*
		 * print body: additional PMDs recorded
		 * PMD are recorded in increasing index order
		 */
		reg = (uint64_t *)(ent+1);

		n = num_smpl_pmds;
		for(j=0; n; j++) {	
			if (pfm_bv_isset(smpl_pmds, j)) {
				printf("PMD%-2d = 0x%016"PRIx64"\n", j, *reg);
				reg++;
				n--;
			}
		}
		pos += entry_size;
		ent = (smpl_entry_t *)pos;
		entry++;
	}
	collected_samples = entry;
}

static void
program_pmu_xeon(int fd, int cpu, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	unsigned int npmcs = 0;
	unsigned int npmds = 0;

	printf("program set up for P4/Xeon/EM64T\n");
	/*
	 * measuring instr_retired with CRU_ESCR0, IQ_CCCR0, IQ_CTR0
	 * CRU_ESCR0.event_select=2
	 * CRU_ESCR0.usr=1 (at the user level)
	 * CRU_ESCR0.tag_enable=1
	 * CRU_ESCR0.event_mask=NBOGUSTAG
	 * IQ_CCCR0.cccr_select=4
	 * IQ_CCCR0.enable=1
	 * IQ_CCCR0.active_thread=3 (either logical CPU is active)
	 */
	pc[npmcs].reg_num   = 20; /* CRU_ESCR0 */
	pc[npmcs].reg_value = (2ULL <<25) | (1ULL<<9) |(1ULL<<2) | (1ULL<<4);
	npmcs++;
	pc[npmcs].reg_num   = 29; /* IQ_CCCR0 */
	pc[npmcs].reg_value = (4ULL<<13) | (1ULL<<12) | (3ULL <<16);
	npmcs++;

	*num_pmcs = npmcs;

	/*
	 * IQ_CTR0
	 */
	pd[npmds].reg_num   = 6;
	pd[npmds].reg_flags = PFM_REGFL_OVFL_NOTIFY|PFM_REGFL_RANDOM;
	pd[npmds].reg_value       = - SMPL_PERIOD;
	pd[npmds].reg_long_reset  = - SMPL_PERIOD;
	pd[npmds].reg_short_reset = - SMPL_PERIOD;
	pd[npmds].reg_random_seed = 5;
	pd[npmds].reg_random_mask = 0xff;
	npmds++;

	*num_pmds = npmds;
}

static void
program_pmu_mips(int fd, int cpu, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	unsigned int npmcs = 0;
	unsigned int npmds = 0;

	if (cpu == STANDALONE_MIPS20K) {
	    pc[npmcs].reg_num = 0;
	    pc[npmcs].reg_value = 0xfULL << 5 | 0x8ULL; /* INST, user mode*/
	    npmcs++;
	    pd[npmds].reg_num = 0;
	} else if (cpu == STANDALONE_MIPS5K) {
	    pc[npmcs].reg_num = 1;
	    pc[npmcs].reg_value = 0xfULL << 5 | 0x8ULL; /* INST, user mode*/
	    npmcs++;
	    pd[npmds].reg_num   = 1;
	} 

	pd[npmds].reg_flags = PFM_REGFL_OVFL_NOTIFY|PFM_REGFL_RANDOM;
	pd[npmds].reg_value       = - SMPL_PERIOD;
	pd[npmds].reg_long_reset  = - SMPL_PERIOD;
	pd[npmds].reg_short_reset = - SMPL_PERIOD;
	pd[npmds].reg_random_seed = 5;
	pd[npmds].reg_random_mask = 0xff;
	npmds++;

	printf("Measuring floating point instr. user mode: reg. %u value 0x%"PRIx64"\n",pc[0].reg_num,pc[0].reg_value);

	*num_pmcs = npmcs;
	*num_pmds = npmds;
}

static void
program_pmu(int fd, pfarg_pmc_t *pc, unsigned int *num_pmcs, pfarg_pmd_t *pd, unsigned int *num_pmds)
{
	int cpu;

	cpu = cpu_detect();
	switch(cpu) {
		case STANDALONE_MIPS5K:
		case STANDALONE_MIPS20K:
			program_pmu_mips(fd, cpu, pc, num_pmcs, pd, num_pmds);
			break;
		case STANDALONE_P4:
			program_pmu_xeon(fd, cpu, pc, num_pmcs, pd, num_pmds);
			break;
		default:
			fatal_error("PMU model not supported by this program. It may be directly supported by libpfm, try the other examples\n");
	}

}
int
main(int argc, char **argv)
{
	pfarg_pmd_t pd[NUM_PMDS];
	pfarg_pmc_t pc[NUM_PMCS];
	pfarg_ctx_t ctx;
	smpl_arg_t buf_arg;
	pfarg_load_t load_args;
	pfm_msg_t msg;
	smpl_hdr_t *hdr;
	void *buf_addr;
	size_t entry_size;
	pid_t pid;
	int ret, fd, status;
	unsigned int num_pmcs, num_pmds, num_smpl_pmds;


	if (argc < 2)
		fatal_error("you need to pass a program to sample\n");

	num_pmcs = num_pmds = num_smpl_pmds = 0;

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&ctx, 0, sizeof(ctx));
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&load_args, 0, sizeof(load_args));

	buf_arg.buf_size = 3*getpagesize();
	ctx.ctx_flags = PFM_FL_NOTIFY_BLOCK;

	fd = pfm_create_context(&ctx, "default", &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}

	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	buf_addr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (buf_addr == MAP_FAILED)
		fatal_error("cannot mmap sampling buffer errno %d\n", errno);

	printf("context [%d] buffer mapped @%p\n", fd, buf_addr);

	hdr = (smpl_hdr_t *)buf_addr;
	printf("hdr_cur_offs=%llu version=%u.%u\n",
			(unsigned long long)hdr->hdr_cur_offs,
			PFM_VERSION_MAJOR(hdr->hdr_version),
			PFM_VERSION_MINOR(hdr->hdr_version));

	if (PFM_VERSION_MAJOR(hdr->hdr_version) < 1)
		fatal_error("invalid buffer format version\n");

	program_pmu(fd, pc, &num_pmcs, pd, &num_pmds);

	entry_size = sizeof(smpl_entry_t)+(num_smpl_pmds<<3);
	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(fd, pc, num_pmcs))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);
	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */
	if (pfm_write_pmds(fd, pd, num_pmds))
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(fd, F_SETOWN, getpid());
	if (ret == -1)
		fatal_error("cannot setown: %s\n", strerror(errno));

	signal(SIGCHLD, SIG_IGN);
	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1)
		fatal_error("Cannot fork process\n");

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
		warning("task %s [%d] exited already status %d\n", argv[1], pid, WEXITSTATUS(status));
		goto terminate_session;
	}

	/*
	 *  attach the context to self
	 */
	load_args.load_pid = pid;
	if (pfm_load_context(fd, &load_args))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * start monitoring
	 */
	if (pfm_start(fd, NULL))
		fatal_error("pfm_start error errno %d\n",errno);

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
				process_smpl_buf(hdr, pd[0].reg_smpl_pmds, num_smpl_pmds, entry_size);
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
					else
						warning("pfm_restart: task has probably terminated \n");
				}
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
	wait4(pid, &status, 0, NULL);

	/*
	 * check for any leftover samples
	 */
	process_smpl_buf(hdr, pd[0].reg_smpl_pmds, num_smpl_pmds, entry_size);

	munmap(buf_addr, (size_t)buf_arg.buf_size);

	close(fd);

	return 0;
}
