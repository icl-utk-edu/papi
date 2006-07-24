/*
 * smpl_pebs.c - PEBS self-contained (no libpfm) sampling example for 32-bit P4/Xeon ONLY
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
#include <perfmon/perfmon_p4_pebs_smpl.h>

#define NUM_PMCS	32
#define NUM_PMDS	32

#define SMPL_PERIOD	100000ULL	 /* must not use more bits than actual HW counter width */

typedef pfm_p4_pebs_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_p4_pebs_smpl_hdr_t		smpl_hdr_t;
typedef pfm_p4_pebs_smpl_entry_t	smpl_entry_t;
typedef pfm_p4_pebs_smpl_arg_t		smpl_arg_t;
#define FMT_UUID		 	PFM_P4_PEBS_SMPL_UUID

static uint64_t collected_samples;
static pfm_uuid_t buf_fmt_id = FMT_UUID;

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
	smpl_entry_t *ent;
	unsigned long count;
	uint64_t entry;

	if (hdr->hdr_overflows <= last_overflow && last_overflow != ~0) {
		warning("skipping identical set of samples %"PRIu64" <= %"PRIu64"\n",
			hdr->hdr_overflows, last_overflow);
		return;	
	}
	last_overflow = hdr->hdr_overflows;

	count = (hdr->hdr_ds.pebs_index - hdr->hdr_ds.pebs_buf_base)/sizeof(*ent);
	/*
	 * the beginning of the buffer does not necessarily follow the header
	 * due to alignement.
	 */
	ent   = (smpl_entry_t *)((unsigned long)(hdr+1)+ hdr->hdr_start_offs);
	entry = collected_samples;

	while(count--) {
		printf("entry %06"PRIu64" eflags:0x%08x EAX:0x%08x ESP:0x%08x IP:0x%08x\n",
			entry,
			ent->eflags,
			ent->eax,
			ent->esp,
			ent->ip);
		ent++;
		entry++;
	}
	collected_samples = entry;
}

static inline int
bit_weight(unsigned long x)
{
	int sum = 0;
	for (; x ; x>>=1) {
		if (x & 0x1UL) sum++;
	}
	return sum;

}

static inline unsigned int cpuid_eax(unsigned int op)
{
	unsigned long eax;

	__asm__("pushl %%ebx; cpuid; popl %%ebx"
			: "=a" (eax)
			: "0" (op)
			: "cx", "dx");
	return eax;
}

/*
 * check if processor has HT and if it is on
 * PEBS does not work with HT on.
 *
 * Not clear how to reliably test this from user space.
 * I have seen mentions of /dev/cpu/cpuid but this requires
 * that this option be enabled.
 *
 * Here we use another approach with /proc/cpuinfo, looking
 * at the siblings line
 */
static int
has_ht_on(void)
{
	FILE *fp1;
	uint32_t sib = 0;
	char buffer[128], *p, *value;

	memset(buffer, 0, sizeof(buffer));

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return 0;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL)
			break;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			break;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp("siblings", buffer, 8)) {
			sscanf(value, "%u", &sib);
			break;
		}
	}
	fclose(fp1);
	return sib < 2 ? 0 : 1;
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
	pid_t pid;
	unsigned long eax;
	int ret, fd, status, npmcs = 0;
	int family, model;

	eax = cpuid_eax(1);
	family = (eax>>8) & 0xf;
	model  = (eax>>4) & 0xf;
	/*
	 * the sanity test may be relaxed for other models in family 15
	 */
	if (family == 15 && model != 2) {
		fatal_error("this program only works for P4/Xeon with PEBS (found family=%d model=%d)\n", family, model);
	}

	if (sysconf(_SC_NPROCESSORS_ONLN) > 1 && has_ht_on())
		fatal_error("PEBS is not supported when HyperThreading is enabled\n");

	if (argc < 2)
		fatal_error("you need to pass a program to sample\n");

	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&ctx, 0, sizeof(ctx));
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&load_args, 0, sizeof(load_args));

	memcpy(ctx.ctx_smpl_buf_id, buf_fmt_id, sizeof(pfm_uuid_t));

	buf_arg.buf_size = getpagesize();
	buf_arg.cnt_reset = -SMPL_PERIOD;
	ctx.ctx_flags = 0;
	/*
	 * trigger interrupt when reached 90% of buffer
	 */
	buf_arg.intr_thres = (buf_arg.buf_size/sizeof(smpl_entry_t))*90/100;

	if (pfm_create_context(&ctx, &buf_arg, sizeof(buf_arg)) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s, maybe you do not have the P4/Xeon PEBS sampling format in the kernel.\n Check /sys/kernel/perfmon\n", strerror(errno));
	}

	/*
	 * extract the unique identifier for our context, a regular file descriptor
	 */
	fd = ctx.ctx_fd;

	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	buf_addr = mmap(NULL, ctx.ctx_smpl_buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	printf("context [%d] buffer mapped @%p\n", fd, buf_addr);
	if (buf_addr == MAP_FAILED)
		fatal_error("cannot mmap sampling buffer errno %d\n", errno);

	hdr = (smpl_hdr_t *)buf_addr;

	printf("pebs_base=0x%x pebs_end=0x%x index=0x%x\n"
	       "intr=0x%x (thres=%zu) version=%u.%u\n"
	       "entry_size=%zu ds_size=%zu\n",
			hdr->hdr_ds.pebs_buf_base,
			hdr->hdr_ds.pebs_abs_max,
			hdr->hdr_ds.pebs_index,
			hdr->hdr_ds.pebs_intr_thres,
			buf_arg.intr_thres,
			PFM_VERSION_MAJOR(hdr->hdr_version),
			PFM_VERSION_MINOR(hdr->hdr_version),
			sizeof(smpl_entry_t),
			sizeof(hdr->hdr_ds));

	if (PFM_VERSION_MAJOR(hdr->hdr_version) < 1)
		fatal_error("invalid buffer format version\n");

	/*
	 * using the replay_event event
	 *
	 * CRU_ESCR2.usr=1
	 * CRU_ESCR2.event_mask=1 (NBOGUS)
	 * CRU_ESCR2.event_select=0x9 (replay_event)
	 */
	pc[npmcs].reg_num   = 21;
	pc[npmcs].reg_value = (9ULL <<25) | (1ULL<<9) |(1ULL<<2);
	npmcs++;

	/*
	 * for PEBS, must use IQ_CCCR4 for thread0
	 * IQ_CCCR4.escr_select = 5
	 * IQ_CCCR4.enable= 1
	 * IQ_CCCR4.active_thread= 3
	 *
	 * We must disable 64-bit emulation by the kernel
	 * on the associated counter when using PEBS. Otherwise
	 * we received a spurious interrupt for every counter overflow.
	 */
	pc[npmcs].reg_num   = 31;
	pc[npmcs].reg_flags = PFM_REGFL_NO_EMUL64;
	pc[npmcs].reg_value = (5ULL << 13) | (1ULL<<12) | (3ULL<<16);
	npmcs++;

	/*
	 * PEBS_MATRIX_VERT.bit0=1 (1st level cache load miss retired)
	 */
	pc[npmcs].reg_num   = 63;
	pc[npmcs].reg_value = 1;
	npmcs++;

	/*
	 * PEBS_ENABLE.enable=1 (bit0)
	 * PEBS_ENABLE.uops=1 (bit 24)
	 * PEBS_ENABLE.my_thr=1 (bit 25)
	 */
	pc[npmcs].reg_num   = 64;
	pc[npmcs].reg_value = (1ULL<<25)|(1ULL<<24) | 1ULL;
	npmcs++;

	/*
	 * Must use IQ_CCCR4/IQ_CTR4 with PEBS for thread0
	 *
	 * IMPORTANT:
	 * 	SMPL_PERIOD MUST not exceed width of HW counter
	 * 	because no 64-bit virtualization is done by the
	 * 	kernel.
	 */
	pd[0].reg_num = 8;
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	pd[0].reg_value = -SMPL_PERIOD;
	pd[0].reg_long_reset = -SMPL_PERIOD;
	pd[0].reg_short_reset = -SMPL_PERIOD;

	/*
	 * Now program the registers
	 */
	if (pfm_write_pmcs(fd, pc, npmcs) == -1) {
		fatal_error("pfm_write_pmcs error errno %d\n",errno);
	}
	if (pfm_write_pmds(fd, pd, 1) == -1) {
		fatal_error("pfm_write_pmds error errno %d\n",errno);
	}

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(fd, F_SETOWN, getpid());
	if (ret == -1) {
		fatal_error("cannot setown: %s\n", strerror(errno));
	}
	signal(SIGCHLD, SIG_IGN);
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
	if (pfm_load_context(fd, &load_args) == -1) {
		fatal_error("pfm_load_context error errno %d\n",errno);
	}
	/*
	 * start monitoring
	 */
	if (pfm_start(fd, NULL) == -1) {
		fatal_error("pfm_start error errno %d\n",errno);
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
				if (pfm_restart(fd) == -1) {
					if (errno != EBUSY)
						fatal_error("pfm_restart error errno %d\n",errno);
					else
						warning("pfm_restart: task has probably terminated \n");
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

	close(fd);

	return 0;
}
