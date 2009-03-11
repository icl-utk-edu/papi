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
#include <getopt.h>
#include <time.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

#include "detect_pmcs.h"

#define SAMPLING_PERIOD	100000

typedef struct {
	int opt_no_show;
	int opt_block;
} options_t;

typedef pfm_dfl_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_dfl_smpl_hdr_t		smpl_hdr_t;
typedef pfm_dfl_smpl_entry_t		smpl_entry_t;
typedef pfm_dfl_smpl_arg_t		smpl_arg_t;
#define FMT_NAME			PFM_DFL_SMPL_NAME

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

static uint64_t collected_samples, collected_partial;
static options_t options;

static struct option the_options[]={
	{ "help", 0, 0,  1},
	{ "ovfl-block", 0, &options.opt_block, 1},
	{ "no-show", 0, &options.opt_no_show, 1},
	{ 0, 0, 0, 0}
};

static void fatal_error(char *fmt,...) __attribute__((noreturn));

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

static inline void pfm_bv_copy(uint64_t *d, uint64_t *j, uint16_t n)
{
	if (n <= BPL)
		*d = *j;
	else {
		memcpy(d, j, (n>>LBPL)*sizeof(uint64_t));
	}
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

void
show_task_rusage(const struct timeval *start, const struct timeval *end, const struct rusage *ru)
{
	long secs, suseconds, end_usec;

	 secs     =  end->tv_sec - start->tv_sec;
	 end_usec = end->tv_usec;

	if (end_usec < start->tv_usec) {
      		end_usec += 1000000;
      		secs--;
    	}

  	suseconds = end_usec - start->tv_usec;

	printf ("real %ldh%02ldm%02ld.%03lds user %ldh%02ldm%02ld.%03lds sys %ldh%02ldm%02ld.%03lds\n", 
		secs / 3600, 
		(secs % 3600) / 60, 
		secs % 60,
		suseconds / 1000,

		ru->ru_utime.tv_sec / 3600, 
		(ru->ru_utime.tv_sec % 3600) / 60, 
		ru->ru_utime.tv_sec% 60,
		(long)(ru->ru_utime.tv_usec / 1000),

		ru->ru_stime.tv_sec / 3600, 
		(ru->ru_stime.tv_sec % 3600) / 60, 
		ru->ru_stime.tv_sec% 60,
		(long)(ru->ru_stime.tv_usec / 1000)
		);
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
	
	if (hdr->hdr_overflows == last_overflow && hdr->hdr_count == last_count) {
		warning("skipping identical set of samples %"PRIu64" = %"PRIu64"\n",
			hdr->hdr_overflows, last_overflow);
		return;	
	}

	count = hdr->hdr_count;

	if (options.opt_no_show) {
		collected_samples += count;
		return;
	}

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
				printf("PMD%-3d:0x%016"PRIx64"\n", j, *reg);
				reg++;
				n--;
			}
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
	smpl_arg_t buf_arg;
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfarg_pmd_attr_t pd[NUM_PMDS];
	pfarg_pmr_t pc[NUM_PMCS];
	pfarg_sinfo_t sif;
	struct timeval start_time, end_time;
	struct rusage rusage;
	pfarg_msg_t msg;
	uint64_t ovfl_count = 0;
	uint32_t ctx_flags;
	size_t entry_size;
	void *buf_addr;
	pid_t pid;
	int status, ret, fd;
	unsigned int i, num_counters;
	unsigned int max_pmd = 0, num_smpl_pmds = 0;

	/*
	 * intialize all locals
	 */
	memset(&buf_arg, 0, sizeof(buf_arg));
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(pd, 0, sizeof(pd));
	memset(pc, 0, sizeof(pc));
	memset(&sif, 0, sizeof(sif));

	pfm_get_num_counters(&num_counters);

	if (pfm_get_cycle_event(&inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event\n");

	if (pfm_get_inst_retired_event(&inp.pfp_events[1]) != PFMLIB_SUCCESS)
		fatal_error("cannot find inst retired event\n");

	i = 2;

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 * 	PFM_PLM0 : kernel level
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
	get_sif(0, &sif);
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
	for (i=0; i < outp.pfp_pmd_count; i++) {
		pd[i].reg_num = outp.pfp_pmds[i].reg_num;
		/*
		 * skip first counter (sampling period)
		 * track highest PMD 
		 */
		if (i) {
			pfm_bv_set(pd[0].reg_smpl_pmds, pd[i].reg_num);
			if (pd[i].reg_num > max_pmd)
				max_pmd = pd[i].reg_num;
			num_smpl_pmds++;
		}
	}

	/*
	 * we our sampling counter overflow, we want to be notified.
	 * The notification will come ONLY when the sampling buffer
	 * becomes full.
	 *
	 * We also activate randomization of the sampling period.
	 */
	pd[0].reg_flags	|= PFM_REGFL_OVFL_NOTIFY | PFM_REGFL_RANDOM;

	/*
	 * we also want to reset the other PMDs on
	 * every overflow. If we do not set
	 * this, the non-overflowed counters
	 * will be untouched.
	 */
	pfm_bv_copy(pd[0].reg_reset_pmds, pd[0].reg_smpl_pmds, max_pmd);

	pd[0].reg_value       = - SAMPLING_PERIOD;
	pd[0].reg_short_reset = - SAMPLING_PERIOD;
	pd[0].reg_long_reset  = - SAMPLING_PERIOD;

	/*
	 * setup randomization parameters, we allow a range of up to +256 here.
	 */
	pd[0].reg_random_mask = 0xff;

	/*
	 * in this example program, we use fixed-size entries, therefore we
	 * can compute the entry size in advance. Perfmon-2 supports variable
	 * size entries.
	 */
	entry_size = sizeof(smpl_entry_t)+(num_smpl_pmds<<3);

	printf("programming %u PMCS and %u PMDS\n", outp.pfp_pmc_count, inp.pfp_event_count);

	/*
 	 * indicate we are using a smapling format, i.e., extra arguments
 	 * passed to pfm_create_session()
 	 */
	ctx_flags = PFM_FL_SMPL_FMT;

	/*
	 * add overflow blocking is necessary
	 */
	ctx_flags |= options.opt_block ? PFM_FL_NOTIFY_BLOCK : 0;

	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 *
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	buf_arg.buf_size = 3*getpagesize()+512;

	/*
	 * now create our session
	 */
	fd = pfm_create(ctx_flags, NULL, FMT_NAME, &buf_arg, sizeof(buf_arg));
	if (fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("cannot create session %s\n", strerror(errno));
	}

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
	 * Now program the registers
	 */
	if (pfm_write(fd, 0, PFM_RW_PMC, pc, outp.pfp_pmc_count * sizeof(*pc)))
		fatal_error("pfm_write error errno %d\n",errno);
	/*
	 * initialize the PMDs
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds, reg_reset_pmds)
	 */
	if (pfm_write(fd, 0, PFM_RW_PMD_ATTR, pd, outp.pfp_pmd_count * sizeof(*pd)))
		fatal_error("pfm_write(PMD) error errno %d\n",errno);

	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1)
		fatal_error("Cannot fork process\n");

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
	 * attach session to stopped task
	 */
	if (pfm_attach(fd, 0, pid))
		fatal_error("pfm_attach error errno %d\n",errno);

	/*
	 * activate monitoring for stopped task.
	 * (nothing will be measured at this point
	 */
	if (pfm_set_state(fd, 0, PFM_ST_START))
		fatal_error("pfm_set_state(start) error errno %d\n",errno);
	/*
	 * detach child. Side effect includes
	 * activation of monitoring.
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	gettimeofday(&start_time, NULL);

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
				ovfl_count++;
				/*
				 * reactivate monitoring once we are done with the samples
				 *
				 * Note that this call can fail with EBUSY in non-blocking mode
				 * as the task may have disappeared while we were processing
				 * the samples.
				 */
				if (pfm_set_state(fd, 0, PFM_ST_RESTART)) {
					if (errno != EBUSY)
						fatal_error("pfm_set_state(restart) error errno %d\n",errno);
					else
						warning("pfm_set_state(restart): task probably terminated \n");
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
	wait4(pid, &status, 0, &rusage);
	gettimeofday(&end_time, NULL);

	/*
	 * check for any leftover samples
	 */
	process_smpl_buf(hdr, pd[0].reg_smpl_pmds, num_smpl_pmds, entry_size);

	/*
	 * close file descritor. Because of mmap() the number of reference to the
	 * "file" is 2, thus the session is only freed when the last reference is closed
	 * either by closed or munmap() depending on the order in which those calls are
	 * made:
	 * 	- close() -> munmap(): session and buffer destroyed after munmap().
	 * 			       buffer remains accessible after close().
	 * 	- munmap() -> close(): buffer unaccessible after munmap(), session and
	 * 			       buffer destroyed after close().
	 *
	 * It is important to free the resources cleanly, especially because the sampling
	 * buffer reserves locked memory.
	 */
	close(fd);

	/*
	 * unmap buffer, actually free the buffer and session because placed after
	 * the close(), i.e. is the last reference. See comments about close() above.
	 */
	ret = munmap(hdr, (size_t)buf_arg.buf_size);
	if (ret)
		fatal_error("cannot unmap buffer: %s\n", strerror(errno));

	printf("%"PRIu64" samples (%"PRIu64" in partial buffer) collected in %"PRIu64" buffer overflows\n",
		collected_samples,
		collected_partial,
		ovfl_count);
	show_task_rusage(&start_time, &end_time, &rusage);

	return 0;
}

static void
usage(void)
{
	printf("usage: task_smpl [-h] [--help] [--no-show] [--ovfl-block] cmd\n");
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	int c, ret;

	while ((c=getopt_long(argc, argv,"h", the_options, 0)) != -1) {
		switch(c) {
			case 0: continue;

			case 1:
			case 'h':
				usage();
				exit(0);
			default:
				fatal_error("");
		}
	}

	if (argv[optind] == NULL) {
		fatal_error("You must specify a command to execute\n");
	}
	
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

	return mainloop(argv+optind);
}
