/*
 * smpl_amd64_ibs.c - AMD64 Family 10h IBS sampling
 *
 * Copyright (c) 2007 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Copyright (c) 2008 Advanced Mirco Devices Inc.
 * Contributed by Robert Richter <robert.richter@amd.com>
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

#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_amd64.h>

#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

typedef struct {
	int opt_no_show;
	int opt_block;
	int opt_setup;
} options_t;

enum {
	OPT_IBSOP,		/* 0: default */
	OPT_IBSFETCH,
	OPT_IBSOP_NATIVE,
};

typedef pfm_dfl_smpl_arg_t		smpl_fmt_arg_t;
typedef pfm_dfl_smpl_hdr_t		smpl_hdr_t;
typedef pfm_dfl_smpl_entry_t		smpl_entry_t;
typedef pfm_dfl_smpl_arg_t		smpl_arg_t;
#define FMT_NAME			PFM_DFL_SMPL_NAME

#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define PMD_IBSOP_NUM		7
#define PMD_IBSFETCH_NUM	3

static uint64_t collected_samples, collected_partial;
static options_t options;

static struct option the_options[]={
	{ "help", 0, 0,  1},
	{ "ovfl-block", 0, &options.opt_block, 1},
	{ "no-show", 0, &options.opt_no_show, 1},
	{ "ibsop", 0, &options.opt_setup, OPT_IBSOP},
	{ "ibsfetch", 0, &options.opt_setup, OPT_IBSFETCH},
	{ "ibsop-native", 0, &options.opt_setup, OPT_IBSOP_NATIVE},
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

static void
process_smpl_buf(smpl_hdr_t *hdr, uint64_t *smpl_pmds, unsigned int num_smpl_pmds, size_t entry_size)
{
	static uint64_t last_overflow = ~0; /* initialize to biggest value possible */
	static uint64_t last_count;
	smpl_entry_t *ent;
	size_t pos, count;
	ibsopdata_t *opdata;
	ibsopdata2_t *opdata2;
	ibsopdata3_t *opdata3;
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
				switch(j) {
					case 7:
						printf("PMD%-3d:0x%016"PRIx64"\n", j, *reg);
						/* check valid "record" bit */
						if ((*reg & (1ull<<18)) == 0) {
							printf("no data captured\n");
							goto skip;
						}
						break;
					case 9: /*IBSOPSDATA */
						opdata = (ibsopdata_t *)reg;
						printf("PMD%-3d:0x%016"PRIx64" : comptoret=%u tagtoretctr=%u opbrnresync=%u opmispret=%u opret=%u brntk=%u brnmips=%u bnrret=%u\n",
							j,
							*reg,
							opdata->reg.ibscomptoretctr,
							opdata->reg.ibstagtoretctr,
							opdata->reg.ibsopbrnresync,
							opdata->reg.ibsopmispreturn,
							opdata->reg.ibsopreturn,
							opdata->reg.ibsopbrntaken,
							opdata->reg.ibsopbrnmisp,
							opdata->reg.ibsopbrnret);
						break;
					case 10:
						opdata2 = (ibsopdata2_t *)reg;
						printf("PMD%-3d:0x%016"PRIx64" : reqsrc=%u reqdstproc=%s reqcachehitst=%u\n",
							j,
							*reg,
							opdata2->reg.nbibsreqsrc,
							opdata2->reg.nbibsreqdstproc ? "local" : "remote",
							opdata2->reg.nbibsreqcachehitst);
						break;
					case 11:
						opdata3 = (ibsopdata3_t *)reg;
						printf("PMD%-3d:0x%016"PRIx64" : ld=%u st=%u L1TLBmiss=%u L2TLBmiss=%u L1TLBhit2M=%u L1TLBhit1G=%u L2TLBhit2M=%u miss=%u misalign=%u ld_bankconf=%u  st_bankconf=%u st_to_ld_conf=%u st_to_ld_canc=%u UCaccess=%u WCaccess=%u lock=%u MAB=%u linevalid=%u physvalid=%u miss_lat=%u\n",
							j,
							*reg,
							opdata3->reg.ibsldop,
							opdata3->reg.ibsstop,
							opdata3->reg.ibsdcl1tlbmiss,
							opdata3->reg.ibsdcl2tlbmiss,
							opdata3->reg.ibsdcl1tlbhit2m,
							opdata3->reg.ibsdcl1tlbhit1g,
							opdata3->reg.ibsdcl2tlbhit2m,
							opdata3->reg.ibsdcmiss,
							opdata3->reg.ibsdcmissacc,
							opdata3->reg.ibsdcldbnkcon,
							opdata3->reg.ibsdcstbnkcon,
							opdata3->reg.ibsdcsttoldfwd,
							opdata3->reg.ibsdcsttoldcan,
							opdata3->reg.ibsdcucmemacc,
							opdata3->reg.ibsdcwcmemacc,
							opdata3->reg.ibsdclockedop,
							opdata3->reg.ibsdcmabhit,
							opdata3->reg.ibsdclinaddrvalid,
							opdata3->reg.ibsdcphyaddrvalid,
							opdata3->reg.ibsdcmisslat);
						break;

					default:
						printf("PMD%-3d:0x%016"PRIx64"\n", j, *reg);
				}
				reg++;
				n--;
			}
		}
skip:
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

static int
setup_pmu_ibsop_native(pfarg_pmr_t *pc, pfarg_pmr_t *pd, pfarg_pmd_attr_t *pa)
{
	uint64_t ibs_ops_smpl;

	/*
	 * OBSCTL sampling period (20 bits)
	 * bits 3:0 must be zero
	 */
	ibs_ops_smpl = 0xffff0;

	/*
	 * IBSOPSCTL config
	 *
	 * bit    17: enable
	 * bits 0-15: bit 19-4 of sampling period
	 */
	pc[0].reg_num = 5;
	pc[0].reg_value = (1ULL <<17) | ((ibs_ops_smpl >> 4) & 0xffffULL);

	/* IBSOPSCTL data
	 *
	 * point to the same MSR register. It correspond to the associated
	 * data register, i.e., the register to which the IBS interrupt will
	 * be associated.
	 *
	 * Randomization on IBS control register (IBSOPSCTL, IBSFETCHCTL) is
	 * ignored.
	 * 
	 * The value, short_reset, long_reset values are ignored. Use the
	 * corresponding PMC registers to set sampling period.
	 *
	 * If the last_reset-value is important for your program, then you can
	 * get it frmo the controlling PMC (4, 5). Alternatively, you can set
	 * the reg_value field to the value of the corresponding PMC register.
	 */
	pd[0].reg_num = 7;
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	pd[0].reg_value =  pc[0].reg_value;

	pa[0].reg_long_reset  = pc[0].reg_value;
	pa[0].reg_short_reset = pc[0].reg_value;

	pfm_bv_set(pa[0].reg_smpl_pmds, 7);
	pfm_bv_set(pa[0].reg_smpl_pmds, 8);
	pfm_bv_set(pa[0].reg_smpl_pmds, 9);
	pfm_bv_set(pa[0].reg_smpl_pmds, 10);
	pfm_bv_set(pa[0].reg_smpl_pmds, 11);
	pfm_bv_set(pa[0].reg_smpl_pmds, 12);
	pfm_bv_set(pa[0].reg_smpl_pmds, 13);

	return PFMLIB_SUCCESS;
}

static int
setup_pmu_ibsop(pfarg_pmr_t *pc, pfarg_pmr_t *pd, pfarg_pmd_attr_t *pa)
{
	pfmlib_amd64_input_param_t inp_mod;
	pfmlib_output_param_t outp;
	pfmlib_amd64_output_param_t outp_mod;
	int ret;
	
	memset(&inp_mod,0, sizeof(inp_mod));
	memset(&outp,0, sizeof(outp));
	memset(&outp_mod,0, sizeof(outp_mod));

	/* setup ibsopctl register */
	inp_mod.ibsop.maxcnt = 0xFFFF0;
	inp_mod.flags |= PFMLIB_AMD64_USE_IBSOP;

	/* setup Perfmon2 registers */
	ret = pfm_dispatch_events(NULL, &inp_mod, &outp, &outp_mod);
	if (ret != PFMLIB_SUCCESS) {
		fprintf(stderr, "cannot dispatch events: %s\n", pfm_strerror(ret));
		return ret;
	}
	if (outp.pfp_pmc_count != 1) {
		fprintf(stderr, "Unexpected PMC register count: %d\n",
			outp.pfp_pmc_count);
		return PFMLIB_ERR_INVAL;
	}
	if (outp.pfp_pmd_count != 1) {
		fprintf(stderr, "Unexpected PMD register count: %d\n",
			outp.pfp_pmd_count);
		return PFMLIB_ERR_INVAL;
	}
	if (outp_mod.ibsop_base != 0) {
		fprintf(stderr, "Unexpected IBSOP base register: %d\n",
			outp_mod.ibsop_base);
		return PFMLIB_ERR_INVAL;
	}

	/* PMC_IBSOPCTL */
	pc[0].reg_num   = outp.pfp_pmcs[0].reg_num;
	pc[0].reg_value = outp.pfp_pmcs[0].reg_value;
	/* PMD_IBSOPCTL */
	pd[0].reg_num   = outp.pfp_pmds[0].reg_num;
	pd[0].reg_value = 0;

	/* setup all IBSOP registers for sampling */
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	if (pd[0].reg_num > 64 - PMD_IBSOP_NUM) {
		fprintf(stderr, "Unexpected IBSOP base: %d\n",
			(int)pd[0].reg_num);
		return PFMLIB_ERR_INVAL;
	}
	pa[0].reg_smpl_pmds[0] =
		((1UL << PMD_IBSOP_NUM) - 1) << outp.pfp_pmds[0].reg_num;

	return PFMLIB_SUCCESS;
}

static int
setup_pmu_ibsfetch(pfarg_pmr_t *pc, pfarg_pmr_t *pd, pfarg_pmd_attr_t *pa)
{
	pfmlib_amd64_input_param_t inp_mod;
	pfmlib_output_param_t outp;
	pfmlib_amd64_output_param_t outp_mod;
	int ret;

	memset(&inp_mod,0, sizeof(inp_mod));
	memset(&outp,0, sizeof(outp));
	memset(&outp_mod,0, sizeof(outp_mod));

	/* setup ibsfetchctl register */
	inp_mod.ibsfetch.maxcnt = 0xFFFF0;
	inp_mod.flags |= PFMLIB_AMD64_USE_IBSFETCH;

	/* setup Perfmon2 registers */
	ret = pfm_dispatch_events(NULL, &inp_mod, &outp, &outp_mod);
	if (ret != PFMLIB_SUCCESS) {
		fprintf(stderr, "cannot dispatch events: %s\n", pfm_strerror(ret));
		return ret;
	}
	if (outp.pfp_pmc_count != 1) {
		fprintf(stderr, "Unexpected PMC register count: %d\n",
			outp.pfp_pmc_count);
		return PFMLIB_ERR_INVAL;
	}
	if (outp.pfp_pmd_count != 1) {
		fprintf(stderr, "Unexpected PMD register count: %d\n",
			outp.pfp_pmd_count);
		return PFMLIB_ERR_INVAL;
	}
	if (outp_mod.ibsfetch_base != 0) {
		fprintf(stderr, "Unexpected IBSFETCH base register: %d\n",
			outp_mod.ibsfetch_base);
		return PFMLIB_ERR_INVAL;
	}

	/* PMC_IBSFETCHCTL */
	pc[0].reg_num   = outp.pfp_pmcs[0].reg_num;
	pc[0].reg_value = outp.pfp_pmcs[0].reg_value;
	/* PMD_IBSFETCHCTL */
	pd[0].reg_num   = outp.pfp_pmds[0].reg_num;
	pd[0].reg_value = 0;

	/* setup all IBSFETCH registers for sampling */
	pd[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;
	if (pd[0].reg_num > 64 - PMD_IBSFETCH_NUM) {
		fprintf(stderr, "Unexpected IBSFETCH base: %d\n",
			(int)pd[0].reg_num);
		return PFMLIB_ERR_INVAL;
	}
	pa[0].reg_smpl_pmds[0] =
		((1UL << PMD_IBSFETCH_NUM) - 1) << outp.pfp_pmds[0].reg_num;

	return PFMLIB_SUCCESS;
}

int
mainloop(char **arg)
{
	pfarg_pmr_t pc[1];
	pfarg_pmr_t pd[1];
	pfarg_pmd_attr_t pa[1];

	smpl_hdr_t *hdr;
	smpl_arg_t buf_arg;
	struct timeval start_time, end_time;
	pfarg_msg_t msg;
	uint64_t ovfl_count = 0;
	size_t entry_size;
	void *buf_addr;
	pid_t pid;
	int status, ret, fd;
	int pmc_count, pmd_count;
	unsigned int num_smpl_pmds = 0;
	uint32_t ctx_flags;

	memset(pd, 0, sizeof(pd));
	memset(pa, 0, sizeof(pa));
	memset(pc, 0, sizeof(pc));

	/* defaults */
	num_smpl_pmds = 7;
	pmc_count = pmd_count = 1;

	switch (options.opt_setup) {
	case OPT_IBSOP:
		ret = setup_pmu_ibsop(pc, pd, pa);
		break;
	case OPT_IBSOP_NATIVE:
		ret = setup_pmu_ibsop_native(pc, pd, pa);
		break;
	case OPT_IBSFETCH:
		num_smpl_pmds = 3;
		ret = setup_pmu_ibsfetch(pc, pd, pa);
		break;
	default:
		ret = PFMLIB_ERR_NOTSUPP;
		break;
	}

	if (ret != PFMLIB_SUCCESS)
		fatal_error("cannot setup #%d\n", options.opt_setup);

	/*
	 * in this example program, we use fixed-size entries, therefore we
	 * can compute the entry size in advance. Perfmon-2 supports variable
	 * size entries.
	 */
	entry_size = sizeof(smpl_entry_t)+(num_smpl_pmds<<3);

	/*
	 * prepare session flags
	 */

	/*
	 * We initialize the format specific information.
	 * The format is identified by its UUID which must be copied
	 * into the ctx_buf_fmt_id field.
	 */
	ctx_flags = options.opt_block ? PFM_FL_NOTIFY_BLOCK : 0;

	/*
 	 * we use a samplig format, thus we are passing extra arguments
 	 */
	ctx_flags |= PFM_FL_SMPL_FMT;

	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 *
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	buf_arg.buf_size = 3*getpagesize();

	/*
	 * now create our perfmon session.
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
	if (pfm_write(fd, 0, PFM_RW_PMC, pc, pmc_count * sizeof(*pc)))
		fatal_error("pfm_write error errno %d\n",errno);
	/*
	 * initialize the PMDs
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds, reg_reset_pmds)
	 */
	if (pfm_write(fd, 0, PFM_RW_PMD_ATTR, pd, pmd_count * sizeof(*pd)))
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
		fatal_error("pfm_start error errno %d\n",errno);
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
				process_smpl_buf(hdr, pa[0].reg_smpl_pmds, num_smpl_pmds, entry_size);
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
	wait4(pid, &status, 0, NULL);
	gettimeofday(&end_time, NULL);

	/*
	 * check for any leftover samples
	 */
	process_smpl_buf(hdr, pa[0].reg_smpl_pmds, num_smpl_pmds, entry_size);

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

	return 0;
}

static void
usage(void)
{
	printf("usage: smpl_amd64_ibs [-hdv] [--help] [--no-show] "
	       "[--ovfl-block] [--ibsop] [--ibsfetch] [--ibsop-native] cmd\n");
}

int
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	int c, ret;

	/*
	 * pass options to library
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug   = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for verbose */

	while ((c=getopt_long(argc, argv,"+hvd", the_options, 0)) != -1) {
		switch(c) {
			case 0: continue;

			case 1:
			case 'h':
				usage();
				exit(0);
			case 'v':
				pfmlib_options.pfm_verbose = 1;
				continue;
			case 'd':
				pfmlib_options.pfm_debug = 1;
				continue;
			default:
				fatal_error("");
		}
	}

	if (argv[optind] == NULL) {
		fatal_error("You must specify a command to execute\n");
	}
	
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	pfm_get_pmu_type(&c);
	if (c != PFMLIB_AMD64_PMU) {
		fatal_error("not running on an AMD64 processor\n");
	}
	/*
	 * XXX: would need to check for family 10h
	 */

	return mainloop(argv+optind);
}
