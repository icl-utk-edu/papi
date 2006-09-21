/*
 * mont_btb.c - example of how use the BTB with the Dual-Core Itanium 2  PMU
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <perfmon/perfmon.h>
#include <perfmon/perfmon_default_smpl.h>
#include <perfmon/pfmlib_montecito.h>

typedef pfm_default_smpl_hdr_t		etb_hdr_t;
typedef pfm_default_smpl_entry_t	etb_entry_t;
typedef pfm_default_smpl_ctx_arg_t	etb_ctx_arg_t;
#define BTB_FMT_UUID	       		PFM_DEFAULT_SMPL_UUID

static pfm_uuid_t buf_fmt_id = BTB_FMT_UUID;


#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#define MAX_EVT_NAME_LEN	128
#define MAX_PMU_NAME_LEN	32

/*
 * The BRANCH_EVENT is increment by 1 for each branch event. Such event is composed of
 * two entries in the BTB: a source and a target entry. The BTB is full after 4 branch
 * events.
 */
#define SMPL_PERIOD	(4UL*256)

/*
 * We use a small buffer size to exercise the overflow handler
 */
#define SMPL_BUF_NENTRIES	64

#define M_PMD(x)		(1UL<<(x))

#define ETB_REGS_MASK		(M_PMD(38)| M_PMD(39)| \
		                 M_PMD(48)|M_PMD(49)|M_PMD(50)|M_PMD(51)|M_PMD(52)|M_PMD(53)|M_PMD(54)|M_PMD(55)|\
				 M_PMD(56)|M_PMD(57)|M_PMD(58)|M_PMD(59)|M_PMD(60)|M_PMD(61)|M_PMD(62)|M_PMD(63))
static void *smpl_vaddr;
static size_t entry_size;
static int id;

#if defined(__ECC) && defined(__INTEL_COMPILER)
/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>

#define hweight64(x)	_m64_popcnt(x)

#elif defined(__GNUC__)

static __inline__ int
hweight64 (unsigned long x)
{
	unsigned long result;
	__asm__ ("popcnt %0=%1" : "=r" (result) : "r" (x));
	return (int)result;
}

#else
#error "you need to provide inline assembly from your compiler"
#endif

/*
 * we don't use static to make sure the compiler does not inline the function
 */
long func1(void) { return random();}
long func2(void) { return random();}

long
do_test(unsigned long loop)
{
	long sum  = 0;

	while(loop--) {
		if (loop & 0x1)
			sum += func1();
		else
			sum += loop + func2();
	}
	return sum;
}


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
 * print content of sampling buffer
 *
 * XXX: using stdio to print from a signal handler is not safe with multi-threaded
 * applications
 */
#define safe_printf	printf

static void
show_etb_reg(int j, pfm_mont_pmd_reg_t reg, pfm_mont_pmd_reg_t pmd39)
{
	unsigned long bruflush, b1, etb_ext;
	unsigned long addr;
	int is_valid;

	is_valid = reg.pmd48_63_etb_mont_reg.etb_s == 0 && reg.pmd48_63_etb_mont_reg.etb_mp == 0 ? 0 : 1;

	/*
	 * the joy of the ETB extension register layout!
	 */
	if (j < 8)
		etb_ext = (pmd39.pmd_val>>(8*j)) & 0xf;
	else
		etb_ext = (pmd39.pmd_val>>(4+8*(j-1))) & 0xf;

	b1       = etb_ext & 0x1;
	bruflush = (etb_ext >> 1) & 0x1;

	safe_printf("\tPMD%-2d: 0x%016lx s=%d mp=%d bru=%ld b1=%ld valid=%c\n",
		j+48,
		reg.pmd_val,
		reg.pmd48_63_etb_mont_reg.etb_s,
		reg.pmd48_63_etb_mont_reg.etb_mp,
		bruflush, b1,
		is_valid ? 'Y' : 'N');


	if (!is_valid) return;

	if (reg.pmd48_63_etb_mont_reg.etb_s) {
		addr = (reg.pmd48_63_etb_mont_reg.etb_addr+b1)<<4;
		addr |= reg.pmd48_63_etb_mont_reg.etb_slot < 3 ? reg.pmd48_63_etb_mont_reg.etb_slot : 0;

		safe_printf("\t       Source Address: 0x%016lx\n"
			    "\t       Taken=%c Prediction:%s\n\n",
			 addr,
			 reg.pmd48_63_etb_mont_reg.etb_slot < 3 ? 'Y' : 'N',
			 reg.pmd48_63_etb_mont_reg.etb_mp ? "FE Failure" : 
			 bruflush ? "BE Failure" : "Success");
	} else {
		safe_printf("\t       Target Address:0x%016lx\n\n",
			 (unsigned long)(reg.pmd48_63_etb_mont_reg.etb_addr<<4));
	}
}

static void
show_etb(pfm_mont_pmd_reg_t *etb)
{
	int i, last;
	pfm_mont_pmd_reg_t pmd38, pmd39;

	pmd38.pmd_val = etb[0].pmd_val;
	pmd39.pmd_val = etb[1].pmd_val;

	i    = pmd38.pmd38_mont_reg.etbi_full ? pmd38.pmd38_mont_reg.etbi_ebi : 0;
	last = pmd38.pmd38_mont_reg.etbi_ebi;

	safe_printf("btb_trace: i=%d last=%d bbi=%d full=%d\n",
		i,
		last,
		pmd38.pmd38_mont_reg.etbi_ebi,
		pmd38.pmd38_mont_reg.etbi_full);

	do {
		show_etb_reg(i, etb[i], pmd39);
		i = (i+1) % 16;
	} while (i != last);
}

void
process_smpl_buffer(void)
{
	etb_hdr_t	*hdr;
	etb_entry_t	*ent;
	unsigned long pos;
	unsigned long smpl_entry = 0;
	pfm_mont_pmd_reg_t *reg;
	size_t count;
	static unsigned long last_ovfl = ~0UL;


	hdr = (etb_hdr_t *)smpl_vaddr;

	/*
	 * check that we are not diplaying the previous set of samples again.
	 * Required to take care of the last batch of samples.
	 */
	if (hdr->hdr_overflows <= last_ovfl && last_ovfl != ~0UL) {
		printf("skipping identical set of samples %lu <= %lu\n", hdr->hdr_overflows, last_ovfl);
		return;
	}

	pos = (unsigned long)(hdr+1);
	count = hdr->hdr_count;
	/*
	 * walk through all the entries recored in the buffer
	 */
	while(count--) {

		ent = (etb_entry_t *)pos;
		/*
		 * print entry header
		 */
		safe_printf("Entry %ld PID:%d TID:%d CPU:%d STAMP:0x%lx IIP:0x%016lx\n",
			smpl_entry++,
			ent->tgid,
			ent->pid,
			ent->cpu,
			ent->tstamp,
			ent->ip);

		/*
		 * point to first recorded register (always contiguous with entry header)
		 */
		reg = (pfm_mont_pmd_reg_t*)(ent+1);

		/*
		 * in this particular example, we have pmd48-pmd63 has the ETB. We have also
		 * included pmd38/pmd39 (ETB index and extenseion) has part of the registers
		 * to record. This trick allows us to get the index to decode the sequential
		 * order of the BTB.
		 *
		 * Recorded registers are always recorded in increasing index order. So we know
		 * that where to find pmd38/pmd39.
		 */
		show_etb(reg);

		/*
		 * move to next entry
		 */
		pos += entry_size;
	}
}

static void
overflow_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	process_smpl_buffer();

	/*
	 * And resume monitoring
	 */
	if (perfmonctl(id, PFM_RESTART, NULL, 0))
		fatal_error("pfm_restart errno %d\n", errno);
}


int
main(void)
{
	int ret;
	int type = 0;
	pfarg_reg_t pd[NUM_PMDS];
	pfarg_reg_t pc[NUM_PMCS];
	pfmlib_input_param_t inp;
	pfmlib_output_param_t outp;
	pfmlib_mont_input_param_t mont_inp;
	etb_ctx_arg_t ctx;
	pfarg_load_t load_args;
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	unsigned int i;

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS)
		fatal_error("Can't initialize library\n");

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	if (type != PFMLIB_MONTECITO_PMU) {
		char model[MAX_PMU_NAME_LEN];
		pfm_get_pmu_name(model, MAX_PMU_NAME_LEN);
		fatal_error("this program does not work with %s PMU\n", model);
	}

	/*
	 * Install the overflow handler (SIGIO)
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = (sig_t)overflow_handler;
	sigaction (SIGIO, &act, 0);


	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfmlib_options.pfm_verbose = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	memset(pd, 0, sizeof(pd));
	memset(&ctx, 0, sizeof(ctx));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&inp,0, sizeof(inp));
	memset(&outp,0, sizeof(outp));
	memset(&mont_inp,0, sizeof(mont_inp));

	/*
	 * Before calling pfm_find_dispatch(), we must specify what kind
	 * of branches we want to capture. We are interested in all taken
	 * branches * therefore we program we set the various fields to:
	 */
	mont_inp.pfp_mont_etb.etb_used = 1;

	mont_inp.pfp_mont_etb.etb_tm  = 0x2;
	mont_inp.pfp_mont_etb.etb_ptm = 0x3;
	mont_inp.pfp_mont_etb.etb_ppm = 0x3;
	mont_inp.pfp_mont_etb.etb_brt = 0x0;
	mont_inp.pfp_mont_etb.etb_plm = PFM_PLM3;

	/*
	 * To count the number of occurence of this instruction, we must
	 * program a counting monitor with the IA64_TAGGED_INST_RETIRED_PMC8
	 * event.
	 */
	if (pfm_find_full_event("BRANCH_EVENT", &inp.pfp_events[0]) != PFMLIB_SUCCESS)
		fatal_error("cannot find event BRANCH_EVENT\n");

	/*
	 * set the (global) privilege mode:
	 * 	PFM_PLM3 : user level only
	 */
	inp.pfp_dfl_plm   = PFM_PLM3;

	/*
	 * how many counters we use
	 */
	inp.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&inp, &mont_inp, &outp, NULL)) != PFMLIB_SUCCESS)
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

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
	ctx.buf_arg.buf_size = getpagesize();

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(0, PFM_CREATE_CONTEXT, &ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/*
	 * extract our file descriptor
	 */
	id = ctx.ctx_arg.ctx_fd;

	/*
	 * retrieve the virtual address at which the sampling
	 * buffer has been mapped
	 */
	smpl_vaddr = ctx.ctx_arg.ctx_smpl_vaddr;
	if (smpl_vaddr == MAP_FAILED)
		fatal_error("cannot mmap sampling buffer errno %d\n", errno);

	printf("Sampling buffer mapped at %p\n", smpl_vaddr);

	/*
	 * Now prepare the argument to initialize the PMDs and PMCS.
	 * We must pfp_pmc_count to determine the number of PMC to intialize.
	 * We must use pfp_event_count to determine the number of PMD to initialize.
	 * Some events cause extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
	 */

	for (i=0; i < outp.pfp_pmc_count; i++) {
		pc[i].reg_num   = outp.pfp_pmcs[i].reg_num;
		pc[i].reg_value = outp.pfp_pmcs[i].reg_value;
	}

	/*
	 * figure out pmd mapping from output pmc
	 * PMD38 returned as used PMD by libpfm, will be reset
	 */
	for (i=0; i < outp.pfp_pmd_count; i++)
		pd[i].reg_num   = outp.pfp_pmds[i].reg_num;

	/*
	 * indicate we want notification when buffer is full and randomization
	 */
	pc[0].reg_flags |= PFM_REGFL_OVFL_NOTIFY | PFM_REGFL_RANDOM;

	/*
	 * Now prepare the argument to initialize the PMD and the sampling period
	 * We know we use only one PMD in this case, therefore pmd[0] corresponds
	 * to our first event which is our sampling period.
	 */
	pd[0].reg_value       = - SMPL_PERIOD;
	pd[0].reg_long_reset  = - SMPL_PERIOD;
	pd[0].reg_short_reset = - SMPL_PERIOD;

	/*
	 * indicate PMD to collect in each sample (good up to PMD63)
	 */
	pc[0].reg_smpl_pmds[0] = ETB_REGS_MASK;

	/*
	 * compute size of each sample: fixed-size header + all our BTB regs
	 */
	entry_size = sizeof(etb_entry_t)+(hweight64(ETB_REGS_MASK)<<3);

	/*
	 * When our counter overflows, we want to ETB index to be reset, so that we keep
	 * in sync. 
	 */
	pc[0].reg_reset_pmds[0] = M_PMD(38);

	/*
	 * Now program the registers
	 */
	if (perfmonctl(id, PFM_WRITE_PMCS, pc, outp.pfp_pmc_count))
		fatal_error("pfm_write_pmcs error errno %d\n",errno);

	/*
	 * we use 2 registers = 1 for the branch_event + 1 to reset PMD38
	 */
	if (perfmonctl(id, PFM_WRITE_PMDS, pd, outp.pfp_pmd_count))
		fatal_error("pfm_write_pmds error errno %d\n",errno);

	/*
	 * now we load (i.e., attach) the context to ourself
	 */
	load_args.load_pid = getpid();
	if (perfmonctl(id, PFM_LOAD_CONTEXT, &load_args, 1))
		fatal_error("pfm_load_context error errno %d\n",errno);

	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret = fcntl(id, F_SETFL, fcntl(id, F_GETFL, 0) | O_ASYNC);
	if (ret == -1)
		fatal_error("cannot set ASYNC: %s\n", strerror(errno));

	/*
	 * get ownership of the descriptor
	 */
	ret = fcntl(id, F_SETOWN, getpid());
	if (ret == -1)
		fatal_error("cannot setown: %s\n", strerror(errno));

	/*
	 * Let's roll now.
	 */
	pfm_self_start(id);

	do_test(1000);

	pfm_self_stop(id);

	/*
	 * We must call the processing routine to cover the last entries recorded
	 * in the sampling buffer. Note that the buffer may not be full at this point.
	 *
	 */
	process_smpl_buffer();
	/*
	 * let's stop this now
	 */
	close(id);

	return 0;
}
