/*
 * PPC64 performance-monitoring counters driver.
 *
 * based on Mikael Pettersson's 32 bit ppc code
 * Copyright (C) 2004  David Gibson, IBM Corporation.
 * Copyright (C) 2004, 2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <asm/prom.h>
#include <asm/time.h>		/* tb_ticks_per_jiffy */
#include <asm/pmc.h>
#include <asm/cputable.h>

#include "ppc64_tests.h"

extern void ppc64_enable_pmcs(void);

/* Support for lazy perfctr SPR updates. */
struct per_cpu_cache {	/* roughly a subset of perfctr_cpu_state */
	unsigned int id;	/* cache owner id */
	/* Physically indexed cache of the MMCRs. */
	unsigned long ppc64_mmcr0, ppc64_mmcr1, ppc64_mmcra;
};
static DEFINE_PER_CPU(struct per_cpu_cache, per_cpu_cache);
#define __get_cpu_cache(cpu) (&per_cpu(per_cpu_cache, cpu))
#define get_cpu_cache()	(&__get_cpu_var(per_cpu_cache))

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	u64 tsc;
	u32 pmc[8];
};

static unsigned int new_id(void)
{
	static DEFINE_SPINLOCK(lock);
	static unsigned int counter;
	int id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

static inline u32 read_pmc(int pmc)
{
	switch (pmc) {
	case 0:
		return mfspr(SPRN_PMC1);
		break;
	case 1:
		return mfspr(SPRN_PMC2);
		break;
	case 2:
		return mfspr(SPRN_PMC3);
		break;
	case 3:
		return mfspr(SPRN_PMC4);
		break;
	case 4:
		return mfspr(SPRN_PMC5);
		break;
	case 5:
		return mfspr(SPRN_PMC6);
		break;
	case 6:
		return mfspr(SPRN_PMC7);
		break;
	case 7:
		return mfspr(SPRN_PMC8);
		break;

	default:
		return -EINVAL;
	}
}

static inline void write_pmc(int pmc, u32 val)
{
	switch (pmc) {
	case 0:
		mtspr(SPRN_PMC1, val);
		break;
	case 1:
		mtspr(SPRN_PMC2, val);
		break;
	case 2:
		mtspr(SPRN_PMC3, val);
		break;
	case 3:
		mtspr(SPRN_PMC4, val);
		break;
	case 4:
		mtspr(SPRN_PMC5, val);
		break;
	case 5:
		mtspr(SPRN_PMC6, val);
		break;
	case 6:
		mtspr(SPRN_PMC7, val);
		break;
	case 7:
		mtspr(SPRN_PMC8, val);
		break;
	}
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static void perfctr_default_ihandler(unsigned long pc)
{
	unsigned int mmcr0 = mfspr(SPRN_MMCR0);

	mmcr0 &= ~MMCR0_PMXE;
	mtspr(SPRN_MMCR0, mmcr0);
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;

void do_perfctr_interrupt(struct pt_regs *regs)
{
	unsigned long mmcr0;

	/* interrupts are disabled here, so we don't need to
	 * preempt_disable() */

	(*perfctr_ihandler)(instruction_pointer(regs));

	/* clear PMAO so the interrupt doesn't reassert immediately */
	mmcr0 = mfspr(SPRN_MMCR0) & ~MMCR0_PMAO;
	mtspr(SPRN_MMCR0, mmcr0);
}

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}

#else
#define perfctr_cstatus_has_ictrs(cstatus)	0
#endif


#if defined(CONFIG_SMP) && defined(CONFIG_PERFCTR_INTERRUPT_SUPPORT)

static inline void
set_isuspend_cpu(struct perfctr_cpu_state *state, int cpu)
{
	state->isuspend_cpu = cpu;
}

static inline int
is_isuspend_cpu(const struct perfctr_cpu_state *state, int cpu)
{
	return state->isuspend_cpu == cpu;
}

static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state)
{
	state->isuspend_cpu = NR_CPUS;
}

#else
static inline void set_isuspend_cpu(struct perfctr_cpu_state *state, int cpu) { }
static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state, int cpu) { return 1; }
static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state) { }
#endif


static void ppc64_clear_counters(void)
{
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_MMCR1, 0);
	mtspr(SPRN_MMCRA, 0);

	if (cur_cpu_spec->num_pmcs >= 1)
		mtspr(SPRN_PMC1, 0);
	if (cur_cpu_spec->num_pmcs >= 2)
		mtspr(SPRN_PMC2, 0);
	if (cur_cpu_spec->num_pmcs >= 3)
		mtspr(SPRN_PMC3, 0);
	if (cur_cpu_spec->num_pmcs >= 4)
		mtspr(SPRN_PMC4, 0);
	if (cur_cpu_spec->num_pmcs >= 5)
		mtspr(SPRN_PMC5, 0);
	if (cur_cpu_spec->num_pmcs >= 6)
		mtspr(SPRN_PMC6, 0);
	if (cur_cpu_spec->num_pmcs >= 7)
		mtspr(SPRN_PMC7, 0);
	if (cur_cpu_spec->num_pmcs >= 8)
		mtspr(SPRN_PMC8, 0);
}

/*
 * Driver methods, internal and exported.
 */

static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned long long value;

	cache = get_cpu_cache();
	/*
	 * Order matters here: update threshmult and event
	 * selectors before updating global control, which
	 * potentially enables PMIs.
	 *
	 * Since mtspr doesn't accept a runtime value for the
	 * SPR number, unroll the loop so each mtspr targets
	 * a constant SPR.
	 *
	 * For processors without MMCR2, we ensure that the
	 * cache and the state indicate the same value for it,
	 * preventing any actual mtspr to it. Ditto for MMCR1.
	 */
	value = state->control.mmcra;
	if (value != cache->ppc64_mmcra) {
		cache->ppc64_mmcra = value;
		mtspr(SPRN_MMCRA, value);
	}
	value = state->control.mmcr1;
	if (value != cache->ppc64_mmcr1) {
		cache->ppc64_mmcr1 = value;
		mtspr(SPRN_MMCR1, value);
	}
	value = state->control.mmcr0;
	if (perfctr_cstatus_has_ictrs(state->user.cstatus))
	    value |= MMCR0_PMXE;
	if (value != cache->ppc64_mmcr0) {
		cache->ppc64_mmcr0 = value;
		mtspr(SPRN_MMCR0, value);
	}
	cache->id = state->id;
}

static void perfctr_cpu_read_counters(struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, i, pmc;

	cstatus = state->user.cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		ctrs->tsc = mftb();

	for (i = 0; i < perfctr_cstatus_nractrs(cstatus); ++i) {
		pmc = state->control.pmc_map[i];
		ctrs->pmc[i] = read_pmc(pmc);
	}
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	set_isuspend_cpu(state, cpu); /* early to limit cpu's live range */
	cstatus = state->user.cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for (i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		int pmc = state->control.pmc_map[i];
		u32 now = read_pmc(pmc);

		state->user.pmc[i].sum += (u32)(now-state->user.pmc[i].start);
		state->user.pmc[i].start = now;
	}
}

static void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	cache = __get_cpu_cache(cpu);
	if (cache->id == state->id) {
		/* Clearing cache->id to force write_control()
		   to unfreeze MMCR0 would be done here, but it
		   is subsumed by resume()'s MMCR0 reload logic. */
		if (is_isuspend_cpu(state, cpu)) {
			return; /* skip reload of PMCs */
		}
	}
	/*
	 * The CPU state wasn't ours.
	 *
	 * The counters must be frozen before being reinitialised,
	 * to prevent unexpected increments and missed overflows.
	 *
	 * All unused counters must be reset to a non-overflow state.
	 */
	if (!(cache->ppc64_mmcr0 & MMCR0_FC)) {
		cache->ppc64_mmcr0 |= MMCR0_FC;
		mtspr(SPRN_MMCR0, cache->ppc64_mmcr0);
	}
	cstatus = state->user.cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for (i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		write_pmc(state->control.pmc_map[i], state->user.pmc[i].start);
	}
}

/* Call perfctr_cpu_ireload() just before perfctr_cpu_resume() to
   bypass internal caching and force a reload if the I-mode PMCs. */
void perfctr_cpu_ireload(struct perfctr_cpu_state *state)
{
#ifdef CONFIG_SMP
	clear_isuspend_cpu(state);
#else
	get_cpu_cache()->id = 0;
#endif
}

/* PRE: the counters have been suspended and sampled by perfctr_cpu_suspend() */
unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nractrs, nrctrs, i;
	unsigned int pmc_mask = 0;
	int nr_pmcs = cur_cpu_spec->num_pmcs;

	cstatus = state->user.cstatus;
	nractrs = perfctr_cstatus_nractrs(cstatus);
	nrctrs = perfctr_cstatus_nrctrs(cstatus);

	/* Ickity, ickity, ick.  We don't have fine enough interrupt
	 * control to disable interrupts on all the counters we're not
	 * interested in.  So, we have to deal with overflows on actrs
	 * amd unused PMCs as well as the ones we actually care
	 * about. */
	for (i = 0; i < nractrs; ++i) {
		int pmc = state->control.pmc_map[i];
		u32 val = read_pmc(pmc);

		/* For actrs, force a sample if they overflowed */

		if ((s32)val < 0) {
			state->user.pmc[i].sum += (u32)(val - state->user.pmc[i].start);
			state->user.pmc[i].start = 0;
			write_pmc(pmc, 0);
		}
	}
	for (; i < nrctrs; ++i) {
		if ((s32)state->user.pmc[i].start < 0) { /* PPC64-specific */
			int pmc = state->control.pmc_map[i];
			/* XXX: "+=" to correct for overshots */
			state->user.pmc[i].start = state->control.ireset[pmc];
			pmc_mask |= (1 << i);
		}
	}

	/* Clear any unused overflowed counters, so we don't loop on
	 * the interrupt */
	for (i = 0; i < nr_pmcs; ++i) {
		if (! (state->unused_pmcs & (1<<i)))
			continue;

		if ((int)read_pmc(i) < 0)
			write_pmc(i, 0);
	}

	/* XXX: HW cleared MMCR0[ENINT]. We presumably cleared the entire
	   MMCR0, so the re-enable occurs automatically later, no? */
	return pmc_mask;
}

static inline int check_ireset(struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i;

	i = state->control.header.nractrs;
	nrctrs = i + state->control.header.nrictrs;
	for(; i < nrctrs; ++i) {
		unsigned int pmc = state->control.pmc_map[i];
		if ((int)state->control.ireset[pmc] < 0) /* PPC64-specific */
			return -EINVAL;
		state->user.pmc[i].start = state->control.ireset[pmc];
	}
	return 0;
}

#else	/* CONFIG_PERFCTR_INTERRUPT_SUPPORT */
static inline void perfctr_cpu_isuspend(struct perfctr_cpu_state *state) { }
static inline void perfctr_cpu_iresume(const struct perfctr_cpu_state *state) { }
static inline int check_ireset(struct perfctr_cpu_state *state) { return 0; }
#endif	/* CONFIG_PERFCTR_INTERRUPT_SUPPORT */

static int check_control(struct perfctr_cpu_state *state)
{
	unsigned int i, nractrs, nrctrs, pmc_mask, pmc;
	unsigned int nr_pmcs = cur_cpu_spec->num_pmcs;

	nractrs = state->control.header.nractrs;
	nrctrs = nractrs + state->control.header.nrictrs;
	if (nrctrs < nractrs || nrctrs > nr_pmcs)
		return -EINVAL;

	pmc_mask = 0;
	for (i = 0; i < nrctrs; ++i) {
		pmc = state->control.pmc_map[i];
		if (pmc >= nr_pmcs || (pmc_mask & (1<<pmc)))
			return -EINVAL;
		pmc_mask |= (1<<pmc);
	}

	/* We need to retain internal control of PMXE and PMAO.  PMXE
	 * will be set when ictrs are active.  We can't really handle
	 * TB interrupts, so we don't allow those either. */
	if ( (state->control.mmcr0 & MMCR0_PMXE)
	     || (state->control.mmcr0 & MMCR0_PMAO)
	     || (state->control.mmcr0 & MMCR0_TBEE) )
		return -EINVAL;

	state->unused_pmcs = ((1 << nr_pmcs)-1) & ~pmc_mask;

	state->id = new_id();

	return 0;
}

int perfctr_cpu_update_control(struct perfctr_cpu_state *state, int is_global)
{
	int err;

	clear_isuspend_cpu(state);
	state->user.cstatus = 0;

	/* disallow i-mode counters if we cannot catch the interrupts */
	if (!(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
	    && state->control.header.nrictrs)
		return -EPERM;

	err = check_control(state); /* may initialise state->cstatus */
	if (err < 0)
		return err;
	err = check_ireset(state);
	if (err < 0)
		return err;
	state->user.cstatus |= perfctr_mk_cstatus(state->control.header.tsc_on,
						  state->control.header.nractrs,
						  state->control.header.nrictrs);
	return 0;
}

/*
 * get_reg_offset() maps SPR numbers to offsets into struct perfctr_cpu_control.
 */
static const struct {
	unsigned int spr;
	unsigned int offset;
	unsigned int size;
} reg_offsets[] = {
	{ SPRN_MMCR0, offsetof(struct perfctr_cpu_control, mmcr0), sizeof(long) },
	{ SPRN_MMCR1, offsetof(struct perfctr_cpu_control, mmcr1), sizeof(long) },
	{ SPRN_MMCRA, offsetof(struct perfctr_cpu_control, mmcra), sizeof(long) },
	{ SPRN_PMC1,  offsetof(struct perfctr_cpu_control, ireset[1-1]), sizeof(int) },
	{ SPRN_PMC2,  offsetof(struct perfctr_cpu_control, ireset[2-1]), sizeof(int) },
	{ SPRN_PMC3,  offsetof(struct perfctr_cpu_control, ireset[3-1]), sizeof(int) },
	{ SPRN_PMC4,  offsetof(struct perfctr_cpu_control, ireset[4-1]), sizeof(int) },
	{ SPRN_PMC5,  offsetof(struct perfctr_cpu_control, ireset[5-1]), sizeof(int) },
	{ SPRN_PMC6,  offsetof(struct perfctr_cpu_control, ireset[6-1]), sizeof(int) },
	{ SPRN_PMC7,  offsetof(struct perfctr_cpu_control, ireset[7-1]), sizeof(int) },
	{ SPRN_PMC8,  offsetof(struct perfctr_cpu_control, ireset[8-1]), sizeof(int) },
};

static int get_reg_offset(unsigned int spr, unsigned int *size)
{
	unsigned int i;

	for(i = 0; i < ARRAY_SIZE(reg_offsets); ++i)
		if (spr == reg_offsets[i].spr) {
			*size = reg_offsets[i].size;
			return reg_offsets[i].offset;
		}
	return -1;
}

static int access_regs(struct perfctr_cpu_control *control,
		       void *argp, unsigned int argbytes, int do_write)
{
	struct perfctr_cpu_reg *regs;
	unsigned int i, nr_regs, size;
	int offset;

	nr_regs = argbytes / sizeof(struct perfctr_cpu_reg);
	if (nr_regs * sizeof(struct perfctr_cpu_reg) != argbytes)
		return -EINVAL;
	regs = (struct perfctr_cpu_reg*)argp;

	for(i = 0; i < nr_regs; ++i) {
		offset = get_reg_offset(regs[i].nr, &size);
		if (offset < 0)
			return -EINVAL;
		if (size == sizeof(long)) {
			unsigned long *where = (unsigned long*)((char*)control + offset);
			if (do_write)
				*where = regs[i].value;
			else
				regs[i].value = *where;
		} else {
			unsigned int *where = (unsigned int*)((char*)control + offset);
			if (do_write)
				*where = regs[i].value;
			else
				regs[i].value = *where;
		}
	}
	return argbytes;
}

int perfctr_cpu_control_write(struct perfctr_cpu_control *control, unsigned int domain,
			      const void *srcp, unsigned int srcbytes)
{
	if (domain != PERFCTR_DOMAIN_CPU_REGS)
		return -EINVAL;
	return access_regs(control, (void*)srcp, srcbytes, 1);
}

int perfctr_cpu_control_read(const struct perfctr_cpu_control *control, unsigned int domain,
			     void *dstp, unsigned int dstbytes)
{
	if (domain != PERFCTR_DOMAIN_CPU_REGS)
		return -EINVAL;
	return access_regs((struct perfctr_cpu_control*)control, dstp, dstbytes, 0);
}

void perfctr_cpu_suspend(struct perfctr_cpu_state *state)
{
	unsigned int i, cstatus;
	struct perfctr_low_ctrs now;

	write_perfseq_begin(&state->user.sequence);

	/* quiesce the counters */
	mtspr(SPRN_MMCR0, MMCR0_FC);
	get_cpu_cache()->ppc64_mmcr0 = MMCR0_FC;

	if (perfctr_cstatus_has_ictrs(state->user.cstatus))
		perfctr_cpu_isuspend(state);

	perfctr_cpu_read_counters(state, &now);
	cstatus = state->user.cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		state->user.tsc_sum += now.tsc - state->user.tsc_start;

	for (i = 0; i < perfctr_cstatus_nractrs(cstatus); ++i)
		state->user.pmc[i].sum += (u32)(now.pmc[i]-state->user.pmc[i].start);

	write_perfseq_end(&state->user.sequence);
}

void perfctr_cpu_resume(struct perfctr_cpu_state *state)
{
	struct perfctr_low_ctrs now;
	unsigned int i, cstatus;

	write_perfseq_begin(&state->user.sequence);
	if (perfctr_cstatus_has_ictrs(state->user.cstatus))
	    perfctr_cpu_iresume(state);
	perfctr_cpu_write_control(state);

	perfctr_cpu_read_counters(state, &now);
	cstatus = state->user.cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		state->user.tsc_start = now.tsc;

	for (i = 0; i < perfctr_cstatus_nractrs(cstatus); ++i)
		state->user.pmc[i].start = now.pmc[i];

	write_perfseq_end(&state->user.sequence);
}

void perfctr_cpu_sample(struct perfctr_cpu_state *state)
{
	unsigned int i, cstatus, nractrs;
	struct perfctr_low_ctrs now;

	write_perfseq_begin(&state->user.sequence);
	perfctr_cpu_read_counters(state, &now);
	cstatus = state->user.cstatus;
	if (perfctr_cstatus_has_tsc(cstatus)) {
		state->user.tsc_sum += now.tsc - state->user.tsc_start;
		state->user.tsc_start = now.tsc;
	}
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i) {
		state->user.pmc[i].sum += (u32)(now.pmc[i]-state->user.pmc[i].start);
		state->user.pmc[i].start = now.pmc[i];
	}
	write_perfseq_end(&state->user.sequence);
}

static void perfctr_cpu_clear_counters(void)
{
	struct per_cpu_cache *cache;

	cache = get_cpu_cache();
	memset(cache, 0, sizeof *cache);
	cache->id = 0;

	ppc64_clear_counters();
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

static void ppc64_cpu_setup(void)
{
	/* allow user to initialize these???? */

        unsigned long long mmcr0 = mfspr(SPRN_MMCR0);
        unsigned long long mmcra = mfspr(SPRN_MMCRA);


        ppc64_enable_pmcs();

        mmcr0 |= MMCR0_FC;
        mtspr(SPRN_MMCR0, mmcr0);

        mmcr0 |= MMCR0_FCM1|MMCR0_PMXE|MMCR0_FCECE;
        mmcr0 |= MMCR0_PMC1CE|MMCR0_PMCjCE;
        mtspr(SPRN_MMCR0, mmcr0);

        mmcra |= MMCRA_SAMPLE_ENABLE;
        mtspr(SPRN_MMCRA, mmcra);

	printk("setup on cpu %d, mmcr0 %lx\n", smp_processor_id(),
            mfspr(SPRN_MMCR0));
	printk("setup on cpu %d, mmcr1 %lx\n", smp_processor_id(),
            mfspr(SPRN_MMCR1));
	printk("setup on cpu %d, mmcra %lx\n", smp_processor_id(),
            mfspr(SPRN_MMCRA));

/*         mtmsrd(mfmsr() | MSR_PMM); */

	ppc64_clear_counters();

	mmcr0 = mfspr(SPRN_MMCR0);
        mmcr0 &= ~MMCR0_PMAO;
        mmcr0 &= ~MMCR0_FC;
        mtspr(SPRN_MMCR0, mmcr0);

        printk("start on cpu %d, mmcr0 %llx\n", smp_processor_id(), mmcr0);
}


static void perfctr_cpu_clear_one(void *ignore)
{
	/* PREEMPT note: when called via on_each_cpu(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
}

static void perfctr_cpu_reset(void)
{
	on_each_cpu(perfctr_cpu_clear_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
}

int __init perfctr_cpu_init(void)
{
	extern unsigned long ppc_proc_freq;
	extern unsigned long ppc_tb_freq;

	perfctr_info.cpu_features = PERFCTR_FEATURE_RDTSC
		| PERFCTR_FEATURE_RDPMC | PERFCTR_FEATURE_PCINT;

	perfctr_cpu_name = "PowerPC64";

	perfctr_info.cpu_khz = ppc_proc_freq / 1000;
	/* We need to round here rather than truncating, because in a
	 * few cases the raw ratio can end up being 7.9999 or
	 * suchlike */
	perfctr_info.tsc_to_cpu_mult =
		(ppc_proc_freq + ppc_tb_freq - 1) / ppc_tb_freq;

	on_each_cpu((void *)ppc64_cpu_setup, NULL, 0, 1);

	perfctr_ppc64_init_tests();

	perfctr_cpu_reset();
	return 0;
}

void __exit perfctr_cpu_exit(void)
{
	perfctr_cpu_reset();
}

/****************************************************************
 *								*
 * Hardware reservation.					*
 *								*
 ****************************************************************/

static spinlock_t service_mutex = SPIN_LOCK_UNLOCKED;
static const char *current_service = NULL;

const char *perfctr_cpu_reserve(const char *service)
{
	const char *ret;

	spin_lock(&service_mutex);

	ret = current_service;
	if (ret)
		goto out;

	ret = "unknown driver (oprofile?)";
	if (reserve_pmc_hardware(do_perfctr_interrupt) != 0)
		goto out;

	current_service = service;
	ret = NULL;

 out:
	spin_unlock(&service_mutex);
	return ret;
}

void perfctr_cpu_release(const char *service)
{
	spin_lock(&service_mutex);

	if (service != current_service) {
		printk(KERN_ERR "%s: attempt by %s to release while reserved by %s\n",
		       __FUNCTION__, service, current_service);
		goto out;
	}

	/* power down the counters */
	perfctr_cpu_reset();
	current_service = NULL;
	release_pmc_hardware();

 out:
	spin_unlock(&service_mutex);
}
