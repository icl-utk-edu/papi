/* $Id: arm.c,v 1.1.2.3 2009/06/11 12:33:51 mikpe Exp $
 * ARM/XScale performance-monitoring counters driver.
 *
 * Copyright (C) 2005-2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/perfctr.h>
#include <asm/cputype.h>

#include "compat.h"

/* Support for lazy evntsel and perfctr register updates. */
struct per_cpu_cache {	/* roughly a subset of perfctr_cpu_state */
	union {
		unsigned int id;	/* cache owner id */
	} k1;
	union {
		struct {
			unsigned int pmnc;
		} xsc1;
		struct {
			unsigned int evtsel;
			unsigned int inten;
			unsigned int pmnc;
		} xsc2;
	} arm;
} ____cacheline_aligned;
static struct per_cpu_cache per_cpu_cache[NR_CPUS] __cacheline_aligned;
#define __get_cpu_cache(cpu) (&per_cpu_cache[cpu])
#define get_cpu_cache()	(&per_cpu_cache[smp_processor_id()])

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[4];
};

#define PMU_XSC1	1
#define PMU_XSC2	2
static unsigned int pmu_type;

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

#ifndef CONFIG_PERFCTR_INTERRUPT_SUPPORT
#define perfctr_cstatus_has_ictrs(cstatus)	0
#endif

#if defined(CONFIG_SMP) && defined(CONFIG_PERFCTR_INTERRUPT_SUPPORT)

static inline void
set_isuspend_cpu(struct perfctr_cpu_state *state, int cpu)
{
	state->k1.isuspend_cpu = cpu;
}

static inline int
is_isuspend_cpu(const struct perfctr_cpu_state *state, int cpu)
{
	return state->k1.isuspend_cpu == cpu;
}

static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state)
{
	state->k1.isuspend_cpu = NR_CPUS;
}

#else
static inline void set_isuspend_cpu(struct perfctr_cpu_state *state, int cpu) { }
static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state, int cpu) { return 1; }
static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state) { }
#endif

/****************************************************************
 *								*
 * Driver procedures.						*
 *								*
 ****************************************************************/

/*
 * XScale1 driver procedures.
 */

static u32 xsc1_read_pmnc(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c0, c0, 0" : "=r"(val));
	/* bits 1, 2, 7, 11, 28-31 are read-unpredictable */
	val &= 0x0ffff779;
	return val;
}

static void xsc1_write_pmnc(u32 val)
{
	/* bits 7, 11, 28-31 are write-as-0 */
	val &= 0x0ffff77f;
	__asm__ __volatile__("mcr p14, 0, %0, c0, c0, 0" : : "r"(val));
}

static u32 xsc1_read_ccnt(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c1, c0, 0" : "=r"(val));
	return val;
}

#if 0
static void xsc1_write_ccnt(u32 val)
{
	__asm__ __volatile__("mcr p14, 0, %0, c1, c0, 0" : : "r"(val));
}
#endif

static u32 xsc1_read_pmc(u32 counter)
{
	u32 val;

	switch (counter) {
	default: /* impossible, but silences gcc warning */
	case 0:
		__asm__ __volatile__("mrc p14, 0, %0, c2, c0, 0" : "=r"(val));
		break;
	case 1:
		__asm__ __volatile__("mrc p14, 0, %0, c3, c0, 0" : "=r"(val));
		break;
	}
	return val;
}

#if 0
static void xsc1_write_pmc(u32 counter, u32 val)
{
	switch (counter) {
	case 0:
		__asm__ __volatile__("mcr p14, 0, %0, c2, c0, 0" : : "r"(val));
		break;
	case 1:
		__asm__ __volatile__("mcr p14, 0, %0, c3, c0, 0" : : "r"(val));
		break;
	}
}
#endif

static void xsc1_clear_counters(void)
{
	u32 pmnc;

	pmnc = xsc1_read_pmnc();
	/* preserve CCNT settings */
	pmnc &= (1 << 10) | (1 << 6) | (1 << 3) | (1 << 0);
	/* update non-CCNT settings: set event selectors to idle, and
	   reset the performance counters and their overflow flags. */
	pmnc |= (0xFF << 20) | (0xFF << 12) | (0x3 << 8) | (1 << 1);
	xsc1_write_pmnc(pmnc);
}

static unsigned int xsc1_nr_pmcs(void)
{
	return 2;
}

/*
 * XScale2 driver procedures.
 */

static u32 xsc2_read_pmnc(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c0, c1, 0" : "=r"(val));
	/* bits 1, 2, 4-23 are read-unpredictable */
	val &= 0xff000009;
	return val;
}

static void xsc2_write_pmnc(u32 val)
{
	/* bits 4-23 are write-as-0, 24-31 are write ignored */
	val &= 0x0000000f;
	__asm__ __volatile__("mcr p14, 0, %0, c0, c1, 0" : : "r"(val));
}

static u32 xsc2_read_ccnt(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c1, c1, 0" : "=r"(val));
	return val;
}

#if 0
static void xsc2_write_ccnt(u32 val)
{
	__asm__ __volatile__("mcr p14, 0, %0, c1, c1, 0" : : "r"(val));
}
#endif

static u32 xsc2_read_inten(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c4, c1, 0" : "=r"(val));
	/* bits 5-31 are read-unpredictable */
	val &= 0x0000001f;
	return val;
}

static void xsc2_write_inten(u32 val)
{
	/* bits 5-31 are write-as-zero */
	val &= 0x0000001f;
	__asm__ __volatile__("mcr p14, 0, %0, c4, c1, 0" : : "r"(val));
}

#if 0
static u32 xsc2_read_flag(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c5, c1, 0" : "=r"(val));
	/* bits 5-31 are read-unpredictable */
	val &= 0x0000001f;
	return val;
}
#endif

static void xsc2_write_flag(u32 val)
{
	/* bits 5-31 are write-as-zero */
	val &= 0x0000001f;
	__asm__ __volatile__("mcr p14, 0, %0, c5, c1, 0" : : "r"(val));
}

#if 0
static u32 xsc2_read_evtsel(void)
{
	u32 val;

	__asm__ __volatile__("mrc p14, 0, %0, c8, c1, 0" : "=r"(val));
	return val;
}
#endif

static void xsc2_write_evtsel(u32 val)
{
	__asm__ __volatile__("mcr p14, 0, %0, c8, c1, 0" : : "r"(val));
}

static u32 xsc2_read_pmc(u32 counter)
{
	u32 val;

	switch (counter) {
	default: /* impossible, but silences gcc warning */
	case 0:
		__asm__ __volatile__("mrc p14, 0, %0, c0, c2, 0" : "=r"(val));
		break;
	case 1:
		__asm__ __volatile__("mrc p14, 0, %0, c1, c2, 0" : "=r"(val));
		break;
	case 2:
		__asm__ __volatile__("mrc p14, 0, %0, c2, c2, 0" : "=r"(val));
		break;
	case 3:
		__asm__ __volatile__("mrc p14, 0, %0, c3, c2, 0" : "=r"(val));
		break;
	}
	return val;
}

#if 0
static void xsc2_write_pmc(u32 counter, u32 val)
{
	switch (counter) {
	case 0:
		__asm__ __volatile__("mcr p14, 0, %0, c0, c2, 0" : : "r"(val));
		break;
	case 1:
		__asm__ __volatile__("mcr p14, 0, %0, c1, c2, 0" : : "r"(val));
		break;
	case 2:
		__asm__ __volatile__("mcr p14, 0, %0, c2, c2, 0" : : "r"(val));
		break;
	case 3:
		__asm__ __volatile__("mcr p14, 0, %0, c3, c2, 0" : : "r"(val));
		break;
	}
}
#endif

static void xsc2_clear_counters(void)
{
	u32 val;

	/* clear interrupt enable bits */
	val = xsc2_read_inten();
	val &= (1 << 0);		/* all but CCNT */
	xsc2_write_inten(val);

	/* set event selectors to idle */
	xsc2_write_evtsel(0xFFFFFFFF);

	/* reset the performance counters */
	val = xsc2_read_pmnc();
	val &= (1 << 3) | (1 << 0);	/* preserve CCNT settings */
	val |= (1 << 1);		/* reset the performance counters */
	xsc2_write_pmnc(val);

	/* clear overflow status bits */
	xsc2_write_flag(0x1E);		/* all but CCNT */
}

static unsigned int xsc2_nr_pmcs(void)
{
	return 4;
}

/*
 * XScale driver procedures.
 */

static u32 xscale_read_ccnt(void)
{
	if (pmu_type == PMU_XSC1)
		return xsc1_read_ccnt();
	else
		return xsc2_read_ccnt();
}

static u32 xscale_read_pmc(u32 counter)
{
	if (pmu_type == PMU_XSC1)
		return xsc1_read_pmc(counter);
	else
		return xsc2_read_pmc(counter);
}

static void xscale_clear_counters(void)
{
	if (pmu_type == PMU_XSC1)
		xsc1_clear_counters();
	else
		xsc2_clear_counters();
}

static unsigned int xscale_nr_pmcs(void)
{
	if (pmu_type == PMU_XSC1)
		return xsc1_nr_pmcs();
	else
		return xsc2_nr_pmcs();
}

static unsigned int xscale_check_event(unsigned int evntsel)
{
	/* Events 0x00-0x0D are defined for the XScale core
	   and the IXP42x family.
	   Event 0xFF is defined as an "idle" event, but users
	   have no reason to specify it, so we reject it.
	   Events 0x10-0x16 are defined in oprofile, but not
	   in the XScale core or IXP42x manuals. */
	return (evntsel <= 0x0D) ? 0 : -1;
}

static void xscale_read_counters(struct perfctr_cpu_state *state,
				 struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		ctrs->tsc = xscale_read_ccnt();
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		ctrs->pmc[i] = xscale_read_pmc(pmc);
	}
}

static int xscale_check_control(struct perfctr_cpu_state *state)
{
	unsigned int i, nractrs, nrctrs, pmc_mask, pmi_mask, pmc;
	unsigned int nr_pmcs, evntsel[4];

	nr_pmcs = xscale_nr_pmcs();
	nractrs = state->control.nractrs;
	nrctrs = nractrs + state->control.nrictrs;
	if (nrctrs < nractrs || nrctrs > nr_pmcs)
		return -EINVAL;

	pmc_mask = 0;
	pmi_mask = 0;
	memset(evntsel, 0, sizeof evntsel);
	for(i = 0; i < nrctrs; ++i) {
		pmc = state->control.pmc_map[i];
		state->pmc[i].map = pmc;
		if (pmc >= nr_pmcs || (pmc_mask & (1<<pmc)))
			return -EINVAL;
		pmc_mask |= (1<<pmc);

		if (i >= nractrs)
			pmi_mask |= (1<<pmc);

		evntsel[pmc] = state->control.evntsel[i];
		if (xscale_check_event(evntsel[pmc]))
			return -EINVAL;
	}

	switch (pmu_type) {
	case PMU_XSC1:
		state->arm.xsc1.pmnc =
			((evntsel[1] << 20) |
			 (evntsel[0] << 12) |
			 (pmi_mask << 4) |	/* inten field */
			 1);			/* enable */
		break;
	case PMU_XSC2:
		state->arm.xsc2.evtsel =
			((evntsel[3] << 24) |
			 (evntsel[2] << 16) |
			 (evntsel[1] << 8) |
			 evntsel[0]);
		state->arm.xsc2.inten = pmi_mask << 1;
		break;
	}

	state->k1.id = new_id();

	return 0;
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
/* PRE: counters frozen *//* XXX: that is FALSE on XScale! */
static void xscale_isuspend(struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	set_isuspend_cpu(state, cpu); /* early to limit cpu's live range */
	cache = __get_cpu_cache(cpu);
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		unsigned int now = read_pmc(pmc);
		state->pmc[i].sum += now - state->pmc[i].start;
		state->pmc[i].start = now;
	}
	/* cache->k1.id is still == state->k1.id */
}

static void xscale_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;
	unsigned int pmc[4];

	cpu = smp_processor_id();
	cache = __get_cpu_cache(cpu);
	if (cache->k1.id == state->k1.id) {
		cache->k1.id = 0; /* force reload of cleared EVNTSELs */
		if (is_isuspend_cpu(state, cpu))
			return; /* skip reload of PMCs */
	}
	/* XXX: the rest is still the PPC code */
	/*
	 * The CPU state wasn't ours.
	 *
	 * The counters must be frozen before being reinitialised,
	 * to prevent unexpected increments and missed overflows.
	 *
	 * All unused counters must be reset to a non-overflow state.
	 */
	if (!(cache->ppc_mmcr[0] & MMCR0_FC)) {
		cache->ppc_mmcr[0] |= MMCR0_FC;
		mtspr(SPRN_MMCR0, cache->ppc_mmcr[0]);
	}
	memset(&pmc[0], 0, sizeof pmc);
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i)
		pmc[state->pmc[i].map] = state->pmc[i].start;

	switch (pm_type) {
	case PM_7450:
		mtspr(SPRN_PMC6, pmc[6-1]);
		mtspr(SPRN_PMC5, pmc[5-1]);
	case PM_7400:
	case PM_750:
	case PM_604e:
		mtspr(SPRN_PMC4, pmc[4-1]);
		mtspr(SPRN_PMC3, pmc[3-1]);
	case PM_604:
		mtspr(SPRN_PMC2, pmc[2-1]);
		mtspr(SPRN_PMC1, pmc[1-1]);
	case PM_NONE:
		;
	}
	/* cache->k1.id remains != state->k1.id */
}
#endif

static void xscale_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int value;

	cache = get_cpu_cache();
	if (cache->k1.id == state->k1.id)
		return;
	switch (pmu_type) {
	case PMU_XSC1:
		value = state->arm.xsc1.pmnc;
		if (value != cache->arm.xsc1.pmnc) {
			cache->arm.xsc1.pmnc = value;
			xsc1_write_pmnc(value);
		}
		break;
	case PMU_XSC2:
		value = cache->arm.xsc2.pmnc;
		if (value & 1) {
			value &= ~1;
			cache->arm.xsc2.pmnc = value;
			xsc2_write_pmnc(value);
		}
		value = state->arm.xsc2.evtsel;
		if (value != cache->arm.xsc2.evtsel) {
			cache->arm.xsc2.evtsel = value;
			xsc2_write_evtsel(value);
		}
		value = state->arm.xsc2.inten;
		if (value != cache->arm.xsc2.inten) {
			cache->arm.xsc2.inten = value;
			xsc2_write_inten(value);
		}
		value = cache->arm.xsc2.pmnc;
		value |= 1;
		cache->arm.xsc2.pmnc = value;
		xsc2_write_pmnc(value);
	}
	cache->k1.id = state->k1.id;
}

/*
 * Driver methods, internal and exported.
 */

static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	return xscale_write_control(state);
}

static void perfctr_cpu_read_counters(struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	return xscale_read_counters(state, ctrs);
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	return xscale_isuspend(state);
}

static void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	return xscale_iresume(state);
}

/* Call perfctr_cpu_ireload() just before perfctr_cpu_resume() to
   bypass internal caching and force a reload if the I-mode PMCs. */
void perfctr_cpu_ireload(struct perfctr_cpu_state *state)
{
	//state->ppc_mmcr[0] |= MMCR0_PMXE;
#ifdef CONFIG_SMP
	clear_isuspend_cpu(state);
#else
	get_cpu_cache()->k1.id = 0;
#endif
}

/* PRE: the counters have been suspended and sampled by perfctr_cpu_suspend() */
unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state *state)
{
	/* XXX: XScale has an overflow status register */
	/* XXX: XSC1 A stepping has an erratum making the
	   overflow status bits unreliable */
	/* XXX: use different procedures for XSC1 and XSC2 */

	/* XXX: the rest is still the X86 code */
	unsigned int cstatus, nrctrs, pmc, pmc_mask;

	cstatus = state->cstatus;
	pmc = perfctr_cstatus_nractrs(cstatus);
	nrctrs = perfctr_cstatus_nrctrs(cstatus);

	for(pmc_mask = 0; pmc < nrctrs; ++pmc) {
		if ((int)state->pmc[pmc].start >= 0) { /* XXX: ">" ? */
			/* XXX: "+=" to correct for overshots */
			state->pmc[pmc].start = state->control.ireset[pmc];
			pmc_mask |= (1 << pmc);
		}
	}
	return pmc_mask;
}

static inline int check_ireset(const struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i;

	i = state->control.nractrs;
	nrctrs = i + state->control.nrictrs;
	for(; i < nrctrs; ++i)
		if (state->control.ireset[i] >= 0)	/* XScale-specific */
			return -EINVAL;
	return 0;
}

static inline void setup_imode_start_values(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i)
		state->pmc[i].start = state->control.ireset[i];
}

#else	/* CONFIG_PERFCTR_INTERRUPT_SUPPORT */
static inline void perfctr_cpu_isuspend(struct perfctr_cpu_state *state) { }
static inline void perfctr_cpu_iresume(const struct perfctr_cpu_state *state) { }
static inline int check_ireset(const struct perfctr_cpu_state *state) { return 0; }
static inline void setup_imode_start_values(struct perfctr_cpu_state *state) { }
#endif	/* CONFIG_PERFCTR_INTERRUPT_SUPPORT */

static int check_control(struct perfctr_cpu_state *state)
{
	return xscale_check_control(state);
}

int perfctr_cpu_update_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	int err;

	clear_isuspend_cpu(state);
	state->cstatus = 0;

	/* disallow i-mode counters if we cannot catch the interrupts */
	if (!(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
	    && state->control.nrictrs)
		return -EPERM;

	err = check_ireset(state);
	if (err < 0)
		return err;
	err = check_control(state); /* may initialise state->cstatus */
	if (err < 0)
		return err;
	state->cstatus |= perfctr_mk_cstatus(state->control.tsc_on,
					     state->control.nractrs,
					     state->control.nrictrs);
	setup_imode_start_values(state);
	return 0;
}

void perfctr_cpu_suspend(struct perfctr_cpu_state *state)
{
	unsigned int i, cstatus, nractrs;
	struct perfctr_low_ctrs now;

	if (perfctr_cstatus_has_ictrs(state->cstatus))
		perfctr_cpu_isuspend(state);
	perfctr_cpu_read_counters(state, &now);
	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		state->tsc_sum += now.tsc - state->tsc_start;
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i)
		state->pmc[i].sum += now.pmc[i] - state->pmc[i].start;
}

void perfctr_cpu_resume(struct perfctr_cpu_state *state)
{
	if (perfctr_cstatus_has_ictrs(state->cstatus))
	    perfctr_cpu_iresume(state);
	perfctr_cpu_write_control(state);
	//perfctr_cpu_read_counters(state, &state->start);
	{
		struct perfctr_low_ctrs now;
		unsigned int i, cstatus, nrctrs;
		perfctr_cpu_read_counters(state, &now);
		cstatus = state->cstatus;
		if (perfctr_cstatus_has_tsc(cstatus))
			state->tsc_start = now.tsc;
		nrctrs = perfctr_cstatus_nractrs(cstatus);
		for(i = 0; i < nrctrs; ++i)
			state->pmc[i].start = now.pmc[i];
	}
	/* XXX: if (SMP && start.tsc == now.tsc) ++now.tsc; */
}

void perfctr_cpu_sample(struct perfctr_cpu_state *state)
{
	unsigned int i, cstatus, nractrs;
	struct perfctr_low_ctrs now;

	perfctr_cpu_read_counters(state, &now);
	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus)) {
		state->tsc_sum += now.tsc - state->tsc_start;
		state->tsc_start = now.tsc;
	}
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i) {
		state->pmc[i].sum += now.pmc[i] - state->pmc[i].start;
		state->pmc[i].start = now.pmc[i];
	}
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

static int __init xscale_init(void)
{
	static char xsc1_name[] __initdata = "XScale1";
	static char xsc2_name[] __initdata = "XScale2";
	u32 id;

	//asm("mrc p15, 0, %0, c0, c0, 0" : "=r"(id));
	id = read_cpuid(CPUID_ID);

	/* check for Intel/V5TE */
	if ((id & 0xffff0000) != 0x69050000)
		return -ENODEV;
	/* check coregen for XSC1 or XSC2 */
	switch ((id >> 13) & 0x7) {
	case 0x1:
		pmu_type = PMU_XSC1;
		perfctr_info.cpu_type = PERFCTR_ARM_XSC1;
		perfctr_cpu_name = xsc1_name;
		break;
	case 0x2:
		pmu_type = PMU_XSC2;
		perfctr_info.cpu_type = PERFCTR_ARM_XSC2;
		perfctr_cpu_name = xsc2_name;
		break;
	default:
		return -ENODEV;
	}
	perfctr_info.cpu_features = 0;
	/* XXX: detect cpu_khz by sampling CCNT over mdelay()? */
	return 0;
}

static void perfctr_cpu_clear_counters(void)
{
	struct per_cpu_cache *cache;

	cache = get_cpu_cache();
	memset(cache, 0, sizeof *cache);
	cache->k1.id = -1;

	xscale_clear_counters();
}

static void perfctr_cpu_clear_one(void *ignore)
{
	/* PREEMPT note: when called via on_each_cpu(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
}

static void perfctr_cpu_reset(void)
{
	on_each_cpu(perfctr_cpu_clear_one, NULL, 1);
	perfctr_cpu_set_ihandler(NULL);
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static int reserve_pmu_irq(void)
{
	int ret;

	ret = request_irq(XSCALE_PMU_IRQ, xscale_pmu_interrupt, IRQF_DISABLED,
			  "XScale PMU", NULL);
	if (ret < 0)
		return ret;
	/* fiddle pmnc PMU_ENABLE, PMU_CNT64 */
	return 0;
}

static void release_pmu_irq(void)
{
	write_pmnc(read_pmnc() & ~PMU_ENABLE);
	free_irq(XSCALE_PMU_IRQ, NULL);
}

#else
static inline int reserve_pmu_irq(void) { return 0; }
static inline void release_pmu_irq(void) { }
#endif

static void do_init_tests(void)
{
#ifdef CONFIG_PERFCTR_INIT_TESTS
	if (reserve_pmu_irq() >= 0) {
		perfctr_xscale_init_tests();
		release_pmu_irq();
	}
#endif
}

int __init perfctr_cpu_init(void)
{
	int err;

	preempt_disable();

	err = xscale_init();
	if (err)
		goto out;

	do_init_tests();

	perfctr_info.cpu_khz = 266; /* XXX: perfctr_cpu_khz(); */
	perfctr_info.tsc_to_cpu_mult = 1;
 out:
	preempt_enable();
	return err;
}

void __exit perfctr_cpu_exit(void)
{
}

/****************************************************************
 *								*
 * Hardware reservation.					*
 *								*
 ****************************************************************/

static DEFINE_MUTEX(mutex);
static const char *current_service = 0;

const char *perfctr_cpu_reserve(const char *service)
{
	const char *ret;

	mutex_lock(&mutex);
	ret = current_service;
	if (ret)
		goto out_unlock;
	ret = "unknown driver (oprofile? ixp425_eth?)";
	if (reserve_pmu_irq() < 0)
		goto out_unlock;
	current_service = service;
	__module_get(THIS_MODULE);
	perfctr_cpu_reset();
	ret = NULL;
 out_unlock:
	mutex_unlock(&mutex);
	return ret;
}

void perfctr_cpu_release(const char *service)
{
	mutex_lock(&mutex);
	if (service != current_service) {
		printk(KERN_ERR "%s: attempt by %s to release while reserved by %s\n",
		       __FUNCTION__, service, current_service);
		goto out_unlock;
	}
	/* power down the counters */
	perfctr_cpu_reset();
	current_service = 0;
	release_pmu_irq();
	module_put(THIS_MODULE);
 out_unlock:
	mutex_unlock(&mutex);
}
