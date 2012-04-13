/* $Id: ppc.c,v 1.3.2.22 2009/06/11 12:33:51 mikpe Exp $
 * PPC32 performance-monitoring counters driver.
 *
 * Copyright (C) 2004-2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <asm/prom.h>
#include <asm/time.h>		/* tb_ticks_per_jiffy, get_tbl() */

#include "compat.h"
#include "ppc_compat.h"
#include "ppc_tests.h"

/* Support for lazy evntsel and perfctr SPR updates. */
struct per_cpu_cache {	/* roughly a subset of perfctr_cpu_state */
	union {
		unsigned int id;	/* cache owner id */
	} k1;
	/* Physically indexed cache of the MMCRs. */
	unsigned int ppc_mmcr[3];
} ____cacheline_aligned;
static struct per_cpu_cache per_cpu_cache[NR_CPUS] __cacheline_aligned;
#define __get_cpu_cache(cpu) (&per_cpu_cache[cpu])
#define get_cpu_cache()	(&per_cpu_cache[smp_processor_id()])

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[6];
};

enum pm_type {
	PM_NONE,
	PM_604,
	PM_604e,
	PM_750,	/* XXX: Minor event set diffs between IBM and Moto. */
	PM_7400,
	PM_7450,
};
static enum pm_type pm_type;

/* Bits users shouldn't set in control.ppc.mmcr0:
 * - PMC1SEL/PMC2SEL because event selectors are in control.evntsel[]
 */
#define MMCR0_RESERVED		(MMCR0_PMC1SEL | MMCR0_PMC2SEL)

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

/* The ppc driver internally uses cstatus & (1<<30) to record that
   a context has an asynchronously changing MMCR0. */
static inline unsigned int perfctr_cstatus_set_mmcr0_quirk(unsigned int cstatus)
{
	return cstatus | (1 << 30);
}

static inline int perfctr_cstatus_has_mmcr0_quirk(unsigned int cstatus)
{
	return cstatus & (1 << 30);
}

/****************************************************************
 *								*
 * Driver procedures.						*
 *								*
 ****************************************************************/

/*
 * The PowerPC 604/750/74xx family.
 *
 * Common features
 * ---------------
 * - Per counter event selection data in subfields of control registers.
 *   MMCR0 contains both global control and PMC1/PMC2 event selectors.
 * - Overflow interrupt support is present in all processors, but an
 *   erratum makes it difficult to use in 750/7400/7410 processors.
 * - There is no concept of per-counter qualifiers:
 *   - User-mode/supervisor-mode restrictions are global.
 *   - Two groups of counters, PMC1 and PMC2-PMC<highest>. Each group
 *     has a single overflow interrupt/event enable/disable flag.
 * - The instructions used to read (mfspr) and write (mtspr) the control
 *   and counter registers (SPRs) only support hardcoded register numbers.
 *   There is no support for accessing an SPR via a runtime value.
 * - Each counter supports its own unique set of events. However, events
 *   0-1 are common for PMC1-PMC4, and events 2-4 are common for PMC1-PMC4.
 * - There is no separate high-resolution core clock counter.
 *   The time-base counter is available, but it typically runs an order of
 *   magnitude slower than the core clock.
 *   Any performance counter can be programmed to count core clocks, but
 *   doing this (a) reserves one PMC, and (b) needs indirect accesses
 *   since the SPR number in general isn't known at compile-time.
 *
 * 604
 * ---
 * 604 has MMCR0, PMC1, PMC2, SIA, and SDA.
 *
 * MMCR0[THRESHOLD] is not automatically multiplied.
 *
 * On the 604, software must always reset MMCR0[ENINT] after
 * taking a PMI. This is not the case for the 604e.
 *
 * 604e
 * ----
 * 604e adds MMCR1, PMC3, and PMC4.
 * Bus-to-core multiplier is available via HID1[PLL_CFG].
 *
 * MMCR0[THRESHOLD] is automatically multiplied by 4.
 *
 * When the 604e vectors to the PMI handler, it automatically
 * clears any pending PMIs. Unlike the 604, the 604e does not
 * require MMCR0[ENINT] to be cleared (and possibly reset)
 * before external interrupts can be re-enabled.
 *
 * 750
 * ---
 * 750 adds user-readable MMCRn/PMCn/SIA registers, and removes SDA.
 *
 * MMCR0[THRESHOLD] is not automatically multiplied.
 *
 * Motorola MPC750UM.pdf, page C-78, states: "The performance monitor
 * of the MPC755 functions the same as that of the MPC750, (...), except
 * that for both the MPC750 and MPC755, no combination of the thermal
 * assist unit, the decrementer register, and the performance monitor
 * can be used at any one time. If exceptions for any two of these
 * functional blocks are enabled together, multiple exceptions caused
 * by any of these three blocks cause unpredictable results."
 *
 * IBM 750CXe_Err_DD2X.pdf, Erratum #13, states that a PMI which
 * occurs immediately after a delayed decrementer exception can
 * corrupt SRR0, causing the processor to hang. It also states that
 * PMIs via TB bit transitions can be used to simulate the decrementer.
 *
 * 750FX adds dual-PLL support and programmable core frequency switching.
 *
 * 750FX DD2.3 fixed the DEC/PMI SRR0 corruption erratum.
 *
 * 74xx
 * ----
 * 7400 adds MMCR2 and BAMR.
 *
 * MMCR0[THRESHOLD] is multiplied by 2 or 32, as specified
 * by MMCR2[THRESHMULT].
 *
 * 74xx changes the semantics of several MMCR0 control bits,
 * compared to 604/750.
 *
 * PPC7410 Erratum No. 10: Like the MPC750 TAU/DECR/PMI erratum.
 * Erratum No. 14 marks TAU as unsupported in 7410, but this leaves
 * perfmon and decrementer interrupts as being mutually exclusive.
 * Affects PPC7410 1.0-1.2 (PVR 0x800C1100-0x800C1102). 1.3 and up
 * (PVR 0x800C1103 up) are Ok.
 *
 * 7450 adds PMC5 and PMC6.
 *
 * 7455/7445 V3.3 (PVR 80010303) and later use the 7457 PLL table,
 * earlier revisions use the 7450 PLL table
 */

static inline unsigned int read_pmc(unsigned int pmc)
{
	switch (pmc) {
	default: /* impossible, but silences gcc warning */
	case 0:
		return mfspr(SPRN_PMC1);
	case 1:
		return mfspr(SPRN_PMC2);
	case 2:
		return mfspr(SPRN_PMC3);
	case 3:
		return mfspr(SPRN_PMC4);
	case 4:
		return mfspr(SPRN_PMC5);
	case 5:
		return mfspr(SPRN_PMC6);
	}
}

static void ppc_read_counters(struct perfctr_cpu_state *state,
			      struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		ctrs->tsc = get_tbl();
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		ctrs->pmc[i] = read_pmc(pmc);
	}
}

static unsigned int pmc_max_event(unsigned int pmc)
{
	switch (pmc) {
	default: /* impossible, but silences gcc warning */
	case 0:
		return 127;
	case 1:
		return 63;
	case 2:
		return 31;
	case 3:
		return 31;
	case 4:
		return 31;
	case 5:
		return 63;
	}
}

static unsigned int get_nr_pmcs(void)
{
	switch (pm_type) {
	case PM_7450:
		return 6;
	case PM_7400:
	case PM_750:
	case PM_604e:
		return 4;
	case PM_604:
		return 2;
	default: /* PM_NONE, but silences gcc warning */
		return 0;
	}
}

static int ppc_check_control(struct perfctr_cpu_state *state)
{
	unsigned int i, nractrs, nrctrs, pmc_mask, pmi_mask, pmc;
	unsigned int nr_pmcs, evntsel[6];

	nr_pmcs = get_nr_pmcs();
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
		if (evntsel[pmc] > pmc_max_event(pmc))
			return -EINVAL;
	}

	/* XXX: temporary limitation */
	if ((pmi_mask & ~1) && (pmi_mask & ~1) != (pmc_mask & ~1))
		return -EINVAL;

	switch (pm_type) {
	case PM_7450:
	case PM_7400:
		if (state->control.ppc.mmcr2 & MMCR2_RESERVED)
			return -EINVAL;
		state->ppc_mmcr[2] = state->control.ppc.mmcr2;
		break;
	default:
		if (state->control.ppc.mmcr2)
			return -EINVAL;
		state->ppc_mmcr[2] = 0;
	}

	/* We do not yet handle TBEE as the only exception cause,
	   so PMXE requires at least one interrupt-mode counter. */
	if ((state->control.ppc.mmcr0 & MMCR0_PMXE) && !state->control.nrictrs)
		return -EINVAL;
	if (state->control.ppc.mmcr0 & MMCR0_RESERVED)
		return -EINVAL;
	state->ppc_mmcr[0] = (state->control.ppc.mmcr0
			      | (evntsel[0] << (31-25))
			      | (evntsel[1] << (31-31)));

	state->ppc_mmcr[1] = ((  evntsel[2] << (31-4))
			      | (evntsel[3] << (31-9))
			      | (evntsel[4] << (31-14))
			      | (evntsel[5] << (31-20)));

	state->k1.id = new_id();

	/*
	 * MMCR0[FC] and MMCR0[TRIGGER] may change on 74xx if FCECE or
	 * TRIGGER is set. At suspends we must read MMCR0 back into
	 * the state and the cache and then freeze the counters, and
	 * at resumes we must unfreeze the counters and reload MMCR0.
	 */
	switch (pm_type) {
	case PM_7450:
	case PM_7400:
		if (state->ppc_mmcr[0] & (MMCR0_FCECE | MMCR0_TRIGGER))
			state->cstatus = perfctr_cstatus_set_mmcr0_quirk(state->cstatus);
	default:
		;
	}

	/* The MMCR0 handling for FCECE and TRIGGER is also needed for PMXE. */
	if (state->ppc_mmcr[0] & (MMCR0_PMXE | MMCR0_FCECE | MMCR0_TRIGGER))
		state->cstatus = perfctr_cstatus_set_mmcr0_quirk(state->cstatus);

	return 0;
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
/* PRE: counters frozen */
static void ppc_isuspend(struct perfctr_cpu_state *state)
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

static void ppc_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;
	unsigned int pmc[6];

	cpu = smp_processor_id();
	cache = __get_cpu_cache(cpu);
	if (cache->k1.id == state->k1.id) {
		/* Clearing cache->k1.id to force write_control()
		   to unfreeze MMCR0 would be done here, but it
		   is subsumed by resume()'s MMCR0 reload logic. */
		if (is_isuspend_cpu(state, cpu))
			return; /* skip reload of PMCs */
	}
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

static void ppc_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int value;

	cache = get_cpu_cache();
	if (cache->k1.id == state->k1.id)
		return;
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
	value = state->ppc_mmcr[2];
	if (value != cache->ppc_mmcr[2]) {
		cache->ppc_mmcr[2] = value;
		mtspr(SPRN_MMCR2, value);
	}
	value = state->ppc_mmcr[1];
	if (value != cache->ppc_mmcr[1]) {
		cache->ppc_mmcr[1] = value;
		mtspr(SPRN_MMCR1, value);
	}
	value = state->ppc_mmcr[0];
	if (value != cache->ppc_mmcr[0]) {
		cache->ppc_mmcr[0] = value;
		mtspr(SPRN_MMCR0, value);
	}
	cache->k1.id = state->k1.id;
}

static void ppc_clear_counters(void)
{
	switch (pm_type) {
	case PM_7450:
	case PM_7400:
		mtspr(SPRN_MMCR2, 0);
		mtspr(SPRN_BAMR, 0);
	case PM_750:
	case PM_604e:
		mtspr(SPRN_MMCR1, 0);
	case PM_604:
		mtspr(SPRN_MMCR0, 0);
	case PM_NONE:
		;
	}
	switch (pm_type) {
	case PM_7450:
		mtspr(SPRN_PMC6, 0);
		mtspr(SPRN_PMC5, 0);
	case PM_7400:
	case PM_750:
	case PM_604e:
		mtspr(SPRN_PMC4, 0);
		mtspr(SPRN_PMC3, 0);
	case PM_604:
		mtspr(SPRN_PMC2, 0);
		mtspr(SPRN_PMC1, 0);
	case PM_NONE:
		;
	}
}

/*
 * Driver methods, internal and exported.
 */

static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	return ppc_write_control(state);
}

static void perfctr_cpu_read_counters(struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	return ppc_read_counters(state, ctrs);
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	return ppc_isuspend(state);
}

static void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	return ppc_iresume(state);
}

/* Call perfctr_cpu_ireload() just before perfctr_cpu_resume() to
   bypass internal caching and force a reload if the I-mode PMCs. */
void perfctr_cpu_ireload(struct perfctr_cpu_state *state)
{
	state->ppc_mmcr[0] |= MMCR0_PMXE;
#ifdef CONFIG_SMP
	clear_isuspend_cpu(state);
#else
	get_cpu_cache()->k1.id = 0;
#endif
}

/* PRE: the counters have been suspended and sampled by perfctr_cpu_suspend() */
unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nrctrs, pmc, pmc_mask;

	cstatus = state->cstatus;
	pmc = perfctr_cstatus_nractrs(cstatus);
	nrctrs = perfctr_cstatus_nrctrs(cstatus);

	for(pmc_mask = 0; pmc < nrctrs; ++pmc) {
		if ((int)state->pmc[pmc].start < 0) { /* PPC-specific */
			/* XXX: "+=" to correct for overshots */
			state->pmc[pmc].start = state->control.ireset[pmc];
			pmc_mask |= (1 << pmc);
		}
	}
	if (!pmc_mask && (state->ppc_mmcr[0] & MMCR0_TBEE))
		pmc_mask = (1<<8); /* fake TB bit flip indicator */
	return pmc_mask;
}

static inline int check_ireset(const struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i;

	i = state->control.nractrs;
	nrctrs = i + state->control.nrictrs;
	for(; i < nrctrs; ++i)
		if (state->control.ireset[i] < 0)	/* PPC-specific */
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
	return ppc_check_control(state);
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

	if (perfctr_cstatus_has_mmcr0_quirk(state->cstatus)) {
		unsigned int mmcr0 = mfspr(SPRN_MMCR0);
		mtspr(SPRN_MMCR0, mmcr0 | MMCR0_FC);
		get_cpu_cache()->ppc_mmcr[0] = mmcr0 | MMCR0_FC;
		state->ppc_mmcr[0] = mmcr0;
	}
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
	if (perfctr_cstatus_has_mmcr0_quirk(state->cstatus))
		get_cpu_cache()->k1.id = 0; /* force reload of MMCR0 */
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

static void perfctr_cpu_clear_counters(void)
{
	struct per_cpu_cache *cache;

	cache = get_cpu_cache();
	memset(cache, 0, sizeof *cache);
	cache->k1.id = -1;

	ppc_clear_counters();
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

/* Derive CPU core frequency from TB frequency and PLL_CFG. */

enum pll_type {
	PLL_NONE,	/* for e.g. 604 which has no HID1[PLL_CFG] */
	PLL_604e,
	PLL_750,
	PLL_750FX,
	PLL_7400,
	PLL_7450,
	PLL_7457,
};

/* These are the known bus-to-core ratios, indexed by PLL_CFG.
   Multiplied by 2 since half-multiplier steps are present. */

static unsigned char cfg_ratio_604e[16] __initdata = { // *2
	2, 2, 14, 2, 4, 13, 5, 9,
	6, 11, 8, 10, 3, 12, 7, 0
};

static unsigned char cfg_ratio_750[16] __initdata = { // *2
	5, 15, 14, 2, 4, 13, 20, 9, // 0b0110 is 18 if L1_TSTCLK=0, but that is abnormal
	6, 11, 8, 10, 16, 12, 7, 0
};

static unsigned char cfg_ratio_750FX[32] __initdata = { // *2
	0, 0, 2, 2, 4, 5, 6, 7,
	8, 9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 22, 24, 26,
	28, 30, 32, 34, 36, 38, 40, 0
};

static unsigned char cfg_ratio_7400[16] __initdata = { // *2
	18, 15, 14, 2, 4, 13, 5, 9,
	6, 11, 8, 10, 16, 12, 7, 0
};

static unsigned char cfg_ratio_7450[32] __initdata = { // *2
	1, 0, 15, 30, 14, 0, 2, 0,
	4, 0, 13, 26, 5, 0, 9, 18,
	6, 0, 11, 22, 8, 20, 10, 24,
	16, 28, 12, 32, 7, 0, 0, 0
};

static unsigned char cfg_ratio_7457[32] __initdata = { // *2
	23, 34, 15, 30, 14, 36, 2, 40,
	4, 42, 13, 26, 17, 48, 19, 18,
	6, 21, 11, 22, 8, 20, 10, 24,
	16, 28, 12, 32, 27, 56, 0, 25
};

static unsigned int __init pll_tb_to_core(enum pll_type pll_type)
{
	unsigned char *cfg_ratio;
	unsigned int shift = 28, mask = 0xF, hid1, pll_cfg, ratio;

	switch (pll_type) {
	case PLL_604e:
		cfg_ratio = cfg_ratio_604e;
		break;
	case PLL_750:
		cfg_ratio = cfg_ratio_750;
		break;
	case PLL_750FX:
		cfg_ratio = cfg_ratio_750FX;
		hid1 = mfspr(SPRN_HID1);
		switch ((hid1 >> 16) & 0x3) { /* HID1[PI0,PS] */
		case 0:		/* PLL0 with external config */
			shift = 31-4;	/* access HID1[PCE] */
			break;
		case 2:		/* PLL0 with internal config */
			shift = 31-20;	/* access HID1[PC0] */
			break;
		case 1: case 3:	/* PLL1 */
			shift = 31-28;	/* access HID1[PC1] */
			break;
		}
		mask = 0x1F;
		break;
	case PLL_7400:
		cfg_ratio = cfg_ratio_7400;
		break;
	case PLL_7450:
		cfg_ratio = cfg_ratio_7450;
		shift = 12;
		mask = 0x1F;
		break;
	case PLL_7457:
		cfg_ratio = cfg_ratio_7457;
		shift = 12;
		mask = 0x1F;
		break;
	default:
		return 0;
	}
	hid1 = mfspr(SPRN_HID1);
	pll_cfg = (hid1 >> shift) & mask;
	ratio = cfg_ratio[pll_cfg];
	if (!ratio)
		printk(KERN_WARNING "perfctr: unknown PLL_CFG 0x%x\n", pll_cfg);
	return (4/2) * ratio;
}

/* Extract core and timebase frequencies from Open Firmware. */

#ifdef CONFIG_PPC_OF

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
static inline struct device_node *perfctr_of_find_node_by_type(struct device_node *from, const char *type)
{
	return of_find_node_by_type(from, type);
}

static inline void perfctr_of_node_put(struct device_node *node)
{
	of_node_put(node);
}
#else
#define of_get_property(a, b, c)	get_property((a), (b), (c))
static inline struct device_node *perfctr_of_find_node_by_type(struct device_node *from, const char *type)
{
	return find_type_devices(type);
}

static inline void perfctr_of_node_put(struct device_node *node) { }
#endif

static unsigned int __init of_core_khz(void)
{
	struct device_node *cpu;
	unsigned int *fp, core;

	cpu = perfctr_of_find_node_by_type(NULL, "cpu");
	if (!cpu)
		return 0;
	fp = (unsigned int*)of_get_property(cpu, "clock-frequency", NULL);
	core = 0;
	if (fp)
		core = *fp;
	perfctr_of_node_put(cpu);
	return core / 1000;
}

static unsigned int __init of_bus_khz(void)
{
	struct device_node *cpu;
	unsigned int *fp, bus;

	cpu = perfctr_of_find_node_by_type(NULL, "cpu");
	if (!cpu)
		return 0;
	fp = (unsigned int*)of_get_property(cpu, "bus-frequency", NULL);
	bus = 0;
	if (!fp || !(bus = *fp)) {
		fp = (unsigned int*)of_get_property(cpu, "config-bus-frequency", NULL);
		if (fp)
			bus = *fp;
	}
	perfctr_of_node_put(cpu);
	return bus / 1000;
}

static unsigned int __init of_bus_to_core_x2(void)
{
	struct device_node *cpu;
	unsigned int *fp, ratio;

	cpu = perfctr_of_find_node_by_type(NULL, "cpu");
	if (!cpu)
		return 0;
	fp = (unsigned int*)of_get_property(cpu, "processor-to-bus-ratio*2", NULL);
	ratio = 0;
	if (fp)
		ratio = *fp;
	perfctr_of_node_put(cpu);
	return ratio;
}
#else
static inline unsigned int of_core_khz(void) { return 0; }
static inline unsigned int of_bus_khz(void) { return 0; }
static inline unsigned int of_bus_to_core_x2(void) { return 0; }
#endif

static unsigned int __init detect_tb_khz(unsigned int bus_khz, unsigned int tb_to_bus)
{
	unsigned int tb_khz, bus_tb_khz, diff;

	tb_khz = tb_ticks_per_jiffy * (HZ/10) / (1000/10);
	if (bus_khz && tb_to_bus) {
		bus_tb_khz = bus_khz / tb_to_bus;
		if (bus_tb_khz > tb_khz)
			diff = bus_tb_khz - tb_khz;
		else
			diff = tb_khz - bus_tb_khz;
		if (diff >= bus_tb_khz/20) {
			printk(KERN_WARNING "perfctr: timebase frequency %u kHz seems"
			       " out of range, using %u kHz (bus/%u) instead\n",
			       tb_khz, bus_tb_khz, tb_to_bus);
			return bus_tb_khz;
		}
	}
	return tb_khz;
}

static unsigned int __init detect_tb_to_core(enum pll_type pll_type, unsigned int tb_to_bus)
{
	unsigned int tb_to_core;
	unsigned int bus_to_core_x2;

	tb_to_core = pll_tb_to_core(pll_type);
	if (tb_to_core)
		return tb_to_core;
	if (tb_to_bus) {
		bus_to_core_x2 = of_bus_to_core_x2();
		if (bus_to_core_x2)
			return (tb_to_bus * bus_to_core_x2) / 2;
	}
	return 0;
}

static unsigned int __init detect_core_khz(unsigned int tb_khz, unsigned int tb_to_core)
{
	unsigned int core_khz;

	if (tb_to_core) {
		perfctr_info.tsc_to_cpu_mult = tb_to_core;
		return tb_khz * tb_to_core;
	}
	core_khz = of_core_khz();
	perfctr_info.tsc_to_cpu_mult = core_khz / tb_khz;
	return core_khz;
}

/*
 * Detect the timebase and core clock frequencies.
 *
 * Known issues:
 * 1. The OF timebase-frequency property is sometimes way off, and
 *    similarly the ppc32 kernel's tb_ticks_per_jiffy variable.
 *    (Observed on a 7447A-based laptop.)
 *    Workaround: Compute the TB frequency from the bus frequency
 *    and the TB-to-bus ratio.
 * 2. The OF clock-frequency property is sometimes wrong.
 *    (Observed on a Beige G3 with a 7455 upgrade processor.)
 *    Workaround: Compute the core frequency from the TB frequency
 *    and the TB-to-core ratio.
 * 3. The PLL_CFG details may be unknown.
 */
static unsigned int __init detect_cpu_khz(enum pll_type pll_type, unsigned int tb_to_bus)
{
	unsigned int bus_khz;
	unsigned int tb_khz;
	unsigned int tb_to_core;
	unsigned int core_khz;

	bus_khz = of_bus_khz();
	tb_khz = detect_tb_khz(bus_khz, tb_to_bus);
	tb_to_core = detect_tb_to_core(pll_type, tb_to_bus);
	core_khz = detect_core_khz(tb_khz, tb_to_core);
	if (!core_khz)
		printk(KERN_WARNING "perfctr: unable to determine CPU speed\n");
	return core_khz;
}

static int __init known_init(void)
{
	static char known_name[] __initdata = "PowerPC 60x/7xx/74xx";
	unsigned int features;
	enum pll_type pll_type;
	unsigned int pvr;
	int have_mmcr1;
	unsigned int tb_to_bus;

	tb_to_bus = 4; /* default, overridden below if necessary */
	features = PERFCTR_FEATURE_RDTSC | PERFCTR_FEATURE_RDPMC;
	have_mmcr1 = 1;
	pvr = mfspr(SPRN_PVR);
	switch (PVR_VER(pvr)) {
	case 0x0004: /* 604 */
		pm_type = PM_604;
		pll_type = PLL_NONE;
		features = PERFCTR_FEATURE_RDTSC;
		have_mmcr1 = 0;
		break;
	case 0x0009: /* 604e;  */
	case 0x000A: /* 604ev */
		pm_type = PM_604e;
		pll_type = PLL_604e;
		features = PERFCTR_FEATURE_RDTSC;
		break;
	case 0x0008: /* 750/740 */
		pm_type = PM_750;
		pll_type = PLL_750;
		break;
	case 0x7000: case 0x7001: /* IBM750FX */
		if ((pvr & 0xFF0F) >= 0x0203)
			features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_750;
		pll_type = PLL_750FX;
		break;
	case 0x7002: /* IBM750GX */
		features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_750;
		pll_type = PLL_750FX;
		break;
	case 0x000C: /* 7400 */
		pm_type = PM_7400;
		pll_type = PLL_7400;
		break;
	case 0x800C: /* 7410 */
		if ((pvr & 0xFFFF) >= 0x1103)
			features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_7400;
		pll_type = PLL_7400;
		break;
	case 0x8000: /* 7451/7441 */
		features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_7450;
		pll_type = PLL_7450;
		break;
	case 0x8001: /* 7455/7445 */
		features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_7450;
		pll_type = ((pvr & 0xFFFF) < 0x0303) ? PLL_7450 : PLL_7457;
		break;
	case 0x8002: /* 7457/7447 */
	case 0x8003: /* 7447A */
		features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_7450;
		pll_type = PLL_7457;
		break;
	case 0x8004: /* 7448 */
		features |= PERFCTR_FEATURE_PCINT;
		pm_type = PM_7450;
		pll_type = PLL_NONE; /* known to differ from 7447A, no details yet */
		break;
	default:
		return -ENODEV;
	}
	perfctr_info.cpu_features = features;
	perfctr_info.cpu_type = 0; /* user-space should inspect PVR */
	perfctr_cpu_name = known_name;
	perfctr_info.cpu_khz = detect_cpu_khz(pll_type, tb_to_bus);
	perfctr_ppc_init_tests(have_mmcr1);
	return 0;
}

static int __init unknown_init(void)
{
	static char unknown_name[] __initdata = "Generic PowerPC with TB";
	unsigned int khz;

	khz = detect_cpu_khz(PLL_NONE, 0);
	if (!khz)
		return -ENODEV;
	perfctr_info.cpu_features = PERFCTR_FEATURE_RDTSC;
	perfctr_info.cpu_type = 0;
	perfctr_cpu_name = unknown_name;
	perfctr_info.cpu_khz = khz;
	pm_type = PM_NONE;
	return 0;
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

int __init perfctr_cpu_init(void)
{
	int err;

	perfctr_info.cpu_features = 0;

	err = known_init();
	if (err) {
		err = unknown_init();
		if (err)
			goto out;
	}

 out:
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
	ret = "unknown driver (oprofile?)";
	if (perfctr_reserve_pmc_hardware() < 0)
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
	} else {
		/* power down the counters */
		perfctr_cpu_reset();
		current_service = 0;
		perfctr_release_pmc_hardware();
		module_put(THIS_MODULE);
	}
	mutex_unlock(&mutex);
}
