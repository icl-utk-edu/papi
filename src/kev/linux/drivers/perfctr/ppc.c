/* $Id$
 * PPC32 performance-monitoring counters driver.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <linux/seq_file.h>
#include <asm/machdep.h>
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

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[6];
};

enum pm_type {
    PM_604,
    PM_604e,
    PM_750,	/* XXX: Minor event set diffs between IBM and Moto. */
    PM_7400,
    PM_7450,
};
static enum pm_type pm_type;

#define SPRN_MMCR0	0x3B8	/* 604 and up */
#define SPRN_PMC1	0x3B9	/* 604 and up */
#define SPRN_PMC2	0x3BA	/* 604 and up */
#define SPRN_SIA	0x3BB	/* 604 and up */
#define SPRN_MMCR1	0x3BC	/* 604e and up */
#define SPRN_PMC3	0x3BD	/* 604e and up */
#define SPRN_PMC4	0x3BE	/* 604e and up */
#define SPRN_MMCR2	0x3B0	/* 7400 and up */
#define SPRN_BAMR	0x3B7	/* 7400 and up */
#define SPRN_PMC5	0x3B1	/* 7450 and up */
#define SPRN_PMC6	0x3B2	/* 7450 and up */

/* MMCR0 layout (74xx terminology) */
#define MMCR0_FC		0x80000000 /* Freeze counters unconditionally. */
#define MMCR0_FCS		0x40000000 /* Freeze counters while MSR[PR]=0 (supervisor mode). */
#define MMCR0_FCP		0x20000000 /* Freeze counters while MSR[PR]=1 (user mode). */
#define MMCR0_FCM1		0x10000000 /* Freeze counters while MSR[PM]=1. */
#define MMCR0_FCM0		0x08000000 /* Freeze counters while MSR[PM]=0. */
#define MMCR0_PMXE		0x04000000 /* Enable performance monitor exceptions.
					    * Cleared by hardware when a PM exception occurs.
					    * 604: PMXE is not cleared by hardware.
					    */
#define MMCR0_FCECE		0x02000000 /* Freeze counters on enabled condition or event.
					    * FCECE is treated as 0 if TRIGGER is 1.
					    * 74xx: FC is set when the event occurs.
					    * 604/750: ineffective when PMXE=0.
					    */
#define MMCR0_TBSEL		0x01800000 /* Time base lower (TBL) bit selector.
					    * 00: bit 31, 01: bit 23, 10: bit 19, 11: bit 15.
					    */
#define MMCR0_TBEE		0x00400000 /* Enable event on TBL bit transition from 0 to 1. */
#define MMCR0_THRESHOLD		0x003F0000 /* Threshold value for certain events. */
#define MMCR0_PMC1CE		0x00008000 /* Enable event on PMC1 overflow. */
#define MMCR0_PMCjCE		0x00004000 /* Enable event on PMC2-PMC6 overflow.
					    * 604/750: Overrides FCECE (DISCOUNT).
					    */
#define MMCR0_TRIGGER		0x00002000 /* Disable PMC2-PMC6 until PMC1 overflow or other event.
					    * 74xx: cleared by hardware when the event occurs.
					    */
#define MMCR0_PMC1SEL		0x00001FB0 /* PMC1 event selector, 7 bits. */
#define MMCR0_PMC2SEL		0x0000003F /* PMC2 event selector, 6 bits. */
#define MMCR0_RESERVED		(MMCR0_PMXE | MMCR0_PMC1SEL | MMCR0_PMC2SEL)

/* MMCR1 layout (604e-7457) */
#define MMCR1_PMC3SEL		0xF8000000 /* PMC3 event selector, 5 bits. */
#define MMCR1_PMC4SEL		0x07B00000 /* PMC4 event selector, 5 bits. */
#define MMCR1_PMC5SEL		0x003E0000 /* PMC5 event selector, 5 bits. (745x only) */
#define MMCR1_PMC6SEL		0x0001F800 /* PMC6 event selector, 6 bits. (745x only) */
#define MMCR1__RESERVED		0x000007FF /* should be zero */

/* MMCR2 layout (7400-7457) */
#define MMCR2_THRESHMULT	0x80000000 /* MMCR0[THRESHOLD] multiplier. */
#define MMCR2_SMCNTEN		0x40000000 /* 7400/7410 only, should be zero. */
#define MMCR2_SMINTEN		0x20000000 /* 7400/7410 only, should be zero. */
#define MMCR2__RESERVED		0x1FFFFFFF /* should be zero */
#define MMCR2_RESERVED		(MMCR2_SMCNTEN | MMCR2_SMINTEN | MMCR2__RESERVED)

static unsigned int new_id(void)
{
	static spinlock_t lock = SPIN_LOCK_UNLOCKED;
	static unsigned int counter;
	int id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

#if !defined(PERFCTR_INTERRUPT_SUPPORT)
#define perfctr_cstatus_has_ictrs(cstatus)	0
#endif

#if defined(CONFIG_SMP) && PERFCTR_INTERRUPT_SUPPORT

static inline void set_isuspend_cpu(struct perfctr_cpu_state *state,
				    const struct per_cpu_cache *cache)
{
	state->k1.isuspend_cpu = cache;
}

static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state,
				  const struct per_cpu_cache *cache)
{
	return state->k1.isuspend_cpu == cache;
}

static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state)
{
	state->k1.isuspend_cpu = NULL;
}

#else
static inline void set_isuspend_cpu(struct perfctr_cpu_state *state,
				    const struct per_cpu_cache *cache) { }
static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state,
				  const struct per_cpu_cache *cache) { return 1; }
static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state) { }
#endif

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
 * Driver notes
 * ------------
 * - The driver currently does not support performance monitor interrupts,
 *   mostly because of the 750/7400/7410 erratum. Working around it would
 *   require disabling the decrementer interrupt, reserving a performance
 *   counter and setting it up for TBL bit-flip events, and having the PMI
 *   handler invoke the decrementer handler.
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
	switch( pmc ) {
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

static void ppc_read_counters(/*const*/ struct perfctr_cpu_state *state,
			      struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		ctrs->tsc = get_tbl();
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		ctrs->pmc[i] = read_pmc(pmc);
	}
	/* handle MMCR0 changes due to FCECE or TRIGGER on 74xx */
	if( state->cstatus & (1<<30) ) {
		unsigned int mmcr0 = mfspr(SPRN_MMCR0);
		state->ppc_mmcr[0] = mmcr0;
		per_cpu_cache[smp_processor_id()].ppc_mmcr[0] = mmcr0;
	}
}

static unsigned int pmc_max_event(unsigned int pmc)
{
	switch( pmc ) {
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
	switch( pm_type ) {
	case PM_7450:
		return 6;
	case PM_7400:
	case PM_750:
	case PM_604e:
		return 4;
	default: /* impossible, but silences gcc warning */
	case PM_604:
		return 2;
	}
}

static int ppc_check_control(struct perfctr_cpu_state *state)
{
	unsigned int i, nrctrs, pmc_mask, pmc;
	unsigned int nr_pmcs, evntsel[6];

	nr_pmcs = get_nr_pmcs();
	nrctrs = state->control.nractrs;
	if( state->control.nrictrs || nrctrs > nr_pmcs )
		return -EINVAL;

	pmc_mask = 0;
	memset(evntsel, 0, sizeof evntsel);
	for(i = 0; i < nrctrs; ++i) {
		pmc = state->control.pmc_map[i];
		state->pmc[i].map = pmc;
		if( pmc >= nr_pmcs || (pmc_mask & (1<<pmc)) )
			return -EINVAL;
		pmc_mask |= (1<<pmc);

		evntsel[pmc] = state->control.evntsel[i];
		if( evntsel[pmc] > pmc_max_event(pmc) )
			return -EINVAL;
	}

	switch( pm_type ) {
	case PM_7450:
		if( state->control.ppc.mmcr2 & MMCR2_RESERVED )
			return -EINVAL;
		state->ppc_mmcr[2] = state->control.ppc.mmcr2;
		break;
	default:
		if( state->control.ppc.mmcr2 )
			return -EINVAL;
		state->ppc_mmcr[2] = 0;
	}

	if( state->control.ppc.mmcr0 & MMCR0_RESERVED )
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
	 * TRIGGER is set. To avoid undoing those changes, we must read
	 * MMCR0 back into state->ppc_mmcr[0] and the cache at suspends.
	 */
	switch( pm_type ) {
	case PM_7450:
	case PM_7400:
		if( state->ppc_mmcr[0] & (MMCR0_FCECE | MMCR0_TRIGGER) )
			state->cstatus |= (1<<30);
	default:
		;
	}

	return 0;
}

#if PERFCTR_INTERRUPT_SUPPORT
static void ppc_isuspend(struct perfctr_cpu_state *state)
{
	// XXX
}

static void ppc_iresume(const struct perfctr_cpu_state *state)
{
	// XXX
}
#endif

static void ppc_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int value;

	cache = &per_cpu_cache[smp_processor_id()];
	if( cache->k1.id == state->k1.id ) {
		//debug_evntsel_cache(state, cache);
		return;
	}
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
	if( value != cache->ppc_mmcr[2] ) {
		cache->ppc_mmcr[2] = value;
		mtspr(SPRN_MMCR2, value);
	}
	value = state->ppc_mmcr[1];
	if( value != cache->ppc_mmcr[1] ) {
		cache->ppc_mmcr[1] = value;
		mtspr(SPRN_MMCR1, value);
	}
	value = state->ppc_mmcr[0];
	if( value != cache->ppc_mmcr[0] ) {
		cache->ppc_mmcr[0] = value;
		mtspr(SPRN_MMCR0, value);
	}
	cache->k1.id = state->k1.id;
}

static void ppc_clear_counters(void)
{
	switch( pm_type ) {
	case PM_7450:
	case PM_7400:
		mtspr(SPRN_MMCR2, 0);
		mtspr(SPRN_BAMR, 0);
	case PM_750:
	case PM_604e:
		mtspr(SPRN_MMCR1, 0);
	case PM_604:
		mtspr(SPRN_MMCR0, 0);
	}
	switch( pm_type ) {
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
	}
}

/*
 * Driver methods, internal and exported.
 */

static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	return ppc_write_control(state);
}

static void perfctr_cpu_read_counters(/*const*/ struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	return ppc_read_counters(state, ctrs);
}

#if PERFCTR_INTERRUPT_SUPPORT
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
#ifdef CONFIG_SMP
	clear_isuspend_cpu(state);
#else
	per_cpu_cache[smp_processor_id()].k1.id = 0;
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
		if( (int)state->pmc[pmc].start < 0 ) { /* PPC-specific */
			/* XXX: "+=" to correct for overshots */
			state->pmc[pmc].start = state->control.ireset[pmc];
			pmc_mask |= (1 << pmc);
		}
	}
	/* XXX: if pmc_mask == 0, then it must have been a TBL bit flip */
	/* XXX: HW cleared MMCR0[ENINT]. We presumably cleared the entire
	   MMCR0, so the re-enable occurs automatically later, no? */
	return pmc_mask;
}

static inline int check_ireset(const struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i;

	i = state->control.nractrs;
	nrctrs = i + state->control.nrictrs;
	for(; i < nrctrs; ++i)
		if( state->control.ireset[i] < 0 )	/* PPC-specific */
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

static inline void debug_no_imode(const struct perfctr_cpu_state *state)
{
#ifdef CONFIG_PERFCTR_DEBUG
	if( perfctr_cstatus_has_ictrs(state->cstatus) )
		printk(KERN_ERR "perfctr/%s: BUG! updating control in"
		       " perfctr %p on cpu %u while it has cstatus %x"
		       " (pid %d, comm %s)\n",
		       __FILE__, state, smp_processor_id(), state->cstatus,
		       current->pid, current->comm);
#endif
}

#else	/* PERFCTR_INTERRUPT_SUPPORT */
static inline void perfctr_cpu_isuspend(struct perfctr_cpu_state *state) { }
static inline void perfctr_cpu_iresume(const struct perfctr_cpu_state *state) { }
static inline int check_ireset(const struct perfctr_cpu_state *state) { return 0; }
static inline void setup_imode_start_values(struct perfctr_cpu_state *state) { }
static inline void debug_no_imode(const struct perfctr_cpu_state *state) { }
#endif	/* PERFCTR_INTERRUPT_SUPPORT */

static int check_control(struct perfctr_cpu_state *state)
{
	return ppc_check_control(state);
}

int perfctr_cpu_update_control(struct perfctr_cpu_state *state, int is_global)
{
	int err;

	debug_no_imode(state);
	clear_isuspend_cpu(state);
	state->cstatus = 0;

	/* disallow i-mode counters if we cannot catch the interrupts */
	if( !(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
	    && state->control.nrictrs )
		return -EPERM;

	err = check_ireset(state);
	if( err < 0 )
		return err;
	err = check_control(state); /* may initialise state->cstatus */
	if( err < 0 )
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

	if( perfctr_cstatus_has_ictrs(state->cstatus) )
	    perfctr_cpu_isuspend(state);
	perfctr_cpu_read_counters(state, &now);
	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		state->tsc_sum += now.tsc - state->tsc_start;
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i)
		state->pmc[i].sum += now.pmc[i] - state->pmc[i].start;
}

void perfctr_cpu_resume(struct perfctr_cpu_state *state)
{
	if( perfctr_cstatus_has_ictrs(state->cstatus) )
	    perfctr_cpu_iresume(state);
	perfctr_cpu_write_control(state);
	//perfctr_cpu_read_counters(state, &state->start);
	{
		struct perfctr_low_ctrs now;
		unsigned int i, cstatus, nrctrs;
		perfctr_cpu_read_counters(state, &now);
		cstatus = state->cstatus;
		if( perfctr_cstatus_has_tsc(cstatus) )
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
	if( perfctr_cstatus_has_tsc(cstatus) ) {
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

	cache = &per_cpu_cache[smp_processor_id()];
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

static unsigned int __init tb_to_core_ratio(enum pll_type pll_type)
{
	unsigned char *cfg_ratio;
	unsigned int shift = 28, mask = 0xF, hid1, pll_cfg, ratio;

	switch( pll_type ) {
	case PLL_604e:
		cfg_ratio = cfg_ratio_604e;
		break;
	case PLL_750:
		cfg_ratio = cfg_ratio_750;
		break;
	case PLL_750FX:
		cfg_ratio = cfg_ratio_750FX;
		hid1 = mfspr(SPRN_HID1);
		switch( (hid1 >> 16) & 0x3 ) { /* HID1[PI0,PS] */
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
	if( !ratio )
		printk(KERN_WARNING "perfctr/%s: unknown PLL_CFG 0x%x\n",
		       __FILE__, pll_cfg);
	return (4/2) * ratio;
}

static unsigned int __init pll_to_core_khz(enum pll_type pll_type)
{
	unsigned int tb_to_core = tb_to_core_ratio(pll_type);
	perfctr_info.tsc_to_cpu_mult = tb_to_core;
	return tb_ticks_per_jiffy * tb_to_core * (HZ/10) / (1000/10);
}

/* Extract the CPU clock frequency from /proc/cpuinfo. */

static unsigned int __init parse_clock_khz(struct seq_file *m)
{
	/* "/proc/cpuinfo" formats:
	 *
	 * "core clock\t: %d MHz\n"	// 8260 (show_percpuinfo)
	 * "clock\t\t: %ldMHz\n"	// 4xx (show_percpuinfo)
	 * "clock\t\t: %dMHz\n"		// oak (show_percpuinfo)
	 * "clock\t\t: %ldMHz\n"	// prep (show_percpuinfo)
	 * "clock\t\t: %dMHz\n"		// pmac (show_percpuinfo)
	 * "clock\t\t: %dMHz\n"		// gemini (show_cpuinfo!)
	 */
	char *p;
	unsigned int mhz;

	p = m->buf;
	p[m->count] = '\0';

	for(;;) {		/* for each line */
		if( strncmp(p, "core ", 5) == 0 )
			p += 5;
		do {
			if( strncmp(p, "clock\t", 6) != 0 )
				break;
			p += 6;
			while( *p == '\t' )
				++p;
			if( *p != ':' )
				break;
			do {
				++p;
			} while( *p == ' ' );
			mhz = simple_strtoul(p, 0, 10);
			if( mhz )
				return mhz * 1000;
		} while( 0 );
		for(;;) {	/* skip to next line */
			switch( *p++ ) {
			case '\n':
				break;
			case '\0':
				return 0;
			default:
				continue;
			}
			break;
		}
	}
}

static unsigned int __init detect_cpu_khz(enum pll_type pll_type)
{
	char buf[512];
	struct seq_file m;
	unsigned int khz;

	khz = pll_to_core_khz(pll_type);
	if( khz )
		return khz;

	memset(&m, 0, sizeof m);
	m.buf = buf;
	m.size = (sizeof buf)-1;

	m.count = 0;
	if( ppc_md.show_percpuinfo != 0 &&
	    ppc_md.show_percpuinfo(&m, 0) == 0 &&
	    (khz = parse_clock_khz(&m)) != 0 )
		return khz;

	m.count = 0;
	if( ppc_md.show_cpuinfo != 0 &&
	    ppc_md.show_cpuinfo(&m) == 0 &&
	    (khz = parse_clock_khz(&m)) != 0 )
		return khz;

	printk(KERN_WARNING "perfctr/%s: unable to determine CPU speed\n",
	       __FILE__);
	return 0;
}

static int __init generic_init(void)
{
	static char generic_name[] __initdata = "PowerPC 60x/7xx/74xx";
	unsigned int features;
	enum pll_type pll_type;
	unsigned int pvr;

	features = PERFCTR_FEATURE_RDTSC | PERFCTR_FEATURE_RDPMC;
	pvr = mfspr(SPRN_PVR);
	switch( PVR_VER(pvr) ) {
	case 0x0004: /* 604 */
		pm_type = PM_604;
		pll_type = PLL_NONE;
		features = PERFCTR_FEATURE_RDTSC;
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
		pm_type = PM_750;
		pll_type = PLL_750FX;
		break;
	case 0x000C: /* 7400 */
		pm_type = PM_7400;
		pll_type = PLL_7400;
		break;
	case 0x800C: /* 7410 */
		pm_type = PM_7400;
		pll_type = PLL_7400;
		break;
	case 0x8000: /* 7451/7441 */
		pm_type = PM_7450;
		pll_type = PLL_7450;
		break;
	case 0x8001: /* 7455/7445 */
		pm_type = PM_7450;
		pll_type = ((pvr & 0xFFFF) < 0x0303) ? PLL_7450 : PLL_7457;
		break;
	case 0x8002: /* 7457/7447 */
		pm_type = PM_7450;
		pll_type = PLL_7457;
		break;
	default:
		printk(KERN_WARNING "perfctr/%s: unknown PowerPC with "
		       "PVR 0x%08x -- bailing out\n", __FILE__, pvr);
		return -ENODEV;
	}
	perfctr_info.cpu_features = features;
	perfctr_info.cpu_type = 0; /* user-space should inspect PVR */
	perfctr_cpu_name = generic_name;
	perfctr_info.cpu_khz = detect_cpu_khz(pll_type);
	perfctr_ppc_init_tests();
	return 0;
}

static void __init perfctr_cpu_init_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
}

static void __exit perfctr_cpu_exit_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
}

int __init perfctr_cpu_init(void)
{
	int err;

	preempt_disable();

	perfctr_info.cpu_features = 0;

	err = generic_init();
	if( err )
		goto out;

	perfctr_cpu_init_one(NULL);
	smp_call_function(perfctr_cpu_init_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
 out:
	preempt_enable();
	return err;
}

void __exit perfctr_cpu_exit(void)
{
	preempt_disable();
	perfctr_cpu_exit_one(NULL);
	smp_call_function(perfctr_cpu_exit_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	preempt_enable();
}

/****************************************************************
 *								*
 * Hardware reservation.					*
 *								*
 ****************************************************************/

static const char *current_service = 0;

const char *perfctr_cpu_reserve(const char *service)
{
	if( current_service )
		return current_service;
	current_service = service;
	__module_get(THIS_MODULE);
	return 0;
}

static void perfctr_cpu_clear_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
}

void perfctr_cpu_release(const char *service)
{
	if( service != current_service ) {
		printk(KERN_ERR "%s: attempt by %s to release while reserved by %s\n",
		       __FUNCTION__, service, current_service);
	} else {
		/* power down the counters */
		on_each_cpu(perfctr_cpu_clear_one, NULL, 1, 1);
		perfctr_cpu_set_ihandler(NULL);
		current_service = 0;
		module_put(THIS_MODULE);
	}
}
