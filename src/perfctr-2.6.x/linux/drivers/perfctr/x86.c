/* $Id: x86.c,v 1.127.2.70 2010/11/07 19:46:06 mikpe Exp $
 * x86/x86_64 performance-monitoring counters driver.
 *
 * Copyright (C) 1999-2010  Mikael Pettersson
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

#include <asm/msr.h>
#undef MSR_P6_PERFCTR0
#undef MSR_P6_EVNTSEL0
#undef MSR_K7_PERFCTR0
#undef MSR_K7_EVNTSEL0
#undef MSR_CORE_PERF_FIXED_CTR0
#undef MSR_CORE_PERF_FIXED_CTR_CTRL
#undef MSR_CORE_PERF_GLOBAL_CTRL
#undef MSR_IA32_MISC_ENABLE
#undef MSR_IA32_MISC_ENABLE_PEBS_UNAVAIL
#undef MSR_IA32_DEBUGCTLMSR
#include <asm/fixmap.h>
#include <asm/apic.h>
struct hw_interrupt_type;
#include <asm/hw_irq.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,18) && defined(CONFIG_X86_LOCAL_APIC)
#include <asm/nmi.h>
#endif

#include "compat.h"
#include "x86_compat.h"
#include "x86_tests.h"

/* Support for lazy evntsel and perfctr MSR updates. */
struct per_cpu_cache {	/* roughly a subset of perfctr_cpu_state */
	union {
		unsigned int p5_cesr;
		unsigned int id;	/* cache owner id */
	} k1;
	struct {
		/* NOTE: these caches have physical indices, not virtual */
		unsigned int evntsel[18];
		union {
			unsigned int escr[0x3E2-0x3A0];
			unsigned int evntsel_high[18];
		};
		unsigned int pebs_enable;
		unsigned int pebs_matrix_vert;
	} control;
	unsigned int core2_fixed_ctr_ctrl;
	unsigned int nhlm_offcore_rsp[2];
} ____cacheline_aligned;
static struct per_cpu_cache per_cpu_cache[NR_CPUS] __cacheline_aligned;
#define __get_cpu_cache(cpu) (&per_cpu_cache[cpu])
#define get_cpu_cache() __get_cpu_cache(smp_processor_id())

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[18];
};

/* Structures for describing the set of PMU MSRs. */
struct perfctr_msr_range {
	unsigned int first_msr;
	unsigned int nr_msrs;
};

struct perfctr_pmu_msrs {
	const struct perfctr_msr_range *perfctrs; /* for {reserve,release}_perfctr_nmi() */
	const struct perfctr_msr_range *evntsels; /* for {reserve,release}_evntsel_nmi() */
	const struct perfctr_msr_range *extras;
	void (*clear_counters)(int init);
};

/* Intel P5, Cyrix 6x86MX/MII/III, Centaur WinChip C6/2/3 */
#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12		/* .. 0x13 */
#define P5_CESR_CPL		0x00C0
#define P5_CESR_RESERVED	(~0x01FF)
#define MII_CESR_RESERVED	(~0x05FF)
#define C6_CESR_RESERVED	(~0x00FF)

/* Intel P6, VIA C3 */
#define MSR_P6_PERFCTR0		0xC1		/* .. 0xC4 */
#define MSR_P6_EVNTSEL0		0x186		/* .. 0x189 */
#define P6_EVNTSEL_ENABLE	0x00400000
#define P6_EVNTSEL_INT		0x00100000
#define P6_EVNTSEL_CPL		0x00030000
#define P6_EVNTSEL_RESERVED	0x00280000
#define VC3_EVNTSEL1_RESERVED	(~0x1FF)

/* Intel Core */
#define MSR_IA32_DEBUGCTLMSR		0x000001D9
#define MSR_IA32_DEBUGCTLMSR_FREEZE_PERFMON_ON_PMI	(1<<12)
#define MSR_CORE_PERF_FIXED_CTR0	0x309	/* .. 0x30B */
#define MSR_CORE_PERF_FIXED_CTR_CTRL	0x38D
#define MSR_CORE_PERF_FIXED_CTR_CTRL_PMIANY	0x00000888
#define MSR_CORE_PERF_GLOBAL_CTRL	0x38F
#define CORE2_PMC_FIXED_FLAG		(1<<30)
#define CORE2_PMC_FIXED_MASK		0x3

/* Intel Nehalem */
#define MSR_OFFCORE_RSP0	0x1A6		/* Westmere has another at 0x1A7 */
#define OFFCORE_RSP_RESERVED	(~0xF7FF)

/* AMD K7 */
#define MSR_K7_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K7_PERFCTR0		0xC0010004	/* .. 0xC0010007 */

/* AMD K8 */
#define IS_K8_NB_EVENT(EVNTSEL)	((((EVNTSEL) >> 5) & 0x7) == 0x7)

/* AMD Family 10h */
#define FAM10H_EVNTSEL_HIGH_RESERVED	(~0x30F)

/* Intel P4, Intel Pentium M, Intel Core */
#define MSR_IA32_MISC_ENABLE	0x1A0
#define MSR_IA32_MISC_ENABLE_PERF_AVAIL (1<<7)	/* read-only status bit */
#define MSR_IA32_MISC_ENABLE_PEBS_UNAVAIL (1<<12) /* read-only status bit */

/* Intel P4 */
#define MSR_P4_PERFCTR0		0x300		/* .. 0x311 */
#define MSR_P4_CCCR0		0x360		/* .. 0x371 */
#define MSR_P4_ESCR0		0x3A0		/* .. 0x3E1, with some gaps */

#define MSR_P4_PEBS_ENABLE	0x3F1
#define P4_PE_REPLAY_TAG_BITS	0x00000607
#define P4_PE_UOP_TAG		0x01000000
#define P4_PE_RESERVED		0xFEFFF9F8	/* only allow ReplayTagging */

#define MSR_P4_PEBS_MATRIX_VERT	0x3F2
#define P4_PMV_REPLAY_TAG_BITS	0x00000003
#define P4_PMV_RESERVED		0xFFFFFFFC

#define P4_CCCR_OVF		0x80000000
#define P4_CCCR_CASCADE		0x40000000
#define P4_CCCR_OVF_PMI_T1	0x08000000
#define P4_CCCR_OVF_PMI_T0	0x04000000
#define P4_CCCR_FORCE_OVF	0x02000000
#define P4_CCCR_ACTIVE_THREAD	0x00030000
#define P4_CCCR_ENABLE		0x00001000
#define P4_CCCR_ESCR_SELECT(X)	(((X) >> 13) & 0x7)
#define P4_CCCR_EXTENDED_CASCADE	0x00000800
#define P4_CCCR_RESERVED	(0x300007FF|P4_CCCR_OVF|P4_CCCR_OVF_PMI_T1)

#define P4_ESCR_CPL_T1		0x00000003
#define P4_ESCR_CPL_T0		0x0000000C
#define P4_ESCR_TAG_ENABLE	0x00000010
#define P4_ESCR_RESERVED	(0x80000000)

#define P4_FAST_RDPMC		0x80000000
#define P4_MASK_FAST_RDPMC	0x0000001F	/* we only need low 5 bits */

#define rdmsr_low(msr,low) \
	__asm__ __volatile__("rdmsr" : "=a"(low) : "c"(msr) : "edx")
#define rdpmc_low(ctr,low) \
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

static void clear_msr_range(unsigned int base, unsigned int n)
{
	unsigned int i;

	for(i = 0; i < n; ++i)
		wrmsr(base+i, 0, 0);
}

static inline void set_in_cr4_local(unsigned int mask)
{
	write_cr4(read_cr4() | mask);
}

static inline void clear_in_cr4_local(unsigned int mask)
{
	write_cr4(read_cr4() & ~mask);
}

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

#if defined(CONFIG_X86_LOCAL_APIC)

static inline void perfctr_cpu_mask_interrupts(const struct per_cpu_cache *cache)
{
	__perfctr_cpu_mask_interrupts();
}

static inline void perfctr_cpu_unmask_interrupts(const struct per_cpu_cache *cache)
{
	__perfctr_cpu_unmask_interrupts();
}

#else	/* CONFIG_X86_LOCAL_APIC */
#define perfctr_cstatus_has_ictrs(cstatus)	0
#undef cpu_has_apic
#define cpu_has_apic				0
#undef apic_write
#define apic_write(reg,vector)			do{}while(0)
#endif	/* CONFIG_X86_LOCAL_APIC */

#if defined(CONFIG_SMP)

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

#else	/* CONFIG_SMP */
static inline void set_isuspend_cpu(struct perfctr_cpu_state *state, int cpu) { }
static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state, int cpu) { return 1; }
static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state) { }
#endif	/* CONFIG_SMP */

/****************************************************************
 *								*
 * Driver procedures.						*
 *								*
 ****************************************************************/

/*
 * Intel P5 family (Pentium, family code 5).
 * - One TSC and two 40-bit PMCs.
 * - A single 32-bit CESR (MSR 0x11) controls both PMCs.
 *   CESR has two halves, each controlling one PMC.
 *   To keep the API reasonably clean, the user puts 16 bits of
 *   control data in each counter's evntsel; the driver combines
 *   these to a single 32-bit CESR value.
 * - Overflow interrupts are not available.
 * - Pentium MMX added the RDPMC instruction. RDPMC has lower
 *   overhead than RDMSR and it can be used in user-mode code.
 * - The MMX events are not symmetric: some events are only available
 *   for some PMC, and some event codes denote different events
 *   depending on which PMCs they control.
 */

/* shared with MII and C6 */
static int p5_like_check_control(struct perfctr_cpu_state *state,
				 unsigned int reserved_bits, int is_c6)
{
	unsigned short cesr_half[2];
	unsigned int pmc, evntsel, i;

	if (state->control.nrictrs != 0 || state->control.nractrs > 2)
		return -EINVAL;
	cesr_half[0] = 0;
	cesr_half[1] = 0;
	for(i = 0; i < state->control.nractrs; ++i) {
		pmc = state->control.pmc_map[i];
		state->pmc[i].map = pmc;
		if (pmc > 1 || cesr_half[pmc] != 0)
			return -EINVAL;
		evntsel = state->control.evntsel[i];
		/* protect reserved bits */
		if ((evntsel & reserved_bits) != 0)
			return -EPERM;
		/* the CPL field (if defined) must be non-zero */
		if (!is_c6 && !(evntsel & P5_CESR_CPL))
			return -EINVAL;
		cesr_half[pmc] = evntsel;
	}
	state->k1.id = (cesr_half[1] << 16) | cesr_half[0];
	return 0;
}

static int p5_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	return p5_like_check_control(state, P5_CESR_RESERVED, 0);
}

/* shared with MII but not C6 */
static void p5_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cesr;

	cesr = state->k1.id;
	if (!cesr)	/* no PMC is on (this test doesn't work on C6) */
		return;
	cache = get_cpu_cache();
	if (cache->k1.p5_cesr != cesr) {
		cache->k1.p5_cesr = cesr;
		wrmsr(MSR_P5_CESR, cesr, 0);
	}
}

static void p5_read_counters(const struct perfctr_cpu_state *state,
			     struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	/* The P5 doesn't allocate a cache line on a write miss, so do
	   a dummy read to avoid a write miss here _and_ a read miss
	   later in our caller. */
	asm("" : : "r"(ctrs->tsc));

	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		rdmsr_low(MSR_P5_CTR0+pmc, ctrs->pmc[i]);
	}
}

/* used by all except pre-MMX P5 */
static void rdpmc_read_counters(const struct perfctr_cpu_state *state,
				struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if (perfctr_cstatus_has_tsc(cstatus))
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		rdpmc_low(pmc, ctrs->pmc[i]);
	}
}

/* shared with MII and C6 */
static const struct perfctr_msr_range p5_extras[] = {
	{ MSR_P5_CESR, 1+2 },
	{ 0, 0 },
};

static const struct perfctr_pmu_msrs p5_pmu_msrs = {
	.extras = p5_extras,
};

/*
 * Cyrix 6x86/MII/III.
 * - Same MSR assignments as P5 MMX. Has RDPMC and two 48-bit PMCs.
 * - Event codes and CESR formatting as in the plain P5 subset.
 * - Many but not all P5 MMX event codes are implemented.
 * - Cyrix adds a few more event codes. The event code is widened
 *   to 7 bits, and Cyrix puts the high bit in CESR bit 10
 *   (and CESR bit 26 for PMC1).
 */

static int mii_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	return p5_like_check_control(state, MII_CESR_RESERVED, 0);
}

/*
 * Centaur WinChip C6/2/3.
 * - Same MSR assignments as P5 MMX. Has RDPMC and two 40-bit PMCs.
 * - CESR is formatted with two halves, like P5. However, there
 *   are no defined control fields for e.g. CPL selection, and
 *   there is no defined method for stopping the counters.
 * - Only a few event codes are defined.
 * - The 64-bit TSC is synthesised from the low 32 bits of the
 *   two PMCs, and CESR has to be set up appropriately.
 *   Reprogramming CESR causes RDTSC to yield invalid results.
 *   (The C6 may also hang in this case, due to C6 erratum I-13.)
 *   Therefore, using the PMCs on any of these processors requires
 *   that the TSC is not accessed at all:
 *   1. The kernel must be configured or a TSC-less processor, i.e.
 *      generic 586 or less.
 *   2. The "notsc" boot parameter must be passed to the kernel.
 *   3. User-space libraries and code must also be configured and
 *      compiled for a generic 586 or less.
 */

#if !defined(CONFIG_X86_TSC)
static int c6_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	if (state->control.tsc_on)
		return -EINVAL;
	return p5_like_check_control(state, C6_CESR_RESERVED, 1);
}

static void c6_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cesr;

	if (perfctr_cstatus_nractrs(state->cstatus) == 0) /* no PMC is on */
		return;
	cache = get_cpu_cache();
	cesr = state->k1.id;
	if (cache->k1.p5_cesr != cesr) {
		cache->k1.p5_cesr = cesr;
		wrmsr(MSR_P5_CESR, cesr, 0);
	}
}
#endif	/* !CONFIG_X86_TSC */

/*
 * Intel P6 family (Pentium Pro, Pentium II, Pentium III, Pentium M,
 * Intel Core, Intel Core 2, Atom, and Core i7, including Xeon and Celeron versions.
 * - One TSC and two 40-bit PMCs.
 *   Core i7 has four 48-bit PMCs.
 * - One 32-bit EVNTSEL MSR for each PMC.
 * - EVNTSEL0 contains a global enable/disable bit.
 *   That bit is reserved in EVNTSEL1.
 *   On Core 2, Atom, and Core i7 each EVNTSEL has its own enable/disable bit.
 * - Each EVNTSEL contains a CPL field.
 * - Overflow interrupts are possible, but requires that the
 *   local APIC is available. Some Mobile P6s have no local APIC.
 * - The PMCs cannot be initialised with arbitrary values, since
 *   wrmsr fills the high bits by sign-extending from bit 31.
 * - Most events are symmetric, but a few are not.
 * - Core 2 adds three fixed-function counters. A single shared control
 *   register has the control bits (CPL:2 + PMI:1) for these counters.
 * - Initial Atoms appear to have one fixed-function counter.
 */

static int is_fam10h;
static int amd_is_multicore;		/* northbridge events need special care */
static int amd_is_k8_mc_RevE;
static cpumask_t amd_mc_core0_mask;	/* only these may use NB events */
static int p6_has_separate_enables;	/* affects EVNTSEL.ENable rules */
static unsigned int p6_nr_pmcs;	/* number of general-purpose counters */
static unsigned int p6_nr_ffcs;	/* number of fixed-function counters */
static unsigned int nhlm_nr_offcore_rsps; /* number of OFFCORE_RSP MSRs */

/* shared with K7 */
static int p6_like_check_control(struct perfctr_cpu_state *state, int is_k7, cpumask_t *cpumask)
{
	unsigned int evntsel, i, nractrs, nrctrs, pmc_mask, pmc;
	unsigned int core2_fixed_ctr_ctrl;
	unsigned int max_nrctrs;
	unsigned int amd_mc_nb_event_seen;

	max_nrctrs = is_k7 ? 4 : p6_nr_pmcs + p6_nr_ffcs;

	nractrs = state->control.nractrs;
	nrctrs = nractrs + state->control.nrictrs;
	if (nrctrs < nractrs || nrctrs > max_nrctrs)
		return -EINVAL;

	pmc_mask = 0;
	core2_fixed_ctr_ctrl = 0;	/* must be zero on CPUs != Core 2 */
	amd_mc_nb_event_seen = 0;
	for(i = 0; i < nrctrs; ++i) {
		pmc = state->control.pmc_map[i];
		state->pmc[i].map = pmc;
		/* pmc_map[i] is what we pass to RDPMC
		 * to check that pmc_map[] is well-defined on Core 2,
		 * we map FIXED_CTR 0x40000000+N to PMC p6_nr_pmcs+N
		 */
		if (!is_k7 && p6_nr_ffcs != 0) {
			if (pmc & CORE2_PMC_FIXED_FLAG)
				pmc = p6_nr_pmcs + (pmc & ~CORE2_PMC_FIXED_FLAG);
			else if (pmc >= p6_nr_pmcs)
				return -EINVAL;
		}
		if (pmc >= max_nrctrs || (pmc_mask & (1<<pmc)))
			return -EINVAL;
		pmc_mask |= (1<<pmc);
		/*
		 * check evntsel_high on AMD Fam10h
		 * on others we force it to zero (should return -EINVAL but
		 * having zeroes there has not been a requirement before)
		 */
		if (is_fam10h) {
			unsigned int evntsel_high = state->control.evntsel_high[i];
			if (evntsel_high & FAM10H_EVNTSEL_HIGH_RESERVED)
				return -EINVAL;
		} else
			state->control.evntsel_high[i] = 0;
		/* check evntsel */
		evntsel = state->control.evntsel[i];
		/* handle per-thread counting of AMD multicore northbridge events */
		if (cpumask != NULL && amd_is_multicore && IS_K8_NB_EVENT(evntsel)) {
			/* K8 RevE NB event erratum is incompatible with per-thread counters */
			if (amd_is_k8_mc_RevE)
				return -EPERM;
			/* remember to restrict this session to amd_mc_core0_mask */
			amd_mc_nb_event_seen = 1;
		}
		/* protect reserved bits */
		if (evntsel & P6_EVNTSEL_RESERVED)
			return -EPERM;
		/* check ENable bit */
		if (is_k7 || p6_has_separate_enables) {
			/* ENable bit must be set in each evntsel */
			if (!(evntsel & P6_EVNTSEL_ENABLE))
				return -EINVAL;
		} else {
			/* only evntsel[0] has the ENable bit */
			if (evntsel & P6_EVNTSEL_ENABLE) {
				if (pmc > 0)
					return -EPERM;
			} else {
				if (pmc == 0)
					return -EINVAL;
			}
		}
		/* the CPL field must be non-zero */
		if (!(evntsel & P6_EVNTSEL_CPL))
			return -EINVAL;
		/* INT bit must be off for a-mode and on for i-mode counters */
		if (evntsel & P6_EVNTSEL_INT) {
			if (i < nractrs)
				return -EINVAL;
		} else {
			if (i >= nractrs)
				return -EINVAL;
		}
		if (!is_k7 && p6_nr_ffcs != 0) {
			pmc = state->control.pmc_map[i];
			if (pmc & CORE2_PMC_FIXED_FLAG) {
				unsigned int ctl = 0;
				ctl |= ((evntsel >> 17) & 1) << 0;	/* CPL.OS */
				ctl |= ((evntsel >> 16) & 1) << 1;	/* CPL.USR */
				ctl |= ((evntsel >> 20) & 1) << 3;	/* INT/PMI */
				core2_fixed_ctr_ctrl |= ctl << (pmc & CORE2_PMC_FIXED_MASK) * 4;
			}
		}
	}
	/*
	 * check offcore_rsp[] on Intel Nehalem
	 * on others we force it to zero (should return -EINVAL but
	 * having zeroes there has not been a requirement before)
	 */
	for (i = 0; i < 2; ++i) {
		if (i < nhlm_nr_offcore_rsps) {
			unsigned int offcore_rsp = state->control.nhlm.offcore_rsp[i];
			if (offcore_rsp & OFFCORE_RSP_RESERVED)
				return -EINVAL;
		} else
			state->control.nhlm.offcore_rsp[i] = 0;
	}
	state->core2_fixed_ctr_ctrl = core2_fixed_ctr_ctrl;
	state->k1.id = new_id();
	if (amd_mc_nb_event_seen)
		*cpumask = amd_mc_core0_mask;
	return 0;
}

static int p6_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	return p6_like_check_control(state, 0, cpumask);
}

#ifdef CONFIG_X86_LOCAL_APIC
/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
/* shared with K7 and P4 */
static void p6_like_isuspend(struct perfctr_cpu_state *state,
			     unsigned int msr_evntsel0)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;
	unsigned int pending = 0;

	cpu = smp_processor_id();
	set_isuspend_cpu(state, cpu); /* early to limit cpu's live range */
	cache = __get_cpu_cache(cpu);
	perfctr_cpu_mask_interrupts(cache);
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	if (state->core2_fixed_ctr_ctrl & MSR_CORE_PERF_FIXED_CTR_CTRL_PMIANY) {
		cache->core2_fixed_ctr_ctrl = 0;
		wrmsr(MSR_CORE_PERF_FIXED_CTR_CTRL, 0, 0);
	}
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		unsigned int pmc_raw, pmc_idx, now;
		pmc_raw = state->pmc[i].map;
		if (!(pmc_raw & CORE2_PMC_FIXED_FLAG)) {
			/* Note: P4_MASK_FAST_RDPMC is a no-op for P6 and K7.
			   We don't need to make it into a parameter. */
			pmc_idx = pmc_raw & P4_MASK_FAST_RDPMC;
			cache->control.evntsel[pmc_idx] = 0;
			cache->control.evntsel_high[pmc_idx] = 0;
			/* On P4 this intensionally also clears the CCCR.OVF flag. */
			wrmsr(msr_evntsel0+pmc_idx, 0, 0);
		}
		/* P4 erratum N17 does not apply since we read only low 32 bits. */
		rdpmc_low(pmc_raw, now);
		state->pmc[i].sum += now - state->pmc[i].start;
		state->pmc[i].start = now;
		if ((int)now >= 0)
			++pending;
	}
	state->pending_interrupt = pending;
	/* cache->k1.id is still == state->k1.id */
}

/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
/* shared with K7 and P4 */
static void p6_like_iresume(const struct perfctr_cpu_state *state,
			    unsigned int msr_evntsel0,
			    unsigned int msr_perfctr0)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	cache = __get_cpu_cache(cpu);
	perfctr_cpu_unmask_interrupts(cache);
	if (cache->k1.id == state->k1.id) {
		cache->k1.id = 0; /* force reload of cleared EVNTSELs */
		if (is_isuspend_cpu(state, cpu))
			return; /* skip reload of PERFCTRs */
	}
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	/* If the control wasn't ours we must disable the
	   counters before reinitialising them. */
	if ((state->core2_fixed_ctr_ctrl & MSR_CORE_PERF_FIXED_CTR_CTRL_PMIANY) &&
	    cache->core2_fixed_ctr_ctrl != 0) {
		cache->core2_fixed_ctr_ctrl = 0;
		wrmsr(MSR_CORE_PERF_FIXED_CTR_CTRL, 0, 0);
	}
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		unsigned int pmc_raw = state->pmc[i].map;
		unsigned int msr_perfctr;
		unsigned int pmc_value_hi;

		if (pmc_raw & CORE2_PMC_FIXED_FLAG) {
			msr_perfctr = MSR_CORE_PERF_FIXED_CTR0 + (pmc_raw & CORE2_PMC_FIXED_MASK);
			/* Limit the value written to a fixed-function counter's MSR
			 * to 40 bits. Extraneous high bits cause GP faults on Model 23
			 * Core2s, while earlier processors would just ignore them.
			 */
			pmc_value_hi = 0xff;
		} else {
			/* Note: P4_MASK_FAST_RDPMC is a no-op for P6 and K7.
			   We don't need to make it into a parameter. */
			unsigned int pmc_idx = pmc_raw & P4_MASK_FAST_RDPMC;
			/* If the control wasn't ours we must disable the evntsels
			   before reinitialising the counters, to prevent unexpected
			   counter increments and missed overflow interrupts. */
			if (cache->control.evntsel[pmc_idx]) {
				cache->control.evntsel[pmc_idx] = 0;
				cache->control.evntsel_high[pmc_idx] = 0;
				wrmsr(msr_evntsel0+pmc_idx, 0, 0);
			}
			msr_perfctr = msr_perfctr0 + pmc_idx;
			pmc_value_hi = -1;
		}
		/* P4 erratum N15 does not apply since the CCCR is disabled. */
		wrmsr(msr_perfctr, state->pmc[i].start, pmc_value_hi);
	}
	/* cache->k1.id remains != state->k1.id */
}

static void p6_isuspend(struct perfctr_cpu_state *state)
{
	p6_like_isuspend(state, MSR_P6_EVNTSEL0);
}

static void p6_iresume(const struct perfctr_cpu_state *state)
{
	p6_like_iresume(state, MSR_P6_EVNTSEL0, MSR_P6_PERFCTR0);
}
#endif	/* CONFIG_X86_LOCAL_APIC */

/* shared with K7 and VC3 */
static void p6_like_write_control(const struct perfctr_cpu_state *state,
				  unsigned int msr_evntsel0)
{
	struct per_cpu_cache *cache;
	unsigned int nrctrs, i;

	cache = get_cpu_cache();
	if (cache->k1.id == state->k1.id)
		return;
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc, evntsel, evntsel_high;

		pmc = state->pmc[i].map;
		if (pmc & CORE2_PMC_FIXED_FLAG)
			continue;
		evntsel = state->control.evntsel[i];
		evntsel_high = state->control.evntsel_high[i];
		if (evntsel != cache->control.evntsel[pmc] ||
		    evntsel_high != cache->control.evntsel_high[pmc]) {
			cache->control.evntsel[pmc] = evntsel;
			cache->control.evntsel_high[pmc] = evntsel_high;
			wrmsr(msr_evntsel0+pmc, evntsel, evntsel_high);
		}
	}
	if (state->core2_fixed_ctr_ctrl != 0 &&
	    state->core2_fixed_ctr_ctrl != cache->core2_fixed_ctr_ctrl) {
		cache->core2_fixed_ctr_ctrl = state->core2_fixed_ctr_ctrl;
		wrmsr(MSR_CORE_PERF_FIXED_CTR_CTRL, state->core2_fixed_ctr_ctrl, 0);
	}
	for (i = 0; i < 2; ++i) {
		unsigned int offcore_rsp;
		
		offcore_rsp = state->control.nhlm.offcore_rsp[i];
		if (offcore_rsp != cache->nhlm_offcore_rsp[i]) {
			cache->nhlm_offcore_rsp[i] = offcore_rsp;
			wrmsr(MSR_OFFCORE_RSP0+i, offcore_rsp, 0);
		}
	}
	cache->k1.id = state->k1.id;
}

/* shared with VC3, Generic*/
static void p6_write_control(const struct perfctr_cpu_state *state)
{
	p6_like_write_control(state, MSR_P6_EVNTSEL0);
}

static struct perfctr_msr_range p6_perfctrs[] = {
	{ MSR_P6_PERFCTR0, 2 }, /* on Core i7 we'll update this count */
	{ 0, 0 },
};

static struct perfctr_msr_range p6_evntsels[] = {
	{ MSR_P6_EVNTSEL0, 2 }, /* on Core i7 we'll update this count */
	{ 0, 0 },
};

static const struct perfctr_pmu_msrs p6_pmu_msrs = {
	.perfctrs = p6_perfctrs,
	.evntsels = p6_evntsels,
};

static struct perfctr_msr_range core2_extras[] = {
	{ MSR_CORE_PERF_FIXED_CTR0, 3 }, /* on Atom we'll update this count */
	{ MSR_CORE_PERF_FIXED_CTR_CTRL, 1 },
	{ MSR_OFFCORE_RSP0, 0 },	/* on Nehalem we'll update this count */
	{ 0, 0 },
};

static void core2_clear_counters(int init)
{
	if (init) {
		unsigned int low, high;
		rdmsr(MSR_IA32_DEBUGCTLMSR, low, high);
		low &= ~MSR_IA32_DEBUGCTLMSR_FREEZE_PERFMON_ON_PMI;
		wrmsr(MSR_IA32_DEBUGCTLMSR, low, high);
		wrmsr(MSR_CORE_PERF_GLOBAL_CTRL, (1 << p6_nr_pmcs) - 1, (1 << p6_nr_ffcs) - 1);
	}
}

static const struct perfctr_pmu_msrs core2_pmu_msrs = {
	.perfctrs = p6_perfctrs,
	.evntsels = p6_evntsels,
	.extras = core2_extras,
	.clear_counters = core2_clear_counters,
};

/*
 * AMD K7 family (Athlon, Duron).
 * - Somewhat similar to the Intel P6 family.
 * - Four 48-bit PMCs.
 * - Four 32-bit EVNTSEL MSRs with similar layout as in P6.
 * - Completely different MSR assignments :-(
 * - Fewer countable events defined :-(
 * - The events appear to be completely symmetric.
 * - The EVNTSEL MSRs are symmetric since each has its own enable bit.
 * - Publicly available documentation is incomplete.
 * - K7 model 1 does not have a local APIC. AMD Document #22007
 *   Revision J hints that it may use debug interrupts instead.
 *
 * The K8 has the same hardware layout as the K7. It also has
 * better documentation and a different set of available events.
 *
 * AMD Family 10h is similar to the K7, but the EVNTSEL MSRs
 * have been widened to 64 bits.
 */

static int k7_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	return p6_like_check_control(state, 1, cpumask);
}

#ifdef CONFIG_X86_LOCAL_APIC
static void k7_isuspend(struct perfctr_cpu_state *state)
{
	p6_like_isuspend(state, MSR_K7_EVNTSEL0);
}

static void k7_iresume(const struct perfctr_cpu_state *state)
{
	p6_like_iresume(state, MSR_K7_EVNTSEL0, MSR_K7_PERFCTR0);
}
#endif	/* CONFIG_X86_LOCAL_APIC */

static void k7_write_control(const struct perfctr_cpu_state *state)
{
	p6_like_write_control(state, MSR_K7_EVNTSEL0);
}

static const struct perfctr_msr_range k7_perfctrs[] = {
	{ MSR_K7_PERFCTR0, 4 },
	{ 0, 0 },
};

static const struct perfctr_msr_range k7_evntsels[] = {
	{ MSR_K7_EVNTSEL0, 4 },
	{ 0, 0 },
};

static const struct perfctr_pmu_msrs k7_pmu_msrs = {
	.perfctrs = k7_perfctrs,
	.evntsels = k7_evntsels,
};

/*
 * VIA C3 family.
 * - A Centaur design somewhat similar to the P6/Celeron.
 * - PERFCTR0 is an alias for the TSC, and EVNTSEL0 is read-only.
 * - PERFCTR1 is 32 bits wide.
 * - EVNTSEL1 has no defined control fields, and there is no
 *   defined method for stopping the counter.
 * - According to testing, the reserved fields in EVNTSEL1 have
 *   no function. We always fill them with zeroes.
 * - Only a few event codes are defined.
 * - No local APIC or interrupt-mode support.
 * - pmc_map[0] must be 1, if nractrs == 1.
 */
static int vc3_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	if (state->control.nrictrs || state->control.nractrs > 1)
		return -EINVAL;
	if (state->control.nractrs == 1) {
		if (state->control.pmc_map[0] != 1)
			return -EINVAL;
		state->pmc[0].map = 1;
		if (state->control.evntsel[0] & VC3_EVNTSEL1_RESERVED)
			return -EPERM;
		state->k1.id = state->control.evntsel[0];
	} else
		state->k1.id = 0;
	return 0;
}

static void vc3_clear_counters(int init)
{
	/* Not documented, but seems to be default after boot. */
	wrmsr(MSR_P6_EVNTSEL0+1, 0x00070079, 0);
}

static const struct perfctr_pmu_msrs vc3_pmu_msrs = {
	.clear_counters = vc3_clear_counters,
};

/*
 * Intel Pentium 4.
 * Current implementation restrictions:
 * - No DS/PEBS support.
 *
 * Known quirks:
 * - OVF_PMI+FORCE_OVF counters must have an ireset value of -1.
 *   This allows the regular overflow check to also handle FORCE_OVF
 *   counters. Not having this restriction would lead to MAJOR
 *   complications in the driver's "detect overflow counters" code.
 *   There is no loss of functionality since the ireset value doesn't
 *   affect the counter's PMI rate for FORCE_OVF counters.
 * - In experiments with FORCE_OVF counters, and regular OVF_PMI
 *   counters with small ireset values between -8 and -1, it appears
 *   that the faulting instruction is subjected to a new PMI before
 *   it can complete, ad infinitum. This occurs even though the driver
 *   clears the CCCR (and in testing also the ESCR) and invokes a
 *   user-space signal handler before restoring the CCCR and resuming
 *   the instruction.
 */

/*
 * Table 15-4 in the IA32 Volume 3 manual contains a 18x8 entry mapping
 * from counter/CCCR number (0-17) and ESCR SELECT value (0-7) to the
 * actual ESCR MSR number. This mapping contains some repeated patterns,
 * so we can compact it to a 4x8 table of MSR offsets:
 *
 * 1. CCCRs 16 and 17 are mapped just like CCCRs 13 and 14, respectively.
 *    Thus, we only consider the 16 CCCRs 0-15.
 * 2. The CCCRs are organised in pairs, and both CCCRs in a pair use the
 *    same mapping. Thus, we only consider the 8 pairs 0-7.
 * 3. In each pair of pairs, the second odd-numbered pair has the same domain
 *    as the first even-numbered pair, and the range is 1+ the range of the
 *    the first even-numbered pair. For example, CCCR(0) and (1) map ESCR
 *    SELECT(7) to 0x3A0, and CCCR(2) and (3) map it to 0x3A1.
 *    The only exception is that pair (7) [CCCRs 14 and 15] does not have
 *    ESCR SELECT(3) in its domain, like pair (6) [CCCRs 12 and 13] has.
 *    NOTE: Revisions of IA32 Volume 3 older than #245472-007 had an error
 *    in this table: CCCRs 12, 13, and 16 had their mappings for ESCR SELECT
 *    values 2 and 3 swapped.
 * 4. All MSR numbers are on the form 0x3??. Instead of storing these as
 *    16-bit numbers, the table only stores the 8-bit offsets from 0x300.
 */

static const unsigned char p4_cccr_escr_map[4][8] = {
	/* 0x00 and 0x01 as is, 0x02 and 0x03 are +1 */
	[0x00/4] {	[7] 0xA0,
			[6] 0xA2,
			[2] 0xAA,
			[4] 0xAC,
			[0] 0xB2,
			[1] 0xB4,
			[3] 0xB6,
			[5] 0xC8, },
	/* 0x04 and 0x05 as is, 0x06 and 0x07 are +1 */
	[0x04/4] {	[0] 0xC0,
			[2] 0xC2,
			[1] 0xC4, },
	/* 0x08 and 0x09 as is, 0x0A and 0x0B are +1 */
	[0x08/4] {	[1] 0xA4,
			[0] 0xA6,
			[5] 0xA8,
			[2] 0xAE,
			[3] 0xB0, },
	/* 0x0C, 0x0D, and 0x10 as is,
	   0x0E, 0x0F, and 0x11 are +1 except [3] is not in the domain */
	[0x0C/4] {	[4] 0xB8,
			[5] 0xCC,
			[6] 0xE0,
			[0] 0xBA,
			[2] 0xBC,
			[3] 0xBE,
			[1] 0xCA, },
};

static unsigned int p4_escr_addr(unsigned int pmc, unsigned int cccr_val)
{
	unsigned int escr_select, pair, escr_offset;

	escr_select = P4_CCCR_ESCR_SELECT(cccr_val);
	if (pmc > 0x11)
		return 0;	/* pmc range error */
	if (pmc > 0x0F)
		pmc -= 3;	/* 0 <= pmc <= 0x0F */
	pair = pmc / 2;		/* 0 <= pair <= 7 */
	escr_offset = p4_cccr_escr_map[pair / 2][escr_select];
	if (!escr_offset || (pair == 7 && escr_select == 3))
		return 0;	/* ESCR SELECT range error */
	return escr_offset + (pair & 1) + 0x300;
};

static int p4_IQ_ESCR_ok;	/* only models <= 2 can use IQ_ESCR{0,1} */
static int p4_is_ht;		/* affects several CCCR & ESCR fields */
static int p4_extended_cascade_ok;	/* only models >= 2 can use extended cascading */

static int p4_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	unsigned int i, nractrs, nrctrs, pmc_mask;

	nractrs = state->control.nractrs;
	nrctrs = nractrs + state->control.nrictrs;
	if (nrctrs < nractrs || nrctrs > 18)
		return -EINVAL;

	pmc_mask = 0;
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc, cccr_val, escr_val, escr_addr;
		/* check that pmc_map[] is well-defined;
		   pmc_map[i] is what we pass to RDPMC, the PMC itself
		   is extracted by masking off the FAST_RDPMC flag */
		pmc = state->control.pmc_map[i] & ~P4_FAST_RDPMC;
		state->pmc[i].map = state->control.pmc_map[i];
		if (pmc >= 18 || (pmc_mask & (1<<pmc)))
			return -EINVAL;
		pmc_mask |= (1<<pmc);
		/* check CCCR contents */
		cccr_val = state->control.evntsel[i];
		if (cccr_val & P4_CCCR_RESERVED)
			return -EPERM;
		if (cccr_val & P4_CCCR_EXTENDED_CASCADE) {
			if (!p4_extended_cascade_ok)
				return -EPERM;
			if (!(pmc == 12 || pmc >= 15))
				return -EPERM;
		}
		if ((cccr_val & P4_CCCR_ACTIVE_THREAD) != P4_CCCR_ACTIVE_THREAD && !p4_is_ht)
			return -EINVAL;
		if (!(cccr_val & (P4_CCCR_ENABLE | P4_CCCR_CASCADE | P4_CCCR_EXTENDED_CASCADE)))
			return -EINVAL;
		if (cccr_val & P4_CCCR_OVF_PMI_T0) {
			if (i < nractrs)
				return -EINVAL;
			if ((cccr_val & P4_CCCR_FORCE_OVF) &&
			    state->control.ireset[i] != -1)
				return -EINVAL;
		} else {
			if (i >= nractrs)
				return -EINVAL;
		}
		/* check ESCR contents */
		escr_val = state->control.p4.escr[i];
		if (escr_val & P4_ESCR_RESERVED)
			return -EPERM;
		if ((escr_val & P4_ESCR_CPL_T1) && (!p4_is_ht || cpumask != NULL))
			return -EINVAL;
		/* compute and cache ESCR address */
		escr_addr = p4_escr_addr(pmc, cccr_val);
		if (!escr_addr)
			return -EINVAL;		/* ESCR SELECT range error */
		/* IQ_ESCR0 and IQ_ESCR1 only exist in models <= 2 */
		if ((escr_addr & ~0x001) == 0x3BA && !p4_IQ_ESCR_ok)
			return -EINVAL;
		/* XXX: Two counters could map to the same ESCR. Should we
		   check that they use the same ESCR value? */
		state->p4_escr_map[i] = escr_addr - MSR_P4_ESCR0;
	}
	/* check ReplayTagging control (PEBS_ENABLE and PEBS_MATRIX_VERT) */
	if (state->control.p4.pebs_enable) {
		if (!nrctrs)
			return -EPERM;
		if (state->control.p4.pebs_enable & P4_PE_RESERVED)
			return -EPERM;
		if (!(state->control.p4.pebs_enable & P4_PE_UOP_TAG))
			return -EINVAL;
		if (!(state->control.p4.pebs_enable & P4_PE_REPLAY_TAG_BITS))
			return -EINVAL;
		if (state->control.p4.pebs_matrix_vert & P4_PMV_RESERVED)
			return -EPERM;
		if (!(state->control.p4.pebs_matrix_vert & P4_PMV_REPLAY_TAG_BITS))
			return -EINVAL;
	} else if (state->control.p4.pebs_matrix_vert)
		return -EPERM;
	state->k1.id = new_id();
	if (nrctrs != 0 && cpumask != NULL)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,32)
		cpumask_complement(cpumask, &perfctr_cpus_forbidden_mask);
#else
		cpus_complement(*cpumask, perfctr_cpus_forbidden_mask);
#endif
	return 0;
}

#ifdef CONFIG_X86_LOCAL_APIC
static void p4_isuspend(struct perfctr_cpu_state *state)
{
	return p6_like_isuspend(state, MSR_P4_CCCR0);
}

static void p4_iresume(const struct perfctr_cpu_state *state)
{
	return p6_like_iresume(state, MSR_P4_CCCR0, MSR_P4_PERFCTR0);
}
#endif	/* CONFIG_X86_LOCAL_APIC */

static void p4_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int nrctrs, i;

	/* XXX: temporary debug check */
	if (cpu_isset(smp_processor_id(), perfctr_cpus_forbidden_mask) &&
	    perfctr_cstatus_nrctrs(state->cstatus))
		printk(KERN_ERR "%s: BUG! CPU %u is in the forbidden set\n",
		       __FUNCTION__, smp_processor_id());
	cache = get_cpu_cache();
	if (cache->k1.id == state->k1.id)
		return;
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int escr_val, escr_off, cccr_val, pmc;
		escr_val = state->control.p4.escr[i];
		escr_off = state->p4_escr_map[i];
		if (escr_val != cache->control.escr[escr_off]) {
			cache->control.escr[escr_off] = escr_val;
			wrmsr(MSR_P4_ESCR0+escr_off, escr_val, 0);
		}
		cccr_val = state->control.evntsel[i];
		pmc = state->pmc[i].map & P4_MASK_FAST_RDPMC;
		if (cccr_val != cache->control.evntsel[pmc]) {
			cache->control.evntsel[pmc] = cccr_val;
			wrmsr(MSR_P4_CCCR0+pmc, cccr_val, 0);
		}
	}
	if (state->control.p4.pebs_enable != cache->control.pebs_enable) {
		cache->control.pebs_enable = state->control.p4.pebs_enable;
		wrmsr(MSR_P4_PEBS_ENABLE, state->control.p4.pebs_enable, 0);
	}
	if (state->control.p4.pebs_matrix_vert != cache->control.pebs_matrix_vert) {
		cache->control.pebs_matrix_vert = state->control.p4.pebs_matrix_vert;
		wrmsr(MSR_P4_PEBS_MATRIX_VERT, state->control.p4.pebs_matrix_vert, 0);
	}
	cache->k1.id = state->k1.id;
}

static const struct perfctr_msr_range p4_perfctrs[] = {
	{ MSR_P4_PERFCTR0, 18 },
	{ 0, 0 },
};

static const struct perfctr_msr_range p4_evntsels[] = {
	{ 0x3BA, 2 },	/* IQ_ESCR{0,1}: only models <= 2 have them */
	{ 0x3A0, 26 },
	{ 0x3BC, 3 },
	{ 0x3C0, 6 },
	{ 0x3C8, 6 },
	{ 0x3E0, 2 },
	{ 0, 0 },
};

static const struct perfctr_msr_range p4_extras[] = {
	/* MSR 0x3F0 seems to have a default value of 0xFC00, but current
	   docs doesn't fully define it, so leave it alone for now. */
	/* PEBS_ENABLE and PEBS_MATRIX_VERT handle both PEBS and
	   ReplayTagging, and should exist even if PEBS is disabled */
	{ 0x3F1, 2 },
	{ MSR_P4_CCCR0, 18 },
	{ 0, 0 },
};

static const struct perfctr_pmu_msrs p4_pmu_msrs_models_0to2 = {
	.perfctrs = p4_perfctrs,
	.evntsels = p4_evntsels,
	.extras = p4_extras,
};

static const struct perfctr_pmu_msrs p4_pmu_msrs_models_3up = {
	.perfctrs = p4_perfctrs,
	.evntsels = p4_evntsels+1,
	.extras = p4_extras,
};

/*
 * Generic driver for any x86 with a working TSC.
 */

static int generic_check_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	if (state->control.nractrs || state->control.nrictrs)
		return -EINVAL;
	return 0;
}

/*
 * Driver methods, internal and exported.
 *
 * Frequently called functions (write_control, read_counters,
 * isuspend and iresume) are back-patched to invoke the correct
 * processor-specific methods directly, thereby saving the
 * overheads of indirect function calls.
 *
 * Backpatchable call sites must have been "finalised" after
 * initialisation. The reason for this is that unsynchronised code
 * modification doesn't work in multiprocessor systems, due to
 * Intel P6 errata. Consequently, all backpatchable call sites
 * must be known and local to this file.
 *
 * Backpatchable calls must initially be to 'noinline' stubs.
 * Otherwise the compiler may inline the stubs, which breaks
 * redirect_call() and finalise_backpatching().
 */

static int redirect_call_disable;

static noinline void redirect_call(void *ra, void *to)
{
	/* XXX: make this function __init later */
	if (redirect_call_disable)
		printk(KERN_ERR __FILE__ ":%s: unresolved call to %p at %p\n",
		       __FUNCTION__, to, ra);
	/* we can only redirect `call near relative' instructions */
	if (*((unsigned char*)ra - 5) != 0xE8) {
		printk(KERN_WARNING __FILE__ ":%s: unable to redirect caller %p to %p\n",
		       __FUNCTION__, ra, to);
		return;
	}
	*(int*)((char*)ra - 4) = (char*)to - (char*)ra;
}

static void (*write_control)(const struct perfctr_cpu_state*);
static noinline void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), write_control);
	return write_control(state);
}

static void (*read_counters)(const struct perfctr_cpu_state*,
			     struct perfctr_low_ctrs*);
static noinline void perfctr_cpu_read_counters(const struct perfctr_cpu_state *state,
					       struct perfctr_low_ctrs *ctrs)
{
	redirect_call(__builtin_return_address(0), read_counters);
	return read_counters(state, ctrs);
}

#ifdef CONFIG_X86_LOCAL_APIC
static void (*cpu_isuspend)(struct perfctr_cpu_state*);
static noinline void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), cpu_isuspend);
	return cpu_isuspend(state);
}

static void (*cpu_iresume)(const struct perfctr_cpu_state*);
static noinline void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), cpu_iresume);
	return cpu_iresume(state);
}

/* Call perfctr_cpu_ireload() just before perfctr_cpu_resume() to
   bypass internal caching and force a reload if the I-mode PMCs. */
void perfctr_cpu_ireload(struct perfctr_cpu_state *state)
{
#ifdef CONFIG_SMP
	clear_isuspend_cpu(state);
#else
	get_cpu_cache()->k1.id = 0;
#endif
}

/* PRE: the counters have been suspended and sampled by perfctr_cpu_suspend() */
static int lvtpc_reinit_needed;
unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nrctrs, pmc, pmc_mask;

	cstatus = state->cstatus;
	pmc = perfctr_cstatus_nractrs(cstatus);
	nrctrs = perfctr_cstatus_nrctrs(cstatus);

	state->pending_interrupt = 0;
	for(pmc_mask = 0; pmc < nrctrs; ++pmc) {
		if ((int)state->pmc[pmc].start >= 0) { /* XXX: ">" ? */
			/* XXX: "+=" to correct for overshots */
			state->pmc[pmc].start = state->control.ireset[pmc];
			pmc_mask |= (1 << pmc);
			/* On a P4 we should now clear the OVF flag in the
			   counter's CCCR. However, p4_isuspend() already
			   did that as a side-effect of clearing the CCCR
			   in order to stop the i-mode counters. */
		}
	}
	if (lvtpc_reinit_needed)
		apic_write(APIC_LVTPC, LOCAL_PERFCTR_VECTOR);
	return pmc_mask;
}

static inline int check_ireset(const struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i;

	i = state->control.nractrs;
	nrctrs = i + state->control.nrictrs;
	for(; i < nrctrs; ++i)
		if (state->control.ireset[i] >= 0)
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
	if (perfctr_cstatus_has_ictrs(state->cstatus))
		printk(KERN_ERR "perfctr/x86.c: BUG! updating control in"
		       " perfctr %p on cpu %u while it has cstatus %x"
		       " (pid %d, comm %s)\n",
		       state, smp_processor_id(), state->cstatus,
		       current->pid, current->comm);
#endif
}

#else	/* CONFIG_X86_LOCAL_APIC */
static inline void perfctr_cpu_isuspend(struct perfctr_cpu_state *state) { }
static inline void perfctr_cpu_iresume(const struct perfctr_cpu_state *state) { }
static inline int check_ireset(const struct perfctr_cpu_state *state) { return 0; }
static inline void setup_imode_start_values(struct perfctr_cpu_state *state) { }
static inline void debug_no_imode(const struct perfctr_cpu_state *state) { }
#endif	/* CONFIG_X86_LOCAL_APIC */

static int (*check_control)(struct perfctr_cpu_state*, cpumask_t*);
int perfctr_cpu_update_control(struct perfctr_cpu_state *state, cpumask_t *cpumask)
{
	int err;

	debug_no_imode(state);
	clear_isuspend_cpu(state);
	state->cstatus = 0;

	/* disallow i-mode counters if we cannot catch the interrupts */
	if (!(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
	    && state->control.nrictrs)
		return -EPERM;

	err = check_control(state, cpumask);
	if (err < 0)
		return err;
	err = check_ireset(state);
	if (err < 0)
		return err;
	state->cstatus = perfctr_mk_cstatus(state->control.tsc_on,
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
	/* perfctr_cpu_disable_rdpmc(); */	/* not for x86 */
}

void perfctr_cpu_resume(struct perfctr_cpu_state *state)
{
	if (perfctr_cstatus_has_ictrs(state->cstatus))
	    perfctr_cpu_iresume(state);
	/* perfctr_cpu_enable_rdpmc(); */	/* not for x86 or global-mode */
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

static const struct perfctr_pmu_msrs *pmu_msrs;

static void perfctr_cpu_clear_counters(int init)
{
	const struct perfctr_pmu_msrs *pmu;
	const struct perfctr_msr_range *msrs;
	int i;

	pmu = pmu_msrs;
	if (!pmu)
		return;

	/* The order below is significant: evntsels must be cleared
	   before the perfctrs. */
	msrs = pmu->evntsels;
	if (msrs)
		for(i = 0; msrs[i].first_msr; ++i)
			clear_msr_range(msrs[i].first_msr, msrs[i].nr_msrs);
	msrs = pmu->extras;
	if (msrs)
		for(i = 0; msrs[i].first_msr; ++i)
			clear_msr_range(msrs[i].first_msr, msrs[i].nr_msrs);
	msrs = pmu->perfctrs;
	if (msrs)
		for(i = 0; msrs[i].first_msr; ++i)
			clear_msr_range(msrs[i].first_msr, msrs[i].nr_msrs);
	if (pmu->clear_counters)
		(*pmu->clear_counters)(init);
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

static inline void clear_perfctr_cpus_forbidden_mask(void)
{
#if !defined(perfctr_cpus_forbidden_mask)
	cpus_clear(perfctr_cpus_forbidden_mask);
#endif
}

static inline void set_perfctr_cpus_forbidden_mask(cpumask_t mask)
{
#if !defined(perfctr_cpus_forbidden_mask)
	perfctr_cpus_forbidden_mask = mask;
#endif
}

/* see comment above at redirect_call() */
static void __init finalise_backpatching(void)
{
	struct per_cpu_cache *cache;
	struct perfctr_cpu_state state;
	cpumask_t old_mask;

	old_mask = perfctr_cpus_forbidden_mask;
	clear_perfctr_cpus_forbidden_mask();

	cache = get_cpu_cache();
	memset(cache, 0, sizeof *cache);
	memset(&state, 0, sizeof state);
	if (perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT) {
		state.cstatus = __perfctr_mk_cstatus(0, 1, 0, 0);
		perfctr_cpu_sample(&state);
		perfctr_cpu_resume(&state);
		perfctr_cpu_suspend(&state);
	}
	state.cstatus = 0;
	perfctr_cpu_sample(&state);
	perfctr_cpu_resume(&state);
	perfctr_cpu_suspend(&state);

	set_perfctr_cpus_forbidden_mask(old_mask);

	redirect_call_disable = 1;
}

#ifdef CONFIG_SMP

cpumask_t perfctr_cpus_forbidden_mask;

static inline unsigned int find_mask(unsigned int nrvals)
{
	unsigned int tmp = nrvals;
	unsigned int index_msb = 31;

	if (!tmp)
		return 0;
	while (!(tmp & (1<<31))) {
		tmp <<= 1;
		--index_msb;
	}
	if (nrvals & (nrvals - 1))
		++index_msb;
	return ~(~0 << index_msb);
}

static void __init p4_ht_mask_setup_cpu(void *forbidden)
{
	int cpu = smp_processor_id();
	unsigned int cpuid_maxlev;
	unsigned int cpuid1_ebx, cpuid1_edx, cpuid4_eax;
	unsigned int initial_APIC_ID;
	unsigned int max_cores_per_package;
	unsigned int max_lp_per_package;
	unsigned int max_lp_per_core;
	unsigned int smt_id;

	/*
	 * The following big chunk of code detects the current logical processor's
	 * SMT ID (thread number). This is quite complicated, see AP-485 and Volume 3
	 * of Intel's IA-32 Manual (especially section 7.10) for details.
	 */

	/* Ensure that CPUID reports all levels. */
	if (cpu_data(cpu).x86_model == 3) { /* >= 3? */
		unsigned int low, high;
		rdmsr(MSR_IA32_MISC_ENABLE, low, high);
		if (low & (1<<22)) { /* LIMIT_CPUID_MAXVAL */
			low &= ~(1<<22);
			wrmsr(MSR_IA32_MISC_ENABLE, low, high);
			printk(KERN_INFO "perfctr/x86.c: CPU %d: removed CPUID level limitation\n",
			       cpu);
		}
	}

	/* Find the highest standard CPUID level. */
	cpuid_maxlev = cpuid_eax(0);
	if (cpuid_maxlev < 1) {
		printk(KERN_INFO "perfctr/x86: CPU %d: impossibly low # of CPUID levels: %u\n",
		       cpu, cpuid_maxlev);
		return;
	}
	cpuid1_ebx = cpuid_ebx(1);
	cpuid1_edx = cpuid_edx(1);

	/* Find the initial (HW-assigned) APIC ID of this logical processor. */
	initial_APIC_ID = cpuid1_ebx >> 24;

	/* Find the max number of logical processors per physical processor package. */
	if (cpuid1_edx & (1 << 28))	/* HT is supported */
		max_lp_per_package = (cpuid1_ebx >> 16) & 0xFF;
	else				/* HT is not supported */
		max_lp_per_package = 1;
	
	/* Find the max number of processor cores per physical processor package. */
	if (cpuid_maxlev >= 4) {
		/* For CPUID level 4 we need a zero in ecx as input to CPUID, but
		   cpuid_eax() doesn't do that. So we resort to using cpuid_count()
		   with reference parameters and dummy outputs. */
		unsigned int dummy;
		cpuid_count(4, 0, &cpuid4_eax, &dummy, &dummy, &dummy);
		max_cores_per_package = (cpuid4_eax >> 26) + 1;
	} else {
		cpuid4_eax = 0;
		max_cores_per_package = 1;
	}

	max_lp_per_core = max_lp_per_package / max_cores_per_package;

	smt_id = initial_APIC_ID & find_mask(max_lp_per_core);

	printk(KERN_INFO "perfctr/x86.c: CPU %d: cpuid_ebx(1) 0x%08x, cpuid_edx(1) 0x%08x, cpuid_eax(4) 0x%08x, cpuid_maxlev %u, max_cores_per_package %u, SMT_ID %u\n",
	       cpu, cpuid1_ebx, cpuid1_edx, cpuid4_eax, cpuid_maxlev, max_cores_per_package, smt_id);

	/*
	 * Now (finally!) check the SMT ID. The CPU numbers for non-zero SMT ID
	 * threads are recorded in the forbidden set, to allow performance counter
	 * hardware resource conflicts between sibling threads to be prevented.
	 */
	if (smt_id != 0)
		/* We rely on cpu_set() being atomic! */
		cpu_set(cpu, *(cpumask_t*)forbidden);
}

static int __init p4_ht_smp_init(void)
{
	cpumask_t forbidden;
	unsigned int cpu;

	cpus_clear(forbidden);
	smp_call_function(p4_ht_mask_setup_cpu, &forbidden, 1);
	p4_ht_mask_setup_cpu(&forbidden);
	if (cpus_empty(forbidden))
		return 0;
	perfctr_cpus_forbidden_mask = forbidden;
	printk(KERN_INFO "perfctr/x86.c: hyper-threaded P4s detected:"
	       " restricting access for CPUs");
	for(cpu = 0; cpu < NR_CPUS; ++cpu)
		if (cpu_isset(cpu, forbidden))
			printk(" %u", cpu);
	printk("\n");
	return 0;
}
#else /* SMP */
#define p4_ht_smp_init()	(0)
#endif /* SMP */

static int __init p4_ht_init(void)
{
	unsigned int nr_siblings;

	if (!cpu_has_ht)
		return 0;
	nr_siblings = (cpuid_ebx(1) >> 16) & 0xFF;
	if (nr_siblings < 2)
		return 0;
	p4_is_ht = 1;	/* needed even in a UP kernel */
	return p4_ht_smp_init();
}

static int __init intel_p4_init(void)
{
	static char p4_name[] __initdata = "Intel P4";
	unsigned int misc_enable;

	/* Detect things that matter to the driver. */
	rdmsr_low(MSR_IA32_MISC_ENABLE, misc_enable);
	if (!(misc_enable & MSR_IA32_MISC_ENABLE_PERF_AVAIL))
		return -ENODEV;
	if (p4_ht_init() != 0)
		return -ENODEV;
	if (current_cpu_data.x86_model <= 2)
		p4_IQ_ESCR_ok = 1;
	if (current_cpu_data.x86_model >= 2)
		p4_extended_cascade_ok = 1;
	/* Detect and set up legacy cpu_type for user-space. */
	if (current_cpu_data.x86_model >= 3) {
		/* Model 3 removes IQ_ESCR{0,1} and adds one event. */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P4M3;
	} else if (current_cpu_data.x86_model >= 2) {
		/* Model 2 changed the ESCR Event Mask programming
		   details for several events. */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P4M2;
	} else {
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P4;
	}
	perfctr_set_tests_type(PTT_P4);
	perfctr_cpu_name = p4_name;
	read_counters = rdpmc_read_counters;
	write_control = p4_write_control;
	check_control = p4_check_control;
	if (current_cpu_data.x86_model <= 2)
		pmu_msrs = &p4_pmu_msrs_models_0to2;
	else
		pmu_msrs = &p4_pmu_msrs_models_3up;
#ifdef CONFIG_X86_LOCAL_APIC
	if (cpu_has_apic) {
		perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
		cpu_isuspend = p4_isuspend;
		cpu_iresume = p4_iresume;
		lvtpc_reinit_needed = 1;
	}
#endif
	return 0;
}

static int __init intel_p5_init(void)
{
	static char p5_name[] __initdata = "Intel P5";

	/* Detect things that matter to the driver. */
	if (cpu_has_mmx) {
		read_counters = rdpmc_read_counters;

		/* Avoid Pentium Erratum 74. */
		if (current_cpu_data.x86_model == 4 &&
		    (current_cpu_data.x86_mask == 4 ||
		     (current_cpu_data.x86_mask == 3 &&
		      ((cpuid_eax(1) >> 12) & 0x3) == 1)))
			perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	} else {
		perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
		read_counters = p5_read_counters;
	}
	/* Detect and set up legacy cpu_type for user-space. */
	if (cpu_has_mmx) {
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P5MMX;
	} else {
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P5;
	}
	perfctr_set_tests_type(PTT_P5);
	perfctr_cpu_name = p5_name;
	write_control = p5_write_control;
	check_control = p5_check_control;
	pmu_msrs = &p5_pmu_msrs;
	return 0;
}

static int __init intel_p6_init(void)
{
	static char p6_name[] __initdata = "Intel P6";
	static char core2_name[] __initdata = "Intel Core 2";
	static char atom_name[] __initdata = "Intel Atom";
	static char nhlm_name[] __initdata = "Intel Nehalem";
	static char wstmr_name[] __initdata = "Intel Westmere";
	unsigned int misc_enable;

	/*
	 * Post P4 family 6 models (Pentium M, Core, Core 2, Atom)
	 * have MISC_ENABLE.PERF_AVAIL like the P4.
	 */
	switch (current_cpu_data.x86_model) {
	case 9:		/* Pentium M */
	case 13:	/* Pentium M */
	case 14:	/* Core */
	case 15:	/* Core 2 */
	case 22:	/* Core 2 based Celeron model 16h */
	case 23:	/* Core 2 */
	case 26:	/* Nehalem: Core i7-900, Xeon 5500, Xeon 3500 */
	case 28:	/* Atom */
	case 29:	/* Core 2 based Xeon 7400 */
	case 30:	/* Nehalem: Core i7-800/i5-700, i7-900XM/i7-800M/i7-700M, Xeon 3400 */
	case 37:	/* Westmere: Core i5-600/i3-500/Pentium-G6950, i7-600M/i5-500M/i5-400M/i3-300M, Xeno L3406 */
	case 44:	/* Westmere: Core i7-980X (Gulftown), Xeon 5600, Xeon 3600 */
	case 46:	/* Nehalem: Xeon 7500 */
		rdmsr_low(MSR_IA32_MISC_ENABLE, misc_enable);
		if (!(misc_enable & MSR_IA32_MISC_ENABLE_PERF_AVAIL))
			return -ENODEV;
	}

	/*
	 * Core 2 made each EVNTSEL have its own ENable bit,
	 * and added three fixed-function counters.
	 * On Atom cpuid tells us the number of fixed-function counters.
	 * Core i7 extended the number of PMCs to four.
	 */
	p6_nr_pmcs = 2;
	switch (current_cpu_data.x86_model) {
	case 15:	/* Core 2 */
	case 22:	/* Core 2 based Celeron model 16h */
	case 23:	/* Core 2 */
	case 29:	/* Core 2 based Xeon 7400 */
		perfctr_cpu_name = core2_name;
		p6_has_separate_enables = 1;
		p6_nr_ffcs = 3;
		break;
	case 26:	/* Nehalem: Core i7-900, Xeon 5500, Xeon 3500 */
	case 30:	/* Nehalem: Core i7-800/i5-700, i7-900XM/i7-800M/i7-700M, Xeon 3400 */
	case 46:	/* Nehalem: Xeon 7500 */
		perfctr_cpu_name = nhlm_name;
		p6_has_separate_enables = 1;
		p6_nr_ffcs = 3;
		p6_nr_pmcs = 4;
		nhlm_nr_offcore_rsps = 1;
		break;
	case 37:	/* Westmere: Core i5-600/i3-500/Pentium-G6950, i7-600M/i5-500M/i5-400M/i3-300M, Xeon L3406 */
	case 44:	/* Westmere: Core i7-980X (Gulftown), Xeon 5600, Xeon 3600 */
		perfctr_cpu_name = wstmr_name;
		p6_has_separate_enables = 1;
		p6_nr_ffcs = 3;
		p6_nr_pmcs = 4;
		/* Westmere adds MSR_OFFCORE_RSP1 and drops some events */
		nhlm_nr_offcore_rsps = 2;
		break;
	case 28: {	/* Atom */
		unsigned int maxlev, eax, ebx, dummy, edx;

		perfctr_cpu_name = atom_name;
		p6_has_separate_enables = 1;

		maxlev = cpuid_eax(0);
		if (maxlev < 0xA) {
			printk(KERN_WARNING "%s: cpuid[0].eax == %u, unable to query 0xA leaf\n",
			       __FUNCTION__, maxlev);
			return -EINVAL;
		}
		cpuid(0xA, &eax, &ebx, &dummy, &edx);
		/* ensure we have at least APM V2 with 2 40-bit general-purpose counters */
		if ((eax & 0xff) < 2 ||
		    ((eax >> 8) & 0xff) != 2 ||
		    ((eax >> 16) & 0xff) < 40) {
			printk(KERN_WARNING "%s: cpuid[0xA].eax == 0x%08x appears bogus\n",
			       __FUNCTION__, eax);
			return -EINVAL;
		}
		/* extract the number of fixed-function counters: Core2 has 3,
		   and initial Atoms appear to have 1; play it safe and reject
		   excessive values */
		p6_nr_ffcs = edx & 0x1f;
		if (p6_nr_ffcs > 3) {
			printk(KERN_WARNING "%s: cpuid[0xA] == { edx == 0x%08x, "
			       "eax == 0x%08x } appears bogus\n",
			       __FUNCTION__, edx, eax);
			p6_nr_ffcs = 0;
		}
		break;
	}
	default:
		perfctr_cpu_name = p6_name;
		break;
	}

	/*
	 * Avoid Pentium Pro Erratum 26.
	 */
	if (current_cpu_data.x86_model < 3) {	/* Pentium Pro */
		if (current_cpu_data.x86_mask < 9)
			perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	}

	/*
	 * Detect and set up legacy cpu_type for user-space.
	 */
	switch (current_cpu_data.x86_model) {
	default:
		printk(KERN_WARNING __FILE__ "%s: unknown model %u processor, "
		       "please report this to perfctr-devel or mikpe@it.uu.se\n",
		       __FUNCTION__, current_cpu_data.x86_model);
		/*FALLTHROUGH*/
	case 0:		/* Pentium Pro A-step */
	case 1:		/* Pentium Pro */
	case 4:		/* Pentium Pro based P55CT overdrive for P54 */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P6;
		break;
	case 3:		/* Pentium II or PII-based overdrive for PPro */
	case 5:		/* Pentium II */
	case 6:		/* Pentium II */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_PII;
		break;
	case 7:		/* Pentium III */
	case 8:		/* Pentium III */
	case 10:	/* Pentium III Xeon model A */
	case 11:	/* Pentium III */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_PIII;
		break;
	case 9:		/* Pentium M */
	case 13:	/* Pentium M */
		/* Erratum Y3 probably does not apply since we
		   read only the low 32 bits. */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_PENTM;
		break;
	case 14:	/* Core */
		/* XXX: what about erratum AE19? */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_CORE;
		break;
	case 15:	/* Core 2 */
	case 22:	/* Core 2 based Celeron model 16h */
	case 23:	/* Core 2 */
	case 29:	/* Core 2 based Xeon 7400 */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_CORE2;
		break;
	case 26:	/* Nehalem: Core i7-900, Xeon 5500, Xeon 3500 */
	case 30:	/* Nehalem: Core i7-800/i5-700, i7-900XM/i7-800M/i7-700M, Xeon 3400 */
	case 46:	/* Nehalem: Xeon 7500 */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_NHLM;
		break;
	case 37:	/* Westmere: Core i5-600/i3-500/Pentium-G6950, i7-600M/i5-500M/i5-400M/i3-300M, Xeon L3406 */
	case 44:	/* Westmere: Core i7-980X (Gulftown), Xeon 5600, Xeon 3600 */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_WSTMR;
		break;
	case 28:	/* Atom */
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_ATOM;
		break;
	}

	perfctr_set_tests_type(p6_nr_ffcs != 0 ? PTT_CORE2 : PTT_P6);
	read_counters = rdpmc_read_counters;
	write_control = p6_write_control;
	check_control = p6_check_control;
	p6_perfctrs[0].nr_msrs = p6_nr_pmcs;
	p6_evntsels[0].nr_msrs = p6_nr_pmcs;
	core2_extras[0].nr_msrs = p6_nr_ffcs;
	core2_extras[2].nr_msrs = nhlm_nr_offcore_rsps;
	pmu_msrs = p6_nr_ffcs != 0 ? &core2_pmu_msrs : &p6_pmu_msrs;

#ifdef CONFIG_X86_LOCAL_APIC
	if (cpu_has_apic) {
		perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
		cpu_isuspend = p6_isuspend;
		cpu_iresume = p6_iresume;
		/*
		 * Post P4 family 6 models (Pentium M, Core, Core 2, Atom)
		 * have LVTPC auto-masking like the P4.
		 */
		switch (current_cpu_data.x86_model) {
		case 9:		/* Pentium M */
		case 13:	/* Pentium M */
		case 14:	/* Core */
		case 15:	/* Core 2 */
		case 22:	/* Core 2 based Celeron model 16h */
		case 23:	/* Core 2 */
		case 26:	/* Nehalem: Core i7-900, Xeon 5500, Xeon 3500 */
		case 28:	/* Atom */
		case 29:	/* Core 2 based Xeon 7400 */
		case 30:	/* Nehalem: Core i7-800/i5-700, i7-900XM/i7-800M/i7-700M, Xeon 3400 */
		case 37:	/* Westmere: Core i5-600/i3-500/Pentium-G6950, i7-600M/i5-500M/i5-400M/i3-300M, Xeon L3406 */
		case 44:	/* Westmere: Core i7-980X (Gulftown), Xeon 5600, Xeon 3600 */
		case 46:	/* Nehalem: Xeon 7500 */
			lvtpc_reinit_needed = 1;
		}
	}
#endif

	return 0;
}

static int __init intel_init(void)
{
	if (!cpu_has_tsc)
		return -ENODEV;
	switch (current_cpu_data.x86) {
	case 5:
		return intel_p5_init();
	case 6:
		return intel_p6_init();
	case 15:
		return intel_p4_init();
	}
	return -ENODEV;
}

/*
 * Multicore K8s have issues with northbridge events:
 * 1. The NB is shared between the cores, so two different cores
 *    in the same node cannot count NB events simultaneously.
 *    This is handled by using a cpumask to restrict NB-using
 *    threads to core0 of all processors.
 * 2. The initial multicore chips (Revision E) have an erratum
 *    which causes the NB counters to be reset when either core
 *    reprograms its evntsels (even for non-NB events).
 *    This is only an issue because of scheduling of threads, so
 *    we restrict NB events to the non thread-centric API.
 */
#ifdef CONFIG_SMP
struct amd_mc_init_data {
	atomic_t non0core_seen;
	cpumask_t core0_mask;
};

static void __init amd_mc_init_cpu(void *data)
{
	int cpu = smp_processor_id();
	unsigned int apic_core_id_size, core_id;
	struct amd_mc_init_data *amd_mc_init_data = data;

	if ((cpuid_edx(1) & (1<<28)) == 0 ||		/* HTT is off */
	    cpuid_eax(0x80000000) < 0x80000008)	{	/* no Core Count info */
		/* each processor is single-core */
		apic_core_id_size = 0;
	} else {
		unsigned int ecx = cpuid_ecx(0x80000008);

		apic_core_id_size = (ecx >> 12) & 0xF; /* XXX: resvd in early CPUs */
		if (apic_core_id_size == 0) {
			unsigned int max_cores = (ecx & 0xFF) + 1;

			while ((1 << apic_core_id_size) < max_cores)
				++apic_core_id_size;
		}
	}

	core_id = (cpuid_ebx(1) >> 24) & ((1 << apic_core_id_size) - 1);
	printk(KERN_INFO "%s: cpu %d core_id %u\n", __FUNCTION__, cpu, core_id);

	if (core_id != 0) {
		atomic_set(&amd_mc_init_data->non0core_seen, 1);
	} else {
		/* We rely on cpu_set() being atomic! */
		cpu_set(cpu, amd_mc_init_data->core0_mask);
	}
}

static int __init amd_multicore_init(void)
{
	struct amd_mc_init_data amd_mc_init_data;

	atomic_set(&amd_mc_init_data.non0core_seen, 0);
	cpus_clear(amd_mc_init_data.core0_mask);

	smp_call_function(amd_mc_init_cpu, &amd_mc_init_data, 1);
	amd_mc_init_cpu(&amd_mc_init_data);

	if (atomic_read(&amd_mc_init_data.non0core_seen) == 0) {
		printk(KERN_INFO "%s: !non0core_seen\n", __FUNCTION__);
		return 0;
	}
#if 1	/* XXX: temporary sanity check, should be impossible */
	if (cpus_empty(amd_mc_init_data.core0_mask)) {
		printk(KERN_ERR "%s: Error: cpus_empty(core0_mask)\n", __FUNCTION__);
		return -ENODEV;
	}
#endif

	amd_is_multicore = 1;
	if (current_cpu_data.x86 == 15 &&
	    current_cpu_data.x86_model >= 0x20 &&
	    current_cpu_data.x86_model < 0x40) {
		amd_is_k8_mc_RevE = 1;
		printk(KERN_INFO "perfctr/x86.c: multi-core K8 RevE detected:"
		       " restricting access to northbridge events\n");
	} else {
		amd_mc_core0_mask = amd_mc_init_data.core0_mask;
		printk(KERN_INFO "perfctr/x86.c: multi-core AMDs detected:"
		       " forcing northbridge events to core0 CPUs\n");
	}
	return 0;
}
#else	/* CONFIG_SMP */
#define amd_multicore_init()	(0)
#endif	/* CONFIG_SMP */

static int __init amd_init(void)
{
	static char amd_name[] __initdata = "AMD K7/K8/Fam10h/Fam11h";

	if (!cpu_has_tsc)
		return -ENODEV;
	switch (current_cpu_data.x86) {
	case 6: /* K7 */
		perfctr_info.cpu_type = PERFCTR_X86_AMD_K7;
		break;
	case 15: /* K8. Like a K7 with a different event set. */
		if ((current_cpu_data.x86_model > 5) ||
		    (current_cpu_data.x86_model >= 4 && current_cpu_data.x86_mask >= 8)) {
			perfctr_info.cpu_type = PERFCTR_X86_AMD_K8C;
		} else {
			perfctr_info.cpu_type = PERFCTR_X86_AMD_K8;
		}
		if (amd_multicore_init() < 0)
			return -ENODEV;
		break;
	case 16:
	case 17:
		is_fam10h = 1;
		perfctr_info.cpu_type = PERFCTR_X86_AMD_FAM10H;
		if (amd_multicore_init() < 0)
			return -ENODEV;
		break;
	default:
		return -ENODEV;
	}
	perfctr_set_tests_type(PTT_AMD);
	perfctr_cpu_name = amd_name;
	read_counters = rdpmc_read_counters;
	write_control = k7_write_control;
	check_control = k7_check_control;
	pmu_msrs = &k7_pmu_msrs;
#ifdef CONFIG_X86_LOCAL_APIC
	if (cpu_has_apic) {
		perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
		cpu_isuspend = k7_isuspend;
		cpu_iresume = k7_iresume;
	}
#endif
	return 0;
}

static int __init cyrix_init(void)
{
	static char mii_name[] __initdata = "Cyrix 6x86MX/MII/III";
	if (!cpu_has_tsc)
		return -ENODEV;
	switch (current_cpu_data.x86) {
	case 6:	/* 6x86MX, MII, or III */
		perfctr_info.cpu_type = PERFCTR_X86_CYRIX_MII;
		perfctr_set_tests_type(PTT_P5);
		perfctr_cpu_name = mii_name;
		read_counters = rdpmc_read_counters;
		write_control = p5_write_control;
		check_control = mii_check_control;
		pmu_msrs = &p5_pmu_msrs;
		return 0;
	}
	return -ENODEV;
}

static int __init centaur_init(void)
{
#if !defined(CONFIG_X86_TSC)
	static char winchip_name[] __initdata = "WinChip C6/2/3";
#endif
	static char vc3_name[] __initdata = "VIA C3";
	switch (current_cpu_data.x86) {
#if !defined(CONFIG_X86_TSC)
	case 5:
		switch (current_cpu_data.x86_model) {
		case 4: /* WinChip C6 */
			perfctr_info.cpu_type = PERFCTR_X86_WINCHIP_C6;
			break;
		case 8: /* WinChip 2, 2A, or 2B */
		case 9: /* WinChip 3, a 2A with larger cache and lower voltage */
			perfctr_info.cpu_type = PERFCTR_X86_WINCHIP_2;
			break;
		default:
			return -ENODEV;
		}
		perfctr_set_tests_type(PTT_WINCHIP);
		perfctr_cpu_name = winchip_name;
		/*
		 * TSC must be inaccessible for perfctrs to work.
		 */
		if (!(read_cr4() & X86_CR4_TSD) || cpu_has_tsc)
			return -ENODEV;
		perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDTSC;
		read_counters = rdpmc_read_counters;
		write_control = c6_write_control;
		check_control = c6_check_control;
		pmu_msrs = &p5_pmu_msrs;
		return 0;
#endif	/* !CONFIG_X86_TSC */
	case 6: /* VIA C3 */
		if (!cpu_has_tsc)
			return -ENODEV;
		switch (current_cpu_data.x86_model) {
		case 6:	/* Cyrix III */
		case 7:	/* Samuel 2, Ezra (steppings >= 8) */
		case 8:	/* Ezra-T */
		case 9: /* Antaur/Nehemiah */
			break;
		default:
			return -ENODEV;
		}
		perfctr_info.cpu_type = PERFCTR_X86_VIA_C3;
		perfctr_set_tests_type(PTT_VC3);
		perfctr_cpu_name = vc3_name;
		read_counters = rdpmc_read_counters;
		write_control = p6_write_control;
		check_control = vc3_check_control;
		pmu_msrs = &vc3_pmu_msrs;
		return 0;
	}
	return -ENODEV;
}

static int __init generic_init(void)
{
	static char generic_name[] __initdata = "Generic x86 with TSC";
	if (!cpu_has_tsc)
		return -ENODEV;
	perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	perfctr_info.cpu_type = PERFCTR_X86_GENERIC;
	perfctr_set_tests_type(PTT_GENERIC);
	perfctr_cpu_name = generic_name;
	check_control = generic_check_control;
	write_control = p6_write_control;
	read_counters = rdpmc_read_counters;
	pmu_msrs = NULL;
	return 0;
}

static void perfctr_cpu_invalidate_cache(void)
{
	struct per_cpu_cache *cache = get_cpu_cache();

	/*
	 * per_cpu_cache[] is initialised to contain "impossible"
	 * evntsel values guaranteed to differ from anything accepted
	 * by perfctr_cpu_update_control().
	 * All-bits-one works for all currently supported processors.
	 * The memset also sets the ids to -1, which is intentional.
	 */
	memset(cache, ~0, sizeof(struct per_cpu_cache));

	/*
	 * To ensure that MSR_CORE_PERF_FIXED_CTR_CTRL is not written
	 * to on processors that do not have it, each CPU cache must
	 * indicate that it has an all-bits-zero value.
	 */
	cache->core2_fixed_ctr_ctrl = 0;

	/*
	 * To ensure that MSR_OFFCORE_RSP[0..1] are not written to
	 * on processors that do not have them, each CPU cache must
	 * indicate that they have all-bits-zero values.
	 */
	cache->nhlm_offcore_rsp[0] = 0;
	cache->nhlm_offcore_rsp[1] = 0;
}

static void perfctr_cpu_init_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters(1);
	perfctr_cpu_invalidate_cache();
	if (cpu_has_apic)
		apic_write(APIC_LVTPC, LOCAL_PERFCTR_VECTOR);
	if (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		set_in_cr4_local(X86_CR4_PCE);
}

static void perfctr_cpu_exit_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters(0);
	perfctr_cpu_invalidate_cache();
	if (cpu_has_apic)
		apic_write(APIC_LVTPC, APIC_DM_NMI | APIC_LVT_MASKED);
	if (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		clear_in_cr4_local(X86_CR4_PCE);
}

#if defined(CONFIG_X86_LOCAL_APIC) && defined(CONFIG_PM)

static void perfctr_pm_suspend(void)
{
	/* XXX: clear control registers */
	printk("perfctr/x86: PM suspend\n");
}

static void perfctr_pm_resume(void)
{
	/* XXX: reload control registers */
	printk("perfctr/x86: PM resume\n");
}

#include <linux/sysdev.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,14)
typedef pm_message_t perfctr_suspend_state_t;
#else
typedef u32 perfctr_suspend_state_t;
#endif

static int perfctr_device_suspend(struct sys_device *dev,
				  perfctr_suspend_state_t state)
{
	perfctr_pm_suspend();
	return 0;
}

static int perfctr_device_resume(struct sys_device *dev)
{
	perfctr_pm_resume();
	return 0;
}

static struct sysdev_class perfctr_sysclass = {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,25)
	.name		= "perfctr",
#else
	set_kset_name("perfctr"),
#endif
	.resume		= perfctr_device_resume,
	.suspend	= perfctr_device_suspend,
};

static struct sys_device device_perfctr = {
	.id	= 0,
	.cls	= &perfctr_sysclass,
};

static void x86_pm_init(void)
{
	if (sysdev_class_register(&perfctr_sysclass) == 0)
		sysdev_register(&device_perfctr);
}

static void x86_pm_exit(void)
{
	sysdev_unregister(&device_perfctr);
	sysdev_class_unregister(&perfctr_sysclass);
}

#else	/* CONFIG_X86_LOCAL_APIC && CONFIG_PM */

static inline void x86_pm_init(void) { }
static inline void x86_pm_exit(void) { }

#endif	/* CONFIG_X86_LOCAL_APIC && CONFIG_PM */

#ifdef CONFIG_X86_LOCAL_APIC

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,6)
static int reserve_lapic_nmi(void)
{
	int ret = 0;
	if (nmi_perfctr_msr) {
		nmi_perfctr_msr = 0;
		disable_lapic_nmi_watchdog();
		ret = 1;
	}
	return ret;
}

static inline void release_lapic_nmi(void) { }
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,19)

static void perfctr_release_perfctr_range(unsigned int first_msr, unsigned int nr_msrs)
{
	unsigned int i;

	for(i = 0; i < nr_msrs; ++i)
		release_perfctr_nmi(first_msr + i);
}

static int perfctr_reserve_perfctr_range(unsigned int first_msr, unsigned int nr_msrs)
{
	unsigned int i;

	for(i = 0; i < nr_msrs; ++i)
		if (!reserve_perfctr_nmi(first_msr + i)) {
			printk(KERN_ERR "perfctr/x86.c: failed to reserve perfctr MSR %#x\n",
			       first_msr + i);
			perfctr_release_perfctr_range(first_msr, i);
			return -1;
		}
	return 0;
}

static void perfctr_release_evntsel_range(unsigned int first_msr, unsigned int nr_msrs)
{
	unsigned int i;

	for(i = 0; i < nr_msrs; ++i)
		release_evntsel_nmi(first_msr + i);
}

static int perfctr_reserve_evntsel_range(unsigned int first_msr, unsigned int nr_msrs)
{
	unsigned int i;

	for(i = 0; i < nr_msrs; ++i)
		if (!reserve_evntsel_nmi(first_msr + i)) {
			printk(KERN_ERR "perfctr/x86.c: failed to reserve evntsel MSR %#x\n",
			       first_msr + i);
			perfctr_release_evntsel_range(first_msr, i);
			return -1;
		}
	return 0;
}

static void perfctr_release_counters_cpu(void *ignore)
{
	const struct perfctr_pmu_msrs *pmu;
	const struct perfctr_msr_range *msrs;
	int i;

	pmu = pmu_msrs;
	if (!pmu)
		return;
	msrs = pmu->perfctrs;
	if (msrs)
		for(i = 0; msrs[i].first_msr; ++i)
			perfctr_release_perfctr_range(msrs[i].first_msr, msrs[i].nr_msrs);
	msrs = pmu->evntsels;
	if (msrs)
		for(i = 0; msrs[i].first_msr; ++i)
			perfctr_release_evntsel_range(msrs[i].first_msr, msrs[i].nr_msrs);
}

static void perfctr_release_counters(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,21)
	perfctr_release_counters_cpu(NULL);
#else
	on_each_cpu(perfctr_release_counters_cpu, NULL, 1);
#endif
}

static void perfctr_reserve_counters_cpu(void *error)
{
	const struct perfctr_pmu_msrs *pmu;
	const struct perfctr_msr_range *msrs;
	int i;

	pmu = pmu_msrs;
	if (!pmu)
		return;
	msrs = pmu->perfctrs;
	if (msrs) {
		for(i = 0; msrs[i].first_msr; ++i)
			if (perfctr_reserve_perfctr_range(msrs[i].first_msr, msrs[i].nr_msrs))
				goto err_perfctrs;
	}
	msrs = pmu->evntsels;
	if (msrs) {
		for(i = 0; msrs[i].first_msr; ++i)
			if (perfctr_reserve_evntsel_range(msrs[i].first_msr, msrs[i].nr_msrs))
				goto err_evntsels;
	}
	return;

 err_evntsels:
	while (--i >= 0)
		perfctr_release_evntsel_range(msrs[i].first_msr, msrs[i].nr_msrs);

	msrs = pmu->perfctrs;
	if (!msrs)
		goto err;
	for(i = 0; msrs[i].first_msr; ++i)
		;
 err_perfctrs:
	while (--i >= 0)
		perfctr_release_perfctr_range(msrs[i].first_msr, msrs[i].nr_msrs);
 err:
	atomic_set((atomic_t*)error, -1);
}

static int perfctr_reserve_counters(void)
{
	atomic_t error = ATOMIC_INIT(0);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,21)
	perfctr_reserve_counters_cpu(&error);
#else
	on_each_cpu(perfctr_reserve_counters_cpu, &error, 1);
#endif
	return atomic_read(&error);
}

static int reserve_lapic_nmi(void)
{
	if (nmi_watchdog != NMI_LOCAL_APIC)
		return 0;
	if (atomic_read(&nmi_active) <= 0)
		return 0;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
	disable_lapic_nmi_watchdog();
#else
	on_each_cpu(stop_apic_nmi_watchdog, NULL, 1);
#endif
	return perfctr_reserve_counters();
}

static void release_lapic_nmi(void)
{
	perfctr_release_counters();
	if (nmi_watchdog != NMI_LOCAL_APIC)
		return;
	if (atomic_read(&nmi_active) != 0)
		return;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
	enable_lapic_nmi_watchdog();
#else
	on_each_cpu(setup_apic_nmi_watchdog, NULL, 1);
#endif
}
#endif	/* LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,19) */

#else	/* CONFIG_X86_LOCAL_APIC */
static inline int reserve_lapic_nmi(void) { return 0; }
static inline void release_lapic_nmi(void) { }
#endif	/* CONFIG_X86_LOCAL_APIC */

static void __init do_init_tests(void)
{
#ifdef CONFIG_PERFCTR_INIT_TESTS
	if (reserve_lapic_nmi() >= 0) {
		perfctr_x86_init_tests();
		release_lapic_nmi();
	}
#endif
}

int __init perfctr_cpu_init(void)
{
	int err = -ENODEV;

	preempt_disable();

	/* RDPMC and RDTSC are on by default. They will be disabled
	   by the init procedures if necessary. */
	perfctr_info.cpu_features = PERFCTR_FEATURE_RDPMC | PERFCTR_FEATURE_RDTSC;

	if (cpu_has_msr) {
		switch (current_cpu_data.x86_vendor) {
		case X86_VENDOR_INTEL:
			err = intel_init();
			break;
		case X86_VENDOR_AMD:
			err = amd_init();
			break;
		case X86_VENDOR_CYRIX:
			err = cyrix_init();
			break;
		case X86_VENDOR_CENTAUR:
			err = centaur_init();
		}
	}
	if (err) {
		err = generic_init();	/* last resort */
		if (err)
			goto out;
	}
	do_init_tests();
	finalise_backpatching();

	perfctr_info.cpu_khz = perfctr_cpu_khz();
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
	ret = "unknown driver (oprofile?)";
	if (reserve_lapic_nmi() < 0)
		goto out_unlock;
	current_service = service;
	__module_get(THIS_MODULE);
	if (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		mmu_cr4_features |= X86_CR4_PCE;
	on_each_cpu(perfctr_cpu_init_one, NULL, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_init();
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
	if (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		mmu_cr4_features &= ~X86_CR4_PCE;
	on_each_cpu(perfctr_cpu_exit_one, NULL, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_exit();
	current_service = 0;
	release_lapic_nmi();
	module_put(THIS_MODULE);
 out_unlock:
	mutex_unlock(&mutex);
}
