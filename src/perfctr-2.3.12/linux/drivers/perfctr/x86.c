/* $Id$
 * x86 performance-monitoring counters driver.
 *
 * Copyright (C) 1999-2001  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/perfctr.h>

#include <asm/msr.h>

#include "compat.h"
#include "x86_compat.h"
#include "x86_tests.h"

/* Support for lazy evntsel and perfctr MSR updates. */
struct per_cpu_cache {	/* subset of perfctr_cpu_state */
	union {
		unsigned int p5_cesr;
		unsigned int id;	/* cache owner id */
	} k1;
	struct {
		unsigned int evntsel[18];
		unsigned int evntsel_aux[18];
	} control;
} __attribute__((__aligned__(SMP_CACHE_BYTES)));
static struct per_cpu_cache per_cpu_cache[NR_CPUS] __cacheline_aligned;

/* Intel P5, Cyrix 6x86MX/MII/III, Centaur WinChip C6/2/3 */
#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12		/* .. 0x13 */
#define P5_CESR_CPL		0x00C0
#define P5_CESR_RESERVED	(~0x01FF)
#define MII_CESR_RESERVED	(~0x05FF)
#define C6_CESR_RESERVED	(~0x00FF)

/* Intel P6, VIA C3 */
#define MSR_P6_PERFCTR0		0xC1		/* .. 0xC2 */
#define MSR_P6_EVNTSEL0		0x186		/* .. 0x187 */
#define P6_EVNTSEL_ENABLE	0x00400000
#define P6_EVNTSEL_INT		0x00100000
#define P6_EVNTSEL_CPL		0x00030000
#define P6_EVNTSEL_RESERVED	0x00280000
#define VC3_EVNTSEL1_RESERVED	(~0x1FF)

/* AMD K7 */
#define MSR_K7_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K7_PERFCTR0		0xC0010004	/* .. 0xC0010007 */

/* Intel P4 */
#define MSR_IA32_MISC_ENABLE	0x1A0
#define MSR_P4_PERFCTR0		0x300		/* .. 0x311 */
#define MSR_P4_CCCR0		0x360		/* .. 0x371 */
#define P4_CCCR_RESERVED	0xB8000FFF	/* must be zeros */
#define P4_CCCR_REQUIRED	0x00031000	/* must be ones */
#define P4_CCCR_CASCADE		0x40000000
#define P4_CCCR_OVF_PMI		0x04000000
#define P4_CCCR_ESCR_SELECT(X)	(((X) >> 13) & 0x7)
#define P4_ESCR_RESERVED	0x80000003	/* must be zeros */
#define P4_ESCR_CPL		0x0000000C	/* must be non-zero */
#define P4_ESCR_TAG_ENABLE	0x00000010
#define P4_FAST_RDPMC		0x80000000
#define P4_MASK_FAST_RDPMC	0x0000001F	/* we only need low 5 bits */

#define rdmsrl(msr,low) \
	__asm__ __volatile__("rdmsr" : "=a"(low) : "c"(msr) : "edx")
#define rdpmcl(ctr,low) \
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

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
	static spinlock_t lock = SPIN_LOCK_UNLOCKED;
	static unsigned int counter;
	int id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

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
 * - pmc_map[] is not required to be the identity function.
 */

/* shared with MII and C6 */
static int p5_like_check_control(struct perfctr_cpu_state *state,
				 unsigned int reserved_bits, int is_c6)
{
	unsigned short cesr_half[2];
	unsigned int pmc, evntsel, i;

	if( state->control.nrictrs != 0 || state->control.nractrs > 2 )
		return -EINVAL;
	cesr_half[0] = 0;
	cesr_half[1] = 0;
	for(i = 0; i < state->control.nractrs; ++i) {
		pmc = state->control.pmc_map[i];
		if( pmc > 1 || cesr_half[pmc] != 0 )
			return -EINVAL;
		evntsel = state->control.evntsel[i];
		/* protect reserved bits */
		if( (evntsel & reserved_bits) != 0 )
			return -EPERM;
		/* the CPL field (if defined) must be non-zero */
		if( !is_c6 && !(evntsel & P5_CESR_CPL) )
			return -EINVAL;
		cesr_half[pmc] = evntsel;
	}
	state->k1.p5_cesr = (cesr_half[1] << 16) | cesr_half[0];
	return 0;
}

static int p5_check_control(struct perfctr_cpu_state *state)
{
	return p5_like_check_control(state, P5_CESR_RESERVED, 0);
}

/* shared with MII but not C6 */
static void p5_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned cesr;

	cesr = state->k1.p5_cesr;
	if( !cesr )	/* no PMC is on (this test doesn't work on C6) */
		return;
	cpu = &per_cpu_cache[smp_processor_id()];
	if( cpu->k1.p5_cesr != cesr ) {
		cpu->k1.p5_cesr = cesr;
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
	if( perfctr_cstatus_has_tsc(cstatus) )
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->control.pmc_map[i];
		rdmsrl(MSR_P5_CTR0+pmc, ctrs->pmc[i]);
	}
}

/* shared with MII, C6, and VC3 */
static void p5mmx_read_counters(const struct perfctr_cpu_state *state,
				struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	/* The P5 doesn't allocate a cache line on a write miss, so do
	   a dummy read to avoid a write miss here _and_ a read miss
	   later in our caller. */
	asm("" : : "r"(ctrs->tsc));

	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->control.pmc_map[i];
		rdpmcl(pmc, ctrs->pmc[i]);
	}
}

/* shared with MII and C6 */
static void p5_clear_counters(void)
{
	wrmsr(MSR_P5_CESR, 0, 0);
	wrmsr(MSR_P5_CTR0+0, 0, 0);
	wrmsr(MSR_P5_CTR0+1, 0, 0);
}

/*
 * Cyrix 6x86/MII/III.
 * - Same MSR assignments as P5 MMX. Has RDPMC and two 48-bit PMCs.
 * - Event codes and CESR formatting as in the plain P5 subset.
 * - Many but not all P5 MMX event codes are implemented.
 * - Cyrix adds a few more event codes. The event code is widened
 *   to 7 bits, and Cyrix puts the high bit in CESR bit 10
 *   (and CESR bit 26 for PMC1).
 */

static int mii_check_control(struct perfctr_cpu_state *state)
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
static int c6_check_control(struct perfctr_cpu_state *state)
{
	if( state->control.tsc_on )
		return -EINVAL;
	return p5_like_check_control(state, C6_CESR_RESERVED, 1);
}

static void c6_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned cesr;

	if( perfctr_cstatus_nractrs(state->cstatus) == 0 ) /* no PMC is on */
		return;
	cpu = &per_cpu_cache[smp_processor_id()];
	cesr = state->k1.p5_cesr;
	if( cpu->k1.p5_cesr != cesr ) {
		cpu->k1.p5_cesr = cesr;
		wrmsr(MSR_P5_CESR, cesr, 0);
	}
}
#endif

/*
 * Intel P6 family (Pentium Pro, Pentium II, and Pentium III cores,
 * and Xeon and Celeron versions of Pentium II and III cores).
 * - One TSC and two 40-bit PMCs.
 * - One 32-bit EVNTSEL MSR for each PMC.
 * - EVNTSEL0 contains a global enable/disable bit.
 *   That bit is reserved in EVNTSEL1.
 * - Each EVNTSEL contains a CPL field.
 * - Overflow interrupts are possible, but requires that the
 *   local APIC is available. Mobile P6 CPUs have no local APIC.
 *   Additional kernel patches are also required.
 * - The PMCs cannot be initialised with arbitrary values, since
 *   wrmsr fills the high bits by sign-extending from bit 31.
 * - Most events are symmetric, but a few are not.
 * - pmc_map[] is required to be the identity function. PMC1 cannot
 *   be used if PMC0 is skipped (since EVNTSEL0 has the global
 *   enable bit), so the counters might as well be listed in the
 *   natural order.
 */

/* shared with K7 */
static int p6_like_check_control(struct perfctr_cpu_state *state, int is_k7)
{
	unsigned int evntsel, i, nractrs, nrctrs;

	nractrs = state->control.nractrs;
	nrctrs = nractrs + state->control.nrictrs;
	if( nrctrs < nractrs || nrctrs > (is_k7 ? 4 : 2) )
		return -EINVAL;

	for(i = 0; i < nrctrs; ++i) {
		/* pmc_map[] should be the identity function */
		if( state->control.pmc_map[i] != i )
			return -EINVAL;
		evntsel = state->control.evntsel[i];
		/* protect reserved bits */
		if( evntsel & P6_EVNTSEL_RESERVED )
			return -EPERM;
		/* check ENable bit */
		if( is_k7 ) {
			/* ENable bit must be set in each evntsel */
			if( !(evntsel & P6_EVNTSEL_ENABLE) )
				return -EINVAL;
		} else {
			/* only evntsel[0] has the ENable bit */
			if( evntsel & P6_EVNTSEL_ENABLE ) {
				if( i > 0 )
					return -EPERM;
			} else {
				if( i == 0 )
					return -EINVAL;
			}
		}
		/* the CPL field must be non-zero */
		if( !(evntsel & P6_EVNTSEL_CPL) )
			return -EINVAL;
		/* INT bit must be off for a-mode and on for i-mode counters */
		if( evntsel & P6_EVNTSEL_INT ) {
			if( i < nractrs )
				return -EINVAL;
		} else {
			if( i >= nractrs )
				return -EINVAL;
		}
	}
	state->k1.id = new_id();
	return 0;
}

static int p6_check_control(struct perfctr_cpu_state *state)
{
	return p6_like_check_control(state, 0);
}

#ifdef CONFIG_PERFCTR_DEBUG
static void debug_evntsel_cache(const struct perfctr_cpu_state *state,
				const struct per_cpu_cache *cpu)
{
	unsigned int nrctrs, i;

	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int evntsel = state->control.evntsel[i];
		if( evntsel != cpu->control.evntsel[i] ) {
			printk(KERN_ERR "perfctr/x86.c: (pid %d, comm %s) "
			       "evntsel[%u] is %#x, should be %#x\n",
			       current->pid, current->comm,
			       i, cpu->control.evntsel[i], evntsel);
			return;
		}
	}
}
#else
static inline void debug_evntsel_cache(const struct perfctr_cpu_state *s,
				       const struct per_cpu_cache *c)
{ }
#endif

static void p6_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	if( cpu->k1.id == state->k1.id ) {
		debug_evntsel_cache(state, cpu);
		return;
	}
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int evntsel = state->control.evntsel[i];
		if( evntsel != cpu->control.evntsel[i] ) {
			cpu->control.evntsel[i] = evntsel;
			wrmsr(MSR_P6_EVNTSEL0+i, evntsel, 0);
		}
	}
	cpu->k1.id = state->k1.id;
}

/* shared with K7 */
static void p6_read_counters(const struct perfctr_cpu_state *state,
			     struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )	/* XXX: ignore this test? */
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i)
		rdpmcl(i, ctrs->pmc[i]);
}

static void p6_clear_counters(void)
{
	int i;

	for(i = 0; i < 2; ++i) {
		wrmsr(MSR_P6_EVNTSEL0+i, 0, 0);
		wrmsr(MSR_P6_PERFCTR0+i, 0, 0);
	}
}

#if PERFCTR_INTERRUPT_SUPPORT
/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
static void p6_isuspend(struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int cstatus, nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	cpu->control.evntsel[0] = 0;
	wrmsr(MSR_P6_EVNTSEL0, 0, 0);
	/* cpu->k1.id is still == state->k1.id */
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i)
		rdpmcl(i, state->start.pmc[i]);
}

/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
static void p6_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int cstatus, nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	/* XXX: move k1.id test up here? */
	if( cpu->control.evntsel[0] ) {
		cpu->control.evntsel[0] = 0;
		wrmsr(MSR_P6_EVNTSEL0, 0, 0);
		cpu->k1.id = 0;
	} else if( cpu->k1.id == state->k1.id ) {
		/* isuspend() cleared EVNTSEL0, so invalidate the cache
		   here to force write_control() to reload the EVNTSELs.
		   The k1.id cache still allows us to avoid reloading
		   the PERFCTRs. */
		cpu->k1.id = 0;
		return;
	}
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i)
		wrmsr(MSR_P6_PERFCTR0+i, state->start.pmc[i], 0);
	/* cpu->k1.id remains != state->k1.id */
}
#endif	/* PERFCTR_INTERRUPT_SUPPORT */

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
 * - pmc_map[] is required to be the identity function.
 */

static int k7_check_control(struct perfctr_cpu_state *state)
{
	return p6_like_check_control(state, 1);
}

static void k7_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	if( cpu->k1.id == state->k1.id ) {
		debug_evntsel_cache(state, cpu);
		return;
	}
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int evntsel = state->control.evntsel[i];
		if( evntsel != cpu->control.evntsel[i] ) {
			cpu->control.evntsel[i] = evntsel;
			wrmsr(MSR_K7_EVNTSEL0+i, evntsel, 0);
		}
	}
	cpu->k1.id = state->k1.id;
}

static void k7_clear_counters(void)
{
	int i;

	for(i = 0; i < 4; ++i) {
		wrmsr(MSR_K7_EVNTSEL0+i, 0, 0);
		wrmsr(MSR_K7_PERFCTR0+i, 0, 0);
	}
}

#if PERFCTR_INTERRUPT_SUPPORT
/* PRE: perfctr_cstatus_has_ictrs(control->cstatus) != 0 */
static void k7_isuspend(struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int cstatus, nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		cpu->control.evntsel[i] = 0;
		wrmsr(MSR_K7_EVNTSEL0+i, 0, 0);
		rdpmcl(i, state->start.pmc[i]);
	}
	/* cpu->k1.id is still == state->k1.id */
}

/* PRE: perfctr_cstatus_has_ictrs(state->cstatus) != 0 */
static void k7_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int cstatus, nrctrs, nractrs, i;
	int id_valid;

	cpu = &per_cpu_cache[smp_processor_id()];
	id_valid = 1;
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	nractrs = perfctr_cstatus_nractrs(cstatus);
	/* XXX: move k1.id test up here? */
	for(i = nractrs; i < nrctrs; ++i) {
		if( cpu->control.evntsel[i] ) {
			cpu->control.evntsel[i] = 0;
			wrmsr(MSR_K7_EVNTSEL0+i, 0, 0);
			id_valid = 0;
		}
	}
	if( !id_valid )
		cpu->k1.id = 0;
	else if( cpu->k1.id == state->k1.id ) {
		cpu->k1.id = 0; /* see comment in p6_iresume() */
		return;
	}
	for(i = nractrs; i < nrctrs; ++i)
		wrmsr(MSR_K7_PERFCTR0+i, state->start.pmc[i], -1);
	/* cpu->k1.id remains != state->k1.id */
}
#endif	/* PERFCTR_INTERRUPT_SUPPORT */

/*
 * VIA C3 family.
 * - A Centaur design somewhat similar to the Intel P6.
 * - PERFCTR0 is an alias for the TSC, and EVNTSEL0 is read-only.
 * - PERFCTR1 is 32 bits wide.
 * - EVNTSEL1 has no defined control fields, and there is no
 *   defined method for stopping the counter.
 * - According to testing, the reserved fields in EVNTSEL1 have
 *   no function. We always fill them with zeroes.
 * - Only a few event codes are defined.
 * - No local APIC or interrupt-mode support.
 * - pmc_map[] is NOT the identity function: pmc_map[0] must be 1,
 *   if nractrs == 1.
 */
static int vc3_check_control(struct perfctr_cpu_state *state)
{
	if( state->control.nrictrs || state->control.nractrs > 1 )
		return -EINVAL;
	if( state->control.nractrs == 1 ) {
		if( state->control.pmc_map[0] != 1 )
			return -EINVAL;
		if( state->control.evntsel[0] & VC3_EVNTSEL1_RESERVED )
			return -EPERM;
	}
	return 0;
}

static void vc3_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned evntsel;

	/* check if PERFCTR1 is enabled */
	if( perfctr_cstatus_nractrs(state->cstatus) == 0 )
		return;
	cpu = &per_cpu_cache[smp_processor_id()];
	evntsel = state->control.evntsel[0];
	if( cpu->control.evntsel[0] != evntsel ) {
		cpu->control.evntsel[0] = evntsel;
		wrmsr(MSR_P6_EVNTSEL0+1, evntsel, 0);
	}
}

/*
 * Intel Pentium 4.
 * Current implementation restrictions:
 * - No cascading counters support.
 * - No overflow interrupts support.
 * - No tagging of micro-ops support.
 * - No DS/PEBS support.
 */

/*
 * Table 14-4 in the IA32 Volume 3 manual contains a 18x8 entry mapping
 * from counter/CCCR number (0-17) and ESCR SELECT value (0-7) to the
 * actual ESCR MSR number. This mapping contains some repeated patterns,
 * so we can compact it to a 5x8 table of MSR offsets:
 *
 * 1. CCCRs 16 and 17 are mapped just like CCCRs 12 and 13, respectively.
 *    Thus, we only consider the 16 CCCRs 0-15.
 * 2. The CCCRs are organised in pairs, and both CCCRs in a pair use the
 *    same mapping. Thus, we only consider the 8 pairs 0-7.
 * 3. In each pair of pairs, the second odd-numbered pair has the same domain
 *    as the first even-numbered pair, and the range is 1+ the range of the
 *    the first even-numbered pair. For example, CCCR(0) and (1) map ESCR
 *    SELECT(7) to 0x3A0, and CCCR(2) and (3) map it to 0x3A1.
 *    However, pairs (6) and (7) [CCCRs 12-15] do not follow this pattern
 *    due to some strange irregularities in the maps for CCCRs 14 and 15.
 *    We reduce the 8 pairs to 5, and store those 5 mappings in the table.
 * 4. All MSR numbers are on the form 0x3??. Instead of storing these as
 *    16-bit numbers, the table only stores the 8-bit offsets from 0x300.
 */

static const unsigned char p4_cccr_escr_map[5][8] = {
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
	/* 0x0C, 0x0D, and 0x10 as is */
	[0x0C/4] {	[4] 0xB8,
			[5] 0xCC,
			[6] 0xE0,
			[0] 0xBA,
			[3] 0xBC,
			[2] 0xBE,
			[1] 0xCA, },
	/* 0x0E, 0x0F, and 0x11 as is */
	[0x11/4] {	[4] 0xB9,
			[5] 0xCD,
			[6] 0xE1,
			[0] 0xBB,
			[2] 0xBD,
			[1] 0xCB, },
};

static unsigned int p4_escr_addr(unsigned int pmc, unsigned int cccr_val)
{
	unsigned int pair, index, escr_offset;

	if( pmc > 0x11 )
		return 0;	/* pmc range error */
	if( pmc > 0x0F )
		pmc -= 4;	/* 0 <= pmc <= 0x0F */
	pair = pmc / 2;		/* 0 <= pair <= 7 */
	index = (pair == 7) ? 4 : (pair / 2);	/* 0 <= index <= 4 */
	escr_offset = p4_cccr_escr_map[index][P4_CCCR_ESCR_SELECT(cccr_val)];
	if( !escr_offset )
		return 0;	/* ESCR SELECT range error */
	if( pair < 6 )
		escr_offset += (pair & 1);
	return escr_offset + 0x300;
};

static int p4_check_control(struct perfctr_cpu_state *state)
{
	unsigned int nrctrs, i, pmc, cccr_val, escr_val, escr_addr, pmc_mask;

	nrctrs = state->control.nractrs;
	if( nrctrs > 18 || state->control.nrictrs != 0 )
		return -EINVAL;

	pmc_mask = 0;
	for(i = 0; i < nrctrs; ++i) {
		/* check that pmc_map[] is well-defined;
		   pmc_map[i] is what we pass to RDPMC, the PMC itself
		   is extracted by masking off the FAST RDPMC flag */
		pmc = state->control.pmc_map[i] & ~P4_FAST_RDPMC;
		if( pmc >= 18 || (pmc_mask & (1<<pmc)) )
			return -EINVAL;
		pmc_mask |= (1<<pmc);
		/* check CCCR contents */
		cccr_val = state->control.evntsel[i];
		if( cccr_val & P4_CCCR_RESERVED )
			return -EPERM;
		if( (cccr_val & P4_CCCR_REQUIRED) != P4_CCCR_REQUIRED )
			return -EINVAL;
		if( cccr_val & P4_CCCR_CASCADE )	/* XXX: NYI */
			return -EPERM;
		if( cccr_val & P4_CCCR_OVF_PMI )	/* XXX: NYI */
			return -EPERM;
		/* check ESCR contents */
		escr_val = state->control.evntsel_aux[i];
		if( escr_val & P4_ESCR_RESERVED )
			return -EPERM;
		if( !(escr_val & P4_ESCR_CPL) )
			return -EINVAL;
		if( escr_val & P4_ESCR_TAG_ENABLE )	/* XXX: NYI */
			return -EPERM;
		/* compute and cache ESCR address */
		escr_addr = p4_escr_addr(pmc, cccr_val);
		if( !escr_addr )
			return -EINVAL;		/* ESCR SELECT range error */
		/* XXX: It's possible to have two CCCRs using the same ESCR.
		   Is it worthwhile to check here that they use the same
		   ESCR value? */
		state->k2.p4_escr_map[i] = escr_addr;
	}
	state->k1.id = new_id();
	return 0;
}

static void p4_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cpu;
	unsigned int nrctrs, i;

	cpu = &per_cpu_cache[smp_processor_id()];
	if( cpu->k1.id == state->k1.id )
		return;
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int escr_val, cccr_val, pmc;
		escr_val = state->control.evntsel_aux[i];
		if( escr_val != cpu->control.evntsel_aux[i] ) {
			cpu->control.evntsel_aux[i] = escr_val;
			wrmsr(state->k2.p4_escr_map[i], escr_val, 0);
		}
		cccr_val = state->control.evntsel[i];
		if( cccr_val != cpu->control.evntsel[i] ) {
			pmc = state->control.pmc_map[i] & P4_MASK_FAST_RDPMC;
			cpu->control.evntsel[i] = cccr_val;
			wrmsr(MSR_P4_CCCR0+pmc, cccr_val, 0);
		}
	}
	cpu->k1.id = state->k1.id;
}

static void p4_read_counters(const struct perfctr_cpu_state *state,
			     struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	/* XXX: could be shared with P5 et al if we remove the
	   prefetch in p5mmx_read_counters() */

	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->control.pmc_map[i];
		rdpmcl(pmc, ctrs->pmc[i]);
	}
}

static void p4_clear_counters(void)
{
	int i;

	for(i = 0; i < 18; ++i) {
		wrmsr(MSR_P4_PERFCTR0+i, 0, 0);
		wrmsr(MSR_P4_CCCR0+i, 0, 0);
	}
	for(i = 0x3A0; i <= 0x3BE; ++i)
		wrmsr(i, 0, 0);
	for(i = 0x3C0; i <= 0x3C5; ++i)
		wrmsr(i, 0, 0);
	for(i = 0x3C8; i <= 0x3CD; ++i)
		wrmsr(i, 0, 0);
	for(i = 0x3E0; i <= 0x3E1; ++i)
		wrmsr(i, 0, 0);
	/* XXX: 0x3F0, 0x3F1, 0x3F2 ??? */
}

/*
 * Generic driver for any x86 with a working TSC.
 */

static int generic_check_control(struct perfctr_cpu_state *state)
{
	if( state->control.nractrs || state->control.nrictrs )
		return -EINVAL;
	return 0;
}

static void generic_write_control(const struct perfctr_cpu_state *state)
{
}

static void generic_read_counters(const struct perfctr_cpu_state *state,
				  struct perfctr_low_ctrs *ctrs)
{
	rdtscl(ctrs->tsc);
}

static void generic_clear_counters(void)
{
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
 */

static int redirect_call_disable;

static void redirect_call(void *ra, void *to)
{
	/* XXX: make this function __init later */
	if( redirect_call_disable )
		printk(KERN_ERR __FILE__ ":" __FUNCTION__
		       ": unresolved call to %p at %p\n",
		       to, ra);
	/* we can only redirect `call near relative' instructions */
	if( *((unsigned char*)ra - 5) != 0xE8 ) {
		printk(KERN_WARNING __FILE__ ":" __FUNCTION__
		       ": unable to redirect caller %p to %p\n",
		       ra, to);
		return;
	}
	*(int*)((char*)ra - 4) = (char*)to - (char*)ra;
}

static void (*write_control)(const struct perfctr_cpu_state*);
static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), write_control);
	return write_control(state);
}

static void (*read_counters)(const struct perfctr_cpu_state*,
			     struct perfctr_low_ctrs*);
static void perfctr_cpu_read_counters(const struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	redirect_call(__builtin_return_address(0), read_counters);
	return read_counters(state, ctrs);
}

#if PERFCTR_INTERRUPT_SUPPORT
static void (*cpu_isuspend)(struct perfctr_cpu_state*);
static void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), cpu_isuspend);
	return cpu_isuspend(state);
}

static void (*cpu_iresume)(const struct perfctr_cpu_state*);
static void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	redirect_call(__builtin_return_address(0), cpu_iresume);
	return cpu_iresume(state);
}

void perfctr_cpu_ireload(const struct perfctr_cpu_state *state)
{
	/* Call ireload() just before iresume() to bypass
	   internal caching and force a reload of i-mode PMCs. */
	struct per_cpu_cache *cpu;
	cpu = &per_cpu_cache[smp_processor_id()];
	cpu->k1.id = 0;
}

/* PRE: the counters have been suspended and sampled */
unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state *state)
{
	unsigned int cstatus, nrctrs, pmc, pmc_mask;

	cstatus = state->cstatus;
	pmc = perfctr_cstatus_nractrs(cstatus);
	nrctrs = perfctr_cstatus_nrctrs(cstatus);

	/* Only one i-mode PMC: we don't have to poll. */
	if( nrctrs == pmc+1 ) {
		state->start.pmc[pmc] = state->control.ireset[pmc];
		return (1 << pmc);
	}

	/* Multiple i-mode PMCs: must poll and accumulate bitmask. */
	for(pmc_mask = 0; pmc < nrctrs; ++pmc) {
		if( (int)state->start.pmc[pmc] >= 0 ) {
			state->start.pmc[pmc] = state->control.ireset[pmc];
			pmc_mask |= (1 << pmc);
		}
	}
	return pmc_mask;
}
#endif	/* PERFCTR_INTERRUPT_SUPPORT */

static inline int check_ireset(const struct perfctr_cpu_state *state)
{
#if PERFCTR_INTERRUPT_SUPPORT
	unsigned int nrctrs, i;

	i = state->control.nractrs;
	nrctrs = i + state->control.nrictrs;
	for(; i < nrctrs; ++i)
		if( state->control.ireset[i] >= 0 )
			return -EINVAL;
#endif
	return 0;
}

static inline void setup_imode_start_values(struct perfctr_cpu_state *state)
{
#if PERFCTR_INTERRUPT_SUPPORT
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i)
		state->start.pmc[i] = state->control.ireset[i];
#endif
}

static int (*check_control)(struct perfctr_cpu_state*);
int perfctr_cpu_update_control(struct perfctr_cpu_state *state)
{
	int err;

#if PERFCTR_INTERRUPT_SUPPORT
	if( perfctr_cstatus_has_ictrs(state->cstatus) )
	    perfctr_cpu_isuspend(state);
#endif
	state->cstatus = 0;
	if( !(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT) ) {
		/* disallow i-mode counters if we cannot catch the interrupts */
		if( state->control.nrictrs )
			return -EPERM;
	}
	err = check_control(state);
	if( err < 0 )
		return err;
	err = check_ireset(state);
	if( err < 0 )
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

#if PERFCTR_INTERRUPT_SUPPORT
	if( perfctr_cstatus_has_ictrs(state->cstatus) )
	    perfctr_cpu_isuspend(state);
#endif
	perfctr_cpu_read_counters(state, &now);
	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		state->sum.tsc += now.tsc - state->start.tsc;
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i)
		state->sum.pmc[i] += now.pmc[i] - state->start.pmc[i];
	/* perfctr_cpu_disable_rdpmc(); */	/* not for x86 */
}

void perfctr_cpu_resume(struct perfctr_cpu_state *state)
{
#if PERFCTR_INTERRUPT_SUPPORT
	if( perfctr_cstatus_has_ictrs(state->cstatus) )
	    perfctr_cpu_iresume(state);
#endif
	/* perfctr_cpu_enable_rdpmc(); */	/* not for x86 or global-mode */
	perfctr_cpu_write_control(state);
	perfctr_cpu_read_counters(state, &state->start);
	/* XXX: if (SMP && start.tsc == now.tsc) ++now.tsc; */
}

void perfctr_cpu_sample(struct perfctr_cpu_state *state)
{
	unsigned int i, cstatus, nractrs;
	struct perfctr_low_ctrs now;

	perfctr_cpu_read_counters(state, &now);
	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) ) {
		state->sum.tsc += now.tsc - state->start.tsc;
		state->start.tsc = now.tsc;
	}
	nractrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nractrs; ++i) {
		state->sum.pmc[i] += now.pmc[i] - state->start.pmc[i];
		state->start.pmc[i] = now.pmc[i];
	}
}

static void (*clear_counters)(void);
static void perfctr_cpu_clear_counters(void)
{
	return clear_counters();
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

/* see comment above at redirect_call() */
static void __init finalise_backpatching(void)
{
	struct per_cpu_cache *cpu;
	struct perfctr_cpu_state state;

	cpu = &per_cpu_cache[smp_processor_id()];
	memset(cpu, 0, sizeof *cpu);
	memset(&state, 0, sizeof state);
	state.cstatus =
		(perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
		? perfctr_mk_cstatus(0, 0, 1)
		: 0;
	perfctr_cpu_sample(&state);
	perfctr_cpu_resume(&state);
	perfctr_cpu_suspend(&state);
	perfctr_cpu_update_control(&state);

	redirect_call_disable = 1;
}

static int __init intel_init(void)
{
	unsigned int misc_enable;

	if( !cpu_has_tsc )
		return -ENODEV;
	switch( boot_cpu_data.x86 ) {
	case 5:
		if( cpu_has_mmx ) {
			perfctr_info.cpu_type = PERFCTR_X86_INTEL_P5MMX;
			read_counters = p5mmx_read_counters;
		} else {
			perfctr_info.cpu_type = PERFCTR_X86_INTEL_P5;
			perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
			read_counters = p5_read_counters;
		}
		write_control = p5_write_control;
		check_control = p5_check_control;
		clear_counters = p5_clear_counters;
		perfctr_p5_init_tests();
		return 0;
	case 6:
		if( boot_cpu_data.x86_model >= 7 )	/* PIII */
			perfctr_info.cpu_type = PERFCTR_X86_INTEL_PIII;
		else if( boot_cpu_data.x86_model >= 3 )	/* PII or Celeron */
			perfctr_info.cpu_type = PERFCTR_X86_INTEL_PII;
		else
			perfctr_info.cpu_type = PERFCTR_X86_INTEL_P6;
		read_counters = p6_read_counters;
		write_control = p6_write_control;
		check_control = p6_check_control;
		clear_counters = p6_clear_counters;
#if PERFCTR_INTERRUPT_SUPPORT
		if( cpu_has_apic ) {
			perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
			cpu_isuspend = p6_isuspend;
			cpu_iresume = p6_iresume;
		}
#endif
		perfctr_p6_init_tests();
		return 0;
	case 15:	/* Pentium 4 */
		printk("perfctr: Pentium 4 detected\n");
		rdmsrl(0x1A0, misc_enable);
		if( !(misc_enable & (1 << 7)) ) {
			printk("perfctr: Performance Monitoring is unavailable\n");
			return -ENODEV;
		}
		perfctr_info.cpu_type = PERFCTR_X86_INTEL_P4;
		read_counters = p4_read_counters;
		write_control = p4_write_control;
		check_control = p4_check_control;
		clear_counters = p4_clear_counters;
		/* XXX: set up isuspend/iresume here later */
		return 0;
	}
	return -ENODEV;
}

static int __init amd_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	switch( boot_cpu_data.x86 ) {
	case 6:	/* K7. Model 1 does not have a local APIC.
		   AMD Document #22007 Revision J hints that APIC-less
		   K7s signal overflows as debug interrupts. */
		perfctr_info.cpu_type = PERFCTR_X86_AMD_K7;
		read_counters = p6_read_counters;
		write_control = k7_write_control;
		check_control = k7_check_control;
		clear_counters = k7_clear_counters;
#if PERFCTR_INTERRUPT_SUPPORT
		if( cpu_has_apic ) {
			perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
			cpu_isuspend = k7_isuspend;
			cpu_iresume = k7_iresume;
		}
#endif
		perfctr_k7_init_tests();
		return 0;
	}
	return -ENODEV;
}

static int __init cyrix_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	switch( boot_cpu_data.x86 ) {
	case 6:	/* 6x86MX, MII, or III */
		perfctr_info.cpu_type = PERFCTR_X86_CYRIX_MII;
		read_counters = p5mmx_read_counters;
		write_control = p5_write_control;
		check_control = mii_check_control;
		clear_counters = p5_clear_counters;
		perfctr_mii_init_tests();
		return 0;
	}
	return -ENODEV;
}

static int __init centaur_init(void)
{
	switch( boot_cpu_data.x86 ) {
#if !defined(CONFIG_X86_TSC)
	case 5:
		switch( boot_cpu_data.x86_model ) {
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
		/*
		 * TSC must be inaccessible for perfctrs to work.
		 */
		if( !(read_cr4() & X86_CR4_TSD) || cpu_has_tsc )
			return -ENODEV;
		perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDTSC;
		read_counters = p5mmx_read_counters;
		write_control = c6_write_control;
		check_control = c6_check_control;
		clear_counters = p5_clear_counters;
		perfctr_c6_init_tests();
		return 0;
#endif
	case 6: /* VIA C3 */
		if( !cpu_has_tsc )
			return -ENODEV;
		switch( boot_cpu_data.x86_model ) {
		case 6:	/* VIA C3 (Cyrix III) */
		case 7:	/* VIA C3 Samuel 2 or Ezra */
			break;
		default:
			return -ENODEV;
		}
		perfctr_info.cpu_type = PERFCTR_X86_VIA_C3;
		read_counters = p5mmx_read_counters;
		write_control = vc3_write_control;
		check_control = vc3_check_control;
		clear_counters = generic_clear_counters;
		perfctr_vc3_init_tests();
		return 0;
	}
	return -ENODEV;
}

static int __init generic_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	perfctr_info.cpu_type = PERFCTR_X86_GENERIC;
	check_control = generic_check_control;
	write_control = generic_write_control;
	read_counters = generic_read_counters;
	clear_counters = generic_clear_counters;
	return 0;
}

static char generic_name[] __initdata = "Generic x86 with TSC";
static char p5_name[] __initdata = "Intel Pentium";
static char p5mmx_name[] __initdata = "Intel Pentium MMX";
static char p6_name[] __initdata = "Intel Pentium Pro";
static char pii_name[] __initdata = "Intel Pentium II";
static char piii_name[] __initdata = "Intel Pentium III";
static char mii_name[] __initdata = "Cyrix 6x86MX/MII/III";
static char wcc6_name[] __initdata = "WinChip C6";
static char wc2_name[] __initdata = "WinChip 2/3";
static char k7_name[] __initdata = "AMD K7";
static char vc3_name[] __initdata = "VIA C3";
static char p4_name[] __initdata = "Intel Pentium 4";

char *perfctr_cpu_name[] __initdata = {
	[PERFCTR_X86_GENERIC] generic_name,
	[PERFCTR_X86_INTEL_P5] p5_name,
	[PERFCTR_X86_INTEL_P5MMX] p5mmx_name,
	[PERFCTR_X86_INTEL_P6] p6_name,
	[PERFCTR_X86_INTEL_PII] pii_name,
	[PERFCTR_X86_INTEL_PIII] piii_name,
	[PERFCTR_X86_CYRIX_MII] mii_name,
	[PERFCTR_X86_WINCHIP_C6] wcc6_name,
	[PERFCTR_X86_WINCHIP_2] wc2_name,
	[PERFCTR_X86_AMD_K7] k7_name,
	[PERFCTR_X86_VIA_C3] vc3_name,
	[PERFCTR_X86_INTEL_P4] p4_name,
};

static void __init perfctr_cpu_init_one(void *ignore)
{
	perfctr_cpu_clear_counters();
#if PERFCTR_INTERRUPT_SUPPORT
	if( cpu_has_apic )
		apic_write(APIC_LVTPC, LOCAL_PERFCTR_VECTOR);
#endif
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		set_in_cr4_local(X86_CR4_PCE);
}

static void __exit perfctr_cpu_exit_one(void *ignore)
{
	perfctr_cpu_clear_counters();
#if PERFCTR_INTERRUPT_SUPPORT
	if( cpu_has_apic )
		apic_write(APIC_LVTPC, APIC_DM_NMI | APIC_LVT_MASKED);
#endif
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		clear_in_cr4_local(X86_CR4_PCE);
}

#if defined(NMI_LOCAL_APIC) && defined(CONFIG_PM)

static void __init unregister_nmi_pmdev(void)
{
	if( nmi_pmdev ) {
		apic_pm_unregister(nmi_pmdev);
		nmi_pmdev = 0;
	}
}

static int x86_pm_callback(struct pm_dev *dev, pm_request_t rqst, void *data)
{
	/* XXX: incomplete */
	return 0;
}

static struct pm_dev *x86_pmdev;

static void __init x86_pm_init(void)
{
	x86_pmdev = apic_pm_register(PM_SYS_DEV, 0, x86_pm_callback);
}

static void __exit x86_pm_exit(void)
{
	if( x86_pmdev ) {
		apic_pm_unregister(x86_pmdev);
		x86_pmdev = NULL;
	}
}

#else

static inline void unregister_nmi_pmdev(void) { }
static inline void x86_pm_init(void) { }
static inline void x86_pm_exit(void) { }

#endif	/* NMI_LOCAL_APIC && CONFIG_PM */

#if defined(NMI_LOCAL_APIC)

static void __init disable_nmi_watchdog(void)
{
	if( nmi_perfctr_msr ) {
		nmi_perfctr_msr = 0;
		printk(KERN_NOTICE "perfctr: disabled nmi_watchdog\n");
		unregister_nmi_pmdev();
	}
}

#else

static inline void disable_nmi_watchdog(void) { }

#endif

int __init perfctr_cpu_init(void)
{
	int err = -ENODEV;

	/* RDPMC and RDTSC are on by default. They will be disabled
	   by the init procedures if necessary. */
	perfctr_info.cpu_features = PERFCTR_FEATURE_RDPMC | PERFCTR_FEATURE_RDTSC;

	if( cpu_has_msr ) {
		switch( boot_cpu_data.x86_vendor ) {
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
	if( err ) {
		err = generic_init();	/* last resort */
		if( err )
			return err;
	}
	/*
	 * Put the hardware in a sane state:
	 * - finalise resolution of backpatchable call sites
	 * - clear perfctr MSRs
	 * - set up APIC_LVTPC
	 * - set CR4.PCE [on permanently due to __flush_tlb_global()]
	 * - install our default interrupt handler
	 */
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features |= X86_CR4_PCE;
	finalise_backpatching();
	perfctr_cpu_init_one(NULL);
	smp_call_function(perfctr_cpu_init_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	/*
	 * Fix up the connection to the local APIC:
	 * - disable and disconnect the NMI watchdog
	 * - register our PM callback
	 */
	disable_nmi_watchdog();
	x86_pm_init();
	/*
	 * per_cpu_cache[] is initialised to contain "impossible"
	 * evntsel values guaranteed to differ from anything accepted
	 * by perfctr_cpu_check_control(). This way, initialisation of
	 * a CPU's evntsel MSRs will happen automatically the first time
	 * perfctr_cpu_write_control() executes on it.
	 * All-bits-one works for all currently supported processors.
	 * The memset also sets the ids to -1, which is intentional.
	 */
	memset(per_cpu_cache, ~0, sizeof per_cpu_cache);

	perfctr_info.cpu_khz = cpu_khz;
	perfctr_info.nrcpus = smp_num_cpus;

	return 0;
}

void __exit perfctr_cpu_exit(void)
{
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features &= ~X86_CR4_PCE;
	perfctr_cpu_exit_one(NULL);
	smp_call_function(perfctr_cpu_exit_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_exit();
	/* XXX: restart nmi watchdog? */
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
	MOD_INC_USE_COUNT;
	return 0;
}

void perfctr_cpu_release(const char *service)
{
	if( service != current_service ) {
		printk(KERN_ERR __FUNCTION__
		       ": attempt by %s to release while reserved by %s\n",
		       service, current_service);
	} else {
		perfctr_cpu_set_ihandler(NULL);
		current_service = 0;
		MOD_DEC_USE_COUNT;
	}
}
