/* $Id$
 * x86_64 performance-monitoring counters driver.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>

#include <asm/msr.h>
#include <asm/fixmap.h>
#include <asm/apic.h>
struct hw_interrupt_type;
#include <asm/hw_irq.h>

#include "compat.h"
#include "x86_compat.h"
#include "x86_tests.h"

/* Support for lazy evntsel and perfctr MSR updates. */
struct per_cpu_cache {	/* roughly a subset of perfctr_cpu_state */
	union {
		unsigned int id;	/* cache owner id */
	} k1;
	struct {
		/* NOTE: these caches have physical indices, not virtual */
		unsigned int evntsel[4];
	} control;
} ____cacheline_aligned;
static struct per_cpu_cache per_cpu_cache[NR_CPUS] __cacheline_aligned;

/* Structure for counter snapshots, as 32-bit values. */
struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[4];
};

/* AMD K8 */
#define MSR_K8_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K8_PERFCTR0		0xC0010004	/* .. 0xC0010007 */
#define K8_EVNTSEL_ENABLE	0x00400000
#define K8_EVNTSEL_INT		0x00100000
#define K8_EVNTSEL_CPL		0x00030000
#define K8_EVNTSEL_RESERVED	0x00280000

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
	static spinlock_t lock = SPIN_LOCK_UNLOCKED;
	static unsigned int counter;
	int id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

#if defined(CONFIG_SMP)

static inline void set_isuspend_cpu(struct perfctr_cpu_state *state,
				    int cpu)
{
	state->k1.isuspend_cpu = cpu;
}

static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state,
				  int cpu)
{
	return state->k1.isuspend_cpu == cpu;
}

static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state)
{
	state->k1.isuspend_cpu = NR_CPUS;
}

#else
static inline void set_isuspend_cpu(struct perfctr_cpu_state *state,
				    int cpu) { }
static inline int is_isuspend_cpu(const struct perfctr_cpu_state *state,
				  int cpu) { return 1; }
static inline void clear_isuspend_cpu(struct perfctr_cpu_state *state) { }
#endif

/* XXX: disabled: called from switch_to() where printk() is disallowed */
#if 0 && defined(CONFIG_PERFCTR_DEBUG)
static void debug_evntsel_cache(const struct perfctr_cpu_state *state,
				const struct per_cpu_cache *cache)
{
	unsigned int nrctrs, i;

	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int evntsel = state->control.evntsel[i];
		unsigned int pmc = state->control.pmc_map[i];
		if( evntsel != cache->control.evntsel[pmc] ) {
			printk(KERN_ERR "perfctr: (pid %d, comm %s) "
			       "evntsel[%u] is %#x, should be %#x\n",
			       current->pid, current->comm,
			       i, cache->control.evntsel[pmc], evntsel);
			return;
		}
	}
}
#else
static inline void debug_evntsel_cache(const struct perfctr_cpu_state *s,
				       const struct per_cpu_cache *c)
{ }
#endif

/****************************************************************
 *								*
 * Driver procedures.						*
 *								*
 ****************************************************************/

static void perfctr_cpu_read_counters(const struct perfctr_cpu_state *state,
				      struct perfctr_low_ctrs *ctrs)
{
	unsigned int cstatus, nrctrs, i;

	cstatus = state->cstatus;
	if( perfctr_cstatus_has_tsc(cstatus) )
		rdtscl(ctrs->tsc);
	nrctrs = perfctr_cstatus_nractrs(cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		rdpmc_low(pmc, ctrs->pmc[i]);
	}
}

static int k8_check_control(struct perfctr_cpu_state *state)
{
	unsigned int evntsel, i, nractrs, nrctrs, pmc_mask, pmc;

	nractrs = state->control.nractrs;
	nrctrs = nractrs + state->control.nrictrs;
	if( nrctrs < nractrs || nrctrs > 4 )
		return -EINVAL;

	pmc_mask = 0;
	for(i = 0; i < nrctrs; ++i) {
		pmc = state->control.pmc_map[i];
		state->pmc[i].map = pmc;
		if( pmc >= 4 || (pmc_mask & (1<<pmc)) )
			return -EINVAL;
		pmc_mask |= (1<<pmc);
		evntsel = state->control.evntsel[i];
		/* protect reserved bits */
		if( evntsel & K8_EVNTSEL_RESERVED )
			return -EPERM;
		/* ENable bit must be set in each evntsel */
		if( !(evntsel & K8_EVNTSEL_ENABLE) )
			return -EINVAL;
		/* the CPL field must be non-zero */
		if( !(evntsel & K8_EVNTSEL_CPL) )
			return -EINVAL;
		/* INT bit must be off for a-mode and on for i-mode counters */
		if( evntsel & K8_EVNTSEL_INT ) {
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

static void perfctr_cpu_isuspend(struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	cache = &per_cpu_cache[cpu];
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		unsigned int pmc, now;
		pmc = state->pmc[i].map;
		cache->control.evntsel[pmc] = 0;
		wrmsr(MSR_K8_EVNTSEL0+pmc, 0, 0);
		rdpmc_low(pmc, now);
		state->pmc[i].sum += now - state->pmc[i].start;
		state->pmc[i].start = now;
	}
	/* cache->k1.id is still == state->k1.id */
	set_isuspend_cpu(state, cpu);
}

static void perfctr_cpu_iresume(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int cstatus, nrctrs, i;
	int cpu;

	cpu = smp_processor_id();
	cache = &per_cpu_cache[cpu];
	if( cache->k1.id == state->k1.id ) {
		cache->k1.id = 0; /* force reload of cleared EVNTSELs */
		if( is_isuspend_cpu(state, cpu) )
			return; /* skip reload of PERFCTRs */
	}
	cstatus = state->cstatus;
	nrctrs = perfctr_cstatus_nrctrs(cstatus);
	for(i = perfctr_cstatus_nractrs(cstatus); i < nrctrs; ++i) {
		unsigned int pmc = state->pmc[i].map;
		/* If the control wasn't ours we must disable the evntsels
		   before reinitialising the counters, to prevent unexpected
		   counter increments and missed overflow interrupts. */
		if( cache->control.evntsel[pmc] ) {
			cache->control.evntsel[pmc] = 0;
			wrmsr(MSR_K8_EVNTSEL0+pmc, 0, 0);
		}
		wrmsr(MSR_K8_PERFCTR0+pmc, state->pmc[i].start, -1);
	}
	/* cache->k1.id remains != state->k1.id */
}

static void perfctr_cpu_write_control(const struct perfctr_cpu_state *state)
{
	struct per_cpu_cache *cache;
	unsigned int nrctrs, i;

	cache = &per_cpu_cache[smp_processor_id()];
	if( cache->k1.id == state->k1.id ) {
		debug_evntsel_cache(state, cache);
		return;
	}
	nrctrs = perfctr_cstatus_nrctrs(state->cstatus);
	for(i = 0; i < nrctrs; ++i) {
		unsigned int evntsel = state->control.evntsel[i];
		unsigned int pmc = state->pmc[i].map;
		if( evntsel != cache->control.evntsel[pmc] ) {
			cache->control.evntsel[pmc] = evntsel;
			wrmsr(MSR_K8_EVNTSEL0+pmc, evntsel, 0);
		}
	}
	cache->k1.id = state->k1.id;
}

static void k8_clear_counters(void)
{
	clear_msr_range(MSR_K8_EVNTSEL0, 4+4);
}

/*
 * Generic driver for any x86-64 with a working TSC.
 * (Mainly for testing with Screwdriver.)
 */

static int generic_check_control(struct perfctr_cpu_state *state)
{
	if( state->control.nractrs || state->control.nrictrs )
		return -EINVAL;
	return 0;
}

static void generic_clear_counters(void)
{
}

/*
 * Driver methods, internal and exported.
 */

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
		if( (int)state->pmc[pmc].start >= 0 ) { /* XXX: ">" ? */
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
		if( state->control.ireset[i] >= 0 )
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
		printk(KERN_ERR "perfctr: BUG! updating control in"
		       " perfctr %p on cpu %u while it has cstatus %x"
		       " (pid %d, comm %s)\n",
		       state, smp_processor_id(), state->cstatus,
		       current->pid, current->comm);
#endif
}

static int (*check_control)(struct perfctr_cpu_state*);
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

static int __init amd_init(void)
{
	static char k8_name[] __initdata = "AMD K8";
	static char k8c_name[] __initdata = "AMD K8C";

	if( !cpu_has_tsc )
		return -ENODEV;
	if( boot_cpu_data.x86 != 15 )
		return -ENODEV;
	if( (boot_cpu_data.x86_model > 5) ||
	    (boot_cpu_data.x86_model >= 4 && boot_cpu_data.x86_mask >= 8) ) {
		perfctr_info.cpu_type = PERFCTR_X86_AMD_K8C;
		perfctr_cpu_name = k8c_name;
	} else {
		perfctr_info.cpu_type = PERFCTR_X86_AMD_K8;
		perfctr_cpu_name = k8_name;
	}
	check_control = k8_check_control;
	clear_counters = k8_clear_counters;
	if( cpu_has_apic )
		perfctr_info.cpu_features |= PERFCTR_FEATURE_PCINT;
	return 0;
}

/* For testing on Screwdriver. */
static int __init generic_init(void)
{
	static char generic_name[] __initdata = "Generic x86-64 with TSC";
	if( !cpu_has_tsc )
		return -ENODEV;
	perfctr_info.cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	perfctr_info.cpu_type = PERFCTR_X86_GENERIC;
	perfctr_cpu_name = generic_name;
	check_control = generic_check_control;
	clear_counters = generic_clear_counters;
	return 0;
}

static void perfctr_cpu_init_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
	if( cpu_has_apic )
		apic_write(APIC_LVTPC, LOCAL_PERFCTR_VECTOR);
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		set_in_cr4_local(X86_CR4_PCE);
}

static void perfctr_cpu_exit_one(void *ignore)
{
	/* PREEMPT note: when called via smp_call_function(),
	   this is in IRQ context with preemption disabled. */
	perfctr_cpu_clear_counters();
	if( cpu_has_apic )
		apic_write(APIC_LVTPC, APIC_DM_NMI | APIC_LVT_MASKED);
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		clear_in_cr4_local(X86_CR4_PCE);
}

#if defined(CONFIG_PM)

static void perfctr_pm_suspend(void)
{
	/* XXX: clear control registers */
	printk("perfctr: PM suspend\n");
}

static void perfctr_pm_resume(void)
{
	/* XXX: reload control registers */
	printk("perfctr: PM resume\n");
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,71)

#include <linux/sysdev.h>

static int perfctr_device_suspend(struct sys_device *dev, u32 state)
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
	set_kset_name("perfctr"),
	.resume		= perfctr_device_resume,
	.suspend	= perfctr_device_suspend,
};

static struct sys_device device_perfctr = {
	.id	= 0,
	.cls	= &perfctr_sysclass,
};

static void x86_pm_init(void)
{
	if( sysdev_class_register(&perfctr_sysclass) == 0 )
		sysdev_register(&device_perfctr);
}

static void x86_pm_exit(void)
{
	sysdev_unregister(&device_perfctr);
	sysdev_class_unregister(&perfctr_sysclass);
}

#else	/* 2.4 kernel */

static int x86_pm_callback(struct pm_dev *dev, pm_request_t rqst, void *data)
{
	switch( rqst ) {
	case PM_SUSPEND:
		perfctr_pm_suspend();
		break;
	case PM_RESUME:
		perfctr_pm_resume();
		break;
	}
	return 0;
}

static struct pm_dev *x86_pmdev;

static void x86_pm_init(void)
{
	x86_pmdev = apic_pm_register(PM_SYS_DEV, 0, x86_pm_callback);
}

static void x86_pm_exit(void)
{
	if( x86_pmdev ) {
		apic_pm_unregister(x86_pmdev);
		x86_pmdev = NULL;
	}
}

#endif	/* 2.4 kernel */

#else

static inline void x86_pm_init(void) { }
static inline void x86_pm_exit(void) { }

#endif	/* CONFIG_PM */

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,71)
static void disable_lapic_nmi_watchdog(void)
{
#ifdef CONFIG_PM
	if( nmi_pmdev ) {
		apic_pm_unregister(nmi_pmdev);
		nmi_pmdev = 0;
	}
#endif
}
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,6)
static int reserve_lapic_nmi(void)
{
	int ret = 0;
	if( nmi_perfctr_msr ) {
		nmi_perfctr_msr = 0;
		disable_lapic_nmi_watchdog();
		ret = 1;
	}
	return ret;
}

static inline void release_lapic_nmi(void) { }
#endif

static void do_init_tests(void)
{
#ifdef CONFIG_PERFCTR_INIT_TESTS
	if( reserve_lapic_nmi() >= 0 ) {
		perfctr_x86_init_tests();
		release_lapic_nmi();
	}
#endif
}

static void invalidate_per_cpu_cache(void)
{
	/*
	 * per_cpu_cache[] is initialised to contain "impossible"
	 * evntsel values guaranteed to differ from anything accepted
	 * by perfctr_cpu_update_control(). This way, initialisation of
	 * a CPU's evntsel MSRs will happen automatically the first time
	 * perfctr_cpu_write_control() executes on it.
	 * All-bits-one works for all currently supported processors.
	 * The memset also sets the ids to -1, which is intentional.
	 */
	memset(per_cpu_cache, ~0, sizeof per_cpu_cache);
}

int __init perfctr_cpu_init(void)
{
	int err = -ENODEV;

	preempt_disable();

	/* RDPMC and RDTSC are on by default. They will be disabled
	   by the init procedures if necessary. */
	perfctr_info.cpu_features = PERFCTR_FEATURE_RDPMC | PERFCTR_FEATURE_RDTSC;

	switch( boot_cpu_data.x86_vendor ) {
	case X86_VENDOR_AMD:
		err = amd_init();
		break;
	}
	if( err ) {
		err = generic_init();	/* last resort */
		if( err )
			goto out;
	}
	do_init_tests();
#if 0
	/*
	 * Put the hardware in a sane state:
	 * - clear perfctr MSRs
	 * - set up APIC_LVTPC
	 * - set CR4.PCE [on permanently due to __flush_tlb_global()]
	 * - install our default interrupt handler
	 */
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features |= X86_CR4_PCE;
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
#endif

	invalidate_per_cpu_cache();

	perfctr_info.cpu_khz = perfctr_cpu_khz();
	perfctr_info.tsc_to_cpu_mult = 1;

 out:
	preempt_enable();
	return err;
}

void __exit perfctr_cpu_exit(void)
{
#if 0
	preempt_disable();
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features &= ~X86_CR4_PCE;
	perfctr_cpu_exit_one(NULL);
	smp_call_function(perfctr_cpu_exit_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_exit();
	/* XXX: restart nmi watchdog? */
	preempt_enable();
#endif
}

/****************************************************************
 *								*
 * Hardware reservation.					*
 *								*
 ****************************************************************/

static DECLARE_MUTEX(mutex);
static const char *current_service = 0;

const char *perfctr_cpu_reserve(const char *service)
{
	const char *ret;

	down(&mutex);
	ret = current_service;
	if( ret )
		goto out_up;
	ret = "unknown driver (oprofile?)";
	if( reserve_lapic_nmi() < 0 )
		goto out_up;
	current_service = service;
	__module_get(THIS_MODULE);
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features |= X86_CR4_PCE;
	on_each_cpu(perfctr_cpu_init_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_init();
	ret = NULL;
 out_up:
	up(&mutex);
	return ret;
}

void perfctr_cpu_release(const char *service)
{
	down(&mutex);
	if( service != current_service ) {
		printk(KERN_ERR "%s: attempt by %s to release while reserved by %s\n",
		       __FUNCTION__, service, current_service);
		goto out_up;
	}
	/* power down the counters */
	invalidate_per_cpu_cache();
	if( perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC )
		mmu_cr4_features &= ~X86_CR4_PCE;
	on_each_cpu(perfctr_cpu_exit_one, NULL, 1, 1);
	perfctr_cpu_set_ihandler(NULL);
	x86_pm_exit();
	current_service = 0;
	release_lapic_nmi();
	module_put(THIS_MODULE);
 out_up:
	up(&mutex);
}
