/* $Id$
 * x86 performance-monitoring counters driver.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
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

static union {	/* Support for lazy evntsel MSR updates. */
	struct perfctr_control control;	/* 16 bytes */
	char __align[SMP_CACHE_BYTES];	/* 32 bytes */
} per_cpu_control[NR_CPUS] __cacheline_aligned;

/* Intel P5, Cyrix 6x86MX/MII/III, Centaur WinChip C6/2/3 */
#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12
#define MSR_P5_CTR1		0x13

/* Intel P6 */
#define MSR_P6_PERFCTR0		0xC1
#define MSR_P6_PERFCTR1		0xC2
#define MSR_P6_EVNTSEL0		0x186
#define MSR_P6_EVNTSEL1		0x187

/* AMD K7 Athlon */
#define MSR_K7_EVNTSEL0		0xC0010000	/* .. 0xC0010003 */
#define MSR_K7_PERFCTR0		0xC0010004	/* .. 0xC0010007 */

static __inline__ unsigned int get_cr4(void)
{
	unsigned int cr4;
	__asm__ __volatile__("movl %%cr4,%0" : "=r"(cr4));
	return cr4;
}

#define rdmsrl(msr,low) \
	__asm__ __volatile__("rdmsr" : "=a"(low) : "c"(msr) : "edx")
#define rdpmcl(ctr,low) \
	__asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")

/* Detected CPU type. */
unsigned char perfctr_cpu_type;

/* CPU features optionally supported.
 * RDPMC and RDTSC are on by default. They will be disabled
 * by the init procedures if necessary.
 */
unsigned char perfctr_cpu_features
	= PERFCTR_FEATURE_RDPMC | PERFCTR_FEATURE_RDTSC;

/* CPU/TSC speed in kHz. */
unsigned long perfctr_cpu_khz;

/****************************************************************
 *								*
 * Driver procedures.						*
 *								*
 ****************************************************************/

static int p5_check_control(const struct perfctr_control *control)
{
	/* protect reserved and pin control bits */
	if( control->evntsel[0] & 0xFE00FE00 )
		return -EPERM;
	/* CTR1 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x00C00000 )
		return 3;
	/* CTR0 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x000000C0 )
		return 2;
	/* Only TSC is on. */
	return 1;
}

static void p5_write_control(int nrctrs, const struct perfctr_control *control)
{
	struct perfctr_control *cpu;
	unsigned evntsel;

	if( nrctrs <= 1 )	/* no evntsel to write if only TSC is on */
		return;
	cpu = &per_cpu_control[smp_processor_id()].control;
	evntsel = control->evntsel[0];
	if( cpu->evntsel[0] != evntsel ) {
		cpu->evntsel[0] = evntsel;
		wrmsr(MSR_P5_CESR, evntsel, 0);
	}
}

static void p5_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	rdtscl(ctrs->ctr[0]);
	if( nrctrs >= 2 ) {
		rdmsrl(MSR_P5_CTR0, ctrs->ctr[1]);
		if( nrctrs == 3 )
			rdmsrl(MSR_P5_CTR1, ctrs->ctr[2]);
	}
}

static void p5mmx_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	rdtscl(ctrs->ctr[0]);
	if( nrctrs >= 2 ) {
		rdpmcl(0, ctrs->ctr[1]);
		if( nrctrs == 3 )
			rdpmcl(1, ctrs->ctr[2]);
	}
}

static int p6_check_control(const struct perfctr_control *control)
{
	/* protect reserved, interrupt control, and pin control bits */
	if( control->evntsel[0] & 0x00380000 )
		return -EPERM;
	if( control->evntsel[1] & 0x00780000 )
		return -EPERM;
	/* check global enable bit */
	if( control->evntsel[0] & 0x00400000 ) {
		/* check CPL field */
		if( control->evntsel[1] & 0x00030000 )
			return 3;
		if( control->evntsel[0] & 0x00030000 )
			return 2;
	}
	return 1;
}

static void p6_write_control(int nrctrs, const struct perfctr_control *control)
{
	struct perfctr_control *cpu;
	unsigned evntsel;

	cpu = &per_cpu_control[smp_processor_id()].control;
	if( nrctrs == 3 ) {
		evntsel = control->evntsel[1];
		if( evntsel != cpu->evntsel[1] ) {
			cpu->evntsel[1] = evntsel;
			wrmsr(MSR_P6_EVNTSEL1, evntsel, 0);
		}
	}	
	if( nrctrs >= 2 ) {
		evntsel = control->evntsel[0];
		if( evntsel != cpu->evntsel[0] ) {
			cpu->evntsel[0] = evntsel;
			wrmsr(MSR_P6_EVNTSEL0, evntsel, 0);
		}
	}
}

static int k7_check_control(const struct perfctr_control *control)
{
	int i, last_pmc_on;

	last_pmc_on = -1;
	for(i = 0; i < 4; ++i) {
		/* protect reserved, interrupt control, and pin control bits */
		if( control->evntsel[i] & 0x00380000 )
			return -EPERM;
		/* check enable bit */
		if( !(control->evntsel[i] & 0x00400000) )
			continue;
		/* check CPL field */
		if( control->evntsel[i] & 0x00030000 )
			last_pmc_on = i;
	}
	return last_pmc_on + 2;
}

static void k7_write_control(int nrctrs, const struct perfctr_control *control)
{
	struct perfctr_control *cpu;
	int i;
	unsigned evntsel;

	cpu = &per_cpu_control[smp_processor_id()].control;
	for(i = nrctrs - 1; --i >= 0;) {	/* -1 for TSC */
		evntsel = control->evntsel[i];
		if( evntsel != cpu->evntsel[i] ) {
			cpu->evntsel[i] = evntsel;
			wrmsr(MSR_K7_EVNTSEL0+i, evntsel, 0);
		}
	}
}

static void k7_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	int i;

	rdtscl(ctrs->ctr[0]);
	for(i = nrctrs - 1; --i >= 0;)	/* -1 for TSC */
		rdpmcl(i, ctrs->ctr[i+1]);
}

static int mii_check_control(const struct perfctr_control *control)
{
	/* Protect reserved and pin control bits.
	 * CESR bits 9 and 25 are reserved in the Cyrix III,
	 * but not in the earlier processors where they control
	 * the external PM pins. In either case, we do not
	 * allow the user to touch them.
	 */
	if( control->evntsel[0] & 0xFA00FA00 )
		return -EPERM;
	/* CTR1 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x00C00000 )
		return 3;
	/* CTR0 is on if its CPL field is non-zero */
	if( control->evntsel[0] & 0x000000C0 )
		return 2;
	/* Only TSC is on. */
	return 1;
}

#if defined(CONFIG_PERFCTR_WINCHIP)
static int c6_check_control(const struct perfctr_control *control)
{
	/* protect reserved bits */
	if( control->evntsel[0] & 0xFF00FF00 )
		return -EPERM;
	return 3;	/* fake TSC and two perfctrs */
}

static void c6_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	ctrs->ctr[0] = 0;
	rdpmcl(0, ctrs->ctr[1]);
	rdpmcl(1, ctrs->ctr[2]);
}
#endif	/* CONFIG_PERFCTR_WINCHIP */

static int generic_check_control(const struct perfctr_control *control)
{
	return 1;
}

static void generic_write_control(int nrctrs, const struct perfctr_control *control)
{
}

static void generic_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	rdtscl(ctrs->ctr[0]);
}

static void redirect_call(void *ra, void *to)
{
	/* we can only redirect `call near relative' instructions */
	if( *((unsigned char*)ra - 5) != 0xE8 ) {
		printk(KERN_INFO __FILE__ ":" __FUNCTION__
		       ": unable to redirect caller %p to %p\n",
		       ra, to);
		return;
	}
	*(int*)((char*)ra - 4) = (char*)to - (char*)ra;
}

static int (*check_control)(const struct perfctr_control *control);
int perfctr_cpu_check_control(const struct perfctr_control *control)
{
	redirect_call(__builtin_return_address(0), check_control);
	return check_control(control);
}

static void (*write_control)(int nrctrs, const struct perfctr_control *control);
void perfctr_cpu_write_control(int nrctrs, const struct perfctr_control *control)
{
	redirect_call(__builtin_return_address(0), write_control);
	return write_control(nrctrs, control);
}

static void (*read_counters)(int, struct perfctr_low_ctrs*);
void perfctr_cpu_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs)
{
	redirect_call(__builtin_return_address(0), read_counters);
	return read_counters(nrctrs, ctrs);
}

/****************************************************************
 *								*
 * Processor detection and initialisation procedures.		*
 *								*
 ****************************************************************/

static int __init intel_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	switch( boot_cpu_data.x86 ) {
	case 5:
		if( cpu_has_mmx ) {
			perfctr_cpu_type = PERFCTR_X86_INTEL_P5MMX;
			read_counters = p5mmx_read_counters;
		} else {
			perfctr_cpu_type = PERFCTR_X86_INTEL_P5;
			perfctr_cpu_features &= ~PERFCTR_FEATURE_RDPMC;
			read_counters = p5_read_counters;
		}
		write_control = p5_write_control;
		check_control = p5_check_control;
		perfctr_p5_init_tests();
		return 0;
	case 6:
		if( boot_cpu_data.x86_model >= 7 )	/* PIII */
			perfctr_cpu_type = PERFCTR_X86_INTEL_PIII;
		else if( boot_cpu_data.x86_model >= 3 )	/* PII or Celeron */
			perfctr_cpu_type = PERFCTR_X86_INTEL_PII;
		else
			perfctr_cpu_type = PERFCTR_X86_INTEL_P6;
		read_counters = p5mmx_read_counters;
		write_control = p6_write_control;
		check_control = p6_check_control;
		perfctr_p6_init_tests();
		return 0;
	}
	return -ENODEV;
}

static int __init amd_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	switch( boot_cpu_data.x86 ) {
	case 6:	/* K7 Athlon. Model 1 does not have a local APIC. */
		perfctr_cpu_type = PERFCTR_X86_AMD_K7;
		read_counters = k7_read_counters;
		write_control = k7_write_control;
		check_control = k7_check_control;
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
		perfctr_cpu_type = PERFCTR_X86_CYRIX_MII;
		read_counters = p5mmx_read_counters;
		write_control = p5_write_control;
		check_control = mii_check_control;
		perfctr_mii_init_tests();
		return 0;
	}
	return -ENODEV;
}

static int __init centaur_init(void)
{
#if defined(CONFIG_PERFCTR_WINCHIP)
	switch( boot_cpu_data.x86 ) {
	case 5:
		switch( boot_cpu_data.x86_model ) {
		case 4: /* WinChip C6 */
			perfctr_cpu_type = PERFCTR_X86_WINCHIP_C6;
			break;
		case 8: /* WinChip 2, 2A, or 2B */
		case 9: /* WinChip 3, a 2A with larger cache and lower voltage */
			perfctr_cpu_type = PERFCTR_X86_WINCHIP_2;
			break;
		default:
			return -ENODEV;
		}
		/*
		 * TSC must be inaccessible for perfctrs to work.
		 */
		if( !(get_cr4() & X86_CR4_TSD) || cpu_has_tsc )
			return -ENODEV;
		perfctr_cpu_features &= ~PERFCTR_FEATURE_RDTSC;
		read_counters = c6_read_counters;
		write_control = p5_write_control;
		check_control = c6_check_control;
		perfctr_c6_init_tests();
		return 0;
	}
#endif
	return -ENODEV;
}

static int __init generic_init(void)
{
	if( !cpu_has_tsc )
		return -ENODEV;
	perfctr_cpu_features &= ~PERFCTR_FEATURE_RDPMC;
	perfctr_cpu_type = PERFCTR_X86_GENERIC;
	check_control = generic_check_control;
	write_control = generic_write_control;
	read_counters = generic_read_counters;
	return 0;
}

char *perfctr_cpu_name[] __initdata = {
	"Generic x86 with TSC",
	"Intel Pentium",
	"Intel Pentium MMX",
	"Intel Pentium Pro",
	"Intel Pentium II",
	"Intel Pentium III",
	"Cyrix 6x86MX/MII/III",
	"WinChip C6",
	"WinChip 2/3",
	"AMD K7",
};

int __init perfctr_cpu_init(void)
{
	int err = -ENODEV;
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
	 * per_cpu_control[] is initialised to contain "impossible"
	 * evntsel values guaranteed to differ from anything accepted
	 * by perfctr_cpu_check_control(). This way, initialisation of
	 * a CPU's evntsel MSRs will happen automatically the first time
	 * perfctr_cpu_write_control() executes on it.
	 * All-bits-one works for all currently supported processors.
	 * [XXX: Should be set up by the cpu-specific init procedure.]
	 */
	memset(per_cpu_control, ~0, sizeof per_cpu_control);

	/*
	 * Set CR4.PCE, if the processor supports user-mode RDPMC.
	 * CR4.PCE is enabled globally, for two reasons:
	 *
	 * 1. Writing %cr4 to toggle CR4.PCE at process suspend/resume
	 *    is expensive (a Pentium II/III needs 42 cycles just to
	 *    write to %cr4).
	 *
	 * 2. Starting with the 2.3.40 kernel, %cr4 is written to from
	 *    a single global variable mmu_cr4_features in post-boot
	 *    operation. A change to CR4.PCE on one CPU would be undone
	 *    asynchronously whenever flush_tlb_all() is called.
	 */
	if( perfctr_cpu_features & PERFCTR_FEATURE_RDPMC )
		set_cr4_pce_global();

	perfctr_cpu_khz = get_cpu_khz();

	return 0;
}

void __exit perfctr_cpu_exit(void)
{
	clear_cr4_pce_global();
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
		current_service = 0;
		MOD_DEC_USE_COUNT;
	}
}
