/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific compatibility definitions for 2.2/2.4 kernels.
 *
 * Copyright (C) 2000  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

/* missing from <asm-i386/processor.h> */
#define cpu_has_mmx	(test_bit(X86_FEATURE_MMX,  boot_cpu_data.x86_capability))
#define cpu_has_msr	(test_bit(X86_FEATURE_MSR,  boot_cpu_data.x86_capability))
#define clr_cap_tsc()	(clear_bit(X86_FEATURE_TSC, boot_cpu_data.x86_capability))

#else	/* 2.4 simulation for 2.2 */

#define cpu_has_mmx	(boot_cpu_data.x86_capability & X86_FEATURE_MMX)
#define cpu_has_msr	(boot_cpu_data.x86_capability & X86_FEATURE_MSR)
#define cpu_has_tsc	(boot_cpu_data.x86_capability & X86_FEATURE_TSC)
#define clr_cap_tsc()	(boot_cpu_data.x86_capability &= ~X86_FEATURE_TSC)

#endif	/* 2.4 simulation for 2.2 */

/*
 * Access and export CPU speed.
 * 2.2.xx: cpu_khz when xx >= 16, otherwise cpu_hz
 * 2.4.xx: cpu_khz when xx >= test9-pre, otherwise cpu_hz
 * fast_gettimeoffset_quotient has the same meaning in all
 * kernels, but is local to arch/i386/kernel/time.c in <= 2.2.14.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,2,16)
extern unsigned long cpu_khz;
#define EXPORT_cpu_khz	EXPORT_SYMBOL(cpu_khz)
#define get_cpu_khz()	(cpu_khz)
#else
extern unsigned long cpu_hz;
#define EXPORT_cpu_khz	EXPORT_SYMBOL(cpu_hz)
#define get_cpu_khz()	(cpu_hz/1000)
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,3,40)
/* mmu_cr4_features is __initdata */

#define MAYBE_EXPORT_mmu_cr4_features	/*empty*/

static __inline__ void __write_cr4(unsigned int x)
{
	__asm__ __volatile__("movl %0,%%cr4" : : "r"(x));
}

static __inline__ unsigned int __read_cr4(void)
{
	unsigned int x;
	__asm__ __volatile__("movl %%cr4,%0" : "=r"(x));
	return x;
}

static __inline__ void __set_cr4_pce(void *ignore)
{
	__write_cr4(__read_cr4() | X86_CR4_PCE);
}

static __inline__ void __clear_cr4_pce(void *ignore)
{
	__write_cr4(__read_cr4() & ~X86_CR4_PCE);
}

static __inline__ void set_cr4_pce_global(void)
{
	__set_cr4_pce(NULL);
	smp_call_function(__set_cr4_pce, NULL, 1, 1);
}

static __inline__ void clear_cr4_pce_global(void)
{
	__clear_cr4_pce(NULL);
	smp_call_function(__clear_cr4_pce, NULL, 1, 1);
}

#define set_cr4_tsd_local()	__write_cr4(__read_cr4() | X86_CR4_TSD)

#else	/* >= 2.3.40, mmu_cr4_features is a global variable */

#define MAYBE_EXPORT_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)

#ifdef CONFIG_SMP
static void update_cr4(void *ignore)
{
	__asm__ __volatile__("movl %0,%%cr4" : : "r"(mmu_cr4_features));
}
#endif

static __inline__ void set_cr4_pce_global(void)
{
	set_in_cr4(X86_CR4_PCE);
	smp_call_function(update_cr4, NULL, 1, 1);
}

static __inline__ void clear_cr4_pce_global(void)
{
	clear_in_cr4(X86_CR4_PCE);
	smp_call_function(update_cr4, NULL, 1, 1);
}

#define set_cr4_tsd_local()	set_in_cr4(X86_CR4_TSD)

#endif
