/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific compatibility definitions for 2.2/2.4 kernels.
 *
 * Copyright (C) 2000-2001  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/version.h>

/* 2.4.9-ac3 added {read,write}_cr4() macros in <asm-i386/system.h> */
#if !defined(write_cr4)
static inline void write_cr4(unsigned int x)
{
	__asm__ __volatile__("movl %0,%%cr4" : : "r"(x));
}

static inline unsigned int read_cr4(void)
{
	unsigned int x;
	__asm__ __volatile__("movl %%cr4,%0" : "=r"(x));
	return x;
}
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

/* missing from <asm-i386/processor.h> */
#define cpu_has_mmx	(test_bit(X86_FEATURE_MMX,  boot_cpu_data.x86_capability))
#define cpu_has_msr	(test_bit(X86_FEATURE_MSR,  boot_cpu_data.x86_capability))

#else	/* 2.4 simulation for 2.2 */

#define cpu_has_mmx	(boot_cpu_data.x86_capability & X86_FEATURE_MMX)
#define cpu_has_msr	(boot_cpu_data.x86_capability & X86_FEATURE_MSR)
#define cpu_has_tsc	(boot_cpu_data.x86_capability & X86_FEATURE_TSC)

#define X86_CR4_TSD	0x0004
#define X86_CR4_PCE	0x0100

unsigned long mmu_cr4_features;	/*fake*/

#endif	/* 2.4 simulation for 2.2 */

extern unsigned long cpu_khz;
