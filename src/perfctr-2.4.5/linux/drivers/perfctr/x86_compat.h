/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific compatibility definitions for 2.2/2.4/2.5 kernels.
 *
 * Copyright (C) 2000-2002  Mikael Pettersson
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

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,18)

/* missing from <asm-i386/cpufeature.h> */
#define cpu_has_msr	boot_cpu_has(X86_FEATURE_MSR)

#elif LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

/* missing from <asm-i386/processor.h> */
#define cpu_has_mmx	(test_bit(X86_FEATURE_MMX,  boot_cpu_data.x86_capability))
#define cpu_has_msr	(test_bit(X86_FEATURE_MSR,  boot_cpu_data.x86_capability))
#define cpu_has_ht	(test_bit(28, boot_cpu_data.x86_capability))

#else	/* 2.4 simulation for 2.2 */

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,2,21)
static inline unsigned int cpuid_eax(unsigned int op)
{
	unsigned int eax;

	__asm__("cpuid"
		: "=a" (eax)
		: "0" (op)
		: "bx", "cx", "dx");
	return eax;
}

static inline unsigned int cpuid_ebx(unsigned int op)
{
	unsigned int eax, ebx;

	__asm__("cpuid"
		: "=a" (eax), "=b" (ebx)
		: "0" (op)
		: "cx", "dx");
	return ebx;
}
#endif

#define cpu_has_mmx	(boot_cpu_data.x86_capability & X86_FEATURE_MMX)
#define cpu_has_msr	(boot_cpu_data.x86_capability & X86_FEATURE_MSR)
#define cpu_has_tsc	(boot_cpu_data.x86_capability & X86_FEATURE_TSC)
#define cpu_has_ht	(boot_cpu_data.x86_capability & (1 << 28))

#define X86_CR4_TSD	0x0004
#define X86_CR4_PCE	0x0100

unsigned long mmu_cr4_features;	/*fake*/

#endif	/* 2.4 simulation for 2.2 */

/* irq_enter() and irq_exit() take two parameters in 2.4. However,
   we only use them to disable preemption in the interrupt handler,
   which isn't needed in non-preemptive 2.4 kernels. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,0)
#ifdef CONFIG_PREEMPT
#error "not yet ported to 2.4+PREEMPT"
#endif
#undef irq_enter
#undef irq_exit
#define irq_enter()	do{}while(0)
#define irq_exit()	do{}while(0)
#endif

extern unsigned long perfctr_cpu_khz(void);
