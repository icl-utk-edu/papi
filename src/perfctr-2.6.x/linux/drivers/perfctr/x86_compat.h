/* $Id$
 * Performance-monitoring counters driver.
 * x86/x86_64-specific compatibility definitions for 2.4/2.6 kernels.
 *
 * Copyright (C) 2000-2006  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,18)

/* missing from <asm-i386/cpufeature.h> */
#define cpu_has_msr	boot_cpu_has(X86_FEATURE_MSR)

#else	/* 2.4 */

/* missing from <asm-i386/processor.h> */
#ifndef cpu_has_mmx	/* added in 2.4.22-pre3 */
#define cpu_has_mmx	(test_bit(X86_FEATURE_MMX,  boot_cpu_data.x86_capability))
#endif
#define cpu_has_msr	(test_bit(X86_FEATURE_MSR,  boot_cpu_data.x86_capability))
#ifndef cpu_has_ht	/* added in 2.4.22-pre3 */
#define cpu_has_ht	(test_bit(28, boot_cpu_data.x86_capability))
#endif

#endif	/* 2.4 */

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

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,16) && !defined(CONFIG_X86_64)
/* Stop speculative execution */
static inline void sync_core(void)
{
	int tmp;
	asm volatile("cpuid" : "=a" (tmp) : "0" (1) : "ebx","ecx","edx","memory");
}
#endif

/* cpuid_count() was added in the 2.6.12 standard kernel, but it's been
   backported to some distribution kernels including the 2.6.9-22 RHEL4
   kernel. For simplicity, always use our version in older kernels. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,12)
/* Some CPUID calls want 'count' to be placed in ecx */
static inline void perfctr_cpuid_count(int op, int count, int *eax, int *ebx, int *ecx,
	       	int *edx)
{
	__asm__("cpuid"
		: "=a" (*eax),
		  "=b" (*ebx),
		  "=c" (*ecx),
		  "=d" (*edx)
		: "0" (op), "c" (count));
}
#undef cpuid_count
#define cpuid_count(o,c,eax,ebx,ecx,edx)	perfctr_cpuid_count((o),(c),(eax),(ebx),(ecx),(edx))
#endif

extern unsigned int perfctr_cpu_khz(void);
