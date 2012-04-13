/* $Id: x86_compat.h,v 1.33.2.5 2009/01/23 17:01:02 mikpe Exp $
 * Performance-monitoring counters driver.
 * x86/x86_64-specific compatibility definitions for 2.6 kernels.
 *
 * Copyright (C) 2000-2007, 2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif

/* missing from <asm-i386/cpufeature.h> */
#define cpu_has_msr	boot_cpu_has(X86_FEATURE_MSR)

/* 2.6.24-rc1 changed cpu_data from behaving like an array indexed
   by cpu to being a macro with a cpu parameter. This emulates the
   macro-with-parameter form in older kernels. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,24)
static inline struct cpuinfo_x86 *perfctr_cpu_data(int cpu)
{
	return &cpu_data[cpu];
}
#undef cpu_data
#define cpu_data(cpu)	(*perfctr_cpu_data((cpu)))
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
