/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific compatibility definitions for 2.4/2.6 kernels.
 *
 * Copyright (C) 2000-2003  Mikael Pettersson
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

extern unsigned int perfctr_cpu_khz(void);
