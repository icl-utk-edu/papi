/* $Id$
 * Performance-monitoring counters driver.
 * x86_64-specific compatibility definitions for 2.4/2.5 kernels.
 *
 * Copyright (C) 2003  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,0)

/* irq_enter() and irq_exit() take two parameters in 2.4. However,
   we only use them to disable preemption in the interrupt handler,
   which isn't needed in non-preemptive 2.4 kernels. */
#ifdef CONFIG_PREEMPT
#error "not yet ported to 2.4+PREEMPT"
#endif
#undef irq_enter
#undef irq_exit
#define irq_enter()	do{}while(0)
#define irq_exit()	do{}while(0)

#endif

extern unsigned int perfctr_cpu_khz(void);
