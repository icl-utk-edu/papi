/* $Id$
 * Performance-monitoring counters driver.
 * x86_86-specific kernel-resident code.
 *
 * Copyright (C) 2003  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include <asm/fixmap.h>
#include <asm/apic.h>
#include "x86_64_compat.h"
#include "compat.h"

static void perfctr_default_ihandler(unsigned long pc)
{
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;

void do_perfctr_interrupt(struct pt_regs *regs)
{
	/* PREEMPT note: invoked via an interrupt gate, which
	   masks interrupts. We're still on the originating CPU. */
	ack_APIC_irq();
	irq_enter();
	(*perfctr_ihandler)(regs->rip);
	irq_exit();
}

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}

extern unsigned int cpu_khz;

/* Wrapper to avoid namespace clash in RedHat 8.0's 2.4.18-14 kernel. */
unsigned int perfctr_cpu_khz(void)
{
	return cpu_khz;
}

#ifdef CONFIG_PERFCTR_MODULE
EXPORT_SYMBOL_mmu_cr4_features;
EXPORT_SYMBOL(perfctr_cpu_khz);

EXPORT_SYMBOL(nmi_perfctr_msr);

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,71) && defined(CONFIG_PM)
EXPORT_SYMBOL(apic_pm_register);
EXPORT_SYMBOL(apic_pm_unregister);
EXPORT_SYMBOL(nmi_pmdev);
#endif

EXPORT_SYMBOL(perfctr_cpu_set_ihandler);

#endif /* MODULE */
