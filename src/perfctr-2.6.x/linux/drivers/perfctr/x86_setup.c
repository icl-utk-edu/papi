/* $Id: x86_setup.c,v 1.47.2.9 2009/01/23 17:21:20 mikpe Exp $
 * Performance-monitoring counters driver.
 * x86/x86_64-specific kernel-resident code.
 *
 * Copyright (C) 1999-2007, 2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include <asm/fixmap.h>
#include <asm/apic.h>
#include "x86_compat.h"
#include "compat.h"

#ifdef CONFIG_X86_LOCAL_APIC
static void perfctr_default_ihandler(unsigned long pc)
{
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;
static unsigned int interrupts_masked[NR_CPUS] __cacheline_aligned;

void __perfctr_cpu_mask_interrupts(void)
{
	interrupts_masked[smp_processor_id()] = 1;
}

void __perfctr_cpu_unmask_interrupts(void)
{
	interrupts_masked[smp_processor_id()] = 0;
}

asmlinkage void smp_perfctr_interrupt(struct pt_regs *regs)
{
	/* PREEMPT note: invoked via an interrupt gate, which
	   masks interrupts. We're still on the originating CPU. */
	/* XXX: recursive interrupts? delay the ACK, mask LVTPC, or queue? */
	ack_APIC_irq();
	if (interrupts_masked[smp_processor_id()])
		return;
	irq_enter();
	(*perfctr_ihandler)(instruction_pointer(regs));
	irq_exit();
}

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}
#endif

#if defined(__x86_64__) || LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,13)
extern unsigned int cpu_khz;
#else
extern unsigned long cpu_khz;
#endif

unsigned int perfctr_cpu_khz(void)
{
	return cpu_khz;
}

#ifdef CONFIG_PERFCTR_MODULE
EXPORT_SYMBOL_mmu_cr4_features;
EXPORT_SYMBOL(perfctr_cpu_khz);

#ifdef CONFIG_X86_LOCAL_APIC

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,19)
#include <asm/nmi.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
EXPORT_SYMBOL(disable_lapic_nmi_watchdog);
EXPORT_SYMBOL(enable_lapic_nmi_watchdog);
#else
EXPORT_SYMBOL(setup_apic_nmi_watchdog);
EXPORT_SYMBOL(stop_apic_nmi_watchdog);
#endif
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,6)
EXPORT_SYMBOL(nmi_perfctr_msr);
#endif

EXPORT_SYMBOL(__perfctr_cpu_mask_interrupts);
EXPORT_SYMBOL(__perfctr_cpu_unmask_interrupts);
EXPORT_SYMBOL(perfctr_cpu_set_ihandler);
#endif /* CONFIG_X86_LOCAL_APIC */

#endif /* MODULE */
