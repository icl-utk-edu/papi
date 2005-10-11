/* $Id$
 * Performance-monitoring counters driver.
 * PPC32-specific kernel-resident code.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include "ppc_compat.h"
#include "compat.h"

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
static void perfctr_default_ihandler(unsigned long pc)
{
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;

void do_perfctr_interrupt(struct pt_regs *regs)
{
	preempt_disable();
	(*perfctr_ihandler)(instruction_pointer(regs));
	preempt_enable_no_resched();
}

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}

#ifdef CONFIG_PERFCTR_MODULE
EXPORT_SYMBOL(perfctr_cpu_set_ihandler);
#endif /* MODULE */
#endif /* CONFIG_PERFCTR_INTERRUPT_SUPPORT */
