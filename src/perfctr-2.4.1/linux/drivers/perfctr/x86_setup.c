/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific kernel-resident code.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include "x86_compat.h"

#ifdef CONFIG_PERFCTR_VIRTUAL	/* XXX: actually generic, not x86-specific */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,32)
#include <linux/ptrace.h>
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,15)
#include <linux/mm.h>
#else	/* < 2.4.15 */
#include "compat.h"
int ptrace_check_attach(struct task_struct *child, int kill)
{
	if (!TASK_IS_PTRACED(child))
		return -ESRCH;

	if (child->p_pptr != current)
		return -ESRCH;

	if (!kill) {
		if (child->state != TASK_STOPPED)
			return -ESRCH;
#ifdef CONFIG_SMP
		/* Make sure the child gets off its CPU.. */
		for (;;) {
			task_lock(child);
			if (!task_has_cpu(child))
				break;
			task_unlock(child);
			do {
				if (child->state != TASK_STOPPED)
					return -ESRCH;
				barrier();
				cpu_relax();
			} while (task_has_cpu(child));
		}
		task_unlock(child);
#endif		
	}

	/* All systems go.. */
	return 0;
}
#endif	/* < 2.4.15 */
EXPORT_SYMBOL(ptrace_check_attach);
#endif

#if PERFCTR_INTERRUPT_SUPPORT
unsigned int apic_lvtpc_irqs[NR_CPUS];

static void perfctr_default_ihandler(unsigned long pc)
{
	++apic_lvtpc_irqs[smp_processor_id()];
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;

void do_perfctr_interrupt(struct pt_regs *regs)
{
	/* XXX: 2.5.28+: irq_enter() or preempt_disable() here */
	/* XXX: should be rewritten in assembly and inlined below */
	/* XXX: recursive interrupts? delay the ACK, mask LVTPC, or queue? */
	ack_APIC_irq();
	(*perfctr_ihandler)(regs->eip);
	/* XXX: 2.5.28+: irq_exit() or preempt_enable() here */
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,8)
extern asmlinkage void perfctr_interrupt(void);
#else	/* < 2.5.8 */
#define BUILD_PERFCTR_INTERRUPT(x,v) XBUILD_PERFCTR_INTERRUPT(x,v)
#define XBUILD_PERFCTR_INTERRUPT(x,v) \
asmlinkage void x(void); \
__asm__( \
	"\n.text\n\t" \
	__ALIGN_STR "\n\t" \
	".type " SYMBOL_NAME_STR(x) ",@function\n" \
	".globl " SYMBOL_NAME_STR(x) "\n" \
SYMBOL_NAME_STR(x) ":\n\t" \
	"pushl $" #v "-256\n\t" \
	SAVE_ALL \
	"pushl %esp\n\t" \
	"call " SYMBOL_NAME_STR(do_ ## x) "\n\t" \
	"addl $4,%esp\n\t" \
	"jmp ret_from_intr\n\t" \
	".size " SYMBOL_NAME_STR(x) ",.-" SYMBOL_NAME_STR(x) "\n" \
	".previous\n");
BUILD_PERFCTR_INTERRUPT(perfctr_interrupt,LOCAL_PERFCTR_VECTOR)
#endif	/* < 2.5.8 */

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}
#endif

#ifdef CONFIG_PERFCTR_MODULE
EXPORT_SYMBOL(mmu_cr4_features);
EXPORT_SYMBOL(cpu_khz);

#ifdef NMI_LOCAL_APIC
EXPORT_SYMBOL(nmi_perfctr_msr);

#ifdef CONFIG_PM
EXPORT_SYMBOL(apic_pm_register);
EXPORT_SYMBOL(apic_pm_unregister);
EXPORT_SYMBOL(nmi_pmdev);
#endif /* CONFIG_PM */

#endif /* NMI_LOCAL_APIC */

#if PERFCTR_INTERRUPT_SUPPORT
EXPORT_SYMBOL(perfctr_cpu_set_ihandler);
#endif /* PERFCTR_INTERRUPT_SUPPORT */

#endif /* MODULE */
