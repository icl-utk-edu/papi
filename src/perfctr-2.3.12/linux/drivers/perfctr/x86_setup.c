/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific kernel-resident code.
 *
 * Copyright (C) 1999-2001  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include "x86_compat.h"
#include <linux/perfctr.h>	/* for DEBUG */

#if PERFCTR_INTERRUPT_SUPPORT
unsigned int apic_lvtpc_irqs[NR_CPUS];

static void perfctr_default_ihandler(unsigned long pc)
{
	++apic_lvtpc_irqs[smp_processor_id()];
}

static perfctr_ihandler_t perfctr_ihandler = perfctr_default_ihandler;

static void __attribute__((unused))
do_perfctr_interrupt(struct pt_regs *regs)
{
	/* XXX: should be rewritten in assembly and inlined below */
	/* XXX: recursive interrupts? delay the ACK, mask LVTPC, or queue? */
	ack_APIC_irq();
	(*perfctr_ihandler)(regs->eip);
	/* XXX: on P4 LVTPC must now be unmasked */
}

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

void perfctr_cpu_set_ihandler(perfctr_ihandler_t ihandler)
{
	perfctr_ihandler = ihandler ? ihandler : perfctr_default_ihandler;
}
#endif

#if defined(CONFIG_PERFCTR_DEBUG) && defined(CONFIG_PERFCTR_VIRTUAL)
struct shadow_vperfctr {
	unsigned int pad[512];
	void *magic2;
};

void _vperfctr_set_thread(struct thread_struct *thread, struct vperfctr *perfctr)
{
	thread->perfctr = perfctr;
	if( perfctr ) {
		struct shadow_vperfctr *shadow;
		shadow = (struct shadow_vperfctr*)perfctr;
		shadow->magic2 = &shadow->magic2;
	}
}

struct vperfctr *__vperfctr_get_thread(const struct thread_struct *thread,
				       const char *function)
{
	struct vperfctr *perfctr;
	struct shadow_vperfctr *shadow;

	perfctr = thread->perfctr;
	if( !perfctr )
		return NULL;
	if( (long)perfctr & (4096-1) ) {
		printk(KERN_ERR "%s: BUG! perfctr 0x%08lx is not page aligned (pid %d, comm %s)\n",
		       function, (long)perfctr, current->pid, current->comm);
		return NULL;
	}
	if( ((struct vperfctr_state*)perfctr)->magic != VPERFCTR_MAGIC ) {
		printk(KERN_ERR "%s: BUG! perfctr 0x%08lx has invalid magic 0x%08x\n (pid %d, comm %s)\n",
		       function, (long)perfctr, ((struct vperfctr_state*)perfctr)->magic, current->pid, current->comm);
		return NULL;
	}
	shadow = (struct shadow_vperfctr*)perfctr;
	if( shadow->magic2 != &shadow->magic2 ) {
		printk(KERN_ERR "%s: BUG! perfctr 0x%08lx has invalid magic2 0x%08lx\n (pid %d, comm %s)\n",
		       function, (long)perfctr, (long)shadow->magic2, current->pid, current->comm);
		return NULL;
	}
	return perfctr;
}

#ifdef CONFIG_PERFCTR_MODULE
EXPORT_SYMBOL(_vperfctr_set_thread);
EXPORT_SYMBOL(__vperfctr_get_thread);
#endif

#endif	/* DEBUG && VIRTUAL */

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
