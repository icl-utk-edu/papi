/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.4 kernels.
 *
 * Copyright (C) 1999-2003  Mikael Pettersson
 */
#include <linux/mm.h>	/* for remap_page_range() [redefined here] */

#include "cpumask.h"

/* 2.4.18-redhat had BUG_ON() before 2.4.19 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,19) && !defined(BUG_ON)
#define BUG_ON(condition)	do { if ((condition) != 0) BUG(); } while(0)
#endif

/* 2.4.18-redhat had set_cpus_allowed() before 2.4.21-pre5 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,21) && !defined(HAVE_SET_CPUS_ALLOWED)
#if defined(CONFIG_SMP)
extern void set_cpus_allowed(struct task_struct*, unsigned long);
#else
#define set_cpus_allowed(tsk, mask)	do{}while(0)
#endif
#endif

/* 2.4.20-8-redhat added cpu_online() */
#if !defined(cpu_online)
#define cpu_online(cpu)		(cpu_online_map & (1UL << (cpu)))
#endif

/* 2.4.20-8-redhat added put_task_struct() */
#if defined(put_task_struct)	/* RH 2.4.20-8 */
#define EXPORT_SYMBOL___put_task_struct	EXPORT_SYMBOL(__put_task_struct)
#else				/* standard 2.4 */
#define put_task_struct(tsk)	free_task_struct((tsk))
#define EXPORT_SYMBOL___put_task_struct	/*empty*/
#endif

/* remap_page_range() changed in 2.5.3-pre1 and 2.4.20-8-redhat */
#if !defined(HAVE_5ARG_REMAP_PAGE_RANGE)
static inline int perfctr_remap_page_range(struct vm_area_struct *vma, unsigned long from, unsigned long to, unsigned long size, pgprot_t prot)
{
	return remap_page_range(from, to, size, prot);
}
#undef remap_page_range
#define remap_page_range(vma,from,to,size,prot) perfctr_remap_page_range((vma),(from),(to),(size),(prot))
#endif

/* 2.4.22-rc1 added EXPORT_SYMBOL(mmu_cr4_features) */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,22) || defined(HAVE_EXPORT_mmu_cr4_features)
#define EXPORT_SYMBOL_mmu_cr4_features	/*empty*/
#else
#define EXPORT_SYMBOL_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)
#endif

/* not in 2.4 proper, but some people use 2.4 with preemption patches */
#ifdef CONFIG_PREEMPT
#error "not yet ported to 2.4+PREEMPT"
#endif
#ifndef preempt_disable
#define preempt_disable()	do{}while(0)
#define preempt_enable()	do{}while(0)
#endif

#ifdef MODULE
#define __module_get(module)	do { if ((module)) __MOD_INC_USE_COUNT((module)); } while(0)
#define module_put(module)	do { if ((module)) __MOD_DEC_USE_COUNT((module)); } while(0)
#else
#define __module_get(module)	do{}while(0)
#define module_put(module)	do{}while(0)
#endif

#define MODULE_ALIAS(alias)	/*empty*/

/* introduced in 2.5.64; backported to 2.4.22-1.2115.nptl (FC1) */
static inline int
perfctr_on_each_cpu(void (*func) (void *info), void *info,
		    int retry, int wait)
{
        int ret = 0;

        preempt_disable();
        ret = smp_call_function(func, info, retry, wait);
        func(info);
        preempt_enable();
        return ret;
}
#undef on_each_cpu
#define on_each_cpu(f,i,r,w)	perfctr_on_each_cpu((f),(i),(r),(w))
