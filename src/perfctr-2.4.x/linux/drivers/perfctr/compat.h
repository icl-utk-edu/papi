/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.2/2.4/2.5 kernels.
 *
 * Copyright (C) 1999-2003  Mikael Pettersson
 */
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,2,17)
#define EXPORT_SYMBOL_pidhash	EXPORT_SYMBOL(pidhash)
#else
#define EXPORT_SYMBOL_pidhash	/*empty*/
#endif

#if defined(CONFIG_X86) && LINUX_VERSION_CODE < KERNEL_VERSION(2,2,18)
static inline void rep_nop(void)
{
	__asm__ __volatile__("rep;nop");
}
#endif

#if defined(CONFIG_X86) && LINUX_VERSION_CODE < KERNEL_VERSION(2,4,11) && !defined(cpu_relax)
#define cpu_relax()		rep_nop()
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,15)
#define task_has_cpu(tsk)	((tsk)->has_cpu)
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,15)
extern int ptrace_check_attach(struct task_struct *child, int kill);
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,2,20)
#define TASK_IS_PTRACED(tsk)	((tsk)->flags & PF_PTRACED)
#else
#define TASK_IS_PTRACED(tsk)	((tsk)->ptrace & PT_PTRACED)
#endif

/* /proc/pid/ inodes changes */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,4)
#include <linux/proc_fs.h>
#define proc_pid_inode_denotes_task(inode,tsk)	\
	((tsk) == PROC_I((inode))->task)
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
#define proc_pid_inode_denotes_task(inode,tsk)	\
	((tsk) == (inode)->u.proc_i.task)
#else
#define proc_pid_inode_denotes_task(inode,tsk)	\
	((tsk)->pid == ((inode)->i_ino >> 16))
#endif

/* remap_page_range() changed in 2.5.3-pre1 and 2.4.20-8-redhat */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,3) && !defined(HAVE_5ARG_REMAP_PAGE_RANGE)
#include <linux/mm.h>
static inline int perfctr_remap_page_range(struct vm_area_struct *vma, unsigned long from, unsigned long to, unsigned long size, pgprot_t prot)
{
	return remap_page_range(from, to, size, prot);
}
#undef remap_page_range
#define remap_page_range(vma,from,to,size,prot) perfctr_remap_page_range((vma),(from),(to),(size),(prot))
#endif

/* module changes in 2.5.48 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,48)
static inline void try_module_get(void *m) { MOD_INC_USE_COUNT; }
#define module_put(MOD)		MOD_DEC_USE_COUNT
#endif

/* added in 2.4.9-ac */
#if !defined(MODULE_LICENSE)
#define MODULE_LICENSE(license)	/*empty*/
#endif

/* preemption wrappers for old kernels */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,0)
#ifdef CONFIG_PREEMPT
#error "not yet ported to 2.4+PREEMPT"
#endif
#define preempt_disable()	do{}while(0)
#define preempt_enable()	do{}while(0)
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

#define EXPORT_SYMBOL_tasklist_lock	/*empty*/

/* 2.4.20-8-redhat added 2.5-like put_task_struct() */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,4) || defined(put_task_struct)
#define EXPORT_SYMBOL___put_task_struct	EXPORT_SYMBOL(__put_task_struct)
#else
#define put_task_struct(tsk)	free_task_struct((tsk))
#define EXPORT_SYMBOL___put_task_struct	/* empty */
#endif

#define vma_pgoff(vma)		((vma)->vm_pgoff)
#define task_thread(tsk)	(&(tsk)->thread)

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,19) && !defined(BUG_ON)
#define BUG_ON(condition)	do { if ((condition) != 0) BUG(); } while(0)
#endif

#define get_cpus_allowed(tsk)	((tsk)->cpus_allowed)
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,21) && !defined(HAVE_SET_CPUS_ALLOWED)
#ifdef CONFIG_SMP
extern void set_cpus_allowed(struct task_struct*, unsigned long);
#else
#define set_cpus_allowed(tsk, mask)	do{}while(0)
#endif
#endif

/* 2.4.22-rc1 added EXPORT_SYMBOL(mmu_cr4_features) */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,22)
#define EXPORT_SYMBOL_mmu_cr4_features	/*empty*/
#else
#define EXPORT_SYMBOL_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)
#endif

#else	/* 2.4 simulation for 2.2 */

#define EXPORT_SYMBOL_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)
#define EXPORT_SYMBOL_tasklist_lock	EXPORT_SYMBOL(tasklist_lock)

#if !defined(CONFIG_KMOD) && defined(request_module)	/* < 2.2.20pre10 */
#undef request_module
static inline int request_module(const char *name) { return -ENOSYS; }
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,2,18)
#ifdef MODULE
#define module_init(x)	int init_module(void) { return x(); }
#define module_exit(x)	void cleanup_module(void) { x(); }
#else
#define module_init(x)	/* explicit call is needed */
#define module_exit(x)	/* empty */
#endif /* MODULE */
#define DECLARE_MUTEX(name)	struct semaphore name = MUTEX
#endif /* < 2.2.18 */

/* {get,put}_task_struct() are unsafe no-ops */
#define get_task_struct(tsk)	do{}while(0)
#define put_task_struct(tsk)	do{}while(0)
#define EXPORT_SYMBOL___put_task_struct	/* empty */

/* XXX: is this the correct 2.2 replacement for task_{,un}lock() ??? */
#define task_lock(tsk)		spin_lock_irq(&runqueue_lock)
#define task_unlock(tsk)	spin_unlock_irq(&runqueue_lock)

#define virt_to_page(kaddr)	(mem_map + MAP_NR(kaddr))

#define fops_get(fops)		(fops)

#define vma_pgoff(vma)	((vma)->vm_offset) /* NOT, but suffices for != 0 */

#define get_zeroed_page(mask)	get_free_page((mask))
#define SetPageReserved(page)	set_bit(PG_reserved, &(page)->flags)
#define ClearPageReserved(page)	clear_bit(PG_reserved, &(page)->flags)

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

#ifdef MODULE
#define __cacheline_aligned __attribute__((__aligned__(SMP_CACHE_BYTES)))
#define __exit		/* empty */
#else
#define __exit		__attribute__((unused, __section__("text.init")))
#endif

#define task_thread(tsk)	(&(tsk)->tss)

#define get_cpus_allowed(tsk)		(-1UL)
#define set_cpus_allowed(tsk, mask)	do{}while(0)

#endif	/* 2.4 simulation for 2.2 */
