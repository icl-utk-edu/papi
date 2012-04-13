/* $Id: compat.h,v 1.42.2.17 2009/01/23 17:01:02 mikpe Exp $
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.6 kernels.
 *
 * Copyright (C) 1999-2009  Mikael Pettersson
 */
#include <linux/version.h>

#include "cpumask.h"

#define EXPORT_SYMBOL_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)

/* Starting with 2.6.16-rc1, put_task_struct() uses an RCU callback
   __put_task_struct_cb() instead of the old __put_task_struct().
   2.6.16-rc6 dropped the EXPORT_SYMBOL() of __put_task_struct_cb().
   2.6.17-rc1 reverted to using __put_task_struct() again. */
#if defined(HAVE_EXPORT___put_task_struct)
/* 2.6.5-7.201-suse added EXPORT_SYMBOL_GPL(__put_task_struct) */
/* 2.6.16.46-0.12-suse added EXPORT_SYMBOL(__put_task_struct_cb) */
#define EXPORT_SYMBOL___put_task_struct	/*empty*/
#elif LINUX_VERSION_CODE == KERNEL_VERSION(2,6,16)
#define EXPORT_SYMBOL___put_task_struct	EXPORT_SYMBOL(__put_task_struct_cb)
#else
#define EXPORT_SYMBOL___put_task_struct	EXPORT_SYMBOL(__put_task_struct)
#endif

#define task_siglock(tsk)	((tsk)->sighand->siglock)

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,4)	/* names changed in 2.6.4-rc2 */
#define sysdev_register(dev)	sys_device_register((dev))
#define sysdev_unregister(dev)	sys_device_unregister((dev))
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,10) /* remap_page_range() obsoleted in 2.6.10-rc1 */
#include <linux/mm.h>
static inline int
remap_pfn_range(struct vm_area_struct *vma, unsigned long uvaddr,
		unsigned long pfn, unsigned long size, pgprot_t prot)
{
	return remap_page_range(vma, uvaddr, pfn << PAGE_SHIFT, size, prot);
}
#endif

#if !defined(DEFINE_SPINLOCK) /* added in 2.6.11-rc1 */
#define DEFINE_SPINLOCK(x)	spinlock_t x = SPIN_LOCK_UNLOCKED
#endif

/* 2.6.16 introduced a new mutex type, replacing mutex-like semaphores. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,16)
#define DEFINE_MUTEX(mutex)	DECLARE_MUTEX(mutex)
#define mutex_lock(mutexp)	down(mutexp)
#define mutex_unlock(mutexp)	up(mutexp)
#endif

/* 2.6.18-8.1.1.el5 replaced ptrace with utrace */
#if defined(CONFIG_UTRACE)
/* alas, I don't yet know how to convert this to utrace */
static inline int ptrace_check_attach(struct task_struct *task, int kill) { return -ESRCH; }
#endif

/* 2.6.20-rc1 moved filp->f_dentry and filp->f_vfsmnt into filp->fpath */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
#define filp_dentry(filp)	((filp)->f_path.dentry)
#define filp_vfsmnt(filp)	((filp)->f_path.mnt)
#else
#define filp_dentry(filp)	((filp)->f_dentry)
#define filp_vfsmnt(filp)	((filp)->f_vfsmnt)
#endif

/* 2.6.24 introduced find_task_by_vpid() and task_pid_vnr().
   2.6.26 deprecated find_task_by_pid() and 2.6.27-rc1 removed it.
   We'll use 2.6.26 as the switch-over point. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26)
static inline struct task_struct *perfctr_find_task_by_vpid(pid_t nr)
{
	return find_task_by_pid(nr);
}
#undef find_task_by_vpid
#define find_task_by_vpid(nr)	perfctr_find_task_by_vpid((nr))
static inline pid_t perfctr_task_pid_vnr(const struct task_struct *tsk)
{
	return tsk->pid;
}
#undef task_pid_vnr
#define task_pid_vnr(tsk)	perfctr_task_pid_vnr((tsk))
#endif

/* 2.6.27-rc1 dropped the retry parameter from smp_call_function()
   and on_each_cpu() -- we always called it with retry == 1 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27)
static inline int perfctr_smp_call_function(
	void (*func)(void *info), void *info, int wait)
{
	return smp_call_function(func, info, 1, wait);
}
#undef smp_call_function
#define smp_call_function(f,i,w)	perfctr_smp_call_function((f),(i),(w))
static inline int perfctr_on_each_cpu(
	void (*func)(void *info), void *info, int wait)
{
	return on_each_cpu(func, info, 1, wait);
}
#undef on_each_cpu
#define on_each_cpu(f,i,w)	perfctr_on_each_cpu((f),(i),(w))
#endif

/* 2.6.29-rc1 changed how ->fsuid and ->fsgid are accessed */
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,29)) && !defined(current_fsuid)
#define current_fsuid()	(current->fsuid)
#define current_fsgid()	(current->fsgid)
#endif
