/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.6 kernels.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */
#include <linux/version.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,0)
#include "compat24.h"
#else

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

#endif
