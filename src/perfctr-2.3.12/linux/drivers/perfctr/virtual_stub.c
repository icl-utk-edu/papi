/* $Id$
 * Kernel stub used to support virtual perfctrs when the
 * perfctr driver is built as a module.
 *
 * Copyright (C) 2000-2002  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/perfctr.h>
#include <linux/kmod.h>
#include "compat.h"

static void bug(const char *func, void *callee)
{
	printk(KERN_ERR __FILE__ ": BUG! call to __vperfctr_%s "
	       "from %p, pid %u, '%s' when perfctr module is not loaded\n",
	       func, callee, current->pid, current->comm);
	task_thread(current)->perfctr = NULL;
}

static void bug_exit(struct vperfctr *perfctr)
{
	bug("exit", __builtin_return_address(0));
}

static void bug_suspend(struct vperfctr *perfctr)
{
	bug("suspend", __builtin_return_address(0));
}

static void bug_resume(struct vperfctr *perfctr)
{
	bug("resume", __builtin_return_address(0));
}

#ifdef CONFIG_SMP
static void bug_sample(struct vperfctr *perfctr)
{
	bug("sample", __builtin_return_address(0));
}
#endif

static int vperfctr_stub_open(struct inode *inode, struct file *filp)
{
	struct file_operations *fops;

	if( !proc_pid_inode_denotes_task(inode, current) )
		return -EPERM;
	read_lock(&vperfctr_stub_lock);
	fops = fops_get(vperfctr_stub.file_ops);
	read_unlock(&vperfctr_stub_lock);
	if( !fops && request_module("perfctr") == 0 ) {
		read_lock(&vperfctr_stub_lock);
		fops = fops_get(vperfctr_stub.file_ops);
		read_unlock(&vperfctr_stub_lock);
	}
	if( !fops )
		return -ENOSYS;
	filp->f_op = fops;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
	inode->i_fop = fops; /* no fops_get since only filp->f_op counts */
#endif
	return fops->open(inode, filp);
}

static struct file_operations vperfctr_stub_file_ops = {
	.open = vperfctr_stub_open,
};

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
void perfctr_set_proc_pid_ops(struct inode *inode)
{
	inode->i_fop = &vperfctr_stub_file_ops;
}
#else
#include <linux/proc_fs.h>
struct inode_operations perfctr_proc_pid_inode_operations = {
	.default_file_ops = &vperfctr_stub_file_ops,
	.permission = proc_permission,
};
#endif

struct vperfctr_stub vperfctr_stub = {
	.exit = bug_exit,
	.suspend = bug_suspend,
	.resume = bug_resume,
#ifdef CONFIG_SMP
	.sample = bug_sample,
#endif
	.file_ops = NULL,
};
rwlock_t vperfctr_stub_lock = RW_LOCK_UNLOCKED;

EXPORT_SYMBOL(vperfctr_stub);
EXPORT_SYMBOL(vperfctr_stub_lock);
EXPORT_SYMBOL_pidhash;
EXPORT_SYMBOL___put_task_struct;
#ifdef CONFIG_SMP
EXPORT_SYMBOL_tasklist_lock;
#endif
