/* $Id$
 * Kernel stub used to support virtual perfctrs when the
 * perfctr driver is built as a module.
 *
 * Copyright (C) 2000  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/perfctr.h>
#ifdef CONFIG_VPERFCTR_PROC
#include <linux/kmod.h>
#endif
#include "compat.h"
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0)) && defined(CONFIG_VPERFCTR_PROC)
#include <linux/proc_fs.h>
#endif

static void bug(const char *func)
{
	static int count = 0;
	if( ++count > 5 )	/* don't spam the log too much */
		return;
	printk(KERN_ERR __FILE__ ": BUG! call to __vperfctr_%s "
	       "and perfctr module is not loaded\n",
	       func);
}

static struct vperfctr *bug_copy(struct vperfctr *perfctr)
{
	bug("copy");
	return NULL;
}

static void bug_exit(struct vperfctr *perfctr)
{
	bug("exit");
}

static void bug_release(struct vperfctr *parent, struct vperfctr *child)
{
	bug("release");
}

static void bug_suspend(struct vperfctr *perfctr)
{
	bug("suspend");
}

static void bug_resume(struct vperfctr *perfctr)
{
	bug("resume");
}

#ifdef CONFIG_VPERFCTR_PROC
static int vperfctr_stub_open(struct inode *inode, struct file *filp)
{
	struct file_operations *fops;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
	if( current != inode->u.proc_i.task )
		return -EPERM;
#else
	if( current->pid != (inode->i_ino >> 16) )
		return -EPERM;
#endif
	read_lock(&vperfctr_stub_lock);
	fops = fops_get(vperfctr_stub.file_ops);
	read_unlock(&vperfctr_stub_lock);
#ifdef CONFIG_KMOD
	if( !fops ) {
		request_module("perfctr");
		read_lock(&vperfctr_stub_lock);
		fops = fops_get(vperfctr_stub.file_ops);
		read_unlock(&vperfctr_stub_lock);
	}
#endif
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
EXPORT_SYMBOL(perfctr_set_proc_pid_ops);
#else
struct inode_operations perfctr_proc_pid_inode_operations = {
	.default_file_ops = &vperfctr_stub_file_ops,
	.permission = proc_permission,
};
EXPORT_SYMBOL(perfctr_proc_pid_inode_operations);
#endif

#endif	/* CONFIG_VPERFCTR_PROC */

struct vperfctr_stub vperfctr_stub = {
	.copy = bug_copy,
	.exit = bug_exit,
	.release = bug_release,
	.suspend = bug_suspend,
	.resume = bug_resume,
#ifdef CONFIG_VPERFCTR_PROC
	.file_ops = NULL,
#endif
};
rwlock_t vperfctr_stub_lock = RW_LOCK_UNLOCKED;

EXPORT_SYMBOL(vperfctr_stub);
EXPORT_SYMBOL(vperfctr_stub_lock);
