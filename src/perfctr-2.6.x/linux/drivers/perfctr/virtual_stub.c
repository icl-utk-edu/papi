/* $Id: virtual_stub.c,v 1.26.2.9 2009/01/23 17:21:20 mikpe Exp $
 * Kernel stub used to support virtual perfctrs when the
 * perfctr driver is built as a module.
 *
 * Copyright (C) 2000-2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/perfctr.h>
#include "compat.h"

static void bug_void_perfctr(struct vperfctr *perfctr)
{
	current->thread.perfctr = NULL;
	BUG();
}

#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
static void bug_set_cpus_allowed(struct task_struct *owner, struct vperfctr *perfctr, cpumask_t new_mask)
{
	owner->thread.perfctr = NULL;
	BUG();
}
#endif

struct vperfctr_stub vperfctr_stub = {
	.exit = bug_void_perfctr,
	.flush = bug_void_perfctr,
	.suspend = bug_void_perfctr,
	.resume = bug_void_perfctr,
	.sample = bug_void_perfctr,
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
	.set_cpus_allowed = bug_set_cpus_allowed,
#endif
};

/*
 * exit_thread() calls __vperfctr_exit() via vperfctr_stub.exit().
 * If the process' reference was the last reference to this
 * vperfctr object, and this was the last live vperfctr object,
 * then the perfctr module's use count will drop to zero.
 * This is Ok, except for the fact that code is still running
 * in the module (pending returns back to exit_thread()). This
 * could race with rmmod in a preemptive UP kernel, leading to
 * code running in freed memory. The race also exists in SMP
 * kernels, but the time window is extremely small.
 *
 * Since exit() isn't performance-critical, we wrap the call to
 * vperfctr_stub.exit() with code to increment the module's use
 * count before the call, and decrement it again afterwards. Thus,
 * the final drop to zero occurs here and not in the module itself.
 * (All other code paths that drop the use count do so via a file
 * object, and VFS also refcounts the module.)
 */
void _vperfctr_exit(struct vperfctr *perfctr)
{
	__module_get(vperfctr_stub.owner);
	vperfctr_stub.exit(perfctr);
	module_put(vperfctr_stub.owner);
}

/* __vperfctr_flush() is a conditional __vperfctr_exit(),
 * so it needs the same protection.
 */
void _vperfctr_flush(struct vperfctr *perfctr)
{
	__module_get(vperfctr_stub.owner);
	vperfctr_stub.flush(perfctr);
	module_put(vperfctr_stub.owner);
}

EXPORT_SYMBOL(vperfctr_stub);
EXPORT_SYMBOL___put_task_struct;

#if !defined(CONFIG_UTRACE)
#include <linux/mm.h>
#include <linux/ptrace.h>
EXPORT_SYMBOL(ptrace_check_attach);
#endif
