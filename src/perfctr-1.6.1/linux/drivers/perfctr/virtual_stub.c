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
#include "compat.h"

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

struct vperfctr_stub vperfctr_stub = {
	.copy = bug_copy,
	.exit = bug_exit,
	.release = bug_release,
	.suspend = bug_suspend,
	.resume = bug_resume,
};

EXPORT_SYMBOL(vperfctr_stub);
