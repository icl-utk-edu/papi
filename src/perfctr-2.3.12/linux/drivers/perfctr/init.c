/* $Id$
 * Performance-monitoring counters driver.
 * Top-level initialisation code.
 *
 * Copyright (C) 1999-2001  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/perfctr.h>

#include <asm/uaccess.h>

#include "compat.h"
#include "virtual.h"
#include "global.h"
#include "version.h"

MODULE_AUTHOR("Mikael Pettersson <mikpe@csd.uu.se>");
MODULE_DESCRIPTION("Performance-monitoring counters driver");
MODULE_LICENSE("GPL");
EXPORT_NO_SYMBOLS;

struct perfctr_info perfctr_info = {
	.version = VERSION
#ifdef CONFIG_PERFCTR_DEBUG
	" DEBUG"
#endif
};

int sys_perfctr_info(struct perfctr_info *argp)
{
	if( copy_to_user(argp, &perfctr_info, sizeof perfctr_info) )
		return -EFAULT;
	return 0;
}

int __init perfctr_init(void)
{
	int err;
	if( (err = perfctr_cpu_init()) != 0 ) {
		printk(KERN_INFO "perfctr: not supported by this processor\n");
		return err;
	}
	if( (err = vperfctr_init()) != 0 )
		return err;
	if( (err = gperfctr_init()) != 0 )
		return err;
	printk(KERN_INFO "perfctr: driver %s, cpu type %s at %lu kHz\n",
	       perfctr_info.version,
	       perfctr_cpu_name[perfctr_info.cpu_type],
	       perfctr_info.cpu_khz);
	return 0;
}

void __exit perfctr_exit(void)
{
	gperfctr_exit();
	vperfctr_exit();
	perfctr_cpu_exit();
}

module_init(perfctr_init)
module_exit(perfctr_exit)
