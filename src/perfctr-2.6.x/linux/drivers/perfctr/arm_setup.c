/* $Id$
 * Performance-monitoring counters driver.
 * ARM-specific kernel-resident code.
 *
 * Copyright (C) 2005-2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/module.h>
#include <asm/processor.h>

//#ifdef CONFIG_PERFCTR_MODULE
//EXPORT_SYMBOL(__free_task_struct);
//#endif /* MODULE */
