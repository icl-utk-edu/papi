/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.6 kernels.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */
#include <linux/version.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,0)
#include "compat24.h"
#else

#include "cpumask.h"

#define EXPORT_SYMBOL_mmu_cr4_features	EXPORT_SYMBOL(mmu_cr4_features)
#define EXPORT_SYMBOL___put_task_struct	EXPORT_SYMBOL(__put_task_struct)

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,4)	/* names changed in 2.6.4-rc2 */
#define sysdev_register(dev)	sys_device_register((dev))
#define sysdev_unregister(dev)	sys_device_unregister((dev))
#endif

#endif
