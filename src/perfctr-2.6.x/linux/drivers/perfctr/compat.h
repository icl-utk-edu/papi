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

#define task_siglock(tsk)	((tsk)->sighand->siglock)

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,4)	/* names changed in 2.6.4-rc2 */
#define sysdev_register(dev)	sys_device_register((dev))
#define sysdev_unregister(dev)	sys_device_unregister((dev))
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,10) /* obsoleted in 2.6.10-rc1 */
#include <linux/mm.h>
static inline int
remap_pfn_range(struct vm_area_struct *vma, unsigned long uvaddr,
		unsigned long pfn, unsigned long size, pgprot_t prot)
{
	return remap_page_range(vma, uvaddr, pfn << PAGE_SHIFT, size, prot);
}
#endif

#endif
