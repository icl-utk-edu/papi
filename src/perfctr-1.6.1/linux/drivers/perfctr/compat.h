/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.2/2.4 kernels.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <linux/fs.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

#define OWNER_THIS_MODULE	.owner = THIS_MODULE,
#define MOD_INC_OPEN(func)	/*empty*/
#define MOD_DEC_RELEASE(func)	/*empty*/
#define vma_pgoff(vma)		((vma)->vm_pgoff)
#define TASK_VPERFCTR(tsk)	((tsk)->thread.perfctr)

#else	/* 2.4 simulation for 2.2 */

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,2,18)
#ifdef MODULE
#define module_init(x)	int init_module(void) { return x(); }
#define module_exit(x)	void cleanup_module(void) { x(); }
#else
#define module_init(x)	/* explicit call is needed */
#define module_exit(x)	/* empty */
#endif /* MODULE */
#define DECLARE_MUTEX(name)	struct semaphore name = MUTEX
#endif /* < 2.2.18 */

#define virt_to_page(kaddr)	(mem_map + MAP_NR(kaddr))

#define fops_get(fops)		(fops)
#define OWNER_THIS_MODULE	/*empty*/
#define MOD_INC_OPEN(func)	.open = func,
#define MOD_DEC_RELEASE(func)	.release = func,
#define NEED_MOD_INC_OPEN

#define vma_pgoff(vma)	((vma)->vm_offset) /* NOT, but suffices for != 0 */

#define get_zeroed_page(mask)	get_free_page((mask))
#define SetPageReserved(page)	set_bit(PG_reserved, &(page)->flags)
#define ClearPageReserved(page)	clear_bit(PG_reserved, &(page)->flags)

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

#ifdef MODULE
#define __cacheline_aligned __attribute__((__aligned__(SMP_CACHE_BYTES)))
#define __exit		/* empty */
#else
#define __exit		__attribute__((unused, __section__("text.init")))
#endif

#define X86_CR4_TSD	0x0004
#define X86_CR4_PCE	0x0100

#define TASK_VPERFCTR(tsk)	((tsk)->tss.perfctr)

#define get_file(x)	((x)->f_count++)
#define file_count(x)	((x)->f_count)

#endif	/* 2.4 simulation for 2.2 */
