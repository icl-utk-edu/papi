/* $Id$
 * Performance-monitoring counters driver.
 * Compatibility definitions for 2.2/2.4/2.5 kernels.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */
#include <linux/fs.h>
#include <linux/version.h>

/* remap_page_range() changed in 2.5.3-pre1 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,5,3)
#include <linux/mm.h>
static inline int perfctr_remap_page_range(struct vm_area_struct *vma, unsigned long from, unsigned long to, unsigned long size, pgprot_t prot)
{
	return remap_page_range(from, to, size, prot);
}
#undef remap_page_range
#define remap_page_range(vma,from,to,size,prot) perfctr_remap_page_range((vma),(from),(to),(size),(prot))
#endif

/* added in 2.4.9-ac */
#if !defined(MODULE_LICENSE)
#define MODULE_LICENSE(license)	/*empty*/
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

#define vma_pgoff(vma)		((vma)->vm_pgoff)
#define task_thread(tsk)	(&(tsk)->thread)
#define proc_pid_inode_denotes_task(inode,tsk)	\
	((tsk) == (inode)->u.proc_i.task)

#else	/* 2.4 simulation for 2.2 */

#if !defined(CONFIG_KMOD) && defined(request_module)	/* < 2.2.20pre10 */
#undef request_module
static inline int request_module(const char *name) { return -ENOSYS; }
#endif

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

#define proc_pid_inode_denotes_task(inode,tsk)	\
	((tsk)->pid == ((inode)->i_ino >> 16))

#define virt_to_page(kaddr)	(mem_map + MAP_NR(kaddr))

#define fops_get(fops)		(fops)

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

#define task_thread(tsk)	(&(tsk)->tss)

#endif	/* 2.4 simulation for 2.2 */
