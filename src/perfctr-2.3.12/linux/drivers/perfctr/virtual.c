/* $Id$
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/perfctr.h>

#include <asm/io.h>
#include <asm/uaccess.h>

#include "compat.h"
#include "virtual.h"

/****************************************************************
 *								*
 * Data types and macros.					*
 *								*
 ****************************************************************/

struct vperfctr {
/* User-visible fields: (must be first for mmap()) */
	struct vperfctr_state state;
/* Kernel-private fields: */
	atomic_t count;
#ifdef CONFIG_SMP
	unsigned int sampling_timer;
#endif
#ifdef CONFIG_PERFCTR_DEBUG
	unsigned start_smp_id;
	unsigned suspended;
#endif
#if PERFCTR_INTERRUPT_SUPPORT
	unsigned int iresume_cstatus;
#endif
};
#define IS_RUNNING(perfctr)	perfctr_cstatus_enabled((perfctr)->state.cpu_state.cstatus)
#define IS_IMODE(perfctr)	perfctr_cstatus_has_ictrs((perfctr)->state.cpu_state.cstatus)

#ifdef CONFIG_PERFCTR_DEBUG
#define debug_free(perfctr) \
do { \
	int i; \
	for(i = 0; i < PAGE_SIZE/sizeof(int); ++i) \
		((int*)(perfctr))[i] = 0xfedac0ed; \
} while( 0 )
#define debug_init(perfctr)	do { (perfctr)->suspended = 1; } while( 0 )
#define debug_suspend(perfctr) \
do { \
	if( (perfctr)->suspended ) \
		printk(KERN_ERR "%s: BUG! suspending non-running perfctr (pid %d, comm %s)\n", \
		       __FUNCTION__, current->pid, current->comm); \
	(perfctr)->suspended = 1; \
} while( 0 )
#define debug_resume(perfctr) \
do { \
	if( !(perfctr)->suspended ) \
		printk(KERN_ERR "%s: BUG! resuming non-suspended perfctr (pid %d, comm %s)\n", \
		       __FUNCTION__, current->pid, current->comm); \
	(perfctr)->suspended = 0; \
} while( 0 )
#define debug_check_smp_id(perfctr) \
do { \
	if( (perfctr)->start_smp_id != smp_processor_id() ) { \
		printk(KERN_ERR "%s: BUG! current cpu %u differs from start cpu %u (pid %d, comm %s)\n", \
		       __FUNCTION__, smp_processor_id(), (perfctr)->start_smp_id, \
		       current->pid, current->comm); \
		return; \
	} \
} while( 0 )
#define debug_set_smp_id(perfctr) \
	do { (perfctr)->start_smp_id = smp_processor_id(); } while( 0 )
#else	/* CONFIG_PERFCTR_DEBUG */
#define debug_free(perfctr)		do{}while(0)
#define debug_init(perfctr)		do{}while(0)
#define debug_suspend(perfctr)		do{}while(0)
#define debug_resume(perfctr)		do{}while(0)
#define debug_check_smp_id(perfctr)	do{}while(0)
#define debug_set_smp_id(perfctr)	do{}while(0)
#endif	/* CONFIG_PERFCTR_DEBUG */

/****************************************************************
 *								*
 * Resource management.						*
 *								*
 ****************************************************************/

/* XXX: perhaps relax this to number of _live_ perfctrs */
static spinlock_t nrctrs_lock = SPIN_LOCK_UNLOCKED;
int nrctrs = 0;
static const char this_service[] = __FILE__;
#if PERFCTR_INTERRUPT_SUPPORT
static void vperfctr_ihandler(unsigned long pc);
#endif

static int inc_nrctrs(void)
{
	const char *other;

	other = NULL;
	spin_lock(&nrctrs_lock);
	if( ++nrctrs == 1 )
		other = perfctr_cpu_reserve(this_service);
	spin_unlock(&nrctrs_lock);
	if( other ) {
		printk(KERN_ERR __FILE__
		       ": cannot operate, perfctr hardware taken by '%s'\n",
		       other);
		return -EBUSY;
	}
#if PERFCTR_INTERRUPT_SUPPORT
	perfctr_cpu_set_ihandler(vperfctr_ihandler);
#endif
	return 0;
}

static void dec_nrctrs(void)
{
	spin_lock(&nrctrs_lock);
	if( --nrctrs == 0 )
		perfctr_cpu_release(this_service);
	spin_unlock(&nrctrs_lock);
}

static struct vperfctr *vperfctr_alloc(void)
{
	unsigned long page;

	if( inc_nrctrs() != 0 )
		return NULL;
	page = get_zeroed_page(GFP_KERNEL);
	if( !page ) {
		dec_nrctrs();
		return NULL;
	}
	SetPageReserved(virt_to_page(page));
	return (struct vperfctr*) page;
}

static void vperfctr_free(struct vperfctr *perfctr)
{
	debug_free(perfctr);
	ClearPageReserved(virt_to_page(perfctr));
	free_page((unsigned long)perfctr);
	dec_nrctrs();
}

static struct vperfctr *get_empty_vperfctr(void)
{
	struct vperfctr *perfctr = vperfctr_alloc();
	if( perfctr ) {
		perfctr->state.magic = VPERFCTR_MAGIC;
		atomic_set(&perfctr->count, 1);
		debug_init(perfctr);
	}
	return perfctr;
}

static void put_vperfctr(struct vperfctr *perfctr)
{
	if( atomic_dec_and_test(&perfctr->count) )
		vperfctr_free(perfctr);
}

/****************************************************************
 *								*
 * Basic counter operations.					*
 *								*
 ****************************************************************/

/* PRE: perfctr == TASK_VPERFCTR(current) && IS_RUNNING(perfctr)
 * Suspend the counters.
 */
static inline void vperfctr_suspend(struct vperfctr *perfctr)
{
	debug_suspend(perfctr);
	debug_check_smp_id(perfctr);
	perfctr_cpu_suspend(&perfctr->state.cpu_state);
}

static inline void vperfctr_reset_sampling_timer(struct vperfctr *perfctr)
{
#ifdef CONFIG_SMP
	/* XXX: base the value on perfctr_info.cpu_khz instead! */
	perfctr->sampling_timer = HZ/2;
#endif
}

/* PRE: perfctr == TASK_VPERFCTR(current) && IS_RUNNING(perfctr)
 * Restart the counters.
 */
static inline void vperfctr_resume(struct vperfctr *perfctr)
{
	debug_resume(perfctr);
	perfctr_cpu_resume(&perfctr->state.cpu_state);
	vperfctr_reset_sampling_timer(perfctr);
	debug_set_smp_id(perfctr);
}

/* Sample the counters but do not suspend them. */
static void vperfctr_sample(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) ) {
		debug_check_smp_id(perfctr);
		perfctr_cpu_sample(&perfctr->state.cpu_state);
		vperfctr_reset_sampling_timer(perfctr);
	}
}

#if PERFCTR_INTERRUPT_SUPPORT
/* vperfctr interrupt handler (XXX: add buffering support) */
static void vperfctr_ihandler(unsigned long pc)
{
	struct task_struct *tsk = current;
	struct vperfctr *perfctr;
	unsigned int pmc_mask;
	siginfo_t si;

	perfctr = task_thread(tsk)->perfctr;
	if( !perfctr ) {
		printk(KERN_ERR "%s: BUG! pid %d has no vperfctr\n",
		       __FUNCTION__, tsk->pid);
		return;
	}
	if( !IS_IMODE(perfctr) ) {
		printk(KERN_ERR "%s: BUG! vperfctr has cstatus %#x (pid %d, comm %s)\n",
		       __FUNCTION__, perfctr->state.cpu_state.cstatus, tsk->pid, tsk->comm);
		return;
	}
	vperfctr_suspend(perfctr);
	pmc_mask = perfctr_cpu_identify_overflow(&perfctr->state.cpu_state);
	if( !pmc_mask ) {
		printk(KERN_ERR "%s: BUG! pid %d has unidentifiable overflow source\n",
		       __FUNCTION__, tsk->pid);
		return;
	}
	/* suspend a-mode and i-mode PMCs, leaving only TSC on */
	/* XXX: some people also want to suspend the TSC */
	perfctr->iresume_cstatus = perfctr->state.cpu_state.cstatus;
	if( perfctr_cstatus_has_tsc(perfctr->iresume_cstatus) ) {
		perfctr->state.cpu_state.cstatus = perfctr_mk_cstatus(1, 0, 0);
		vperfctr_resume(perfctr);
	} else
		perfctr->state.cpu_state.cstatus = 0;
	si.si_signo = perfctr->state.si_signo;
	si.si_errno = 0;
	si.si_code = SI_PMC_OVF;
	si.si_pmc_ovf_mask = pmc_mask;
	if( !send_sig_info(si.si_signo, &si, tsk) )
		send_sig(si.si_signo, tsk, 1);
}
#endif

/****************************************************************
 *								*
 * Process management operations.				*
 *								*
 ****************************************************************/

/* Called from exit_thread() or sys_vperfctr_unlink().
 * Current has just detached its vperfctr.
 * If the counters are running, stop them and sample their final values.
 * Mark this perfctr as dead and decrement its use count.
 */
void __vperfctr_exit(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);
	perfctr->state.cpu_state.cstatus = 0;
#if PERFCTR_INTERRUPT_SUPPORT
	perfctr->iresume_cstatus = 0;
#endif
	put_vperfctr(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_suspend().
 * If the counters are running, suspend them.
 */
void __vperfctr_suspend(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_resume().
 * PRE: perfctr == TASK_VPERFCTR(current)
 * If the counters are runnable, resume them.
 */
void __vperfctr_resume(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_resume(perfctr);
}

#ifdef CONFIG_SMP
/* Called from update_one_process() [triggered by timer interrupt].
 * PRE: perfctr == TASK_VPERFCTR(current).
 * Sample the counters but do not suspend them.
 * Needed on SMP to avoid precision loss due to multiple counter
 * wraparounds between resume/suspend for CPU-bound processes.
 */
void __vperfctr_sample(struct vperfctr *perfctr)
{
	if( --perfctr->sampling_timer == 0 )
		vperfctr_sample(perfctr);
}
#endif

/****************************************************************
 *								*
 * Virtual perfctr "system calls".				*
 *								*
 ****************************************************************/

/* PRE: perfctr == TASK_VPERFCTR(current) */
static int sys_vperfctr_stop(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) ) {
		vperfctr_suspend(perfctr);
		perfctr->state.cpu_state.cstatus = 0;
#if PERFCTR_INTERRUPT_SUPPORT
		perfctr->iresume_cstatus = 0;
#endif
	}
	return 0;
}

static int
sys_vperfctr_control(struct vperfctr *perfctr, struct vperfctr_control *argp)
{
	struct vperfctr_control control;
	int err;
	unsigned int next_cstatus;

	if( copy_from_user(&control, argp, sizeof control) )
		return -EFAULT;
	sys_vperfctr_stop(perfctr);
	perfctr->state.cpu_state.control = control.cpu_control;
	err = perfctr_cpu_update_control(&perfctr->state.cpu_state);
	if( err < 0 )
		return err;
	next_cstatus = perfctr->state.cpu_state.cstatus;
	if( !perfctr_cstatus_enabled(next_cstatus) )
		return 0;

	/* XXX: validate si_signo? */
	perfctr->state.si_signo = control.si_signo;

	/*
	 * Clear the perfctr sums and restart the perfctrs.
	 * Preserve the time-stamp counter's sum if possible.
	 */
	if( !perfctr_cstatus_has_tsc(next_cstatus) )
		perfctr->state.cpu_state.sum.tsc = 0;
	memset(&perfctr->state.cpu_state.sum.pmc, 0,
	       sizeof perfctr->state.cpu_state.sum.pmc);
	vperfctr_resume(perfctr);

	return 0;
}

static int sys_vperfctr_iresume(struct vperfctr *perfctr)
{
#if PERFCTR_INTERRUPT_SUPPORT
	unsigned int iresume_cstatus;

	iresume_cstatus = perfctr->iresume_cstatus;
	if( !perfctr_cstatus_has_ictrs(iresume_cstatus) )
		return -EPERM;

	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);

	perfctr->state.cpu_state.cstatus = iresume_cstatus;
	perfctr->iresume_cstatus = 0;

	perfctr_cpu_ireload(&perfctr->state.cpu_state);
	vperfctr_resume(perfctr);
	return 0;
#else
	return -ENOSYS;
#endif
}

/* PRE: perfctr == TASK_VPERFCTR(current) */
static int sys_vperfctr_unlink(struct vperfctr *perfctr)
{
	task_thread(current)->perfctr = NULL;
	__vperfctr_exit(perfctr);
	return 0;
}

/* PRE: perfctr == TASK_VPERFCTR(current)
 * Sample the current process' counters and update state.
 * This operation is used on processors like the pre-MMX Intel P5,
 * which cannot sample the counter registers in user-mode.
 */
static int sys_vperfctr_sample(struct vperfctr *perfctr)
{
	vperfctr_sample(perfctr);
	return 0;
}

/****************************************************************
 *								*
 * Virtual perfctr file operations.				*
 *								*
 ****************************************************************/

static int vperfctr_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct vperfctr *perfctr;

	/* Only allow read-only mapping of first page. */
	if( (vma->vm_end - vma->vm_start) != PAGE_SIZE ||
	    vma_pgoff(vma) != 0 ||
	    (pgprot_val(vma->vm_page_prot) & _PAGE_RW) ||
	    (vma->vm_flags & (VM_WRITE | VM_MAYWRITE)) )
		return -EPERM;
	perfctr = filp->private_data;
	if( !perfctr )
		return -EPERM;
	return remap_page_range(vma, vma->vm_start, virt_to_phys(perfctr),
				PAGE_SIZE, vma->vm_page_prot);
}

static int vperfctr_release(struct inode *inode, struct file *filp)
{
	struct vperfctr *perfctr = filp->private_data;
	filp->private_data = NULL;
	if( perfctr )
		put_vperfctr(perfctr);
	return 0;
}

static int vperfctr_ioctl(struct inode *inode, struct file *filp,
			  unsigned int cmd, unsigned long arg)
{
	struct vperfctr *perfctr;

	switch( cmd ) {
	case PERFCTR_INFO:
		return sys_perfctr_info((struct perfctr_info*)arg);
	}
	perfctr = filp->private_data;
	if( !perfctr || perfctr != task_thread(current)->perfctr )
		return -EPERM;
	switch( cmd ) {
	case VPERFCTR_CONTROL:
		return sys_vperfctr_control(perfctr, (struct vperfctr_control*)arg);
	case VPERFCTR_STOP:
		return sys_vperfctr_stop(perfctr);
	case VPERFCTR_UNLINK:
		return sys_vperfctr_unlink(perfctr);
	case VPERFCTR_SAMPLE:
		return sys_vperfctr_sample(perfctr);
	case VPERFCTR_IRESUME:
		return sys_vperfctr_iresume(perfctr);
	}
	return -EINVAL;
}

static int vperfctr_init_done;

static int vperfctr_open(struct inode *inode, struct file *filp)
{
	struct task_struct *tsk;
	struct vperfctr *perfctr;

	/* The link from /proc/<pid>/perfctr exists even if the
	   hardware detection failed. Disallow open in this case. */
	if( !vperfctr_init_done )
		return -ENODEV;

	/* XXX:
	 * - permit read-only open of other process' vperfctr, using
	 *   same permission check as in the old ATTACH interface
	 * - or add a spinlock to the thread_struct and allow a
	 *   "remote open" even if the target proc isn't stopped?
	 */
	tsk = current;
	if( !proc_pid_inode_denotes_task(inode, tsk) )
		return -EPERM;
	perfctr = task_thread(tsk)->perfctr;
	if( filp->f_flags & O_CREAT ) {
		if( perfctr )
			return -EEXIST;
		perfctr = get_empty_vperfctr();
		if( !perfctr )
			return -ENOMEM;
	}
	filp->private_data = perfctr;
	if( perfctr )
		atomic_inc(&perfctr->count);
	if( !task_thread(tsk)->perfctr )
		task_thread(tsk)->perfctr = perfctr;
	return 0;
}

#if 1
#include <linux/smp_lock.h>	/* for {,un}lock_kernel() */
int perfctr_stub_for_new_remote_control_code(int pid)
{
    struct task_struct *tsk;
    int ret;

    lock_kernel();
    read_lock(&tasklist_lock);
    tsk = find_task_by_pid(pid);
    if( tsk )
	get_task_struct(tsk);
    read_unlock(&tasklist_lock);
    ret = -EPERM;
    if( !tsk )
	goto out;
    ret = ptrace_check_attach(tsk, 0);
    if( ret < 0 )
	goto out_tsk;
    ret = 0;	/* do some work here */
 out_tsk:
    put_task_struct(tsk);
 out:
    unlock_kernel();
    return ret;
}
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)

static struct file_operations vperfctr_file_ops = {
	.owner = THIS_MODULE,
	.mmap = vperfctr_mmap,
	.release = vperfctr_release,
	.ioctl = vperfctr_ioctl,
	.open = vperfctr_open,
};

#if !defined(MODULE)
void perfctr_set_proc_pid_ops(struct inode *inode)
{
	inode->i_fop = &vperfctr_file_ops;
}
#endif

#else	/* 2.2 :-( */

#include <linux/proc_fs.h>

#if defined(MODULE)
static int vperfctr_release_22(struct inode *inode, struct file *filp)
{
	vperfctr_release(inode, filp);
	MOD_DEC_USE_COUNT;	/* 2.4 kernel does this for us */
	return 0;
}
static int vperfctr_open_22(struct inode *inode, struct file *filp)
{
	int ret;
	MOD_INC_USE_COUNT;	/* 2.4 kernel does this for us */
	ret = vperfctr_open(inode, filp);
	if( ret < 0 )
		MOD_DEC_USE_COUNT;
	return ret;
}
#else	/* !MODULE */
#define vperfctr_release_22	vperfctr_release
#define vperfctr_open_22	vperfctr_open
#endif	/* MODULE */

static struct file_operations vperfctr_file_ops = {
	.mmap = vperfctr_mmap,
	.release = vperfctr_release_22,
	.ioctl = vperfctr_ioctl,
	.open = vperfctr_open_22,
};

#if !defined(MODULE)
struct inode_operations perfctr_proc_pid_inode_operations = {
	.default_file_ops = &vperfctr_file_ops,
	.permission = proc_permission,
};
#endif

#endif	/* LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */

/****************************************************************
 *								*
 * module_init/exit						*
 *								*
 ****************************************************************/

#ifdef MODULE
static struct vperfctr_stub off;

static void vperfctr_stub_init(void)
{
	write_lock(&vperfctr_stub_lock);
	off = vperfctr_stub;
	vperfctr_stub.exit = __vperfctr_exit;
	vperfctr_stub.suspend = __vperfctr_suspend;
	vperfctr_stub.resume = __vperfctr_resume;
#ifdef CONFIG_SMP
	vperfctr_stub.sample = __vperfctr_sample;
#endif
	vperfctr_stub.file_ops = &vperfctr_file_ops;
	write_unlock(&vperfctr_stub_lock);
}

static void vperfctr_stub_exit(void)
{
	write_lock(&vperfctr_stub_lock);
	vperfctr_stub = off;
	write_unlock(&vperfctr_stub_lock);
}
#else
static inline void vperfctr_stub_init(void) { }
static inline void vperfctr_stub_exit(void) { }
#endif	/* MODULE */

int __init vperfctr_init(void)
{
	vperfctr_stub_init();
	vperfctr_init_done = 1;
	return 0;
}

void __exit vperfctr_exit(void)
{
	vperfctr_stub_exit();
}
