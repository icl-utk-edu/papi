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
#include <linux/ptrace.h>
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
	spinlock_t owner_lock;
	struct task_struct *owner;
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
static int nrctrs;
static const char this_service[] = __FILE__;
#if PERFCTR_INTERRUPT_SUPPORT
static void vperfctr_ihandler(unsigned long pc);
#endif

static int inc_nrctrs(void)
{
	const char *other;

	other = NULL;
	spin_lock(&nrctrs_lock);
	if( ++nrctrs == 1 ) {
		other = perfctr_cpu_reserve(this_service);
		if( other )
			nrctrs = 0;
	}
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
		return ERR_PTR(-EBUSY);
	page = get_zeroed_page(GFP_KERNEL);
	if( !page ) {
		dec_nrctrs();
		return ERR_PTR(-ENOMEM);
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
	if( !IS_ERR(perfctr) ) {
		perfctr->state.magic = VPERFCTR_MAGIC;
		atomic_set(&perfctr->count, 1);
		spin_lock_init(&perfctr->owner_lock);
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
 * These must all be called by the owner process only.		*
 * These must all be called with preemption disabled.		*
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
/* PREEMPT note: called in IRQ context with preemption disabled. */
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
 * These must all, with the exception of __vperfctr_exit(),	*
 * be called by the owner process only.				*
 *								*
 ****************************************************************/

/* Called from exit_thread() or sys_vperfctr_unlink().
 * The vperfctr has just been detached from its owner.
 * If the counters are running, stop them and sample their final values.
 * Mark this perfctr as dead and decrement its use count.
 * PREEMPT note: exit_thread() does not run with preemption disabled.
 */
void __vperfctr_exit(struct vperfctr *perfctr)
{
	struct task_struct *owner;

	spin_lock(&perfctr->owner_lock);
	owner = perfctr->owner;
	/* task_thread(owner)->perfctr = NULL was done by the caller */
	perfctr->owner = NULL;
	spin_unlock(&perfctr->owner_lock);

	if( IS_RUNNING(perfctr) && owner == current ) {
		preempt_disable();
		vperfctr_suspend(perfctr);
		preempt_enable();
	}
	perfctr->state.cpu_state.cstatus = 0;
#if PERFCTR_INTERRUPT_SUPPORT
	perfctr->iresume_cstatus = 0;
#endif
	put_vperfctr(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_suspend().
 * If the counters are running, suspend them.
 * PREEMPT note: switch_to() runs with preemption disabled.
 */
void __vperfctr_suspend(struct vperfctr *perfctr)
{
	if( IS_RUNNING(perfctr) )
		vperfctr_suspend(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_resume().
 * PRE: perfctr == TASK_VPERFCTR(current)
 * If the counters are runnable, resume them.
 * PREEMPT note: switch_to() runs with preemption disabled.
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
 * PREEMPT note: called in IRQ context with preemption disabled.
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
 * These can be called by the owner process, or by a monitor	*
 * process which has the owner under ptrace ATTACH control.	*
 *								*
 ****************************************************************/

/* obsolete. subsumed by control(). must be called with preemption disabled */
static int sys_vperfctr_stop(struct vperfctr *perfctr, struct task_struct *tsk)
{
	if( IS_RUNNING(perfctr) ) {
		if( tsk == current )
			vperfctr_suspend(perfctr);
		perfctr->state.cpu_state.cstatus = 0;
#if PERFCTR_INTERRUPT_SUPPORT
		perfctr->iresume_cstatus = 0;
#endif
	}
	return 0;
}

static int sys_vperfctr_control(struct vperfctr *perfctr,
				struct vperfctr_control *argp,
				struct task_struct *tsk)
{
	struct vperfctr_control control;
	int err;
	unsigned int next_cstatus;
	unsigned int nrctrs, i;

	if( copy_from_user(&control, argp, sizeof control) )
		return -EFAULT;

	if( control.cpu_control.nractrs || control.cpu_control.nrictrs ) {
		unsigned long old_mask = get_cpus_allowed(tsk);
		unsigned long new_mask = old_mask & ~perfctr_cpus_forbidden_mask;
		if( !new_mask )
			return -EINVAL;
		if( new_mask != old_mask )
			set_cpus_allowed(tsk, new_mask);
	}

	/* PREEMPT note: preemption is disabled over the entire
	   region since we're updating an active perfctr. */
	preempt_disable();
	sys_vperfctr_stop(perfctr, tsk);
	perfctr->state.cpu_state.control = control.cpu_control;
	/* remote access note: perfctr_cpu_update_control() is ok */
	err = perfctr_cpu_update_control(&perfctr->state.cpu_state);
	if( err < 0 )
		goto out;
	next_cstatus = perfctr->state.cpu_state.cstatus;
	if( !perfctr_cstatus_enabled(next_cstatus) )
		goto out;

	/* XXX: validate si_signo? */
	perfctr->state.si_signo = control.si_signo;

	if( !perfctr_cstatus_has_tsc(next_cstatus) )
		perfctr->state.cpu_state.sum.tsc = 0;

	nrctrs = perfctr_cstatus_nrctrs(next_cstatus);
	for(i = 0; i < nrctrs; ++i)
		if( !(control.preserve & (1<<i)) )
			perfctr->state.cpu_state.sum.pmc[i] = 0;

	if( tsk == current )
		vperfctr_resume(perfctr);

 out:
	preempt_enable();
	return err;
}

static int sys_vperfctr_iresume(struct vperfctr *perfctr, struct task_struct *tsk)
{
#if PERFCTR_INTERRUPT_SUPPORT
	unsigned int iresume_cstatus;

	iresume_cstatus = perfctr->iresume_cstatus;
	if( !perfctr_cstatus_has_ictrs(iresume_cstatus) )
		return -EPERM;

	/* PREEMPT note: preemption is disabled over the entire
	   region because we're updating an active perfctr. */
	preempt_disable();

	if( IS_RUNNING(perfctr) && tsk == current )
		vperfctr_suspend(perfctr);

	perfctr->state.cpu_state.cstatus = iresume_cstatus;
	perfctr->iresume_cstatus = 0;

	/* remote access note: perfctr_cpu_ireload() is ok */
	perfctr_cpu_ireload(&perfctr->state.cpu_state);

	if( tsk == current )
		vperfctr_resume(perfctr);

	preempt_enable();

	return 0;
#else
	return -ENOSYS;
#endif
}

static int sys_vperfctr_unlink(struct vperfctr *perfctr, struct task_struct *tsk)
{
	task_thread(tsk)->perfctr = NULL;
	__vperfctr_exit(perfctr);
	return 0;
}

/*
 * Sample the counters and update state.
 * This operation is used on processors like the pre-MMX Intel P5,
 * which cannot sample the counter registers in user-mode.
 */
static int sys_vperfctr_sample(struct vperfctr *perfctr, struct task_struct *tsk)
{
	if( tsk == current ) {
		preempt_disable();
		vperfctr_sample(perfctr);
		preempt_enable();
	}
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
	struct task_struct *tsk;
	int ret;

	switch( cmd ) {
	case PERFCTR_INFO:
		return sys_perfctr_info((struct perfctr_info*)arg);
	}
	perfctr = filp->private_data;
	if( !perfctr )
		return -EINVAL;
	tsk = current;
	if( perfctr != task_thread(current)->perfctr ) {
		spin_lock(&perfctr->owner_lock);
		tsk = perfctr->owner;
		if( tsk )
			get_task_struct(tsk);
		spin_unlock(&perfctr->owner_lock);
		if( !tsk )
			return -ESRCH;
		ret = ptrace_check_attach(tsk, 0);
		if( ret < 0 )
			goto out;
	}
	switch( cmd ) {
	case VPERFCTR_CONTROL:
		ret = sys_vperfctr_control(perfctr, (struct vperfctr_control*)arg, tsk);
		break;
	case VPERFCTR_STOP:
		preempt_disable();
		ret = sys_vperfctr_stop(perfctr, tsk);
		preempt_enable();
		break;
	case VPERFCTR_UNLINK:
		ret = sys_vperfctr_unlink(perfctr, tsk);
		break;
	case VPERFCTR_SAMPLE:
		ret = sys_vperfctr_sample(perfctr, tsk);
		break;
	case VPERFCTR_IRESUME:
		ret = sys_vperfctr_iresume(perfctr, tsk);
		break;
	default:
		ret = -EINVAL;
	}
 out:
	if( tsk != current )
		put_task_struct(tsk);
	return ret;
}

/* Map a /proc/$pid/$file inode to its task struct. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0)
static struct task_struct *get_task_by_proc_pid_inode(struct inode *inode)
{
	int pid = inode->i_ino >> 16;
	struct task_struct *tsk;

	read_lock(&tasklist_lock);
	tsk = find_task_by_pid(pid);
	if( tsk )
		get_task_struct(tsk);	/* dummy in 2.2 */
	read_unlock(&tasklist_lock);
	return tsk;
}
#else /* LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0) */
static struct task_struct *get_task_by_proc_pid_inode(struct inode *inode)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,4)
	struct task_struct *tsk = PROC_I(inode)->task;
#else
	struct task_struct *tsk = inode->u.proc_i.task;
#endif
	get_task_struct(tsk);
	return tsk;
}
#endif /* LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0) */

static int vperfctr_init_done;

static int vperfctr_open(struct inode *inode, struct file *filp)
{
	struct task_struct *tsk;
	struct vperfctr *perfctr;
	int err;

	/* The link from /proc/<pid>/perfctr exists even if the
	   hardware detection failed. Disallow open in this case. */
	if( !vperfctr_init_done )
		return -ENODEV;

	/*
	 * Allocating a new vperfctr object for the O_CREAT case is
	 * done before the self-or-remote-control check.
	 * This is because get_empty_vperfctr() may sleep, and in the
	 * remote control case, the child may have been killed while we
	 * slept. Instead of dealing with the ugly revalidation issues,
	 * we allocate ahead of time, and remember to deallocate in case
	 * of errors.
	 * If we only supported 2.4+ kernels, this would be much less of
	 * an issue, since the task pointer itself remains valid across
	 * a sleep thanks to get_task_struct().
	 */
	perfctr = NULL;
	if( filp->f_flags & O_CREAT ) {
		perfctr = get_empty_vperfctr(); /* may sleep */
		if( IS_ERR(perfctr) )
			return PTR_ERR(perfctr);
	}
	tsk = current;
	if( !proc_pid_inode_denotes_task(inode, tsk) ) { /* remote? */
		tsk = get_task_by_proc_pid_inode(inode);
		err = -ESRCH;
		if( !tsk )
			goto err_perfctr;
		err = ptrace_check_attach(tsk, 0);
		if( err < 0 )
			goto err_tsk;
	}
	if( filp->f_flags & O_CREAT ) {
		err = -EEXIST;
		if( task_thread(tsk)->perfctr )
			goto err_tsk;
		perfctr->owner = tsk;
		task_thread(tsk)->perfctr = perfctr;
	} else {
		perfctr = task_thread(tsk)->perfctr;
		/* In the /proc/pid/perfctr API, there is one user, viz.
		   ioctl PERFCTR_INFO, for which it's ok for perfctr to
		   be NULL. Hence no non-NULL check here. */
	}
	filp->private_data = perfctr;
	if( perfctr )
		atomic_inc(&perfctr->count);
	if( tsk != current )
		put_task_struct(tsk);
	return 0;
 err_tsk:
	if( tsk != current )
		put_task_struct(tsk);
 err_perfctr:
	if( perfctr )	/* can only occur if filp->f_flags & O_CREAT */
		put_vperfctr(perfctr);
	return err;
}

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
