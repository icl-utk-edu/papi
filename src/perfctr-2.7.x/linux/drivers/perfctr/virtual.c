/* $Id: virtual.c,v 1.117 2007/10/06 13:02:07 mikpe Exp $
 * Virtual per-process performance counters.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/init.h>
#include <linux/compiler.h>	/* for unlikely() in 2.4.18 and older */
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/ptrace.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/perfctr.h>

#include <asm/io.h>
#include <asm/uaccess.h>

#include "cpumask.h"
#include "virtual.h"

/****************************************************************
 *								*
 * Data types and macros.					*
 *								*
 ****************************************************************/

struct vperfctr {
/* User-visible fields: (must be first for mmap()) */
	struct perfctr_cpu_state cpu_state;
/* Kernel-private fields: */
	int si_signo;
	atomic_t count;
	spinlock_t owner_lock;
	struct task_struct *owner;
	/* sampling_timer and bad_cpus_allowed are frequently
	   accessed, so they get to share a cache line */
	unsigned int sampling_timer ____cacheline_aligned;
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
	atomic_t bad_cpus_allowed;
#endif
	unsigned int preserve;
	unsigned int resume_cstatus;
#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
	unsigned int ireload_needed; /* only valid if resume_cstatus != 0 */
#endif
	/* children_lock protects inheritance_id and children,
	   when parent is not the one doing release_task() */
	spinlock_t children_lock;
	unsigned long long inheritance_id;
	struct perfctr_sum_ctrs children;
	/* schedule_work() data for when an operation cannot be
	   done in the current context due to locking rules */
	struct work_struct work;
	struct task_struct *parent_tsk;
};
#define IS_RUNNING(perfctr)	perfctr_cstatus_enabled((perfctr)->cpu_state.user.cstatus)

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT

static void vperfctr_ihandler(unsigned long pc);
static void vperfctr_handle_overflow(struct task_struct*, struct vperfctr*);

static inline void vperfctr_set_ihandler(void)
{
	perfctr_cpu_set_ihandler(vperfctr_ihandler);
}

#else
static inline void vperfctr_set_ihandler(void) { }
#endif

#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK

static inline void vperfctr_init_bad_cpus_allowed(struct vperfctr *perfctr)
{
	atomic_set(&perfctr->bad_cpus_allowed, 0);
}

#else	/* !CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK */
static inline void vperfctr_init_bad_cpus_allowed(struct vperfctr *perfctr) { }
#endif	/* !CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK */

/****************************************************************
 *								*
 * Resource management.						*
 *								*
 ****************************************************************/

/* XXX: perhaps relax this to number of _live_ perfctrs */
static DEFINE_MUTEX(nrctrs_mutex);
static int nrctrs;
static const char this_service[] = __FILE__;

static int inc_nrctrs(void)
{
	const char *other;

	other = NULL;
	mutex_lock(&nrctrs_mutex);
	if (++nrctrs == 1) {
		other = perfctr_cpu_reserve(this_service);
		if (other)
			nrctrs = 0;
	}
	mutex_unlock(&nrctrs_mutex);
	if (other) {
		printk(KERN_ERR __FILE__
		       ": cannot operate, perfctr hardware taken by '%s'\n",
		       other);
		return -EBUSY;
	}
	vperfctr_set_ihandler();
	return 0;
}

static void dec_nrctrs(void)
{
	mutex_lock(&nrctrs_mutex);
	if (--nrctrs == 0)
		perfctr_cpu_release(this_service);
	mutex_unlock(&nrctrs_mutex);
}

/* Allocate a `struct vperfctr'. Claim and reserve
   an entire page so that it can be mmap():ed. */
static struct vperfctr *vperfctr_alloc(void)
{
	unsigned long page;

	if (inc_nrctrs() != 0)
		return ERR_PTR(-EBUSY);
	page = get_zeroed_page(GFP_KERNEL);
	if (!page) {
		dec_nrctrs();
		return ERR_PTR(-ENOMEM);
	}
	SetPageReserved(virt_to_page(page));
	return (struct vperfctr*) page;
}

static void vperfctr_free(struct vperfctr *perfctr)
{
	ClearPageReserved(virt_to_page(perfctr));
	free_page((unsigned long)perfctr);
	dec_nrctrs();
}

static struct vperfctr *get_empty_vperfctr(void)
{
	struct vperfctr *perfctr = vperfctr_alloc();
	if (!IS_ERR(perfctr)) {
		atomic_set(&perfctr->count, 1);
		vperfctr_init_bad_cpus_allowed(perfctr);
		spin_lock_init(&perfctr->owner_lock);
		spin_lock_init(&perfctr->children_lock);
	}
	return perfctr;
}

static void put_vperfctr(struct vperfctr *perfctr)
{
	if (atomic_dec_and_test(&perfctr->count))
		vperfctr_free(perfctr);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
static void scheduled_vperfctr_free(struct work_struct *work)
{
	struct vperfctr *perfctr = container_of(work, struct vperfctr, work);
	vperfctr_free(perfctr);
}
#else
static void scheduled_vperfctr_free(void *data)
{
	vperfctr_free((struct vperfctr*)data);
}
#endif

static void schedule_put_vperfctr(struct vperfctr *perfctr)
{
	if (!atomic_dec_and_test(&perfctr->count))
		return;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
	INIT_WORK(&perfctr->work, scheduled_vperfctr_free);
#else
	INIT_WORK(&perfctr->work, scheduled_vperfctr_free, perfctr);
#endif
	schedule_work(&perfctr->work);
}

static unsigned long long new_inheritance_id(void)
{
	static spinlock_t lock = SPIN_LOCK_UNLOCKED;
	static unsigned long long counter;
	unsigned long long id;

	spin_lock(&lock);
	id = ++counter;
	spin_unlock(&lock);
	return id;
}

/****************************************************************
 *								*
 * Basic counter operations.					*
 * These must all be called by the owner process only.		*
 * These must all be called with preemption disabled.		*
 *								*
 ****************************************************************/

/* PRE: IS_RUNNING(perfctr)
 * Suspend the counters.
 */
static inline void vperfctr_suspend(struct vperfctr *perfctr)
{
	perfctr_cpu_suspend(&perfctr->cpu_state);
}

static inline void vperfctr_reset_sampling_timer(struct vperfctr *perfctr)
{
	/* XXX: base the value on perfctr_info.cpu_khz instead! */
	perfctr->sampling_timer = HZ/2;
}

/* PRE: perfctr == current->thread.perfctr && IS_RUNNING(perfctr)
 * Restart the counters.
 */
static inline void vperfctr_resume(struct vperfctr *perfctr)
{
	perfctr_cpu_resume(&perfctr->cpu_state);
	vperfctr_reset_sampling_timer(perfctr);
}

static inline void vperfctr_resume_with_overflow_check(struct vperfctr *perfctr)
{
#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
	if (perfctr_cpu_has_pending_interrupt(&perfctr->cpu_state)) {
		vperfctr_handle_overflow(current, perfctr);
		return;
	}
#endif
	vperfctr_resume(perfctr);
}

/* Sample the counters but do not suspend them. */
static void vperfctr_sample(struct vperfctr *perfctr)
{
	if (IS_RUNNING(perfctr)) {
		perfctr_cpu_sample(&perfctr->cpu_state);
		vperfctr_reset_sampling_timer(perfctr);
	}
}

#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
/* vperfctr interrupt handler (XXX: add buffering support) */
/* PREEMPT note: called in IRQ context with preemption disabled. */
static void vperfctr_ihandler(unsigned long pc)
{
	struct task_struct *tsk = current;
	struct vperfctr *perfctr;

	perfctr = tsk->thread.perfctr;
	if (!perfctr) {
		printk(KERN_ERR "%s: BUG! pid %d has no vperfctr\n",
		       __FUNCTION__, tsk->pid);
		return;
	}
	if (!perfctr_cstatus_has_ictrs(perfctr->cpu_state.user.cstatus)) {
		printk(KERN_ERR "%s: BUG! vperfctr has cstatus %#x (pid %d, comm %s)\n",
		       __FUNCTION__, perfctr->cpu_state.user.cstatus, tsk->pid, tsk->comm);
		return;
	}
	vperfctr_suspend(perfctr);
	vperfctr_handle_overflow(tsk, perfctr);
}

static void vperfctr_handle_overflow(struct task_struct *tsk,
				     struct vperfctr *perfctr)
{
	unsigned int pmc_mask;
	siginfo_t si;
	sigset_t old_blocked;

	pmc_mask = perfctr_cpu_identify_overflow(&perfctr->cpu_state);
	if (!pmc_mask) {
#ifdef CONFIG_PPC64
		/* On some hardware (ppc64, in particular) it's
		 * impossible to control interrupts finely enough to
		 * eliminate overflows on counters we don't care
		 * about.  So in this case just restart the counters
		 * and keep going. */
		vperfctr_resume(perfctr);
#else
		printk(KERN_ERR "%s: BUG! pid %d has unidentifiable overflow source\n",
		       __FUNCTION__, tsk->pid);
#endif
		return;
	}
	perfctr->ireload_needed = 1;
	/* suspend a-mode and i-mode PMCs, leaving only TSC on */
	/* XXX: some people also want to suspend the TSC */
	perfctr->resume_cstatus = perfctr->cpu_state.user.cstatus;
	if (perfctr_cstatus_has_tsc(perfctr->resume_cstatus)) {
		perfctr->cpu_state.user.cstatus = perfctr_mk_cstatus(1, 0, 0);
		vperfctr_resume(perfctr);
	} else
		perfctr->cpu_state.user.cstatus = 0;
	si.si_signo = perfctr->si_signo;
	si.si_errno = 0;
	si.si_code = SI_PMC_OVF;
	si.si_pmc_ovf_mask = pmc_mask;

	/* deliver signal without waking up the receiver */
	spin_lock_irq(&tsk->sighand->siglock);
	old_blocked = tsk->blocked;
	sigaddset(&tsk->blocked, si.si_signo);
	spin_unlock_irq(&tsk->sighand->siglock);

	if (!send_sig_info(si.si_signo, &si, tsk))
		send_sig(si.si_signo, tsk, 1);

	spin_lock_irq(&tsk->sighand->siglock);
	tsk->blocked = old_blocked;
	recalc_sigpending();
	spin_unlock_irq(&tsk->sighand->siglock);
}
#endif

/****************************************************************
 *								*
 * Process management operations.				*
 * These must all, with the exception of vperfctr_unlink()	*
 * and __vperfctr_set_cpus_allowed(), be called by the owner	*
 * process only.						*
 *								*
 ****************************************************************/

/* do_fork() -> copy_process() -> copy_thread() -> __vperfctr_copy().
 * Inherit the parent's perfctr settings to the child.
 * PREEMPT note: do_fork() etc do not run with preemption disabled.
*/
void __vperfctr_copy(struct task_struct *child_tsk, struct pt_regs *regs)
{
	struct vperfctr *parent_perfctr;
	struct vperfctr *child_perfctr;

	/* Do not inherit perfctr settings to kernel-generated
	   threads, like those created by kmod. */
	child_perfctr = NULL;
	if (!user_mode(regs))
		goto out;

	/* Allocation may sleep. Do it before the critical region. */
	child_perfctr = get_empty_vperfctr();
	if (IS_ERR(child_perfctr)) {
		child_perfctr = NULL;
		goto out;
	}

	/* Although we're executing in the parent, if it is scheduled
	   then a remote monitor may attach and change the perfctr
	   pointer or the object it points to. This may already have
	   occurred when we get here, so the old copy of the pointer
	   in the child cannot be trusted. */
	preempt_disable();
	parent_perfctr = current->thread.perfctr;
	if (parent_perfctr) {
		child_perfctr->cpu_state.control = parent_perfctr->cpu_state.control;
		child_perfctr->si_signo = parent_perfctr->si_signo;
		child_perfctr->inheritance_id = parent_perfctr->inheritance_id;
	}
	preempt_enable();
	if (!parent_perfctr) {
		put_vperfctr(child_perfctr);
		child_perfctr = NULL;
		goto out;
	}
	(void)perfctr_cpu_update_control(&child_perfctr->cpu_state, 0);
	child_perfctr->owner = child_tsk;
 out:
	child_tsk->thread.perfctr = child_perfctr;
}

/* Called from exit_thread() or do_vperfctr_unlink().
 * If the counters are running, stop them and sample their final values.
 * Mark the vperfctr object as dead.
 * Optionally detach the vperfctr object from its owner task.
 * PREEMPT note: exit_thread() does not run with preemption disabled.
 */
static void vperfctr_unlink(struct task_struct *owner, struct vperfctr *perfctr, int do_unlink)
{
	/* this synchronises with sys_vperfctr() */
	spin_lock(&perfctr->owner_lock);
	perfctr->owner = NULL;
	spin_unlock(&perfctr->owner_lock);

	/* perfctr suspend+detach must be atomic wrt process suspend */
	/* this also synchronises with perfctr_set_cpus_allowed() */
	task_lock(owner);
	if (IS_RUNNING(perfctr) && owner == current)
		vperfctr_suspend(perfctr);
	if (do_unlink)
		owner->thread.perfctr = NULL;
	task_unlock(owner);

	perfctr->cpu_state.user.cstatus = 0;
	perfctr->resume_cstatus = 0;
	if (do_unlink)
		put_vperfctr(perfctr);
}

void __vperfctr_exit(struct vperfctr *perfctr)
{
	vperfctr_unlink(current, perfctr, 0);
}

/* release_task() -> perfctr_release_task() -> __vperfctr_release().
 * A task is being released. If it inherited its perfctr settings
 * from its parent, then merge its final counts back into the parent.
 * Then unlink the child's perfctr.
 * PRE: caller has write_lock_irq(&tasklist_lock).
 * PREEMPT note: preemption is disabled due to tasklist_lock.
 *
 * When current == parent_tsk, the child's counts can be merged
 * into the parent's immediately. This is the common case.
 *
 * When current != parent_tsk, the parent must be task_lock()ed
 * before its perfctr state can be accessed. task_lock() is illegal
 * here due to the write_lock_irq(&tasklist_lock) in release_task(),
 * so the operation is done via schedule_work().
 */
static void do_vperfctr_release(struct vperfctr *child_perfctr, struct task_struct *parent_tsk)
{
	struct vperfctr *parent_perfctr;
	unsigned int cstatus, nrctrs, i;

	parent_perfctr = parent_tsk->thread.perfctr;
	if (parent_perfctr && child_perfctr) {
		spin_lock(&parent_perfctr->children_lock);
		if (parent_perfctr->inheritance_id == child_perfctr->inheritance_id) {
			cstatus = parent_perfctr->cpu_state.user.cstatus;
			if (perfctr_cstatus_has_tsc(cstatus))
				parent_perfctr->children.tsc +=
					child_perfctr->cpu_state.user.tsc_sum +
					child_perfctr->children.tsc;
			nrctrs = perfctr_cstatus_nrctrs(cstatus);
			for(i = 0; i < nrctrs; ++i)
				parent_perfctr->children.pmc[i] +=
					child_perfctr->cpu_state.user.pmc[i].sum +
					child_perfctr->children.pmc[i];
		}
		spin_unlock(&parent_perfctr->children_lock);
	}
	schedule_put_vperfctr(child_perfctr);
}

static void do_scheduled_release(struct vperfctr *child_perfctr)
{
	struct task_struct *parent_tsk = child_perfctr->parent_tsk;

	task_lock(parent_tsk);
	do_vperfctr_release(child_perfctr, parent_tsk);
	task_unlock(parent_tsk);
	put_task_struct(parent_tsk);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
static void scheduled_release(struct work_struct *work)
{
	struct vperfctr *perfctr = container_of(work, struct vperfctr, work);
	do_scheduled_release(perfctr);
}
#else
static void scheduled_release(void *data)
{
	do_scheduled_release((struct vperfctr*)data);
}
#endif

void __vperfctr_release(struct task_struct *child_tsk)
{
	struct task_struct *parent_tsk = child_tsk->parent;
	struct vperfctr *child_perfctr = child_tsk->thread.perfctr;

	child_tsk->thread.perfctr = NULL;
	if (parent_tsk == current)
		do_vperfctr_release(child_perfctr, parent_tsk);
	else {
		get_task_struct(parent_tsk);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
		INIT_WORK(&child_perfctr->work, scheduled_release);
#else
		INIT_WORK(&child_perfctr->work, scheduled_release, child_perfctr);
#endif
		child_perfctr->parent_tsk = parent_tsk;
		schedule_work(&child_perfctr->work);
	}
}

/* schedule() --> switch_to() --> .. --> __vperfctr_suspend().
 * If the counters are running, suspend them.
 * PREEMPT note: switch_to() runs with preemption disabled.
 */
void __vperfctr_suspend(struct vperfctr *perfctr)
{
	if (IS_RUNNING(perfctr))
		vperfctr_suspend(perfctr);
}

/* schedule() --> switch_to() --> .. --> __vperfctr_resume().
 * PRE: perfctr == current->thread.perfctr
 * If the counters are runnable, resume them.
 * PREEMPT note: switch_to() runs with preemption disabled.
 */
void __vperfctr_resume(struct vperfctr *perfctr)
{
	if (IS_RUNNING(perfctr)) {
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
		if (unlikely(atomic_read(&perfctr->bad_cpus_allowed)) &&
		    perfctr_cstatus_nrctrs(perfctr->cpu_state.user.cstatus)) {
			perfctr->cpu_state.user.cstatus = 0;
			perfctr->resume_cstatus = 0;
			BUG_ON(current->state != TASK_RUNNING);
			send_sig(SIGILL, current, 1);
			return;
		}
#endif
		vperfctr_resume_with_overflow_check(perfctr);
	}
}

/* Called from update_one_process() [triggered by timer interrupt].
 * PRE: perfctr == current->thread.perfctr.
 * Sample the counters but do not suspend them.
 * Needed to avoid precision loss due to multiple counter
 * wraparounds between resume/suspend for CPU-bound processes.
 * PREEMPT note: called in IRQ context with preemption disabled.
 */
void __vperfctr_sample(struct vperfctr *perfctr)
{
	if (--perfctr->sampling_timer == 0)
		vperfctr_sample(perfctr);
}

#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
/* Called from set_cpus_allowed().
 * PRE: current holds task_lock(owner)
 * PRE: owner->thread.perfctr == perfctr
 */
void __vperfctr_set_cpus_allowed(struct task_struct *owner,
				 struct vperfctr *perfctr,
				 cpumask_t new_mask)
{
	if (cpus_intersects(new_mask, perfctr_cpus_forbidden_mask)) {
		atomic_set(&perfctr->bad_cpus_allowed, 1);
		if (printk_ratelimit())
			printk(KERN_WARNING "perfctr: process %d (comm %s) issued unsafe"
			       " set_cpus_allowed() on process %d (comm %s)\n",
			       current->pid, current->comm, owner->pid, owner->comm);
	} else
		atomic_set(&perfctr->bad_cpus_allowed, 0);
}
#endif

/****************************************************************
 *								*
 * Virtual perfctr system calls implementation.			*
 * These can be called by the owner process (tsk == current),	*
 * a monitor process which has the owner under ptrace ATTACH	*
 * control (tsk && tsk != current), or anyone with a handle to	*
 * an unlinked perfctr (!tsk).					*
 *								*
 ****************************************************************/

static int do_vperfctr_write(struct vperfctr *perfctr,
			     unsigned int domain,
			     const void __user *srcp,
			     unsigned int srcbytes,
			     struct task_struct *tsk)
{
	void *tmp;
	int err;

	if (!tsk)
		return -ESRCH;	/* attempt to update unlinked perfctr */

	if (srcbytes > PAGE_SIZE) /* primitive sanity check */
		return -EINVAL;
	tmp = kmalloc(srcbytes, GFP_USER);
	if (!tmp)
		return -ENOMEM;
	err = -EFAULT;
	if (copy_from_user(tmp, srcp, srcbytes))
		goto out_kfree;

	/* PREEMPT note: preemption is disabled over the entire
	   region since we're updating an active perfctr. */
	preempt_disable();
	if (IS_RUNNING(perfctr)) {
		if (tsk == current)
			vperfctr_suspend(perfctr);
		perfctr->cpu_state.user.cstatus = 0;
		perfctr->resume_cstatus = 0;
	}

	switch (domain) {
	case VPERFCTR_DOMAIN_CONTROL: {
		struct vperfctr_control control;

		err = -EINVAL;
		if (srcbytes > sizeof(control))
			break;
		control.si_signo = perfctr->si_signo;
		control.preserve = perfctr->preserve;
		memcpy(&control, tmp, srcbytes);
		/* XXX: validate si_signo? */
		perfctr->si_signo = control.si_signo;
		perfctr->preserve = control.preserve;
		err = 0;
		break;
	}
	case PERFCTR_DOMAIN_CPU_CONTROL:
		err = -EINVAL;
		if (srcbytes > sizeof(perfctr->cpu_state.control.header))
			break;
		memcpy(&perfctr->cpu_state.control.header, tmp, srcbytes);
		err = 0;
		break;
	case PERFCTR_DOMAIN_CPU_MAP:
		err = -EINVAL;
		if (srcbytes > sizeof(perfctr->cpu_state.control.pmc_map))
			break;
		memcpy(perfctr->cpu_state.control.pmc_map, tmp, srcbytes);
		err = 0;
		break;
	default:
		err = perfctr_cpu_control_write(&perfctr->cpu_state.control,
						domain, tmp, srcbytes);
	}

	preempt_enable();
 out_kfree:
	kfree(tmp);
	return err;
}

static int vperfctr_enable_control(struct vperfctr *perfctr, struct task_struct *tsk)
{
	int err;
	unsigned int next_cstatus;
	unsigned int nrctrs, i;

	if (perfctr->cpu_state.control.header.nractrs ||
	    perfctr->cpu_state.control.header.nrictrs) {
		cpumask_t old_mask, new_mask;

		old_mask = tsk->cpus_allowed;
		cpus_andnot(new_mask, old_mask, perfctr_cpus_forbidden_mask);

		if (cpus_empty(new_mask))
			return -EINVAL;
		if (!cpus_equal(new_mask, old_mask))
			set_cpus_allowed(tsk, new_mask);
	}

	perfctr->cpu_state.user.cstatus = 0;
	perfctr->resume_cstatus = 0;

	/* remote access note: perfctr_cpu_update_control() is ok */
	err = perfctr_cpu_update_control(&perfctr->cpu_state, 0);
	if (err < 0)
		return err;
	next_cstatus = perfctr->cpu_state.user.cstatus;
	if (!perfctr_cstatus_enabled(next_cstatus))
		return 0;

	if (!perfctr_cstatus_has_tsc(next_cstatus))
		perfctr->cpu_state.user.tsc_sum = 0;

	nrctrs = perfctr_cstatus_nrctrs(next_cstatus);
	for(i = 0; i < nrctrs; ++i)
		if (!(perfctr->preserve & (1<<i)))
			perfctr->cpu_state.user.pmc[i].sum = 0;

	spin_lock(&perfctr->children_lock);
	perfctr->inheritance_id = new_inheritance_id();
	memset(&perfctr->children, 0, sizeof perfctr->children);
	spin_unlock(&perfctr->children_lock);

	return 0;
}

static inline void vperfctr_ireload(struct vperfctr *perfctr)
{
#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
	if (perfctr->ireload_needed) {
		perfctr->ireload_needed = 0;
		/* remote access note: perfctr_cpu_ireload() is ok */
		perfctr_cpu_ireload(&perfctr->cpu_state);
	}
#endif
}

static int do_vperfctr_resume(struct vperfctr *perfctr, struct task_struct *tsk)
{
	unsigned int resume_cstatus;
	int ret;

	if (!tsk)
		return -ESRCH;	/* attempt to update unlinked perfctr */

	/* PREEMPT note: preemption is disabled over the entire
	   region because we're updating an active perfctr. */
	preempt_disable();

	if (IS_RUNNING(perfctr) && tsk == current)
		vperfctr_suspend(perfctr);

	resume_cstatus = perfctr->resume_cstatus;
	if (perfctr_cstatus_enabled(resume_cstatus)) {
		perfctr->cpu_state.user.cstatus = resume_cstatus;
		perfctr->resume_cstatus = 0;
		vperfctr_ireload(perfctr);
		ret = 0;
	} else {
		ret = vperfctr_enable_control(perfctr, tsk);
		resume_cstatus = perfctr->cpu_state.user.cstatus;
	}

	if (ret >= 0 && perfctr_cstatus_enabled(resume_cstatus) && tsk == current)
		vperfctr_resume(perfctr);

	preempt_enable();

	return ret;
}

static int do_vperfctr_suspend(struct vperfctr *perfctr, struct task_struct *tsk)
{
	if (!tsk)
		return -ESRCH;	/* attempt to update unlinked perfctr */

	/* PREEMPT note: preemption is disabled over the entire
	   region since we're updating an active perfctr. */
	preempt_disable();

	if (IS_RUNNING(perfctr)) {
		if (tsk == current)
			vperfctr_suspend(perfctr);
		perfctr->resume_cstatus = perfctr->cpu_state.user.cstatus;
		perfctr->cpu_state.user.cstatus = 0;
	}

	preempt_enable();

	return 0;
}

static int do_vperfctr_unlink(struct vperfctr *perfctr, struct task_struct *tsk)
{
	if (tsk)
		vperfctr_unlink(tsk, perfctr, 1);
	return 0;
}

static int do_vperfctr_clear(struct vperfctr *perfctr, struct task_struct *tsk)
{
	if (!tsk)
		return -ESRCH;	/* attempt to update unlinked perfctr */

	/* PREEMPT note: preemption is disabled over the entire
	   region because we're updating an active perfctr. */
	preempt_disable();

	if (IS_RUNNING(perfctr) && tsk == current)
		vperfctr_suspend(perfctr);

	memset(&perfctr->cpu_state, 0, sizeof perfctr->cpu_state);
	perfctr->resume_cstatus = 0;

	spin_lock(&perfctr->children_lock);
	perfctr->inheritance_id = 0;
	memset(&perfctr->children, 0, sizeof perfctr->children);
	spin_unlock(&perfctr->children_lock);

	preempt_enable();

	return 0;
}

static int do_vperfctr_control(struct vperfctr *perfctr,
			       unsigned int cmd,
			       struct task_struct *tsk)
{
	switch (cmd) {
	case VPERFCTR_CONTROL_UNLINK:
		return do_vperfctr_unlink(perfctr, tsk);
	case VPERFCTR_CONTROL_SUSPEND:
		return do_vperfctr_suspend(perfctr, tsk);
	case VPERFCTR_CONTROL_RESUME:
		return do_vperfctr_resume(perfctr, tsk);
	case VPERFCTR_CONTROL_CLEAR:
		return do_vperfctr_clear(perfctr, tsk);
	default:
		return -EINVAL;
	}
}

static int do_vperfctr_read(struct vperfctr *perfctr,
			    unsigned int domain,
			    void __user *dstp,
			    unsigned int dstbytes,
			    struct task_struct *tsk)
{
	union {
		struct perfctr_sum_ctrs sum;
		struct vperfctr_control control;
		struct perfctr_sum_ctrs children;
	} *tmp;
	unsigned int tmpbytes;
	int ret;

	tmpbytes = dstbytes;
	if (tmpbytes > PAGE_SIZE) /* primitive sanity check */
		return -EINVAL;
	if (tmpbytes < sizeof(*tmp))
		tmpbytes = sizeof(*tmp);
	tmp = kmalloc(tmpbytes, GFP_USER);
	if (!tmp)
		return -ENOMEM;

	/* PREEMPT note: While we're reading our own control, another
	   process may ptrace ATTACH to us and update our control.
	   Disable preemption to ensure we get a consistent copy.
	   Not needed for other cases since the perfctr is either
	   unlinked or its owner is ptrace ATTACH suspended by us. */
	if (tsk == current)
		preempt_disable();

	switch (domain) {
	case VPERFCTR_DOMAIN_SUM: {
		int j;

		vperfctr_sample(perfctr);
		tmp->sum.tsc = perfctr->cpu_state.user.tsc_sum;
		for(j = 0; j < ARRAY_SIZE(tmp->sum.pmc); ++j)
			tmp->sum.pmc[j] = perfctr->cpu_state.user.pmc[j].sum;
		ret = sizeof(tmp->sum);
		break;
	}
	case VPERFCTR_DOMAIN_CONTROL:
		tmp->control.si_signo = perfctr->si_signo;
		tmp->control.preserve = perfctr->preserve;
		ret = sizeof(tmp->control);
		break;
	case VPERFCTR_DOMAIN_CHILDREN:
		if (tsk)
			spin_lock(&perfctr->children_lock);
		tmp->children = perfctr->children;
		if (tsk)
			spin_unlock(&perfctr->children_lock);
		ret = sizeof(tmp->children);
		break;
	case PERFCTR_DOMAIN_CPU_CONTROL:
		if (tmpbytes > sizeof(perfctr->cpu_state.control.header))
			tmpbytes = sizeof(perfctr->cpu_state.control.header);
		memcpy(tmp, &perfctr->cpu_state.control.header, tmpbytes);
		ret = tmpbytes;
		break;
	case PERFCTR_DOMAIN_CPU_MAP:
		if (tmpbytes > sizeof(perfctr->cpu_state.control.pmc_map))
			tmpbytes = sizeof(perfctr->cpu_state.control.pmc_map);
		memcpy(tmp, perfctr->cpu_state.control.pmc_map, tmpbytes);
		ret = tmpbytes;
		break;
	default:
		ret = -EFAULT;
		if (copy_from_user(tmp, dstp, dstbytes) == 0)
			ret = perfctr_cpu_control_read(&perfctr->cpu_state.control,
						       domain, tmp, dstbytes);
	}

	if (tsk == current)
		preempt_enable();

	if (ret > 0) {
		if (ret > dstbytes)
			ret = dstbytes;
		if (ret > 0 && copy_to_user(dstp, tmp, ret))
			ret = -EFAULT;
	}
	kfree(tmp);
	return ret;
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
	if ((vma->vm_end - vma->vm_start) != PAGE_SIZE ||
	    vma->vm_pgoff != 0 ||
	    (pgprot_val(vma->vm_page_prot) & _PAGE_RW) ||
	    (vma->vm_flags & (VM_WRITE | VM_MAYWRITE)))
		return -EPERM;
	perfctr = filp->private_data;
	if (!perfctr)
		return -EPERM;
	return remap_pfn_range(vma, vma->vm_start,
				virt_to_phys(perfctr) >> PAGE_SHIFT,
				PAGE_SIZE, vma->vm_page_prot);
}

static int vperfctr_release(struct inode *inode, struct file *filp)
{
	struct vperfctr *perfctr = filp->private_data;
	filp->private_data = NULL;
	if (perfctr)
		put_vperfctr(perfctr);
	return 0;
}

static struct file_operations vperfctr_file_ops = {
	.mmap = vperfctr_mmap,
	.release = vperfctr_release,
};

/****************************************************************
 *								*
 * File system for virtual perfctrs. Based on pipefs.		*
 *								*
 ****************************************************************/

#define VPERFCTRFS_MAGIC (('V'<<24)|('P'<<16)|('M'<<8)|('C'))

/* The code to set up a `struct file_system_type' for a pseudo fs
   is unfortunately not the same in 2.4 and 2.6. */
#include <linux/mount.h> /* needed for 2.6, included by fs.h in 2.4 */

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,18)
static int
vperfctrfs_get_sb(struct file_system_type *fs_type,
		  int flags, const char *dev_name, void *data,
		  struct vfsmount *mnt)
{
	return get_sb_pseudo(fs_type, "vperfctr:", NULL, VPERFCTRFS_MAGIC, mnt);
}
#else
static struct super_block *
vperfctrfs_get_sb(struct file_system_type *fs_type,
		  int flags, const char *dev_name, void *data)
{
	return get_sb_pseudo(fs_type, "vperfctr:", NULL, VPERFCTRFS_MAGIC);
}
#endif

static struct file_system_type vperfctrfs_type = {
	.name		= "vperfctrfs",
	.get_sb		= vperfctrfs_get_sb,
	.kill_sb	= kill_anon_super,
};

/* XXX: check if s/vperfctr_mnt/vperfctrfs_type.kern_mnt/ would work */
static struct vfsmount *vperfctr_mnt;
#define vperfctr_fs_init_done()	(vperfctr_mnt != NULL)

static int __init vperfctrfs_init(void)
{
	int err = register_filesystem(&vperfctrfs_type);
	if (!err) {
		vperfctr_mnt = kern_mount(&vperfctrfs_type);
		if (!IS_ERR(vperfctr_mnt))
			return 0;
		err = PTR_ERR(vperfctr_mnt);
		unregister_filesystem(&vperfctrfs_type);
		vperfctr_mnt = NULL;
	}
	return err;
}

static void __exit vperfctrfs_exit(void)
{
	unregister_filesystem(&vperfctrfs_type);
	mntput(vperfctr_mnt);
}

static struct inode *vperfctr_get_inode(void)
{
	struct inode *inode;

	inode = new_inode(vperfctr_mnt->mnt_sb);
	if (!inode)
		return NULL;
	inode->i_fop = &vperfctr_file_ops;
	inode->i_state = I_DIRTY;
	inode->i_mode = S_IFCHR | S_IRUSR | S_IWUSR;
	inode->i_uid = current->fsuid;
	inode->i_gid = current->fsgid;
	inode->i_atime = inode->i_mtime = inode->i_ctime = CURRENT_TIME;
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19) && !DONT_HAVE_i_blksize
	inode->i_blksize = 0;
#endif
	return inode;
}

static int vperfctrfs_delete_dentry(struct dentry *dentry)
{
	return 1;
}

static struct dentry_operations vperfctrfs_dentry_operations = {
	.d_delete	= vperfctrfs_delete_dentry,
};

static struct dentry *vperfctr_d_alloc_root(struct inode *inode)
{
	struct qstr this;
	char name[32];
	struct dentry *dentry;

	sprintf(name, "[%lu]", inode->i_ino);
	this.name = name;
	this.len = strlen(name);
	this.hash = inode->i_ino; /* will go */
	dentry = d_alloc(vperfctr_mnt->mnt_sb->s_root, &this);
	if (dentry) {
		dentry->d_op = &vperfctrfs_dentry_operations;
		d_add(dentry, inode);
	}
	return dentry;
}

static struct file *vperfctr_get_filp(void)
{
	struct file *filp;
	struct inode *inode;
	struct dentry *dentry;

	filp = get_empty_filp();
	if (!filp)
		goto out;
	inode = vperfctr_get_inode();
	if (!inode)
		goto out_filp;
	dentry = vperfctr_d_alloc_root(inode);
	if (!dentry)
		goto out_inode;

	filp->f_vfsmnt = mntget(vperfctr_mnt);
	filp->f_dentry = dentry;
	filp->f_mapping = dentry->d_inode->i_mapping;

	filp->f_pos = 0;
	filp->f_flags = 0;
	filp->f_op = &vperfctr_file_ops; /* fops_get() if MODULE */
	filp->f_mode = FMODE_READ;
	filp->f_version = 0;

	return filp;

 out_inode:
	iput(inode);
 out_filp:
	put_filp(filp);	/* doesn't run ->release() like fput() does */
 out:
	return NULL;
}

/****************************************************************
 *								*
 * Virtual perfctr actual system calls.				*
 *								*
 ****************************************************************/

/* tid is the actual task/thread id (née pid, stored as ->pid),
   pid/tgid is that 2.6 thread group id crap (stored as ->tgid) */
asmlinkage long sys_vperfctr_open(int tid, int creat)
{
	struct file *filp;
	struct task_struct *tsk;
	struct vperfctr *perfctr;
	int err;
	int fd;

	if (!vperfctr_fs_init_done())
		return -ENODEV;
	filp = vperfctr_get_filp();
	if (!filp)
		return -ENOMEM;
	err = fd = get_unused_fd();
	if (err < 0)
		goto err_filp;
	perfctr = NULL;
	if (creat) {
		perfctr = get_empty_vperfctr(); /* may sleep */
		if (IS_ERR(perfctr)) {
			err = PTR_ERR(perfctr);
			goto err_fd;
		}
	}
	tsk = current;
	if (tid != 0 && tid != tsk->pid) { /* remote? */
		read_lock(&tasklist_lock);
		tsk = find_task_by_pid(tid);
		if (tsk)
			get_task_struct(tsk);
		read_unlock(&tasklist_lock);
		err = -ESRCH;
		if (!tsk)
			goto err_perfctr;
		err = ptrace_check_attach(tsk, 0);
		if (err < 0)
			goto err_tsk;
	}
	if (creat) {
		/* check+install must be atomic to prevent remote-control races */
		task_lock(tsk);
		if (!tsk->thread.perfctr) {
			perfctr->owner = tsk;
			tsk->thread.perfctr = perfctr;
			err = 0;
		} else
			err = -EEXIST;
		task_unlock(tsk);
		if (err)
			goto err_tsk;
	} else {
		perfctr = tsk->thread.perfctr;
		/* XXX: Old API needed to allow NULL perfctr here.
		   Do we want to keep or change that rule? */
	}
	filp->private_data = perfctr;
	if (perfctr)
		atomic_inc(&perfctr->count);
	if (tsk != current)
		put_task_struct(tsk);
	fd_install(fd, filp);
	return fd;
 err_tsk:
	if (tsk != current)
		put_task_struct(tsk);
 err_perfctr:
	if (perfctr)	/* can only occur if creat != 0 */
		put_vperfctr(perfctr);
 err_fd:
	put_unused_fd(fd);
 err_filp:
	fput(filp);
	return err;
}

static struct vperfctr *fd_get_vperfctr(int fd)
{
	struct vperfctr *perfctr;
	struct file *filp;
	int err;

	err = -EBADF;
	filp = fget(fd);
	if (!filp)
		goto out;
	err = -EINVAL;
	if (filp->f_op != &vperfctr_file_ops)
		goto out_filp;
	perfctr = filp->private_data;
	if (!perfctr)
		goto out_filp;
	atomic_inc(&perfctr->count);
	fput(filp);
	return perfctr;
 out_filp:
	fput(filp);
 out:
	return ERR_PTR(err);
}

static struct task_struct *vperfctr_get_tsk(struct vperfctr *perfctr)
{
	struct task_struct *tsk;

	tsk = current;
	if (perfctr != current->thread.perfctr) {
		/* this synchronises with vperfctr_unlink() and itself */
		spin_lock(&perfctr->owner_lock);
		tsk = perfctr->owner;
		if (tsk)
			get_task_struct(tsk);
		spin_unlock(&perfctr->owner_lock);
		if (tsk) {
			int ret = ptrace_check_attach(tsk, 0);
			if (ret < 0) {
				put_task_struct(tsk);
				return ERR_PTR(ret);
			}
		}
	}
	return tsk;
}

static void vperfctr_put_tsk(struct task_struct *tsk)
{
	if (tsk && tsk != current)
		put_task_struct(tsk);
}

asmlinkage long sys_vperfctr_write(int fd, unsigned int domain,
				   const void __user *argp,
				   unsigned int argbytes)
{
	struct vperfctr *perfctr;
	struct task_struct *tsk;
	int ret;

	perfctr = fd_get_vperfctr(fd);
	if (IS_ERR(perfctr))
		return PTR_ERR(perfctr);
	tsk = vperfctr_get_tsk(perfctr);
	if (IS_ERR(tsk)) {
		ret = PTR_ERR(tsk);
		goto out;
	}
	ret = do_vperfctr_write(perfctr, domain, argp, argbytes, tsk);
	vperfctr_put_tsk(tsk);
 out:
	put_vperfctr(perfctr);
	return ret;
}

asmlinkage long sys_vperfctr_control(int fd, unsigned int cmd)
{
	struct vperfctr *perfctr;
	struct task_struct *tsk;
	int ret;

	perfctr = fd_get_vperfctr(fd);
	if (IS_ERR(perfctr))
		return PTR_ERR(perfctr);
	tsk = vperfctr_get_tsk(perfctr);
	if (IS_ERR(tsk)) {
		ret = PTR_ERR(tsk);
		goto out;
	}
	ret = do_vperfctr_control(perfctr, cmd, tsk);
	vperfctr_put_tsk(tsk);
 out:
	put_vperfctr(perfctr);
	return ret;
}

asmlinkage long sys_vperfctr_read(int fd, unsigned int domain,
				  void __user *argp, unsigned int argbytes)
{
	struct vperfctr *perfctr;
	struct task_struct *tsk;
	int ret;

	perfctr = fd_get_vperfctr(fd);
	if (IS_ERR(perfctr))
		return PTR_ERR(perfctr);
	tsk = vperfctr_get_tsk(perfctr);
	if (IS_ERR(tsk)) {
		ret = PTR_ERR(tsk);
		goto out;
	}
	ret = do_vperfctr_read(perfctr, domain, argp, argbytes, tsk);
	vperfctr_put_tsk(tsk);
 out:
	put_vperfctr(perfctr);
	return ret;
}

/****************************************************************
 *								*
 * module_init/exit						*
 *								*
 ****************************************************************/

int __init vperfctr_init(void)
{
	return vperfctrfs_init();
}

void __exit vperfctr_exit(void)
{
	vperfctrfs_exit();
}
