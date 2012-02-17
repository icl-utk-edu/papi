/* $Id: perfctr.h,v 1.95 2005/10/02 13:01:30 mikpe Exp $
 * Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2005  Mikael Pettersson
 */
#ifndef _LINUX_PERFCTR_H
#define _LINUX_PERFCTR_H

#ifdef CONFIG_PERFCTR	/* don't break archs without <asm/perfctr.h> */

#include <asm/perfctr.h>

/* cpu_features flag bits */
#define PERFCTR_FEATURE_RDPMC	0x01
#define PERFCTR_FEATURE_RDTSC	0x02
#define PERFCTR_FEATURE_PCINT	0x04

/* virtual perfctr control object */
struct vperfctr_control {
	__s32 si_signo;
	__u32 preserve;
};

/* commands for sys_vperfctr_control() */
#define VPERFCTR_CONTROL_UNLINK		0x01
#define VPERFCTR_CONTROL_SUSPEND	0x02
#define VPERFCTR_CONTROL_RESUME		0x03
#define VPERFCTR_CONTROL_CLEAR		0x04

/* common description of an arch-specific control register */
struct perfctr_cpu_reg {
	__u64 nr;
	__u64 value;
};

/* state and control domain numbers
   0-127 are for architecture-neutral domains
   128-255 are for architecture-specific domains */
#define VPERFCTR_DOMAIN_SUM		1	/* struct perfctr_sum_ctrs */
#define VPERFCTR_DOMAIN_CONTROL		2	/* struct vperfctr_control */
#define VPERFCTR_DOMAIN_CHILDREN	3	/* struct perfctr_sum_ctrs */

/* domain numbers for common arch-specific control data */
#define PERFCTR_DOMAIN_CPU_CONTROL	128	/* struct perfctr_cpu_control_header */
#define PERFCTR_DOMAIN_CPU_MAP		129	/* __u32[] */
#define PERFCTR_DOMAIN_CPU_REGS		130	/* struct perfctr_cpu_reg[] */

#endif	/* CONFIG_PERFCTR */

#ifdef __KERNEL__

/*
 * The perfctr system calls.
 */
asmlinkage long sys_vperfctr_open(int tid, int creat);
asmlinkage long sys_vperfctr_control(int fd, unsigned int cmd);
asmlinkage long sys_vperfctr_write(int fd, unsigned int domain,
				   const void __user *argp,
				   unsigned int argbytes);
asmlinkage long sys_vperfctr_read(int fd, unsigned int domain,
				  void __user *argp,
				  unsigned int argbytes);

struct perfctr_info {
	unsigned int cpu_features;
	unsigned int cpu_khz;
	unsigned int tsc_to_cpu_mult;
};

extern struct perfctr_info perfctr_info;

#ifdef CONFIG_PERFCTR_VIRTUAL

/*
 * Virtual per-process performance-monitoring counters.
 */
struct vperfctr;	/* opaque */

/* process management operations */
extern void __vperfctr_copy(struct task_struct*, struct pt_regs*);
extern void __vperfctr_release(struct task_struct*);
extern void __vperfctr_exit(struct vperfctr*);
extern void __vperfctr_suspend(struct vperfctr*);
extern void __vperfctr_resume(struct vperfctr*);
extern void __vperfctr_sample(struct vperfctr*);
extern void __vperfctr_set_cpus_allowed(struct task_struct*, struct vperfctr*, cpumask_t);

static inline void perfctr_copy_task(struct task_struct *tsk, struct pt_regs *regs)
{
	if (tsk->thread.perfctr)
		__vperfctr_copy(tsk, regs);
}

static inline void perfctr_release_task(struct task_struct *tsk)
{
	if (tsk->thread.perfctr)
		__vperfctr_release(tsk);
}

static inline void perfctr_exit_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if (perfctr)
		__vperfctr_exit(perfctr);
}

static inline void perfctr_suspend_thread(struct thread_struct *prev)
{
	struct vperfctr *perfctr;
	perfctr = prev->perfctr;
	if (perfctr)
		__vperfctr_suspend(perfctr);
}

static inline void perfctr_resume_thread(struct thread_struct *next)
{
	struct vperfctr *perfctr;
	perfctr = next->perfctr;
	if (perfctr)
		__vperfctr_resume(perfctr);
}

static inline void perfctr_sample_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if (perfctr)
		__vperfctr_sample(perfctr);
}

static inline void perfctr_set_cpus_allowed(struct task_struct *p, cpumask_t new_mask)
{
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
	struct vperfctr *perfctr;

	task_lock(p);
	perfctr = p->thread.perfctr;
	if (perfctr)
		__vperfctr_set_cpus_allowed(p, perfctr, new_mask);
	task_unlock(p);
#endif
}

#else	/* !CONFIG_PERFCTR_VIRTUAL */

static inline void perfctr_copy_task(struct task_struct *p, struct pt_regs *r) { }
static inline void perfctr_release_task(struct task_struct *p) { }
static inline void perfctr_exit_thread(struct thread_struct *t) { }
static inline void perfctr_suspend_thread(struct thread_struct *t) { }
static inline void perfctr_resume_thread(struct thread_struct *t) { }
static inline void perfctr_sample_thread(struct thread_struct *t) { }
static inline void perfctr_set_cpus_allowed(struct task_struct *p, cpumask_t m) { }

#endif	/* CONFIG_PERFCTR_VIRTUAL */

/* These routines are identical to write_seqcount_begin() and
 * write_seqcount_end(), except they take an explicit __u32 rather
 * than a seqcount_t.  That's because this sequence lock is user from
 * userspace, so we have to pin down the counter's type explicitly to
 * have a clear ABI.  They also omit the SMP write barriers since we
 * only support mmap() based sampling for self-monitoring tasks.
 */
static inline void write_perfseq_begin(__u32 *seq)
{
	++*seq;
}

static inline void write_perfseq_end(__u32 *seq)
{
	++*seq;
}

#endif	/* __KERNEL__ */

#endif	/* _LINUX_PERFCTR_H */
