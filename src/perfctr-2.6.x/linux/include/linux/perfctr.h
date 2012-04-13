/* $Id: perfctr.h,v 1.69.2.6 2009/01/23 17:31:57 mikpe Exp $
 * Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2009  Mikael Pettersson
 */
#ifndef _LINUX_PERFCTR_H
#define _LINUX_PERFCTR_H

#ifdef CONFIG_KPERFCTR	/* don't break archs without <asm/perfctr.h> */

#include <asm/perfctr.h>

struct perfctr_info {
	unsigned int abi_version;
	char driver_version[32];
	unsigned int cpu_type;
	unsigned int cpu_features;
	unsigned int cpu_khz;
	unsigned int tsc_to_cpu_mult;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

struct perfctr_cpu_mask {
	unsigned int nrwords;
	unsigned int mask[1];	/* actually 'nrwords' */
};

/* abi_version values: Lower 16 bits contain the CPU data version, upper
   16 bits contain the API version. Each half has a major version in its
   upper 8 bits, and a minor version in its lower 8 bits. */
#define PERFCTR_API_VERSION	0x0502	/* 5.2 */
#define PERFCTR_ABI_VERSION	((PERFCTR_API_VERSION<<16)|PERFCTR_CPU_VERSION)

/* cpu_features flag bits */
#define PERFCTR_FEATURE_RDPMC	0x01
#define PERFCTR_FEATURE_RDTSC	0x02
#define PERFCTR_FEATURE_PCINT	0x04

/* user's view of mmap:ed virtual perfctr */
struct vperfctr_state {
	struct perfctr_cpu_state cpu_state;
};

/* parameter in VPERFCTR_CONTROL command */
struct vperfctr_control {
	int si_signo;
	struct perfctr_cpu_control cpu_control;
	unsigned int preserve;
	unsigned int flags;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

/* vperfctr_control flags bits */
#define VPERFCTR_CONTROL_CLOEXEC	0x01	/* close (unlink) state before exec */

/* parameter in GPERFCTR_CONTROL command */
struct gperfctr_cpu_control {
	unsigned int cpu;
	struct perfctr_cpu_control cpu_control;
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

/* returned by GPERFCTR_READ command */
struct gperfctr_cpu_state {
	unsigned int cpu;
	struct perfctr_cpu_control cpu_control;
	struct perfctr_sum_ctrs sum;
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

/* buffer for encodings of most of the above structs */
struct perfctr_struct_buf {
	unsigned int rdsize;
	unsigned int wrsize;
	unsigned int buffer[1]; /* actually 'max(rdsize,wrsize)' */
};

#include <linux/ioctl.h>
#define _PERFCTR_IOCTL	0xD0	/* 'P'+128, currently unassigned */

#define PERFCTR_ABI		 _IOR(_PERFCTR_IOCTL,0,unsigned int)
#define PERFCTR_INFO		 _IOR(_PERFCTR_IOCTL,1,struct perfctr_struct_buf)
#define PERFCTR_CPUS		_IOWR(_PERFCTR_IOCTL,2,struct perfctr_cpu_mask)
#define PERFCTR_CPUS_FORBIDDEN	_IOWR(_PERFCTR_IOCTL,3,struct perfctr_cpu_mask)
#define VPERFCTR_CREAT		  _IO(_PERFCTR_IOCTL,6)/*int tid*/
#define VPERFCTR_OPEN		  _IO(_PERFCTR_IOCTL,7)/*int tid*/

#define VPERFCTR_READ_SUM	 _IOR(_PERFCTR_IOCTL,8,struct perfctr_struct_buf)
#define VPERFCTR_UNLINK		  _IO(_PERFCTR_IOCTL,9)
#define VPERFCTR_CONTROL	 _IOW(_PERFCTR_IOCTL,10,struct perfctr_struct_buf)
#define VPERFCTR_IRESUME	  _IO(_PERFCTR_IOCTL,11)
#define VPERFCTR_READ_CONTROL	 _IOR(_PERFCTR_IOCTL,12,struct perfctr_struct_buf)

#define GPERFCTR_CONTROL	_IOWR(_PERFCTR_IOCTL,16,struct perfctr_struct_buf)
#define GPERFCTR_READ		_IOWR(_PERFCTR_IOCTL,17,struct perfctr_struct_buf)
#define GPERFCTR_STOP		  _IO(_PERFCTR_IOCTL,18)
#define GPERFCTR_START		  _IO(_PERFCTR_IOCTL,19)/*unsigned int*/

#ifdef __KERNEL__
extern struct perfctr_info perfctr_info;
extern int sys_perfctr_abi(unsigned int*);
extern int sys_perfctr_info(struct perfctr_struct_buf*);
extern int sys_perfctr_cpus(struct perfctr_cpu_mask*);
extern int sys_perfctr_cpus_forbidden(struct perfctr_cpu_mask*);
#endif	/* __KERNEL__ */

#endif	/* CONFIG_KPERFCTR */

#ifdef __KERNEL__

#ifdef CONFIG_PERFCTR_VIRTUAL

/*
 * Virtual per-process performance-monitoring counters.
 */
struct vperfctr;	/* opaque */

/* process management operations */
extern void __vperfctr_exit(struct vperfctr*);
extern void __vperfctr_flush(struct vperfctr*);
extern void __vperfctr_suspend(struct vperfctr*);
extern void __vperfctr_resume(struct vperfctr*);
extern void __vperfctr_sample(struct vperfctr*);
extern void __vperfctr_set_cpus_allowed(struct task_struct*, struct vperfctr*, cpumask_t);

#ifdef CONFIG_PERFCTR_MODULE
extern struct vperfctr_stub {
	struct module *owner;
	void (*exit)(struct vperfctr*);
	void (*flush)(struct vperfctr*);
	void (*suspend)(struct vperfctr*);
	void (*resume)(struct vperfctr*);
	void (*sample)(struct vperfctr*);
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
	void (*set_cpus_allowed)(struct task_struct*, struct vperfctr*, cpumask_t);
#endif
} vperfctr_stub;
extern void _vperfctr_exit(struct vperfctr*);
extern void _vperfctr_flush(struct vperfctr*);
#define _vperfctr_suspend(x)	vperfctr_stub.suspend((x))
#define _vperfctr_resume(x)	vperfctr_stub.resume((x))
#define _vperfctr_sample(x)	vperfctr_stub.sample((x))
#define _vperfctr_set_cpus_allowed(x,y,z) (*vperfctr_stub.set_cpus_allowed)((x),(y),(z))
#else	/* !CONFIG_PERFCTR_MODULE */
#define _vperfctr_exit(x)	__vperfctr_exit((x))
#define _vperfctr_flush(x)	__vperfctr_flush((x))
#define _vperfctr_suspend(x)	__vperfctr_suspend((x))
#define _vperfctr_resume(x)	__vperfctr_resume((x))
#define _vperfctr_sample(x)	__vperfctr_sample((x))
#define _vperfctr_set_cpus_allowed(x,y,z) __vperfctr_set_cpus_allowed((x),(y),(z))
#endif	/* CONFIG_PERFCTR_MODULE */

static inline void perfctr_copy_task(struct task_struct *tsk, struct pt_regs *regs)
{
	tsk->thread.perfctr = NULL; /* inheritance is not yet implemented */
}

static inline void perfctr_release_task(struct task_struct *tsk)
{
	/* nothing to do until inheritance is implemented */
}

static inline void perfctr_exit_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if (perfctr)
		_vperfctr_exit(perfctr);
}

static inline void perfctr_flush_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if (perfctr)
		_vperfctr_flush(perfctr);
}

static inline void perfctr_suspend_thread(struct thread_struct *prev)
{
	struct vperfctr *perfctr;
	perfctr = prev->perfctr;
	if (perfctr)
		_vperfctr_suspend(perfctr);
}

static inline void perfctr_resume_thread(struct thread_struct *next)
{
	struct vperfctr *perfctr;
	perfctr = next->perfctr;
	if (perfctr)
		_vperfctr_resume(perfctr);
}

static inline void perfctr_sample_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if (perfctr)
		_vperfctr_sample(perfctr);
}

static inline void perfctr_set_cpus_allowed(struct task_struct *p, cpumask_t new_mask)
{
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
	struct vperfctr *perfctr;

	task_lock(p);
	perfctr = p->thread.perfctr;
	if (perfctr)
		_vperfctr_set_cpus_allowed(p, perfctr, new_mask);
	task_unlock(p);
#endif
}

#else	/* !CONFIG_PERFCTR_VIRTUAL */

static inline void perfctr_copy_task(struct task_struct *p, struct pt_regs *r) { }
static inline void perfctr_release_task(struct task_struct *p) { }
static inline void perfctr_exit_thread(struct thread_struct *t) { }
static inline void perfctr_flush_thread(struct thread_struct *t) { }
static inline void perfctr_suspend_thread(struct thread_struct *t) { }
static inline void perfctr_resume_thread(struct thread_struct *t) { }
static inline void perfctr_sample_thread(struct thread_struct *t) { }
static inline void perfctr_set_cpus_allowed(struct task_struct *p, cpumask_t m) { }

#endif	/* CONFIG_PERFCTR_VIRTUAL */

#endif	/* __KERNEL__ */

#endif	/* _LINUX_PERFCTR_H */
