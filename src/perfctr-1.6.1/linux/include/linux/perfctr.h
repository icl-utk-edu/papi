/* $Id$
 * Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#ifndef _LINUX_PERFCTR_H
#define _LINUX_PERFCTR_H

#include <asm/perfctr.h>

/* returned by the PERFCTR_INFO ioctl */
struct perfctr_info {
	unsigned char version_major;
	unsigned char version_minor;
	unsigned char version_micro;
	unsigned char nrcpus;
	unsigned char cpu_type;
	unsigned char cpu_features;
	unsigned long cpu_khz;
};
#define PERFCTR_X86_GENERIC	0	/* any x86 with rdtsc */
#define PERFCTR_X86_INTEL_P5	1	/* no rdpmc */
#define PERFCTR_X86_INTEL_P5MMX	2
#define PERFCTR_X86_INTEL_P6	3
#define PERFCTR_X86_INTEL_PII	4
#define PERFCTR_X86_INTEL_PIII	5
#define PERFCTR_X86_CYRIX_MII	6
#define PERFCTR_X86_WINCHIP_C6	7	/* no rdtsc */
#define PERFCTR_X86_WINCHIP_2	8	/* no rdtsc */
#define PERFCTR_X86_AMD_K7	9

#define PERFCTR_FEATURE_RDPMC	0x01
#define PERFCTR_FEATURE_RDTSC	0x02

/* user-visible state of an mmap:ed virtual perfctr */
struct vperfctr_state {
	int status;	/* DEAD, STOPPED, or # of active counters */
	unsigned int control_id;	/* identify inheritance trees */
	struct perfctr_control control;
	struct perfctr_sum_ctrs sum;
	struct perfctr_low_ctrs start;
	struct perfctr_sum_ctrs children;
};
#define VPERFCTR_STATUS_DEAD	-1
#define VPERFCTR_STATUS_STOPPED	0

/* parameter in GPERFCTR_CONTROL ioctl */
struct gperfctr_control {
	unsigned long interval_usec;
	unsigned int nrcpus;
	struct perfctr_control cpu_control[0];
};

/* returned by GPERFCTR_READ ioctl */
struct gperfctr_cpu_state {
	int nrctrs;
	struct perfctr_control control;
	struct perfctr_sum_ctrs sum;
};
struct gperfctr_state {
	unsigned nrcpus;
	struct gperfctr_cpu_state cpu_state[0];
};

#ifdef __KERNEL__

extern int sys_perfctr_info(struct perfctr_info*);

/*
 * Virtual per-process performance-monitoring counters.
 */
struct vperfctr;	/* opaque */

#ifdef CONFIG_PERFCTR_VIRTUAL

/* process management callbacks */
extern struct vperfctr *__vperfctr_copy(struct vperfctr*);
extern void __vperfctr_exit(struct vperfctr*);
extern void __vperfctr_release(struct vperfctr*, struct vperfctr*);
extern void __vperfctr_suspend(struct vperfctr*);
extern void __vperfctr_resume(struct vperfctr*);

#ifdef CONFIG_PERFCTR_MODULE
extern struct vperfctr_stub {
	struct vperfctr *(*copy)(struct vperfctr*);
	void (*exit)(struct vperfctr*);
	void (*release)(struct vperfctr*, struct vperfctr*);
	void (*suspend)(struct vperfctr*);
	void (*resume)(struct vperfctr*);
#ifdef CONFIG_VPERFCTR_PROC
	struct file_operations *file_ops;
#endif
} vperfctr_stub;
/* lock taken on module load/unload and ->file_ops access;
   the process scheduling callbacks don't take the lock
   because the module is known to be loaded and in use */
extern rwlock_t vperfctr_stub_lock;
#define _vperfctr_copy(x)	vperfctr_stub.copy((x))
#define _vperfctr_exit(x)	vperfctr_stub.exit((x))
#define _vperfctr_release(x,y)	vperfctr_stub.release((x),(y))
#define _vperfctr_suspend(x)	vperfctr_stub.suspend((x))
#define _vperfctr_resume(x)	vperfctr_stub.resume((x))
#else	/* !CONFIG_PERFCTR_MODULE */
#define _vperfctr_copy(x)	__vperfctr_copy((x))
#define _vperfctr_exit(x)	__vperfctr_exit((x))
#define _vperfctr_release(x,y)	__vperfctr_release((x),(y))
#define _vperfctr_suspend(x)	__vperfctr_suspend((x))
#define _vperfctr_resume(x)	__vperfctr_resume((x))
#endif	/* CONFIG_PERFCTR_MODULE */

static __inline__ void perfctr_copy_thread(struct thread_struct *thread, unsigned long clone_flags)
{
	struct vperfctr *perfctr = thread->perfctr;
	if( perfctr ) {
		if( clone_flags & CLONE_PERFCTR )
			thread->perfctr = _vperfctr_copy(perfctr);
		else
			thread->perfctr = NULL;
	}
}

static __inline__ void perfctr_exit_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr = thread->perfctr;
	if( perfctr )
		_vperfctr_exit(perfctr);
}

static __inline__ void perfctr_release_thread(struct thread_struct *parent_thread,
					      struct thread_struct *child_thread)
{
	struct vperfctr *child = child_thread->perfctr;
	if( child ) {
		child_thread->perfctr = NULL;
		_vperfctr_release(parent_thread->perfctr, child);
	}
}

static __inline__ void perfctr_suspend_thread(struct thread_struct *prev)
{
	struct vperfctr *perfctr = prev->perfctr;
	if( perfctr )
		_vperfctr_suspend(perfctr);
}

/* PRE: next is current */
static __inline__ void perfctr_resume_thread(struct thread_struct *next)
{
	struct vperfctr *perfctr = next->perfctr;
	if( perfctr )
		_vperfctr_resume(perfctr);
}

#define PERFCTR_PROC_PID_MODE	(0 | S_IRUSR)
extern void perfctr_set_proc_pid_ops(struct inode *inode);
/* for 2.2: */
extern struct inode_operations perfctr_proc_pid_inode_operations;

#else	/* !CONFIG_PERFCTR_VIRTUAL */

static __inline__ void perfctr_copy_thread(struct thread_struct *t, unsigned long clone_flags) { }
static __inline__ void perfctr_exit_thread(struct thread_struct *t) { }
static __inline__ void perfctr_release_thread(struct thread_struct *p, struct thread_struct *c) { }
static __inline__ void perfctr_suspend_thread(struct thread_struct *t) { }
static __inline__ void perfctr_resume_thread(struct thread_struct *t) { }

#endif	/* CONFIG_PERFCTR_VIRTUAL */

#endif	/* __KERNEL__ */

#include <linux/ioctl.h>
#define _PERFCTR_IOCTL	0xD0	/* 'P'+128, currently unassigned */

#define PERFCTR_INFO		_IOR(_PERFCTR_IOCTL,0,struct perfctr_info)
#define VPERFCTR_ATTACH		 _IO(_PERFCTR_IOCTL,1)
#define VPERFCTR_SAMPLE		 _IO(_PERFCTR_IOCTL,2)
#define VPERFCTR_UNLINK		 _IO(_PERFCTR_IOCTL,3)
#define VPERFCTR_CONTROL	_IOW(_PERFCTR_IOCTL,4,struct perfctr_control)
#define VPERFCTR_STOP		 _IO(_PERFCTR_IOCTL,5)
#define GPERFCTR_CONTROL	_IOW(_PERFCTR_IOCTL,6,struct gperfctr_control)
#define GPERFCTR_READ		_IOR(_PERFCTR_IOCTL,7,struct gperfctr_state)
#define GPERFCTR_STOP		 _IO(_PERFCTR_IOCTL,8)

#endif	/* _LINUX_PERFCTR_H */
