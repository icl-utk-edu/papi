/* $Id$
 * Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */
#ifndef _LINUX_PERFCTR_H
#define _LINUX_PERFCTR_H

#include <asm/perfctr.h>

struct perfctr_info {
	char version[32];
	unsigned char nrcpus;
	unsigned char cpu_type;
	unsigned char cpu_features;
	unsigned long cpu_khz;
};

/* cpu_type values */
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
#define PERFCTR_X86_VIA_C3	10	/* no pmc0 */
#define PERFCTR_X86_INTEL_P4	11	/* model 0 and 1 */
#define PERFCTR_X86_INTEL_P4M2	12	/* model 2 and above */

/* cpu_features flag bits */
#define PERFCTR_FEATURE_RDPMC	0x01
#define PERFCTR_FEATURE_RDTSC	0x02
#define PERFCTR_FEATURE_PCINT	0x04

/* user's view of mmap:ed virtual perfctr */
struct vperfctr_state {
	unsigned int magic;
	int si_signo;
	struct perfctr_cpu_state cpu_state;
};

/* `struct vperfctr_state' binary layout version number */
#define VPERFCTR_STATE_MAGIC	0x0201	/* 2.1 */
#define VPERFCTR_MAGIC	((VPERFCTR_STATE_MAGIC<<16)|PERFCTR_CPU_STATE_MAGIC)

/* parameter in VPERFCTR_CONTROL command */
struct vperfctr_control {
	int si_signo;
	struct perfctr_cpu_control cpu_control;
};

/* parameter in GPERFCTR_CONTROL command */
struct gperfctr_control {
	unsigned long interval_usec;
	unsigned int nrcpus;
	struct perfctr_cpu_control cpu_control[1]; /* actually 'nrcpus' */
};

/* returned by GPERFCTR_READ command */
struct gperfctr_cpu_state {
	struct perfctr_cpu_control cpu_control;
	struct perfctr_sum_ctrs sum;
};
struct gperfctr_state {
	unsigned nrcpus;
	struct gperfctr_cpu_state cpu_state[1]; /* actually 'nrcpus' */
};

#include <linux/ioctl.h>
#define _PERFCTR_IOCTL	0xD0	/* 'P'+128, currently unassigned */

#define PERFCTR_INFO		_IOR(_PERFCTR_IOCTL,0,struct perfctr_info)

#define VPERFCTR_SAMPLE		 _IO(_PERFCTR_IOCTL,1)
#define VPERFCTR_UNLINK		 _IO(_PERFCTR_IOCTL,2)
#define VPERFCTR_CONTROL	_IOW(_PERFCTR_IOCTL,3,struct vperfctr_control)
#define VPERFCTR_STOP		 _IO(_PERFCTR_IOCTL,4)
#define VPERFCTR_IRESUME	 _IO(_PERFCTR_IOCTL,5)

#define GPERFCTR_CONTROL	_IOW(_PERFCTR_IOCTL,16,struct gperfctr_control)
#define GPERFCTR_READ		_IOR(_PERFCTR_IOCTL,17,struct gperfctr_state)
#define GPERFCTR_STOP		 _IO(_PERFCTR_IOCTL,18)

#ifdef __KERNEL__

extern struct perfctr_info perfctr_info;
extern int sys_perfctr_info(struct perfctr_info*);

/*
 * Virtual per-process performance-monitoring counters.
 */
struct vperfctr;	/* opaque */

#ifdef CONFIG_PERFCTR_VIRTUAL

/* process management operations */
extern struct vperfctr *__vperfctr_copy(struct vperfctr*);
extern void __vperfctr_exit(struct vperfctr*);
extern void __vperfctr_suspend(struct vperfctr*);
extern void __vperfctr_resume(struct vperfctr*);
extern void __vperfctr_sample(struct vperfctr*);

#ifdef CONFIG_PERFCTR_MODULE
extern struct vperfctr_stub {
	void (*exit)(struct vperfctr*);
	void (*suspend)(struct vperfctr*);
	void (*resume)(struct vperfctr*);
#ifdef CONFIG_SMP
	void (*sample)(struct vperfctr*);
#endif
	struct file_operations *file_ops;
} vperfctr_stub;
/* lock taken on module load/unload and ->file_ops access;
   the process management operations don't take the lock
   because the module is known to be loaded and in use */
extern rwlock_t vperfctr_stub_lock;
#define _vperfctr_exit(x)	vperfctr_stub.exit((x))
#define _vperfctr_suspend(x)	vperfctr_stub.suspend((x))
#define _vperfctr_resume(x)	vperfctr_stub.resume((x))
#define _vperfctr_sample(x)	vperfctr_stub.sample((x))
#else	/* !CONFIG_PERFCTR_MODULE */
#define _vperfctr_exit(x)	__vperfctr_exit((x))
#define _vperfctr_suspend(x)	__vperfctr_suspend((x))
#define _vperfctr_resume(x)	__vperfctr_resume((x))
#define _vperfctr_sample(x)	__vperfctr_sample((x))
#endif	/* CONFIG_PERFCTR_MODULE */

static inline void perfctr_copy_thread(struct thread_struct *thread)
{
	thread->perfctr = NULL;
}

static inline void perfctr_exit_thread(struct thread_struct *thread)
{
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if( perfctr ) {
		thread->perfctr = NULL;
		_vperfctr_exit(perfctr);
	}
}

static inline void perfctr_suspend_thread(struct thread_struct *prev)
{
	struct vperfctr *perfctr;
	perfctr = prev->perfctr;
	if( perfctr )
		_vperfctr_suspend(perfctr);
}

/* PRE: next is current */
static inline void perfctr_resume_thread(struct thread_struct *next)
{
	struct vperfctr *perfctr;
	perfctr = next->perfctr;
	if( perfctr )
		_vperfctr_resume(perfctr);
}

static inline void perfctr_sample_thread(struct thread_struct *thread)
{
#ifdef CONFIG_SMP
	struct vperfctr *perfctr;
	perfctr = thread->perfctr;
	if( perfctr )
		_vperfctr_sample(perfctr);
#endif
}

#define PERFCTR_PROC_PID_MODE	(0 | S_IRUSR)
extern void perfctr_set_proc_pid_ops(struct inode *inode);
/* for 2.2: */
extern struct inode_operations perfctr_proc_pid_inode_operations;

#else	/* !CONFIG_PERFCTR_VIRTUAL */

static inline void perfctr_copy_thread(struct thread_struct *t) { }
static inline void perfctr_exit_thread(struct thread_struct *t) { }
static inline void perfctr_suspend_thread(struct thread_struct *t) { }
static inline void perfctr_resume_thread(struct thread_struct *t) { }
static inline void perfctr_sample_thread(struct thread_struct *t) { }

#endif	/* CONFIG_PERFCTR_VIRTUAL */

#endif	/* __KERNEL__ */

#endif	/* _LINUX_PERFCTR_H */
