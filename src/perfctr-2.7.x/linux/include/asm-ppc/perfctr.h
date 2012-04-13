/* $Id: perfctr.h,v 1.20 2005/05/26 00:37:36 mikpe Exp $
 * PPC32 Performance-Monitoring Counters driver
 *
 * Copyright (C) 2004-2005  Mikael Pettersson
 */
#ifndef _ASM_PPC_PERFCTR_H
#define _ASM_PPC_PERFCTR_H

#include <asm/types.h>

struct perfctr_sum_ctrs {
	__u64 tsc;
	__u64 pmc[8];	/* the size is not part of the user ABI */
};

struct perfctr_cpu_control_header {
	__u32 tsc_on;
	__u32 nractrs;	/* number of accumulation-mode counters */
	__u32 nrictrs;	/* number of interrupt-mode counters */
};

struct perfctr_cpu_state_user {
	__u32 cstatus;
	/* This is a sequence counter to ensure atomic reads by
	 * userspace.  The mechanism is identical to that used for
	 * seqcount_t in include/linux/seqlock.h. */
	__u32 sequence;
	__u64 tsc_start;
	__u64 tsc_sum;
	struct {
		__u64 start;
		__u64 sum;
	} pmc[8];	/* the size is not part of the user ABI */
};

/* cstatus is a re-encoding of control.tsc_on/nractrs/nrictrs
   which should have less overhead in most cases */
/* XXX: ppc driver internally also uses cstatus&(1<<30) */

static inline
unsigned int perfctr_mk_cstatus(unsigned int tsc_on, unsigned int nractrs,
				unsigned int nrictrs)
{
	return (tsc_on<<31) | (nrictrs<<16) | ((nractrs+nrictrs)<<8) | nractrs;
}

static inline unsigned int perfctr_cstatus_enabled(unsigned int cstatus)
{
	return cstatus;
}

static inline int perfctr_cstatus_has_tsc(unsigned int cstatus)
{
	return (int)cstatus < 0;	/* test and jump on sign */
}

static inline unsigned int perfctr_cstatus_nractrs(unsigned int cstatus)
{
	return cstatus & 0x7F;		/* and with imm8 */
}

static inline unsigned int perfctr_cstatus_nrctrs(unsigned int cstatus)
{
	return (cstatus >> 8) & 0x7F;
}

static inline unsigned int perfctr_cstatus_has_ictrs(unsigned int cstatus)
{
	return cstatus & (0x7F << 16);
}

/*
 * 'struct siginfo' support for perfctr overflow signals.
 * In unbuffered mode, si_code is set to SI_PMC_OVF and a bitmask
 * describing which perfctrs overflowed is put in si_pmc_ovf_mask.
 * A bitmask is used since more than one perfctr can have overflowed
 * by the time the interrupt handler runs.
 *
 * glibc's <signal.h> doesn't seem to define __SI_FAULT or __SI_CODE(),
 * and including <asm/siginfo.h> as well may cause redefinition errors,
 * so the user and kernel values are different #defines here.
 */
#ifdef __KERNEL__
#define SI_PMC_OVF	(__SI_FAULT|'P')
#else
#define SI_PMC_OVF	('P')
#endif
#define si_pmc_ovf_mask	_sifields._pad[0] /* XXX: use an unsigned field later */

#ifdef __KERNEL__

#if defined(CONFIG_PERFCTR)

struct perfctr_cpu_control {
	struct perfctr_cpu_control_header header;
	unsigned int mmcr0;
	unsigned int mmcr1;
	unsigned int mmcr2;
	/* IABR/DABR/BAMR not supported */
	unsigned int ireset[8];		/* [0,0x7fffffff], for i-mode counters, physical indices */
	unsigned int pmc_map[8];	/* virtual to physical index map */
};

struct perfctr_cpu_state {
	/* Don't change field order here without first considering the number
	   of cache lines touched during sampling and context switching. */
	unsigned int id;
	int isuspend_cpu;
	struct perfctr_cpu_state_user user;
	struct perfctr_cpu_control control;
};

/* Driver init/exit. */
extern int perfctr_cpu_init(void);
extern void perfctr_cpu_exit(void);

/* CPU type name. */
extern char *perfctr_cpu_name;

/* Hardware reservation. */
extern const char *perfctr_cpu_reserve(const char *service);
extern void perfctr_cpu_release(const char *service);

/* PRE: state has no running interrupt-mode counters.
   Check that the new control data is valid.
   Update the driver's private control data.
   Returns a negative error code if the control data is invalid. */
extern int perfctr_cpu_update_control(struct perfctr_cpu_state *state, int is_global);

/* Parse and update control for the given domain. */
extern int perfctr_cpu_control_write(struct perfctr_cpu_control *control,
				     unsigned int domain,
				     const void *srcp, unsigned int srcbytes);

/* Retrieve and format control for the given domain.
   Returns number of bytes written. */
extern int perfctr_cpu_control_read(const struct perfctr_cpu_control *control,
				    unsigned int domain,
				    void *dstp, unsigned int dstbytes);

/* Read a-mode counters. Subtract from start and accumulate into sums.
   Must be called with preemption disabled. */
extern void perfctr_cpu_suspend(struct perfctr_cpu_state *state);

/* Write control registers. Read a-mode counters into start.
   Must be called with preemption disabled. */
extern void perfctr_cpu_resume(struct perfctr_cpu_state *state);

/* Perform an efficient combined suspend/resume operation.
   Must be called with preemption disabled. */
extern void perfctr_cpu_sample(struct perfctr_cpu_state *state);

/* The type of a perfctr overflow interrupt handler.
   It will be called in IRQ context, with preemption disabled. */
typedef void (*perfctr_ihandler_t)(unsigned long pc);

/* Operations related to overflow interrupt handling. */
#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
extern void perfctr_cpu_set_ihandler(perfctr_ihandler_t);
extern void perfctr_cpu_ireload(struct perfctr_cpu_state*);
extern unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state*);
#else
static inline void perfctr_cpu_set_ihandler(perfctr_ihandler_t x) { }
#endif
static inline int perfctr_cpu_has_pending_interrupt(const struct perfctr_cpu_state *state)
{
	return 0;
}

#endif	/* CONFIG_PERFCTR */

#endif	/* __KERNEL__ */

#endif	/* _ASM_PPC_PERFCTR_H */
