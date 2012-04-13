/* $Id: perfctr.h,v 1.1.2.16 2010/11/07 19:46:06 mikpe Exp $
 * x86/x86_64 Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2010  Mikael Pettersson
 */
#ifndef _ASM_X86_PERFCTR_H
#define _ASM_X86_PERFCTR_H

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
#define PERFCTR_X86_INTEL_P4M2	12	/* model 2 */
#define PERFCTR_X86_AMD_K8	13
#define PERFCTR_X86_INTEL_PENTM	14	/* Pentium M */
#define PERFCTR_X86_AMD_K8C	15	/* Revision C */
#define PERFCTR_X86_INTEL_P4M3	16	/* model 3 and above */
#define PERFCTR_X86_INTEL_CORE	17	/* family 6 model 14 */
#define PERFCTR_X86_INTEL_CORE2	18	/* family 6, models 15, 22, 23, 29 */
#define PERFCTR_X86_AMD_FAM10H	19	/* family 10h, family 11h */
#define PERFCTR_X86_AMD_FAM10	PERFCTR_X86_AMD_FAM10H /* XXX: compat crap, delete soon */
#define PERFCTR_X86_INTEL_ATOM	20	/* family 6 model 28 */
#define PERFCTR_X86_INTEL_NHLM	21	/* Nehalem: family 6 models 26, 30, 46 */
#define PERFCTR_X86_INTEL_COREI7 PERFCTR_X86_INTEL_NHLM /* XXX: compat crap, delete soon */
#define PERFCTR_X86_INTEL_WSTMR	22	/* Westmere: family 6 models 37, 44 */

struct perfctr_sum_ctrs {
	unsigned long long tsc;
	unsigned long long pmc[18];
};

struct perfctr_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[18];
	unsigned int evntsel[18];	/* one per counter, even on P5 */
	union {
		/* Note: envtsel_high[] and p4.escr[] must occupy the same locations */
		struct {
			unsigned int escr[18];
			unsigned int pebs_enable;	/* for replay tagging */
			unsigned int pebs_matrix_vert;	/* for replay tagging */
		} p4;
		unsigned int evntsel_high[18];
		/* Note: nhlm.offcore_rsp[] must not overlap evntsel_high[],
		   instead we make it overlap the p4.pebs_ fields */
		struct {
			unsigned int _padding[18];
			unsigned int offcore_rsp[2];
		} nhlm;
	}; /* XXX: should be a named field 'u', but that breaks source-code compatibility */
	int ireset[18];			/* < 0, for i-mode counters */
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

struct perfctr_cpu_state {
	unsigned int cstatus;
	struct {	/* k1 is opaque in the user ABI */
		unsigned int id;
		int isuspend_cpu;
	} k1;
	/* The two tsc fields must be inlined. Placing them in a
	   sub-struct causes unwanted internal padding on x86-64. */
	unsigned int tsc_start;
	unsigned long long tsc_sum;
	struct {
		unsigned int map;
		unsigned int start;
		unsigned long long sum;
	} pmc[18];	/* the size is not part of the user ABI */
#ifdef __KERNEL__
	struct perfctr_cpu_control control;
	unsigned int core2_fixed_ctr_ctrl;
	unsigned int p4_escr_map[18];
#ifdef CONFIG_PERFCTR_INTERRUPT_SUPPORT
	unsigned int pending_interrupt;
#endif
#endif
};

/* cstatus is a re-encoding of control.tsc_on/nractrs/nrictrs
   which should have less overhead in most cases */

static inline
unsigned int __perfctr_mk_cstatus(unsigned int tsc_on, unsigned int have_ictrs,
				  unsigned int nrictrs, unsigned int nractrs)
{
	return (tsc_on<<31) | (have_ictrs<<16) | ((nractrs+nrictrs)<<8) | nractrs;
}

static inline
unsigned int perfctr_mk_cstatus(unsigned int tsc_on, unsigned int nractrs,
				unsigned int nrictrs)
{
	return __perfctr_mk_cstatus(tsc_on, nrictrs, nrictrs, nractrs);
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

/* version number for user-visible CPU-specific data */
#define PERFCTR_CPU_VERSION	0x0501	/* 5.1 */

#ifdef __KERNEL__

#if defined(CONFIG_PERFCTR) || defined(CONFIG_PERFCTR_MODULE)

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
   cpumask must be NULL for global-mode counters and non-NULL
   for per-thread counters. If cpumask is non-NULL and the control
   data requires the task to be restricted to a specific set of
   CPUs, then *cpumask will be updated accordingly.
   Returns a negative error code if the control data is invalid. */
extern int perfctr_cpu_update_control(struct perfctr_cpu_state *state, cpumask_t *cpumask);

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
#ifdef CONFIG_X86_LOCAL_APIC
extern void __perfctr_cpu_mask_interrupts(void);
extern void __perfctr_cpu_unmask_interrupts(void);
extern void perfctr_cpu_set_ihandler(perfctr_ihandler_t);
extern void perfctr_cpu_ireload(struct perfctr_cpu_state*);
extern unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state*);
static inline int perfctr_cpu_has_pending_interrupt(const struct perfctr_cpu_state *state)
{
	return state->pending_interrupt;
}
#else
static inline void perfctr_cpu_set_ihandler(perfctr_ihandler_t x) { }
static inline int perfctr_cpu_has_pending_interrupt(const struct perfctr_cpu_state *state)
{
	return 0;
}
#endif

#endif	/* CONFIG_PERFCTR */

#if defined(CONFIG_KPERFCTR) && defined(CONFIG_X86_LOCAL_APIC)
asmlinkage void perfctr_interrupt(struct pt_regs*);
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,27)	/* 2.6.27-rc1 */
#define perfctr_vector_init()	\
	alloc_intr_gate(LOCAL_PERFCTR_VECTOR, perfctr_interrupt)
#else
#define perfctr_vector_init()	\
	set_intr_gate(LOCAL_PERFCTR_VECTOR, perfctr_interrupt)
#endif
#else
#define perfctr_vector_init()	do{}while(0)
#endif

#endif	/* __KERNEL__ */

#endif	/* _ASM_X86_PERFCTR_H */
