/* $Id$
 * x86 Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */
#ifndef _ASM_I386_PERFCTR_H
#define _ASM_I386_PERFCTR_H

struct perfctr_sum_ctrs {
	unsigned long long tsc;
	unsigned long long pmc[18];
};

struct perfctr_low_ctrs {
	unsigned int tsc;
	unsigned int pmc[18];
};

struct perfctr_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[18];
	unsigned int evntsel[18];	/* one per counter, even on P5 */
	unsigned int evntsel_aux[18];	/* e.g. P4 ESCR contents */
	struct {
		unsigned int pebs_enable;	/* for replay tagging */
		unsigned int pebs_matrix_vert;	/* for replay tagging */
	} p4;
	int ireset[18];			/* <= 0, for i-mode counters */
};

struct perfctr_cpu_state {
	unsigned int cstatus;
	union {
		unsigned int p5_cesr;
		unsigned int id;	/* cache owner id */
	} k1;
	struct perfctr_sum_ctrs sum;
	struct perfctr_low_ctrs start;
	struct perfctr_cpu_control control;
	struct {
		unsigned int p4_escr_map[18];
		const void *isuspend_cpu;
	} k2;
};

/* `struct perfctr_cpu_state' binary layout version number */
#define PERFCTR_CPU_STATE_MAGIC	0x0201	/* 2.1 */

/* cstatus is a re-encoding of control.tsc_on/nractrs/nrictrs
   which should have less overhead in most cases */

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

#if defined(CONFIG_PERFCTR) || defined(CONFIG_PERFCTR_MODULE)

/* Driver init/exit. */
extern int perfctr_cpu_init(void);
extern void perfctr_cpu_exit(void);

/* CPU type name. */
extern char *perfctr_cpu_name[];

/* Hardware reservation. */
extern const char *perfctr_cpu_reserve(const char *service);
extern void perfctr_cpu_release(const char *service);

/* Check that the new control data is valid.
   Update the driver's private control data.
   Returns a negative error code if the control data is invalid. */
extern int perfctr_cpu_update_control(struct perfctr_cpu_state *state,
				      const struct perfctr_cpu_control *control);

/* Read a-mode counters. Subtract from start and accumulate into sums. */
extern void perfctr_cpu_suspend(struct perfctr_cpu_state *state);

/* Write control registers. Read a-mode counters into start. */
extern void perfctr_cpu_resume(struct perfctr_cpu_state *state);

/* Perform an efficient combined suspend/resume operation. */
extern void perfctr_cpu_sample(struct perfctr_cpu_state *state);

typedef void (*perfctr_ihandler_t)(unsigned long pc);

#ifdef CONFIG_X86_LOCAL_APIC
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0)
#include <asm/fixmap.h>
#include <asm/apic.h>
struct hw_interrupt_type;
#include <asm/hw_irq.h>
#ifdef LOCAL_PERFCTR_VECTOR
#define PERFCTR_INTERRUPT_SUPPORT 1
#endif
#endif
#endif

#if PERFCTR_INTERRUPT_SUPPORT
extern unsigned int apic_lvtpc_irqs[];
extern void perfctr_interrupt(void);
extern void perfctr_cpu_set_ihandler(perfctr_ihandler_t);
extern void perfctr_cpu_ireload(const struct perfctr_cpu_state*);
extern unsigned int perfctr_cpu_identify_overflow(struct perfctr_cpu_state*);
#else
static inline void perfctr_cpu_set_ihandler(perfctr_ihandler_t x) { }
#endif

#endif	/* CONFIG_PERFCTR */

#endif	/* __KERNEL__ */

#endif	/* _ASM_I386_PERFCTR_H */
