/* $Id$
 * x86 Performance-Monitoring Counters driver
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#ifndef _ASM_I386_PERFCTR_H
#define _ASM_I386_PERFCTR_H

struct perfctr_sum_ctrs {
	unsigned long long ctr[5];	/* tsc, pmc0, ..., pmc3 */
};

struct perfctr_low_ctrs {
	unsigned int ctr[5];		/* tsc, pmc0, ..., pmc3 */
};

struct perfctr_control {
	unsigned int evntsel[4];
};

#ifdef __KERNEL__

#if defined(CONFIG_PERFCTR) || defined(CONFIG_PERFCTR_MODULE)

/* called from arch/i386/kernel/setup.c:dodgy_tsc() */
extern void perfctr_dodgy_tsc(void);

/* Driver init/exit. */
extern int perfctr_cpu_init(void);
extern void perfctr_cpu_exit(void);

/* CPU type/features. */
extern unsigned char perfctr_cpu_type;
extern unsigned char perfctr_cpu_features;
extern unsigned long perfctr_cpu_khz;

/* CPU type name. */
extern char *perfctr_cpu_name[];

/* Hardware reservation. */
extern const char *perfctr_cpu_reserve(const char *service);
extern void perfctr_cpu_release(const char *service);

/* Compute `nrctrs', the number of enabled counters.
   On x86, we order the counters as TSC, PMC0, PMC1, ..., PMCn,
   and TSC is implicitly enabled. We do not cater for "holes":
   if counter i is enabled, then counters 0..i-1 will also be enabled.
   Returns < 0 if the control data is invalid. */
extern int perfctr_cpu_check_control(const struct perfctr_control *control);

/* Update the CPU's control and event selection data. PRE: nrctrs > 0. */
extern void perfctr_cpu_write_control(int nrctrs, const struct perfctr_control *control);

/* Read and save current counter values. PRE: nrctrs > 0. */
extern void perfctr_cpu_read_counters(int nrctrs, struct perfctr_low_ctrs *ctrs);

/* Process resume/suspend hooks for virtual perfctrs. */
static __inline__ void perfctr_cpu_enable_rdpmc(void) { }
static __inline__ void perfctr_cpu_disable_rdpmc(void) { }

#else	/* !CONFIG_PERFCTR */

static __inline__ void perfctr_dodgy_tsc(void) { }

#endif	/* CONFIG_PERFCTR */

#endif	/* __KERNEL__ */

#endif	/* _ASM_I386_PERFCTR_H */
