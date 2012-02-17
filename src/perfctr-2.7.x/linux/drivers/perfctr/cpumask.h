/* $Id: cpumask.h,v 1.11 2004/07/12 22:44:14 mikpe Exp $
 * Performance-monitoring counters driver.
 * Partial simulation of cpumask_t on non-cpumask_t kernels.
 * Extension to allow inspecting a cpumask_t as array of ulong.
 * Appropriate definition of perfctr_cpus_forbidden_mask.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 */

#ifdef CPU_ARRAY_SIZE
#define PERFCTR_CPUMASK_NRLONGS	CPU_ARRAY_SIZE
#else
#define PERFCTR_CPUMASK_NRLONGS	1
#endif

/* CPUs in `perfctr_cpus_forbidden_mask' must not use the
   performance-monitoring counters. TSC use is unrestricted.
   This is needed to prevent resource conflicts on hyper-threaded P4s. */
#ifdef CONFIG_PERFCTR_CPUS_FORBIDDEN_MASK
extern cpumask_t perfctr_cpus_forbidden_mask;
#define perfctr_cpu_is_forbidden(cpu)	cpu_isset((cpu), perfctr_cpus_forbidden_mask)
#else
#define perfctr_cpus_forbidden_mask	CPU_MASK_NONE
#define perfctr_cpu_is_forbidden(cpu)	0 /* cpu_isset() needs an lvalue :-( */
#endif
