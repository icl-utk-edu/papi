/* $Id: cpumask.h,v 1.6.2.3 2009/01/23 17:01:02 mikpe Exp $
 * Performance-monitoring counters driver.
 * Partial simulation of cpumask_t on non-cpumask_t kernels.
 * Extension to allow inspecting a cpumask_t as array of ulong.
 * Appropriate definition of perfctr_cpus_forbidden_mask.
 *
 * Copyright (C) 2003-2004, 2009  Mikael Pettersson
 */

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,1)

/* 2.6.1-rc1 introduced cpus_addr() */
#ifdef CPU_ARRAY_SIZE
#define cpus_addr(map)		((map).mask)
#else
#define cpus_addr(map)		(&(map))
#endif

#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8) && !defined(cpus_andnot)
#define cpus_andnot(dst, src1, src2) \
do { \
    cpumask_t _tmp2; \
    _tmp2 = (src2); \
    cpus_complement(_tmp2); \
    cpus_and((dst), (src1), _tmp2); \
} while(0)
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,8) && !defined(CONFIG_SMP)
#undef cpu_online_map
#define cpu_online_map	cpumask_of_cpu(0)
#endif

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
