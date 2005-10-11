/* $Id$
 * Performance-monitoring counters driver.
 * Partial simulation of cpumask_t on non-cpumask_t kernels.
 * Extension to allow inspecting a cpumask_t as array of ulong.
 * Appropriate definition of perfctr_cpus_forbidden_mask.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 */

/* 2.6.0-test4 changed set-of-CPUs values from ulong to cpumask_t */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0)

#if !defined(PERFCTR_HAVE_CPUMASK_T) && !defined(HAVE_CPUMASK_T)
typedef unsigned long cpumask_t;
#endif

/* RH/FC1 kernel 2.4.22-1.2115.nptl added cpumask_t, but with
   an incomplete API and a broken cpus_and() [misspelled parameter
   in its body]. Sigh.
   Assume cpumask_t is unsigned long and use our own code. */
#undef cpu_set
#define cpu_set(cpu, map)	atomic_set_mask((1UL << (cpu)), &(map))
#undef cpu_isset
#define cpu_isset(cpu, map)	((map) & (1UL << (cpu)))
#undef cpus_and
#define cpus_and(dst,src1,src2)	do { (dst) = (src1) & (src2); } while(0)
#undef cpus_clear
#define cpus_clear(map)		do { (map) = 0UL; } while(0)
#undef cpus_complement
#define cpus_complement(map)	do { (map) = ~(map); } while(0)
#undef cpus_empty
#define cpus_empty(map)		((map) == 0UL)
#undef cpus_equal
#define cpus_equal(map1, map2)	((map1) == (map2))
#undef cpus_addr
#define cpus_addr(map)		(&(map))

#undef CPU_MASK_NONE
#define CPU_MASK_NONE		0UL

#elif LINUX_VERSION_CODE < KERNEL_VERSION(2,6,1)

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
