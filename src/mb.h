#ifndef _MB_H
#define _MB_H

/* These definitions are not yet in distros, so I have cut and pasted just
   the needed definitions in here */

#ifdef __powerpc__
#define mb()   __asm__ __volatile__ ("sync" : : : "memory")
#define rmb()  __asm__ __volatile__ ("sync" : : : "memory")
#define wmb()  __asm__ __volatile__ ("sync" : : : "memory")

#elif defined(__mips__)
#define mb()   __asm__ __volatile__ ("sync" : : : "memory")
#define rmb()  __asm__ __volatile__ ("sync" : : : "memory")
#define wmb()  __asm__ __volatile__ ("sync" : : : "memory")

#elif defined(__arm__)
#define rmb() __sync_synchronize()
#define mb()  rmb()
#define wmb() rmb()

#elif defined(__ia64__)
#define ia64_mf() __asm__ __volatile__ ("mf" ::: "memory")
#define mb()  ia64_mf()
#define rmb() mb()
#define wmb() mb()

#elif defined(__x86_64__) || defined(__i386__)
#ifdef CONFIG_X86_32
/*
 * Some non-Intel clones support out of order store. wmb() ceases to be a
 * nop for these.
 */
#define mb() alternative("lock; addl $0,0(%%esp)", "mfence", X86_FEATURE_XMM2)
#define rmb() alternative("lock; addl $0,0(%%esp)", "lfence", X86_FEATURE_XMM2)
#define wmb() alternative("lock; addl $0,0(%%esp)", "sfence", X86_FEATURE_XMM)
#else
#define mb()    asm volatile("mfence":::"memory")
#define rmb()   asm volatile("lfence":::"memory")
#define wmb()   asm volatile("sfence" ::: "memory")
#endif

#else
#error Need to define mb and rmb for this architecture!
#error See the kernel source directory: arch/<arch>/include/asm/system.h file
#endif

#endif
