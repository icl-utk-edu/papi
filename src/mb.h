/* These definitions are not yet in distros, so I have cut and pasted just
   the needed definitions in here */

#ifdef __powerpc__
#define mb()   __asm__ __volatile__ ("sync" : : : "memory")
#define rmb()  __asm__ __volatile__ ("sync" : : : "memory")
#define wmb()  __asm__ __volatile__ ("sync" : : : "memory")

#elif defined __arm__

/* FIXME!  not all arm have barriers! */
#warning "WARNING! ARM memory barriers currently not working!"

#define dsb() __asm__ __volatile__ ("dsb" : : : "memory")
#define dmb() __asm__ __volatile__ ("dmb" : : : "memory")

/* Use __kuser_memory_barrier helper from the CPU helper page. See
 * arch/arm/kernel/entry-armv.S in the kernel source for details.  */
#define rmb()    ((void(*)(void))0xffff0fa0)()
#define mb()     asm volatile("":::"memory")

#define rmb()           {}
#define wmb()           {}


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
