/* These definitions are not yet in distros, so I have cut and pasted just
   the needed definitions in here */

#ifdef __powerpc__
#define mb()   __asm__ __volatile__ ("sync" : : : "memory")
#define rmb()  __asm__ __volatile__ ("sync" : : : "memory")
#define wmb()  __asm__ __volatile__ ("sync" : : : "memory")
#else
#error Need to define mb and rmb for this architecture!
#error See the kernel source's arch/<arch>/include/asm/system.h file
#endif
