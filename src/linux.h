#ifndef _PAPI_LINUX_H
#define _PAPI_LINUX_H 

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif
#ifndef __USE_UNIX98
#define __USE_UNIX98
#endif
#ifndef __USE_XOPEN_EXTENDED
#define __USE_XOPEN_EXTENDED
#endif

/* #include "papi_sys_headers.h" */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>

#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>

#include <sys/times.h>
#include <sys/time.h>

#ifndef __BSD__ /* #include <malloc.h> */
#include <malloc.h>
#endif

#define inline_static inline static

#define HAVE_FFSLL
#include <time.h>

#ifdef __BSD__
#include <ucontext.h>
#else
#include <sys/ucontext.h>
#endif

#ifndef __BSD__ /* #include <linux/unistd.h> */
  #ifndef __CATAMOUNT__
    #include <linux/unistd.h>
  #endif
#endif

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#endif

/* generalized definitions for signal handling */
//typedef siginfo_t hwd_siginfo_t;
//typedef ucontext_t hwd_ucontext_t;

/* Lock macros. Non Itanium*/
#ifdef ITANIUM2
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#ifdef ALTIX
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sn/mmtimer.h>
#endif

#ifdef __INTEL_COMPILER
#include <ia64intrin.h>
#include <ia64regs.h>
#endif

#define MUTEX_OPEN (unsigned int)1
#define MUTEX_CLOSED (unsigned int)0
extern volatile unsigned int lock[PAPI_MAX_LOCK];

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */
#define  _papi_hwd_lock(lck)                    \
{                                               \
struct sembuf sem_lock = { lck, -1, 0 }; \
if (semop(sem_set, &sem_lock, 1) == -1 ) {      \
abort(); } }
// PAPIERROR("semop errno %d",errno); abort(); } }

#define  _papi_hwd_unlock(lck)                   \
{                                                \
struct sembuf sem_unlock = { lck, 1, 0 }; \
if (semop(sem_set, &sem_unlock, 1) == -1 ) {     \
abort(); } }
// PAPIERROR("semop errno %d",errno); abort(); } }

#if 0
#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck) { while(_InterlockedCompareExchange_acq(&lock[lck],MUTEX_CLOSED,MUTEX_OPEN) != MUTEX_OPEN) { ; } }
#define _papi_hwd_unlock(lck) { _InterlockedExchange((volatile int *)&lock[lck], MUTEX_OPEN); }
#else                           /* GCC */
#define _papi_hwd_lock(lck)                                                   \
   { uint64_t res = 0;                                                        \
    do {                                                                      \
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));            \
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&lock[lck]), "r"(MUTEX_CLOSED) : "memory");                                    \
    } while (res != (uint64_t)MUTEX_OPEN); }

#define _papi_hwd_unlock(lck)                                                 \
    { uint64_t res = 0;                                                       \
       __asm__ __volatile__ ("xchg4 %0=[%1],%2" : "=r"(res) : "r"(&lock[lck]), "r"(MUTEX_OPEN) : "memory"); }
#endif
#endif
#else
extern volatile unsigned int lock[PAPI_MAX_LOCK];
#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0

/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

#define  _papi_hwd_lock(lck)                    \
do                                              \
{                                               \
   unsigned int res = 0;                        \
   do {                                         \
      __asm__ __volatile__ ("lock ; " "cmpxchg %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(lock[lck]), "0"(MUTEX_OPEN) : "memory");  \
   } while(res != (unsigned int)MUTEX_OPEN);   \
} while(0)

#define  _papi_hwd_unlock(lck)                  \
do                                              \
{                                               \
   unsigned int res = 0;                       \
   __asm__ __volatile__ ("xchg %0,%1" : "=r"(res) : "m"(lock[lck]), "0"(MUTEX_OPEN) : "memory");                                \
} while(0)

#endif
#endif

