/**
* @file:   papi_lock.h
* CVS:     $Id$
* @author  Philip Mucci
*          mucci@cs.utk.edu
*/

#ifndef PAPI_LOCK_H
#define PAPI_LOCK_H

#include "papi_defines.h"

#define MUTEX_OPEN 0
#define MUTEX_CLOSED 1

#ifdef _AIX
volatile int lock_var[PAPI_MAX_LOCK] = { 0 };
atomic_p lock[PAPI_MAX_LOCK];
#else
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#endif

inline_static void 
_papi_hwd_lock_init( void )
{
#if defined(__bgp__) 
    /* PAPI on BG/P does not need locks. */ 
    return;
#elif defined(_AIX)
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ )
		lock[i] = ( int * ) ( lock_var + i );
#else
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ )
		_papi_hwd_lock_data[i] = MUTEX_OPEN;
#endif
}

inline_static void
_papi_hwd_lock_fini( void )
{
#if defined(_AIX) || defined(__bgp__)
    return;
#else
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ )
		_papi_hwd_lock_data[i] = MUTEX_OPEN;
#endif
}

#ifdef _AIX

#define _papi_hwd_lock(lck)   { while(_check_lock(lock[lck],0,1) == TRUE) { ; } }
#define _papi_hwd_unlock(lck) { _clear_lock(lock[lck], 0); }

#elif defined(__bgp__)

/* PAPI on BG/P does not need locks. */ 
#define _papi_hwd_lock(lck)   {}
#define _papi_hwd_unlock(lck) {}

#elif defined(__ia64__)

#ifdef __INTEL_COMPILER
#define _papi_hwd_lock(lck) { while(_InterlockedCompareExchange_acq(&_papi_hwd_lock_data[lck],MUTEX_CLOSED,MUTEX_OPEN) != MUTEX_OPEN) { ; } }

#define _papi_hwd_unlock(lck) { _InterlockedExchange((volatile int *)&_papi_hwd_lock_data[lck], MUTEX_OPEN); }
#else  /* GCC */
#define _papi_hwd_lock(lck)			 			      \
   { int res = 0;							      \
    do {								      \
      __asm__ __volatile__ ("mov ar.ccv=%0;;" :: "r"(MUTEX_OPEN));            \
      __asm__ __volatile__ ("cmpxchg4.acq %0=[%1],%2,ar.ccv" : "=r"(res) : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_CLOSED) : "memory");				      \
    } while (res != MUTEX_OPEN); }

#define _papi_hwd_unlock(lck) {  __asm__ __volatile__ ("st4.rel [%0]=%1" : : "r"(&_papi_hwd_lock_data[lck]), "r"(MUTEX_OPEN) : "memory"); }
#endif /* __INTEL_COMPILER */

#elif defined(__i386__)||defined(__x86_64__)
#define  _papi_hwd_lock(lck)                    \
do                                              \
{                                               \
   unsigned int res = 0;                        \
   do {                                         \
      __asm__ __volatile__ ("lock ; " "cmpxchg %1,%2" : "=a"(res) : "q"(MUTEX_CLOSED), "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");  \
   } while(res != (unsigned int)MUTEX_OPEN);   \
} while(0)
#define  _papi_hwd_unlock(lck)                  \
do                                              \
{                                               \
   unsigned int res = 0;                       \
   __asm__ __volatile__ ("xchg %0,%1" : "=r"(res) : "m"(_papi_hwd_lock_data[lck]), "0"(MUTEX_OPEN) : "memory");                                \
} while(0)

#elif defined(__powerpc__)

/*
 * These functions are slight modifications of the functions in
 * /usr/include/asm-ppc/system.h.
 *
 *  We can't use the ones in system.h directly because they are defined
 *  only when __KERNEL__ is defined.
 */

static __inline__ unsigned long
papi_xchg_u32( volatile void *p, unsigned long val )
{
	unsigned long prev;

	__asm__ __volatile__( "\n\
        sync \n\
1:      lwarx   %0,0,%2 \n\
        stwcx.  %3,0,%2 \n\
        bne-    1b \n\
        isync":"=&r"( prev ), "=m"( *( volatile unsigned long * ) p )
						  :"r"( p ), "r"( val ),
						  "m"( *( volatile unsigned long * ) p )
						  :"cc", "memory" );

	return prev;
}

#define  _papi_hwd_lock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  do {                                                  \
  retval = papi_xchg_u32(&_papi_hwd_lock_data[lck],MUTEX_CLOSED);  \
  } while(retval != (unsigned int)MUTEX_OPEN);	        \
} while(0)
#define  _papi_hwd_unlock(lck)                          \
do {                                                    \
  unsigned int retval;                                 \
  retval = papi_xchg_u32(&_papi_hwd_lock_data[lck],MUTEX_OPEN); \
} while(0)

#elif defined(__sparc__)
#include <synch.h>
extern void cpu_sync( void );
extern unsigned long long get_tick( void );
extern caddr_t _start, _end, _etext, _edata;
rwlock_t lock[PAPI_MAX_LOCK];
#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);
#define _papi_hwd_unlock(lck) rw_unlock(&lock[lck]);
#else
#error "_papi_hwd_lock/unlock undefined!"
#endif

#endif
