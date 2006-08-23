/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-mx.h
* CVS:     $Id$
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_GM_H
#define _PAPI_GM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dirent.h>

#define _GNU_SOURCE
#define __USE_GNU
#define __USE_UNIX98
#define __USE_XOPEN_EXTENDED

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>

#ifndef __BSD__ /* #include <malloc.h> */
#include <malloc.h>
#endif

#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#ifdef XML
#include <expat.h>
#endif

#ifdef _WIN32
#undef HAVE_FFSLL
#define inline_static static __inline
#include <errno.h>
#include "cpuinfo.h"
#include "pmclib.h"
#else
#define inline_static inline static
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>

#ifdef __BSD__
#include <ucontext.h>
#else
#include <sys/ucontext.h>
#endif

#include <sys/times.h>
#include <sys/time.h>

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
#include <inttypes.h>
/*#include "libperfctr.h"*/
#endif

#define MAX_COUNTERS 100
#define MAX_COUNTER_TERMS  MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

/*#define inline_static inline static*/
#ifdef _WIN32

/* Lock macros. */
extern CRITICAL_SECTION lock[PAPI_MAX_LOCK];

#define  _papi_hwd_lock(lck) EnterCriticalSection(&lock[lck])
#define  _papi_hwd_unlock(lck) LeaveCriticalSection(&lock[lck])

/*typedef siginfo_t hwd_siginfo_t;*/
typedef int hwd_siginfo_t;
/*typedef ucontext_t hwd_ucontext_t;*/
typedef CONTEXT hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) ((caddr_t)(ctx->ucontext->Eip))

/* Windows DOES NOT support hardware overflow */
#define HW_OVERFLOW 0

#else

/* Lock macros. */
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

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

/* Overflow macros */
#ifdef __x86_64__
  #ifdef __CATAMOUNT__
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext))->sc_rip)
  #else
    #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
  #endif
#else
  #define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif

/* Linux DOES support hardware overflow */
#define HW_OVERFLOW 1

#endif /* _WIN32 */

#define LINELEN 128
/*#define GMPATH "/usr/gm/bin/gm_counters"*/

typedef struct gm_register {
   /* indicate which counters this event can live on */
   unsigned int selector;
   /* Buffers containing counter cmds for each possible metric */
   char *counter_cmd[PAPI_MAX_STR_LEN];
} GM_register_t;

typedef GM_register_t hwd_register_t;

typedef struct native_event_entry {
   /* description of the resources required by this native event */
   hwd_register_t resources;
   /* If it exists, then this is the name of this event */
   char *name;
   /* If it exists, then this is the description of this event */
   char *description;
} native_event_entry_t;

typedef struct hwd_reg_alloc {
  hwd_register_t ra_bits;
} hwd_reg_alloc_t;

typedef struct hwd_control_state {
  long_long counts[MAX_COUNTERS];
} hwd_control_state_t;

typedef struct hwd_context {
  hwd_control_state_t state; 
} hwd_context_t;

/*
#define _papi_hwd_lock_init() { ; }
#define _papi_hwd_lock(a) { ; }
#define _papi_hwd_unlock(a) { ; }
#define GET_OVERFLOW_ADDRESS(ctx) (0x80000000)
*/
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;
#endif /* _PAPI_MX_H */
