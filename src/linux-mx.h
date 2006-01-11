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

#ifndef _PAPI_MX_H
#define _PAPI_MX_H

#include "papi_sys_headers.h"

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
  #define __USE_GNU
  #define __USE_UNIX98
  #define __USE_XOPEN_EXTENDED
#endif

#ifndef __BSD__ /* #include <malloc.h> */
#include <malloc.h>
#endif

#ifdef XML
#include <expat.h>
#endif

#ifdef _WIN32
#define NEED_FFSLL
#define inline_static static __inline
#include <errno.h>
#include "cpuinfo.h"
#include "pmclib.h"
#else
#define inline_static inline static
#define HAVE_FFSLL
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

//#define MAX_COUNTERS 100
#define MX_MAX_COUNTERS 16
#define MX_MAX_COUNTER_TERMS  8

#include "papi.h"
#include "papi_preset.h"

#define LINELEN 128
/*#define GMPATH "/usr/gm/bin/gm_counters"*/

typedef struct mx_register {
   /* indicate which counters this event can live on */
   unsigned int selector;
   /* Buffers containing counter cmds for each possible metric */
//   char *counter_cmd[PAPI_MAX_STR_LEN];
   char *counter_cmd;
} MX_register_t;

typedef MX_register_t hwd_register_t;

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
  long_long counts[MX_MAX_COUNTERS];
} hwd_control_state_t;

typedef struct hwd_context {
  hwd_control_state_t state; 
} hwd_context_t;

extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;
#endif /* _PAPI_MX_H */
