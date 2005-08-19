/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-acpi.h
* CVS:     $Id$
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu 
*          <your name here>
*          <your email address>
*/

#ifndef _PAPI_ACPI_H
#define _PAPI_ACPI_H

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

#define MAX_COUNTERS 8
#define MAX_COUNTER_TERMS  MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

typedef struct ACPI_register {
   /* indicate which counters this event can live on */
   unsigned int selector;
   /* Buffers containing counter cmds for each possible metric */
   char *counter_cmd;
} ACPI_register_t;

typedef ACPI_register_t hwd_register_t;

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
#endif /* _PAPI_ACPI_H */
