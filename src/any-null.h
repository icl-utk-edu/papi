/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    any-null.h
* CVS:     $Id$
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_ANY_NULL_H
#define _PAPI_ANY_NULL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>
#include <time.h>
#if defined(linux)
#include <signal.h>
#include <syscall.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/times.h>
#endif

#define MAX_COUNTERS 2
#define MAX_COUNTER_TERMS  MAX_COUNTERS

#define inline_static inline static

typedef struct hwd_siginfo {
  int placeholder;
} hwd_siginfo_t;

typedef struct hwd_register {
  int placeholder;
} hwd_register_t;

typedef struct hwd_reg_alloc {
  int placeholder;
} hwd_reg_alloc_t;

typedef struct hwd_control_state {
  int placeholder;
} hwd_control_state_t;

typedef struct hwd_context {
#if defined(USE_PROC_PTTIMER)
  int stat_fd;
#endif
  int placeholder; 
} hwd_context_t;

typedef struct hwd_ucontext {
  int placeholder; 
} hwd_ucontext_t;

#define _papi_hwd_lock_init() { ; }
#define _papi_hwd_lock(a) { ; }
#define _papi_hwd_unlock(a) { ; }
#define GET_OVERFLOW_ADDRESS(ctx) (ctx)

#endif /* _PAPI_ANY_NULL_H */



