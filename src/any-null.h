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
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

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
  int placeholder; 
} hwd_context_t;

typedef struct hwd_ucontext {
  int placeholder; 
} hwd_ucontext_t;

#define _papi_hwd_lock_init() { ; }
#define _papi_hwd_lock(a) { ; }
#define _papi_hwd_unlock(a) { ; }
#define GET_OVERFLOW_ADDRESS(ctx) (0x80000000)

#endif /* _PAPI_ANY_NULL_H */
