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

#if defined(linux)
#include "linux.h"
#endif

#define MAX_COUNTERS 4
#define MAX_COUNTER_TERMS  MAX_COUNTERS

#define inline_static inline static

typedef struct _any_reg_alloc {
  int placeholder;
} _any_reg_alloc_t;

typedef struct _any_register {
  int placeholder;
} _any_register_t;

typedef struct _any_control_state {
  int placeholder;
} _any_control_state_t;

typedef struct _any_context {
#if defined(USE_PROC_PTTIMER)
  int stat_fd;
#endif
  int placeholder; 
} _any_context_t;

typedef struct _any_siginfo {
  int placeholder;
} _any_siginfo_t;

typedef struct _any_ucontext {
  int placeholder; 
} _any_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) (ctx)

#define MY_VECTOR _any_vector

#endif /* _PAPI_ANY_NULL_H */

