/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    linux-ppc64.h
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _LINUX_PPC64_H               /* _LINUX_PPC64_H */
#define _LINUX_PPC64_H

#include "papi_sys_headers.h"

#include <time.h>
#include <stddef.h>

#define inline_static inline static

#define POWER_MAX_COUNTERS MAX_COUNTERS
#define MAX_COUNTER_TERMS MAX_COUNTERS


/* overflow */
typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)ctx->ucontext->uc_mcontext.regs->nip


#endif                          /* _LINUX_PPC64_H */
