/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    any-null.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#ifndef _PAPI_ANY_NULL_H
#define _PAPI_ANY_NULL_H

#define _GNU_SOURCE
#define __USE_GNU
#define __USE_UNIX98
#define __USE_XOPEN_EXTENDED

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <inttypes.h>
#include <sys/ucontext.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <linux/unistd.h>
#include "libperfctr.h"

#define PERF_MAX_COUNTERS 4
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define P3_MAX_REGS_PER_EVENT 2

#define gettid() syscall(SYS_gettid)
#define inline_static inline static

#define vperfctr_open() ((struct vperfctr *)gettid())
#define vperfctr_info(a,b) (assert(a==(struct vperfctr *)gettid()),0)
#define vperfctr_control(a,b) (assert(a==(struct vperfctr *)gettid()),0)
#define vperfctr_unlink(a) (assert(a==(struct vperfctr *)gettid()),0)
#define vperfctr_stop(a) (assert(a==(struct vperfctr *)gettid()),0)
#define vperfctr_read_ctrs(a,b) { struct perfctr_sum_ctrs *c = b; assert(a==(struct vperfctr *)gettid()); c->pmc[0] = ++cntr[0]; c->pmc[1] = ++cntr[1]; }
#define vperfctr_read_tsc(a) (assert(a==(struct vperfctr *)gettid()),++virt_tsc)
#define vperfctr_close(a) (assert(a==(struct vperfctr *)gettid()))

#define PERF_MAX_COUNTERS 4
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTER_TERMS  MAX_COUNTERS
#define P3_MAX_REGS_PER_EVENT 2
#define PERFCTR_CPU_NAME(a)   "null"
#define PERFCTR_CPU_NRCTRS(a) 2

/* Used in determining on which counters an event can live. */
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8
#define CNTRS12 (CNTR1|CNTR2)
#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4)

#define HAS_MESI  0x100 /* indicates this event supports MESI modifiers */ 
#define HAS_MOESI 0x200 /* indicates this event supports MOESI modifiers */
#define MOESI_M   0x1000 /* Modified bit */
#define MOESI_O   0x0800 /* Owner bit */
#define MOESI_E   0x0400 /* Exclusive bit */
#define MOESI_S   0x0200 /* Shared bit */
#define MOESI_I   0x0100 /* Invalid bit */
#define MOESI_M_INTEL   MOESI_O /* Modified bit on Intel processors */
#define MOESI_ALL 0x1F00 /* mask for MOESI bits in event code or counter_cmd */

/* Masks to craft an eventcode to perfctr's liking */
#define PERF_ENABLE            0x00400000
#define PERF_INT_ENABLE        0x00100000
#define PERF_OS                0x00020000
#define PERF_USR               0x00010000

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a Pentium I,II,III, Athlon or Opteron"

/* Lock macros. */
extern int sem_set;
#define MUTEX_OPEN 1
#define MUTEX_CLOSED 0

/* Overflow macros */
#ifdef __x86_64__
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->rip)
#else
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
#endif
/* If lock == MUTEX_OPEN, lock = MUTEX_CLOSED, val = MUTEX_OPEN
 * else val = MUTEX_CLOSED */

#define  _papi_hwd_lock(lck)                    \
{                                               \
extern void PAPIERROR(char *format, ...); \
struct sembuf sem_lock = { lck, -1, 0 }; \
if (semop(sem_set, &sem_lock, 1) == -1 ) {      \
PAPIERROR("semop errno %d",errno); abort(); } }

#define  _papi_hwd_unlock(lck)                   \
{                                                \
extern void PAPIERROR(char *format, ...);\
struct sembuf sem_unlock = { lck, 1, 0 }; \
if (semop(sem_set, &sem_unlock, 1) == -1 ) {     \
PAPIERROR("semop errno %d",errno); abort(); } }

/* Overflow-related defines and declarations */
typedef struct {
   siginfo_t *si;
   struct sigcontext *ucontext;
} _papi_hwd_context_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

typedef struct P3_register {
   unsigned int selector;       /* Mask for which counters in use */
   int counter_cmd;             /* The event code */
} P3_register_t;

typedef struct P3_reg_alloc {
   P3_register_t ra_bits;       /* Info about this native event mapping */
   unsigned ra_selector;        /* Bit mask showing which counters can carry this metric */
   unsigned ra_rank;            /* How many counters can carry this metric */
} P3_reg_alloc_t;

/* Per eventset data structure for thread level counters */

typedef struct hwd_native {
   /* index in the native table, required */
   int index;
   /* Which counters can be used?  */
   unsigned int selector;
   /* Rank determines how many counters carry each metric */
   unsigned char rank;
   /* which counter this native event stays */
   int position;
   int mod;
   int link;
} hwd_native_t;

typedef struct native_event_entry {
   /* If it exists, then this is the name of this event */
   char name[PAPI_MAX_STR_LEN];
   /* If it exists, then this is the description of this event */
   char *description;
   /* description of the resources required by this native event */
   P3_register_t resources;
} native_event_entry_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef P3_reg_alloc_t hwd_reg_alloc_t;
typedef P3_register_t hwd_register_t;

typedef struct P3_perfctr_control {
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   unsigned char master_selector;
   P3_register_t allocated_registers;
   struct vperfctr_control control;
   struct perfctr_sum_ctrs state;
} P3_perfctr_control_t;

typedef struct P3_perfctr_context {
   struct vperfctr *perfctr;
/*  P3_perfctr_control_t start; */
} P3_perfctr_context_t;

/* typedefs to conform to hardware independent PAPI code. */
typedef P3_perfctr_control_t hwd_control_state_t;
typedef P3_perfctr_context_t hwd_context_t;

#endif /* _PAPI_ANY_NULL_H */
