#ifndef _DADD_ALPHA_H
#define _DADD_ALPHA_H

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/timers.h>
#include <stropts.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/processor.h>
#include <sys/times.h>
#include <sys/sysinfo.h>
#include <sys/procfs.h>
#include <sys/clu.h>
#include <machine/hal_sysinfo.h>
#include <machine/cpuconf.h>
#include <assert.h>
#include <sys/ucontext.h>
/* Below can be removed when we stop using rusuage for PAPI_get_virt_usec -KSL*/
#include <sys/resource.h>

#include "dadd.h"
#include "virtual_counters.h"

#define inline_static static

#define VC_TOTAL_CYCLES 0
#define VC_BCACHE_MISSES 1
#define VC_TOTAL_DTBMISS 2
#define VC_NYP_EVENTS 3
#define VC_TAKEN_EVENTS 4
#define VC_MISPREDICT_EVENTS 5
#define VC_LD_ST_ORDER_TRAPS 6
#define VC_TOTAL_INSTR_ISSUED 7
#define VC_TOTAL_INSTR_EXECUTED 8
#define VC_INT_INSTR_EXECUTED 9
#define VC_LOAD_INSTR_EXECUTED 10
#define VC_STORE_INSTR_EXECUTED 11
#define VC_TOTAL_LOAD_STORE_EXECUTED 12
#define VC_SYNCH_INSTR_EXECUTED 13
#define VC_NOP_INSTR_EXECUTED 14
#define VC_PREFETCH_INSTR_EXECUTED 15
#define VC_FA_INSTR_EXECUTED 16
#define VC_FM_INSTR_EXECUTED 17
#define VC_FD_INSTR_EXECUTED 18
#define VC_FSQ_INSTR_EXECUTED 19
#define VC_FP_INSTR_EXECUTED 20
#define VC_UNCOND_BR_EXECUTED 21
#define VC_COND_BR_EXECUTED 22
#define VC_COND_BR_TAKEN 23
#define VC_COND_BR_NOT_TAKEN 24
#define VC_COND_BR_MISPREDICTED 25
#define VC_COND_BR_PREDICTED 26
#define VC_ITBMISS_TRAPS 38


#define MAX_COUNTERS	       47
#define MAX_COUNTER_TERMS	2
#define MAX_NATIVE_EVENT 28
#define PAPI_MAX_NATIVE_EVENTS MAX_NATIVE_EVENT


typedef struct dadd_alpha_control_state {
   virtual_counters *ptr_vc;
   /* counter numbers when started */
   long_long start_value[MAX_COUNTERS];
   /* latest counter numbers */
   long_long latest[MAX_COUNTERS];
   /* Interrupt interval */
   int timer_ms;
   /* latest value for cycles */
   long_long latestcycles;
} dadd_alpha_control_state_t;

typedef struct dadd_alpha_context {
   virtual_counters *ptr_vc;
} dadd_alpha_context_t;

typedef dadd_alpha_control_state_t hwd_control_state_t;

typedef int hwd_register_t;     /* don't need this on dadd-alpha */

typedef dadd_alpha_context_t hwd_context_t;

typedef int hwd_reg_alloc_t;    /* don't need this structure on dadd-alpha */
typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void*)(ctx->ucontext->sc_pc)

typedef struct native_info_t {
   char name[40];               /* native name */
   int encode;
} native_info_t;


extern unsigned long _etext, _ftext, _fdata, _edata;

#endif
