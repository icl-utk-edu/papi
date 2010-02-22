#ifndef _LINUX_BGL_H
#define _LINUX_BGL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/profil.h>
#include <assert.h>
#include <limits.h>
#include <signal.h>
#include <errno.h>
#include <sys/ucontext.h>
#include <bgl_perfctr.h>

#include <stdarg.h>
#include <ctype.h>

#define inline_static inline static
#define SIGNAL45 45			 /* SIGBGLUPC */
#define BGL_PAPI_TIMEBASE 4711	/* Special event for getting the timebase reg */

//#define PERF_MAX_COUNTERS 48 
//#define BGL_PERFCTR_NUM_COUNTERS PERF_MAX_COUNTERS
#define MAX_COUNTERS BGL_PERFCTR_NUM_COUNTERS+1
#define MAX_COUNTER_TERMS  MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"

#define GET_OVERFLOW_ADDRESS(ctx) 0x0
//#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)
//#define GET_OVERFLOW_ADDRESS(ctx) ((caddr_t)(ctx->ucontext->Eip))
//#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(((struct sigcontext *)(&ctx->ucontext->uc_mcontext))->eip)

typedef struct bgl_control_state
{
	bgl_perfctr_control_t perfctr;
	unsigned long long cycles;
} bgl_control_state_t;

typedef BGL_PERFCTR_event_encoding_t hwd_register_t;

typedef struct bgl_reg_alloc
{
	hwd_register_t ra_bits;
	u_int64_t ra_selector;			   /*mapping */
	int ra_rank;					   /*num_encoding */
} bgl_reg_alloc_t;

typedef bgl_control_state_t hwd_control_state_t;

typedef struct bgl_context
{
	bgl_perfctr_control_t *perfstate;
	unsigned long long cycles;
//   bgl_control_state_t ctr_state;
} bgl_context_t;

typedef bgl_context_t hwd_context_t;


/* useless, but defined... */
typedef bgl_reg_alloc_t hwd_reg_alloc_t;
typedef BGL_PERFCTR_event_descr_t native_event_entry_t;

extern void _papi_hwd_lock( int );
extern void _papi_hwd_unlock( int );

typedef int hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;

extern void get_bgl_native_event( int Event, BGL_PERFCTR_event_t * event );

#endif
