#ifndef _PAPI_UNICOS_H
#define _PAPI_UNICOS_H

#define UMK
#include <stdio.h>
#include <signal.h>
#include <assert.h>
#include <infoblk.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <stdarg.h>
#include <sys/ucontext.h>
#include <sys/times.h>
#include <sys/stat.h>
#include <sys/unistd.h>
#include <mpp/globals.h>

#define MAX_COUNTER_TERMS  3
#define MAX_COUNTERS  3
#define inline_static static

#include "papi.h"
#include "papi_preset.h"

#define PERFCNT_ON      1
#define PERFCNT_OFF     2
#define PERFCNT_EV5     1
#define CTL_OFF         0
#define CTL_ON          2
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define ALLCNTRS (CNTR1|CNTR2|CNTR3)

#if defined (_SC_PAGESIZE)
#    define getpagesize() sysconf(_SC_PAGESIZE)
#else
#    if defined (_SC_PAGE_SIZE)
#      define getpagesize() sysconf(_SC_PAGE_SIZE)
#    endif                      /* _SC_PAGE_SIZE */
#endif                          /* _SC_PAGESIZE */

/* Some of this code comes from Cray, originally in perfctr.h
   I assume they own the copyright, so be careful */

typedef struct {
   unsigned int CTR0:16;        /* counter 0 */
   unsigned int CTR1:16;        /* counter 1 */
   unsigned int SEL0:1;         /* select 0 */
   unsigned int Ku:1;           /* Kill user counts */
   unsigned int CTR2:14;        /* counter 2 */
   unsigned int CTL0:2;         /* control 0 */
   unsigned int CTL1:2;         /* control 1 */
   unsigned int CTL2:2;         /* control 2 */
   unsigned int Kp:1;           /* Kill PAL counts */
   unsigned int Kk:1;           /* Kill kernel counts */
   unsigned int SEL1:4;         /* select 1 */
   unsigned int SEL2:4;         /* select 2 */
} pmctr_t;

/********   "sel0" list   ******************************/
/*sel0 = 0x0;*//* count machine cycles */
/*sel0 = 0x1;*//* count instructions */
/********   "sel1" list   ******************************/
/*sel1 = 0x0;*//* count non-issue cycles */
/*sel1 = 0x1;*//* count split-issue cycles */
/*sel1 = 0x2;*//* count issue-pipe-dry cycles */
/*sel1 = 0x3;*//* count replay trap cycles */
/*sel1 = 0x4;*//* count single-issue cycles */
/*sel1 = 0x5;*//* count dual-issue cycles */
/*sel1 = 0x6;*//* count triple-issue cycles */
/*sel1 = 0x7;*//* count quad-issue cycles */
/*sel1 = 0x8;*//* count jsr-ret, cond-branch, flow-change instrucs */
/*sel1 = 0x9;*//* count int instructions issued */
/*sel1 = 0xA;*//* count fp instructions issued */
/*sel1 = 0xB;*//* count loads issued */
/*sel1 = 0xC;*//* count stores issued */
/*sel1 = 0xD;*//* count Icache issued */
/*sel1 = 0xE;*//* count dcache accesses */
/*sel1 = 0xF;*//* count Scache accesses, CBOX input 1 */
/* for sel1=0x8: count jsr-ret if sel2=0x2 or cond-branch */
/* if sel2=0x3 or all flow change instructions */
/* if sel2!=2 or 3 */
/********   "sel2" list   ******************************/
/*sel2 = 0x0;*//* count >15 cycle stalls */
/*sel2 = 0x2;*//* count PC-mispredicts */
/*sel2 = 0x3;*//* count BR-mispredicts */
/*sel2 = 0x4;*//* count icache/RFB misses */
/*sel2 = 0x5;*//* count ITB misses */
/*sel2 = 0x6;*//* count dcache misses */
/*sel2 = 0x7;*//* count DTB misses */
/*sel2 = 0x8;*//* count LDs merged in MAF */
/*sel2 = 0x9;*//* count LDU replay traps */
/*sel2 = 0xA;*//* count WB/MAF full replay traps */
/*sel2 = 0xB;*//* count perf_mon_h input at sysclock intervals */
/*sel2 = 0xC;*//* count CPU cycles */
/*sel2 = 0xD;*//* count MB stall cycles */
/*sel2 = 0xE;*//* count LDxL instructions issued */
/*sel2 = 0xF;*//* count Scache misses, CBOX input 2 */
/*******************************************************/

/* Begin my code */

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

typedef struct hwd_control_state {
   /* Array of events in the cotrol state */
   hwd_native_t native[MAX_COUNTERS];
   int native_idx;
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   int selector;
   /* Is this event derived? */
   int derived;
   /* Buffer to pass to the PAL code to control the counters */
   pmctr_t counter_cmd;
   /* Milliseconds between timer interrupts for various things */
   int timer_ms;
   /* Counter values */
   long_long values[3];
} hwd_control_state_t;

typedef struct hwd_register { /* The entries != -1 imply which mask to use. */
   int selector[3];           /* Holds codes implying which event to count. */
} hwd_register_t;

typedef struct native_event_entry {
   /* If it exists, then this is the name of this event */
   char name[PAPI_MAX_STR_LEN];
   /* If it exists, then this is the description of this event */
   char *description;
   /* The event's resource listing */
   hwd_register_t resources;
} native_event_entry_t;

typedef struct hwd_context {
   pmctr_t *pmctr;
} hwd_context_t;

typedef struct hwd_reg_alloc {
   hwd_register_t ra_bits;      /* Info about this native event mapping */
   unsigned ra_selector;        /* Bit mask showing which counters can carry this metric */
   unsigned ra_rank;            /* How many counters can carry this metric */
} hwd_reg_alloc_t;

#pragma _CRI soft $MULTION
extern $MULTION(void);

#define _papi_hwd_lock(lck)             \
do {                                    \
 if ($MULTION == 0) _semts(lck);        \
} while(0)

#define _papi_hwd_unlock(lck)           \
do {                                    \
    if ($MULTION == 0) _semclr(lck);    \
} while(0)

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(ctx->ucontext->uc_mcontext.gregs[31])

extern native_event_entry_t *native_table;
extern hwi_search_t *preset_search_map;
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __bss_start;

#endif
