#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <infoblk.h>
#include <mpp/globals.h> 
#include <sys/ucontext.h>
#include <sys/times.h>
#include <limits.h>

#include "papi.h"

#define PERFCNT_ON      1
#define PERFCNT_OFF     2
#define PERFCNT_EV5     1
#define CTL_OFF		0
#define CTL_ON		2
#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4

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
/*sel0 = 0x0;*/    /* count machine cycles */
/*sel0 = 0x1;*/    /* count instructions */
/********   "sel1" list   ******************************/
/*sel1 = 0x0;*/    /* count non-issue cycles */
/*sel1 = 0x1;*/    /* count split-issue cycles */
/*sel1 = 0x2;*/    /* count issue-pipe-dry cycles */
/*sel1 = 0x3;*/    /* count replay trap cycles */
/*sel1 = 0x4;*/    /* count single-issue cycles */
/*sel1 = 0x5;*/    /* count dual-issue cycles */
/*sel1 = 0x6;*/    /* count triple-issue cycles */
/*sel1 = 0x7;*/    /* count quad-issue cycles */
/*sel1 = 0x8;*/    /* count jsr-ret, cond-branch, flow-change instrucs */
/*sel1 = 0x9;*/    /* count int instructions issued */
/*sel1 = 0xA;*/    /* count fp instructions issued */
/*sel1 = 0xB;*/    /* count loads issued */
/*sel1 = 0xC;*/    /* count stores issued */
/*sel1 = 0xD;*/    /* count Icache issued */
/*sel1 = 0xE;*/    /* count dcache accesses */
/*sel1 = 0xF;*/    /* count Scache accesses, CBOX input 1 */
/* for sel1=0x8: count jsr-ret if sel2=0x2 or cond-branch */
/* if sel2=0x3 or all flow change instructions */
/* if sel2!=2 or 3 */
/********   "sel2" list   ******************************/
/*sel2 = 0x0;*/    /* count >15 cycle stalls */
/*sel2 = 0x2;*/    /* count PC-mispredicts */
/*sel2 = 0x3;*/    /* count BR-mispredicts */
/*sel2 = 0x4;*/    /* count icache/RFB misses */
/*sel2 = 0x5;*/    /* count ITB misses */
/*sel2 = 0x6;*/    /* count dcache misses */
/*sel2 = 0x7;*/    /* count DTB misses */
/*sel2 = 0x8;*/    /* count LDs merged in MAF */
/*sel2 = 0x9;*/    /* count LDU replay traps */
/*sel2 = 0xA;*/    /* count WB/MAF full replay traps */
/*sel2 = 0xB;*/    /* count perf_mon_h input at sysclock intervals */
/*sel2 = 0xC;*/    /* count CPU cycles */
/*sel2 = 0xD;*/    /* count MB stall cycles */
/*sel2 = 0xE;*/    /* count LDxL instructions issued */
/*sel2 = 0xF;*/    /* count Scache misses, CBOX input 2 */
/*******************************************************/

/* Begin my code */

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the PAL code to control the counters */
  pmctr_t counter_cmd;
  /* Milliseconds between timer interrupts for various things */  
  int timer_ms;  
  /* The native encoding of the domain of this eventset */
  int domain;    
} hwd_control_state_t;

/* Preset structure */

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  unsigned char counter_cmd[3];
  /* Footnote to append to the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

#include "papi_internal.h"
