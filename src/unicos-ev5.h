/* $Id$ */

#include <mpp/globals.h>
#include <stdio.h>
#include <unistd.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

/* This code comes from Cray, originally in perfctr.h
   I assume they own the copyright, so be careful */

#define PERFCNT_ON      1
#define PERFCNT_OFF     2
#define PERFCNT_EV5     1

typedef unsigned int    uint;

typedef struct {
        uint    CTR0:16;        /* counter 0 */
        uint    CTR1:16;        /* counter 1 */
        uint    SEL0:1;         /* select 0 */
        uint    Ku:1;           /* Kill user counts */
        uint    CTR2:14;        /* counter 2 */
        uint    CTL0:2;         /* control 0 */
        uint    CTL1:2;         /* control 1 */
        uint    CTL2:2;         /* control 2 */
        uint    Kp:1;           /* Kill PAL counts */
        uint    Kk:1;           /* Kill kernel counts */
        uint    SEL1:4;         /* select 1 */
        uint    SEL2:4;         /* select 2 */
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
   /*               if sel2=0x3 or all flow change instructions */
   /*               if sel2!=2 or 3 */

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
  int mask;             /* Counter select mask */
  pmctr_t pmctr;        /* Counter control register to activate this EventSet */ 
} hwd_control_state_t;

/* Preset structure */

typedef struct hwd_preset {
  int sel0;             /* Counter control register */ 
  int sel1;             /* Counter control register */
  int sel2;             /* Counter control register */
  int mask;             /* Counter select mask */ } hwd_preset_t;
