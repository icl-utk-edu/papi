#ifndef _PAPI_UNICOS_H
#define _PAPI_UNICOS_H

#define  UMK
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <infoblk.h>
#include <limits.h>
#include <sys/ucontext.h>
#include <sys/times.h>
#include <sys/unistd.h>
#include <mpp/globals.h>

#define PERFCNT_ON      1
#define PERFCNT_OFF     2
#define PERFCNT_EV5     1
#define CTL_OFF		0
#define CTL_ON		2
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

enum {
   PNE_T3E_MACHINE_CYC = 0x40000000,
   PNE_T3E_INS,
   PNE_T3E_NON_ISSUE_CYC,
   PNE_T3E_SPLIT_ISSUE_CYC,
   PNE_T3E_ISSUE_PIPE_DRY_CYC,
   PNE_T3E_REPLAY_TRAP_CYC,
   PNE_T3E_SINGLE_ISSUE_CYC,
   PNE_T3E_DUAL_ISSUE_CYC,
   PNE_T3E_TRIPLE_ISSUE_CYC,
   PNE_T3E_QUAD_ISSUE_CYC,
   PNE_T3E_JSR_RET,
   PNE_T3E_INT_INS,
   PNE_T3E_FP_INS,
   PNE_T3E_LOADS,
   PNE_T3E_STORES,
   PNE_T3E_ICACHE,
   PNE_T3E_DCACHE,
   PNE_T3E_SCACHE_ACCESSES_CBOX1,
   PNE_T3E_15_CYC_STALLS,
   PNE_T3E_PC_MISPREDICTS,
   PNE_T3E_BR_MISPREDICTS,
   PNE_T3E_ICACHE_RFB_MISSES,
   PNE_T3E_ITB_MISSES,
   PNE_T3E_DCACHE_MISSES,
   PNE_T3E_DTB_MISSES,
   PNE_T3E_LD_MERGED_IN_MAF,
   PNE_T3E_LDU_REPLAY_TRAPS,
   PNE_T3E_WB_MAF_FULL_REPLAY_TRAPS,
   PNE_T3E_PERF_MON_H_INPUT,
   PNE_T3E_CPU_CYC,
   PNE_T3E_MB_STALL_CYC,
   PNE_T3E_LDXL_INS,
   PNE_T3E_SCACHE_MISSES_CBOX2
};

/* First entry is preset event name.
   Then is the derived event specification and the counter mapping.
   Notes:
   Resource stalls only count long(>15 cycle) stalls and not MB stall cycles
*/
const hwi_search_t _papi_hwd_preset_map[] = {
   {PAPI_L1_DCM, {0, {PAPI_NULL, PNE_T3E_DCACHE, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_ICACHE_RFB_MISSES}, {0,}}},
   {PAPI_TLB_DM, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_DTB_MISSES}, {0,}}},
   {PAPI_TLB_IM, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_ITB_MISSES}, {0,}}},
   {PAPI_MEM_SCY, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_MB_STALL_CYC}, {0,}}},
   {PAPI_STL_ICY, {0, {PAPI_NULL, PNE_T3E_NON_ISSUE_CYC, PAPI_NULL}, {0,}}},
   {PAPI_FUL_ICY, {0, {PAPI_NULL,  PNE_T3E_QUAD_ISSUE_CYC, PAPI_NULL}, {0,}}},
   {PAPI_STL_CCY, {0, {PAPI_NULL, PNE_T3E_NON_ISSUE_CYC, PAPI_NULL}, {0,}}},
   {PAPI_FUL_CCY, {0, {PAPI_NULL, PNE_T3E_QUAD_ISSUE_CYC, PAPI_NULL}, {0,}}},
   {PAPI_BR_UCN, {DERIVED_ADD, {PAPI_NULL, PNE_T3E_JSR_RET, PNE_T3E_PC_MISPREDICTS}, {0,}}},
   {PAPI_BR_CN, {DERIVED_ADD, {PAPI_NULL, PNE_T3E_JSR_RET, PNE_T3E_BR_MISPREDICTS}, {0,}}},
   {PAPI_BR_MSP, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_BR_MISPREDICTS}, {0,}}},
   {PAPI_TOT_ITS, {0, {PAPI_NULL, PNE_T3E_ICACHE, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {PNE_T3E_INS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_INT_INS, {0, {PAPI_NULL, PNE_T3E_INT_INS, PAPI_NULL}, {0,}}},
   {PAPI_FP_INS, {0, {PAPI_NULL, PNE_T3E_FP_INS, PAPI_NULL}, {0,}}},
   {PAPI_LD_INS, {0, {PAPI_NULL, PNE_T3E_LOADS, PAPI_NULL}, {0,}}},
   {PAPI_SR_INS, {0, {PAPI_NULL, PNE_T3E_STORES, PAPI_NULL}, {0,}}},
   {PAPI_BR_INS, {0, {PAPI_NULL, PNE_T3E_JSR_RET, PAPI_NULL}, {0,}}},
   {PAPI_RES_STL, {0, {PAPI_NULL, PAPI_NULL, PNE_T3E_15_CYC_STALLS}, {0,}}},
   {PAPI_TOT_CYC, {0, {PNE_T3E_MACHINE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}}
};

const native_event_entry_t native_table[] = {
   {"MACHINE_CYCLES",
    "Count machine cycles.",
    {CNTR1, 0x0}},
   {"INSTRUCTIONS",
    "Count instructions.",
    {CNTR1, 0x1}},
   {"NON_ISSUE_CYCLES",
    "Count non issue cycles.",
    {CNTR2, 0x0}},
   {"SPLIT_ISSUE_CYCLES",
    "Count split issue cycles.",
    {CNTR2, 0x1}},
   {"ISSUE_PIPE_DRY_CYCLES",
    "Count issue-pipe-dry cycles.",
    {CNTR2, 0x2}},
   {"REPLAY_TRAP_CYCLES",
    "Count replay trap cycles.",
    {CNTR2, 0x3}},
   {"SINGLE_ISSUE_CYCLES",
    "Count single-issue cycles.",
    {CNTR2, 0x4}},
   {"DUAL_ISSUE_CYCLES",
    "Count dual-issue cycles.",
    {CNTR2, 0x5}},
   {"TRIPLE_ISSUE_CYCLES",
    "Count triple-issue cycles.",
    {CNTR2, 0x6}},
   {"QUAD_ISSUE_CYCLES",
    "Count quad-issue cycles.",
    {CNTR2, 0x7}},
   {"JSR_RET",
    "Count jsr-ret, cond-branch, flow-change instrucs.",
    {CNTR2, 0x8}},
   {"INTEGER_INSTRUCTIONS",
    "Count int instructions issued.",
    {CNTR2, 0x9}},
   {"FP_INSTRUCTIONS",
    "Count fp instructions issued.",
    {CNTR2, 0xA}},
   {"LOADS_ISSUED",
    "Count loads issued.",
    {CNTR2, 0xB}},
   {"STORES_ISSUED",
    "Count stores issued.",
    {CNTR2, 0xC}},
   {"ICACHE_ISSUED",
    "Count Icache issued.",
    {CNTR2, 0xD}},
   {"DCACHE_ACCESSES",
    "Count Dcache accessed.",
    {CNTR2, 0xE}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count Scache accesses, CBOX input 1.",
    {CNTR2, 0xF}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count >15 cycle stalls.",
    {CNTR3, 0x0}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count PC-mispredicts.",
    {CNTR3, 0x2}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count BR-mispredicts.",
    {CNTR3, 0x3}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count icache/RFB misses.",
    {CNTR3, 0x4}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count ITB misses.",
    {CNTR3, 0x5}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count dcache misses.",
    {CNTR3, 0x6}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count DTB misses.",
    {CNTR3, 0x7}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count LDs merged in MAF.",
    {CNTR3, 0x8}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count LDU replay traps.",
    {CNTR3, 0x9}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count WB/MAF full replay traps.",
    {CNTR3, 0xA}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count perf_mon_h input at sysclock intervals.",
    {CNTR3, 0xB}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count CPU cycles.",
    {CNTR3, 0xC}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count MB stall cycles.",
    {CNTR3, 0xD}},
   {"SCACHE_ACCESSES_CBOX1"
    "Count LDxL instructions issued.",
    {CNTR3, 0xE}},
   {"SCACHE_MISSES_CBOX2"
    "Count Scache misses, CBOX input 2.",
    {CNTR3, 0xF}},
   {"", "", {0, 0}}
};

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
} hwd_control_state_t;

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

typedef struct hwd_register {
   unsigned int selector;       /* Mask for which counter in used */
   int code;                    /* The event code */
} hwd_register_t;

typedef struct native_event_entry {
   /* If it exists, then this is the name of this event */
   char name[PAPI_MAX_STR_LEN];
   /* If it exists, then this is the description of this event */
   char *description;
   /* The event's resource listing */
   hwd_register_t resources;
} native_event_entry_t;

#define GET_OVERFLOW_ADDRESS(ctx) (void*)(ctx->uc_mcontext.gregs[31])

#endif
