/*
* File:    t3e_events.c
* CVS:     $Id$
* Author:  Joseph Thomas
*          jthomas@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

hwi_search_t *preset_search_map;

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
   PNE_T3E_JSR_RET_FC,
   PNE_T3E_COND_BR_FC,
   PNE_T3E_ALL_FC,
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
   {PAPI_L1_DCM, {0, {PNE_T3E_DCACHE, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_L1_ICM, {0, {PNE_T3E_ICACHE_RFB_MISSES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_DM, {0, {PNE_T3E_DTB_MISSES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TLB_IM, {0, {PNE_T3E_ITB_MISSES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_MEM_SCY, {0, {PNE_T3E_MB_STALL_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_STL_ICY, {0, {PNE_T3E_NON_ISSUE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FUL_ICY, {0, {PNE_T3E_QUAD_ISSUE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_STL_CCY, {0, {PNE_T3E_NON_ISSUE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FUL_CCY, {0, {PNE_T3E_QUAD_ISSUE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_UCN, {DERIVED_ADD, {PNE_T3E_JSR_RET_FC, PNE_T3E_PC_MISPREDICTS, PAPI_NULL}, {0,}}},
   {PAPI_BR_CN, {DERIVED_ADD, {PNE_T3E_JSR_RET_FC, PNE_T3E_BR_MISPREDICTS, PAPI_NULL}, {0,}}},
   {PAPI_BR_MSP, {0, {PNE_T3E_BR_MISPREDICTS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_IIS, {0, {PNE_T3E_ICACHE, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_INS, {0, {PNE_T3E_INS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_INT_INS, {0, {PNE_T3E_INT_INS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_FP_INS, {0, {PNE_T3E_FP_INS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_LD_INS, {0, {PNE_T3E_LOADS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_SR_INS, {0, {PNE_T3E_STORES, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_BR_INS, {0, {PNE_T3E_JSR_RET_FC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_RES_STL, {0, {PNE_T3E_15_CYC_STALLS, PAPI_NULL, PAPI_NULL}, {0,}}},
   {PAPI_TOT_CYC, {0, {PNE_T3E_MACHINE_CYC, PAPI_NULL, PAPI_NULL}, {0,}}},
   {0, {0, {PAPI_NULL, PAPI_NULL, PAPI_NULL}, {0,}}}
};

native_event_entry_t native_table[] = {
   {"MACHINE_CYCLES",
    "Count machine cycles.",
    {0x0, -1, -1}},
   {"INSTRUCTIONS",
    "Count instructions.",
    {0x1, -1, -1}},
   {"NON_ISSUE_CYCLES",
    "Count non issue cycles.",
    {-1, 0x0, -1}},
   {"SPLIT_ISSUE_CYCLES",
    "Count split issue cycles.",
    {-1, 0x1, -1}},
   {"ISSUE_PIPE_DRY_CYCLES",
    "Count issue-pipe-dry cycles.",
    {-1, 0x2, -1}},
   {"REPLAY_TRAP_CYCLES",
    "Count replay trap cycles.",
    {-1, 0x3, -1}},
   {"SINGLE_ISSUE_CYCLES",
    "Count single-issue cycles.",
    {-1, 0x4, -1}},
   {"DUAL_ISSUE_CYCLES",
    "Count dual-issue cycles.",
    {-1, 0x5, -1}},
   {"TRIPLE_ISSUE_CYCLES",
    "Count triple-issue cycles.",
    {-1, 0x6, -1}},
   {"QUAD_ISSUE_CYCLES",
    "Count quad-issue cycles.",
    {-1, 0x7, -1}},
   {"JSR_RET_FC",
    "Count jsr-ret flow-change instrucs.",
    {-1, 0x8, 0x2}},
   {"COND_BR_FC",
    "Count cond-branch flow-change instrucs.",
    {-1, 0x8, 0x3}},
   {"ALL_FC",
    "Count all flow-change instrucs.",
    {-1, 0x8, -1}},
   {"INTEGER_INSTRUCTIONS",
    "Count int instructions issued.",
    {-1, 0x9, -1}},
   {"FP_INSTRUCTIONS",
    "Count fp instructions issued.",
    {-1, 0xA, -1}},
   {"LOADS_ISSUED",
    "Count loads issued.",
    {-1, 0xB, -1}},
   {"STORES_ISSUED",
    "Count stores issued.",
    {-1, 0xC, -1}},
   {"ICACHE_ISSUED",
    "Count Icache issued.",
    {-1, 0xD, -1}},
   {"DCACHE_ACCESSES",
    "Count Dcache accessed.",
    {-1, 0xE, -1}},
   {"SCACHE_ACCESSES_CBOX1",
    "Count Scache accesses, CBOX input 1.",
    {-1, 0xF, -1}},
   {"15_CYC_STALLS",
    "Count >15 cycle stalls.",
    {-1, -1, 0x0}},
   {"PC_MISPREDICTS",
    "Count PC-mispredicts.",
    {-1, -1, 0x2}},
   {"BR_MISPREDICTS",
    "Count BR-mispredicts.",
    {-1, -1, 0x3}},
   {"ICACHE_RFB_MISSES",
    "Count icache/RFB misses.",
    {-1, -1, 0x4}},
   {"ITB_MISSES",
    "Count ITB misses.",
    {-1, -1, 0x5}},
   {"DCACHE_MISSES",
    "Count dcache misses.",
    {-1, -1, 0x6}},
   {"DTB_MISSES",
    "Count DTB misses.",
    {-1, -1, 0x7}},
   {"LD_MERGED_IN_MAF",
    "Count LDs merged in MAF.",
    {-1, -1, 0x8}},
   {"LDU_REPLAY_TRAPS",
    "Count LDU replay traps.",
    {-1, -1, 0x9}},
   {"WB_MAF_FULL_REPLAY_TRAPS",
    "Count WB/MAF full replay traps.",
    {-1, -1, 0xA}},
   {"PERF_MON_H_INPUT",
    "Count perf_mon_h input at sysclock intervals.",
    {-1, -1, 0xB}},
   {"CPU_CYC",
    "Count CPU cycles.",
    {-1, -1, 0xC}},
   {"MB_STALL_CYC",
    "Count MB stall cycles.",
    {-1, -1, 0xD}},
   {"LDXL_INS",
    "Count LDxL instructions issued.",
    {-1, -1, 0xE}},
   {"SCACHE_MISSES_CBOX2",
    "Count Scache misses, CBOX input 2.",
    {-1, -1, 0xF}},
   {"", "", {0, 0, 0}}
};

/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/* Given a native event code, returns the short text label. */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].name);
}

/* Given a native event code, returns the longer native event
   description. */
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & PAPI_NATIVE_AND_MASK].description);
}

/* Given a native event code, assigns the native event's
   information to a given pointer. */
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   if(native_table[(EventCode & PAPI_NATIVE_AND_MASK)].resources.selector[0] == 0)
{
      return (PAPI_ENOEVNT);
   }
   bits = &native_table[EventCode & PAPI_NATIVE_AND_MASK].resources;
   return (PAPI_OK);
}

/* Given a native event code, looks for the next event in the table
   if the next one exists.  If not, returns the proper error code. */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (native_table[(*EventCode & PAPI_NATIVE_AND_MASK) + 1].resources.selector[0]) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else {
      return (PAPI_ENOEVNT);
   }
}
