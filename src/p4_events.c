/* 
* File:    p4_native.c
* CVS:     $Id$
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. */

/* You requested all the ESCR/CCCR/Counter triplets that allow one to
count cycles.  Well, this is a special case in that an ESCR is not
needed at all. By configuring the threshold comparison appropriately
in a CCCR, you can get the counter to count every cycle, independent
of whatever ESCR the CCCR happens to be listening to.  To do this, set
the COMPARE and COMPLEMENT bits in the CCCR and set the THRESHOLD
value to "1111" (binary).  This works because the setting the
COMPLEMENT bit makes the threshold comparison to be "less than or
equal" and, with THRESHOLD set to its maximum value, the comparison
will always succeed and the counter will increment by one on every
clock cycle. */

#ifdef __i386__

// Definitions of the indexes of all currently defined P4 native events.
// If a new native event is defined below, a new entry must be made in the
// proper place in this enum.
enum {
  PNE_branch_retired_all = 0x40000000,
  PNE_branch_retired_not_taken,
  PNE_branch_retired_taken,
  PNE_branch_retired_predicted,
  PNE_branch_retired_mispredicted,
  PNE_branch_retired_not_taken_predicted,
  PNE_branch_retired_taken_predicted,
  PNE_branch_retired_taken_mispredicted,
  PNE_branch_retired_not_taken_mispredicted,
  PNE_cycles,
  PNE_page_walk_type_data_miss,
  PNE_page_walk_type_instr_miss,
  PNE_page_walk_type_all,
  PNE_x87_FP_uop_tag0,
  PNE_execution_event_nbogus0,
  PNE_replay_event,
  PNE_replay_event_L1_load_miss,
  PNE_replay_event_L1_store_miss,
  PNE_replay_event_L1_data_miss,
  PNE_replay_event_L1_data_access,
  PNE_replay_event_L2_load_miss,
  PNE_replay_event_L2_store_miss,
  PNE_replay_event_L2_data_miss,
  PNE_instr_retired_non_bogus,
  PNE_instr_retired_all
};


const preset_search_t _papi_hwd_pentium4_mlt2_preset_map[] = {
/* preset, derived, native index array */
  {PAPI_RES_STL, 0, { PNE_replay_event,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_INS,	 0, { PNE_branch_retired_all,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_TKN,  0, { PNE_branch_retired_taken,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_NTK,  0, { PNE_branch_retired_not_taken,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_MSP,  0, { PNE_branch_retired_mispredicted,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_PRC,  0, { PNE_branch_retired_predicted,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_DM,  0, { PNE_page_walk_type_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_IM,  0, { PNE_page_walk_type_instr_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_TL,  0, { PNE_page_walk_type_all,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TOT_INS, 0, { PNE_instr_retired_non_bogus,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_FP_INS,  0, { PNE_execution_event_nbogus0,PNE_x87_FP_uop_tag0,-1,-1,-1,-1,-1,-1}},
  {PAPI_TOT_CYC, 0, { PNE_cycles,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_LDM,  0, { PNE_replay_event_L1_load_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_STM,  0, { PNE_replay_event_L1_store_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_DCM,  0, { PNE_replay_event_L1_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_LDM,  0, { PNE_replay_event_L2_load_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_STM,  0, { PNE_replay_event_L2_store_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_DCM,  0, { PNE_replay_event_L2_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  { 0,		 0, { -1,-1,-1,-1,-1,-1,-1,-1}}
};

const preset_search_t _papi_hwd_pentium4_mge2_preset_map[] = {
/* preset, derived, native index array */
  {PAPI_RES_STL, 0, { PNE_replay_event,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_INS,	 0, { PNE_branch_retired_all,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_TKN,  0, { PNE_branch_retired_taken,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_NTK,  0, { PNE_branch_retired_not_taken,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_MSP,  0, { PNE_branch_retired_mispredicted,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_BR_PRC,  0, { PNE_branch_retired_predicted,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_DM,  0, { PNE_page_walk_type_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_IM,  0, { PNE_page_walk_type_instr_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TLB_TL,  0, { PNE_page_walk_type_all,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TOT_INS, 0, { PNE_instr_retired_non_bogus,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_TOT_IIS, 0, { PNE_instr_retired_all,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_FP_INS,  0, { PNE_execution_event_nbogus0,PNE_x87_FP_uop_tag0,-1,-1,-1,-1,-1,-1}},
  {PAPI_TOT_CYC, 0, { PNE_cycles,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_LDM,  0, { PNE_replay_event_L1_load_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_STM,  0, { PNE_replay_event_L1_store_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_DCM,  0, { PNE_replay_event_L1_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L1_DCA,  0, { PNE_replay_event_L1_data_access,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_LDM,  0, { PNE_replay_event_L2_load_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_STM,  0, { PNE_replay_event_L2_store_miss,-1,-1,-1,-1,-1,-1,-1}},
  {PAPI_L2_DCM,  0, { PNE_replay_event_L2_data_miss,-1,-1,-1,-1,-1,-1,-1}},
  { 0,		 0, { -1,-1,-1,-1,-1,-1,-1,-1}}
};

// list of all possible ESCR registers
// these values are used as bit-shifters to build a mask of ESCRs in use
// the ESCR is a limiting resource, just like counters
enum { 
  MSR_BSU_ESCR0 = 0,
  MSR_BSU_ESCR1,
  MSR_FSB_ESCR0,
  MSR_FSB_ESCR1,
  MSR_FIRM_ESCR0,
  MSR_FIRM_ESCR1,
  MSR_FLAME_ESCR0,
  MSR_FLAME_ESCR1,
  MSR_DAC_ESCR0,
  MSR_DAC_ESCR1,
  MSR_MOB_ESCR0,
  MSR_MOB_ESCR1,
  MSR_PMH_ESCR0,
  MSR_PMH_ESCR1,
  MSR_SAAT_ESCR0,
  MSR_SAAT_ESCR1,
  MSR_U2L_ESCR0,
  MSR_U2L_ESCR1,
  MSR_BPU_ESCR0,
  MSR_BPU_ESCR1,
  MSR_IS_ESCR0,
  MSR_IS_ESCR1,
  MSR_ITLB_ESCR0,
  MSR_ITLB_ESCR1,
  MSR_CRU_ESCR0,
  MSR_CRU_ESCR1,
  MSR_IQ_ESCR0,
  MSR_IQ_ESCR1,
  MSR_RAT_ESCR0,
  MSR_RAT_ESCR1,
  MSR_SSU_ESCR0,
  MSR_MS_ESCR0,
  MSR_MS_ESCR1,
  MSR_TBPU_ESCR0,
  MSR_TBPU_ESCR1,
  MSR_TC_ESCR0,
  MSR_TC_ESCR1,
  MSR_IX_ESCR0,
  MSR_IX_ESCR1,
  MSR_ALF_ESCR0,
  MSR_ALF_ESCR1,
  MSR_CRU_ESCR2,
  MSR_CRU_ESCR3,
  MSR_CRU_ESCR4,
  MSR_CRU_ESCR5
};

hwd_p4_native_map_t _papi_hwd_pentium4_native_map[PAPI_MAX_NATIVE_EVENTS] = {
  {
    "branch_retired_all",
    "This event counts the retirement of all taken, not-taken, predicted and mispredicted branches.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK(0xf) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_not_taken",
    "This event counts the retirement of branches not-taken.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK((BR_RET_ESCR_MASK_NT_PR | BR_RET_ESCR_MASK_NT_MPR)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_taken",
    "This event counts the retirement of branches taken.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK((BR_RET_ESCR_MASK_T_PR | BR_RET_ESCR_MASK_T_MPR)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_predicted",
    "This event counts the retirement of branches predicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK((BR_RET_ESCR_MASK_T_PR | BR_RET_ESCR_MASK_NT_PR)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_mispredicted",
    "This event counts the retirement of branches mispredicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK((BR_RET_ESCR_MASK_NT_MPR | BR_RET_ESCR_MASK_T_MPR)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_not_taken_predicted",
    "This event counts the retirement of branches not-taken and predicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK(BR_RET_ESCR_MASK_NT_PR) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_taken_predicted",
    "This event counts the retirement of branches taken and predicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK(BR_RET_ESCR_MASK_T_PR) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_taken_mispredicted",
    "This event counts the retirement of branches taken and mispredicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK(BR_RET_ESCR_MASK_T_MPR) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "branch_retired_not_taken_mispredicted",
    "This event counts the retirement of branches not taken and mispredicted.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(BR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(BR_RET_ESCR) | 
      ESCR_EVENT_MASK(BR_RET_ESCR_MASK_NT_MPR) | 
      CPL(1),
      0,0,0
    },
    0
  },

  /* This one's a bit of a hack.
  It can live on ANY counter and use ANY ESCR, but it must be a valid pairing.
  For now, let's just code it using the TC_deliver_mode group, because that one
  doesn't seem to be used for much else...
  This is a good opportunity to define synonyms for use with other groups...
  */
  {
    "cycles",
    "This event counts every cycle by setting the threshold of a random event to count every tick.",
    {
      { MSR_MS_COUNTER01, MSR_MS_COUNTER23 },
      { MSR_TC_ESCR0,	MSR_TC_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(TC_DLVR_CCCR) | CCCR_ENABLE | CCCR_COMPARE | CCCR_COMPLEMENT | CCCR_THRESHOLD(0xf), 
      ESCR_EVENT_SEL(TC_DLVR_ESCR) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "page_walk_type_data_miss",
    "This event counts data TLB page walks that the page miss handler (PMH) performs.",
    {
      { MSR_BPU_COUNTER01, MSR_BPU_COUNTER23 },
      { MSR_PMH_ESCR0,	 MSR_PMH_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(PG_WLK_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(PG_WLK_ESCR) | 
      ESCR_EVENT_MASK(PG_WLK_ESCR_MASK_DTMISS) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "page_walk_type_instr_miss",
    "This event counts instruction TLB page walks that the page miss handler (PMH) performs.",
    {
      { MSR_BPU_COUNTER01, MSR_BPU_COUNTER23 },
      { MSR_PMH_ESCR0,	 MSR_PMH_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(PG_WLK_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(PG_WLK_ESCR) | 
      ESCR_EVENT_MASK(PG_WLK_ESCR_MASK_ITMISS) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "page_walk_type_all",
    "This event counts data and instruction page walks that the page miss handler (PMH) performs.",
    {
      { MSR_BPU_COUNTER01, MSR_BPU_COUNTER23 },
      { MSR_PMH_ESCR0,	 MSR_PMH_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(PG_WLK_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(PG_WLK_ESCR) | 
      ESCR_EVENT_MASK((PG_WLK_ESCR_MASK_DTMISS | PG_WLK_ESCR_MASK_ITMISS)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "x87_FP_uop_tag0",
    "This event increments for each x87 floating-point 發p, specified through the event mask for detection.",
    {
      { MSR_FLAME_COUNTER01, MSR_FLAME_COUNTER23 },
      { MSR_FIRM_ESCR0,	   MSR_FIRM_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(X87_FP_UOP_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(X87_FP_UOP_ESCR) | 
      ESCR_EVENT_MASK(X87_FP_UOP_ESCR_MASK_ALL) | ESCR_TAG_ENABLE | ESCR_TAG_VAL(1) |
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "execution_event_nbogus0",
    "This event counts the retirement of tagged 發ps, which are specified through the execution tagging mechanism. The event mask allows from one to four types of 發ps to be specified as either bogus or non-bogus 發ps to be tagged.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(EXECUTION_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(EXECUTION_ESCR) | 
      ESCR_EVENT_MASK(EXECUTION_ESCR_MASK_NBOGUS0) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "replay_event",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "replay_event_L1_load_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD, 0
    },
    0
  },
  {
    "replay_event_L1_store_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE, 0
    },
    0
  },
  {
    "replay_event_L1_data_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD, 0
    },
    0
  },
  {
    "replay_event_L1_data_access",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG, PEBS_MV_STORE | PEBS_MV_LOAD, 0
    },
    0
  },
  {
    "replay_event_L2_load_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD, 0
    },
    0
  },
  {
    "replay_event_L2_store_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE, 0
    },
    0
  },
  {
    "replay_event_L2_data_miss",
    "This event counts the retirement of tagged 發ps, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus 發ps..",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR2,	 MSR_CRU_ESCR3 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(REPLAY_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(REPLAY_ESCR) | 
      ESCR_EVENT_MASK(REPLAY_ESCR_MASK_NBOGUS) | 
      CPL(1),
      PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD, 0
    },
    0
  },
  {
    "instr_retired_non_bogus",
    "This event counts instructions that are retired during a clock cycle. Mask bits specify bogus or non-bogus (and whether they are tagged via the front-end tagging mechanism.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR0,	 MSR_CRU_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(INSTR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(INSTR_RET_ESCR) | 
      ESCR_EVENT_MASK((INSTR_RET_ESCR_MASK_NBOGUSNTAG | INSTR_RET_ESCR_MASK_NBOGUSTAG)) | 
      CPL(1),
      0,0,0
    },
    0
  },
  {
    "instr_retired_all",
    "This event counts instructions that are retired during a clock cycle. Mask bits specify bogus or non-bogus (and whether they are tagged via the front-end tagging mechanism.",
    {
      { MSR_IQ_COUNTER014, MSR_IQ_COUNTER235 },
      { MSR_CRU_ESCR0,	 MSR_CRU_ESCR1 },
      CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ESCR_SEL(INSTR_RET_CCCR) | CCCR_ENABLE, 
      ESCR_EVENT_SEL(INSTR_RET_ESCR) | 
      ESCR_EVENT_MASK((INSTR_RET_ESCR_MASK_NBOGUSNTAG | INSTR_RET_ESCR_MASK_NBOGUSTAG | INSTR_RET_ESCR_MASK_BOGUSNTAG | INSTR_RET_ESCR_MASK_BOGUSTAG)) | 
      CPL(1),
      0,0,0
    },
    0
  }
};

/*
  Not sure how to encode this one; can live on ANY counter and use ANY ESCR.
  Problem is, you have to pick a valid pairing between counter and ESCR
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	CCCR_THR_MODE(CCCR_THR_ANY) | CCCR_ENABLE | CCCR_COMPARE | CCCR_COMPLEMENT | CCCR_THRESHOLD(0xf), 
	CPL(1)} }}},
  { 0, NULL, }
*/


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

char *_papi_hwd_native_code_to_name(unsigned int EventCode)
{
  return(_papi_hwd_pentium4_native_map[EventCode & NATIVE_AND_MASK].name);
}

char *_papi_hwd_native_code_to_descr(unsigned int EventCode)
{
  return(_papi_hwd_pentium4_native_map[EventCode & NATIVE_AND_MASK].description);
}

int _papi_hwd_native_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
  *bits = _papi_hwd_pentium4_native_map[EventCode & NATIVE_AND_MASK].resources;
  return(PAPI_OK);
}

//    _papi_hwd_encode_native();
//    _papi_hwd_decode_native();



/* These table are the deprecated way of doing things & need to be deleted...
const P4_search_t _papi_hwd_pentium4_mlt2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};

const P4_search_t _papi_hwd_pentium4_mge2_preset_map[] = {
  { PAPI_RES_STL, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x9) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_BR_INS, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_BR_TKN, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xc) | CPL(1)} }}},
  { PAPI_BR_NTK, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_BR_MSP, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0xa) | CPL(1)} }}},
  { PAPI_BR_PRC, NULL, 1,
   {{{ COUNTER(14) | COUNTER(15) | COUNTER(17), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(0x6) | ESCR_EVENT_MASK(0x5) | CPL(1)} }}},
  { PAPI_TLB_DM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x1) | CPL(1)} }}},
  { PAPI_TLB_IM, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x2) | CPL(1)} }}},
  { PAPI_TLB_TL, NULL, 1,
   {{{ COUNTER(0) | COUNTER(1), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x1) | ESCR_EVENT_MASK(0x3) | CPL(1)} }}},
  { PAPI_TOT_INS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | NBOGUSNTAG | CPL(1)} }}},
  { PAPI_TOT_IIS, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(4) | ENABLE, 
       EVENT(0x2) | ESCR_EVENT_MASK(0xf) | CPL(1)} }}},
  { PAPI_FP_INS,  NULL, 2,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(0xC) | NBOGUSNTAG | CPL(1) }, 
      { COUNTER(8) | COUNTER(9), 
	HYPERTHREAD_ANY | ESCR(1) | ENABLE, 
	EVENT(0x4) | (1 << 24) | (1 << 5) | (1 << 4) | CPL(1)} }}},
  { PAPI_TOT_CYC, NULL, 1,
    {{{ COUNTER(4), 
	HYPERTHREAD_ANY | ENABLE | COMPARE | COMPLEMENT | THRESHOLD(0xf), 
	CPL(1)} }}},
  { PAPI_L1_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L1_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16),  
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE} }}},
  { PAPI_L1_DCM, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG | PEBS_L1_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L1_DCA, NULL, 1,
   {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
       HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
       EVENT(9) | EVENTMASK(0x1) | CPL(1), 
       PEBS_TAG , PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { PAPI_L2_LDM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_LOAD} }}},
  { PAPI_L2_STM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE} }}},
  { PAPI_L2_DCM, NULL, 1,
    {{{ COUNTER(12) | COUNTER(13) | COUNTER(16), 
	HYPERTHREAD_ANY | ESCR(5) | ENABLE, 
	EVENT(9) | EVENTMASK(0x1) | CPL(1), 
	PEBS_TAG | PEBS_L2_MISS, PEBS_MV_STORE | PEBS_MV_LOAD} }}},
  { 0, NULL, }
};
*/

#endif


#ifdef __x86_64__
#define ALLCNTRS 0xf
const P4_search_t _papi_hwd_x86_64_opteron_map[] = {
  { PAPI_L1_DCM, NULL, 1,
    {{{ ALLCNTRS, 0x0041}}}
  },
  { PAPI_L1_ICM, NULL, 1,
    {{{ ALLCNTRS, 0x0081}}}
  },
  { PAPI_L2_DCM, NULL, 1,
    {{{ ALLCNTRS, 0x027E}}}
  },
  { PAPI_L2_ICM, NULL, 1,
    {{{ ALLCNTRS, 0x017E}}}
  },
  /*{ PAPI_L1_TCM, DERIVED_ADD, 2,
    {{{ ALLCNTRS, 0x0041,0x0081}}}
    },*/
  { PAPI_L2_TCM, NULL, 1,
    {{{ ALLCNTRS, 0x037E}}}
  },
  /* Need to think a lot about these events */
  /*  { PAPI_CA_SNP, NULL, 1, 
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_SHR, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_CLN, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_INV, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },
      { PAPI_CA_ITV, NULL, 1,
      {{{ ALLCNTRS, 0x}}}
      },*/
  { PAPI_FPU_IDL, NULL, 1,
    {{{ ALLCNTRS, 0x01}}}
  },
  { PAPI_TLB_DM, NULL, 1,
    {{{ ALLCNTRS, 0x46}}}
  },
  { PAPI_TLB_IM, NULL, 1,
    {{{ ALLCNTRS, 0x85}}}
  },
    /*  { PAPI_TLB_TL, DERIVED_ADD, 2,
	{{{ ALLCNTRS, 0x46, 0x85}}}
	},*/
  { PAPI_MEM_SCY, NULL, 1,
    {{{ ALLCNTRS, 0xD8}}}
  },
  { PAPI_STL_ICY, NULL, 1,
    {{{ ALLCNTRS, 0xD0}}}
  },
  { PAPI_HW_INT, NULL, 1,
    {{{ ALLCNTRS, 0xCF}}}
  },
  { PAPI_BR_TKN, NULL, 1,
    {{{ ALLCNTRS, 0xC4}}}
  },
  { PAPI_BR_MSP, NULL, 1,
    {{{ ALLCNTRS, 0xC3}}}
  },
  { PAPI_TOT_INS, NULL, 1,
    {{{ ALLCNTRS, 0xC0}}}
  },
  { PAPI_FP_INS, NULL, 1,
    {{{ ALLCNTRS, 0x0300}}}
  },
  { PAPI_BR_INS, NULL, 1,
    {{{ ALLCNTRS, 0xC2}}}
  },
  { PAPI_VEC_INS, NULL, 1,
    {{{ ALLCNTRS, 0x0ECB}}}
  },
  { PAPI_RES_STL, NULL, 1,
    {{{ ALLCNTRS, 0xD1}}}
  },
  { PAPI_FP_STAL, NULL, 1,
    {{{ ALLCNTRS, 0x01}}}
  },
  { PAPI_TOT_CYC, NULL, 1,
    {{{ ALLCNTRS, 0xC0}}}
  },
    /*  { PAPI_L1_DCH, DERIVED_SUB, 2,
	{{{ ALLCNTRS, 0x40, 0x41}}}
	},*/
  { PAPI_L1_DCA, NULL, 1,
    {{{ ALLCNTRS, 0x040}}}
  },
  { PAPI_L2_DCH, NULL, 1,
    {{{ ALLCNTRS, 0x1F42}}}
  },
  { PAPI_L2_DCA, NULL, 1,
    {{{ ALLCNTRS, 0x041}}}
  },
  { PAPI_FML_INS, NULL, 1,
    {{{ ALLCNTRS, 0x100}}}
  },
  { PAPI_FAD_INS, NULL, 1,
    {{{ ALLCNTRS, 0x200}}}
  },
  { 0, NULL, }
};
#endif
