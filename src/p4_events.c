/* 
* File:    p4_native.c
* CVS:     $Id$
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"
#include "perfctr-p4.h"

extern papi_vector_t _p4_vector;

#if defined(__i386__) || defined(__x86_64__)
/*
    Native event index structure:

   +----------+--------+----------------+
   |0100000000| event  |      mask      |
   +----------+--------+----------------+
    31   -  24 23 -- 16 15    ----      0
*/


/*  Definitions of the virtual indexes of all P4 native events used in the preset tables.
    To create a new native event index, make up an appropriate name (used only internally);
    the definition for the name consists of:
    - the PAPI_NATIVE_MASK bit;
    - one of the event group names from the enum table in p4_events.h, shifted to the 3rd byte;
    - any required mask bits from that event group, shifted into position;
      valid mask bits for each group are defined in p4_events.h

   Definitive descriptions of the event groups can be found in:
    IA32 Intel Arch. SW. Dev. Man. V3: Appendix A, Table A-1
    Order Number 245472-012, 2003.
*/

/* branch events */
#define PNE_branch_retired_all (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNP) + (1<<MMNM) + (1<<MMTP) + (1<<MMTM))
#define PNE_branch_retired_not_taken (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNP) + (1<<MMNM))
#define PNE_branch_retired_taken (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMTP) + (1<<MMTM))
#define PNE_branch_retired_predicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNP) + (1<<MMTP))
#define PNE_branch_retired_mispredicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNM) + (1<<MMTM))
#define PNE_branch_retired_not_taken_predicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNP))
#define PNE_branch_retired_taken_predicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMTP))
#define PNE_branch_retired_taken_mispredicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMTM))
#define PNE_branch_retired_not_taken_mispredicted (PAPI_NATIVE_MASK + (P4_branch_retired<<16) + (1<<MMNM))

/* TLB events */
#define PNE_page_walk_type_data_miss (PAPI_NATIVE_MASK + (P4_page_walk_type<<16) + (1<<DTMISS))
#define PNE_page_walk_type_instr_miss (PAPI_NATIVE_MASK + (P4_page_walk_type<<16) + (1<<ITMISS))
#define PNE_page_walk_type_all (PAPI_NATIVE_MASK + (P4_page_walk_type<<16) + (1<<DTMISS) + (1<<ITMISS))

/* trace (instruction) cache events */
#define PNE_bpu_fetch_request_tcmiss (PAPI_NATIVE_MASK + (P4_BPU_fetch_request<<16) + (1<<TCMISS))
#define PNE_uop_queue_writes_from_tc_build_deliver (PAPI_NATIVE_MASK + (P4_uop_queue_writes<<16) + (1<<FROM_TC_BUILD) + (1<<FROM_TC_DELIVER))

/* cache events via replay tagging */
#define PNE_replay_event_L1_load_miss (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_LOAD_BIT)  + (1<<PEBS_L1_MISS_BIT))
#define PNE_replay_event_L1_data_miss (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_LOAD_BIT) + (1<<PEBS_MV_STORE_BIT) + (1<<PEBS_L1_MISS_BIT))
#define PNE_replay_event_L1_data_access (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_LOAD_BIT) + (1<<PEBS_MV_STORE_BIT))
#define PNE_replay_event_L2_load_miss (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_LOAD_BIT)  + (1<<PEBS_L2_MISS_BIT))
#define PNE_replay_event_L2_store_miss (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_STORE_BIT)  + (1<<PEBS_L2_MISS_BIT))
#define PNE_replay_event_L2_data_miss (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS) + (1<<PEBS_MV_LOAD_BIT) + (1<<PEBS_MV_STORE_BIT) + (1<<PEBS_L2_MISS_BIT))
#define PNE_replay_event (PAPI_NATIVE_MASK + (P4_replay_event<<16) + (1<<NBOGUS))

/* cache events via front-end tagging */
#define PNE_front_end_event (PAPI_NATIVE_MASK + (P4_front_end_event<<16) + (1<<NBOGUS))
#define PNE_front_end_event_bogus (PAPI_NATIVE_MASK + (P4_front_end_event<<16) + (1<<BOGUS))
#define PNE_front_end_event_all (PAPI_NATIVE_MASK + (P4_front_end_event<<16) + (1<<NBOGUS) + (1<<BOGUS))
#define PNE_uop_type_load (PAPI_NATIVE_MASK + (P4_uop_type<<16) + (1<<TAGLOADS))
#define PNE_uop_type_store (PAPI_NATIVE_MASK + (P4_uop_type<<16) + (1<<TAGSTORES))
#define PNE_uop_type_load_store (PAPI_NATIVE_MASK + (P4_uop_type<<16) + (1<<TAGLOADS) + (1<<TAGSTORES))

/* L2 and L3 cache events via BSQ_cache_reference */
#define PNE_BSQ_cache_reference (PAPI_NATIVE_MASK + (P4_BSQ_cache_reference<<16))

/* instruction events */
#define PNE_instr_retired_non_bogus (PAPI_NATIVE_MASK + (P4_instr_retired<<16) + (1<<NBOGUSNTAG) + (1<<NBOGUSTAG))
#define PNE_instr_retired_all (PAPI_NATIVE_MASK + (P4_instr_retired<<16) + (1<<NBOGUSNTAG) + (1<<NBOGUSTAG) + (1<<BOGUSNTAG) + (1<<BOGUSTAG))

/* miscellaneous events */
#define PNE_cycles (PAPI_NATIVE_MASK + (P4_custom_event<<16) + 0)	// this is the first custom entry
#define PNE_global_power_running (PAPI_NATIVE_MASK + (P4_global_power_events<<16) + (1<<RUNNING))	// this is the first custom entry
#define PNE_resource_stall (PAPI_NATIVE_MASK + (P4_resource_stall<<16) + (1<<SBFULL))

/* uop tagging events for use with execution event
   Which (of 4) tag to use is arbitrary.
   Orthogonal usage allows events to be mixed without polluting each other.
   Current convention uses the following assignments:
   - Tag 0: x87 uops used to count PAPI_FP_INS
      other Tag 0 native events could be added to the event set to provide interesting results
   - Tag 1: uops useful for PAPI_FP_OPS, like x87, scalar SP and DP
   - Tag 2: uops useful for PAPI_VEC_INS, like MMX uops and packed SP and DP
   - Tag 3: currently unused.
*/
#define PNE_x87_FP_uop_tag0 (PAPI_NATIVE_MASK + (P4_x87_FP_uop<<16) + (1<<TAG0) + (1<<ALL))
#define PNE_x87_FP_uop_tag1 (PAPI_NATIVE_MASK + (P4_x87_FP_uop<<16) + (1<<TAG1) + (1<<ALL))
#define PNE_scalar_DP_uop_tag1 (PAPI_NATIVE_MASK + (P4_scalar_DP_uop<<16) + (1<<TAG1) + (1<<ALL))
#define PNE_scalar_SP_uop_tag1 (PAPI_NATIVE_MASK + (P4_scalar_SP_uop<<16) + (1<<TAG1) + (1<<ALL))
#define PNE_packed_SP_uop_tag2 (PAPI_NATIVE_MASK + (P4_packed_SP_uop<<16) + (1<<TAG2) + (1<<ALL))
#define PNE_packed_DP_uop_tag2 (PAPI_NATIVE_MASK + (P4_packed_DP_uop<<16) + (1<<TAG2) + (1<<ALL))
#define PNE_64bit_MMX_uop_tag2 (PAPI_NATIVE_MASK + (P4_64bit_MMX_uop<<16) + (1<<TAG2) + (1<<ALL))
#define PNE_128bit_MMX_uop_tag2 (PAPI_NATIVE_MASK + (P4_128bit_MMX_uop<<16) + (1<<TAG2) + (1<<ALL))

/* execution events to count retired tagged uops with each of the 4 possible tags */
/* other events could be defined that count combinations of tags */
#define PNE_execution_event_nbogus0 (PAPI_NATIVE_MASK + (P4_execution_event<<16) + (1<<NBOGUS0))
#define PNE_execution_event_nbogus1 (PAPI_NATIVE_MASK + (P4_execution_event<<16) + (1<<NBOGUS1))
#define PNE_execution_event_nbogus2 (PAPI_NATIVE_MASK + (P4_execution_event<<16) + (1<<NBOGUS2))
#define PNE_execution_event_nbogus3 (PAPI_NATIVE_MASK + (P4_execution_event<<16) + (1<<NBOGUS3))


/*
   PAPI preset events are defined in the tables below.
   Each entry consists of a PAPI name, derived info, and up to eight native event indexes
   as defined above.
   Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example.
*/

#if defined(PAPI_PENTIUM4_FP_X87)
#define FPU _p4_FP_X87
#define FPU_DESC _p4_FP_X87_dev_notes
#elif defined(PAPI_PENTIUM4_FP_X87_SSE_SP)
#define FPU _p4_FP_X87_SSE_SP
#define FPU_DESC _p4_FP_X87_SSE_SP_dev_notes
#elif defined(PAPI_PENTIUM4_FP_SSE_SP_DP)
#define FPU _p4_FP_SSE_SP_DP
#define FPU_DESC _p4_FP_SSE_SP_DP_dev_notes
#else
#define FPU _p4_FP_X87_SSE_DP
#define FPU_DESC _p4_FP_X87_SSE_DP_dev_notes
#endif

#if defined(PAPI_PENTIUM4_VEC_MMX)
#define VEC _p4_VEC_MMX
#define VEC_DESC _p4_VEC_MMX_dev_notes
#else
#define VEC _p4_VEC_SSE
#define VEC_DESC _p4_VEC_SSE_dev_notes
#endif

/*
   PAPI preset events are defined in the tables below.
   Each entry consists of a PAPI name, derived info, and up to eight native event indexes
   as defined above.
   Events that require tagging should be defined as DERIVED_CMPD and ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example.
   The tables are defined as a base table that is loaded onto all P4s, followed by a series
   of small tables that can be overloaded onto the base table to customize events for model
   or compiler switch differences.
   Generally, new events should be added to the base table, unless they are specific for a
   certain model.
*/

hwi_search_t _p4_base_preset_map[] = {
/* preset, derived, native index array */
	{PAPI_RES_STL, {0, {PNE_resource_stall, PAPI_NULL,}, {0,}}},
	{PAPI_BR_INS, {0, {PNE_branch_retired_all, PAPI_NULL,}, {0,}}},
	{PAPI_BR_TKN, {0, {PNE_branch_retired_taken, PAPI_NULL,}, {0,}}},
	{PAPI_BR_NTK, {0, {PNE_branch_retired_not_taken, PAPI_NULL,}, {0,}}},
	{PAPI_BR_MSP, {0, {PNE_branch_retired_mispredicted, PAPI_NULL,}, {0,}}},
	{PAPI_BR_PRC, {0, {PNE_branch_retired_predicted, PAPI_NULL,}, {0,}}},
	{PAPI_TLB_DM, {0, {PNE_page_walk_type_data_miss, PAPI_NULL,}, {0,}}},
	{PAPI_TLB_IM, {0, {PNE_page_walk_type_instr_miss, PAPI_NULL,}, {0,}}},
	{PAPI_TLB_TL, {0, {PNE_page_walk_type_all, PAPI_NULL,}, {0,}}},
	{PAPI_TOT_INS, {0, {PNE_instr_retired_non_bogus, PAPI_NULL,}, {0,}}},

	/* NOTE: The following three events rely on a common tagging mechanism.
	   Load and/or store events are tagged at the front of the pipeline;
	   Tagged events are counted at the back of the pipeline. Tags for
	   front end events only come in one color. Therefore if any combination
	   of these events will produce (load + store) in every counter. */
	{PAPI_LD_INS,
	 {DERIVED_CMPD, {PNE_front_end_event, PNE_uop_type_load, PAPI_NULL,},
	  {0,}}},
	{PAPI_SR_INS,
	 {DERIVED_CMPD, {PNE_front_end_event, PNE_uop_type_store, PAPI_NULL,},
	  {0,}}},
	{PAPI_LST_INS,
	 {DERIVED_CMPD, {PNE_front_end_event, PNE_uop_type_load_store, PAPI_NULL,},
	  {0,}}},

	{PAPI_FP_INS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus0, PNE_x87_FP_uop_tag0, PAPI_NULL, PAPI_NULL,},
	  {0,}}},
//   {PAPI_TOT_CYC, {0, {PNE_cycles, PAPI_NULL,}, {0,}}},
	{PAPI_TOT_CYC, {0, {PNE_global_power_running, PAPI_NULL,}, {0,}}},

	/* Level 1 cache events */
	/* we need working/tested definitions of data cache events */
	{PAPI_L1_ICM, {0, {PNE_bpu_fetch_request_tcmiss, PAPI_NULL,}, {0,}}},
	{PAPI_L1_ICA,
	 {0, {PNE_uop_queue_writes_from_tc_build_deliver, PAPI_NULL,}, {0,}}},

	/* Level 2 is a unified cache so we only map total cache events */
	/* NOTE1: These events include speculative accesses */
	/* NOTE2: Intel documentation (2004) reports this event causes 
	   both over and undercounting by as much as a factor of two 
	   due to an erratum on the chip. */
	{PAPI_L2_TCH,
	 {0,
	  {PNE_BSQ_cache_reference + ( 1 << RD_2ndL_HITS ) + ( 1 << RD_2ndL_HITE ) +
	   ( 1 << RD_2ndL_HITM ), PAPI_NULL,}, {0,}}},
	{PAPI_L2_TCM,
	 {0, {PNE_BSQ_cache_reference + ( 1 << RD_2ndL_MISS ), PAPI_NULL,}, {0,}}},
	{PAPI_L2_TCA,
	 {0,
	  {PNE_BSQ_cache_reference + ( 1 << RD_2ndL_MISS ) + ( 1 << RD_2ndL_HITS ) +
	   ( 1 << RD_2ndL_HITE ) + ( 1 << RD_2ndL_HITM ), PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* L3 cache events for machines (Xeon) that have L3 cache */
/* Like L2, L3 is unified so we implement only total cache events */
hwi_search_t _p4_L3_cache_map[] = {
/* preset, derived, native index array */
	{PAPI_L3_TCH,
	 {0,
	  {PNE_BSQ_cache_reference + ( 1 << RD_3rdL_HITS ) + ( 1 << RD_3rdL_HITE ) +
	   ( 1 << RD_3rdL_HITM ), PAPI_NULL,}, {0,}}},
	{PAPI_L3_TCM,
	 {0, {PNE_BSQ_cache_reference + ( 1 << RD_3rdL_MISS ), PAPI_NULL,}, {0,}}},
	{PAPI_L3_TCA,
	 {0,
	  {PNE_BSQ_cache_reference + ( 1 << RD_3rdL_MISS ) + ( 1 << RD_3rdL_HITS ) +
	   ( 1 << RD_3rdL_HITE ) + ( 1 << RD_3rdL_HITM ), PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_TOT_IIS for Pentium 4s >= model 2 */
hwi_search_t _p4_tot_iis_preset_map[] = {
	{PAPI_TOT_IIS, {0, {PNE_instr_retired_all, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as x87 uops retired */
hwi_search_t _p4_FP_X87[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_x87_FP_uop_tag1, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as SSE_SP uops retired */
hwi_search_t _p4_FP_SSE_SP[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_scalar_SP_uop_tag1, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as SSE_DP uops retired */
hwi_search_t _p4_FP_SSE_DP[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_scalar_DP_uop_tag1, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as x87 and scalar_SP SSE uops retired */
hwi_search_t _p4_FP_X87_SSE_SP[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_scalar_SP_uop_tag1, PNE_x87_FP_uop_tag1,
	   PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as x87 and scalar_DP SSE uops retired */
hwi_search_t _p4_FP_X87_SSE_DP[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_scalar_DP_uop_tag1, PNE_x87_FP_uop_tag1,
	   PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_FP_OPS as scalar_SP SSE and scalar_DP SSE uops retired */
hwi_search_t _p4_FP_SSE_SP_DP[] = {
	{PAPI_FP_OPS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus1, PNE_scalar_SP_uop_tag1,
	   PNE_scalar_DP_uop_tag1, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table undefining PAPI_FP_OPS, just as a test */
hwi_search_t _p4_FP_null[] = {
	{PAPI_FP_OPS, {0, {PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_VEC_INS as MMX uops */
hwi_search_t _p4_VEC_MMX[] = {
	{PAPI_VEC_INS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus2, PNE_64bit_MMX_uop_tag2,
	   PNE_128bit_MMX_uop_tag2, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* Table defining PAPI_VEC_INS as SSE uops */
hwi_search_t _p4_VEC_SSE[] = {
	{PAPI_VEC_INS,
	 {DERIVED_CMPD,
	  {PNE_execution_event_nbogus2, PNE_packed_SP_uop_tag2,
	   PNE_packed_DP_uop_tag2, PAPI_NULL,}, {0,}}},
	{0, {0, {0,}, {0,}}}
};

/* These are examples of dense developer notes arrays. Each consists of an array
   of structures containing an event and a note string. Pointers to these strings 
   are inserted into a sparse event description structure at init time. This allows
   the use of rare developer strings with no string copies and very little space
   wasted on unused structure elements.
*/
const hwi_dev_notes_t _p4_base_dev_notes[] = {
/* preset, note */
	{PAPI_FP_INS,
	 "PAPI_FP_INS counts only retired x87 uops tagged with 0. If you add other native events tagged with 0, their counts will be included in PAPI_FP_INS"},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_X87_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS, "PAPI_FP_OPS counts retired x87 uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_SSE_SP_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS,
	 "PAPI_FP_OPS counts retired scalar_SP SSE uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_SSE_DP_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS,
	 "PAPI_FP_OPS counts retired scalar_DP SSE uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_X87_SSE_SP_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS,
	 "PAPI_FP_OPS counts retired x87 and scalar_SP SSE uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_X87_SSE_DP_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS,
	 "PAPI_FP_OPS counts retired x87 and scalar_DP SSE uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_SSE_SP_DP_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS,
	 "PAPI_FP_OPS counts retired scalar_SP and scalar_DP SSE uops tagged with 1."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_FP_null_dev_notes[] = {
/* preset, note */
	{PAPI_FP_OPS, NULL},
	{0, NULL}
};

const hwi_dev_notes_t _p4_VEC_MMX_dev_notes[] = {
/* preset, note */
	{PAPI_VEC_INS,
	 "PAPI_VEC_INS counts retired 64bit and 128bit MMX uops tagged with 2."},
	{0, NULL}
};

const hwi_dev_notes_t _p4_VEC_SSE_dev_notes[] = {
/* preset, note */
	{PAPI_VEC_INS,
	 "PAPI_VEC_INS counts retired packed single and double precision SSE uops tagged with 2."},
	{0, NULL}
};


/*
  Pentium 4 supports 3 separate native event tables, each an array of 
  hwd_p4_native_map_t structures. 
  - The first is a user table consisting of up to 128 user definable entries.
  The count of entries is also maintained. The mechanism for modifying this
  table has not yet been fully elucidated.
  - The second table is a custom table of entries defined for specific purposes.
  Cycles is an example of an exception that falls into this table. Other special
  cases e.g., using PEBS or tags, can also be envisioned for this table.
  - The third table is actually a virtual table. It consists of entries for the
  45 or so event groups defined by Intel, with a portion of the index bits used
  to specific mask bits that modify the event group. _p4_ntv_code_to_bits()
  dynamically uses this information to assemble the event structure at run-time.
*/

/* **THREAD SAFE STATIC**
   The entry count and table below are both declared static and initialized to zero. 
   This is thread- and fork-safe, because inherited values will still be valid. 
   Also modifications made in the thread or forked copies will stay local to those copies.
*/
static int _p4_user_count = 0;
static hwd_p4_native_map_t _p4_user_map[128] = { {0,}, };

/* **THREAD SAFE STATIC** constant preinitialized structure */
static hwd_p4_native_map_t _p4_custom_map[] = {
// following are custom defined events that don't fit the normal structure
	{
	 /* cycles is a special case in that an ESCR is not
	    needed at all. By configuring the threshold comparison appropriately
	    in a CCCR, you can get the counter to count every cycle, independent
	    of whatever ESCR the CCCR happens to be listening to.  To do this, set
	    the COMPARE and COMPLEMENT bits in the CCCR and set the THRESHOLD
	    value to "1111" (binary).  This works because the setting the
	    COMPLEMENT bit makes the threshold comparison to be "less than or
	    equal" and, with THRESHOLD set to its maximum value, the comparison
	    will always succeed and the counter will increment by one on every
	    clock cycle. */
	 /* Cycles can live on ANY counter and use ANY ESCR, but it must be a valid pairing.
	    For now, we'll code it using the TC_deliver_mode group, because that one
	    doesn't seem to be used for much else...
	    This is a good candidate for defining synonyms for use with other groups...
	  */
	 "cycles",
	 "This event counts every cycle by setting the threshold of the TC_deliver_mode event to count every tick",
	 {
	  {CTR45, CTR67}, {MSR_TC_ESCR0, MSR_TC_ESCR1},
	  CCCR_ESCR_SEL( TC_DLVR_CCCR ) | CCCR_THR_MODE( CCCR_THR_ANY ) |
	  CCCR_ENABLE | CCCR_COMPARE | CCCR_COMPLEMENT | CCCR_THRESHOLD( 0xf ),
	  ESCR_EVENT_SEL( TC_DLVR_ESCR ),
	  0, 0, 0},
	 0,
	 0},
};

/* **THREAD SAFE STATIC** constant preinitialized structure */
static hwd_p4_native_map_t _p4_native_map[] = {
// following are the non-retirement events
	{
	 "TC_deliver_mode",
	 "This event counts the duration (in clock cycles) of the operating modes of the trace cache and decode engine in the processor package. The mode is specified by one or more of the event mask bits",
	 {
	  {CTR45, CTR67}, {MSR_TC_ESCR0, MSR_TC_ESCR1},
	  CCCR_ESCR_SEL( TC_DLVR_CCCR ), ESCR_EVENT_SEL( TC_DLVR_ESCR ),
	  0, 0, 0},
	 ( 1 << DD ) | ( 1 << DB ) | ( 1 << DI ) | ( 1 << BD ) | ( 1 << BB ) | ( 1
																			 <<
																			 BI )
	 | ( 1 << ID ) | ( 1 << IB ),
	 0},
	{
	 "BPU_fetch_request",
	 "This event counts instruction fetch requests of specified request type by the Branch Prediction unit. Specify one or more mask bits to qualify the request type(s)",
	 {
	  {CTR01, CTR23}, {MSR_BPU_ESCR0, MSR_BPU_ESCR1},
	  CCCR_ESCR_SEL( BPU_FETCH_RQST_CCCR ),
	  ESCR_EVENT_SEL( BPU_FETCH_RQST_ESCR ),
	  0, 0, 0},
	 ( 1 << TCMISS ),
	 0},
	{
	 "ITLB_reference",
	 "This event counts translations using the Instruction Translation Look-aside Buffer (ITLB)",
	 {
	  {CTR01, CTR23}, {MSR_ITLB_ESCR0, MSR_ITLB_ESCR1},
	  CCCR_ESCR_SEL( ITLB_REF_CCCR ), ESCR_EVENT_SEL( ITLB_REF_ESCR ),
	  0, 0, 0},
	 ( 1 << HIT ) | ( 1 << MISS ) | ( 1 << HIT_UC ),
	 0},
	{
	 "memory_cancel",
	 "This event counts the canceling of various type of request in the Data cache Address Control unit (DAC). Specify one or more mask bits to select the type of requests that are canceled",
	 {
	  {CTR89, CTR1011}, {MSR_DAC_ESCR0, MSR_DAC_ESCR1},
	  CCCR_ESCR_SEL( MEM_CANCEL_CCCR ), ESCR_EVENT_SEL( MEM_CANCEL_ESCR ),
	  0, 0, 0},
	 ( 1 << ST_RB_FULL ) | ( 1 << CONF_64K ),
	 0},
	{
	 "memory_complete",
	 "This event counts the completion of a load split, store split, uncacheable (UC) split, or UC load. Specify one or more mask bits to select the operations to be counted",
	 {
	  {CTR89, CTR1011}, {MSR_SAAT_ESCR0, MSR_SAAT_ESCR1},
	  CCCR_ESCR_SEL( MEM_CANCEL_CCCR ), ESCR_EVENT_SEL( MEM_CANCEL_ESCR ),
	  0, 0, 0},
	 ( 1 << LSC ) | ( 1 << SSC ),
	 0},
	{
	 "load_port_replay",
	 "This event counts replayed events at the load port. Specify one or more mask bits to select the cause of the replay",
	 {
	  {CTR89, CTR1011}, {MSR_SAAT_ESCR0, MSR_SAAT_ESCR1},
	  CCCR_ESCR_SEL( LDPRT_RPL_CCCR ), ESCR_EVENT_SEL( LDPRT_RPL_ESCR ),
	  0, 0, 0},
	 ( 1 << SPLIT_LD ),
	 0},
	{
	 "store_port_replay",
	 "This event counts replayed events at the store port. Specify one or more mask bits to select the cause of the replay",
	 {
	  {CTR89, CTR1011}, {MSR_SAAT_ESCR0, MSR_SAAT_ESCR1},
	  CCCR_ESCR_SEL( SRPRT_RPL_CCCR ), ESCR_EVENT_SEL( SRPRT_RPL_ESCR ),
	  0, 0, 0},
	 ( 1 << SPLIT_ST ),
	 0},
	{
	 "MOB_load_replay",
	 "This event triggers if the memory order buffer (MOB) caused a load operation to be replayed. Specify one or more mask bits to select the cause of the replay",
	 {
	  {CTR01, CTR23}, {MSR_MOB_ESCR0, MSR_MOB_ESCR1},
	  CCCR_ESCR_SEL( MOB_LD_RPL_CCCR ), ESCR_EVENT_SEL( MOB_LD_RPL_ESCR ),
	  0, 0, 0},
	 ( 1 << NO_STA ) | ( 1 << NO_STD ) | ( 1 << PARTIAL_DATA ) | ( 1 <<
																   UNALGN_ADDR ),
	 0},
	{
	 "page_walk_type",
	 "This event counts various types of page walks that the page miss handler (PMH) performs",
	 {
	  {CTR01, CTR23}, {MSR_PMH_ESCR0, MSR_PMH_ESCR1},
	  CCCR_ESCR_SEL( PG_WLK_CCCR ), ESCR_EVENT_SEL( PG_WLK_ESCR ),
	  0, 0, 0},
	 ( 1 << DTMISS ) | ( 1 << ITMISS ),
	 0},
	{
	 "BSQ_cache_reference",
	 "This event counts cache references (2nd level cache or 3rd level cache) as seen by the bus unit. Specify one or more mask bit to select an access according to the access type and the access result. Currently this event causes both over and undercounting by as much as a factor of two due to an erratum.",
	 {
	  {CTR01, CTR23}, {MSR_BSU_ESCR0, MSR_BSU_ESCR1},
	  CCCR_ESCR_SEL( BSQ_CREF_CCCR ), ESCR_EVENT_SEL( BSQ_CREF_ESCR ),
	  0, 0, 0},
	 ( 1 << RD_2ndL_HITS ) | ( 1 << RD_2ndL_HITE ) | ( 1 << RD_2ndL_HITM ) | ( 1
																			   <<
																			   RD_3rdL_HITS )
	 | ( 1 << RD_3rdL_HITE ) | ( 1 << RD_3rdL_HITM ) | ( 1 << RD_2ndL_MISS ) |
	 ( 1 << RD_3rdL_MISS ) | ( 1 << WR_2ndL_MISS ),
	 0},
	{
	 "IOQ_allocation",
	 "This event counts the various types of transactions on the bus. A count is generated each time a transaction is allocated into the IOQ that matches the specified mask bits",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( IOQ_ALLOC_CCCR ), ESCR_EVENT_SEL( IOQ_ALLOC_ESCR ),
	  0, 0, 0},
	 ( 1 << BUS_RQ_TYP0 ) | ( 1 << BUS_RQ_TYP1 ) | ( 1 << BUS_RQ_TYP2 ) | ( 1 <<
																			BUS_RQ_TYP3 )
	 | ( 1 << BUS_RQ_TYP4 ) | ( 1 << ALL_READ ) | ( 1 << ALL_WRITE ) | ( 1 <<
																		 MEM_UC )
	 | ( 1 << MEM_WC ) | ( 1 << MEM_WT ) | ( 1 << MEM_WP ) | ( 1 << MEM_WB ) |
	 ( 1 << OWN ) | ( 1 << OTHER ) | ( 1 << PREFETCH ),
	 0},
	{
	 "IOQ_active_entries",
	 "This event counts the number of entries (clipped at 15) in the IOQ that are active. This event must be programmed in conjunction with IOQ_allocation",
	 {
	  {CTR23, CTR23}, {MSR_FSB_ESCR1, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( IOQ_ACTV_ENTR_CCCR ), ESCR_EVENT_SEL( IOQ_ACTV_ENTR_ESCR ),
	  0, 0, 0},
	 ( 1 << BUS_RQ_TYP0 ) | ( 1 << BUS_RQ_TYP1 ) | ( 1 << BUS_RQ_TYP2 ) | ( 1 <<
																			BUS_RQ_TYP3 )
	 | ( 1 << BUS_RQ_TYP4 ) | ( 1 << ALL_READ ) | ( 1 << ALL_WRITE ) | ( 1 <<
																		 MEM_UC )
	 | ( 1 << MEM_WC ) | ( 1 << MEM_WT ) | ( 1 << MEM_WP ) | ( 1 << MEM_WB ) |
	 ( 1 << OWN ) | ( 1 << OTHER ) | ( 1 << PREFETCH ),
	 0},
	{
	 "FSB_data_activity",
	 "This event increments once for each DRDY or DBSY event that occurs on the front side bus. The event allows selection of a specific DRDY or DBSY event",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( FSB_DATA_CCCR ), ESCR_EVENT_SEL( FSB_DATA_ESCR ),
	  0, 0, 0},
	 ( 1 << DRDY_DRV ) | ( 1 << DRDY_OWN ) | ( 1 << DRDY_OTHER ) | ( 1 <<
																	 DBSY_DRV )
	 | ( 1 << DBSY_OWN ) | ( 1 << DBSY_OTHER ),
	 0},
	{
	 "BSQ_allocation",
	 "This event counts allocations in the Bus Sequence Unit (BSQ) according to the specified mask bit encoding",
	 {
	  {CTR01, CTR01}, {MSR_BSU_ESCR0, MSR_BSU_ESCR0},
	  CCCR_ESCR_SEL( BSQ_ALLOC_CCCR ), ESCR_EVENT_SEL( BSQ_ALLOC_ESCR ),
	  0, 0, 0},
	 ( 1 << REQ_TYPE0 ) | ( 1 << REQ_TYPE1 ) | ( 1 << REQ_LEN0 ) | ( 1 <<
																	 REQ_LEN1 )
	 | ( 1 << REQ_IO_TYPE )
	 | ( 1 << REQ_LOCK_TYPE ) | ( 1 << REQ_CACHE_TYPE ) | ( 1 <<
															REQ_SPLIT_TYPE ) |
	 ( 1 << REQ_DEM_TYPE )
	 | ( 1 << REQ_ORD_TYPE ) | ( 1 << MEM_TYPE0 ) | ( 1 << MEM_TYPE1 ) | ( 1 <<
																		   MEM_TYPE2 ),
	 0},
	{
	 "bsq_active_entries",
	 "This event represents the number of BSQ entries (clipped at 15) currently active (valid) which meet the subevent mask criteria during allocation in the BSQ",
	 {
	  {CTR23, CTR23}, {MSR_BSU_ESCR1, MSR_BSU_ESCR1},
	  CCCR_ESCR_SEL( BSQ_ACTV_ENTR_CCCR ), ESCR_EVENT_SEL( BSQ_ACTV_ENTR_ESCR ),
	  0, 0, 0},
	 ( 1 << REQ_TYPE0 ) | ( 1 << REQ_TYPE1 ) | ( 1 << REQ_LEN0 ) | ( 1 <<
																	 REQ_LEN1 )
	 | ( 1 << REQ_IO_TYPE )
	 | ( 1 << REQ_LOCK_TYPE ) | ( 1 << REQ_CACHE_TYPE ) | ( 1 <<
															REQ_SPLIT_TYPE ) |
	 ( 1 << REQ_DEM_TYPE )
	 | ( 1 << REQ_ORD_TYPE ) | ( 1 << MEM_TYPE0 ) | ( 1 << MEM_TYPE1 ) | ( 1 <<
																		   MEM_TYPE2 ),
	 0},
	{
	 "SSE_input_assist",
	 "This event counts the number of times an assist is requested to handle problems with input operands for SSE and SSE2 operations",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( SSE_ASSIST_CCCR ), ESCR_EVENT_SEL( SSE_ASSIST_ESCR ),
	  0, 0, 0},
	 ( 1 << ALL ),
	 0},
	{
	 "packed_SP_uop",
	 "This event increments for each packed single-precision uop, specified through the event mask for detection",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( PACKED_SP_UOP_CCCR ), ESCR_EVENT_SEL( PACKED_SP_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "packed_DP_uop",
	 "This event increments for each packed double-precision uop, specified through the event mask for detection",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( PACKED_DP_UOP_CCCR ), ESCR_EVENT_SEL( PACKED_DP_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "scalar_SP_uop",
	 "This event increments for each scalar single-precision uop, specified through the event mask for detection",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( SCALAR_SP_UOP_CCCR ), ESCR_EVENT_SEL( SCALAR_SP_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "scalar_DP_uop",
	 "This event increments for each scalar double-precision uop, specified through the event mask for detection",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( SCALAR_DP_UOP_CCCR ), ESCR_EVENT_SEL( SCALAR_DP_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "64bit_MMX_uop",
	 "This event increments for each MMX instruction, which operate on 64 bit SIMD operands",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( MMX_64_UOP_CCCR ), ESCR_EVENT_SEL( MMX_64_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "128bit_MMX_uop",
	 "This event increments for each integer SIMD SSE2 instructions, which operate on 128 bit SIMD operands",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( MMX_128_UOP_CCCR ), ESCR_EVENT_SEL( MMX_128_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "x87_FP_uop",
	 "This event increments for each x87 floating-point uop, specified through the event mask for detection",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( X87_FP_UOP_CCCR ), ESCR_EVENT_SEL( X87_FP_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 << TAG2 ) | ( 1 << TAG3 ) | ( 1 <<
																	   ALL ),
	 0},
	{
	 "x87_SIMD_moves_uop",
	 "This event increments for each x87 FPU, MMX, SSE or SSE2 uop related to load data, store data, or register-to-register moves",
	 {
	  {CTR89, CTR1011}, {MSR_FIRM_ESCR0, MSR_FIRM_ESCR1},
	  CCCR_ESCR_SEL( X87_SIMD_UOP_CCCR ), ESCR_EVENT_SEL( X87_SIMD_UOP_ESCR ),
	  0, 0, 0},
	 ( 1 << ALLP0 ) | ( 1 << ALLP2 ) | ( 1 << TAG0 ) | ( 1 << TAG1 ) | ( 1 <<
																		 TAG2 )
	 | ( 1 << TAG3 ),
	 0},
	{
	 "global_power_events",
	 "This event accumulates the time during which a processor is not stopped",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( GLOBAL_PWR_CCCR ), ESCR_EVENT_SEL( GLOBAL_PWR_ESCR ),
	  0, 0, 0},
	 ( 1 << RUNNING ),
	 0},
	{
	 "tc_ms_xfer",
	 "This event counts the number of times that uop delivery changed from TC to MS ROM",
	 {
	  {CTR45, CTR67}, {MSR_MS_ESCR0, MSR_MS_ESCR1},
	  CCCR_ESCR_SEL( TC_MS_XFER_CCCR ), ESCR_EVENT_SEL( TC_MS_XFER_ESCR ),
	  0, 0, 0},
	 ( 1 << CISC ),
	 0},
	{
	 "uop_queue_writes",
	 "This event counts the number of valid uops written to the uop queue. Specify one or more mask bits to select the source type of writes",
	 {
	  {CTR45, CTR67}, {MSR_MS_ESCR0, MSR_MS_ESCR1},
	  CCCR_ESCR_SEL( UOP_QUEUE_WRITES_CCCR ),
	  ESCR_EVENT_SEL( UOP_QUEUE_WRITES_ESCR ),
	  0, 0, 0},
	 ( 1 << FROM_TC_BUILD ) | ( 1 << FROM_TC_DELIVER ) | ( 1 << FROM_ROM ),
	 0},
	{
	 "retired_mispred_branch_type",
	 "This event counts retiring mispredicted branches by type",
	 {
	  {CTR45, CTR67}, {MSR_TBPU_ESCR0, MSR_TBPU_ESCR1},
	  CCCR_ESCR_SEL( RET_MISPRED_BR_TYPE_CCCR ),
	  ESCR_EVENT_SEL( RET_MISPRED_BR_TYPE_ESCR ),
	  0, 0, 0},
	 ( 1 << CONDITIONAL ) | ( 1 << CALL ) | ( 1 << RETURN ) | ( 1 << INDIRECT ),
	 0},
	{
	 "retired_branch_type",
	 "This event counts retiring branches by type. Specify one or more mask bits to qualify the branch by its type",
	 {
	  {CTR45, CTR67}, {MSR_TBPU_ESCR0, MSR_TBPU_ESCR1},
	  CCCR_ESCR_SEL( RET_BR_TYPE_CCCR ), ESCR_EVENT_SEL( RET_BR_TYPE_ESCR ),
	  0, 0, 0},
	 ( 1 << CONDITIONAL ) | ( 1 << CALL ) | ( 1 << RETURN ) | ( 1 << INDIRECT ),
	 0},
	{
	 // "This event may not be supported in all models of the processor family"
	 "resource_stall",
	 "This event monitors the occurrence or latency of stalls in the Allocator. It may not be supported in all models of the processor family",
	 {
	  {CTR236, CTR457}, {MSR_ALF_ESCR0, MSR_ALF_ESCR1},
	  CCCR_ESCR_SEL( RESOURCE_STALL_CCCR ),
	  ESCR_EVENT_SEL( RESOURCE_STALL_ESCR ),
	  0, 0, 0},
	 ( 1 << SBFULL ),
	 0},
	{
	 "WC_Buffer",
	 "This event counts Write Combining Buffer operations that are selected by the event mask",
	 {
	  {CTR89, CTR1011}, {MSR_DAC_ESCR0, MSR_DAC_ESCR1},
	  CCCR_ESCR_SEL( WC_BUFFER_CCCR ), ESCR_EVENT_SEL( WC_BUFFER_ESCR ),
	  0, 0, 0},
	 ( 1 << WCB_EVICTS ) | ( 1 << WCB_FULL_EVICT ),
	 0},
	{
	 // "This event may not be supported in all models of the processor family"
	 "b2b_cycles",
	 "This event can be configured to count the number back-to-back bus cycles using sub-event mask bits 1 through 6",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( B2B_CYCLES_CCCR ), ESCR_EVENT_SEL( B2B_CYCLES_ESCR ),
	  0, 0, 0},
	 // The documentation suggests that mask bits can be used, but none are defined.
	 0, 0},
	{
	 // "This event may not be supported in all models of the processor family"
	 "bnr",
	 "This event can be configured to count bus not ready conditions using sub-event mask bits 0 through 2",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( BNR_CCCR ), ESCR_EVENT_SEL( BNR_ESCR ),
	  0, 0, 0},
	 // The documentation suggests that mask bits can be used, but none are defined.
	 0, 0},
	{
	 // "This event may not be supported in all models of the processor family"
	 "snoop",
	 "This event can be configured to count snoop hit modified bus traffic using sub-event mask bits 2, 6 and 7",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( SNOOP_CCCR ),
	  ESCR_EVENT_SEL( SNOOP_ESCR ),
	  0, 0, 0},
	 // The documentation suggests that mask bits can be used, but none are defined.
	 0, 0},
	{
	 // "This event may not be supported in all models of the processor family"
	 "response",
	 "This event can be configured to count different types of responses using sub-event mask bits 1,2, 8, and 9",
	 {
	  {CTR01, CTR23}, {MSR_FSB_ESCR0, MSR_FSB_ESCR1},
	  CCCR_ESCR_SEL( RESPONSE_CCCR ), ESCR_EVENT_SEL( RESPONSE_ESCR ),
	  0, 0, 0},
	 // The documentation suggests that mask bits can be used, but none are defined.
	 0, 0},

// following are the at-retirement events
	{
	 "front_end_event",
	 "This event counts the retirement of tagged uops, which are specified through the front-end tagging mechanism. The event mask specifies bogus or non-bogus uops",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( FRONT_END_CCCR ), ESCR_EVENT_SEL( FRONT_END_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUS ) | ( 1 << BOGUS ),
	 0},
	{
	 "execution_event",
	 "This event counts the retirement of tagged uops, which are specified through the execution tagging mechanism. The event mask allows from one to four types of uops to be specified as either bogus or non-bogus uops to be tagged",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( EXECUTION_CCCR ), ESCR_EVENT_SEL( EXECUTION_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUS0 ) | ( 1 << NBOGUS1 ) | ( 1 << NBOGUS2 ) | ( 1 << NBOGUS3 ) |
	 ( 1 << BOGUS0 ) | ( 1 << BOGUS1 ) | ( 1 << BOGUS2 ) | ( 1 << BOGUS3 ),
	 0},
	{
	 "replay_event",
	 "This event counts the retirement of tagged uops, which are specified through the replay tagging mechanism. The event mask specifies bogus or non-bogus uops",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( REPLAY_CCCR ), ESCR_EVENT_SEL( REPLAY_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUS ) | ( 1 << BOGUS ) | ( 1 << PEBS_MV_LOAD_BIT ) | ( 1 <<
																	  PEBS_MV_STORE_BIT )
	 | ( 1 << PEBS_L1_MISS_BIT ) | ( 1 << PEBS_L2_MISS_BIT ) | ( 1 <<
																 PEBS_DTLB_MISS_BIT )
	 | ( 1 << PEBS_MOB_BIT ) | ( 1 << PEBS_SPLIT_BIT ),
	 0},
	{
	 "instr_retired",
	 "This event counts instructions that are retired during a clock cycle. Mask bits specify bogus or non-bogus (and whether they are tagged via the front-end tagging mechanism",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR0, MSR_CRU_ESCR1},
	  CCCR_ESCR_SEL( INSTR_RET_CCCR ), ESCR_EVENT_SEL( INSTR_RET_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUSNTAG ) | ( 1 << NBOGUSTAG ) | ( 1 << BOGUSNTAG ) | ( 1 <<
																	   BOGUSTAG ),
	 0},
	{
	 "uops_retired",
	 "This event counts uops that are retired during a clock cycle. Mask bits specify bogus or non-bogus",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR0, MSR_CRU_ESCR1},
	  CCCR_ESCR_SEL( UOPS_RET_CCCR ), ESCR_EVENT_SEL( UOPS_RET_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUS ) | ( 1 << BOGUS ),
	 0},
	{
	 "uop_type",
	 "This event is used in conjunction with the front-end at-retirement mechanism to tag load and store uops",
	 {
	  {CTR236, CTR457}, {MSR_RAT_ESCR0, MSR_RAT_ESCR1},
	  CCCR_ESCR_SEL( UOP_TYPE_CCCR ), ESCR_EVENT_SEL( UOP_TYPE_ESCR ),
	  0, 0, 0},
	 ( 1 << TAGLOADS ) | ( 1 << TAGSTORES ),
	 0},
	{
	 "branch_retired",
	 "This event counts the retirement of a branch. Specify one or more mask bits to select any combination of taken, not-taken, predicted and mispredicted",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( BR_RET_CCCR ), ESCR_EVENT_SEL( BR_RET_ESCR ),
	  0, 0, 0},
	 ( 1 << MMNP ) | ( 1 << MMNM ) | ( 1 << MMTP ) | ( 1 << MMTM ),
	 0},
	{
	 "mispred_branch_retired",
	 "This event represents the retirement of mispredicted IA-32 branch instructions",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR0, MSR_CRU_ESCR1},
	  CCCR_ESCR_SEL( MPR_BR_RET_CCCR ), ESCR_EVENT_SEL( MPR_BR_RET_ESCR ),
	  0, 0, 0},
	 ( 1 << NBOGUS ),
	 0},
	{
	 "x87_assist",
	 "This event counts the retirement of x87 instructions that required special handling. Specifies one or more event mask bits to select the type of assistance",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( X87_ASSIST_CCCR ), ESCR_EVENT_SEL( X87_ASSIST_ESCR ),
	  0, 0, 0},
	 ( 1 << FPSU ) | ( 1 << FPSO ) | ( 1 << POAO ) | ( 1 << POAU ) | ( 1 <<
																	   PREA ),
	 0},
	{
	 "machine_clear",
	 "This event increments according to the mask bit specified while the entire pipeline of the machine is cleared. Specify one of the mask bit to select the cause",
	 {
	  {CTR236, CTR457}, {MSR_CRU_ESCR2, MSR_CRU_ESCR3},
	  CCCR_ESCR_SEL( MACHINE_CLEAR_CCCR ), ESCR_EVENT_SEL( MACHINE_CLEAR_ESCR ),
	  0, 0, 0},
	 ( 1 << CLEAR ) | ( 1 << MOCLEAR ) | ( 1 << SMCLEAR ),
	 0},
};

hwd_p4_mask_t _TC_deliver_mode_mask[] = {
	{DD, "DD", "Both logical processors are in deliver mode"},
	{DB, "DB", "LP0 is in deliver mode and LP1 is in build mode"},
	{DI, "DI", "LP0 is in deliver mode and LP1 is halted"},
	{BD, "BD", "LP0 is in build mode and LP1 is in deliver mode"},
	{BB, "BB", "Both logical processors are in build mode"},
	{BI, "BI", "LP0 is in build mode and LP1 is halted"},
	{ID, "ID", "LP0 is halted and LP1 is in deliver mode"},
	{IB, "IB", "LP0 is halted and LP1 is in build mode"},
	{-1, NULL, NULL}
};


hwd_p4_mask_t _BPU_fetch_request_mask[] = {
	{TCMISS, "TCMISS", "Trace cache lookup miss"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _ITLB_reference_mask[] = {
	{HIT, "HIT", "ITLB hit"},
	{MISS, "MISS", "ITLB miss"},
	{HIT_UC, "HIT_UC", "Uncacheable ITLB hit"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _memory_cancel_mask[] = {
	{ST_RB_FULL, "ST_RB_FULL",
	 "Replayed because no store request buffer is available"},
	{CONF_64K, "CONF_64K", "Conflicts due to 64K aliasing"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _memory_complete_mask[] = {
	{LSC, "LSC", "Load split completed, excluding UC/WC loads"},
	{SSC, "SSC", "Any split stores completed"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _load_port_replay_mask[] = {
	{SPLIT_LD, "SPLIT_LD", "Split load"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _store_port_replay_mask[] = {
	{SPLIT_ST, "SPLIT_ST", "Split store"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _MOB_load_replay_mask[] = {
	{NO_STA, "NO_STA", "Replayed because of unknown store address"},
	{NO_STD, "NO_STD", "Replayed because of unknown store data"},
	{PARTIAL_DATA, "PARTIAL_DATA",
	 "Replayed because of partially overlapped data access between the load and store operations"},
	{UNALGN_ADDR, "UNALGN_ADDR",
	 "Replayed because the lower 4 bits of the linear address do not match between the load and store operations"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _page_walk_type_mask[] = {
	{DTMISS, "DTMISS", "Page walk for a data TLB miss (either load or store)"},
	{ITMISS, "ITMISS", "Page walk for an instruction TLB miss"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _BSQ_cache_reference_mask[] = {
	{RD_2ndL_HITS, "RD_2ndL_HITS", "Read L2 cache hit Shared"},
	{RD_2ndL_HITE, "RD_2ndL_HITE", "Read L2 cache hit Exclusive"},
	{RD_2ndL_HITM, "RD_2ndL_HITM", "Read L2 cache hit Modified"},
	{RD_3rdL_HITS, "RD_3rdL_HITS", "Read L3 cache hit Shared"},
	{RD_3rdL_HITE, "RD_3rdL_HITE", "Read L3 cache hit Exclusive"},
	{RD_3rdL_HITM, "RD_3rdL_HITM", "Read L3 cache hit Modified"},
	{RD_2ndL_MISS, "RD_2ndL_MISS", "Read L2 cache miss"},
	{RD_3rdL_MISS, "RD_3rdL_MISS", "Read L3 cache miss"},
	{WR_2ndL_MISS, "WR_2ndL_MISS",
	 "A Writeback lookup from DAC misses the L2 cache"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _IOQ_allocation_mask[] = {
	{BUS_RQ_TYP0, "BUS_RQ_TYP0", "Bus request type (5 bit field)"},
	{BUS_RQ_TYP1, "BUS_RQ_TYP1", "Bus request type (5 bit field)"},
	{BUS_RQ_TYP2, "BUS_RQ_TYP2", "Bus request type (5 bit field)"},
	{BUS_RQ_TYP3, "BUS_RQ_TYP3", "Bus request type (5 bit field)"},
	{BUS_RQ_TYP4, "BUS_RQ_TYP4", "Bus request type (5 bit field)"},
	{ALL_READ, "ALL_READ", "Count read entries"},
	{ALL_WRITE, "ALL_WRITE", "Count write entries"},
	{MEM_UC, "MEM_UC", "UC memory access entries"},
	{MEM_WC, "MEM_WC", "WC memory access entries"},
	{MEM_WT, "MEM_WT", "Count write-through (WT) memory access entries"},
	{MEM_WP, "MEM_WP", "Count write-protected (WP) memory access entries"},
	{MEM_WB, "MEM_WB", "Count WB memory access entries"},
	{OWN, "OWN", "Count all store requests driven by processor"},
	{OTHER, "OTHER", "Count all requests driven by other processors or DMA"},
	{PREFETCH, "PREFETCH", "Include HW and SW prefetch requests in the count"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _FSB_data_activity_mask[] = {
	{DRDY_DRV, "DRDY_DRV",
	 "Count when this processor drives data onto the bus"},
	{DRDY_OWN, "DRDY_OWN", "Count when this processor reads data from the bus"},
	{DRDY_OTHER, "DRDY_OTHER",
	 "Count when data is on the bus but not being sampled by the processor"},
	{DBSY_DRV, "DBSY_DRV",
	 "Count when this processor reserves the bus for use in the next bus cycle in order to drive data"},
	{DBSY_OWN, "DBSY_OWN",
	 "Count when some agent reserves the bus for use in the next bus cycle to drive data that this processor will sample"},
	{DBSY_OTHER, "DBSY_OTHER",
	 "Count when some agent reserves the bus to drive data that this processor will NOT sample"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _BSQ_allocation_mask[] = {
	{REQ_TYPE0, "REQ_TYPE0",
	 "Request type: 0 - Read; 1 - Read invalidate; 2 - Write; 3 - Writeback"},
	{REQ_TYPE1, "REQ_TYPE1",
	 "Request type: 0 - Read; 1 - Read invalidate; 2 - Write; 3 - Writeback"},
	{REQ_LEN0, "REQ_LEN0",
	 "Request length: 0 - 0 chunks, 1 - 1 chunk, 3 - 8 chunks"},
	{REQ_LEN1, "REQ_LEN1",
	 "Request length: 0 - 0 chunks, 1 - 1 chunk, 3 - 8 chunks"},
	{REQ_IO_TYPE, "REQ_IO_TYPE", "Request type is input or output"},
	{REQ_LOCK_TYPE, "REQ_LOCK_TYPE", "Request type is bus lock"},
	{REQ_CACHE_TYPE, "REQ_CACHE_TYPE", "Request type is cacheable"},
	{REQ_SPLIT_TYPE, "REQ_SPLIT_TYPE",
	 "Request type is a bus 8-byte chunk split across 8-byte boundary"},
	{REQ_DEM_TYPE, "REQ_DEM_TYPE",
	 "Request type is a demand if set, or HW.SW prefetch if 0"},
	{REQ_ORD_TYPE, "REQ_ORD_TYPE", "Request is an ordered type"},
	{MEM_TYPE0, "MEM_TYPE0",
	 "Memory type: 0 - UC, 1 - USWC, 4 - WT, 5 - WP, 6 - WB"},
	{MEM_TYPE1, "MEM_TYPE1",
	 "Memory type: 0 - UC, 1 - USWC, 4 - WT, 5 - WP, 6 - WB"},
	{MEM_TYPE2, "MEM_TYPE2",
	 "Memory type: 0 - UC, 1 - USWC, 4 - WT, 5 - WP, 6 - WB"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _SSE_input_assist_mask[] = {
	{ALL, "ALL", "Count all uops of this type"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _replay_tag_mask[] = {
	{TAG0, "TAG0", "Tag all uops with bit 0"},
	{TAG1, "TAG1", "Tag all uops with bit 1"},
	{TAG2, "TAG2", "Tag all uops with bit 2"},
	{TAG3, "TAG3", "Tag all uops with bit 3"},
	{ALL, "ALL", "Count all uops of this type"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _x87_SIMD_moves_uop_mask[] = {
	{ALLP0, "ALLP0", "Count all x87/SIMD store/moves uops"},
	{ALLP2, "ALLP2", "Count all x87/SIMD load uops"},
	{TAG0, "TAG0", "Tag all uops with bit 0"},
	{TAG1, "TAG1", "Tag all uops with bit 1"},
	{TAG2, "TAG2", "Tag all uops with bit 2"},
	{TAG3, "TAG3", "Tag all uops with bit 3"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _global_power_events_mask[] = {
	{RUNNING, "RUNNING", "The processor is active"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _tc_ms_xfer_mask[] = {
	{CISC, "CISC", "A TC to MS transfer occurred"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _uop_queue_writes_mask[] = {
	{FROM_TC_BUILD, "FROM_TC_BUILD",
	 "The uops being written are from TC build mode"},
	{FROM_TC_DELIVER, "FROM_TC_DELIVER",
	 "The uops being written are from TC deliver mode"},
	{FROM_ROM, "FROM_ROM", "The uops being written are from microcode ROM"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _retired_branch_mask[] = {
	{CONDITIONAL, "CONDITIONAL", "Conditional jumps"},
	{CALL, "CALL", "Indirect call branches"},
	{RETURN, "RETURN", "Return branches"},
	{INDIRECT, "INDIRECT", "Returns, indirect calls, or indirect jumps"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _resource_stall_mask[] = {
	{SBFULL, "SBFULL", "A Stall due to lack of store buffers"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _WC_Buffer_mask[] = {
	{WCB_EVICTS, "WCB_EVICTS", "WC Buffer evictions of all causes"},
	{WCB_FULL_EVICT, "WCB_FULL_EVICT",
	 "WC Buffer eviction: no WC buffer is available"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _b2b_cycles_mask[] = {
	{WCB_EVICTS, "WCB_EVICTS", "WC Buffer evictions of all causes"},
	{WCB_FULL_EVICT, "WCB_FULL_EVICT",
	 "WC Buffer eviction: no WC buffer is available"},
	{-1, NULL, NULL}
};


hwd_p4_mask_t _execution_event_mask[] = {
	{NBOGUS0, "NBOGUS0", "Tag 0 uops are not bogus"},
	{NBOGUS1, "NBOGUS1", "Tag 1 marked uops are not bogus"},
	{NBOGUS2, "NBOGUS2", "Tag 2 marked uops are not bogus"},
	{NBOGUS3, "NBOGUS3", "Tag 3 marked uops are not bogus"},
	{BOGUS0, "BOGUS0", "Tag 0 marked uops are bogus"},
	{BOGUS1, "BOGUS1", "Tag 1 marked uops are bogus"},
	{BOGUS2, "BOGUS2", "Tag 2 marked uops are bogus"},
	{BOGUS3, "BOGUS3", "Tag 3 marked uops are bogus"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _instr_retired_mask[] = {
	{NBOGUSNTAG, "NBOGUSNTAG", "Non-bogus instructions that are not tagged"},
	{NBOGUSTAG, "NBOGUSTAG", "Non-bogus instructions that are tagged"},
	{BOGUSNTAG, "BOGUSNTAG", "Bogus instructions that are not tagged"},
	{BOGUSTAG, "BOGUSTAG", "Bogus instructions that are tagged"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _bogus_mask[] = {
	{NBOGUS, "NBOGUS", "The marked uops are not bogus"},
	{BOGUS, "BOGUS", "The marked uops are bogus"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _replay_event_mask[] = {
	{NBOGUS, "NBOGUS", "The marked uops are not bogus"},
	{BOGUS, "BOGUS", "The marked uops are bogus"},
	{PEBS_MV_LOAD_BIT, "PEBS_MV_LOAD_BIT", "Measure load operations"},
	{PEBS_MV_STORE_BIT, "PEBS_MV_STORE_BIT", "Measure store operations"},
	{PEBS_L1_MISS_BIT, "PEBS_L1_MISS_BIT", "Count L1 cache misses"},
	{PEBS_L2_MISS_BIT, "PEBS_L2_MISS_BIT", "Count L2 cache misses"},
	{PEBS_DTLB_MISS_BIT, "PEBS_DTLB_MISS_BIT", "Count DTLB misses"},
	{PEBS_MOB_BIT, "PEBS_MOB_BIT", "Count MOB replay events"},
	{PEBS_SPLIT_BIT, "PEBS_SPLIT_BIT", "Count split replay events"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _uop_type_mask[] = {
	{TAGLOADS, "TAGLOADS", "The uop is a load operation"},
	{TAGSTORES, "TAGSTORES", "The uop is a store operation"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _branch_retired_mask[] = {
	{MMNP, "MMNP", "Branch Not-taken Predicted"},
	{MMNM, "MMNM", "Branch Not-taken Mispredicted"},
	{MMTP, "MMTP", "Branch Taken Predicted"},
	{MMTM, "MMTM", "Branch Taken Mispredicted"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _mispred_branch_retired_mask[] = {
	{NBOGUS, "NBOGUS", "The retired instruction is not bogus"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _x87_assist_mask[] = {
	{FPSU, "FPSU", "Handle FP stack underflow"},
	{FPSO, "FPSO", "Handle FP stack overflow"},
	{POAO, "POAO", "Handle x87 output overflow"},
	{POAU, "POAU", "Handle x87 output underflow"},
	{PREA, "PREA", "Handle x87 input assist"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t _machine_clear_mask[] = {
	{CLEAR, "CLEAR",
	 "Counts for a portion of the many cycles while the machine is cleared for any cause"},
	{MOCLEAR, "MOCLEAR",
	 "Increments each time the machine is cleared due to memory ordering issues"},
	{SMCLEAR, "SMCLEAR",
	 "Increments each time the machine is cleared due to self-modifying code issues"},
	{-1, NULL, NULL}
};

hwd_p4_mask_t *mask_array[] = {
	_TC_deliver_mode_mask,
	_BPU_fetch_request_mask,
	_ITLB_reference_mask,
	_memory_cancel_mask,
	_memory_complete_mask,
	_load_port_replay_mask,
	_store_port_replay_mask,
	_MOB_load_replay_mask,
	_page_walk_type_mask,
	_BSQ_cache_reference_mask,
	_IOQ_allocation_mask,
	_IOQ_allocation_mask,
	_FSB_data_activity_mask,
	_BSQ_allocation_mask,
	_BSQ_allocation_mask,
	_SSE_input_assist_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_replay_tag_mask,
	_x87_SIMD_moves_uop_mask,
	_global_power_events_mask,
	_tc_ms_xfer_mask,
	_uop_queue_writes_mask,
	_retired_branch_mask,
	_retired_branch_mask,
	_resource_stall_mask,
	_WC_Buffer_mask,
	NULL,					 // b2b_cycles
	NULL,					 // bnr
	NULL,					 // snoop
	NULL,					 // response
	_bogus_mask,
	_execution_event_mask,
	_replay_event_mask,
	_instr_retired_mask,
	_bogus_mask,
	_uop_type_mask,
	_branch_retired_mask,
	_mispred_branch_retired_mask,
	_x87_assist_mask,
	_machine_clear_mask
};

/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

// This defines the number of events in the custom native event table
#define _p4_custom_count (sizeof(_p4_custom_map) / sizeof(hwd_p4_native_map_t))

/* **NOT THREAD SAFE STATIC!!**
   The name and description strings below are both declared static. 
   This is NOT thread-safe, because these values are returned 
     for use by the calling program, and could be trashed by another thread
     before they are used. To prevent this, any call to routines using these
     variables (_p4_code_to_{name,descr}) should be wrapped in 
     _papi_hwi_{lock,unlock} calls.
   They are declared static to reserve non-volatile space for constructed strings.
*/
static char name[128];
static char description[1024];

static inline void
internal_decode_event( unsigned int EventCode, int *event, int *mask )
{
	*event = ( EventCode & PAPI_NATIVE_AND_MASK ) >> 16;	// event is in the third byte
	*mask = ( EventCode & 0xffff );	// mask bits are in the first two bytes
}

int
_p4_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	/* returns the next valid native event code following the one passed in
	   modifier can have different meaning on different platforms 
	   for p4, (modifier==0) scans major groups; (modifier==1) scans all variations of the groups.
	   this can be important for platforms such as pentium 4 that have event groups and variations
	 */

	int event, mask, this_mask;

	internal_decode_event( *EventCode, &event, &mask );

	if ( event <= P4_machine_clear ) {
		switch ( modifier ) {
		case PAPI_ENUM_EVENTS:
			this_mask = _p4_native_map[event].mask;	// valid bits for this mask
			while ( ( ( ++mask ) & this_mask ) != mask ) {
				if ( mask > this_mask ) {
					mask = 0;
					if ( ++event > P4_machine_clear ) {
						event = P4_custom_event;
						mask = -1;
					}
					break;
				}
			}
			break;

		case PAPI_PENT4_ENUM_GROUPS:
			if ( ++event > P4_machine_clear ) {
				event = P4_custom_event;
				mask = -1;
			} else
				mask = 0;
			break;

		case PAPI_PENT4_ENUM_COMBOS:
			this_mask = _p4_native_map[event].mask;	// valid bits for this mask
			while ( ( ( ++mask ) & this_mask ) != mask ) {
				if ( mask > this_mask )
					return ( PAPI_ENOEVNT );
			}
			break;

		case PAPI_PENT4_ENUM_BITS:
			this_mask = _p4_native_map[event].mask;	// valid bits for this mask
			if ( mask == 0 )
				mask = 1;
			else
				mask = mask << 1;
			while ( ( mask & this_mask ) != mask ) {
				mask = mask << 1;
				if ( mask > this_mask )
					return ( PAPI_ENOEVNT );
			}
			break;
		default:
			return ( PAPI_EINVAL );
		}
	}

	if ( event == P4_custom_event ) {
		if ( ++mask >= _p4_custom_count ) {
			event = P4_user_event;
			mask = -1;
		}
	}

	if ( event == P4_user_event ) {
		if ( ++mask >= _p4_user_count )
			return ( PAPI_ENOEVNT );
	}

	*EventCode = ( event << 16 ) + mask + PAPI_NATIVE_MASK;
	return ( PAPI_OK );
}

/* Called by _p4_ntv_code_to_{name,descr}() to build the return string */
static char *
internal_translate_code( int event, int mask, char *str, char *separator )
{
	int i, j;

	if ( *separator == '_' ) // implied flag for name
		strcpy( str, _p4_native_map[event].name );
	else
		strcpy( str, _p4_native_map[event].description );

	// do some sanity checks for valid mask bits
	if ( !mask )
		return ( str );
	if ( ( _p4_native_map[event].mask & mask ) != mask )
		return ( NULL );

	if ( *separator != '_' ) // implied flag for name
		strcat( str, " Mask bits: " );

	for ( i = 0; i < 16 && mask != 0; i++ ) {
		if ( mask & ( 1 << i ) ) {
			mask ^= ( 1 << i );	// turn off the found bit
			for ( j = 0; j < 16; j++ ) {
				if ( mask_array[event][j].bit_pos == i ) {
					strcat( str, separator );
					if ( *separator == '_' )	// implied flag for name
						strcat( str, mask_array[event][j].name );
					else
						strcat( str, mask_array[event][j].description );
					break;
				}
				if ( mask_array[event][j].bit_pos == -1 )
					return ( NULL );
			}
		}
	}
	return ( str );
}

char *
_p4_ntv_code_to_name( unsigned int EventCode )
{
	int event, mask;

	internal_decode_event( EventCode, &event, &mask );

	if ( event == P4_custom_event ) {
		if ( mask >= _p4_custom_count )
			return ( NULL );
		return ( _p4_custom_map[mask].name );
	}

	if ( event == P4_user_event ) {
		if ( mask >= _p4_user_count )
			return ( NULL );
		return ( _p4_user_map[mask].name );
	}

	if ( event > sizeof ( _p4_native_map ) / sizeof ( hwd_p4_native_map_t ) )
		return ( NULL );

	return ( internal_translate_code( event, mask, name, "_" ) );
}

char *
_p4_ntv_code_to_descr( unsigned int EventCode )
{
	int event, mask;

	internal_decode_event( EventCode, &event, &mask );

	if ( event == P4_custom_event ) {
		if ( mask >= _p4_custom_count )
			return ( NULL );
		return ( _p4_custom_map[mask].description );
	}

	if ( event == P4_user_event ) {
		if ( mask >= _p4_user_count )
			return ( NULL );
		return ( _p4_user_map[mask].description );
	}

	if ( event > sizeof ( _p4_native_map ) / sizeof ( hwd_p4_native_map_t ) )
		return ( NULL );

	return ( internal_translate_code( event, mask, description, ": " ) );
}


/* Given a native event code, assigns the native event's 
   information to a given pointer.
   NOTE: the info must be COPIED to the provided pointer,
   not just referenced!
*/
int
_p4_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int event, mask, tags;

	event = ( EventCode & PAPI_NATIVE_AND_MASK ) >> 16;	// event is in the third byte
	mask = ( EventCode & 0xffff );	// mask bits are in the first two bytes

	*( cmp_register_t * ) bits = _p4_native_map[event].bits;

	if ( ( event > sizeof ( _p4_native_map ) / sizeof ( hwd_p4_native_map_t ) )
		 && ( event != P4_custom_event ) && ( event != P4_user_event ) )
		return ( PAPI_ENOEVNT );

	switch ( event ) {
	case P4_custom_event:
		if ( mask > _p4_custom_count )
			return ( PAPI_ENOEVNT );
		*( cmp_register_t * ) bits = _p4_custom_map[mask].bits;
		return ( PAPI_OK );

	case P4_user_event:
		if ( mask > _p4_user_count )
			return ( PAPI_ENOEVNT );
		*( cmp_register_t * ) bits = _p4_user_map[mask].bits;
		return ( PAPI_OK );

	case P4_packed_SP_uop:
	case P4_packed_DP_uop:
	case P4_scalar_SP_uop:
	case P4_scalar_DP_uop:
	case P4_64bit_MMX_uop:
	case P4_128bit_MMX_uop:
	case P4_x87_FP_uop:
	case P4_x87_SIMD_moves_uop:
		// these event groups can be tagged for use with execution_event.
		// the tag bits are encoded as bits 5 - 8 of the otherwise unused mask bits
		// if a tag bit is set, the enable bit is also set
		tags = mask & 0x01e0;
		if ( tags ) {
			mask ^= tags;
			tags |= ESCR_TAG_ENABLE;
			( ( cmp_register_t * ) bits )->event |= tags;
		}
		break;

		// at retirement compound events; from Table A-2
	case P4_replay_event:
		// add the PEBS enable and cache stuff here
		// this stuff comes from Intel Table A-6
		// it is encoded by shifting into unused mask bits
		// for the replay_event mask list
		tags = ( mask >> PEBS_ENB_SHIFT ) & PEBS_ENB_MASK;
		if ( tags ) {
			mask ^= ( PEBS_ENB_MASK << PEBS_ENB_SHIFT );
			( ( cmp_register_t * ) bits )->pebs_enable = PEBS_UOP_TAG | tags;
		}
		tags = ( mask >> PEBS_MV_SHIFT ) & PEBS_MV_MASK;
		if ( tags ) {
			mask ^= ( PEBS_MV_MASK << PEBS_MV_SHIFT );
			( ( cmp_register_t * ) bits )->pebs_matrix_vert = tags;
		}
		break;

		// "These events may not be supported in all models of the processor family"
		// We need to find out which processors and vector appropriately, returning
		// an error if not supported on the current hardware.
	case P4_resource_stall:
	case P4_b2b_cycles:
	case P4_bnr:
	case P4_snoop:
	case P4_response:
		break;

	default:
		break;
	};

	// these bits are turned on for all event groups
	( ( cmp_register_t * ) bits )->event |= ESCR_EVENT_MASK( mask );
	( ( cmp_register_t * ) bits )->cccr |=
		CCCR_THR_MODE( CCCR_THR_ANY ) | CCCR_ENABLE;

	return ( PAPI_OK );
}

int
_p4_ntv_encode( unsigned int *EventCode, char *name, char *description,
				hwd_register_t * bits )
{
	hwd_p4_native_map_t *new_event;

	if ( _p4_user_count >=
		 ( sizeof ( _p4_user_map ) / sizeof ( hwd_p4_native_map_t ) ) )
		return ( PAPI_ENOEVNT );

	*EventCode = PAPI_NATIVE_MASK + ( P4_user_event << 16 ) + _p4_user_count;
	new_event = &( _p4_user_map[_p4_user_count] );
	_p4_user_count += 1;

	strcpy( new_event->name, name );
	strcpy( new_event->description, description );
	new_event->bits = *( cmp_register_t * ) bits;

	return ( PAPI_OK );
}

int
_p4_ntv_decode( unsigned int EventCode, char *name, char *description,
				hwd_register_t * bits )
{
	char *str;

	str = _p4_ntv_code_to_name( EventCode );
	if ( !str )
		return ( PAPI_ENOEVNT );
	strcpy( name, str );

	str = _p4_ntv_code_to_descr( EventCode );
	if ( !str )
		return ( PAPI_ENOEVNT );
	strcpy( description, str );

	return ( _p4_ntv_code_to_bits( EventCode, bits ) );
}

/* Reports the elements of the hwd_register_t struct as an array of names and a matching array of values.
   Maximum string length is name_len; Maximum number of values is count.
*/
static void
copy_value( unsigned int val, char *nam, char *names, unsigned int *values,
			int len )
{
	*values = val;
	strncpy( names, nam, len );
	names[len - 1] = 0;
}

int
_p4_ntv_bits_to_info( hwd_register_t * bits, char *names,
					  unsigned int *values, int name_len, int count )
{
	int i = 0;
	copy_value( ( ( cmp_register_t * ) bits )->cccr, "P4 CCCR",
				&names[i * name_len], &values[i], name_len );
	if ( ++i == count )
		return ( i );
	copy_value( ( ( cmp_register_t * ) bits )->event, "P4 Event",
				&names[i * name_len], &values[i], name_len );
	if ( ++i == count )
		return ( i );
	copy_value( ( ( cmp_register_t * ) bits )->pebs_enable, "P4 PEBS Enable",
				&names[i * name_len], &values[i], name_len );
	if ( ++i == count )
		return ( i );
	copy_value( ( ( cmp_register_t * ) bits )->pebs_matrix_vert,
				"P4 PEBS Matrix Vertical", &names[i * name_len], &values[i],
				name_len );
	if ( ++i == count )
		return ( i );
	copy_value( ( ( cmp_register_t * ) bits )->ireset, "P4 iReset",
				&names[i * name_len], &values[i], name_len );
	return ( ++i );
}

#define FP_NONE 0
#define  FP_X87 1
#define   FP_SP 2
#define   FP_DP 4
void
_p4_fixup_fp( hwi_search_t ** s, const hwi_dev_notes_t ** n )
{
	char *str = getenv( "PAPI_PENTIUM4_FP" );
	int mask = FP_NONE;

	/* CRAP CODE ALERT */
	/* This should absolutely not be here but the below maps are static */

	if ( MY_VECTOR.cmp_info.num_native_events == 0 )
		MY_VECTOR.cmp_info.num_native_events =
			( sizeof ( _p4_native_map ) +
			  sizeof ( _p4_custom_map ) ) / sizeof ( hwd_p4_native_map_t );

	/* if the env variable isn't set, return the defaults */
	if ( ( str == NULL ) || ( strlen( str ) == 0 ) ) {
		*s = FPU;
		*n = FPU_DESC;
		return;
	}

	if ( strstr( str, "X87" ) )
		mask |= FP_X87;
	if ( strstr( str, "SSE_SP" ) )
		mask |= FP_SP;
	if ( strstr( str, "SSE_DP" ) )
		mask |= FP_DP;

	switch ( mask ) {
	case FP_X87:
		*s = _p4_FP_X87;
		*n = _p4_FP_X87_dev_notes;
		break;
	case FP_SP:
		*s = _p4_FP_SSE_SP;
		*n = _p4_FP_SSE_SP_dev_notes;
		break;
	case FP_DP:
		*s = _p4_FP_SSE_DP;
		*n = _p4_FP_SSE_DP_dev_notes;
		break;
	case FP_X87 + FP_SP:
		*s = _p4_FP_X87_SSE_SP;
		*n = _p4_FP_X87_SSE_SP_dev_notes;
		break;
	case FP_X87 + FP_DP:
		*s = _p4_FP_X87_SSE_DP;
		*n = _p4_FP_X87_SSE_DP_dev_notes;
		break;
	case FP_SP + FP_DP:
		*s = _p4_FP_SSE_SP_DP;
		*n = _p4_FP_SSE_SP_DP_dev_notes;
		break;
	default:
		PAPIERROR( "Improper usage of PAPI_PENTIUM4_FP environment variable" );
		PAPIERROR( "Use one or two of X87,SSE_SP,SSE_DP" );
		*s = NULL;
		*n = NULL;
/* for testing purposes: can we delete an existing event? 
        *s = _p4_FP_null;
        *n = _p4_FP_null_dev_notes;
*/
	}
}

#define VEC_NONE 0
#define  VEC_MMX 1
#define  VEC_SSE 2
void
_p4_fixup_vec( hwi_search_t ** s, const hwi_dev_notes_t ** n )
{
	char *str = getenv( "PAPI_PENTIUM4_VEC" );
	int mask = VEC_NONE;

	/* if the env variable isn't set, return the default */
	if ( ( str == NULL ) || ( strlen( str ) == 0 ) ) {
		*s = VEC;
		*n = VEC_DESC;
		return;
	}

	if ( strstr( str, "MMX" ) )
		mask = VEC_MMX;
	else if ( strstr( str, "SSE" ) )
		mask |= VEC_SSE;

	switch ( mask ) {
	case VEC_MMX:
		*s = _p4_VEC_MMX;
		*n = _p4_VEC_MMX_dev_notes;
		break;
	case VEC_SSE:
		*s = _p4_VEC_SSE;
		*n = _p4_VEC_SSE_dev_notes;
		break;
	default:
		PAPIERROR( "Improper usage of PAPI_PENTIUM4_VEC environment variable" );
		PAPIERROR( "Use either SSE or MMX" );
		*s = NULL;
		*n = NULL;
	}
}

#endif
