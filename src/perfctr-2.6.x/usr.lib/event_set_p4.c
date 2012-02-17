/* $Id: event_set_p4.c,v 1.5 2004/02/20 21:32:06 mikpe Exp $
 * Performance counter event descriptions for Intel P4.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 *
 * This is still preliminary:
 * - need mapping from enum escr_set to <cccr bitmask, escr select>
 * - the current data structures can't describe all P4 side-conditions
 * - replace eventsel in struct perfctr_event with a unique cookie?
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

enum escr_set {
    ALF_ESCR_0_1,	/* CCCR 12/13/14/15/16/17 via ESCR select 0x01 */
    BPU_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x00 */
    BSU_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x07 */
    BSU_ESCR_0,		/* CCCR 0/1 via ESCR select 0x07 */
    BSU_ESCR_1,		/* CCCR 2/3 via ESCR select 0x07 */
    CRU_ESCR_0_1,	/* CCCR 12/13/14/15/16/17 via ESCR select 0x04 */
    CRU_ESCR_2_3,	/* CCCR 12/13/14/15/16/17 via ESCR select 0x05 */
    DAC_ESCR_0_1,	/* CCCR 8/9/10/11 via ESCR select 0x05 */
    FIRM_ESCR_0_1,	/* CCCR 8/9/10/11 via ESCR select 0x01 */
    FSB_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x06 */
    FSB_ESCR_0,		/* CCCR 0/1 via ESCR select 0x06 */
    FSB_ESCR_1,		/* CCCR 2/3 via ESCR select 0x06 */
    ITLB_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x03 */
    MOB_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x02 */
    MS_ESCR_0_1,	/* CCCR 4/5/6/7 via ESCR select 0x00 */
    PMH_ESCR_0_1,	/* CCCR 0/1/2/3 via ESCR select 0x04 */
    RAT_ESCR_0_1,	/* CCCR 12/13/14/15/16/17 via ESCR select 0x02 */
    SAAT_ESCR_0_1,	/* CCCR 8/9/10/11 via ESCR select 0x02 */
    TBPU_ESCR_0_1,	/* CCCR 4/5/6/7 via ESCR select 0x02 */
    TC_ESCR_0_1,	/* CCCR 4/5/6/7 via ESCR select 0x01 */
};

static const struct perfctr_unit_mask_8 p4_um_TC_deliver_mode = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,	/* DD */
      .nvalues = 8 },
    { { 0x01, "DD:both logical processors in deliver mode" },
      { 0x02, "DB:logical processor 0 in deliver mode, 1 in build mode" },
      { 0x04, "DI:logical processor 0 in deliver mode, 1 is inactive" },
      { 0x08, "BD:logical processor 0 in build mode, 1 in deliver mode" },
      { 0x10, "BB:both logical processors in build mode" },
      { 0x20, "BI:logical processor 0 in build mode, 1 is inactive" },
      { 0x40, "ID:logical processor 0 is inactive, 1 in deliver mode" },
      { 0x80, "IB:logical processor 0 is inactive, 1 in build mode" } }
};

static const struct perfctr_unit_mask_1 p4_um_BPU_fetch_request = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x00,
      .nvalues = 1 },
    { { 0x01, "TCMISS:Trace cache lookup miss" } }
};

static const struct perfctr_unit_mask_3 p4_um_ITLB_reference = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "HIT:ITLB hit" },
      { 0x02, "MISS:ITLB miss" },
      { 0x04, "HIT_UC:Uncacheable ITLB hit" } }
};

static const struct perfctr_unit_mask_2 p4_um_memory_cancel = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0C,
      .nvalues = 2 },
    { { 0x04, "ST_RB_FULL:Replayed because no store request buffer is available" },
      { 0x08, "64K_CONF:Conflicts due to 64K aliasing" } }
};

static const struct perfctr_unit_mask_2 p4_um_memory_complete = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x01, "LSC:Load split completed, excluding UC/WC loads" },
      { 0x02, "SSC:Any split stores completed" } }
};

static const struct perfctr_unit_mask_1 p4_um_load_port_replay = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x02,
      .nvalues = 1 },
    { { 0x02, "SPLIT_LD:Split load" } }
};

static const struct perfctr_unit_mask_1 p4_um_store_port_replay = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x02,
      .nvalues = 1 },
    { { 0x02, "SPLIT_ST:Split store" } }
};

static const struct perfctr_unit_mask_4 p4_um_MOB_load_replay = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x3A,
      .nvalues = 4 },
    { { 0x02, "NO_STA:Replayed because of unknown store address" },
      { 0x08, "NO_STD:Replayed because of unknown store data" },
      { 0x10, "PARTIAL_DATA:Replayed because of partially overlapped data access between the load and store operations" },
      { 0x20, "UNALGN_ADDR:Replayed because the lower 4 bits of the linear address do not match between the load and store operations" } }
};

static const struct perfctr_unit_mask_2 p4_um_page_walk_type = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x01, "DTMISS:Page walk for a data TLB miss" },
      { 0x02, "ITMISS:Page walk for an instruction TLB miss" } }
};

static const struct perfctr_unit_mask_9 p4_um_BSQ_cache_reference = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x73F,
      .nvalues = 9 },
    { { 0x001, "RD_2ndL_HITS:Read 2nd level cache hit Shared" },
      { 0x002, "RD_2ndL_HITE:Read 2nd level cache hit Exclusive" },
      { 0x004, "RD_2ndL_HITM:Read 2nd level cache hit Modified" },
      { 0x008, "RD_3rdL_HITS:Read 3rd level cache hit Shared" },
      { 0x010, "RD_3rdL_HITE:Read 3rd level cache hit Exclusive" },
      { 0x020, "RD_3rdL_HITM:Read 3rd level cache hit Modified" },
      { 0x100, "RD_2ndL_MISS:Read 2nd level cache miss" },
      { 0x200, "RD_3rdL_MISS:Read 3rd level cache miss" },
      { 0x400, "WR_2ndL_MISS:Writeback lookup from DAC misses the 2nd level cache" } }
};

/* review P4M0 and P4M2 diffs according to P4 Code Optim manual */
static const struct perfctr_unit_mask_15 p4_um_IOQ = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0xEFE1,
      .nvalues = 15 },
    /* XXX: how should we describe that bits 0-4 are a single field? */
    { { 0x0001, "bus request type bit 0" },
      { 0x0002, "bus request type bit 1" },
      { 0x0004, "bus request type bit 2" },
      { 0x0008, "bus request type bit 3" },
      { 0x0010, "bus request type bit 4" },
      { 0x0020, "ALL_READ:Count read entries" },
      { 0x0040, "ALL_WRITE:Count write entries" },
      { 0x0080, "MEM_UC:Count UC memory access entries" },
      { 0x0100, "MEM_WC:Count WC memory access entries" },
      { 0x0200, "MEM_WT:Count WT memory access entries" },
      { 0x0400, "MEM_WP:Count WP memory access entries" },
      { 0x0800, "MEM_WB:Count WB memory access entries" },
      { 0x2000, "OWN:Count own store requests" },
      { 0x4000, "OTHER:Count other and DMA store requests" },
      { 0x8000, "PREFETCH:Include HW and SW prefetch requests" } }
};

static const struct perfctr_unit_mask_6 p4_um_FSB_data_activity = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x1B,
      .nvalues = 6 },
    /* DRDY_OWN is mutually exclusive with DRDY_OTHER */
    /* DBSY_OWN is mutually exclusive with DBSY_OTHER */
    { { 0x01, "DRDY_DRV:Count when this processor drives data onto the bus" },
      { 0x02, "DRDY_OWN:Count when this processor reads data from the bus" },
      { 0x04, "DRDY_OTHER:Count when data is on the bus but not being sampled by the processor" },
      { 0x08, "DBSY_DRV:Count when this processor reserves the bus for driving data" },
      { 0x10, "DBSY_OWN:Count when this processor reserves the bus for sampling data" },
      { 0x20, "DBSY_OTHER:Count when the bus is reserved for driving data this processor will not sample" } }
};

static const struct perfctr_unit_mask_13 p4_um_BSQ = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0021,
      .nvalues = 13 },
    { { 0x0001, "REQ_TYPE0:Request type encoding bit 0" },
      { 0x0002, "REQ_TYPE1:Request type encoding bit 1" },
      { 0x0004, "REQ_LEN0:Request length encoding bit 0" },
      { 0x0008, "REQ_LEN1:Request length encoding bit 1" },
      { 0x0020, "REQ_IO_TYPE:Request type is input or output" },
      { 0x0040, "REQ_LOCK_TYPE:Request type is bus lock" },
      { 0x0080, "REQ_CACHE_TYPE:Request type is cacheable" },
      { 0x0100, "REQ_SPLIT_TYPE:Request type is a bus 8-byte chunk split across 8-byte boundary" },
      { 0x0200, "REQ_DEM_TYPE:Request type is a demand (1) or prefetch (0)" },
      { 0x0400, "REQ_ORD_TYPE:Request is an ordered type" },
      { 0x0800, "MEM_TYPE0:Memory type encoding bit 0" },
      { 0x1000, "MEM_TYPE1:Memory type encoding bit 1" },
      { 0x2000, "MEM_TYPE2:Memory type encoding bit 2" } }
};

static const struct perfctr_unit_mask_1 p4_um_firm_uop = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x8000,
      .nvalues = 1 },
    { { 0x8000, "ALL:count all uops of this type" } }
};

static const struct perfctr_unit_mask_2 p4_um_x87_SIMD_moves_uop = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x18,
      .nvalues = 2 },
    { { 0x08, "ALLP0:Count all x87/SIMD store/move uops" },
      { 0x10, "ALLP2:count all x87/SIMD load uops" } }
};

static const struct perfctr_unit_mask_1 p4_um_TC_misc = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x10,
      .nvalues = 1 },
    { { 0x10, "FLUSH:Number of flushes" } }
};

static const struct perfctr_unit_mask_1 p4_um_global_power_events = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 1 },
    { { 0x01, "Running:The processor is active" } }
};

static const struct perfctr_unit_mask_1 p4_um_tc_ms_xfer = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 1 },
    { { 0x01, "CISC:A TC to MS transfer ocurred" } }
};

static const struct perfctr_unit_mask_3 p4_um_uop_queue_writes = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "FROM_TC_BUILD:uops written from TC build mode" },
      { 0x02, "FROM_TC_DELIVER:uops written from TC deliver mode" },
      { 0x04, "FROM_ROM:uops written from microcode ROM" } }
};

static const struct perfctr_unit_mask_4 p4_um_branch_type = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x1E,
      .nvalues = 4 },
    { { 0x02, "CONDITIONAL:Conditional jumps" },
      { 0x04, "CALL:Call branches" }, /* XXX: diff MISPRED/non-MISPRED events? */
      { 0x08, "RETURN:Return branches" },
      { 0x10, "INDIRECT:Returns, indirect calls, or indirect jumps" } }
};

static const struct perfctr_unit_mask_1 p4_um_resource_stall = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x20,
      .nvalues = 1 },
    { { 0x20, "SBFULL:A Stall due to lack of store buffers" } }
};

static const struct perfctr_unit_mask_3 p4_um_WC_Buffer = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 3 },
    { { 0x01, "WCB_EVICTS:all causes" },
      { 0x02, "WCB_FULL_EVICT:no WC buffer is available" },
      /* XXX: 245472-011 no longer lists bit 2, but that looks like
	 a table formatting error. Keeping it for now. */
      { 0x04, "WCB_HITM_EVICT:store encountered a Hit Modified condition" } }
};

static const struct perfctr_unit_mask_6 p4_um_b2b_cycles = {
    /* XXX: bits 1-6; no details documented yet */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x7E,
      .nvalues = 6 },
    { { 0x02, "bit 1" },
      { 0x04, "bit 2" },
      { 0x08, "bit 3" },
      { 0x10, "bit 4" },
      { 0x20, "bit 5" },
      { 0x40, "bit 6" } }
};

static const struct perfctr_unit_mask_3 p4_um_bnr = {
    /* XXX: bits 0-2; no details documented yet */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "bit 0" },
      { 0x02, "bit 1" },
      { 0x04, "bit 2" } }
};

static const struct perfctr_unit_mask_3 p4_um_snoop = {
    /* XXX: bits 2, 6, and 7; no details documented yet */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0xC4,
      .nvalues = 3 },
    { { 0x04, "bit 2" },
      { 0x40, "bit 6" },
      { 0x80, "bit 7" } }
};

static const struct perfctr_unit_mask_4 p4_um_response = {
    /* XXX: bits 1, 2, 8, and 9; no details documented yet */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x306,
      .nvalues = 4 },
    { { 0x002, "bit 1" },
      { 0x004, "bit 2" },
      { 0x100, "bit 8" },
      { 0x200, "bit 9" } }
};

static const struct perfctr_unit_mask_2 p4_um_nbogus_bogus = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 2 },
    { { 0x01, "NBOGUS:The marked uops are not bogus" },
      { 0x02, "BOGUS:The marked uops are bogus" } }
};

static const struct perfctr_unit_mask_8 p4_um_execution_event = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 8 },
    { { 0x01, "NBOGUS0:non-bogus uops with tag bit 0 set" },
      { 0x02, "NBOGUS1:non-bogus uops with tag bit 1 set" },
      { 0x04, "NBOGUS2:non-bogus uops with tag bit 2 set" },
      { 0x08, "NBOGUS3:non-bogus uops with tag bit 3 set" },
      { 0x10, "BOGUS0:bogus uops with tag bit 0 set" },
      { 0x20, "BOGUS1:bogus uops with tag bit 1 set" },
      { 0x40, "BOGUS2:bogus uops with tag bit 2 set" },
      { 0x80, "BOGUS3:bogus uops with tag bit 3 set" } }
};

static const struct perfctr_unit_mask_4 p4_um_instr_retired = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 4 },
    { { 0x01, "NBOGUSNTAG:Non-bogus instructions that are not tagged" },
      { 0x02, "NBOGUSTAG:Non-bogus instructions that are tagged" },
      { 0x04, "BOGUSNTAG:Bogus instructions that are not tagged" },
      { 0x08, "BOGUSTAG:Bogus instructions that are tagged" } }
};

static const struct perfctr_unit_mask_2 p4_um_uop_type = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x06,
      .nvalues = 2 },
    { { 0x02, "TAGLOADS:The uop is a load operation" },
      { 0x04, "TAGSTORES:The uop is a store operation" } }
};

static const struct perfctr_unit_mask_4 p4_um_branch_retired = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0C,	/* taken branches */
      .nvalues = 4 },
    { { 0x01, "MMNP:Branch Not-taken Predicted" },
      { 0x02, "MMNM:Branch Not-taken Mispredicted" },
      { 0x04, "MMTP:Branch Taken Predicted" },
      { 0x08, "MMTM:Branch Taken Mispredicted" } }
};

static const struct perfctr_unit_mask_1 p4_um_mispred_branch_retired = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 1 },
    { { 0x01, "NBOGUS:The retired branch is not bogus" } }
};

static const struct perfctr_unit_mask_5 p4_um_x87_assist = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x1F,
      .nvalues = 5 },
    { { 0x01, "FPSU:FP stack underflow" },
      { 0x02, "FPSO:FP stack overflow" },
      { 0x04, "POAO:x87 output overflow" },
      { 0x08, "POAU:x87 output underflow" },
      { 0x10, "PREA:x87 input assist" } }
};

static const struct perfctr_unit_mask_3 p4_um_machine_clear = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x01,
      .nvalues = 3 },
    { { 0x01, "CLEAR:Count a portion of the cycles when the machine is cleared" },
      { 0x04, "MOCLEAR:Count clears due to memory ordering issues" },
      { 0x08, "SMCLEAR:Count clears due to self-modifying code issues" } }
};

static const struct perfctr_event p4_events[] = {
    /* Non-Retirement Events: */
    { 0x01, TC_ESCR_0_1, UM(p4_um_TC_deliver_mode), "TC_deliver_mode",
      "duration of the operating modes of the trace cache and decode engine" },
    { 0x03, BPU_ESCR_0_1, UM(p4_um_BPU_fetch_request), "BPU_fetch_request",
      "instruction fetch requests by the Branch Prediction unit" },
    { 0x18, ITLB_ESCR_0_1, UM(p4_um_ITLB_reference), "ITLB_reference",
      "translations using the Instruction Translation Look-aside Buffer" },
    { 0x02, DAC_ESCR_0_1, UM(p4_um_memory_cancel), "memory_cancel",
      "cancelled requests in the Data cache Address Control unit" },
    { 0x08, SAAT_ESCR_0_1, UM(p4_um_memory_complete), "memory_complete",
      "completed load split, store split, uncacheable split, uncacheable load" },
    { 0x04, SAAT_ESCR_0_1, UM(p4_um_load_port_replay), "load_port_replay",
      /* XXX: only ESCR1 supports at-retirement */
      "replayed events at the load port" },
    { 0x05, SAAT_ESCR_0_1, UM(p4_um_store_port_replay), "store_port_replay",
      /* XXX: only ESCR1 supports at-retirement */
      "replayed events at the store port" },
    { 0x03, MOB_ESCR_0_1, UM(p4_um_MOB_load_replay), "MOB_load_replay",
      "replayed loads at the memory order buffer" },
    { 0x01, PMH_ESCR_0_1, UM(p4_um_page_walk_type), "page_walk_type",
      "page walks by the page miss handler" },
    { 0x0C, BSU_ESCR_0_1, UM(p4_um_BSQ_cache_reference), "BSQ_cache_reference",
      "cache references seen by the bus unit" },
    { 0x03, FSB_ESCR_0_1, UM(p4_um_IOQ), "IOQ_allocation",
      /* XXX: ESCR1 unavailable if CPUID < 0xF27 */
      "bus transactions" },
    { 0x1A, FSB_ESCR_1, UM(p4_um_IOQ), "IOQ_active_entries",
      "number of active IOQ entries" },
    { 0x17, FSB_ESCR_0_1, UM(p4_um_FSB_data_activity), "FSB_data_activity",
      "DRDY or DBSY events on the front side bus" },
    { 0x05, BSU_ESCR_0, UM(p4_um_BSQ), "BSQ_allocation",
      "allocations in the bus sequence unit" },
    { 0x06, BSU_ESCR_1, UM(p4_um_BSQ), "bsq_active_entries",
      "number of active BSQ entries" },
    { 0x34, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "SSE_input_assist",
      "assists requested for SSE and SSE2 input operands" },
    { 0x08, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "packed_SP_uop",
      "packed single-precision uops" },
    { 0x0C, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "packed_DP_uop",
      "packed double-precision uops" },
    { 0x0A, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "scalar_SP_uop",
      "scalar single-precision uops" },
    { 0x0E, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "scalar_DP_uop",
      "scalar double-precision uops" },
    { 0x02, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "64bit_MMX_uop",
      "64 bit SIMD MMX instructions" },
    { 0x1A, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "128bit_MMX_uop",
      "128 bit integer SIMD SSE2 instructions" },
    { 0x04, FIRM_ESCR_0_1, UM(p4_um_firm_uop), "x87_FP_uop",
      "x87 floating-point uops" },
    { 0x2E, FIRM_ESCR_0_1, UM(p4_um_x87_SIMD_moves_uop), "x87_SIMD_moves_uop",
      "x87 FPU, MMX, SSE, or SSE2 load, store, and move uops" },
    { 0x06, TC_ESCR_0_1, UM(p4_um_TC_misc), "TC_misc",
      "miscellaneous events detected by the TC" },
    { 0x13, FSB_ESCR_0_1, UM(p4_um_global_power_events), "global_power_events",
      "time during which the processor is not stopped" },
    { 0x05, MS_ESCR_0_1, UM(p4_um_tc_ms_xfer), "tc_ms_xfer",
      "number of times uop delivery changed from TC to MS ROM" },
    { 0x09, MS_ESCR_0_1, UM(p4_um_uop_queue_writes), "uop_queue_writes",
      "number of valid uops written to the uop queue" },
    { 0x05, TBPU_ESCR_0_1, UM(p4_um_branch_type), "retired_mispred_branch_type",
      "retired mispredicted branches by type" },
    { 0x04, TBPU_ESCR_0_1, UM(p4_um_branch_type), "retired_branch_type",
      "retired branches by type" },
    { 0x01, ALF_ESCR_0_1, UM(p4_um_resource_stall), "resource_stall",
      /* XXX: may not be supported in all P4 models */
      "stalls in the Allocator" },
    { 0x05, DAC_ESCR_0_1, UM(p4_um_WC_Buffer), "WC_Buffer",
      "write combining buffer operations" },
    { 0x16, FSB_ESCR_0_1, UM(p4_um_b2b_cycles), "b2b_cycles",
      /* XXX: may not be supported in all P4 models */
      "back-to-back bus cycles" },
    { 0x08, FSB_ESCR_0_1, UM(p4_um_bnr), "bnr",
      /* XXX: may not be supported in all P4 models */
      "bus not ready conditions" },
    { 0x06, FSB_ESCR_0_1, UM(p4_um_snoop), "snoop",
      /* XXX: may not be supported in all P4 models */
      "snoop hit modified bus traffic" },
    { 0x04, FSB_ESCR_0_1, UM(p4_um_response), "response",
      /* XXX: may not be supported in all P4 models */
      "different types of responses" },
    { 0x08, CRU_ESCR_2_3, UM(p4_um_nbogus_bogus), "front_end_event",
      /* XXX: another ESCR must count uop_type */
      /* XXX: can support PEBS */
      "retired uops, tagged by the front-end tagging mechanism" },
    { 0x0C, CRU_ESCR_2_3, UM(p4_um_execution_event), "execution_event",
      /* XXX: needs upstream ESCR */
      /* XXX: can support PEBS */
      "retired uops, tagged by the execution tagging mechanism" },
    { 0x09, CRU_ESCR_2_3, UM(p4_um_nbogus_bogus), "replay_event",
      /* XXX: needs PEBS_ENABLE, PEBS_MATRIX_VERT, and possibly upstream ESCR */
      /* XXX: can support PEBS */
      "retired uops, tagged by the replay tagging mechanism" },
    { 0x02, CRU_ESCR_0_1, UM(p4_um_instr_retired), "instr_retired",
      "retired instructions" },
    { 0x01, CRU_ESCR_0_1, UM(p4_um_nbogus_bogus), "uops_retired",
      "retired uops" },
    { 0x02, RAT_ESCR_0_1, UM(p4_um_uop_type), "uop_type",
      "tag uops for the front-end tagging mechanism" },
    { 0x06, CRU_ESCR_2_3, UM(p4_um_branch_retired), "branch_retired",
      "retired branches" },
    { 0x03, CRU_ESCR_0_1, UM(p4_um_mispred_branch_retired), "mispred_branch_retired",
      "retired mispredicted branches" },
    { 0x03, CRU_ESCR_2_3, UM(p4_um_x87_assist), "x87_assist",
      "retired x87 instructions that required special handling" },
    { 0x02, CRU_ESCR_2_3, UM(p4_um_machine_clear), "machine_clear",
      "cycles or occurrences when the entire pipeline is cleared" },
};

const struct perfctr_event_set perfctr_p4_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P4,
    .event_prefix = "P4_",
    .include = NULL,
    .nevents = ARRAY_SIZE(p4_events),
    .events = p4_events,
};

/*
 * Intel Pentium 4 Model 3 events.
 */

static const struct perfctr_event p4m3_events[] = {
    { 0x07, CRU_ESCR_0_1, UM(p4_um_nbogus_bogus), "instr_completed",
      "retired and completed instructions" },
};

const struct perfctr_event_set perfctr_p4m3_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P4M3,
    .event_prefix = "P4M3_",
    .include = &perfctr_p4_event_set,
    .nevents = ARRAY_SIZE(p4m3_events),
    .events = p4m3_events,
};
