/* $Id$
 * Performance counter event descriptions for the Intel P6 family.
 *
 * Copyright (C) 2003  Mikael Pettersson
 *
 * References
 * ----------
 * [IA32, Volume 3] "Intel Architecture Software Developer's Manual,
 * Volume 3: System Programming Guide". Intel document number 245472-011.
 * (at http://developer.intel.com/)
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * Intel Pentium Pro events.
 * Note that four L2 events were redefined in Pentium M.
 */

static const struct perfctr_unit_mask_4 p6_um_mesi = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x08, "M (modified cache state)" },
      { 0x04, "E (exclusive cache state)" },
      { 0x02, "S (shared cache state)" },
      { 0x01, "I (invalid cache state)" } }
};

static const struct perfctr_unit_mask_2 p6_um_ebl = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x20,
      .nvalues = 2 },
    { { 0x20, "transactions from any processor" },
      { 0x00, "self-generated transactions" } }
};

static const struct perfctr_event p6_events[] = {
    /* Data Cache Unit (DCU) */
    { 0x43, 0x3, NULL, "DATA_MEM_REFS" },
    { 0x45, 0x3, NULL, "DCU_LINES_IN" },
    { 0x46, 0x3, NULL, "DCU_M_LINES_IN" },
    { 0x47, 0x3, NULL, "DCU_M_LINES_OUT" },
    { 0x48, 0x3, NULL, "DCU_MISS_OUTSTANDING" },
    /* Instruction Fetch Unit (IFU) */
    { 0x80, 0x3, NULL, "IFU_IFETCH" },		/* XXX: was IFU_FETCH */
    { 0x81, 0x3, NULL, "IFU_IFETCH_MISS" },	/* XXX: was IFU_FETCH_MISS */
    { 0x85, 0x3, NULL, "ITLB_MISS" },
    { 0x86, 0x3, NULL, "IFU_MEM_STALL" },
    { 0x87, 0x3, NULL, "ILD_STALL" },
    /* L2 Cache */
    { 0x28, 0x3, UM(p6_um_mesi), "L2_IFETCH" },
    { 0x2A, 0x3, UM(p6_um_mesi), "L2_ST" },
    { 0x25, 0x3, NULL, "L2_M_LINES_INM" },
    { 0x2E, 0x3, UM(p6_um_mesi), "L2_RQSTS" },
    { 0x21, 0x3, NULL, "L2_ADS" },
    { 0x22, 0x3, NULL, "L2_DBUS_BUSY" },
    { 0x23, 0x3, NULL, "L2_DBUS_BUSY_RD" },
    /* External Bus Logic (EBL) */
    { 0x62, 0x3, UM(p6_um_ebl), "BUS_DRDY_CLOCKS" },
    { 0x63, 0x3, UM(p6_um_ebl), "BUS_LOCK_CLOCKS" },
    { 0x60, 0x3, NULL, "BUS_REQ_OUTSTANDING" },
    { 0x65, 0x3, UM(p6_um_ebl), "BUS_TRAN_BRD" },
    { 0x66, 0x3, UM(p6_um_ebl), "BUS_TRAN_RFO" },
    { 0x67, 0x3, UM(p6_um_ebl), "BUS_TRANS_WB" },
    { 0x68, 0x3, UM(p6_um_ebl), "BUS_TRAN_IFETCH" },
    { 0x69, 0x3, UM(p6_um_ebl), "BUS_TRAN_INVAL" },
    { 0x6A, 0x3, UM(p6_um_ebl), "BUS_TRAN_PWR" },
    { 0x6B, 0x3, UM(p6_um_ebl), "BUS_TRANS_P" },
    { 0x6C, 0x3, UM(p6_um_ebl), "BUS_TRANS_IO" },
    { 0x6D, 0x3, UM(p6_um_ebl), "BUS_TRAN_DEF" },
    { 0x6E, 0x3, UM(p6_um_ebl), "BUS_TRAN_BURST" },
    { 0x70, 0x3, UM(p6_um_ebl), "BUS_TRAN_ANY" },
    { 0x6F, 0x3, UM(p6_um_ebl), "BUS_TRAN_MEM" },
    { 0x64, 0x3, NULL, "BUS_DATA_RCV" },
    { 0x61, 0x3, NULL, "BUS_BNR_DRV" },
    { 0x7A, 0x3, NULL, "BUS_HIT_DRV" },
    { 0x7B, 0x3, NULL, "BUS_HITM_DRV" },
    { 0x7E, 0x3, NULL, "BUS_SNOOP_STALL" },
    /* Floating-Point Unit */
    { 0xC1, 0x1, NULL, "FLOPS" },
    { 0x10, 0x1, NULL, "FP_COMP_OPS_EXE" },
    { 0x11, 0x2, NULL, "FP_ASSIST" },
    { 0x12, 0x2, NULL, "MUL" },
    { 0x13, 0x2, NULL, "DIV" },
    { 0x14, 0x1, NULL, "CYCLES_DIV_BUSY" },
    /* Memory Ordering */
    { 0x03, 0x3, NULL, "LD_BLOCKS" },
    { 0x04, 0x3, NULL, "SB_DRAINS" },
    { 0x05, 0x3, NULL, "MISALIGN_MEM_REF" },
    /* Instruction Decoding and Retirement */
    { 0xC0, 0x3, NULL, "INST_RETIRED" },
    { 0xC2, 0x3, NULL, "UOPS_RETIRED" },
    { 0xD0, 0x3, NULL, "INST_DECODED" },
    /* Interrupts */
    { 0xC8, 0x3, NULL, "HW_INT_RX" },
    { 0xC6, 0x3, NULL, "CYCLES_INT_MASKED" },
    { 0xC7, 0x3, NULL, "CYCLES_INT_PENDING_AND_MASKED" },
    /* Branches */
    { 0xC4, 0x3, NULL, "BR_INST_RETIRED" },
    { 0xC5, 0x3, NULL, "BR_MISS_PRED_RETIRED" },
    { 0xC9, 0x3, NULL, "BR_TAKEN_RETIRED" },
    { 0xCA, 0x3, NULL, "BR_MISS_PRED_TAKEN_RET" },
    { 0xE0, 0x3, NULL, "BR_INST_DECODED" },
    { 0xE2, 0x3, NULL, "BTB_MISSES" },
    { 0xE4, 0x3, NULL, "BR_BOGUS" },
    { 0xE6, 0x3, NULL, "BACLEARS" },
    /* Stalls */
    { 0xA2, 0x3, NULL, "RESOURCE_STALLS" },
    { 0xD2, 0x3, NULL, "PARTIAL_RAT_STALLS" },
    /* Segment Register Loads */
    { 0x06, 0x3, NULL, "SEGMENT_REG_LOADS" },
    /* Clocks */
    { 0x79, 0x3, NULL, "CPU_CLK_UNHALTED" },
};

const struct perfctr_event_set p6_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P6,
    .event_prefix = "P6_",
    .include = NULL,
    .nevents = ARRAY_SIZE(p6_events),
    .events = p6_events,
};

static const struct perfctr_event ppro_events[] = {
    /* L2 cache */
    { 0x29, 0x3, UM(p6_um_mesi), "L2_LD" }, /* redefined in Pentium M */
    { 0x24, 0x3, NULL, "L2_LINES_IN" }, /* redefined in Pentium M */
    { 0x26, 0x3, NULL, "L2_LINES_OUT" }, /* redefined in Pentium M */
    { 0x27, 0x3, NULL, "L2_M_LINES_OUTM" }, /* redefined in Pentium M */
};

const struct perfctr_event_set perfctr_ppro_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P6,
    .event_prefix = "P6_",
    .include = &p6_event_set,
    .nevents = ARRAY_SIZE(ppro_events),
    .events = ppro_events,
};

/*
 * Intel Pentium II events.
 * Note that two PII events (0xB0 and 0xCE) are unavailable in the PIII.
 */

static const struct perfctr_unit_mask_0 p2_um_mmx_uops_exec = {
    { .type = perfctr_um_type_fixed,
      .default_value = 0x0F,
      .nvalues = 0 }
};

static const struct perfctr_unit_mask_6 p2_um_mmx_instr_type_exec = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x3F,
      .nvalues = 6 },
    { { 0x01, "MMX packed multiplies" },
      { 0x02, "MMX packed shifts" },
      { 0x04, "MMX pack operations" },
      { 0x08, "MMX unpack operations" },
      { 0x10, "MMX packed logical instructions" },
      { 0x20, "MMX packed arithmetic instructions" } }
};

static const struct perfctr_unit_mask_2 p2_um_fp_mmx_trans = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 2 },
    { { 0x00, "MMX to FP transitions" },
      { 0x01, "FP to MMX transitions" } }
};

static const struct perfctr_unit_mask_4 p2_um_seg_reg_rename = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "segment register ES" },
      { 0x02, "segment register DS" },
      { 0x04, "segment register FS" },
      { 0x08, "segment register GS" } }
};

static const struct perfctr_event p2andp3_events[] = {
    /* MMX Unit */
    { 0xB1, 0x3, NULL, "MMX_SAT_INSTR_EXEC" },
    { 0xB2, 0x3, UM(p2_um_mmx_uops_exec), "MMX_UOPS_EXEC" },
    { 0xB3, 0x3, UM(p2_um_mmx_instr_type_exec), "MMX_INSTR_TYPE_EXEC" },
    { 0xCC, 0x3, UM(p2_um_fp_mmx_trans), "FP_MMX_TRANS" },
    { 0xCD, 0x3, NULL, "MMX_ASSIST" },
    /* Segment Register Renaming */
    { 0xD4, 0x3, UM(p2_um_seg_reg_rename), "SEG_RENAME_STALLS" },
    { 0xD5, 0x3, UM(p2_um_seg_reg_rename), "SEG_REG_RENAMES" },
    { 0xD6, 0x3, NULL, "RET_SEG_RENAMES" },
};

static const struct perfctr_event_set p2andp3_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_PII,
    .event_prefix = "PII_",
    .include = &perfctr_ppro_event_set,
    .nevents = ARRAY_SIZE(p2andp3_events),
    .events = p2andp3_events,
};

static const struct perfctr_event p2_events[] = {	/* not in PIII :-( */
    /* MMX Unit */
    { 0xB0, 0x3, NULL, "MMX_INSTR_EXEC" },
    { 0xCE, 0x3, NULL, "MMX_INSTR_RET" },
};

const struct perfctr_event_set perfctr_p2_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_PII,
    .event_prefix = "PII_",
    .include = &p2andp3_event_set,
    .nevents = ARRAY_SIZE(p2_events),
    .events = p2_events,
};

/*
 * Intel Pentium III events.
 * Note that the two KNI decoding events were redefined in Pentium M.
 */

static const struct perfctr_unit_mask_4 p3_um_kni_prefetch = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 4 },
    { { 0x00, "prefetch NTA" },
      { 0x01, "prefetch T1" },
      { 0x02, "prefetch T2" },
      { 0x03, "weakly ordered stores" } }
};

static const struct perfctr_event p3_events_1[] = {
    /* Memory Ordering */
    { 0x07, 0x3, UM(p3_um_kni_prefetch), "EMON_KNI_PREF_DISPATCHED" },
    { 0x4B, 0x3, UM(p3_um_kni_prefetch), "EMON_KNI_PREF_MISS" },
};

static const struct perfctr_event_set p3_event_set_1 = {
    .cpu_type = PERFCTR_X86_INTEL_PIII,
    .event_prefix = "PIII_",
    .include = &p2andp3_event_set,
    .nevents = ARRAY_SIZE(p3_events_1),
    .events = p3_events_1,
};

static const struct perfctr_unit_mask_2 p3_um_kni_inst_retired = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 2 },
    { { 0x00, "packed and scalar" },
      { 0x01, "scalar" } }
};

static const struct perfctr_event p3_events_2[] = {
    /* Instruction Decoding and Retirement */
    { 0xD8, 0x3, UM(p3_um_kni_inst_retired), "EMON_KNI_INST_RETIRED" }, /* redefined in Pentium M */
    { 0xD9, 0x3, UM(p3_um_kni_inst_retired), "EMON_KNI_COMP_INST_RET" }, /* redefined in Pentium M */
};

const struct perfctr_event_set perfctr_p3_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_PIII,
    .event_prefix = "PIII_",
    .include = &p3_event_set_1,
    .nevents = ARRAY_SIZE(p3_events_2),
    .events = p3_events_2,
};

/*
 * Intel Pentium M events.
 * Note that six PPro/PIII events were redefined. To describe that
 * we have to break up the PPro and PIII event sets, and assemble
 * the Pentium M event set in several steps.
 */

static const struct perfctr_unit_mask_6 pentm_um_mesi_prefetch = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 6 },
    /* XXX: how should we describe that bits 5-4 are a single field? */
    { { 0x01, "I (invalid cache state)" },
      { 0x02, "S (shared cache state)" },
      { 0x04, "E (exclusive cache state)" },
      { 0x08, "M (modified cache state)" },
      /* Bits 5-4: 00: all but HW-prefetched lines, 01: only HW-prefetched
	 lines, 10/11: all lines */
      { 0x10, "prefetch type bit 0" },
      { 0x20, "prefetch type bit 1" } }
};

static const struct perfctr_unit_mask_2 pentm_um_est_trans = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 2 },
    { { 0x00, "All transitions" },
      { 0x02, "Only Frequency transitions" } }
};

static const struct perfctr_unit_mask_4 pentm_um_sse_inst_ret = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 4 },
    { { 0x00, "SSE Packed Single" },
      { 0x01, "SSE Packed-Single and Scalar-Single" },
      { 0x02, "SSE2 Packed-Double" },
      { 0x03, "SSE2 Scalar-Double" } }
};

static const struct perfctr_unit_mask_4 pentm_um_sse_comp_inst_ret = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 4 },
    { { 0x00, "SSE Packed Single" },
      { 0x01, "SSE Scalar-Single" },
      { 0x02, "SSE2 Packed-Double" },
      { 0x03, "SSE2 Scalar-Double" } }
};

static const struct perfctr_unit_mask_3 pentm_um_fused_uops = {
    { .type = perfctr_um_type_exclusive,
      .default_value = 0x00,
      .nvalues = 3 },
    { { 0x00, "All fused micro-ops" },
      { 0x01, "Only load+Op micro-ops" },
      { 0x02, "Only std+sta micro-ops" } }
};

static const struct perfctr_event pentm_events[] = {
    /* L2 cache */
    { 0x24, 0x3, UM(pentm_um_mesi_prefetch), "L2_LINES_IN" }, /* redefined */
    { 0x26, 0x3, UM(pentm_um_mesi_prefetch), "L2_LINES_OUT" }, /* redefined */
    { 0x27, 0x3, UM(pentm_um_mesi_prefetch), "L2_M_LINES_OUT" }, /* redefined */
    { 0x29, 0x3, UM(pentm_um_mesi_prefetch), "L2_LD" }, /* redefined */
    /* Power Management */
    { 0x58, 0x3, UM(pentm_um_est_trans), "EMON_EST_TRANS" },
    { 0x59, 0x3, NULL, "EMON_THERMAL_TRIP" /*XXX: set bit 22(!?) for edge */ },
    /* BPU */
    { 0x88, 0x3, NULL, "BR_INST_EXEC" },
    { 0x89, 0x3, NULL, "BR_MISSP_EXEC" },
    { 0x8A, 0x3, NULL, "BR_BAC_MISSP_EXEC" },
    { 0x8B, 0x3, NULL, "BR_CND_EXEC" },
    { 0x8C, 0x3, NULL, "BR_CND_MISSP_EXEC" },
    { 0x8D, 0x3, NULL, "BR_IND_EXEC" },
    { 0x8E, 0x3, NULL, "BR_IND_MISSP_EXEC" },
    { 0x8F, 0x3, NULL, "BR_RET_EXEC" },
    { 0x90, 0x3, NULL, "BR_RE_MISSP_EXEC" },
    { 0x91, 0x3, NULL, "BR_RET_BAC_MISSP_EXEC" },
    { 0x92, 0x3, NULL, "BR_CALL_EXEC" },
    { 0x93, 0x3, NULL, "BR_CALL_MISSP_EXEC" },
    { 0x94, 0x3, NULL, "BR_IND_CALL_EXEC" },
    /* Decoder */
    { 0xCE, 0x3, NULL, "EMON_SIMD_INSTR_RETIRED" },
    { 0xD3, 0x3, NULL, "EMON_SYNCH_UOPS" },
    { 0xD7, 0x3, NULL, "EMON_ESP_UOPS" },
    { 0xD8, 0x3, UM(pentm_um_sse_inst_ret), "EMON_SSE_SSE2_INST_RETIRED" }, /* redefined */
    { 0xD9, 0x3, UM(pentm_um_sse_comp_inst_ret), "EMON_SSE_SSE2_COMP_INST_RETIRED" }, /* redefined */
    { 0xDA, 0x3, UM(pentm_um_fused_uops), "EMON_FUSED_UOPS_RET" },
    { 0xDB, 0x3, NULL, "EMON_UNFUSION" },
    /* Prefetcher */
    { 0xF0, 0x3, NULL, "EMON_PREF_RQSTS_UP" },
    { 0xF8, 0x3, NULL, "EMON_PREF_RQSTS_DN" },
};

const struct perfctr_event_set pentm_event_set_1 = {
    .cpu_type = PERFCTR_X86_INTEL_PII,
    .event_prefix = "PII_",
    .include = &p6_event_set,
    .nevents = ARRAY_SIZE(p2andp3_events),
    .events = p2andp3_events,
};

const struct perfctr_event_set pentm_event_set_2 = {
    .cpu_type = PERFCTR_X86_INTEL_PIII,
    .event_prefix = "PIII_",
    .include = &pentm_event_set_1,
    .nevents = ARRAY_SIZE(p3_events_1),
    .events = p3_events_1,
};

const struct perfctr_event_set perfctr_pentm_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_PENTM,
    .event_prefix = "PENTM_",
    .include = &pentm_event_set_2,
    .nevents = ARRAY_SIZE(pentm_events),
    .events = pentm_events,
};
