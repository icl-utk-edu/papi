/* $Id: event_set_amd.c,v 1.8.2.1 2004/08/02 22:27:06 mikpe Exp $
 * Performance counter event descriptions for AMD K7 and K8.
 *
 * Copyright (C) 2003  Mikael Pettersson
 *
 * References
 * ----------
 * "AMD Athlon Processor x86 Code Optimization Guide",
 * Appendix D: "Performance Monitoring Counters".
 * AMD Publication #22007
 * Revision E (on AMD Processor Technical Documents CD, Med-12/99-0, 21860F)
 * Revision K (at http://www.amd.com/).
 *
 * "BIOS and Kernel Developer's Guide for AMD Athlon 64 and
 * AMD Opteron Processors", Chapter 10: "Performance Monitoring".
 * AMD Publication #26094, Revision 3.14 (at http://www.amd.com).

 * "Revision Guide for AMD Opteron Processors",
 * AMD Publication #25759, Revision 3.09
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * AMD K7 events.
 *
 * Note: Different revisions of AMD #22007 list different sets of events.
 * We split the K7 event set into an "official" part based on recent
 * revisions of #22007, and an "unofficial" part which includes events
 * only documented in older revisions of #22007 (specifically Rev. E).
 *
 * All official K7 events are also present in K8, as are most of the
 * unofficial K7 events.
 */

static const struct perfctr_unit_mask_5 k7_um_moesi = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x1F,
      .nvalues = 5 },
    { { 0x10, "Modified (M)" },
      { 0x08, "Owner (O)" },
      { 0x04, "Exclusive (E)" },
      { 0x02, "Shared (S)" },
      { 0x01, "Invalid (I)" } }
};

static const struct perfctr_event k7_official_events[] = {
    { 0x40, 0xF, NULL, "DATA_CACHE_ACCESSES",
      "Data cache accesses" },
    { 0x41, 0xF, NULL, "DATA_CACHE_MISSES",
      "Data cache misses" },
    { 0x42, 0xF, UM(k7_um_moesi), "DATA_CACHE_REFILLS_FROM_L2",
      "Data cache refills from L2" },
    { 0x43, 0xF, UM(k7_um_moesi), "DATA_CACHE_REFILLS_FROM_SYSTEM",
      "Data cache refills from system" },
    { 0x44, 0xF, UM(k7_um_moesi), "DATA_CACHE_WRITEBACKS",
      "Data cache writebacks" },
    { 0x45, 0xF, NULL, "L1_DTLB_MISSES_AND_L2_DTLB_HITS",
      "L1 DTLB misses and L2 DTLB hits" },
    { 0x46, 0xF, NULL, "L1_AND_L2_DTLB_MISSES",
      "L1 and L2 DTLB misses" },
    { 0x47, 0xF, NULL, "MISALIGNED_DATA_REFERENCES",
      "Misaligned data references" },
    { 0x80, 0xF, NULL, "INSTRUCTION_CACHE_FETCHES",
      "Instruction cache fetches" },
    { 0x81, 0xF, NULL, "INSTRUCTION_CACHE_MISSES",
      "Instruction cache misses" },
    { 0x84, 0xF, NULL, "L1_ITLB_MISSES_AND_L2_ITLB_HITS", /* XXX: was L1_ITLB_MISSES */
      "L1 ITLB misses (and L2 ITLB hits)" },
    { 0x85, 0xF, NULL, "L1_AND_L2_ITLB_MISSES",	/* XXX: was L2_ITLB_MISSES */
      "(L1 and) L2 ITLB misses" },
    { 0xC0, 0xF, NULL, "RETIRED_INSTRUCTIONS",
      "Retired instructions (includes exceptions, interrupts, resyncs)" },
    { 0xC1, 0xF, NULL, "RETIRED_OPS",
      "Retired Ops" },
    { 0xC2, 0xF, NULL, "RETIRED_BRANCHES",
      "Retired branches (conditional, unconditional, exceptions, interrupts)" },
    { 0xC3, 0xF, NULL, "RETIRED_BRANCHES_MISPREDICTED",
      "Retired branches mispredicted" },
    { 0xC4, 0xF, NULL, "RETIRED_TAKEN_BRANCHES",
      "Retired taken branches" },
    { 0xC5, 0xF, NULL, "RETIRED_TAKEN_BRANCHES_MISPREDICTED",
      "Retired taken branches mispredicted" },
    { 0xC6, 0xF, NULL, "RETIRED_FAR_CONTROL_TRANSFERS",
      "Retired far control transfers" },
    { 0xC7, 0xF, NULL, "RETIRED_RESYNC_BRANCHES",
      "Retired resync branches (only non-control transfer branches counted)" },
    { 0xCD, 0xF, NULL, "INTERRUPTS_MASKED_CYCLES",
      "Interrupts masked cycles (IF=0)" },
    { 0xCE, 0xF, NULL, "INTERRUPTS_MASKED_WHILE_PENDING_CYCLES",
      "Interrupts masked while pending cycles (INTR while IF=0)" },
    { 0xCF, 0xF, NULL, "NUMBER_OF_TAKEN_HARDWARE_INTERRUPTS",
      "Number of taken hardware interrupts" },
};

static const struct perfctr_event_set k7_official_event_set = {
    .cpu_type = PERFCTR_X86_AMD_K7,
    .event_prefix = "K7_",
    .include = NULL,
    .nevents = ARRAY_SIZE(k7_official_events),
    .events = k7_official_events,
};

/* also in K8 */
static const struct perfctr_unit_mask_7 k7_um_seg_reg = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x3F,
      .nvalues = 7 },
    { { 0x40, "HS" },		/* what's this? */
      { 0x20, "GS" },
      { 0x10, "FS" },
      { 0x08, "DS" },
      { 0x04, "SS" },
      { 0x02, "CS" },
      { 0x01, "ES" } }
};

/* not in K8 */
static const struct perfctr_unit_mask_5 k7_um_system_request = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x73,
      .nvalues = 5 },
    { { 0x40, "WB" },
      { 0x20, "WP" },
      { 0x10, "WT" },
      { 0x02, "WC" },
      { 0x01, "UC" } }
};

/* not in K8 */
static const struct perfctr_unit_mask_3 k7_um_snoop_hits = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x04, "L2 (L2 hit and no DC hit)" },
      { 0x02, "Data cache" },
      { 0x01, "Instruction cache" } }
};

/* not in K8 */
static const struct perfctr_unit_mask_2 k7_um_ecc = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x02, "L2 single bit error" },
      { 0x01, "System single bit error" } }
};

/* not in K8 */
static const struct perfctr_unit_mask_4 k7_um_invalidates = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x08, "I invalidates D" },
      { 0x04, "I invalidates I" },
      { 0x02, "D invalidates D" },
      { 0x01, "D invalidates I" } }
};

/* not in K8 */
static const struct perfctr_unit_mask_8 k7_um_L2_requests = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0xFF,
      .nvalues = 8 },
    { { 0x80, "Data block write from the L2 (TBL RMW)" },
      { 0x40, "Data block write from the DC" },
      { 0x20, "Data block write from the system" },
      { 0x10, "Data block read data store" },
      { 0x08, "Data block read data load" },
      { 0x04, "Data block read instruction" },
      { 0x02, "Tag write" },
      { 0x01, "Tag read" } }
};

static const struct perfctr_event k7_unofficial_events[] = {
    { 0x20, 0xF, UM(k7_um_seg_reg), "SEGMENT_REGISTER_LOADS", /* also in K8 */
      "Segment register loads" },
    { 0x21, 0xF, NULL, "STORES_TO_ACTIVE_INSTRUCTION_STREAM", /* also in K8 as SELF_MODIFY_RESYNC */
      "Stores to active instruction stream" },
    { 0x64, 0xF, NULL, "DRAM_SYSTEM_REQUESTS", /* not in K8 */
      "DRAM system requests" },
    { 0x65, 0xF, UM(k7_um_system_request), "SYSTEM_REQUESTS_WITH_THE_SELECTED_TYPE", /* not in K8 */
      "System requests with the selected type" },
    { 0x73, 0xF, UM(k7_um_snoop_hits), "SNOOP_HITS", /* not in K8 */
      "Snoop hits" },
    { 0x74, 0xF, UM(k7_um_ecc), "SINGLE_BIT_ECC_ERRORS_DETECTED_CORRECTED", /* not in K8 */ /* XXX: was SINGLE_BIT_ECC_ERRORS_DETECTED_OR_CORRECTED */
      "Single-bit ECC errors detected/corrected" },
    { 0x75, 0xF, UM(k7_um_invalidates), "INTERNAL_CACHE_LINE_INVALIDATES", /* not in K8 */
      "Internal cache line invalidates" },
    { 0x76, 0xF, NULL, "CYCLES_PROCESSOR_IS_RUNNING", /* also in K8 */
      "Cycles processor is running (not in HLT or STPCLK)" },
    { 0x79, 0xF, UM(k7_um_L2_requests), "L2_REQUESTS", /* not in K8 */
      "L2 requests" },
    { 0x7A, 0xF, NULL, "CYCLES_THAT_AT_LEAST_ONE_FILL_REQUEST_WAITED_TO_USE_THE_L2", /* not in K8 */
      "Cycles that at least one fill request waited to use the L2" },
    { 0x82, 0xF, NULL, "INSTRUCTION_CACHE_REFILLS_FROM_L2", /* also in K8 */
      "Instruction cache refills from L2" },
    { 0x83, 0xF, NULL, "INSTRUCTION_CACHE_REFILLS_FROM_SYSTEM", /* also in K8 */
      "Instruction cache refills from system" },
    { 0x86, 0xF, NULL, "SNOOP_RESYNCS", /* also in K8 */
      "Snoop resyncs" },
    { 0x87, 0xF, NULL, "INSTRUCTION_FETCH_STALL_CYCLES", /* also in K8 */
      "Instruction fetch stall cycles" },
    { 0x88, 0xF, NULL, "RETURN_STACK_HITS", /* also in K8 */
      "Instruction cache hits" },
    { 0x89, 0xF, NULL, "RETURN_STACK_OVERFLOW", /* also in K8 */
      "Return stack overflow" },
    { 0xC8, 0xF, NULL, "RETIRED_NEAR_RETURNS", /* also in K8 */
      "Retired near returns" },
    { 0xC9, 0xF, NULL, "RETIRED_NEAR_RETURNS_MISPREDICTED", /* also in K8 */
      "Retired near returns mispredicted" },
    { 0xCA, 0xF, NULL, "RETIRED_INDIRECT_BRANCHES_WITH_TARGET_MISPREDICTED", /* also in K8 */
      "Retired indirect branches with target mispredicted" },
    { 0xD0, 0xF, NULL, "INSTRUCTION_DECODER_EMPTY", /* also in K8 */
      "Instruction decoder empty" },
    { 0xD1, 0xF, NULL, "DISPATCH_STALLS", /* also in K8 */
      "Dispatch stalls (event masks D2h through DAh below combined)" },
    { 0xD2, 0xF, NULL, "BRANCH_ABORT_TO_RETIRE", /* also in K8 */ /* XXX: was BRANCH_ABORTS_TO_RETIRE */
      "Branch abort to retire" },
    { 0xD3, 0xF, NULL, "SERIALIZE", /* also in K8 */
      "Serialize" },
    { 0xD4, 0xF, NULL, "SEGMENT_LOAD_STALL", /* also in K8 */
      "Segment load stall" },
    { 0xD5, 0xF, NULL, "ICU_FULL", /* also in K8 */
      "ICU full" },
    { 0xD6, 0xF, NULL, "RESERVATION_STATIONS_FULL", /* also in K8 */
      "Reservation stations full" },
    { 0xD7, 0xF, NULL, "FPU_FULL", /* also in K8 */
      "FPU full" },
    { 0xD8, 0xF, NULL, "LS_FULL", /* also in K8 */
      "LS full" },
    { 0xD9, 0xF, NULL, "ALL_QUIET_STALL", /* also in K8 */
      "All quiet stall" },
    { 0xDA, 0xF, NULL, "FAR_TRANSFER_OR_RESYNC_BRANCH_PENDING", /* also in K8 */
      "Fall transfer or resync branch pending" },
    { 0xDC, 0xF, NULL, "BREAKPOINT_MATCHES_FOR_DR0", /* also in K8 */
      "Breakpoint matches for DR0" },
    { 0xDD, 0xF, NULL, "BREAKPOINT_MATCHES_FOR_DR1", /* also in K8 */
      "Breakpoint matches for DR1" },
    { 0xDE, 0xF, NULL, "BREAKPOINT_MATCHES_FOR_DR2", /* also in K8 */
      "Breakpoint matches for DR2" },
    { 0xDF, 0xF, NULL, "BREAKPOINT_MATCHES_FOR_DR3", /* also in K8 */
      "Breakpoint matches for DR3" },
};

const struct perfctr_event_set perfctr_k7_event_set = {
    .cpu_type = PERFCTR_X86_AMD_K7,
    .event_prefix = "K7_",
    .include = &k7_official_event_set,
    .nevents = ARRAY_SIZE(k7_unofficial_events),
    .events = k7_unofficial_events,
};

/*
 * AMD K8 events.
 *
 * Some events are described as being "Revision B and later", but
 * AMD does not document how to distinguish Revision B processors
 * from earlier ones.
 */

static const struct perfctr_unit_mask_6 k8_um_fpu_ops = {
    /* Revision B and later */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x3F,
      .nvalues = 6 },
    { { 0x01, "Add pipe ops excluding junk ops" },
      { 0x02, "Multiply pipe ops excluding junk ops" },
      { 0x04, "Store pipe ops excluding junk ops" },
      { 0x08, "Add pipe junk ops" },
      { 0x10, "Multiply pipe junk ops" },
      { 0x20, "Store pipe junk ops" } }
};

static const struct perfctr_unit_mask_2 k8_um_ecc = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x01, "Scrubber error" },
      { 0x02, "Piggyback scrubber errors" } }
};

static const struct perfctr_unit_mask_3 k8_um_prefetch = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "Load" },
      { 0x02, "Store" },
      { 0x04, "NTA" } }
};

static const struct perfctr_unit_mask_5 k8_um_int_L2_req = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x1F,
      .nvalues = 5 },
    { { 0x01, "IC fill" },
      { 0x02, "DC fill" },
      { 0x04, "TLB reload" },
      { 0x08, "Tag snoop request" },
      { 0x10, "Cancelled request" } }
};

static const struct perfctr_unit_mask_3 k8_um_fill_req = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "IC fill" },
      { 0x02, "DC fill" },
      { 0x04, "TLB reload" } }
};

static const struct perfctr_unit_mask_2 k8_um_fill_L2 = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x01, "Dirty L2 victim" },
      { 0x02, "Victim from L2" } }
};

static const struct perfctr_unit_mask_4 k8_um_fpu_instr = {
    /* Revision B and later */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "x87 instructions" },
      { 0x02, "Combined MMX & 3DNow! instructions" },
      { 0x04, "Combined packed SSE and SSE2 instructions" },
      { 0x08, "Combined scalar SSE and SSE2 instructions" } }
};

static const struct perfctr_unit_mask_3 k8_um_fpu_fastpath = {
    /* Revision B and later */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "With low op in position 0" },
      { 0x02, "With low op in position 1" },
      { 0x04, "With low op in position 2" } }
};

static const struct perfctr_unit_mask_4 k8_um_fpu_exceptions = {
    /* Revision B and later */
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "x87 reclass microfaults" },
      { 0x02, "SSE retype microfaults" },
      { 0x04, "SSE reclass microfaults" },
      { 0x08, "SSE and x87 microtraps" } }
};

static const struct perfctr_unit_mask_3 k8_um_page_access = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "Page hit" },
      { 0x02, "Page miss" },
      { 0x04, "Page conflict" } }
};

static const struct perfctr_unit_mask_3 k8_um_turnaround = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x07,
      .nvalues = 3 },
    { { 0x01, "DIMM turnaround" },
      { 0x02, "Read to write turnaround" },
      { 0x04, "Write to read turnaround" } }
};

static const struct perfctr_unit_mask_4 k8_um_saturation = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "Memory controller high priority bypass" },
      { 0x02, "Memory controller low priority bypass" },
      { 0x04, "DRAM controller interface bypass" },
      { 0x08, "DRAM controller queue bypass" } }
};

static const struct perfctr_unit_mask_7 k8_um_sized_commands = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x7F,
      .nvalues = 7 },
    { { 0x01, "NonPostWrSzByte" },
      { 0x02, "NonPostWrSzDword" },
      { 0x04, "PostWrSzByte" },
      { 0x08, "PostWrSzDword" },
      { 0x10, "RdSzByte" },
      { 0x20, "RdSzDword" },
      { 0x40, "RdModWr" } }
};

static const struct perfctr_unit_mask_4 k8_um_probe = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "Probe miss" },
      { 0x02, "Probe hit" },
      { 0x04, "Probe hit dirty without memory cancel" },
      { 0x08, "Probe hit dirty with memory cancel" } }
};

static const struct perfctr_unit_mask_4 k8_um_ht = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x0F,
      .nvalues = 4 },
    { { 0x01, "Command sent" },
      { 0x02, "Data sent" },
      { 0x04, "Buffer release sent" },
      { 0x08, "Nop sent" } }
};

static const struct perfctr_event k8_common_events[] = {
    { 0x00, 0xF, UM(k8_um_fpu_ops), "DISPATCHED_FPU_OPS",
      /* Revision B and later */
      "Dispatched FPU ops" },
    { 0x01, 0xF, NULL, "NO_FPU_OPS",
      /* Revision B and later */
      "Cycles with no FPU ops retired" },
    { 0x02, 0xF, NULL, "FAST_FPU_OPS",
      /* Revision B and later */
      "Dispatched FPU ops that use the fast flag interface" },
    { 0x20, 0xF, UM(k7_um_seg_reg), "SEG_REG_LOAD",
      "Segment register load" },
    { 0x21, 0xF, NULL, "SELF_MODIFY_RESYNC",
      "Microarchitectural resync caused by self modifying code" },
    { 0x22, 0xF, NULL, "LS_RESYNC_BY_SNOOP",
      /* similar to 0x86, but LS unit instead of IC unit */
      "Microarchitectural resync caused by snoop" },
    { 0x23, 0xF, NULL, "LS_BUFFER_FULL",
      "LS Buffer 2 Full" },
    /* 0x24: changed in Revision C */
    { 0x25, 0xF, NULL, "OP_LATE_CANCEL",
      "Microarchitectural late cancel of an operation" },
    { 0x26, 0xF, NULL, "CFLUSH_RETIRED",
      "Retired CFLUSH instructions" },
    { 0x27, 0xF, NULL, "CPUID_RETIRED",
      "Retired CPUID instructions" },
    /* 0x40-0x47: from K7 official event set */
    { 0x48, 0xF, NULL, "ACCESS_CANCEL_LATE",
      "Microarchitectural late cancel of an access" },
    { 0x49, 0xF, NULL, "ACCESS_CANCEL_EARLY",
      "Microarchitectural early cancel of an access" },
    { 0x4A, 0xF, UM(k8_um_ecc), "ECC_BIT_ERR",
      "One bit ECC error recorded found by scrubber" },
    { 0x4B, 0xF, UM(k8_um_prefetch), "DISPATCHED_PRE_INSTRS",
      "Dispatched prefetch instructions" },
    /* 0x4C: added in Revision C */
    { 0x76, 0xF, NULL, "CPU_CLK_UNHALTED", /* XXX: was CYCLES_PROCESSOR_IS_RUNNING */
      "Cycles processor is running (not in HLT or STPCLK)" },
    { 0x7D, 0xF, UM(k8_um_int_L2_req), "BU_INT_L2_REQ",
      "Internal L2 request" },
    { 0x7E, 0xF, UM(k8_um_fill_req), "BU_FILL_REQ",
      "Fill request that missed in L2" },
    { 0x7F, 0xF, UM(k8_um_fill_L2), "BU_FILL_L2",
      "Fill into L2" },
    /* 0x80-0x81: from K7 official event set */
    { 0x82, 0xF, NULL, "IC_REFILL_FROM_L2",
      "Refill from L2" },
    { 0x83, 0xF, NULL, "IC_REFILL_FROM_SYS",
      "Refill from system" },
    /* 0x84-0x85: from K7 official event set */
    { 0x86, 0xF, NULL, "IC_RESYNC_BY_SNOOP",
      /* similar to 0x22, but IC unit instead of LS unit */
      "Microarchitectural resync caused by snoop" },
    { 0x87, 0xF, NULL, "IC_FETCH_STALL",
      "Instruction fetch stall" },
    { 0x88, 0xF, NULL, "IC_STACK_HIT",
      "Return stack hit" },
    { 0x89, 0xF, NULL, "IC_STACK_OVERFLOW",
      "Return stack overflow" },
    /* 0xC0-0xC7: from K7 official event set */
    { 0xC8, 0xF, NULL, "RETIRED_NEAR_RETURNS",
      "Retired near returns" },
    { 0xC9, 0xF, NULL, "RETIRED_RETURNS_MISPREDICT",
      "Retired near returns mispredicted" },
    { 0xCA, 0xF, NULL, "RETIRED_BRANCH_MISCOMPARE",
      "Retired taken branches mispredicted due to address miscompare" },
    { 0xCB, 0xF, UM(k8_um_fpu_instr), "RETIRED_FPU_INSTRS",
      /* Revision B and later */
      "Retired FPU instructions" },
    { 0xCC, 0xF, UM(k8_um_fpu_fastpath), "RETIRED_FASTPATH_INSTRS",
      /* Revision B and later */
      "Retired fastpath double op instructions" },
    /* 0xCD-0xCF: from K7 official event set */
    { 0xD0, 0xF, NULL, "DECODER_EMPTY",
      "Nothing to dispatch (decoder empty)" },
    { 0xD1, 0xF, NULL, "DISPATCH_STALLS",
      "Dispatch stalls (events 0xD2-0xDA combined)" },
    { 0xD2, 0xF, NULL, "DISPATCH_STALL_FROM_BRANCH_ABORT",
      "Dispatch stall from branch abort to retire" },
    { 0xD3, 0xF, NULL, "DISPATCH_STALL_SERIALIZATION",
      "Dispatch stall for serialization" },
    { 0xD4, 0xF, NULL, "DISPATCH_STALL_SEG_LOAD",
      "Dispatch stall for segment load" },
    { 0xD5, 0xF, NULL, "DISPATCH_STALL_REORDER_BUFFER",
      "Dispatch stall when reorder buffer is full" },
    { 0xD6, 0xF, NULL, "DISPATCH_STALL_RESERVE_STATIONS",
      "Dispatch stall when reservation stations are full" },
    { 0xD7, 0xF, NULL, "DISPATCH_STALL_FPU",
      "Dispatch stall when FPU is full" },
    { 0xD8, 0xF, NULL, "DISPATCH_STALL_LS",
      "Dispatch stall when LS is full" },
    { 0xD9, 0xF, NULL, "DISPATCH_STALL_QUIET_WAIT",
      "Dispatch stall when waiting for all to be quiet" },
    { 0xDA, 0xF, NULL, "DISPATCH_STALL_PENDING",
      "Dispatch stall when far control transfer or resync branch is pending" },
    { 0xDB, 0xF, UM(k8_um_fpu_exceptions), "FPU_EXCEPTIONS",
      /* Revision B and later */
      "FPU exceptions" },
    { 0xDC, 0xF, NULL, "DR0_BREAKPOINTS",
      "Number of breakpoints for DR0" },
    { 0xDD, 0xF, NULL, "DR1_BREAKPOINTS",
      "Number of breakpoints for DR1" },
    { 0xDE, 0xF, NULL, "DR2_BREAKPOINTS",
      "Number of breakpoints for DR2" },
    { 0xDF, 0xF, NULL, "DR3_BREAKPOINTS",
      "Number of breakpoints for DR3" },
    { 0xE0, 0xF, UM(k8_um_page_access), "MEM_PAGE_ACCESS",
      "Memory controller page access" },
    { 0xE1, 0xF, NULL, "MEM_PAGE_TBL_OVERFLOW",
      "Memory controller page table overflow" },
    { 0xE2, 0xF, NULL, "DRAM_SLOTS_MISSED",
      "Memory controller DRAM command slots missed (in MemClks)" },
    { 0xE3, 0xF, UM(k8_um_turnaround), "MEM_TURNAROUND",
      "Memory controller turnaround" },
    { 0xE4, 0xF, UM(k8_um_saturation), "MEM_BYPASS_SAT",
      "Memory controller bypass counter saturation" },
    { 0xEB, 0xF, UM(k8_um_sized_commands), "SIZED_COMMANDS",
      "Sized commands" },
    { 0xEC, 0xF, UM(k8_um_probe), "PROBE_RESULT",
      "Probe result" },
    { 0xF6, 0xF, UM(k8_um_ht), "HYPERTRANSPORT_BUS0_WIDTH",
      "Hypertransport (tm) bus 0 bandwidth" },
    { 0xF7, 0xF, UM(k8_um_ht), "HYPERTRANSPORT_BUS1_WIDTH",
      "Hypertransport (tm) bus 1 bandwidth" },
    { 0xF8, 0xF, UM(k8_um_ht), "HYPERTRANSPORT_BUS2_WIDTH",
      "Hypertransport (tm) bus 2 bandwidth" },
};

static const struct perfctr_event_set k8_common_event_set = {
    .cpu_type = PERFCTR_X86_AMD_K8,
    .event_prefix = "K8_",
    .include = &k7_official_event_set,
    .nevents = ARRAY_SIZE(k8_common_events),
    .events = k8_common_events,
};

static const struct perfctr_event k8_events[] = {
     { 0x24, 0xF, NULL, "LOCKED_OP", /* unit mask changed in Rev. C */
       "Locked operation" },
};

const struct perfctr_event_set perfctr_k8_event_set = {
    .cpu_type = PERFCTR_X86_AMD_K8,
    .event_prefix = "K8_",
    .include = &k8_common_event_set,
    .nevents = ARRAY_SIZE(k8_events),
    .events = k8_events,
};

/*
 * K8 Revision C. Starts at CPUID 0xF58 for Opteron/Athlon64FX and
 * CPUID 0xF48 for Athlon64. (CPUID 0xF51 is Opteron Revision B3.)
 */

static const struct perfctr_unit_mask_3 k8c_um_locked_op = {
     { .type = perfctr_um_type_bitmask,
       .default_value = 0x01,
       .nvalues = 3 },
     { { 0x01, "Number of lock instructions executed" },
       { 0x02, "Number of cycles spent in the lock request/grant stage" },
       { 0x04, "Number of cycles a lock takes to complete once it is "
	 "non-speculative and is the oldest load/store operation "
	 "(non-speculative cycles in Ls2 entry 0)" } }
};

static const struct perfctr_unit_mask_2 k8c_um_lock_accesses = {
    { .type = perfctr_um_type_bitmask,
      .default_value = 0x03,
      .nvalues = 2 },
    { { 0x01, "Number of dcache accesses by lock instructions" },
      { 0x02, "Number of dcache misses by lock instructions" } }
};

static const struct perfctr_event k8c_events[] = {
     { 0x24, 0xF, UM(k8c_um_locked_op), "LOCKED_OP", /* unit mask changed */
       "Locked operation" },
     { 0x4C, 0xF, UM(k8c_um_lock_accesses), "LOCK_ACCESSES",
       "DCACHE accesses by locks" },
};

const struct perfctr_event_set perfctr_k8c_event_set = {
    .cpu_type = PERFCTR_X86_AMD_K8C,
    .event_prefix = "K8C_",
    .include = &k8_common_event_set,
    .nevents = ARRAY_SIZE(k8c_events),
    .events = k8c_events,
};
