/* $Id: event_set_p5.c,v 1.1 2003/02/16 21:08:54 mikpe Exp $
 * Performance counter event descriptions for Intel P5 and P5 MMX
 * processors, and Cyrix 6x86/MII/III processors.
 *
 * Copyright (C) 2003  Mikael Pettersson
 *
 * References
 * ----------
 * [IA32, Volume 3] "Intel Architecture Software Developer's Manual,
 * Volume 3: System Programming Guide". Intel document number 245472-009.
 * (at http://developer.intel.com/)
 *
 * [Cyrix 6x86MX] "Cyrix 6x86MX Processor".
 * [Cyrix MII] "Cyrix M II Data Book".
 * [Cyrix III] "Cyrix III Processor DataBook" Ver. 1.0, 1/25/00.
 * Note: This "Cyrix III" was code-named "Joshua", and it was apparently
 * cancelled by VIA due to disappointing performance.
 * (MII and III docs at http://www.viatech.com/)
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * Intel Pentium (P5) events.
 */

static const struct perfctr_event p5_events[] = {
    { 0x00, 0x3, NULL, "DATA_READ",
      "Number of memory data reads (internal data cache hit and "
      "miss combined)." },
    { 0x01, 0x3, NULL, "DATA_WRITE",
      "Number of memory data writes (internal data cache hit and "
      "miss combined), I/O is not included." },
    { 0x02, 0x3, NULL, "DATA_TLB_MISS",
      "Number of misses to the data cache translation look-aside "
      "buffer." },
    { 0x03, 0x3, NULL, "DATA_READ_MISS",
      "Number of memory read accesses that miss the internal data "
      "cache whether or not the access is cacheable or noncacheable." },
    { 0x04, 0x3, NULL, "DATA_WRITE_MISS",
      "Number of memory write accesses that miss the internal data "
      "cache whether or not the access is cacheable or noncacheable." },
    { 0x05, 0x3, NULL, "WRITE_HIT_TO_M_OR_E_STATE_LINES",
      "Number of write hits to exclusive or modified lines in the "
      "data cache." },
    { 0x06, 0x3, NULL, "DATA_CACHE_LINES_WRITTEN_BACK",
      "Number of dirty lines (all) that are written back, regardless "
      "of the cause." },
    { 0x07, 0x3, NULL, "EXTERNAL_SNOOPS",
      "Number of accepted external snoops whether they hit in the code "
      "cache or data cache or neither." },
    { 0x08, 0x3, NULL, "EXTERNAL_DATA_CACHE_SNOOP_HITS",
      "Number of external snoops to the data cache." },
    { 0x09, 0x3, NULL, "MEMORY_ACCESSES_IN_BOTH_PIPES",
      "Number of data memory reads or writes that are paired in both "
      "pipes of the pipeline." },
    { 0x0A, 0x3, NULL, "BANK_CONFLICTS",
      "Number of actual bank conflicts." },
    { 0x0B, 0x3, NULL, "MISALIGNED_DATA_MEMORY_OR_IO_REFERENCES",
      "Number of memory or I/O reads or writes that are misaligned." },
    { 0x0C, 0x3, NULL, "CODE_READ",
      "Number of instruction reads whether the read is cacheable or "
      "noncacheable." },
    { 0x0D, 0x3, NULL, "CODE_TLB_MISS",
      "Number of instruction reads that miss the code TLB whether "
      "the read is cacheable or noncacheable." },
    { 0x0E, 0x3, NULL, "CODE_CACHE_MISS",
      "Number of instruction reads that miss the internal code cache "
      "whether the read is cacheable or noncacheable." },
    { 0x0F, 0x3, NULL, "ANY_SEGMENT_REGISTER_LOADED",
      "Number of writes into any segment register in real or protected "
      "mode including the LDTR, GDTR, IDTR, and TR." },
    /* 0x10: reserved */
    /* 0x11: reserved */
    { 0x12, 0x3, NULL, "BRANCHES",
      "Number of taken and not taken branches, including conditional "
      "branches, jumps, calls, returns, software interrupts, and "
      "interrupt returns." },
    { 0x13, 0x3, NULL, "BTB_HITS",
      "Number of BTB hits that occur." },
    { 0x14, 0x3, NULL, "TAKEN_BRANCH_OR_BTB_HIT",
      "Number of taken branches or BTB hits that occur." },
    { 0x15, 0x3, NULL, "PIPELINE_FLUSHES",
      "Number of pipeline flushes that occur." },
    { 0x16, 0x3, NULL, "INSTRUCTIONS_EXECUTED",
      "Number of instructions executed (up to two per clock)." },
    { 0x17, 0x3, NULL, "INSTRUCTIONS_EXECUTED_V_PIPE", /* XXX: was INSTRUCTIONS_EXECUTED_IN_V_PIPE */
      "Number of instructions executed in the V_pipe. It indicates "
      "the number of instructions that were paired." },
    { 0x18, 0x3, NULL, "BUS_CYCLE_DURATION",
      "Number of clocks while a bus cycle is in progress." },
    { 0x19, 0x3, NULL, "WRITE_BUFFER_FULL_STALL_DURATION",
      "Number of clocks while the pipeline is stalled due to full "
      "write buffers." },
    { 0x1A, 0x3, NULL, "WAITING_FOR_DATA_MEMORY_READ_STALL_DURATION",
      "Number of clocks while the pipeline is stalled while waiting "
      "for data memory reads." },
    { 0x1B, 0x3, NULL, "STALL_ON_WRITE_TO_AN_E_OR_M_STATE_LINE",
      "Number of stalls on writes to E- or M-state lines." },
    { 0x1C, 0x3, NULL, "LOCKED_BUS_CYCLE",
      "Number of locked bus cycles that occur as the result of "
      "LOCK prefix or LOCK instruction, page-table updates, and "
      "descriptor table updates." },
    { 0x1D, 0x3, NULL, "IO_READ_OR_WRITE_CYCLE",
      "Number of bus cycles directed to I/O space." },
    { 0x1E, 0x3, NULL, "NONCACHEABLE_MEMORY_READS",
      "Number of noncacheable instruction or data memory read bus cycles." },
    { 0x1F, 0x3, NULL, "PIPELINE_AGI_STALLS",
      "Number of adress generation interlock (AGI) stalls." },
    /* 0x20: reserved */
    /* 0x21: reserved */
    { 0x22, 0x3, NULL, "FLOPS",
      "Number of floating-point operations that occur." },
    { 0x23, 0x3, NULL, "BREAKPOINT_MATCH_ON_DR0_REGISTER",
      "Number of matches on DR0 breakpoint." },
    { 0x24, 0x3, NULL, "BREAKPOINT_MATCH_ON_DR1_REGISTER",
      "Number of matches on DR1 breakpoint." },
    { 0x25, 0x3, NULL, "BREAKPOINT_MATCH_ON_DR2_REGISTER",
      "Number of matches on DR2 breakpoint." },
    { 0x26, 0x3, NULL, "BREAKPOINT_MATCH_ON_DR3_REGISTER",
      "Number of matches on DR3 breakpoint." },
    { 0x27, 0x3, NULL, "HARDWARE_INTERRUPTS",
      "Number of taken INTR and NMI interrupts." },
    { 0x28, 0x3, NULL, "DATA_READ_OR_WRITE",
      "Number of memory data reads and/or writes (internal data cache "
      "hit and miss combined)." },
    { 0x29, 0x3, NULL, "DATA_READ_MISS_OR_WRITE_MISS",
      "Number of memory read and/or write accesses that miss the "
      "internal data cache whether or not the acceess is cacheable "
      "or noncacheable." },
};

const struct perfctr_event_set perfctr_p5_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P5,
    .event_prefix = "P5_",
    .include = NULL,
    .nevents = ARRAY_SIZE(p5_events),
    .events = p5_events,
};

/*
 * Intel Pentium MMX (P5MMX) events.
 */

static const struct perfctr_event p5mmx_and_mii_events[] = {
    { 0x2B, 0x1, NULL, "MMX_INSTRUCTIONS_EXECUTED_U_PIPE",
      "Number of MMX instructions executed in the U-pipe." },
    { 0x2B, 0x2, NULL, "MMX_INSTRUCTIONS_EXECUTED_V_PIPE",
      "Number of MMX instructions executed in the V-pipe." },
    { 0x2D, 0x1, NULL, "EMMS_INSTRUCTIONS_EXECUTED",
      "Number of EMMS instructions executed." },
    { 0x2D, 0x2, NULL, "TRANSITIONS_BETWEEN_MMX_AND_FP_INSTRUCTIONS",
      "Number of transitions between MMX and floating-point instructions "
      "or vice versa." },
    { 0x2F, 0x1, NULL, "SATURATING_MMX_INSTRUCTIONS_EXECUTED",
      "Number of saturating MMX instructions executed, independently of "
      "whether they actually saturated." },
    { 0x2F, 0x2, NULL, "SATURATIONS_PERFORMED",
      "Number of MMX instructions that used saturating arithmetic and "
      "that at least one of its results actually saturated." },
    { 0x31, 0x1, NULL, "MMX_INSTRUCTION_DATA_READS",
      "Number of MMX instruction data reads." },
    { 0x32, 0x2, NULL, "TAKEN_BRANCHES",
      "Number of taken branches." },
    { 0x37, 0x1, NULL, "MISPREDICTED_OR_UNPREDICTED_RETURNS",
      "Number of returns predicted incorrectly or not predicted at all." },
    { 0x37, 0x2, NULL, "PREDICTED_RETURNS",
      "Number of predicted returns (whether they are predicted correctly "
      "and incorrectly)." },
    { 0x38, 0x1, NULL, "MMX_MULTIPLY_UNIT_INTERLOCK",
      "Number of clocks the pipe is stalled since the destination of "
      "previous MMX instruction is not ready yet." },
    { 0x38, 0x2, NULL, "MOVD_MOVQ_STORE_STALL_DUE_TO_PREVIOUS_MMX_OPERATION",
      "Number of clocks a MOVD/MOVQ instruction store is stalled in D2 "
      "stage due to a previous MMX operation with a destination to be "
      "used in the store instruction." },
    { 0x39, 0x1, NULL, "RETURNS",
      "Number of returns executed." },
    { 0x3A, 0x1, NULL, "BTB_FALSE_ENTRIES",
      "Number of false entries in the Branch Target Buffer." },
    { 0x3A, 0x2, NULL, "BTB_MISS_PREDICTION_ON_NOT_TAKEN_BRANCH",
      "Number of times the BTB predicted a not-taken branch as taken." },
    { 0x3B, 0x1, NULL, "FULL_WRITE_BUFFER_STALL_DURATION_WHILE_EXECUTING_MMX_INSTRUCTIONS",
      "Number of clocks while the pipeline is stalled due to full write "
      "buffers while executing MMX instructions." },
    { 0x3B, 0x2, NULL, "STALL_ON_MMX_INSTRUCTION_WRITE_TO_E_OR_M_STATE_LINE",
      "Number of clocks during stalls on MMX instructions writing "
      "to E- or M-state lines." },
};

static const struct perfctr_event_set p5mmx_and_mii_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P5MMX,
    .event_prefix = "P5MMX_",
    .include = &perfctr_p5_event_set,
    .nevents = ARRAY_SIZE(p5mmx_and_mii_events),
    .events = p5mmx_and_mii_events,
};

static const struct perfctr_event p5mmx_events[] = {
    { 0x2A, 0x1, NULL, "BUS_OWNERSHIP_LATENCY",
      "The time from LRM bus ownership request to bus ownership granted." },
    { 0x2A, 0x2, NULL, "BUS_OWNERSHIP_TRANSFERS",
      "The number of bus ownership transfers." },
    { 0x2C, 0x1, NULL, "CACHE_M_STATE_LINE_SHARING",
      "Number of times a processor identified a hit to a modified line "
      "due to a memory access in the other processor." },
    { 0x2C, 0x2, NULL, "CACHE_LINE_SHARING",
      "Number of shared data lines in the L1 cache." },
    { 0x2E, 0x1, NULL, "BUS_UTILIZATION_DUE_TO_PROCESSOR_ACTIVITY",
      "Number of clocks the bus is busy due to the processor's own activity." },
    { 0x2E, 0x2, NULL, "WRITES_TO_NONCACHEABLE_MEMORY",
      "Number of write accesses to noncacheable memory." },
    { 0x30, 0x1, NULL, "NUMBER_OF_CYCLES_NOT_IN_HALT_STATE",
      "Number of cycles the processor is not idle due to HLT instruction." },
    { 0x30, 0x2, NULL, "DATA_CACHE_TLB_MISS_STALL_DURATION",
      "Number of clocks the pipeline is stalled due to a data cache "
      "translation look-aside buffer (TLB) miss." },
    { 0x31, 0x2, NULL, "MMX_INSTRUCTION_DATA_READ_MISSES",
      "Number of MMX instruction data read misses." },
    { 0x32, 0x1, NULL, "FLOATING_POINT_STALLS_DURATION",
      "Number of clocks while pipe is stalled due to a floating-point freeze." },
    { 0x33, 0x1, NULL, "D1_STARVATION_AND_FIFO_IS_EMPTY",
      "Number of times D1 stage cannot issue ANY instructions since the "
      "FIFO buffer is empty." },
    { 0x33, 0x2, NULL, "D1_STARVATION_AND_ONLY_ONE_INSTRUCTION_IN_FIFO",
      "Number of times the D1 stage issues just a single instruction since "
      "the FIFO buffer had just one instruction ready." },
    { 0x34, 0x1, NULL, "MMX_INSTRUCTION_DATA_WRITES",
      "Number of data writes caused by MMX instructions." },
    { 0x34, 0x2, NULL, "MMX_INSTRUCTION_DATA_WRITE_MISSES",
      "Number of data write misses caused by MMX instructions." },
    { 0x35, 0x1, NULL, "PIPELINE_FLUSHES_DUE_TO_WRONG_BRANCH_PREDICTIONS",
      "Number of pipeline flushes due to wrong branch prediction resolved "
      "in either the E-stage or the WB-stage." },
    { 0x35, 0x2, NULL, "PIPELINE_FLUSHES_DUE_TO_WRONG_BRANCH_PREDICTIONS_RESOLVED_IN_WB_STAGE",
      "Number of pipeline flushes due to wrong branch prediction resolved "
      "in the WB-stage." },
    { 0x36, 0x1, NULL, "MISALIGNED_DATA_MEMORY_REFERENCE_ON_MMX_INSTRUCTIONS",
      "Number of misaligned data memory references when executing MMX "
      "instructions." },
    { 0x36, 0x2, NULL, "PIPELINE_ISTALL_FOR_MMX_INSTRUCTION_DATA_MEMORY_READS",
      "Number of clocks during pipeline stalls caused by waits from MMX "
      "instructions data memory reads." },
    /* 0x39, counter 1: reserved */
};

const struct perfctr_event_set perfctr_p5mmx_event_set = {
    .cpu_type = PERFCTR_X86_INTEL_P5MMX,
    .event_prefix = "P5MMX_",
    .include = &p5mmx_and_mii_event_set,
    .nevents = ARRAY_SIZE(p5mmx_events),
    .events = p5mmx_events,
};

/*
 * Cyrix 6x86MX, MII, and III events.
 */

static const struct perfctr_event mii_events[] = {
    { 0x039, 0x2, NULL, "RSB_OVERFLOWS" },
    /* NOTE: The manuals list the following events as having codes 40-48.
       However, the 7-bit event code is actually split in the CESR, using
       bits 0-5 and 10, and similarly for the high half of the CESR.
       Since the driver also parses the other fields (bits 6-9) in a user's
       evntsel, the events are listed here with their actual in-CESR values. */
    { 0x400, 0x3, NULL, "L2_TLB_MISSES" },
    { 0x401, 0x3, NULL, "L1_TLB_DATA_MISS" },
    { 0x402, 0x3, NULL, "L1_TLB_CODE_MISS" },
    { 0x403, 0x3, NULL, "L1_TLB_MISS" },
    { 0x404, 0x3, NULL, "TLB_FLUSHES" },
    { 0x405, 0x3, NULL, "TLB_PAGE_INVALIDATES" },
    { 0x406, 0x3, NULL, "TLB_PAGE_INVALIDATES_THAT_HIT" },
    { 0x408, 0x3, NULL, "INSTRUCTIONS_DECODED" },
};

const struct perfctr_event_set perfctr_mii_event_set = {
    .cpu_type = PERFCTR_X86_CYRIX_MII,
    .event_prefix = "MII_",
    .include = &p5mmx_and_mii_event_set,
    .nevents = ARRAY_SIZE(mii_events),
    .events = mii_events,
};
