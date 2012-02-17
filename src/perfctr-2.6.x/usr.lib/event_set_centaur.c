/* $Id: event_set_centaur.c,v 1.1 2003/02/16 21:08:54 mikpe Exp $
 * Performance counter event descriptions for Centaur chips:
 * IDT WinChip C6/2/3 and VIA C3.
 *
 * Copyright (C) 2003  Mikael Pettersson
 *
 * References
 * ----------
 * [WinChip C6] "WinChip C6 Processor Data Sheet".
 * [WinChip 2A] "WinChip 2 Processor Version A Data Sheet".
 * [WinChip 3] "WinChip 3 Processor Data Sheet".
 * (at http://www.centtech.com/)
 *
 * [VIA C3] "VIA C3 Samuel 2 Processor Datasheet", Ver. 1.03, April 2001.
 * Note: The C3 was originally called "Cyrix III", but it is a Centaur
 * design developed as a replacement for Cyrix' "Joshua".
 * (at http://www.viatech.com/)
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * Centaur WinChip C6 events.
 * Note: The manual lists the codes in decimal, not hex as done here.
 */

static const struct perfctr_event wcc6_events[] = {
    { 0x00, 0x3, NULL, "INTERNAL_CLOCKS" },
    { 0x01, 0x3, NULL, "VALID_CYCLES_REACHING_WRITEBACKS" },
    { 0x02, 0x3, NULL, "X86_INSTRUCTIONS" },
    { 0x47, 0x3, NULL, "DATA_READ_CACHE_MISSES" },
    { 0x4A, 0x3, NULL, "DATA_WRITE_CACHE_MISSES" },
    { 0x63, 0x3, NULL, "INSTRUCTION_FETCH_CACHE_MISSES" },
};

const struct perfctr_event_set perfctr_wcc6_event_set = {
    .cpu_type = PERFCTR_X86_WINCHIP_C6,
    .event_prefix = "WCC6_",
    .include = NULL,
    .nevents = ARRAY_SIZE(wcc6_events),
    .events = wcc6_events,
};

/*
 * Centaur WinChip 2 and 3 events.
 * Note: The manual lists the codes in decimal, not hex as done here.
 */

static const struct perfctr_event wc2_events[] = {
    { 0x00, 0x3, NULL, "DATA_READ" },
    { 0x01, 0x3, NULL, "DATA_WRITE" },
    { 0x02, 0x3, NULL, "DATA_TLB_MISS" },
    { 0x03, 0x3, NULL, "DATA_READ_CACHE_MISS" },
    { 0x04, 0x3, NULL, "DATA_WRITE_CACHE_MISS" },
    { 0x06, 0x3, NULL, "DATA_CACHE_WRITEBACKS" },
    { 0x08, 0x3, NULL, "DATA_CACHE_SNOOP_HITS" },
    { 0x09, 0x3, NULL, "PUSH_PUSH_POP_POP_PAIRING" },
    { 0x0B, 0x3, NULL, "MISALIGNED_DATA_MEMORY_NOT_IO" },
    { 0x0C, 0x3, NULL, "CODE_READ" },
    { 0x0D, 0x3, NULL, "CODE_TLB_MISS" },
    { 0x0E, 0x3, NULL, "INSTRUCTION_FETCH_CACHE_MISS" },
    { 0x13, 0x3, NULL, "BHT_HITS" },
    { 0x14, 0x3, NULL, "BHT_CANDIDATE" },
    { 0x16, 0x3, NULL, "INSTRUCTIONS_EXECUTED" },
    { 0x17, 0x3, NULL, "INSTRUCTIONS_IN_PIPE_2" },
    { 0x18, 0x3, NULL, "BUS_UTILIZATION" },
    { 0x1D, 0x3, NULL, "IO_READ_OR_WRITE_CYCLE" },
    { 0x28, 0x3, NULL, "DATA_READ_OR_DATA_WRITE" },
    { 0x2B, 0x1, NULL, "MMX_INSTRUCTIONS_U_PIPE" },
    { 0x2B, 0x2, NULL, "MMX_INSTRUCTIONS_V_PIPE" },
    { 0x37, 0x1, NULL, "RETURNS_PREDICTED_INCORRECTLY" },
    { 0x37, 0x2, NULL, "RETURNS_PREDICTED_CORRECTLY" },
    { 0x3F, 0x3, NULL, "INTERNAL_CLOCKS" },
};

const struct perfctr_event_set perfctr_wc2_event_set = {
    .cpu_type = PERFCTR_X86_WINCHIP_2,
    .event_prefix = "WC2_",
    .include = NULL,
    .nevents = ARRAY_SIZE(wc2_events),
    .events = wc2_events,
};

/*
 * VIA C3 events.
 * This processor is a Centaur design, tweaked to look like a Celeron.
 * Its perfctr MSRs have the same addresses as in the P6, but PERFCTR0
 * is an alias for the TSC and EVNTSEL0 is read-only. It appears that
 * rdpmc(0) returns the TSC truncated to 40 bits. Only EVNTSEL1 and
 * PERFCTR1 can be used. EVNTSEL1 has a different format than in P6: the
 * event selection field is 9 bits, and no other fields are defined.
 * The data sheet only lists the three events defined below.
 */

static const struct perfctr_event vc3_events[] = {
    { 0x079, 0x2, NULL, "INTERNAL_CLOCKS" },
    { 0x0C0, 0x2, NULL, "INSTRUCTIONS_EXECUTED" },
    { 0x1C0, 0x2, NULL, "INSTRUCTIONS_EXECUTED_AND_STRING_ITERATIONS" },
};

const struct perfctr_event_set perfctr_vc3_event_set = {
    .cpu_type = PERFCTR_X86_VIA_C3,
    .event_prefix = "VC3_",
    .include = NULL,
    .nevents = ARRAY_SIZE(vc3_events),
    .events = vc3_events,
};
