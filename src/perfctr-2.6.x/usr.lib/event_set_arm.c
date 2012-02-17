/* $Id: event_set_arm.c,v 1.1.2.1 2007/02/11 20:15:03 mikpe Exp $
 * Descriptions of the events available for different processor types.
 *
 * Copyright (C) 2005-2007  Mikael Pettersson
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * XScale 1 and 2 events for PMC1-PMC4.
 */

static const struct perfctr_event xsc1_events[] = {
    { 0x00, 0x0F, NULL, "IC_MISS",
      "Instruction cache miss requires fetch from external memory" },
    { 0x01, 0x0F, NULL, "IC_CANNOT_DELIVER",
      "Instruction cache cannot deliver an instruction" },
    { 0x02, 0x0F, NULL, "DATA_DEP_STALL",
      "Stall due to a data dependency" },
    { 0x03, 0x0F, NULL, "ITLB_MISS",
      "Instruction TLB miss" },
    { 0x04, 0x0F, NULL, "DTLB_MISS",
      "Data TLB miss" },
    { 0x05, 0x0F, NULL, "BR_INST_EXEC",
      "Branch instruction executed" },
    { 0x06, 0x0F, NULL, "BR_MISPRED",
      "Branch mispredicted" },
    { 0x07, 0x0F, NULL, "INST_EXEC",
      "Instruction executed" },
    { 0x08, 0x0F, NULL, "DC_FULL_CYCLES",
      "Stall because the data cache buffers are full (cycles)" },
    { 0x09, 0x0F, NULL, "DC_FULL_OCCURRENCES",
      "Stall because the data cache buffers are full (occurrences)" },
    { 0x0A, 0x0F, NULL, "DC_ACCESS",
      "Data cache access" },
    { 0x0B, 0x0F, NULL, "DC_MISS",
      "Data cache miss" },
    { 0x0C, 0x0F, NULL, "DC_WRITE_BACK",
      "Data cache write-back" },
    { 0x0D, 0x0F, NULL, "SW_CHANGED_PC",
      "Software changed the PC" },
    { 0xFF, 0x0F, NULL, "IDLE",
      "Power saving event" },
};

static const struct perfctr_event_set perfctr_xsc1_event_set = {
    .cpu_type = PERFCTR_ARM_XSC1,
    .event_prefix = "XSC1_",
    .include = NULL,
    .nevents = ARRAY_SIZE(xsc1_events),
    .events = xsc1_events,
};

/*
 * Helper function to translate a cpu_type code to an event_set pointer.
 */

static const struct perfctr_event_set * const cpu_event_set[] = {
    [PERFCTR_ARM_XSC1] = &perfctr_xsc1_event_set,
    [PERFCTR_ARM_XSC2] = &perfctr_xsc1_event_set,
};

const struct perfctr_event_set *perfctr_cpu_event_set(unsigned cpu_type)
{
    if (cpu_type >= ARRAY_SIZE(cpu_event_set))
	return 0;
    return cpu_event_set[cpu_type];
}
