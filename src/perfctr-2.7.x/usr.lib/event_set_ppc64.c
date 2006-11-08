/* Maynard
 * Descriptions of the events available for different processor types.
 *
 */
#include <stddef.h>	/* for NULL */
#include "libperfctr.h"
#include "event_set.h"

/*
 * XXX: a few events use the TBSEL and THRESHOLD fields in MMCR0.
 * They should have unit mask descriptors.
 */

/*
 * PowerPC common events for PMC1-PMC4, introduced in 604.
 */

static const struct perfctr_event ppc64_common_events[] = {
    { 0x00, 0x0F, NULL, "NOTHING",
      "Nothing. Register counter holds current value" },
    { 0x01, 0x0F, NULL, "PROCESSOR_CYCLES",
      "Processor cycles. Count every cycle" },
    { 0x02, 0x0F, NULL, "INSTRUCTIONS_COMPLETED",
      "Number of instructions completed. Does not include folded branches" },
    { 0x03, 0x0F, NULL, "TBL_BIT_TRANSITIONS", /* XXX: depends on MMCR0[TBSEL] */
      "Time-base (lower) bit transition" },
    { 0x04, 0x0F, NULL, "INSTRUCTIONS_DISPATCHED",
      "Number of instructions dispatched" },
};

static const struct perfctr_event_set ppc64_common_event_set = {
    .cpu_type = PERFCTR_PPC64_POWER4,
    .event_prefix = "PM_",
    .include = NULL,
    .nevents = ARRAY_SIZE(ppc64_common_events),
    .events = ppc64_common_events,
};


static const struct perfctr_event_set * const cpu_event_set[] = {
    [PERFCTR_PPC64_POWER4] = &ppc64_common_event_set,
    [PERFCTR_PPC64_POWER4p] = &ppc64_common_event_set,
    [PERFCTR_PPC64_970] = &ppc64_common_event_set,
    [PERFCTR_PPC64_970MP] = &ppc64_common_event_set,
//    [PERFCTR_PPC_604e] = &perfctr_ppc604e_event_set,
//    [PERFCTR_PPC_750] = &perfctr_ppc750_event_set,
};

const struct perfctr_event_set *perfctr_cpu_event_set(unsigned cpu_type)
{
    if( cpu_type >= ARRAY_SIZE(cpu_event_set) )
	return 0;
    return cpu_event_set[cpu_type];
}
