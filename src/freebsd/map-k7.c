/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-k7.c
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include SUBSTRATE
#include "papiStdEventDefs.h"
#include "map.h"


/****************************************************************************
 K7 SUBSTRATE 
 K7 SUBSTRATE 
 K7 SUBSTRATE (aka Athlon)
 K7 SUBSTRATE
 K7 SUBSTRATE
****************************************************************************/

/*
	NativeEvent_Value_K7Processor must match K7Processor_info 
*/

Native_Event_LabelDescription_t K7Processor_info[] =
{
	{ "k7-dc-accesses", "Count data cache accesses." },
	{ "k7-dc-misses", "Count data cache misses." },
	{ "k7-dc-refills-from-l2", "Count data cache refills from L2 cache." },
	{ "k7-dc-refills-from-system", "Count data cache refills from system memory." },
	{ "k7-dc-writebacks", "Count data cache writebacks." },
	{ "k7-l1-dtlb-miss-and-l2-dtlb-hits", "Count L1 DTLB misses and L2 DTLB hits." },
	{ "k7-l1-and-l2-dtlb-misses", "Count L1 and L2 DTLB misses." },
	{ "k7-misaligned-references", "Count misaligned data references." },
	{ "k7-ic-fetches", "Count instruction cache fetches." },
	{ "k7-ic-misses", "Count instruction cache misses." },
	{ "k7-l1-itlb-misses", "Count L1 ITLB misses that are L2 ITLB hits." },
	{ "k7-l1-l2-itlb-misses", "Count L1 (and L2) ITLB misses." },
	{ "k7-retired-instructions", "Count all retired instructions." },
	{ "k7-retired-ops", "Count retired ops." },
	{ "k7-retired-branches", "Count all retired branches (conditional, unconditional, exceptions and interrupts)."},
	{ "k7-retired-branches-mispredicted", "Count all misprediced retired branches." },
	{ "k7-retired-taken-branches", "Count retired taken branches." },
	{ "k7-retired-taken-branches-mispredicted", "Count mispredicted taken branches that were retired." },
	{ "k7-retired-far-control-transfers", "Count retired far control transfers." },
	{ "k7-retired-resync-branches", "Count retired resync branches (non control transfer branches)." },
	{ "k7-interrupts-masked-cycles", "Count the number of cycles when the processor's IF flag was zero." },
	{ "k7-interrupts-masked-while-pending-cycles", "Count the number of cycles interrupts were masked while pending due to the processor's IF flag being zero." },
	{ "k7-hardware-interrupts", "Count the number of taken hardware interrupts." },
	/* Nearly special counters */
	{ "k7-dc-refills-from-l2,unitmask=+m", "Count data cache refills from L2 cache (in M state)." },
	{ "k7-dc-refills-from-l2,unitmask=+oes", "Count data cache refills from L2 cache (in OES state)." },
	{ "k7-dc-refills-from-system,unitmask=+m", "Count data cache refills from system memory (in M state)." },
	{ "k7-dc-refills-from-system,unitmask=+oes", "Count data cache refills from system memory (in OES state)." },
	{ NULL, NULL }
};

/* PAPI PRESETS */
hwi_search_t K7Processor_map[] = {
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L1_DCM, {DERIVED_ADD, {PNE_K7_DC_REFILLS_FROM_SYSTEM, PNE_K7_DC_REFILLS_FROM_L2}, {0,}}},
#endif
	{PAPI_L1_ICM, {0, {PNE_K7_IC_MISSES}, {0,}}},
	{PAPI_L2_DCM, {0, {PNE_K7_DC_REFILLS_FROM_SYSTEM}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 3
	{PAPI_L1_TCM, {DERIVED_ADD, {PNE_K7_DC_REFILLS_FROM_SYSTEM, PNE_K7_DC_REFILLS_FROM_L2, PNE_K7_IC_MISSES}, {0,}}},
#endif
	{PAPI_TLB_DM, {0, {PNE_K7_L1_AND_L2_DTLB_MISSES}, {0,}}},
	{PAPI_TLB_IM, {0, {PNE_K7_L1_AND_L2_ITLB_MISSES}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_TLB_TL, {DERIVED_ADD, {PNE_K7_L1_AND_L2_DTLB_MISSES, PNE_K7_L1_AND_L2_ITLB_MISSES}, {0,}}},
#endif
	{PAPI_L1_LDM, {0, {PNE_K7_DC_REFILLS_FROM_L2_OES}, {0,}}},
	{PAPI_L1_STM, {0, {PNE_K7_DC_REFILLS_FROM_L2_M}, {0,}}},
	{PAPI_L2_LDM, {0, {PNE_K7_DC_REFILLS_FROM_SYSTEM_OES}, {0,}}},
	{PAPI_L2_STM, {0, {PNE_K7_DC_REFILLS_FROM_SYSTEM_M}, {0,}}},
	{PAPI_HW_INT, {0, {PNE_K7_HARDWARE_INTERRUPTS}, {0,}}},
	{PAPI_BR_UCN, {0, {PNE_K7_RETIRED_FAR_CONTROL_TRANSFERS}, {0,}}},
	{PAPI_BR_CN, {0, {PNE_K7_RETIRED_BRANCHES}, {0,}}},
	{PAPI_BR_TKN, {0, {PNE_K7_RETIRED_TAKEN_BRANCHES}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_BR_NTK, {DERIVED_SUB, {PNE_K7_RETIRED_BRANCHES, PNE_K7_RETIRED_TAKEN_BRANCHES}, {0,}}},
#endif
	{PAPI_BR_MSP, {0, {PNE_K7_RETIRED_BRANCHES_MISPREDICTED}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_BR_PRC, {DERIVED_SUB, {PNE_K7_RETIRED_BRANCHES, PNE_K7_RETIRED_BRANCHES_MISPREDICTED}, {0,}}},
#endif
	{PAPI_TOT_INS, {0, {PNE_K7_RETIRED_INSTRUCTIONS}, {0,}}},
	{PAPI_BR_INS, {0, {PNE_K7_RETIRED_TAKEN_BRANCHES}, {0,}}},
	{PAPI_L1_DCA, {0, {PNE_K7_DC_ACCESSES}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_DCA, {DERIVED_ADD, {PNE_K7_DC_REFILLS_FROM_SYSTEM, PNE_K7_DC_REFILLS_FROM_L2}, {0,}}},
#endif
	{PAPI_L1_ICA, {0, {PNE_K7_IC_FETCHES}, {0,}}},
	{PAPI_L2_ICA, {0, {PNE_K7_IC_MISSES}, {0,}}},
	{PAPI_L1_ICR, {0, {PNE_K7_IC_FETCHES}, {0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L1_TCA, {DERIVED_ADD, {PNE_K7_DC_ACCESSES, PNE_K7_IC_FETCHES}, {0,}}},
#endif
	{0, {0, {PAPI_NULL}, {0,}}}
};
