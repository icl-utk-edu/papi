/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-unknown.c
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include SUBSTRATE
#include "papiStdEventDefs.h"
#include "map.h"

/****************************************************************************
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
****************************************************************************/

/*
	NativeEvent_Value_UnknownProcessor must match UnkProcessor_info 
*/

Native_Event_LabelDescription_t UnkProcessor_info[] =
{
	{ "branches", "Measure the number of branches retired." },
	{ "branch-mispredicts", "Measure the number of retired branches that were mispredicted." },
	/* { "cycles", "Measure processor cycles." }, */
	{ "dc-misses", "Measure the number of data cache misses." },
	{ "ic-misses", "Measure the number of instruction cache misses." },
	{ "instructions", "Measure the number of instructions retired." },
	{ "interrupts", "Measure the number of interrupts seen." },
	{ "unhalted-cycles", "Measure the number of cycles the processor is not in a halted or sleep state." },
	{ NULL, NULL }
};

/* PAPI PRESETS */
hwi_search_t UnkProcessor_map[] = {
	/* {PAPI_TOT_CYC, {0, {PNE_UNK_CYCLES, PAPI_NULL},{0,}}}, */
	{PAPI_TOT_INS, {0, {PNE_UNK_INSTRUCTIONS},{0,}}},
	{PAPI_BR_INS, {0, {PNE_UNK_BRANCHES} ,{0,}}},
	{PAPI_BR_INS, {0, {PNE_UNK_INTERRUPTS} ,{0,}}},
	{PAPI_BR_MSP, {0, {PNE_UNK_BRANCH_MISPREDICTS} ,{0,}}},
	{PAPI_L2_DCM, {0, {PNE_UNK_DC_MISSES} ,{0,}}},
	{PAPI_L2_ICM, {0, {PNE_UNK_IC_MISSES} ,{0,}}},
#if HWPMC_NUM_COUNTERS >= 2
	{PAPI_L2_TCM, {DERIVED_ADD, {PNE_UNK_IC_MISSES, PNE_UNK_DC_MISSES} ,{0,}}},
#endif
	{0, {0, {PAPI_NULL}, {0,}}}
};
