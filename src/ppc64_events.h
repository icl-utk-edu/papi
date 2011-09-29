#ifndef PPC64_EVENTS_H_
#define PPC64_EVENTS_H_
/* 
* File:    ppc64_events.h
* CVS:     
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*          
*/

#ifdef ARCH_EVTS
#include ARCH_EVTS
#else
#define GROUP_INTS 8
#if defined(_POWER5)
#define PAPI_MAX_NATIVE_EVENTS 512
#elif defined(_POWER6)
#define PAPI_MAX_NATIVE_EVENTS 1024
#else
#define PAPI_MAX_NATIVE_EVENTS 1024
#endif
#define MAX_GROUPS (GROUP_INTS * 32)
#endif

typedef struct PPC64_register
{
	/* indicate which counters this event can live on */
	unsigned int selector;
	/* Buffers containing counter cmds for each possible metric */
	int counter_cmd[MAX_COUNTERS];
	/* which group this event belongs */
	unsigned int group[GROUP_INTS];
} PPC64_register_t;

/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef hwd_register_t
typedef PPC64_register_t hwd_register_t;

typedef struct PPC64_groups
{
#ifdef __perfctr__
	unsigned int mmcr0;
	unsigned int mmcr1L;
	unsigned int mmcr1U;
	unsigned int mmcra;
	unsigned int counter_cmd[MAX_COUNTERS];
#else
/* Buffer containing counter cmds for this group */
	unsigned int counter_cmd[MAX_COUNTERS];
#endif
} PPC64_groups_t;

typedef PPC64_groups_t hwd_groups_t;

typedef struct native_event_entry
{
	/* description of the resources required by this native event */
	hwd_register_t resources;
	/* If it exists, then this is the name of this event */
	char *name;
	/* If it exists, then this is the description of this event */
	char *description;
} native_event_entry_t;

typedef struct PPC64_native_map
{
	/* native event name */
	char *name;
	/* real index in the native table */
	int index;
} PPC64_native_map_t;

extern native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];
#ifndef __perfctr__
extern hwd_pminfo_t pminfo;
extern pm_groups_info_t pmgroups;
#endif
extern PPC64_native_map_t native_name_map[PAPI_MAX_NATIVE_EVENTS];
extern hwd_groups_t group_map[MAX_GROUPS];

int check_native_name(  );

#endif /*PPC64_EVENTS_H_ */
