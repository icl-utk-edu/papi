#ifndef _PAPI_PERF_EVENTS_H
#define _PAPI_PERF_EVENTS_H
/*
* File:    perf_events.h
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
*          - based on perfmon.h by -
*          Phil Mucci
*          mucci@cs.utk.edu
*
*/

#include "perfmon/pfmlib.h"
#include PEINCLUDE
#include "papi_defines.h"
#include "linux-lock.h"
#include "linux-context.h"


/* Take a guess at these values for now - FIXME */

#ifndef PFMLIB_MAX_PMCS                                                       
#define PFMLIB_MAX_PMCS 100                                                   
#define PFMLIB_MAX_PMDS 100                                                   
typedef int pfm_register_t;                                                   
#else                                                                         
typedef pfmlib_event_t pfm_register_t;                                        
#endif                                                                        
 
typedef int reg_alloc_t;            

typedef struct
{
	unsigned char wakeup_mode;
} per_event_info_t;

typedef struct
{
	int num_events;
	int num_groups;
	unsigned int domain;
	unsigned int multiplexed;
	unsigned int overflow;
	unsigned int inherit;
	int cpu;
	pid_t tid;
	struct perf_event_attr events[PAPI_MPX_DEF_DEG];
	per_event_info_t per_event_info[PAPI_MPX_DEF_DEG];
	/* Buffer to gather counters */
	long long counts[PFMLIB_MAX_PMDS];
} control_state_t;

typedef struct
{
	int group_leader;				   /* index of leader */
	int event_fd;
	int event_id;
	uint32_t nr_mmap_pages;			   /* number pages in the mmap buffer */
	void *mmap_buf;					   /* used to contain profiling data samples as well as control */
	uint64_t tail;					   /* current location in the mmap buffer to read from */
	uint64_t mask;					   /* mask used for wrapping the pages */
} evt_t;

typedef struct
{
	/* Array of event fd's, event group leader is event_fd[0] */
	int cookie;
	int state;
	int num_evts;
	evt_t evt[PAPI_MPX_DEF_DEG];
} context_t;

#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

#endif
