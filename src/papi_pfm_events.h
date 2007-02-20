#ifndef _PAPI_PFM_EVENTS_H
#define _PAPI_PFM_EVENTS_H
/* 
* File:    papi_pfm_events.h
* CVS:     $Id$
* Author:  Dan Terpstra; extracted from Philip Mucci's perfmon.h
*          mucci@cs.utk.edu
*
*/
/*
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <fcntl.h>
#include <ctype.h>
#include <inttypes.h>
#include <libgen.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/ucontext.h>
#include <sys/ptrace.h>
*/
#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"
//#include "perfmon/perfmon_dfl_smpl.h"

#ifdef __ia64__
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_montecito.h"
#endif

#if defined(DEBUG)
#define DEBUGCALL(a,b) { if (ISLEVEL(a)) { b; } }
#else
#define DEBUGCALL(a,b)
#endif

#define inline_static inline static

//typedef pfmlib_event_t hwd_register_t;
//typedef int hwd_register_map_t;
//typedef int hwd_reg_alloc_t;

//#define MAX_COUNTERS PFMLIB_MAX_PMCS
//#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS
#define PERFMON_EVENT_FILE "perfmon_events.csv"

typedef struct {
   /* Preset code */
   int preset;
   /* Derived code */
   int derived;
   /* Strings to look for, more than 1 means derived */
   char *(findme[MAX_COUNTER_TERMS]);
   /* PostFix operations between terms */
   char *operation;
   /* In case a note is included with a preset */
   char *note;
} pfm_preset_search_entry_t;

//typedef struct hwd_native_register {
//  pfmlib_regmask_t selector;
//  int pfmlib_event_index;
//} hwd_native_register_t;

//typedef struct hwd_native_event_entry {
//   /* If it exists, then this is the name of this event */
//   char name[PAPI_MAX_STR_LEN];
//   /* If it exists, then this is the description of this event */
//   char description[PAPI_HUGE_STR_LEN];
//  /* description of the resources required by this native event */
//  hwd_native_register_t resources;
//} hwd_native_event_entry_t;

#endif // _PAPI_PFM_EVENTS_H
