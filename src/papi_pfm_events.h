#ifndef _PAPI_PFM_EVENTS_H
#define _PAPI_PFM_EVENTS_H
/* 
* File:    papi_pfm_events.h
* CVS:     $Id$
* Author:  Dan Terpstra; extracted from Philip Mucci's perfmon.h
*          mucci@cs.utk.edu
*
*/

#include "perfmon/pfmlib.h"
#include "perfmon/perfmon.h"

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

#define PERFMON_EVENT_FILE "perfmon_events.csv"

typedef struct {
   long long preset; /* Preset code */
   int derived; /* Derived code */
   char *(findme[MAX_COUNTER_TERMS]); /* Strings to look for, more than 1 means derived */
   char *operation; /* PostFix operations between terms */
   char *note; /* In case a note is included with a preset */
} pfm_preset_search_entry_t;

#endif // _PAPI_PFM_EVENTS_H
