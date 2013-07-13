#ifndef _PAPI_LIBPFM4_EVENTS_H
#define _PAPI_LIBPFM4_EVENTS_H

/*
* File:    papi_libpfm4_events.h
*/

#include "perfmon/pfmlib.h"
#include PEINCLUDE

struct native_event_t {
  int component;
  char *pmu;
  int libpfm4_idx;
  char *allocated_name;
  char *base_name;
  char *pmu_plus_name;
  int users;
  long long config;
  long long config1;
  long long config2;
  int type;
};

#define PMU_TYPE_CORE   1
#define PMU_TYPE_UNCORE 2
#define PMU_TYPE_OS     4

struct native_event_table_t {
   struct native_event_t *native_events;
   int num_native_events;
   int allocated_native_events;
   pfm_pmu_info_t default_pmu;
   int pmu_type;
};


/* Prototypes for libpfm name library access */

int _papi_libpfm4_error( int pfm_error );
int _papi_libpfm4_shutdown(void);
int _papi_libpfm4_init(papi_vector_t *my_vector);

#endif // _PAPI_LIBPFM4_EVENTS_H
