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
int _papi_libpfm4_setup_presets( char *name, int type, int cidx );
int _papi_libpfm4_ntv_enum_events( unsigned int *EventCode, int modifier,
		       struct native_event_table_t *event_table);
int _papi_libpfm4_ntv_name_to_code( char *ntv_name,
				    unsigned int *EventCode,
		       struct native_event_table_t *event_table);
int _papi_libpfm4_ntv_code_to_name( unsigned int EventCode, char *name,
				    int len,
		       struct native_event_table_t *event_table);
int _papi_libpfm4_ntv_code_to_descr( unsigned int EventCode, char *name,
				     int len,
		       struct native_event_table_t *event_table);
int _papi_libpfm4_shutdown(struct native_event_table_t *event_table);
int _papi_libpfm4_init(papi_vector_t *my_vector);
int _pe_libpfm4_init(papi_vector_t *my_vector, int cidx,
		       struct native_event_table_t *event_table,
		       int pmu_type);


int _papi_libpfm4_ntv_code_to_info(unsigned int EventCode, 
				   PAPI_event_info_t *info,
		       struct native_event_table_t *event_table);

int _papi_libpfm4_setup_counters( struct perf_event_attr *attr, 
				  int event,
		       struct native_event_table_t *event_table);

#endif // _PAPI_LIBPFM4_EVENTS_H
