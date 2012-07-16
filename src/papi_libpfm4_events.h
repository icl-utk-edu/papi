#ifndef _PAPI_LIBPFM4_EVENTS_H
#define _PAPI_LIBPFM4_EVENTS_H

/* 
* File:    papi_libpfm4_events.h
*/

#include "perfmon/pfmlib.h"
#include PEINCLUDE

/* Prototypes for libpfm name library access */

int _papi_libpfm4_error( int pfm_error );
int _papi_libpfm4_setup_presets( char *name, int type, int cidx );
int _papi_libpfm4_ntv_enum_events( unsigned int *EventCode, int modifier );
int _papi_libpfm4_ntv_name_to_code( char *ntv_name,
				       unsigned int *EventCode );
int _papi_libpfm4_ntv_code_to_name( unsigned int EventCode, char *name,
				       int len );
int _papi_libpfm4_ntv_code_to_descr( unsigned int EventCode, char *name,
					int len );
int _papi_libpfm4_shutdown(void);
int _papi_libpfm4_init(papi_vector_t *my_vector, int cidx);

int _papi_libpfm4_ntv_code_to_info(unsigned int EventCode, 
                                  PAPI_event_info_t *info);

int _papi_libpfm4_setup_counters( struct perf_event_attr *attr, 
				        int event );

#endif // _PAPI_LIBPFM4_EVENTS_H
