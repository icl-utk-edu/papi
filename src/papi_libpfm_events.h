#ifndef _PAPI_LIBPFM_EVENTS_H
#define _PAPI_LIBPFM_EVENTS_H

/* 
* File:    papi_libpfm_events.h
*/

/* Prototypes for libpfm name library access */

extern int _papi_libpfm_error( int pfm_error );
extern int _papi_libpfm_setup_presets( char *name, int type, int cidx );
extern int _papi_libpfm_ntv_enum_events( unsigned int *EventCode, int modifier );
extern int _papi_libpfm_ntv_name_to_code( char *ntv_name,
				       unsigned int *EventCode );
extern int _papi_libpfm_ntv_code_to_name( unsigned int EventCode, char *name,
				       int len );
extern int _papi_libpfm_ntv_code_to_descr( unsigned int EventCode, char *name,
					int len );
extern int _papi_libpfm_ntv_code_to_bits( unsigned int EventCode,
				       hwd_register_t * bits );
extern int _papi_libpfm_shutdown(void);
extern int _papi_libpfm_init(papi_vector_t *my_vector, int cidx);

/* Gross perfctr/perf_events compatability hack */
/* need to think up a better way to handle this */

#ifndef __PERFMON_PERF_EVENT_H__
struct perf_event_attr {
  int config;
  int type;
};

#define PERF_TYPE_RAW 4;

#endif /* !__PERFMON_PERF_EVENT_H__ */


extern int _papi_libpfm_setup_counters( struct perf_event_attr *attr, 
				      hwd_register_t *ni_bits );

#endif // _PAPI_LIBPFM_EVENTS_H
