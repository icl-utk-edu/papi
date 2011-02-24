#ifndef _PAPI_PFM_EVENTS_H
#define _PAPI_PFM_EVENTS_H
/* 
* File:    papi_pfm_events.h
* CVS:     $Id$
* Author:  Dan Terpstra; extracted from Philip Mucci's perfmon.h
*          mucci@cs.utk.edu
*
*/

/* Prototypes for entry points found in papi_pfm_events */
extern int _papi_pfm_error( int pfm_error );
extern int _papi_pfm_setup_presets( char *name, int type );
extern int _papi_pfm_ntv_enum_events( unsigned int *EventCode, int modifier );
extern int _pfm_get_counter_info( unsigned int event, unsigned int *selector,
				  int *code );
extern int _papi_pfm_ntv_name_to_code( char *ntv_name,
				       unsigned int *EventCode );
extern int _papi_pfm_ntv_code_to_name( unsigned int EventCode, char *name,
				       int len );
extern int _papi_pfm_ntv_code_to_descr( unsigned int EventCode, char *name,
					int len );
extern int _papi_pfm_ntv_code_to_bits( unsigned int EventCode,
				       hwd_register_t * bits );
extern int _papi_pfm_ntv_bits_to_info( hwd_register_t * bits, char *names,
				       unsigned int *values, int name_len,
				       int count );
extern int _papi_pfm3_init(void);
extern int _papi_pfm3_vendor_fixups(void);
extern int _papi_pfm3_setup_counters( __u64 *pe_event, 
				      hwd_register_t *ni_bits );

#endif // _PAPI_PFM_EVENTS_H
