#ifndef EXTRAS_H
#define EXTRAS_H

int _papi_hwi_query_native_event( unsigned int EventCode );
int _papi_hwi_get_native_event_info( unsigned int EventCode,
				     PAPI_event_info_t * info );
int _papi_hwi_native_name_to_code( char *in, int *out );
int _papi_hwi_native_code_to_name( unsigned int EventCode, char *hwi_name,
				   int len );
int _papi_hwi_native_code_to_descr( unsigned int EventCode, char *hwi_descr,
				    int len );


int _papi_hwi_stop_timer( int timer, int signal );
int _papi_hwi_start_timer( int timer, int signal, int ms );
int _papi_hwi_stop_signal( int signal );
int _papi_hwi_start_signal( int signal, int need_context, int cidx );
int _papi_hwi_initialize( DynamicArray_t ** );
int _papi_hwi_dispatch_overflow_signal( void *papiContext, caddr_t address,
					int *, long long, int,
					ThreadInfo_t ** master, int cidx );
void _papi_hwi_dispatch_profile( EventSetInfo_t * ESI, caddr_t address,
				 long long over, int profile_index );


#endif /* EXTRAS_H */
