#ifndef MULTIPLEX_H
#define MULTIPLEX_H

int mpx_check( int EventSet );
int mpx_init( int );
int mpx_add_event( MPX_EventSet **, int EventCode, int domain,
		   int granularity );
int mpx_remove_event( MPX_EventSet **, int EventCode );
int MPX_add_events( MPX_EventSet ** mpx_events, int *event_list, int num_events,
		    int domain, int granularity );
int MPX_stop( MPX_EventSet * mpx_events, long long *values );
int MPX_cleanup( MPX_EventSet ** mpx_events );
void MPX_shutdown( void );
int MPX_reset( MPX_EventSet * mpx_events );
int MPX_read( MPX_EventSet * mpx_events, long long *values, int called_by_stop );
int MPX_start( MPX_EventSet * mpx_events );

#endif /* MULTIPLEX_H */
