/* 
* File:    multiplex1_pthreads.c
* Author:  Rick Kufrin
*          rkufrin@ncsa.uiuc.edu                    
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file really bangs on the multiplex pthread functionality */

#include <pthread.h>
#include "papi_test.h"

int *events;
int numevents = 0;
int max_events=0;

double
loop( long n )
{
	long i;
	double a = 0.0012;

	for ( i = 0; i < n; i++ ) {
		a += 0.01;
	}
	return a;
}

void *
thread( void *arg )
{
	( void ) arg;			 /*unused */
	int eventset = PAPI_NULL;
	long long *values;

	int ret = PAPI_register_thread(  );
	if ( ret != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_register_thread", ret );
	ret = PAPI_create_eventset( &eventset );
	if ( ret != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );

	values=calloc(max_events,sizeof(long long));

	printf( "Event set %d created\n", eventset );

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	ret = PAPI_assign_eventset_component( eventset, 0 );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component", ret );
	}

	ret = PAPI_set_multiplex( eventset );
        if ( ret == PAPI_ENOSUPP) {
	   test_skip( __FILE__, __LINE__, "Multiplexing not supported", 1 );
	}
	else if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", ret );
	}

	ret = PAPI_add_events( eventset, events, numevents );
	if ( ret < PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_add_events", ret );
	}

	ret = PAPI_start( eventset );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", ret );
	}

	do_stuff(  );

	ret = PAPI_stop( eventset, values );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
	}

	ret = PAPI_cleanup_eventset( eventset );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", ret );
	}

	ret = PAPI_destroy_eventset( &eventset );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", ret );
	}

	ret = PAPI_unregister_thread(  );
	if ( ret != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_unregister_thread", ret );
	return ( NULL );
}

int
main( int argc, char **argv )
{
	int nthreads = 8, ret, i;
	PAPI_event_info_t info;
	pthread_t *threads;
	const PAPI_hw_info_t *hw_info;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( !TESTS_QUIET ) {
		if ( argc > 1 ) {
			int tmp = atoi( argv[1] );
			if ( tmp >= 1 )
				nthreads = tmp;
		}
	}

	ret = PAPI_library_init( PAPI_VER_CURRENT );
	if ( ret != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
	}

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	if ( strcmp( hw_info->model_string, "POWER6" ) == 0 ) {
		ret = PAPI_set_domain( PAPI_DOM_ALL );
		if ( ret != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_set_domain", ret );
		}
	}

	ret = PAPI_thread_init( ( unsigned long ( * )( void ) ) pthread_self );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_thread_init", ret );
	}

	ret = PAPI_multiplex_init(  );
	if ( ret != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_multiplex_init", ret );
	}

	if ((max_events = PAPI_get_cmp_opt(PAPI_MAX_MPX_CTRS,NULL,0)) <= 0) {
		test_fail( __FILE__, __LINE__, "PAPI_get_cmp_opt", max_events );
	}

	if ((events = calloc(max_events,sizeof(int))) == NULL) {
		test_fail( __FILE__, __LINE__, "calloc", PAPI_ESYS );
	}

	/* Fill up the event set with as many non-derived events as we can */

	i = PAPI_PRESET_MASK;
	do {
		if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
			if ( info.count == 1 ) {
				events[numevents++] = ( int ) info.event_code;
				printf( "Added %s\n", info.symbol );
			} else {
				printf( "Skipping derived event %s\n", info.symbol );
			}
		}
	} while ( ( PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK )
			  && ( numevents < max_events ) );

	printf( "Found %d events\n", numevents );

	do_stuff(  );

	printf( "Creating %d threads:\n", nthreads );

	threads =
		( pthread_t * ) malloc( ( size_t ) nthreads * sizeof ( pthread_t ) );
	if ( threads == NULL ) {
		test_fail( __FILE__, __LINE__, "malloc", PAPI_ENOMEM );
	}

	/* Create the threads */
	for ( i = 0; i < nthreads; i++ ) {
		ret = pthread_create( &threads[i], NULL, thread, NULL );
		if ( ret != 0 ) {
			test_fail( __FILE__, __LINE__, "pthread_create", PAPI_ESYS );
		}
	}

	/* Wait for thread completion */
	for ( i = 0; i < nthreads; i++ ) {
		ret = pthread_join( threads[i], NULL );
		if ( ret != 0 ) {
			test_fail( __FILE__, __LINE__, "pthread_join", PAPI_ESYS );
		}
	}

	printf( "Done." );
	test_pass( __FILE__, NULL, 0 );
	pthread_exit( NULL );
	exit( 0 );
}
