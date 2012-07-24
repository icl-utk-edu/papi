/* 
* File:    multiplex.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file tests the multiplex functionality, originally developed by 
   John May of LLNL. */

#include "papi_test.h"

void
init_papi( void )
{
	int retval;

	/* Initialize the library */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

}

/* Tests that we can really multiplex a lot. */

int
case1( void )
{
	int retval, i, EventSet = PAPI_NULL, j = 0, k = 0, allvalid = 1;
	int max_mux, nev, *events;
	long long *values;
	PAPI_event_info_t pset;
	char evname[PAPI_MAX_STR_LEN];

	init_papi(  );
	init_multiplex(  );

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif
	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif
	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );
#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif

	retval = PAPI_set_multiplex( EventSet );
        if ( retval == PAPI_ENOSUPP) {
	   test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
	}
        else if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );

	max_mux = PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL );
	if ( max_mux > 32 )
		max_mux = 32;

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif

	/* Fill up the event set with as many non-derived events as we can */
	printf
		( "\nFilling the event set with as many non-derived events as we can...\n" );

	i = PAPI_PRESET_MASK;
	do {
		if ( PAPI_get_event_info( i, &pset ) == PAPI_OK ) {
			if ( pset.count && ( strcmp( pset.derived, "NOT_DERIVED" ) == 0 ) ) {
				retval = PAPI_add_event( EventSet, ( int ) pset.event_code );
				if ( retval != PAPI_OK ) {
				   printf("Failed trying to add %s\n",pset.symbol);
				   break;
				}
				else {
					printf( "Added %s\n", pset.symbol );
					j++;
				}
			}
		}
	} while ( ( PAPI_enum_event( &i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK ) &&
			  ( j < max_mux ) );

	events = ( int * ) malloc( ( size_t ) j * sizeof ( int ) );
	if ( events == NULL )
		test_fail( __FILE__, __LINE__, "malloc events", 0 );

	values = ( long long * ) malloc( ( size_t ) j * sizeof ( long long ) );
	if ( values == NULL )
		test_fail( __FILE__, __LINE__, "malloc values", 0 );

	do_stuff(  );

#if 0
	if ( PAPI_set_domain( PAPI_DOM_KERNEL ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_domain", retval );
#endif
	
	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	nev = j;
	retval = PAPI_list_events( EventSet, events, &nev );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_list_events", retval );

	printf( "\nEvent Counts:\n" );
	for ( i = 0, allvalid = 0; i < j; i++ ) {
		PAPI_event_code_to_name( events[i], evname );
		printf( TAB1, evname, values[i] );
		if ( values[i] == 0 )
			allvalid++;
	}
	printf( "\n" );
	if ( allvalid ) {
		printf( "Caution: %d counters had zero values\n", allvalid );
	}
   
        if (allvalid==j) {
	   test_fail( __FILE__, __LINE__, "All counters returned zero", 5 );
	}

	for ( i = 0, allvalid = 0; i < j; i++ ) {
		for ( k = i + 1; k < j; k++ ) {
			if ( ( i != k ) && ( values[i] == values[k] ) ) {
				allvalid++;
				break;
			}
		}
	}

	if ( allvalid ) {
		printf( "Caution: %d counter pair(s) had identical values\n",
				allvalid );
	}

	free( events );
	free( values );

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	return ( SUCCESS );
}

int
main( int argc, char **argv )
{

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	printf( "%s: Does PAPI_multiplex_init() handle lots of events?\n",
			argv[0] );
	printf( "Using %d iterations\n", NUM_ITERS );

	case1(  );
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
