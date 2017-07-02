/*
 * Test example for branch accuracy and functionality, originally
 * provided by Timothy Kaiser, SDSC. It was modified to fit the
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 * and Phil Mucci <mucci@cs.utk.edu>
 * This example verifies the accuracy of branch events
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define MAXEVENTS 4
#define SLEEPTIME 100
#define MINCOUNTS 100000
#define MPX_TOLERANCE .20

int
main( int argc, char **argv )
{
	PAPI_event_info_t info;
	int i, j, retval;
	int iters = 10000000;
	double x = 1.1, y;
	long long t1, t2;
	long long values[MAXEVENTS], refvalues[MAXEVENTS];
	int sleep_time = SLEEPTIME;
	double spread[MAXEVENTS];
	int nevents = MAXEVENTS;
	int eventset = PAPI_NULL;
	int events[MAXEVENTS];
	int quiet;

	/* Set quiet variable */
	quiet = tests_quiet( argc, argv );

	/* Parse command line args */
	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) ) {

		}
		else {
			sleep_time = atoi( argv[1] );
			if ( sleep_time <= 0 )
				sleep_time = SLEEPTIME;
		}
	}

	events[0] = PAPI_BR_NTK;
	events[1] = PAPI_BR_PRC;
	events[2] = PAPI_BR_INS;
	events[3] = PAPI_BR_MSP;

	/* Why were these disabled?
	events[3]=PAPI_BR_CN;
	events[4]=PAPI_BR_UCN;
	events[5]=PAPI_BR_TKN; */


	for ( i = 0; i < MAXEVENTS; i++ ) {
		values[i] = 0;
	}

	if ( !quiet ) {
		printf( "\nAccuracy check of branch presets.\n" );
		printf( "Comparing a measurement with separate measurements.\n\n" );
	}

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

#ifdef MPX
	if ( ( retval = PAPI_multiplex_init(  ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_multiplex_init", retval );

	if ( ( retval = PAPI_set_multiplex( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
#endif

	nevents = 0;

	for ( i = 0; i < MAXEVENTS; i++ ) {
		if ( PAPI_query_event( events[i] ) != PAPI_OK )
			continue;
		if ( PAPI_add_event( eventset, events[i] ) == PAPI_OK ) {
			events[nevents] = events[i];
			nevents++;
		}
	}

	if ( nevents < 1 )
		test_skip( __FILE__, __LINE__, "Not enough events left...", 0 );

	/* Find a reasonable number of iterations (each
	 * event active 20 times) during the measurement
	 */
	t2 = (long long)(10000 * 20) * nevents;	/* Target: 10000 usec/multiplex, 20 repeats */
	if ( t2 > 30e6 )
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );

	/* Measure one run */
	t1 = PAPI_get_real_usec(  );
	y = do_flops3( x, iters, 1 );
	t1 = PAPI_get_real_usec(  ) - t1;

	if ( t2 > t1 )			 /* Scale up execution time to match t2 */
		iters = iters * ( int ) ( t2 / t1 );
	else if ( t1 > 30e6 )	 /* Make sure execution time is < 30s per repeated test */
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );

	x = 1.0;

	if ( !quiet )
		printf( "\nFirst run: Together.\n" );

	t1 = PAPI_get_real_usec(  );
	if ( ( retval = PAPI_start( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	y = do_flops3( x, iters, 1 );
	if ( ( retval = PAPI_stop( eventset, values ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	t2 = PAPI_get_real_usec(  );

	if ( !quiet ) {
		printf( "\tOperations= %.1f Mflop", y * 1e-6 );
		printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
		printf( "PAPI grouped measurement:\n" );
	}
	for ( j = 0; j < nevents; j++ ) {
		PAPI_get_event_info( events[j], &info );
		if ( !quiet ) {
			printf( "%20s = ", info.short_descr );
			printf( LLDFMT, values[j] );
			printf( "\n" );
		}
	}
	if ( !quiet )
		printf( "\n" );


	if ( ( retval = PAPI_remove_events( eventset, events, nevents ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_remove_events", retval );
	if ( ( retval = PAPI_destroy_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );
	eventset = PAPI_NULL;
	if ( ( retval = PAPI_create_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	for ( i = 0; i < nevents; i++ ) {

		if ( ( retval = PAPI_cleanup_eventset( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );
		if ( ( retval = PAPI_add_event( eventset, events[i] ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

		x = 1.0;

		if ( !quiet )
			printf( "\nReference measurement %d (of %d):\n", i + 1, nevents );

		t1 = PAPI_get_real_usec(  );
		if ( ( retval = PAPI_start( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
		y = do_flops3( x, iters, 1 );
		if ( ( retval = PAPI_stop( eventset, &refvalues[i] ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		t2 = PAPI_get_real_usec(  );

		if ( !quiet ) {
			printf( "\tOperations= %.1f Mflop", y * 1e-6 );
			printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
		}
		PAPI_get_event_info( events[i], &info );
		if ( !quiet ) {
			printf( "PAPI results:\n%20s = ", info.short_descr );
			printf( LLDFMT, refvalues[i] );
			printf( "\n" );
		}
	}
	if ( !quiet )
		printf( "\n" );


	if ( !quiet ) {
		printf( "\n\nRelative accuracy:\n" );
		for ( j = 0; j < nevents; j++ )
			printf( "   Event %.2d", j );
		printf( "\n" );
	}

	for ( j = 0; j < nevents; j++ ) {
		spread[j] = abs( ( int ) ( refvalues[j] - values[j] ) );
		if ( values[j] )
			spread[j] /= ( double ) values[j];
		if ( !quiet )
			printf( "%10.3g ", spread[j] );
		/* Make sure that NaN get counted as errors */
		if ( spread[j] < MPX_TOLERANCE )
			i--;
		else if ( refvalues[j] < MINCOUNTS )	/* Neglect inprecise results with low counts */
			i--;
	}
	if ( !quiet ) {
		printf( "\n\n" );
	}

	if ( i ) {
		test_fail( __FILE__, __LINE__, "Values outside threshold", i );
	}

	test_pass( __FILE__ );

	return 0;
}


