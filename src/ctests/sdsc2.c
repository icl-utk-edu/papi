/*
 * $Id$
 *
 * Test example for multiplex functionality, originally 
 * provided by Timothy Kaiser, SDSC. It was modified to fit the 
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 *
 * This example verifies the PAPI_reset function for 
 * multiplexed events
 */

#include "papi_test.h"
#include <stdio.h>
#include <math.h>

#define REPEATS 5
#define MAXEVENTS 9
#define SLEEPTIME 100
#define MINCOUNTS 100000

static double dummy3( double x, int iters );

int
main( int argc, char **argv )
{
	PAPI_event_info_t info;
	int i, j, retval;
	int iters = NUM_FLOPS;
	double x = 1.1, y, dtmp;
	long long t1, t2;
	long long values[MAXEVENTS];
	int sleep_time = SLEEPTIME;
#ifdef STARTSTOP
	long long dummies[MAXEVENTS];
#endif
	double valsample[MAXEVENTS][REPEATS];
	double valsum[MAXEVENTS];
	double avg[MAXEVENTS];
	double spread[MAXEVENTS];
	int nevents = MAXEVENTS;
	int eventset = PAPI_NULL;
	int events[MAXEVENTS];
	int fails;

	events[0] = PAPI_FP_INS;
	events[1] = PAPI_TOT_INS;
	events[2] = PAPI_INT_INS;
	events[3] = PAPI_TOT_CYC;
	events[4] = PAPI_STL_CCY;
	events[5] = PAPI_BR_INS;
	events[6] = PAPI_SR_INS;
	events[7] = PAPI_LD_INS;
	events[8] = PAPI_TOT_IIS;

	for ( i = 0; i < MAXEVENTS; i++ ) {
		values[i] = 0;
		valsum[i] = 0;
	}

	if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) )
			tests_quiet( argc, argv );
		else {
			sleep_time = atoi( argv[1] );
			if ( sleep_time <= 0 )
				sleep_time = SLEEPTIME;
		}
	}

	if ( !TESTS_QUIET ) {
		printf( "\nAccuracy check of multiplexing routines.\n" );
		printf( "Investigating the variance of multiplexed measurements.\n\n" );
	}

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

#ifdef MPX
	init_multiplex(  );
#endif

	if ( ( retval = PAPI_create_eventset( &eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
#ifdef MPX

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( eventset, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

	if ( ( retval = PAPI_set_multiplex( eventset ) ) ) {
	        if ( retval == PAPI_ENOSUPP) {
		       test_skip(__FILE__, __LINE__, "Multiplex not supported", 1);
		}
	   
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex", retval );
	}
#endif

	/* What does this code even do? */
	nevents = MAXEVENTS;
	for ( i = 0; i < nevents; i++ ) {
		if ( ( retval = PAPI_add_event( eventset, events[i] ) ) ) {
		   for ( j = i; j < MAXEVENTS-1; j++ ) {
				events[j] = events[j + 1];
		   }
		   nevents--;
		   i--;
		}
	}
	if ( nevents < 2 )
		test_skip( __FILE__, __LINE__, "Not enough events left...", 0 );

	/* Find a reasonable number of iterations (each 
	 * event active 20 times) during the measurement
	 */
	t2 = 10000 * 20 * nevents;	/* Target: 10000 usec/multiplex, 20 repeats */
	if ( t2 > 30e6 )
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );

	/* Measure one run */
	t1 = PAPI_get_real_usec(  );
	y = dummy3( x, iters );
	t1 = PAPI_get_real_usec(  ) - t1;

	if ( t2 > t1 )			 /* Scale up execution time to match t2 */
		iters = iters * ( int ) ( t2 / t1 );
	else if ( t1 > 30e6 )	 /* Make sure execution time is < 30s per repeated test */
		test_skip( __FILE__, __LINE__, "This test takes too much time",
				   retval );

	if ( ( retval = PAPI_start( eventset ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	for ( i = 1; i <= REPEATS; i++ ) {
		x = 1.0;

#ifndef STARTSTOP
		if ( ( retval = PAPI_reset( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_reset", retval );
#else
		if ( ( retval = PAPI_stop( eventset, dummies ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		if ( ( retval = PAPI_start( eventset ) ) )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
#endif

		if ( !TESTS_QUIET )
			printf( "\nTest %d (of %d):\n", i, REPEATS );
		t1 = PAPI_get_real_usec(  );
		y = dummy3( x, iters );
		PAPI_read( eventset, values );
		t2 = PAPI_get_real_usec(  );

		if ( !TESTS_QUIET ) {
			printf( "\n(calculated independent of PAPI)\n" );
			printf( "\tOperations= %.1f Mflop", y * 1e-6 );
			printf( "\t(%g Mflop/s)\n\n", ( y / ( double ) ( t2 - t1 ) ) );
			printf( "PAPI measurements:\n" );
		}
		for ( j = 0; j < nevents; j++ ) {
			PAPI_get_event_info( events[j], &info );
			if ( !TESTS_QUIET ) {
				printf( "%20s = ", info.short_descr );
				printf( LLDFMT, values[j] );
				printf( "\n" );
			}
			dtmp = ( double ) values[j];
			valsum[j] += dtmp;
			valsample[j][i - 1] = dtmp;
		}
		if ( !TESTS_QUIET )
			printf( "\n" );
	}

	if ( ( retval = PAPI_stop( eventset, values ) ) )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( !TESTS_QUIET ) {
		printf( "\n\nEstimated variance relative to average counts:\n" );
		for ( j = 0; j < nevents; j++ )
			printf( "   Event %.2d", j );
		printf( "\n" );
	}

	fails = nevents;
	/* Due to limited precision of floating point cannot really use
	   typical standard deviation compuation for large numbers with
	   very small variations. Instead compute the std devation
	   problems with precision.
	 */
	for ( j = 0; j < nevents; j++ ) {
		avg[j] = valsum[j] / REPEATS;
		spread[j] = 0;
		for ( i = 0; i < REPEATS; ++i ) {
			double diff = ( valsample[j][i] - avg[j] );
			spread[j] += diff * diff;
		}
		spread[j] = sqrt( spread[j] / REPEATS ) / avg[j];
		if ( !TESTS_QUIET )
			printf( "%9.2g  ", spread[j] );
		/* Make sure that NaN get counted as errors */
		if ( spread[j] < MPX_TOLERANCE )
			--fails;
		else if ( valsum[j] < MINCOUNTS )	/* Neglect inprecise results with low counts */
			--fails;
	}

	if ( !TESTS_QUIET ) {
		printf( "\n\n" );
		for ( j = 0; j < nevents; j++ ) {
			PAPI_get_event_info( events[j], &info );
			printf( "Event %.2d: mean=%10.0f, sdev/mean=%7.2g nrpt=%2d -- %s\n",
					j, avg[j], spread[j], REPEATS, info.short_descr );
		}
		printf( "\n\n" );
	}

	if ( fails )
		test_fail( __FILE__, __LINE__, "Values outside threshold", fails );
	else
		test_pass( __FILE__, NULL, 0 );

	return 0;
}

static double
dummy3( double x, int iters )
{
	int i;
	double w, y, z, a, b, c, d, e, f, g, h;
	double one;
	one = 1.0;
	w = x;
	y = x;
	z = x;
	a = x;
	b = x;
	c = x;
	d = x;
	e = x;
	f = x;
	g = x;
	h = x;
	for ( i = 1; i <= iters; i++ ) {
		w = w * 1.000000000001 + one;
		y = y * 1.000000000002 + one;
		z = z * 1.000000000003 + one;
		a = a * 1.000000000004 + one;
		b = b * 1.000000000005 + one;
		c = c * 0.999999999999 + one;
		d = d * 0.999999999998 + one;
		e = e * 0.999999999997 + one;
		f = f * 0.999999999996 + one;
		g = h * 0.999999999995 + one;
		h = h * 1.000000000006 + one;
	}
	return 2.0 * ( a + b + c + d + e + f + w + x + y + z + g + h );
}
