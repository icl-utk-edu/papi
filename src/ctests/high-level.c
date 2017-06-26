/*  These examples show the essentials in using the PAPI high-level
    interface. The program consists of 4 work-loops. The programmer
    intends to count the total events for loop 1, 2 and 4, but not
    include the number of events in loop 3.

    To accomplish this PAPI_read_counters is used as a counter
    reset function, while PAPI_accum_counters is used to sum
    the contributions of loops 2 and 4 into the total count.
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define NUM_EVENTS 2

int
main( int argc, char **argv )
{
	int retval;
	long long values[NUM_EVENTS], dummyvalues[NUM_EVENTS];
	long long myvalues[NUM_EVENTS];
	int Events[NUM_EVENTS];
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* query and set up the right events to monitor */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		Events[0] = PAPI_FP_INS;
	} else {
		Events[0] = PAPI_TOT_INS;
	}
	Events[1] = PAPI_TOT_CYC;

	retval = PAPI_start_counters( ( int * ) Events, NUM_EVENTS );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Cannot start events\n");
		test_skip( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}

	/* Loop 1 */
	do_flops( NUM_FLOPS );

	retval = PAPI_read_counters( values, NUM_EVENTS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters continuing...)\n" );

	myvalues[0] = values[0];
	myvalues[1] = values[1];
	/* Loop 2 */
	do_flops( NUM_FLOPS );

	retval = PAPI_accum_counters( values, NUM_EVENTS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_accum_counters", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "(Counters being ''held'')\n" );

	/* Loop 3 */
	/* Simulated code that should not be counted */
	do_flops( NUM_FLOPS );

	retval = PAPI_read_counters( dummyvalues, NUM_EVENTS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	if ( !quiet )
		printf( TWO12, dummyvalues[0], dummyvalues[1], "(Skipped counts)\n" );

	if ( !quiet )
		printf( "%12s %12s  (''Continuing'' counting)\n", "xxx", "xxx" );
	/* Loop 4 */
	do_flops( NUM_FLOPS );

	retval = PAPI_accum_counters( values, NUM_EVENTS );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_accum_counters", retval );

	if ( !quiet )
		printf( TWO12, values[0], values[1], "" );

	if ( !quiet ) {
		printf( "----------------------------------\n" );
		printf( "Verification: The last line in each experiment should be\n" );
		printf( "approximately three times the value of the first line.\n" );
	}

	{
		long long min, max;
		min = ( long long ) ( ( double ) myvalues[0] * .9 );
		max = ( long long ) ( ( double ) myvalues[0] * 1.1 );
		if ( values[0] < ( 3 * min ) || values[0] > ( 3 * max ) ) {
			retval = 1;
			if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
				test_fail( __FILE__, __LINE__, "PAPI_FP_INS", 1 );
			} else {
				test_fail( __FILE__, __LINE__, "PAPI_TOT_INS", 1 );
			}
		}
		min = ( long long ) ( ( double ) myvalues[1] * .9 );
		max = ( long long ) ( ( double ) myvalues[1] * 1.1 );
		if ( values[1] < ( 3 * min ) || values[1] > ( 3 * max ) ) {
			retval = 1;
			test_fail( __FILE__, __LINE__, "PAPI_TOT_CYC", 1 );
		}
	}
	/* The values array is not allocated through allocate_test_space
	 * so we need to pass NULL here */
	test_pass( __FILE__ );

	return 0;

}
