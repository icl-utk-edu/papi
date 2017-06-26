/* This test checks that mixing PAPI_flips and the other high
 * level calls does the right thing.
 * by Kevin London
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval;
	int Events, fip = 0;
	long long values, flpins;
	float real_time, proc_time, mflops;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Initialize PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* First see if we have PAPI_FP_INS event */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		fip = 1;
		Events = PAPI_FP_INS;
	/* If not, look for PAPI_FP_OPS */
	} else if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK ) {
		fip = 2;
		Events = PAPI_FP_OPS;
	} else {
		if ( !quiet ) {
			printf( "PAPI_FP_INS and PAPI_FP_OPS are not defined for this platform.\n" );
		}
		test_skip( __FILE__, __LINE__, "FLOPS event not supported", 1);
	}

	/* Start counting flips or flops event */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	} else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	/* If we are flipsing/flopsing, then start_counters should fail */
	retval = PAPI_start_counters( &Events, 1 );
	if (retval == PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}

	/* Try flipsing/flopsing again, should work */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	} else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	/* If we are flipsing/flopsing, then read should fail */
	if ( ( retval = PAPI_read_counters( &values, 1 ) ) == PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}

	/* Stop should still work then */
	if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}

	/* Restart flips/flops */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	} else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval != PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	/* Try reading again, should fail */
	if ( ( retval = PAPI_read_counters( &values, 1 ) ) == PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}

	/* Stop */
	if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}

	/* Now try starting, should work */
	if ( ( retval = PAPI_start_counters( &Events, 1 ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}

	/* Read should work too */
	if ( ( retval = PAPI_read_counters( &values, 1 ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}

	/* flipsing/flopsing should fail */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval == PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	} else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval == PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	/* Stop everything */
	if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}

	test_pass( __FILE__ );

	return 0;
}
