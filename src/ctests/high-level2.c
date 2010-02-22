/* This test checks that mixing PAPI_flips and the other high
 * level calls does the right thing.
 * Kevin 
 */

#include "papi_test.h"
extern int TESTS_QUIET;				   /*Declared in test_utils.c */

int
main( int argc, char **argv )
{
	int retval;
	int Events, fip = 0;
	long long values, flpins;
	float real_time, proc_time, mflops;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		fip = 1;
		Events = PAPI_FP_INS;
	} else if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK ) {
		fip = 2;
		Events = PAPI_FP_OPS;
	} else {
		if ( !TESTS_QUIET )
			printf
				( "PAPI_FP_INS and PAPI_FP_OPS are not defined for this platform.\n" );
	}

	if ( fip > 0 ) {
		if ( fip == 1 ) {
			if ( ( retval =
				   PAPI_flips( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		} else {
			if ( ( retval =
				   PAPI_flops( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
		if ( ( retval = PAPI_start_counters( &Events, 1 ) ) == PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
		if ( fip == 1 ) {
			if ( ( retval =
				   PAPI_flips( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		} else {
			if ( ( retval =
				   PAPI_flops( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
		if ( ( retval = PAPI_read_counters( &values, 1 ) ) == PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
		if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
		if ( fip == 1 ) {
			if ( ( retval =
				   PAPI_flips( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		} else {
			if ( ( retval =
				   PAPI_flops( &real_time, &proc_time, &flpins,
							   &mflops ) ) != PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
		if ( ( retval = PAPI_read_counters( &values, 1 ) ) == PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
		if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
		if ( ( retval = PAPI_start_counters( &Events, 1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
		if ( ( retval = PAPI_read_counters( &values, 1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
		if ( fip == 1 ) {
			if ( ( retval =
				   PAPI_flips( &real_time, &proc_time, &flpins,
							   &mflops ) ) == PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		} else {
			if ( ( retval =
				   PAPI_flops( &real_time, &proc_time, &flpins,
							   &mflops ) ) == PAPI_OK )
				test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
		if ( ( retval = PAPI_stop_counters( &values, 1 ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}
	test_pass( __FILE__, NULL, 0 );
	exit( 0 );				 /* just to make the compiler happy... */
}
