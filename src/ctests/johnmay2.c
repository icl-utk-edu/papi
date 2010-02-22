#include "papi_test.h"
extern int TESTS_QUIET;				   /* Declared in test_utils.c */

int
main( int argc, char **argv )
{
	int FPEventSet = PAPI_NULL;
	long long values;
	int PAPI_event, retval;
	char event_name[PAPI_MAX_STR_LEN];

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	/* query and set up the right instruction to monitor */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK )
		PAPI_event = PAPI_FP_INS;
	else
		PAPI_event = PAPI_TOT_INS;

	if ( ( retval = PAPI_query_event( PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_query_event", retval );

	if ( ( retval = PAPI_create_eventset( &FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( ( retval = PAPI_add_event( FPEventSet, PAPI_event ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	if ( ( retval = PAPI_start( FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	if ( ( retval = PAPI_cleanup_eventset( FPEventSet ) ) != PAPI_EISRUN )	/* JT */
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_EISRUN )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	do_flops( 1000000 );

	if ( ( retval = PAPI_stop( FPEventSet, &values ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_EINVAL )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if ( ( retval = PAPI_cleanup_eventset( FPEventSet ) ) != PAPI_OK )	/* JT */
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	if ( ( retval = PAPI_destroy_eventset( &FPEventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if ( FPEventSet != PAPI_NULL )
		test_fail( __FILE__, __LINE__, "FPEventSet != PAPI_NULL", retval );

	if ( !TESTS_QUIET ) {
		if ( ( retval =
			   PAPI_event_code_to_name( PAPI_event, event_name ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );

		printf( "Test case John May 2: cleanup / destroy eventset.\n" );
		printf( "-------------------------------------------------\n" );
		printf( "Test run    : \t1\n" );
		printf( "%s : \t", event_name );
		printf( LLDFMT, values );
		printf( "\n" );
		printf( "-------------------------------------------------\n" );
		printf( "Verification:\n" );
		printf( "These error messages:\n" );
		printf
			( "PAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n" );
		printf
			( "PAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n" );
		printf( "PAPI Error Code -1: PAPI_EINVAL: Invalid argument\n" );
	}
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
