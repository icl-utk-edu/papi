#include "papi_test.h"

extern int TESTS_QUIET;				   /* Declared in test_utils.c */
extern void clockcore( void );		   /* Declared in clockcore.c */

int
main( int argc, char **argv )
{
	int retval;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_set_debug( PAPI_VERB_ECONT ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_debug", retval );

	if ( !TESTS_QUIET ) {
		printf( "Test case: Clock latency and resolution.\n" );
		printf( "-----------------------------------------------\n" );
	}

	clockcore(  );

	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
