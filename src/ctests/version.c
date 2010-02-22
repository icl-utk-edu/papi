/* This file performs the following test: compare and report versions from papi.h and the papi library */

#include "papi_test.h"
extern int TESTS_QUIET;				   /* Declared in test_utils.c */


int
main( int argc, char **argv )
{
	int retval, init_version, lib_version;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	init_version = PAPI_library_init( PAPI_VER_CURRENT );
	if ( init_version != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", init_version );

	if ( ( lib_version =
		   PAPI_get_opt( PAPI_LIB_VERSION, NULL ) ) == PAPI_EINVAL )
		test_fail( __FILE__, __LINE__, "PAPI_get_opt", PAPI_EINVAL );

	if ( !TESTS_QUIET ) {
		printf
			( "Version.c: Compare and report versions from papi.h and the papi library.\n" );
		printf
			( "-------------------------------------------------------------------------\n" );
		printf( "                    MAJOR  MINOR  REVISION\n" );
		printf
			( "-------------------------------------------------------------------------\n" );

		printf( "PAPI_VER_CURRENT : %4d %6d %7d\n",
				PAPI_VERSION_MAJOR( PAPI_VER_CURRENT ),
				PAPI_VERSION_MINOR( PAPI_VER_CURRENT ),
				PAPI_VERSION_REVISION( PAPI_VER_CURRENT ) );
		printf( "PAPI_library_init: %4d %6d %7d\n",
				PAPI_VERSION_MAJOR( init_version ),
				PAPI_VERSION_MINOR( init_version ),
				PAPI_VERSION_REVISION( init_version ) );
		printf( "PAPI_VERSION     : %4d %6d %7d\n",
				PAPI_VERSION_MAJOR( PAPI_VERSION ),
				PAPI_VERSION_MINOR( PAPI_VERSION ),
				PAPI_VERSION_REVISION( PAPI_VERSION ) );
		printf( "PAPI_get_opt     : %4d %6d %7d\n",
				PAPI_VERSION_MAJOR( lib_version ),
				PAPI_VERSION_MINOR( lib_version ),
				PAPI_VERSION_REVISION( lib_version ) );

		printf
			( "-------------------------------------------------------------------------\n" );
	}

	if ( lib_version != PAPI_VERSION )
		test_fail( __FILE__, __LINE__, "Version Mismatch", PAPI_EINVAL );
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
