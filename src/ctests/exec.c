/* 
* File:    exec.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: start, stop and timer
functionality for a parent and a forked child. */

#include "papi_test.h"
#include <sys/wait.h>

int
main( int argc, char **argv )
{
	int retval;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( argc > 1 ) && ( strcmp( argv[1], "xxx" ) == 0 ) ) {
		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT )
			test_fail( __FILE__, __LINE__, "execed PAPI_library_init", retval );
	} else {
		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT )
			test_fail( __FILE__, __LINE__, "main PAPI_library_init", retval );

		PAPI_shutdown(  );

		if ( execlp( argv[0], argv[0], "xxx", NULL ) == -1 )
			test_fail( __FILE__, __LINE__, "execlp", PAPI_ESYS );
	}

	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
