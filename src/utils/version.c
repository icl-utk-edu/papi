/* This utility displays the current PAPI version number */

#include <stdlib.h>
#include <stdio.h>
#include "papi.h"

int
main(  )
{
	printf( "PAPI Version: %d.%d.%d.%d\n", PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ),
			PAPI_VERSION_INCREMENT( PAPI_VERSION ) );
	exit( 0 );
}
