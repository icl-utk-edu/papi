/**
  * file version.c
  *	@brief papi_version utility.
  * @page papi_version
  *	@section Name
  *	papi_version - provides version information for papi.
  * 
  *	@section Synopsis
  *	papi_version
  *
  *	@section Description
  *	papi_version is a PAPI utility program that reports version
  *	information about the current PAPI installation.
  *
  *	@section Bugs
  *	There are no known bugs in this utility. 
  *	If you find a bug, it should be reported to the PAPI Mailing List at <ptools-perfapi@ptools.org>.
  */
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
