/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    HelloWorld.c
 * CVS:     $Id$
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * test case for Example component 
 * 
 *
 * @brief
 *  This file is a very simple HelloWorld C example which serves (together
 *	with its Makefile) as a guideline on how to add tests to components.
 *  The papi configure and papi Makefile will take care of the compilation
 *	of the component tests (if all tests are added to a directory named
 *	'tests' in the specific component dir).
 *	See components/README for more details.
 */

#include<stdio.h>
#include<stdlib.h>
#include "papi_test.h"

#define NUM_EVENTS 1
#define PAPI

int main ()
{
#ifdef PAPI
	int retval, i;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
    char *EventName[] = { "EXAMPLE_CONSTANT" };
	int events[NUM_EVENTS];
	
	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT )
		fprintf( stderr, "PAPI_library_init failed\n" );
	
	printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	
	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
		retval = PAPI_event_name_to_code( EventName[i], &events[i] );
		if( retval != PAPI_OK )
			fprintf( stderr, "PAPI_event_name_to_code failed\n" );
		else
			printf( "Name %s --- Code: %x\n", EventName[i], events[i] );
	}

	retval = PAPI_create_eventset( &EventSet );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_create_eventset failed\n" );
	
	retval = PAPI_add_events( EventSet, events, NUM_EVENTS );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_add_events failed\n" );
	
	retval = PAPI_start( EventSet );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_start failed\n" );
#endif
	
	
	printf("Example component test: Hello World\n");
	
	
#ifdef PAPI
	retval = PAPI_stop( EventSet, values );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_stop failed\n" );

	for( i = 0; i < NUM_EVENTS; i++ )
		printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
#endif
		
	return 0;
}

