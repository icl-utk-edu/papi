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

#include <stdio.h>
#include <stdlib.h>
#include "papi_test.h"

#define NUM_EVENTS 1

int main (int argc, char **argv)
{

	int retval, i;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
        char *EventName[] = { "EXAMPLE_CONSTANT" };
	int events[NUM_EVENTS];

        /* Set TESTS_QUIET variable */
        tests_quiet( argc, argv );      

	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
	  test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
	
	if (!TESTS_QUIET) {
	   printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	}

	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
		retval = PAPI_event_name_to_code( EventName[i], &events[i] );
		if ( retval != PAPI_OK )
		   test_fail(__FILE__,__LINE__,
			     "PAPI_event_name_to_code failed\n", retval);
		else {
		  if (!TESTS_QUIET) 
                     printf( "Name %s --- Code: %x\n", 
			     EventName[i], events[i] );
		}
	}

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
	   test_fail( __FILE__, __LINE__,
		      "PAPI_create_eventset failed\n", retval );
	
	retval = PAPI_add_events( EventSet, events, NUM_EVENTS );
	if ( retval != PAPI_OK )
	   test_fail( __FILE__, __LINE__,
		      "PAPI_add_events failed\n", retval );
	
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
	   test_fail( __FILE__, __LINE__, 
		      "PAPI_start failed\n",retval );

	
	if (!TESTS_QUIET) {
	   printf("Example component test: Hello World\n");
	}
	

	retval = PAPI_stop( EventSet, values );
	if ( retval != PAPI_OK )
	   test_fail(  __FILE__, __LINE__, "PAPI_stop failed\n", retval);

	if (!TESTS_QUIET) {
	   for( i = 0; i < NUM_EVENTS; i++ )
	      printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
	}

	if (values[0]!=42) {
	   test_fail(  __FILE__, __LINE__, "Result should be 42!\n", 0);
	}

	test_pass( __FILE__, NULL, 0 );
		
	return 0;
}

