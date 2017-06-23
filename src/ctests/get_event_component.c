/*
 * File:    get_event_component.c
 * Author:  Vince Weaver
 *	        vweaver1@eecs.utk.edu
 */

/*
  This test makes sure PAPI_get_event_component() works
*/

#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{

    int i;
    int retval;
    PAPI_event_info_t info;
    int numcmp, cid, our_cid;
    const PAPI_component_info_t* cmpinfo;

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /* Init PAPI library */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
       test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
    }

    numcmp = PAPI_num_components(  );


    /* Loop through all components */
    for( cid = 0; cid < numcmp; cid++ )
    {
        cmpinfo = PAPI_get_component_info( cid );

         if (cmpinfo  == NULL)
         {
            test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 2 );
         }

         if (cmpinfo->disabled && !TESTS_QUIET) {
           printf( "Name:   %-23s %s\n", cmpinfo->name ,cmpinfo->description);
           printf("   \\-> Disabled: %s\n",cmpinfo->disabled_reason);
           continue;
         }


       i = 0 | PAPI_NATIVE_MASK;
       retval = PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );
       if (retval!=PAPI_OK) continue;

       do {
	   if (PAPI_get_event_info( i, &info ) != PAPI_OK) {
	       if (!TESTS_QUIET) {
		   printf("Getting information about event: %#x failed\n", i);
	       }
	       continue;
	   }
	  our_cid=PAPI_get_event_component(i);

	  if (our_cid!=cid) {
	     if (!TESTS_QUIET) {
	        printf("%d %d %s\n",cid,our_cid,info.symbol);
	     }
             test_fail( __FILE__, __LINE__, "component mismatch", 1 );
	  }

	  if (!TESTS_QUIET) {
	    printf("%d %d %s\n",cid,our_cid,info.symbol);
	  }

	  
       } while ( PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid ) == PAPI_OK );

    }

	test_pass( __FILE__ );

	return 0;
}
