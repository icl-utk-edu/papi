#include <stdio.h>
#include <string.h>

#include "papi.h"

#include "do_loops.h"
#include "papi_test.h"

static int total = 0;				   /* total overflows */

static long long values[2];
static long long rapl_values[2];
static long long old_rapl_values[2] = {0,0};
static int rapl_backward=0;

int EventSet2=PAPI_NULL;

int quiet=0;

void handler( int EventSet, void *address, 
	      long long overflow_vector, void *context ) {

	( void ) context;
	( void ) address;
	( void ) overflow_vector;

#if 0
	fprintf( stderr, "handler(%d ) Overflow at %p! bit=%#llx \n",
                         EventSet, address, overflow_vector );
#endif
	
	PAPI_read(EventSet,values);
	if (!quiet) printf("%lld %lld\t",values[0],values[1]);
	PAPI_read(EventSet2,rapl_values);
	if (!quiet) printf("RAPL: %lld %lld\n",rapl_values[0],rapl_values[1]);

	if ((rapl_values[0]<old_rapl_values[0]) ||
	    (rapl_values[1]<old_rapl_values[1])) {
	   if (!quiet) printf("RAPL decreased!\n");
	   rapl_backward=1;
	}
	old_rapl_values[0]=rapl_values[0];
	old_rapl_values[1]=rapl_values[1];


	total++;
}


void do_ints(int n,int quiet)
{
  int i,c=n;

  for(i=0;i<n;i++) {
     c+=c*i*n;
  }
  if (!quiet) printf("%d\n",c);
}



int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL;
	long long values0[2],values1[2],values2[2];
	int num_flops = 3000000, retval;
	int mythreshold = 1000000;
	char event_name1[PAPI_MAX_STR_LEN];
        int PAPI_event;
	int cid,numcmp,rapl_cid;
	const PAPI_component_info_t *cmpinfo = NULL;
	int i;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	  test_fail(__FILE__, __LINE__,"PAPI_library_init",retval);
	}

	numcmp = PAPI_num_components();

	for(cid=0; cid<numcmp; cid++) {

	  if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	    test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n", 0);
	  }

	  if (strstr(cmpinfo->name,"rapl")) {
	    rapl_cid=cid;
	    if (!TESTS_QUIET) printf("Found rapl component at cid %d\n",
				     rapl_cid);
	    if (cmpinfo->num_native_events==0) {
              test_skip(__FILE__,__LINE__,"No rapl events found",0);
	    }
	    break;
	  }
	}

	/* Component not found */
	if (cid==numcmp) {
	  test_skip(__FILE__,__LINE__,"No rapl component found\n",0);
	}


	/* add PAPI_TOT_CYC and PAPI_TOT_INS */
	retval=PAPI_create_eventset(&EventSet);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_create_eventset",retval);
	}

	retval=PAPI_add_event(EventSet,PAPI_TOT_CYC);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_add_event",retval);
	}

	retval=PAPI_add_event(EventSet,PAPI_TOT_INS);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_add_event",retval);
	}

	/* Add some RAPL events */
	retval=PAPI_create_eventset(&EventSet2);
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_create_eventset",retval);
	}

	/* Add an event for each packages 0-n  */
	i = 0;
	do {
		char buffer[80];
		sprintf(&(buffer[0]), "rapl:::PACKAGE_ENERGY:PACKAGE%d", i);
		retval=PAPI_add_named_event(EventSet2,buffer);
		++i;
	/* protect against insane PAPI library, the value 64 is the same value as 
     * RAPL_MAX_COUNTERS in linux-rapl.c, and feels reasonable. */
	} while ( 0 < retval && i < 64 );

	PAPI_event=PAPI_TOT_CYC;

	/* arbitrary */
	mythreshold = 2000000;
	if (!TESTS_QUIET) {
	   printf("Using %#x for the overflow event, threshold %d\n",
		  PAPI_event,mythreshold);
	}

	/* Start the run calibration run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}

	do_ints(num_flops,TESTS_QUIET);
	do_flops( 3000000 );

	/* stop the calibration run */
	retval = PAPI_stop( EventSet, values0 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}


	/* set up overflow handler */
	retval = PAPI_overflow( EventSet,PAPI_event,mythreshold, 0, handler );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_overflow",retval);
	}

	/* Start overflow run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}
	retval = PAPI_start( EventSet2 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_start",retval);
	}

	do_ints(num_flops,TESTS_QUIET);
	do_flops( num_flops );

	/* stop overflow run */
	retval = PAPI_stop( EventSet, values1 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}

	retval = PAPI_stop( EventSet2, values2 );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_stop",retval);
	}

	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_overflow",retval);
	}

	retval = PAPI_event_code_to_name( PAPI_event, event_name1 );
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__,"PAPI_event_code_to_name\n", retval);
	}

	if (!TESTS_QUIET) {
	   printf("%s: %lld %lld\n",event_name1,values0[0],values1[0]);
	}

	retval = PAPI_event_code_to_name( PAPI_TOT_INS, event_name1 );
	if (retval != PAPI_OK) {
	  test_fail(__FILE__, __LINE__,"PAPI_event_code_to_name\n",retval);
	}

	if (!TESTS_QUIET) {
	   printf("%s: %lld %lld\n",event_name1,values0[1],values1[1]);
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK ) {
	  test_fail(__FILE__, __LINE__,"PAPI_cleanup_eventset",retval);
	}

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail(__FILE__, __LINE__,"PAPI_destroy_eventset",retval);
	}

	if (rapl_backward) {
	   test_fail(__FILE__, __LINE__,"RAPL counts went backward!",0);
	}

	test_pass( __FILE__ );

	return 0;
}
