/* This file attempts to test the PAPI_LD_INS			*/
/* performance counter (retired loads).				*/

/* This just does a generic matrix-matrix test			*/
/* Should have a comprehensive assembly language test		*/
/* (see my deterministic benchmark suite) but that would be	*/
/* a lot more complicated.					*/

/* by Vince Weaver, <vincent.weaver@maine.edu>	*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include <time.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"

#include "matrix_multiply.h"
#include "testcode.h"

#define SLEEP_RUNS 3


int main(int argc, char **argv) {

	int quiet;

	double error;

	int i;
	long long count,high=0,low=0,total=0,average=0;
	long long mmm_count;
	long long expected;
	int retval;
	int eventset=PAPI_NULL;

	quiet=tests_quiet(argc,argv);

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if (!quiet) {
		printf("\nTesting PAPI_LD_INS\n\n");
	}

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset,"PAPI_LD_INS");
	if (retval!=PAPI_OK) {
		if (!quiet) printf("Could not add PAPI_LD_INS\n");
		test_skip( __FILE__, __LINE__, "adding PAPI_LD_INS", retval );
	}


	/****************/
	/* Sleep test   */
	/****************/

	if (!quiet) {
		printf("Testing a sleep of 1 second (%d times):\n",SLEEP_RUNS);
	}

	for(i=0;i<SLEEP_RUNS;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		sleep(1);

		retval=PAPI_stop(eventset,&count);
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		}

		if (count>high) high=count;
		if ((low==0) || (count<low)) low=count;
		total+=count;
	}

	average=total/SLEEP_RUNS;

	if (!quiet) {
		printf("\tAverage should be low, as no loads when sleeping\n");
		printf("\tMeasured average: %lld\n",average);
	}

	if (average>100000) {
		if (!quiet) printf("Average cycle count too high!\n");
		test_fail( __FILE__, __LINE__, "idle average", retval );
	}

	/***********************************/
	/* testing a large number of loads */
	/***********************************/

	if (!quiet) {
		printf("\nTesting a large number of loads\n");
	}

	expected=naive_matrix_multiply_estimated_loads(quiet);

	PAPI_reset(eventset);
	PAPI_start(eventset);

	retval = execute_loads(expected);
	if (retval == CODE_UNIMPLEMENTED) {
		if (!quiet) {
			printf("\tNo asm test found for the current hardware. Testing matrix multiply\n");
		}
		naive_matrix_multiply(quiet);
	}

	retval=PAPI_stop(eventset,&count);

	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "Problem stopping!", retval );
	}

	if (!quiet) {
		printf("\tActual measured loads = %lld\n",count);
	}

	error=  100.0 * (double)(count-expected) / (double)expected;

	if (!quiet) {
		printf("\tExpected %lld, got %lld\n",expected,count);
		printf("\tError=%.2f%%\n",error);
	}

	if ((error>10.0) || (error<-10.0)) {

		if (!quiet) printf("Error too high!\n");
		test_fail( __FILE__, __LINE__, "Error too high", retval );
	}


	mmm_count=count;

	/************************************/
	/* Check for Linear Speedup         */
	/************************************/

	if (!quiet) printf("\nTesting for a linear cycle increase\n");

#define REPITITIONS 2

	PAPI_reset(eventset);
	PAPI_start(eventset);

	for(i=0;i<REPITITIONS;i++) {
		retval = execute_loads(expected);
		if (retval == CODE_UNIMPLEMENTED) {
			if (!quiet) {
				printf("\tNo asm test found for the current hardware. Testing matrix multiply\n");
			}
			naive_matrix_multiply(quiet);
		}
	}

	retval=PAPI_stop(eventset,&count);

	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "Problem stopping!", retval );
	}

	expected=mmm_count*REPITITIONS;

	error=  100.0 * (double)(count-expected) / (double)expected;

	if (!quiet) {
		printf("\tExpected %lld, got %lld\n",expected,count);
		printf("\tError=%.2f%%\n",error);
	}

	if ((error>10.0) || (error<-10.0)) {

		if (!quiet) printf("Error too high!\n");
		test_fail( __FILE__, __LINE__, "Error too high", retval );
	}

	if (!quiet) printf("\n");

	test_pass( __FILE__ );

	return 0;
}
