/* This code attempts to test the L1 Data Cache Accesses	*/
/* performance counter PAPI_L1_DCA				*/

/* by Vince Weaver, <vincent.weaver@maine.edu>			*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#include "display_error.h"

#define NUM_RUNS 100

#define ARRAYSIZE 65536

static double array[ARRAYSIZE];

int main(int argc, char **argv) {

	int i;
	int quiet;
	int eventset=PAPI_NULL;

	int retval;
	int num_runs=NUM_RUNS;
	long long high,low,average,expected=ARRAYSIZE;
	long long count,total;
	double aSumm = 0.0;
	double error;

	quiet=tests_quiet(argc,argv);

	if (!quiet) {
		printf("Testing the PAPI_L1_DCA event\n");
	}

	/* Init the PAPI library */
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
		test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);
	}

	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval=PAPI_add_named_event(eventset,"PAPI_L1_DCA");
	if (retval!=PAPI_OK) {
		test_skip( __FILE__, __LINE__, "adding PAPI_L1_DCA", retval );
	}


	/*******************************************************************/
	/* Test if the C compiler uses a sane number of data cache acceess */
	/* This tests writes to memory.				           */
	/*******************************************************************/

	if (!quiet) {
		printf("Write Test: Initializing an array of %d doubles:\n",
			ARRAYSIZE);
	}

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		for(i=0; i<ARRAYSIZE; i++) {
			array[i]=(double)i;
		}
		retval=PAPI_stop(eventset,&count);

                if (retval!=PAPI_OK) {
                        test_fail( __FILE__, __LINE__,
                                "reading PAPI_TOT_INS", retval );
                }

                if (count>high) high=count;
                if ((low==0) || (count<low)) low=count;
                total+=count;
        }

        average=(total/num_runs);

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) printf("\n");


	/*******************************************************************/
	/* Test if the C compiler uses a sane number of data cache acceess */
	/* This tests writes to memory.				           */
	/*******************************************************************/

	if (!quiet) {
		printf("Read Test:  Summing an array of %d doubles:\n",
			ARRAYSIZE);
	}

	high=0; low=0; total=0;

	for(i=0;i<num_runs;i++) {

		PAPI_reset(eventset);
		PAPI_start(eventset);

		for(i=0; i<ARRAYSIZE; i++) {
			aSumm += array[i];
		}

		retval=PAPI_stop(eventset,&count);

                if (retval!=PAPI_OK) {
                        test_fail( __FILE__, __LINE__,
                                "reading PAPI_TOT_INS", retval );
                }

                if (count>high) high=count;
                if ((low==0) || (count<low)) low=count;
                total+=count;
        }

        average=(total/num_runs);

	error=display_error(average,high,low,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		test_fail( __FILE__, __LINE__, "Error too high", 1 );
	}

	if (!quiet) {
		printf("Read test (%lf):\n",aSumm);
		printf("\n");
	}

	test_pass(__FILE__);

	return 0;
}
