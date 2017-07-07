/*
 * A simple example for the use of PAPI, the number of flops you should
 * get is about INDEX^3  on machines that consider add and multiply one flop
 * such as SGI, and 2*(INDEX^3) that don't consider it 1 flop such as INTEL
 * -Kevin London
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"
#include "display_error.h"

int
main( int argc, char **argv )
{
	float real_time, proc_time, mflops;
	long long flpins;
	int retval;
	int fip = 0;
	int quiet=0;
	long long expected;
	double double_result,error;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Initialize PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Try to use one of the FP events */
	if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK ) {
		fip = 1;
	}
	else if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK ) {
		fip = 2;
	}
	else {
		if ( !quiet ) printf( "PAPI_FP_INS and PAPI_FP_OPS are not defined for this platform.\n" );
		test_skip(__FILE__,__LINE__,"No FP events available",1);
	}

	/* Shutdown?  */
	/* I guess because it would interfere with the high-level interface? */
	PAPI_shutdown(  );

	/* Initialize the Matrix arrays */
	expected=flops_float_init_matrix();

	/* Setup PAPI library and begin collecting data from the counters */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	}
	else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	/* Matrix-Matrix multiply */
	double_result=flops_float_matrix_matrix_multiply();

	/* Collect the data into the variables passed in */
	if ( fip == 1 ) {
		retval = PAPI_flips( &real_time, &proc_time, &flpins, &mflops );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
		}
	} else {
		retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops );
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
		}
	}

	if (!quiet) printf("result=%lf\n",double_result);

	if ( !quiet ) {
		printf( "Real_time: %f Proc_time: %f MFLOPS: %f\n",
					real_time, proc_time, mflops );
		if ( fip == 1 ) {
			printf( "Total flpins: ");
		} else {
			printf( "Total flpops: ");
		}
		printf( "%lld\n\n", flpins );
	}

	error=display_error(flpins,flpins,flpins,expected,quiet);

	if ((error > 1.0) || (error<-1.0)) {
		if (!quiet) printf("Instruction count off by more than 1%%\n");
		test_fail( __FILE__, __LINE__, "Validation failed", 1 );
	}

	test_pass( __FILE__ );

	return 0;

}
