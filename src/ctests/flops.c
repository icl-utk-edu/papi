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

#include "do_loops.h"

#define INDEX 1000

static float matrixa[INDEX][INDEX], matrixb[INDEX][INDEX], mresult[INDEX][INDEX];

int
main( int argc, char **argv )
{
	float real_time, proc_time, mflops;
	long long flpins;
	int retval;
	int i, j, k, fip = 0;
	int quiet=0;

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
	for ( i = 0; i < INDEX; i++ ) {
		for ( j = 0; j < INDEX; j++) {
			mresult[j][i] = 0.0;
			matrixa[j][i] = matrixb[j][i] = ( float ) rand(  ) * ( float ) 1.1;
		}
	}

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
	for ( i = 0; i < INDEX; i++ )
		for ( j = 0; j < INDEX; j++ )
			for ( k = 0; k < INDEX; k++ )
				mresult[i][j] = mresult[i][j] + matrixa[i][k] * matrixb[k][j];

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

	dummy( ( void * ) mresult );

	if ( !quiet ) {
		if ( fip == 1 ) {
			printf( "Real_time: %f Proc_time: %f Total flpins: ",
					real_time, proc_time );
		} else {
			printf( "Real_time: %f Proc_time: %f Total flpops: ",
					real_time, proc_time );
		}
		printf( "%lld", flpins );
		printf( " MFLOPS: %f\n", mflops );
	}

	test_pass( __FILE__ );

	return 0;

}
