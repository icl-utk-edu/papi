/*
 * A simple example for the use of PAPI, using PAPI_ipc
 * -Kevin London
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"


#define INDEX 500

int
main( int argc, char **argv )
{
	extern void dummy( void * );
	float matrixa[INDEX][INDEX], matrixb[INDEX][INDEX], mresult[INDEX][INDEX];
	float real_time, proc_time, ipc;
	long long ins;
	int retval;
	int i, j, k;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Initialize the Matrix arrays */
	for( i = 0; i < INDEX; i++ ) {
	   for( j= 0; j < INDEX; j++ ) {
	       mresult[i][j] = 0.0;
	       matrixa[i][j] = matrixb[i][j] = ( float ) rand(  ) * ( float ) 1.1;
	   }
	}

	/* Setup PAPI library and begin collecting data from the counters */
	retval = PAPI_ipc( &real_time, &proc_time, &ins, &ipc );
	if (retval < PAPI_OK ) {
		if (!quiet) printf("Trouble starting IPC\n");
		test_skip( __FILE__, __LINE__, "PAPI_ipc", retval );
	}

	/* Matrix-Matrix multiply */
	for ( i = 0; i < INDEX; i++ )
		for ( j = 0; j < INDEX; j++ )
			for ( k = 0; k < INDEX; k++ )
				mresult[i][j] = mresult[i][j] + matrixa[i][k] * matrixb[k][j];

	/* Collect the data into the variables passed in */
	if ( ( retval = PAPI_ipc( &real_time, &proc_time, &ins, &ipc ) ) < PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	dummy( ( void * ) mresult );

	if ( !quiet ) {
		printf( "Real_time: %f Proc_time: %f Total ins: ", real_time,
				proc_time );
		printf( LLDFMT, ins );
		printf( " IPC: %f\n", ipc );
	}

           /* This should not happen unless the optimizer */
           /* gets too good                               */
        if (ins < INDEX*INDEX) {
	   test_fail( __FILE__, __LINE__, "Instruction count too low.", 
		      5 );
	}
           /* Something is broken, or else you have a really */
           /* slow processor                                 */
        if (ipc<0.01 ) {
	   test_fail( __FILE__, __LINE__, "IPC equals zero.", 
		      5 );
	}

	test_pass( __FILE__ );

	return 0;
}
