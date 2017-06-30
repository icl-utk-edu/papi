/****************************************************************************
 *C
 *C     matrix-hl.f
 *C     An example of matrix-matrix multiplication and using PAPI high level
 *C     to look at the performance. written by Kevin London
 *C     March 2000
 *C     Added to c tests to check stop
 *C****************************************************************************
 */


#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

int
main( int argc, char **argv )
{

#define NROWS1 175
#define NCOLS1 225
#define NROWS2 NCOLS1
#define NCOLS2 150
	double p[NROWS1][NCOLS1], q[NROWS2][NCOLS2], r[NROWS1][NCOLS2];
	int i, j, k, num_events, retval;
	/*     PAPI standardized event to be monitored */
	int event[2];
	/*     PAPI values of the counters */
	long long values[2], tmp;
	int quiet;

	quiet = tests_quiet( argc, argv );

	/*     Setup default values */
	num_events = 0;

	/*     See how many hardware events at one time are supported
	 *     This also initializes the PAPI library */
	num_events = PAPI_num_counters(  );
	if ( num_events < 2 ) {
		if (!quiet) printf( "This example program requries the architecture to "
				"support 2 simultaneous hardware events...shutting down.\n" );
		test_skip( __FILE__, __LINE__, "PAPI_num_counters", 1 );
	}

	if ( !quiet )
		printf( "Number of hardware counters supported: %d\n", num_events );

	if ( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK )
		event[0] = PAPI_FP_OPS;
	else if ( PAPI_query_event( PAPI_FP_INS ) == PAPI_OK )
		event[0] = PAPI_FP_INS;
	else
		event[0] = PAPI_TOT_INS;

	/*     Time used */
	event[1] = PAPI_TOT_CYC;

	/*     matrix 1: read in the matrix values */
	for ( i = 0; i < NROWS1; i++ )
		for ( j = 0; j < NCOLS1; j++ )
			p[i][j] = i * j * 1.0;

	for ( i = 0; i < NROWS2; i++ )
		for ( j = 0; j < NCOLS2; j++ )
			q[i][j] = i * j * 1.0;

	for ( i = 0; i < NROWS1; i++ )
		for ( j = 0; j < NCOLS2; j++ )
			r[i][j] = i * j * 1.0;

	/*     Set up the counters */
	num_events = 2;
	retval = PAPI_start_counters( event, num_events );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );

	/*     Clear the counter values */
	retval = PAPI_read_counters( values, num_events );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );

	/*     Compute the matrix-matrix multiplication  */
	for ( i = 0; i < NROWS1; i++ )
		for ( j = 0; j < NCOLS2; j++ )
			for ( k = 0; k < NCOLS1; k++ )
				r[i][j] = r[i][j] + p[i][k] * q[k][j];

	/*     Stop the counters and put the results in the array values  */
	retval = PAPI_stop_counters( values, num_events );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );

	/*  Make sure the compiler does not optimize away the multiplication
	 *  with dummy(r);
	 */
	dummy( r );

	if ( !quiet ) {
		if ( event[0] == PAPI_TOT_INS ) {
			printf( TAB1, "TOT Instructions:", values[0] );
		} else {
			printf( TAB1, "FP Instructions:", values[0] );
		}
		printf( TAB1, "Cycles:", values[1] );
	}

	/*  
	 *  Intel Core overreports flops by 50% when using -O
	 *  Use -O2 or -O3 to produce the expected # of flops
	 */

	if ( event[0] == PAPI_FP_INS ) {
		/*     Compare measured FLOPS to expected value */
		tmp =
			2 * ( long long ) ( NROWS1 ) * ( long long ) ( NCOLS2 ) *
			( long long ) ( NCOLS1 );
		if ( abs( ( int ) values[0] - ( int ) tmp ) > ( double ) tmp * 0.05 ) {
			/*     Maybe we are counting FMAs? */
			tmp = tmp / 2;
			if ( abs( ( int ) values[0] - ( int ) tmp ) >
				 ( double ) tmp * 0.05 ) {
				printf( "\n" TAB1, "Expected operation count: ", 2 * tmp );
				printf( TAB1, "Or possibly (using FMA):  ", tmp );
				printf( TAB1, "Instead I got:            ", values[0] );
				test_fail( __FILE__, __LINE__,
						   "Unexpected FLOP count (check vector operations)",
						   1 );
			}
		}
	}
	test_pass( __FILE__ );

	return 0;

}
