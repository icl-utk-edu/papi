#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "papi_test.h"

#ifdef SETMAX
#define MAX SETMAX
#else
#define MAX 10000
#endif
#define TIMES 1000

#define PAPI_MAX_EVENTS 2
long long PAPI_values1[PAPI_MAX_EVENTS];
long long PAPI_values2[PAPI_MAX_EVENTS];
long long PAPI_values3[PAPI_MAX_EVENTS];
static int EventSet = PAPI_NULL;

extern int TESTS_QUIET;				   /* Declared in test_utils.c */

int
main( argc, argv )
int argc;
char *argv[];
{
	int i, retval;
	double a[MAX], b[MAX];
	void funcX(  ), funcA(  );

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	for ( i = 0; i < MAX; i++ ) {
		a[i] = 0.0;
		b[i] = 0.;
	}

	for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
		PAPI_values1[i] = PAPI_values2[i] = 0;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

#ifdef MULTIPLEX
	if ( !TESTS_QUIET ) {
		printf( "Activating PAPI Multiplex\n" );
	}
	init_multiplex(  );
#endif

	retval = PAPI_create_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI set event fail\n", retval );

#ifdef MULTIPLEX
	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals.
	   0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

	retval = PAPI_set_multiplex( EventSet );
        if (retval == PAPI_ENOSUPP) {
	   test_skip( __FILE__, __LINE__, "Multiplex not supported", 1 );
	}
	else if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_multiplex fails \n", retval );
#endif

	retval = PAPI_add_event( EventSet, PAPI_FP_INS );
	if ( retval < PAPI_OK ) {
		retval = PAPI_add_event( EventSet, PAPI_TOT_INS );
		if ( retval < PAPI_OK )
			test_fail( __FILE__, __LINE__,
					   "PAPI add PAPI_FP_INS or PAPI_TOT_INS fail\n", retval );
		else if ( !TESTS_QUIET ) {
			printf( "PAPI_TOT_INS\n" );
		}
	} else if ( !TESTS_QUIET ) {
		printf( "PAPI_FP_INS\n" );
	}

	retval = PAPI_add_event( EventSet, PAPI_TOT_CYC );
	if ( retval < PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI add PAPI_TOT_CYC  fail\n",
				   retval );
	if ( !TESTS_QUIET ) {
		printf( "PAPI_TOT_CYC\n" );
	}

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI start fail\n", retval );

	funcX( a, b, MAX );

	retval = PAPI_read( EventSet, PAPI_values1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI read fail \n", retval );

	funcX( a, b, MAX );

	retval = PAPI_read( EventSet, PAPI_values2 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI read fail \n", retval );

#ifdef RESET
	retval = PAPI_reset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI read fail \n", retval );
#endif

	funcA( a, b, MAX );

	retval = PAPI_stop( EventSet, PAPI_values3 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI read fail \n", retval );

	if ( !TESTS_QUIET ) {
		printf( "values1 is:\n" );
		for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
			printf( LLDFMT15, PAPI_values1[i] );

		printf( "\nvalues2 is:\n" );
		for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
			printf( LLDFMT15, PAPI_values2[i] );
		printf( "\nvalues3 is:\n" );
		for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
			printf( LLDFMT15, PAPI_values3[i] );

#ifndef RESET
		printf( "\nPAPI value (2-1) is : \n" );
		for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
			printf( LLDFMT15, PAPI_values2[i] - PAPI_values1[i] );
		printf( "\nPAPI value (3-2) is : \n" );
		for ( i = 0; i < PAPI_MAX_EVENTS; i++ ) {
		  long long diff;
                  diff = PAPI_values3[i] - PAPI_values2[i];
		  printf( LLDFMT15, diff);
		  if (diff<0) {
		    test_fail( __FILE__, __LINE__, "Multiplexed counter decreased", 1 );
		  }
		}
#endif

		printf( "\n\nVerification:\n" );
		printf( "From start to first PAPI_read %d fp operations are made.\n",
				2 * MAX * TIMES );
		printf( "Between 1st and 2nd PAPI_read %d fp operations are made.\n",
				2 * MAX * TIMES );
		printf( "Between 2nd and 3rd PAPI_read %d fp operations are made.\n",
				0 );
		printf( "\n" );
	}
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}

void
funcX( a, b, n )
double a[MAX], b[MAX];
int n;
{
	int i, k;
	for ( k = 0; k < TIMES; k++ )
		for ( i = 0; i < n; i++ )
			a[i] = a[i] * b[i] + 1.;
}

void
funcA( a, b, n )
double a[MAX], b[MAX];
int n;
{
	int i, k;
	double t[MAX];
	for ( k = 0; k < TIMES; k++ )
		for ( i = 0; i < n; i++ ) {
			t[i] = b[n - i];
			b[i] = a[n - i];
			a[i] = t[i];
		}
}
