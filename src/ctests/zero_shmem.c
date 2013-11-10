/* This file performs the following test: start, stop and timer
functionality for 2 slave OMP threads

   - It attempts to use the following two counters. It may use less
depending on hardware counter resource limitations. These are counted
in the default counting domain and default granularity, depending on
the platform. Usually this is the user domain (PAPI_DOM_USER) and
thread context (PAPI_GRN_THR).

     + PAPI_FP_INS
     + PAPI_TOT_CYC

Each of 2 slave pthreads:
   - Get cyc.
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
   - Get cyc.

Master pthread:
   - Get us.
   - Get cyc.
   - Fork threads
   - Wait for threads to exit
   - Get us.
   - Get cyc.
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <pthread.h>
#include "papi_test.h"

void
Thread( int n )
{
	int retval, num_tests = 1;
	int EventSet1 = PAPI_NULL;
	int mask1 = 0x5;
	int num_events1;
	long long **values;
	long long elapsed_us, elapsed_cyc;

	EventSet1 = add_test_events( &num_events1, &mask1, 1 );

	/* num_events1 is greater than num_events2 so don't worry. */

	values = allocate_test_space( num_tests, num_events1 );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	retval = PAPI_start( EventSet1 );
	if ( retval >= PAPI_OK )
		exit( 1 );

	do_flops( n );

	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval >= PAPI_OK )
		exit( 1 );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	remove_test_events( &EventSet1, mask1 );

	printf( "Thread %#x PAPI_FP_INS : \t%lld\n", n / 1000000,
			( values[0] )[0] );
	printf( "Thread %#x PAPI_TOT_CYC: \t%lld\n", n / 1000000,
			( values[0] )[1] );
	printf( "Thread %#x Real usec   : \t%lld\n", n / 1000000,
			elapsed_us );
	printf( "Thread %#x Real cycles : \t%lld\n", n / 1000000,
			elapsed_cyc );

	free_test_space( values, num_tests );
}

int
main( int argc, char **argv )
{
    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

	long long elapsed_us, elapsed_cyc;

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

#ifdef HAVE_OPENSHMEM
	start_pes( 2 );
	Thread( 1000000 * ( _my_pe(  ) + 1 ) );
#else
	test_skip( __FILE__, __LINE__, "OpenSHMEM support not found, skipping.", 0);
#endif

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	printf( "Master real usec   : \t%lld\n", elapsed_us );
	printf( "Master real cycles : \t%lld\n", elapsed_cyc );

	exit( 0 );
}
