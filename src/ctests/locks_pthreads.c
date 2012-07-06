/* This file checks to make sure the locking mechanisms work correctly on the platform.
 * Platforms where the locking mechanisms are not implemented or are incorrectly implemented
 * will fail.  -KSL
 */

#include <pthread.h>
#include "papi_test.h"

volatile long long count = 0;
volatile long long tmpcount = 0;
volatile int num_iters = 0;

void
lockloop( int iters, volatile long long *mycount )
{
	int i;
	for ( i = 0; i < iters; i++ ) {
		PAPI_lock( PAPI_USR1_LOCK );
		*mycount = *mycount + 1;
		PAPI_unlock( PAPI_USR1_LOCK );
	}
}

void *
Slave( void *arg )
{
	long long duration;

	( void ) arg;

	sleep( 1 );
	duration = PAPI_get_real_usec(  );
	lockloop( 10000, &tmpcount );
	duration = PAPI_get_real_usec(  ) - duration;

	/* First one here set's the number */
	PAPI_lock( PAPI_USR2_LOCK );
	if ( num_iters == 0 ) {
		printf( "10000 iterations took %lld us.\n", duration );
		num_iters = ( int ) ( 10 * ( TIME_LIMIT_IN_US / duration ) );
		printf( "Running %d iterations\n", num_iters );
	}
	PAPI_unlock( PAPI_USR2_LOCK );

	lockloop( num_iters, &count );
	pthread_exit( NULL );
}


int
main( int argc, char **argv )
{
	pthread_t slaves[MAX_THREADS];
	int rc, i, nthr;
	int retval;
	const PAPI_hw_info_t *hwinfo = NULL;

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );	

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( hwinfo = PAPI_get_hardware_info(  ) ) == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	retval =
		PAPI_thread_init( ( unsigned long ( * )( void ) ) ( pthread_self ) );
	if ( retval != PAPI_OK ) {
		if ( retval == PAPI_ECMP )
			test_skip( __FILE__, __LINE__, "PAPI_thread_init", retval );
		else
			test_fail( __FILE__, __LINE__, "PAPI_thread_init", retval );
	}

	if ( hwinfo->ncpu > MAX_THREADS )
		nthr = MAX_THREADS;
	else
		nthr = hwinfo->ncpu;

	printf( "Creating %d threads\n", nthr );

	for ( i = 0; i < nthr; i++ ) {
		rc = pthread_create( &slaves[i], NULL, Slave, NULL );
		if ( rc ) {
			retval = PAPI_ESYS;
			test_fail( __FILE__, __LINE__, "pthread_create", retval );
		}
	}

	for ( i = 0; i < nthr; i++ ) {
		pthread_join( slaves[i], NULL );
	}

	printf( "Expected: %lld Received: %lld\n", ( long long ) nthr * num_iters,
			count );
	if ( nthr * num_iters != count )
		test_fail( __FILE__, __LINE__, "Thread Locks", 1 );

	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
