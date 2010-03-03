/*
 *  Test PAPI with fork() and exec().
 */

#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "papi.h"
#include "papi_test.h"

#define MAX_EVENTS  3

int Event[MAX_EVENTS] = {
	PAPI_TOT_CYC,
	PAPI_FP_INS,
	PAPI_FAD_INS,
};

int Threshold[MAX_EVENTS] = {
	8000000,
	4000000,
	4000000,
};

int num_events = 1;
int EventSet = PAPI_NULL;
char *name = "unknown";
struct timeval start, last;
long count, total;

void
my_handler( int EventSet, void *pc, long long ovec, void *context )
{
	( void ) EventSet;
	( void ) pc;
	( void ) ovec;
	( void ) context;

	count++;
	total++;
}

void
zero_count( void )
{
	gettimeofday( &start, NULL );
	last = start;
	count = 0;
	total = 0;
}

#define HERE(str)  printf("[%d] %s, %s\n", getpid(), name, str);

void
print_rate( char *str )
{
	static int last_count = -1;
	struct timeval now;
	double st_secs, last_secs;

	gettimeofday( &now, NULL );
	st_secs = ( double ) ( now.tv_sec - start.tv_sec )
		+ ( ( double ) ( now.tv_usec - start.tv_usec ) ) / 1000000.0;
	last_secs = ( double ) ( now.tv_sec - last.tv_sec )
		+ ( ( double ) ( now.tv_usec - last.tv_usec ) ) / 1000000.0;
	if ( last_secs <= 0.001 )
		last_secs = 0.001;

	printf( "[%d] %s, time = %.3f, total = %ld, last = %ld, rate = %.1f/sec\n",
			getpid(  ), str, st_secs, total, count,
			( ( double ) count ) / last_secs );

	if ( last_count != -1 ) {
		if ( count < .1 * last_count ) {
			test_fail( name, __LINE__, "Interrupt rate changed!", 1 );
			exit( 1 );
		}
	}
	last_count = ( int ) count;
	count = 0;
	last = now;
}

void
do_cycles( int program_time )
{
	struct timeval start, now;
	double x, sum;

	gettimeofday( &start, NULL );

	for ( ;; ) {
		sum = 1.0;
		for ( x = 1.0; x < 250000.0; x += 1.0 )
			sum += x;
		if ( sum < 0.0 )
			printf( "==>>  SUM IS NEGATIVE !!  <<==\n" );

		gettimeofday( &now, NULL );
		if ( now.tv_sec >= start.tv_sec + program_time )
			break;
	}
}

void
my_papi_init( void )
{
	if ( PAPI_library_init( PAPI_VER_CURRENT ) != PAPI_VER_CURRENT )
		test_fail( name, __LINE__, "PAPI_library_init failed", 1 );
}

void
my_papi_start( void )
{
	int ev;

	EventSet = PAPI_NULL;

	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
		test_fail( name, __LINE__, "PAPI_create_eventset failed", 1 );

	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_add_event( EventSet, Event[ev] ) != PAPI_OK )
			test_fail( name, __LINE__, "PAPI_add_event failed", 1 );
	}

	for ( ev = 0; ev < num_events; ev++ ) {
		if ( PAPI_overflow( EventSet, Event[ev], Threshold[ev], 0, my_handler )
			 != PAPI_OK ) {
			test_fail( name, __LINE__, "PAPI_overflow failed", 1 );
		}
	}

	if ( PAPI_start( EventSet ) != PAPI_OK )
		test_fail( name, __LINE__, "PAPI_start failed", 1 );
}

void
my_papi_stop( void )
{
	if ( PAPI_stop( EventSet, NULL ) != PAPI_OK )
		test_fail( name, __LINE__, "PAPI_stop failed", 1 );
}

void
run( char *str, int len )
{
	int n;

	for ( n = 1; n <= len; n++ ) {
		do_cycles( 1 );
		print_rate( str );
	}
}

int
main( int argc, char **argv )
{
	char buf[100];

	if ( argc < 2 || sscanf( argv[1], "%d", &num_events ) < 1 )
		num_events = 1;
	if ( num_events < 0 || num_events > MAX_EVENTS )
		num_events = 1;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */
	do_cycles( 1 );
	zero_count(  );
	my_papi_init(  );
	name = argv[0];
	printf( "[%d] %s, num_events = %d\n", getpid(  ), name, num_events );
	sprintf( buf, "%d", num_events );
	my_papi_start(  );
	run( name, 3 );
#if defined(PCHILD)
	HERE( "stop" );
	my_papi_stop(  );
	HERE( "end" );
	test_pass( name, NULL, 0 );
#elif defined(PEXEC)
	HERE( "stop" );
	my_papi_stop(  );
	HERE( "exec(./child_overflow)" );
	if ( access( "./child_overflow", X_OK ) == 0 )
		execl( "./child_overflow", "./child_overflow",
			   ( TESTS_QUIET ? "TESTS_QUIET" : NULL ), NULL );
	else if ( access( "./ctests/child_overflow", X_OK ) == 0 )
		execl( "./ctests/child_overflow", "./ctests/child_overflow",
			   ( TESTS_QUIET ? "TESTS_QUIET" : NULL ), NULL );
	test_fail( name, __LINE__, "exec failed", 1 );
#elif defined(SYSTEM)
	HERE( "system(./child_overflow)" );
	if ( access( "./child_overflow", X_OK ) == 0 )
		( TESTS_QUIET ? system( "./child_overflow TESTS_QUIET" ) :
		  system( "./child_overflow" ) );
	else if ( access( "./ctests/child_overflow", X_OK ) == 0 )
		( TESTS_QUIET ? system( "./ctests/child_overflow TESTS_QUIET" ) :
		  system( "./ctests/child_overflow" ) );
	test_pass( name, NULL, 0 );
#elif defined(SYSTEM2)
	HERE( "system(./burn)" );
	if ( access( "./burn", X_OK ) == 0 )
		( TESTS_QUIET ? system( "./burn TESTS_QUIET" ) : system( "./burn" ) );
	else if ( access( "./ctests/burn", X_OK ) == 0 )
		( TESTS_QUIET ? system( "./ctests/burn TESTS_QUIET" ) :
		  system( "./ctests/burn" ) );
	test_pass( name, NULL, 0 );
#else
	HERE( "fork" );
	{
		int ret = fork(  );
		if ( ret < 0 )
			test_fail( name, __LINE__, "fork failed", 1 );
		if ( ret == 0 ) {
			/*
			 * Child process.
			 */
			zero_count(  );
			my_papi_init(  );
			my_papi_start(  );
			run( "child", 5 );
			HERE( "stop" );
			my_papi_stop(  );
			sleep( 3 );
			HERE( "end" );
			exit( 0 );
		}
		run( "main", 14 );
		my_papi_stop(  );
		{
			int status;
			wait( &status );
			HERE( "end" );
			if ( WEXITSTATUS( status ) != 0 )
				test_fail( name, __LINE__, "child failed", 1 );
			else
				test_pass( name, NULL, 0 );
		}
	}
#endif
	exit( 0 );
}
