/*
 * File:    freq.c
 * CVS:     $Id$
 * Author:  Brian Sheely
 *          bsheely@eecs.utk.edu
 */

#include <stdio.h>
#include <sys/time.h>

#define MAX_ETIME 86400

typedef void ( *test_funct ) ( void );

static double delta_t = 0.0;
static struct itimerval first_u;	   /* user time */
static struct itimerval first_w;	   /* wall time */

int ax = 0;
int bx = 0;
int cx = 0;

/* Compute peformance by doing 200 repeated additions */
static void
add_test( void )
{
	int a = ax;
	int b = bx;
	int c = cx;
	
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;

	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c; a += b; a += c;
	
	ax = a + b + c;
}

static void
add_dummy( void )
{
	int a = ax;
	int b = bx;
	int c = cx;
	ax = a + b + c;
}

static void
init_etime( void )
{
	first_u.it_interval.tv_sec = 0;
	first_u.it_interval.tv_usec = 0;
	first_u.it_value.tv_sec = MAX_ETIME;
	first_u.it_value.tv_usec = 0;
	setitimer( ITIMER_VIRTUAL, &first_u, NULL );

	first_w.it_interval.tv_sec = 0;
	first_w.it_interval.tv_usec = 0;
	first_w.it_value.tv_sec = MAX_ETIME;
	first_w.it_value.tv_usec = 0;
	setitimer( ITIMER_REAL, &first_w, NULL );
}

static double
get_etime( void )
{
	struct itimerval curr;
	getitimer( ITIMER_VIRTUAL, &curr );
	return ( double ) ( ( first_u.it_value.tv_sec - curr.it_value.tv_sec ) +
						( first_u.it_value.tv_usec -
						  curr.it_value.tv_usec ) * 1e-6 );
}

static double
ftime( test_funct funct, double e )
{
	int cnt = 1;
	double tmin;
	double tmeas = 0.0;

	/* Make sure timer interval has been computed */
	if ( delta_t == 0.0 ) {
		double start;
		init_etime(  );
		start = get_etime(  );
		while ( ( delta_t = get_etime(  ) - start ) <= 1e-6 );
	}

	tmin = delta_t / e + delta_t;

	while ( tmeas < tmin ) {
		int c = cnt;
		double start = get_etime(  );

		while ( c-- > 0 ) {
			funct(  );
		}

		tmeas = get_etime(  ) - start;

		if ( tmeas < tmin )
			cnt += cnt;
	}

	return tmeas / cnt;
}

int
compute_freq(  )
{
	double atime = ftime( add_test, 0.01 );
	double dtime = ftime( add_dummy, 0.01 );
	double secs = atime - dtime;
	double mhz = ( 200.0 / secs ) / 1000000.0;
	printf( "The clock frequency was computed to be %0.1f Megahertz\n", mhz );
	return ( int ) mhz;
}
