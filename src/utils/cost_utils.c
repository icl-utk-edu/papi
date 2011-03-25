#include "papi_test.h"

int num_iters = NUM_ITERS;

/* computes min, max, and mean for an array; returns std deviation */
double
do_stats( long long *array, long long *min, long long *max, double *average )
{
	int i;
	double std, tmp;

	*min = *max = array[0];
	*average = 0;
	for ( i = 0; i < num_iters; i++ ) {
		*average += ( double ) array[i];
		if ( *min > array[i] )
			*min = array[i];
		if ( *max < array[i] )
			*max = array[i];
	}
	*average = *average / ( double ) num_iters;
	std = 0;
	for ( i = 0; i < num_iters; i++ ) {
		tmp = ( double ) array[i] - ( *average );
		std += tmp * tmp;
	}
	std = sqrt( std / ( num_iters - 1 ) );
	return ( std );
}

void
do_std_dev( long long *a, int *s, double std, double ave )
{
	int i, j;
	double dev[10];

	for ( i = 0; i < 10; i++ ) {
		dev[i] = std * ( i + 1 );
		s[i] = 0;
	}

	for ( i = 0; i < num_iters; i++ ) {
		for ( j = 0; j < 10; j++ ) {
			if ( ( ( double ) a[i] - dev[j] ) > ave )
				s[j]++;
		}
	}
}

void
do_dist( long long *a, long long min, long long max, int bins, int *d )
{
	int i, j;
	int dmax = 0;
	int range = ( int ) ( max - min + 1 );	/* avoid edge conditions */

	/* clear the distribution array */
	for ( i = 0; i < bins; i++ ) {
		d[i] = 0;
	}

	/* scan the array to distribute cost per bin */
	for ( i = 0; i < num_iters; i++ ) {
		j = ( ( int ) ( a[i] - min ) * bins ) / range;
		d[j]++;
		if ( j && ( dmax < d[j] ) )
			dmax = d[j];
	}

	/* scale each bin to a max of 100 */
	for ( i = 1; i < bins; i++ ) {
		d[i] = ( d[i] * 100 ) / dmax;
	}
}

