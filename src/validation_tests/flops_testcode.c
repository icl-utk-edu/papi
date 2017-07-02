/* This includes various workloads that had been scattered all over */
/* the various ctests.  The goal is to have them in one place, and */
/* share them, as well as maybe have only one file that has to be */
/* compiled with reduced optimizations */

#include <stdio.h>
#include <stdlib.h>

#include "testcode.h"

#define ROWS	1000
#define COLUMNS	1000

static float matrixa[ROWS][COLUMNS],
		matrixb[ROWS][COLUMNS],
		mresult[ROWS][COLUMNS];


int flops_init_matrix(void) {

	int i,j;

	/* Initialize the Matrix arrays */
	/* Non-optimail row major.  Intentional? */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++) {
			mresult[j][i] = 0.0;
			matrixa[j][i] = ( float ) rand() * ( float ) 1.1;
			matrixb[j][i] = ( float ) rand() * ( float ) 1.1;
		}
	}
	return ROWS;
}

float flops_matrix_matrix_multiply(void) {

	int i,j,k;

	/* Matrix-Matrix multiply */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++ ) {
			for ( k = 0; k < COLUMNS; k++ ) {
				mresult[i][j] += matrixa[i][k] * matrixb[k][j];
			}
		}
	}

	return mresult[10][10];
}

float flops_swapped_matrix_matrix_multiply(void) {

	int i, j, k;

	/* Matrix-Matrix multiply */
	/* With inner loops swapped */

	for (i = 0; i < ROWS; i++) {
		for (k = 0; k < COLUMNS; k++) {
			for (j = 0; j < COLUMNS; j++) {
				mresult[i][j] += matrixa[i][k] * matrixb[k][j];
			}
		}
	}
	return mresult[10][10];
}


/* This was originally called "dummy3" in the various sdsc tests */
/* Does a lot of floating point ops near 1.0 */
/* In theory returns a value roughly equal to the number of flops */
double
do_flops3( double x, int iters, int quiet )
{
	int i;
	double w, y, z, a, b, c, d, e, f, g, h;
	double result;
	double one;
	one = 1.0;
	w = x;
	y = x;
	z = x;
	a = x;
	b = x;
	c = x;
	d = x;
	e = x;
	f = x;
	g = x;
	h = x;
	for ( i = 1; i <= iters; i++ ) {
		w = w * 1.000000000001 + one;
		y = y * 1.000000000002 + one;
		z = z * 1.000000000003 + one;
		a = a * 1.000000000004 + one;
		b = b * 1.000000000005 + one;
		c = c * 0.999999999999 + one;
		d = d * 0.999999999998 + one;
		e = e * 0.999999999997 + one;
		f = f * 0.999999999996 + one;
		g = h * 0.999999999995 + one;
		h = h * 1.000000000006 + one;
	}
	result = 2.0 * ( a + b + c + d + e + f + w + x + y + z + g + h );

	if (!quiet) printf("Result = %lf\n", result);

	return result;
}


volatile double a = 0.5, b = 2.2;

double
do_flops( int n, int quiet )
{
        int i;
        double c = 0.11;

        for ( i = 0; i < n; i++ ) {
                c += a * b;
        }

	if (!quiet) printf("%lf\n",c);

	return c;
}

