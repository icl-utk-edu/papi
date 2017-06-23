#include <stdio.h>
#include <stdlib.h>

#include "testcode.h"

#define INDEX 1000

static float matrixa[INDEX][INDEX],
		matrixb[INDEX][INDEX],
		mresult[INDEX][INDEX];


int flops_init_matrix(void) {

	int i,j;

	/* Initialize the Matrix arrays */
	for ( i = 0; i < INDEX; i++ ) {
		for ( j = 0; j < INDEX; j++) {
			mresult[j][i] = 0.0;
			matrixa[j][i] = matrixb[j][i] =
					( float ) rand() * ( float ) 1.1;
		}
	}
	return INDEX;
}

float flops_matrix_matrix_multiply(void) {

	int i,j,k;

	/* Matrix-Matrix multiply */
	for ( i = 0; i < INDEX; i++ )
		for ( j = 0; j < INDEX; j++ )
			for ( k = 0; k < INDEX; k++ )
				mresult[i][j] = mresult[i][j] + matrixa[i][k] * matrixb[k][j];

	return mresult[10][10];
}

