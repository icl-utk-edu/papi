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

