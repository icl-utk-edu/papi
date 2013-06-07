/* file hl_rates.c
 * This simply tries to add the events listed on the command line one at a time
 * then starts and stops the counters and prints the results
*/

/** 
  *	@page papi_command_line 
  * @brief executes PAPI preset or native events from the command line. 
  *
  *	@section Synopsis
  *		papi_command_line < event > < event > ...
  *
  *	@section Description
  *		papi_command_line is a PAPI utility program that adds named events from the 
  *		command line to a PAPI EventSet and does some work with that EventSet. 
  *		This serves as a handy way to see if events can be counted together, 
  *		and if they give reasonable results for known work.
  *
  *	@section Options
  *		This utility has no command line options.
  *
  *	@section Bugs
  *		There are no known bugs in this utility. 
  *		If you find a bug, it should be reported to the 
  *		PAPI Mailing List at <ptools-perfapi@ptools.org>. 
 */


/*
   int PAPI_flips(float *rtime, float *ptime, long long * flpins, float *mflips);
   int PAPI_flops(float *rtime, float *ptime, long long * flpops, float *mflops);
   int PAPI_ipc(float *rtime, float *ptime, long long * ins, float *ipc);
   int PAPI_epc(char *name, float *rtime, float *ptime, long long *ref, long long *core, long long *evt, float *epc);
*/

#include "papi_test.h"

#define ROWS 1000		// Number of rows in each matrix
#define COLUMNS 1000	// Number of columns in each matrix

static float matrix_a[ROWS][COLUMNS], matrix_b[ROWS][COLUMNS],matrix_c[ROWS][COLUMNS];

static void init_mat()
{
	// Multiply the two matrices
	int i, j;
	for (i = 0; i < ROWS; i++) {
		for (j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float) rand() / RAND_MAX;
			matrix_b[i][j] = (float) rand() / RAND_MAX;
		}
	}

}

static void classic_matmul()
{
	// Multiply the two matrices
	int i, j, k;
	for (i = 0; i < ROWS; i++) {
		for (j = 0; j < COLUMNS; j++) {
			float sum = 0.0;
			for (k = 0; k < COLUMNS; k++) {
				sum += 
					matrix_a[i][k] * matrix_b[k][j];
			}
			matrix_c[i][j] = sum;
		}
	}
}

static void swapped_matmul()
{
	// Multiply the two matrices
	int i, j, k;
	for (i = 0; i < ROWS; i++) {
		for (k = 0; k < COLUMNS; k++) {
			for (j = 0; j < COLUMNS; j++) {
				matrix_c[i][j] += 
					matrix_a[i][k] * matrix_b[k][j];
			}
		}
	}
}

int
main( int argc, char **argv )
{
	int retval;
	float rtime, ptime, mflips, mflops, ipc;
	long long flpins, flpops, ins;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	init_mat();
	printf( "PAPI_flips\n");
	retval = PAPI_flips(&rtime, &ptime, &flpins, &mflips);
	printf( "Start\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS %f\n", mflips);
	classic_matmul();
	retval = PAPI_flips(&rtime, &ptime, &flpins, &mflips);
	printf( "Classic\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS %f\n", mflips);
	swapped_matmul();
	retval = PAPI_flips(&rtime, &ptime, &flpins, &mflips);
	printf( "Swapped\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS %f\n", mflips);

	PAPI_stop_counters(NULL, 0); // turn off flips
	printf( "\n----------------------------------\n" );

	printf( "PAPI_flops\n");
	retval = PAPI_flops(&rtime, &ptime, &flpops, &mflops);
	printf( "Start\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Operations: %lld\n", flpops);
	printf( "MFLOPS %f\n", mflops);
	classic_matmul();
	retval = PAPI_flops(&rtime, &ptime, &flpops, &mflops);
	printf( "Classic\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Operations: %lld\n", flpops);
	printf( "MFLOPS %f\n", mflops);
	swapped_matmul();
	retval = PAPI_flops(&rtime, &ptime, &flpops, &mflops);
	printf( "Swapped\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "FP Operations: %lld\n", flpops);
	printf( "MFLOPS %f\n", mflops);

	PAPI_stop_counters(NULL, 0); // turn off flops
	printf( "\n----------------------------------\n" );

	printf( "PAPI_ipc\n");
	retval = PAPI_ipc(&rtime, &ptime, &ins, &ipc);
	printf( "Start\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "Instructions: %lld\n", ins);
	printf( "IPC %f\n", ipc);
	classic_matmul();
	retval = PAPI_ipc(&rtime, &ptime, &ins, &ipc);
	printf( "Classic\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "Instructions: %lld\n", ins);
	printf( "IPC %f\n", ipc);
	swapped_matmul();
	retval = PAPI_ipc(&rtime, &ptime, &ins, &ipc);
	printf( "Swapped\n");
	printf( "real time: %f\n", rtime);
	printf( "process time: %f\n", ptime);
	printf( "Instructions: %lld\n", ins);
	printf( "IPC %f\n", ipc);

	PAPI_stop_counters(NULL, 0); // turn off ipc
	printf( "\n----------------------------------\n" );
	exit( 1 );
}
