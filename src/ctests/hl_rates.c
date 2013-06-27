/* file hl_rates.c
 * This test exercises the four PAPI High Level rate calls:
 *    PAPI_flops, PAPI_flips, PAPI_ipc, and PAPI_epc
 * flops and flips report cumulative real and process time since the first call,
 * and either floating point operations or instructions since the first call.
 * Also reported is incremental flop or flip rate since the last call.
 *
 * PAPI_ipc reports the same cumulative information, substituting total instructions
 * for flops or flips, and also reports instructions per (process) cycle as
 * a measure of execution efficiency.
 *
 * PAPI_epc is new in PAPI 5.2. It reports the same information as PAPI_IPC, but 
 * for an arbitrary event instead of total cycles. It also reports incremental 
 * core and (where available) reference cycles to allow the computation of 
 * effective clock rates in the presence of clock scaling like speed step or turbo-boost.
 * 
 * This test computes a 1000 x 1000 matrix multiply for orders of indexing for
 * each of the four rate calls. It also accepts a command line parameter for the
 * event to be measured for PAPI_epc. If not provided, PAPI_TOT_INS is measured.
 */

#include "papi_test.h"

#define ROWS 1000		// Number of rows in each matrix
#define COLUMNS 1000	// Number of columns in each matrix

static float matrix_a[ROWS][COLUMNS], matrix_b[ROWS][COLUMNS],matrix_c[ROWS][COLUMNS];

static void init_mat()
{
	// Initialize the two matrices
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
	int retval, event = 0;
	float rtime, ptime, mflips, mflops, ipc, epc;
	long long flpins, flpops, ins, ref, core, evt;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	init_mat();

	printf( "\n----------------------------------\n" );
	printf( "PAPI_flips\n");
	if ( PAPI_flips(&rtime, &ptime, &flpins, &mflips)  != PAPI_OK )
		PAPI_perror( "PAPI_flips" );
	printf( "\nStart\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS           %f\n", mflips);
	classic_matmul();
	if ( PAPI_flips(&rtime, &ptime, &flpins, &mflips)  != PAPI_OK )
		PAPI_perror( "PAPI_flips" );
	printf( "\nClassic\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS           %f\n", mflips);
	swapped_matmul();
	if ( PAPI_flips(&rtime, &ptime, &flpins, &mflips)  != PAPI_OK )
		PAPI_perror( "PAPI_flips" );
	printf( "\nSwapped\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Instructions: %lld\n", flpins);
	printf( "MFLIPS           %f\n", mflips);

	// turn off flips
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK )
		PAPI_perror( "PAPI_stop_counters" );
	printf( "\n----------------------------------\n" );

	printf( "PAPI_flops\n");
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK )
		PAPI_perror( "PAPI_flops" );
	printf( "\nStart\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Operations:   %lld\n", flpops);
	printf( "MFLOPS           %f\n", mflops);
	classic_matmul();
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK )
		PAPI_perror( "PAPI_flops" );
	printf( "\nClassic\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Operations:   %lld\n", flpops);
	printf( "MFLOPS           %f\n", mflops);
	swapped_matmul();
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK )
		PAPI_perror( "PAPI_flops" );
	printf( "\nSwapped\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "FP Operations:   %lld\n", flpops);
	printf( "MFLOPS           %f\n", mflops);

	// turn off flops
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK )
		PAPI_perror( "PAPI_stop_counters" );
	printf( "\n----------------------------------\n" );

	printf( "PAPI_ipc\n");
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );
	printf( "\nStart\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Instructions:    %lld\n", ins);
	printf( "IPC              %f\n", ipc);
	classic_matmul();
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );
	printf( "\nClassic\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Instructions:    %lld\n", ins);
	printf( "IPC              %f\n", ipc);
	swapped_matmul();
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );
	printf( "\nSwapped\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Instructions:    %lld\n", ins);
	printf( "IPC              %f\n", ipc);

	// turn off ipc
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK )
		PAPI_perror( "PAPI_stop_counters" );
	printf( "\n----------------------------------\n" );

	printf( "PAPI_epc\n");
	
	if ( argc >= 2) {
		retval = PAPI_event_name_to_code( argv[1], &event );
		if (retval != PAPI_OK) {
		 	PAPI_perror("PAPI_event_name_to_code");
		 	printf("Can't find %s; Using PAPI_TOT_INS\n", argv[1]);
		 	event = 0;
		} else {
		 	printf("Using event %s\n", argv[1]);
		}
	}

	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK )
		PAPI_perror( "PAPI_epc" );
	printf( "\nStart\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Ref Cycles:      %lld\n", ref);
	printf( "Core Cycles:     %lld\n", core);
	printf( "Events:          %lld\n", evt);
	printf( "EPC:             %f\n", epc);
	classic_matmul();
	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK )
		PAPI_perror( "PAPI_epc" );
	printf( "\nClassic\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Ref Cycles:      %lld\n", ref);
	printf( "Core Cycles:     %lld\n", core);
	printf( "Events:          %lld\n", evt);
	printf( "EPC:             %f\n", epc);
	swapped_matmul();
	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK )
		PAPI_perror( "PAPI_epc" );
	printf( "\nSwapped\n");
	printf( "real time:       %f\n", rtime);
	printf( "process time:    %f\n", ptime);
	printf( "Ref Cycles:      %lld\n", ref);
	printf( "Core Cycles:     %lld\n", core);
	printf( "Events:          %lld\n", evt);
	printf( "EPC:             %f\n", epc);

	// turn off epc
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK )
		PAPI_perror( "PAPI_stop_counters" );
	printf( "\n----------------------------------\n" );
	exit( 1 );
}
