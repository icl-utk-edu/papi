/* file hl_rates.c
 * This test exercises the four PAPI High Level rate calls:
 *    PAPI_flops, PAPI_flips, PAPI_ipc, and PAPI_epc
 * flops and flips report cumulative real and process time since the first call,
 * and either floating point operations or instructions since the first call.
 * Also reported is incremental flop or flip rate since the last call.
 *
 * PAPI_ipc reports the same cumulative information, substituting
 * total instructions for flops or flips, and also reports
 * instructions per (process) cycle as a measure of execution efficiency.
 *
 * PAPI_epc is new in PAPI 5.2. It reports the same information as PAPI_IPC,
 * but for an arbitrary event instead of total cycles. It also reports
 * incremental core and (where available) reference cycles to allow the
 * computation of effective clock rates in the presence of clock scaling
 * like speed step or turbo-boost.
 *
 * This test computes a 1000 x 1000 matrix multiply for orders of indexing for
 * each of the four rate calls. It also accepts a command line parameter
 * for the event to be measured for PAPI_epc. If not provided, PAPI_TOT_INS
 * is measured.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

int
main( int argc, char **argv )
{
	int retval, event = 0;
	float rtime, ptime, mflips, mflops, ipc, epc;
	long long flpins, flpops, ins, ref, core, evt;

	double mflips_classic,mflips_swapped;
	double mflops_classic,mflops_swapped;
	double ipc_classic,ipc_swapped;
	double epc_classic,epc_swapped;

	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Initialize the test matrix */
	flops_float_init_matrix();

	/************************/
	/* FLIPS		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_flips\n");
	}

	/* Run flips at start */
	retval=PAPI_flips(&rtime, &ptime, &flpins, &mflips);
	if (retval!=PAPI_OK) {
		if (!quiet) PAPI_perror( "PAPI_flips" );
		test_skip(__FILE__,__LINE__,"Could not add event",0);
	}

	if (!quiet) {
		printf( "\nStart\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Instructions: %lld\n", flpins);
		printf( "MFLIPS           %f\n", mflips);
	}

	/* Be sure we are all zero at beginning */
	if ((rtime!=0) || (ptime!=0) || (flpins!=0) || (mflips!=0)) {
		test_fail(__FILE__,__LINE__,"Not initialized to zero",0);
	}

	// Flips classic
	flops_float_matrix_matrix_multiply();
	if ( PAPI_flips(&rtime, &ptime, &flpins, &mflips)  != PAPI_OK )
		PAPI_perror( "PAPI_flips" );

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Instructions: %lld\n", flpins);
		printf( "MFLIPS           %f\n", mflips);
	}
	mflips_classic=mflips;

	// Flips swapped
	flops_float_swapped_matrix_matrix_multiply();
	if ( PAPI_flips(&rtime, &ptime, &flpins, &mflips)  != PAPI_OK )
		PAPI_perror( "PAPI_flips" );

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Instructions: %lld\n", flpins);
		printf( "MFLIPS           %f\n", mflips);
	}
	mflips_swapped=mflips;

	// turn off flips
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK ) {
		PAPI_perror( "PAPI_stop_counters" );
	}


	/************************/
	/* FLOPS		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_flops\n");
	}

	// Start flops
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK ) {
		PAPI_perror( "PAPI_flops" );
	}

	if (!quiet) {
		printf( "\nStart\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Operations:   %lld\n", flpops);
		printf( "MFLOPS           %f\n", mflops);
	}

	/* Be sure we are all zero at beginning */
	if ((rtime!=0) || (ptime!=0) || (flpops!=0) || (mflops!=0)) {
		test_fail(__FILE__,__LINE__,"Not initialized to zero",0);
	}

	// Classic flops
	flops_float_matrix_matrix_multiply();
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK )
		PAPI_perror( "PAPI_flops" );

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Operations:   %lld\n", flpops);
		printf( "MFLOPS           %f\n", mflops);
	}
	mflops_classic=mflops;

	// Swapped flops
	flops_float_swapped_matrix_matrix_multiply();
	if ( PAPI_flops(&rtime, &ptime, &flpops, &mflops)  != PAPI_OK )
		PAPI_perror( "PAPI_flops" );

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Operations:   %lld\n", flpops);
		printf( "MFLOPS           %f\n", mflops);
	}
	mflops_swapped=mflops;

	// turn off flops
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK ) {
		PAPI_perror( "PAPI_stop_counters" );
	}


	/************************/
	/* IPC  		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_ipc\n");
	}

	// Start ipc
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );

	if (!quiet) {
		printf( "\nStart\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Instructions:    %lld\n", ins);
		printf( "IPC              %f\n", ipc);
	}

	/* Be sure we are all zero at beginning */
	if ((rtime!=0) || (ptime!=0) || (ins!=0) || (ipc!=0)) {
		test_fail(__FILE__,__LINE__,"Not initialized to zero",0);
	}

	// Classic ipc
	flops_float_matrix_matrix_multiply();
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Instructions:    %lld\n", ins);
		printf( "IPC              %f\n", ipc);
	}
	ipc_classic=ipc;

	// Swapped ipc
	flops_float_swapped_matrix_matrix_multiply();
	if ( PAPI_ipc(&rtime, &ptime, &ins, &ipc)  != PAPI_OK )
		PAPI_perror( "PAPI_ipc" );

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Instructions:    %lld\n", ins);
		printf( "IPC              %f\n", ipc);
	}
	ipc_swapped=ipc;

	// turn off ipc
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK ) {
		PAPI_perror( "PAPI_stop_counters" );
	}


	/************************/
	/* EPC  		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_epc\n");
	}

	/* This unfortunately conflicts a bit with the TESTS_QUIET */
	/* command line paramater nonsense.			   */

	if ( argc >= 2) {
		retval = PAPI_event_name_to_code( argv[1], &event );
		if (retval != PAPI_OK) {
		 	if (!quiet) printf("Can't find %s; Using PAPI_TOT_INS\n", argv[1]);
		 	event = PAPI_TOT_INS;
		} else {
		 	if (!quiet) printf("Using event %s\n", argv[1]);
		}
	}

	// Start epc
	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK )
		PAPI_perror( "PAPI_epc" );

	if (!quiet) {
		printf( "\nStart\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Ref Cycles:      %lld\n", ref);
		printf( "Core Cycles:     %lld\n", core);
		printf( "Events:          %lld\n", evt);
		printf( "EPC:             %f\n", epc);
	}

	/* Be sure we are all zero at beginning */
	if ((rtime!=0) || (ptime!=0) || (ref!=0) || (core!=0)
			|| (evt!=0) || (epc!=0)) {
		test_fail(__FILE__,__LINE__,"Not initialized to zero",0);
	}

	// Classic epc
	flops_float_matrix_matrix_multiply();
	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK )
		PAPI_perror( "PAPI_epc" );

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Ref Cycles:      %lld\n", ref);
		printf( "Core Cycles:     %lld\n", core);
		printf( "Events:          %lld\n", evt);
		printf( "EPC:             %f\n", epc);
	}
	epc_classic=epc;

	// Swapped epc
	flops_float_swapped_matrix_matrix_multiply();
	if ( PAPI_epc(event, &rtime, &ptime, &ref, &core, &evt, &epc)  != PAPI_OK ) {
		PAPI_perror( "PAPI_epc" );
	}

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "Ref Cycles:      %lld\n", ref);
		printf( "Core Cycles:     %lld\n", core);
		printf( "Events:          %lld\n", evt);
		printf( "EPC:             %f\n", epc);
	}
	epc_swapped=epc;

	// turn off epc
	if ( PAPI_stop_counters(NULL, 0)  != PAPI_OK ) {
		PAPI_perror( "PAPI_stop_counters" );
	}

	if (!quiet) {
		printf( "\n----------------------------------\n" );
	}

	/* Validate */
	if (mflips_swapped<mflips_classic) {
		test_fail(__FILE__,__LINE__,"FLIPS should be better when swapped",0);
	}
	if (mflops_swapped<mflops_classic) {
		test_fail(__FILE__,__LINE__,"FLOPS should be better when swapped",0);
	}
	if (ipc_swapped<ipc_classic) {
		test_fail(__FILE__,__LINE__,"IPC should be better when swapped",0);
	}
	if (epc_swapped<epc_classic) {
		test_fail(__FILE__,__LINE__,"EPC should be better when swapped",0);
	}

	test_pass( __FILE__ );

	return 0;
}
