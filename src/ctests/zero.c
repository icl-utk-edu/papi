/* This file performs the following test: start, stop and timer functionality

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include "papi_test.h"

#define MAX_CYCLE_ERROR 30

int
main( int argc, char **argv )
{
	int retval, num_tests = 1, tmp;
	int EventSet1 = PAPI_NULL;
	int PAPI_event, mask1;
	int num_events;
	long long **values;
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];
	double cycles_error;

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );	

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* add PAPI_TOT_CYC and one of the events in 
           PAPI_FP_INS, PAPI_FP_OPS or PAPI_TOT_INS, 
           depending on the availability of the event 
           on the platform                            */
	EventSet1 = add_two_events( &num_events, &PAPI_event, &mask1 );

	retval = PAPI_event_code_to_name( PAPI_event, event_name );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
	}
	sprintf( add_event_str, "PAPI_add_event[%s]", event_name );

	values = allocate_test_space( num_tests, num_events );

	/* warm up the processor to pull it out of idle state */
	do_flops( NUM_FLOPS*10 );

	/* Gather before stats */
	elapsed_us = PAPI_get_real_usec(  );
	elapsed_cyc = PAPI_get_real_cyc(  );
	elapsed_virt_us = PAPI_get_virt_usec(  );
	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	/* Start PAPI */
	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our work code */
	do_flops( NUM_FLOPS );

	/* Stop PAPI */
	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/* Calculate total values */
	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;
	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;
	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	remove_test_events( &EventSet1, mask1 );

	if ( !TESTS_QUIET ) {
	   printf( "Test case 0: start, stop.\n" );
	   printf( "-----------------------------------------------\n" );
	   tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
	   printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
	   tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
	   printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
	   printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
	   printf( "-------------------------------------------------------------------------\n" );

	   printf( "Test type    : \t           1\n" );

	   sprintf( add_event_str, "%-12s : \t", event_name );

	   /* cycles is first, other event second */
	   printf( TAB1, add_event_str, values[0][1] );
	   
	   /* If cycles is there, it's always the first event */
	   if ( mask1 & MASK_TOT_CYC ) {
	      printf( TAB1, "PAPI_TOT_CYC : \t", values[0][0] );
	   }
	   printf( TAB1, "Real usec    : \t", elapsed_us );
	   printf( TAB1, "Real cycles  : \t", elapsed_cyc );
	   printf( TAB1, "Virt usec    : \t", elapsed_virt_us );
	   printf( TAB1, "Virt cycles  : \t", elapsed_virt_cyc );

	   printf( "-------------------------------------------------------------------------\n" );

	   printf( "Verification: PAPI_TOT_CYC should be roughly real_cycles\n" );
	   printf( "NOTE: Not true if dynamic frequency scaling is enabled.\n" );
	   printf( "Verification: PAPI_FP_INS should be roughly %d\n", 2*NUM_FLOPS );
	}
	/* Check that TOT_CYC and real_cycles roughly match */
	cycles_error=100.0*((double)values[0][0] - (double)elapsed_cyc)/((double)elapsed_cyc);
	if ((cycles_error > MAX_CYCLE_ERROR) || (cycles_error < -MAX_CYCLE_ERROR)) {
		printf("PAPI_TOT_CYC Error of %.2f%%\n",cycles_error);
		test_fail( __FILE__, __LINE__, "Cycles validation", 0 );
	}
	/* Check that FP_INS is reasonable */
	if (abs(values[0][1] - (2*NUM_FLOPS)) > (2*NUM_FLOPS)) {
		printf("%s Error of %.2f%%\n", event_name, (100.0 * (double)(values[0][1] - (2*NUM_FLOPS)))/(2*NUM_FLOPS));
		test_fail( __FILE__, __LINE__, "FLOPS validation", 0 );
	}
	if (abs(values[0][1] - (2*NUM_FLOPS)) > (NUM_FLOPS/2)) {
		printf("%s Error of %.2f%%\n", event_name, (100.0 * (double)(values[0][1] - (2*NUM_FLOPS)))/(2*NUM_FLOPS));
		test_warn( __FILE__, __LINE__, "FLOPS validation", 0 );
	}
	
	test_pass( __FILE__, values, num_tests );
	
	return 0;
}
