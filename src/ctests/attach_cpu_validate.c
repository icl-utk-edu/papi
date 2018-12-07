/*
 * This test attempts to attach to each CPU
 * Then it runs some code on one CPU
 * Then it reads the results, they should be different.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#define MAX_CPUS	16

int
main( int argc, char **argv )
{
	int i;
	int retval;
	int num_cpus = 8;
	int EventSet[MAX_CPUS];
	const PAPI_hw_info_t *hwinfo;

	long long values[MAX_CPUS];
	char event_name[PAPI_MAX_STR_LEN] = "PAPI_TOT_INS";
	PAPI_option_t opts;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

        hwinfo = PAPI_get_hardware_info(  );
	if ( hwinfo==NULL) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", retval );
	}

	num_cpus=hwinfo->totalcpus;

	if ( num_cpus < 2 ) {
		if (!quiet) printf("Need at least 1 CPU\n");
		test_skip( __FILE__, __LINE__, "num_cpus", 0 );
	}

	for(i=0;i<num_cpus;i++) {

		EventSet[i]=PAPI_NULL;

		retval = PAPI_create_eventset(&EventSet[i]);
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
		}

		/* Force event set to be associated with component 0 */
		/* (perf_events component provides all core events)  */
		retval = PAPI_assign_eventset_component( EventSet[i], 0 );
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component", retval );
		}

		/* Attach this event set to cpu i */
		opts.cpu.eventset = EventSet[i];
		opts.cpu.cpu_num = i;

		retval = PAPI_set_opt( PAPI_CPU_ATTACH, &opts );
		if ( retval != PAPI_OK ) {
			if (!quiet) printf("Can't PAPI_CPU_ATTACH: %s\n",
					PAPI_strerror(retval));
			test_skip( __FILE__, __LINE__, "PAPI_set_opt", retval );
		}

		retval = PAPI_add_named_event(EventSet[i], event_name);
		if ( retval != PAPI_OK ) {
			if (!quiet) printf("Trouble adding event %s\n",event_name);
			test_skip( __FILE__, __LINE__, "PAPI_add_named_event", retval );
		}
	}


	for(i=0;i<num_cpus;i++) {
		retval = PAPI_start( EventSet[i] );
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );
		}
	}

	// do some work
	do_flops(NUM_FLOPS);

	for(i=0;i<num_cpus;i++) {
		retval = PAPI_stop( EventSet[i], &values[i] );
		if ( retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
		}
	}

	for(i=0;i<num_cpus;i++) {
		if (!quiet) {
			printf ("Event: %s: %8lld on Cpu: %d\n",
				event_name, values[i], i);
		}
	}

	PAPI_shutdown( );

	test_pass( __FILE__ );

	return 0;

}
