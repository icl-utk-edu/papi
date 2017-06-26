/*
 * File:    api.c
 * Author:  Brian Sheely
 *          bsheely@eecs.utk.edu
 *
 * Description: This test is designed to provide unit testing and complete
 *              coverage for all functions which comprise the "Low Level API"
 *              and the "High Level API" as defined in papi.h.
 *
 * Currently only the "high level" API is actually tested.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	const int NUM_COUNTERS = 1;
	int Events[] = { PAPI_TOT_INS };
	long long values[NUM_COUNTERS];
	float rtime, ptime, ipc, mflips, mflops;
	long long ins, flpins, flpops;
	int retval;
	int quiet=0;

	quiet=tests_quiet( argc, argv );

	/* Initialize PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/****** High Level API ******/

	if ( !quiet ) printf( "Testing PAPI_num_components... " );

	/* get the number of components available on the system */
	retval = PAPI_num_components(  );
	if ( !quiet ) printf( "%d\n", retval );

	if ( retval == 0) {
		if ( !quiet ) printf( "No components found, skipping high level tests\n");
		test_skip(__FILE__, __LINE__, "No components found", 0);
	}

	if ( !quiet ) printf( "Testing PAPI_num_counters... " );

	/* get the number of hardware counters available on the system  */
	retval = PAPI_num_counters(  );
	if ( retval != PAPI_get_cmp_opt( PAPI_MAX_HWCTRS, NULL, 0 ) ) {
		test_fail( __FILE__, __LINE__, "PAPI_num_counters", retval );
	}
	else if ( !quiet ) printf( "%d\n", retval );


	/* Test PAPI_start_counters() */
	if ( !quiet ) printf( "Testing PAPI_start_counters... " );
	// pass invalid 1st argument
	retval = PAPI_start_counters( NULL, NUM_COUNTERS );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_start_counters( Events, 0 );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}
	/* Try PAPI_TOT_INS */
	retval = PAPI_start_counters( Events, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("\nCould not start PAPI_TOT_INS\n");
		test_skip( __FILE__, __LINE__, "PAPI_TOT_INS not available.", retval );
	}
	else {
		if ( !quiet ) printf( "started PAPI_TOT_INS\n" );
	}

	/* Test PAPI_stop_counters() */
	if ( !quiet ) printf( "Testing PAPI_stop_counters... " );
	// pass invalid 1st argument
	retval = PAPI_stop_counters( NULL, NUM_COUNTERS );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_stop_counters( values, 0 );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}
	// stop counters and return current counts
	retval = PAPI_stop_counters( values, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}
	else if ( !quiet ) printf( "stopped counting PAPI_TOT_INS\n" );
	//NOTE: There are currently no checks on whether or not counter values are correct


	// start counting hardware events again
	retval = PAPI_start_counters( Events, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start_counters", retval );
	}

	/* Test PAPI_read_counters() */
	if ( !quiet ) printf( "Testing PAPI_read_counters... " );
	// pass invalid 1st argument
	retval = PAPI_read_counters( NULL, NUM_COUNTERS );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_read_counters( values, 0 );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}
	// copy current counts to array and reset counters
	retval = PAPI_read_counters( values, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_read_counters", retval );
	}
	else {
		if ( !quiet ) printf( "read PAPI_TOT_INS counts and reset counter\n" );
	}
	//NOTE: There are currently no checks on whether or not counter values are correct 


	/* Test PAPI_accum_counters() */
	if ( !quiet ) printf( "Testing PAPI_accum_counters... " );
	// pass invalid 1st argument
	retval = PAPI_accum_counters( NULL, NUM_COUNTERS );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_accum_counters", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_accum_counters( values, 0 );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_accum_counters", retval );
	}
	// add current counts to array and reset counters
	retval = PAPI_accum_counters( values, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_accum_counters", retval );
	}
	else {
		if ( !quiet ) printf( "added PAPI_TOT_INS counts and reset counter\n" );
	}
	//NOTE: There are currently no checks on whether or not counter values are correct 


	// stop counting hardware events
	retval = PAPI_stop_counters( values, NUM_COUNTERS );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop_counters", retval );
	}


	/* Test PAPI_ipc() */
	if ( !quiet ) printf( "Testing PAPI_ipc... " );
	// pass invalid 1st argument
	retval = PAPI_ipc( NULL, &ptime, &ins, &ipc );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_ipc( &rtime, NULL, &ins, &ipc );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	}
	// pass invalid 3rd argument
	retval = PAPI_ipc( &rtime, &ptime, NULL, &ipc );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	}
	// pass invalid 4th argument
	retval = PAPI_ipc( &rtime, &ptime, &ins, NULL );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	}
	// get instructions per cycle, real and processor time
	retval = PAPI_ipc( &rtime, &ptime, &ins, &ipc );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_ipc", retval );
	}
	else {
		if ( !quiet ) printf( "got instructions per cycle, real and processor time\n" );
	}
	//NOTE: There are currently no checks on whether or not returned values are correct 


	//NOTE: PAPI_flips and PAPI_flops fail if any other low-level calls have been made!
	PAPI_shutdown(  );
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Test PAPI_flips() */
	if ( !quiet ) printf( "Testing PAPI_flips... " );
	// pass invalid 1st argument
	retval = PAPI_flips( NULL, &ptime, &flpins, &mflips );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_flips( &rtime, NULL, &flpins, &mflips );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
	}
	// pass invalid 3rd argument
	retval = PAPI_flips( &rtime, &ptime, NULL, &mflips );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
	}
	// pass invalid 4th argument
	retval = PAPI_flips( &rtime, &ptime, &flpins, NULL );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
	}
	// get Mflips/s, real and processor time
	retval = PAPI_flips( &rtime, &ptime, &flpins, &mflips );
	if ( retval == PAPI_ENOEVNT ) {
		test_warn( __FILE__, __LINE__, "PAPI_flips", retval);
	}
	else if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_flips", retval );
	}
	else {
		if ( !quiet ) printf( "got Mflips/s, real and processor time\n" );
	}
	//NOTE: There are currently no checks on whether or not returned values are correct 


	PAPI_shutdown(  );
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Test PAPI_flops() */
	if ( !quiet ) printf( "Testing PAPI_flops... " );
	// pass invalid 1st argument
	retval = PAPI_flops( NULL, &ptime, &flpops, &mflops );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
	}
	// pass invalid 2nd argument
	retval = PAPI_flops( &rtime, NULL, &flpops, &mflops );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
	}
	// pass invalid 3rd argument
	retval = PAPI_flops( &rtime, &ptime, NULL, &mflops );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
	}
	// pass invalid 4th argument
	retval = PAPI_flops( &rtime, &ptime, &flpops, NULL );
	if ( retval != PAPI_EINVAL ) {
		test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
	}
	// get Mflops/s, real and processor time
	retval = PAPI_flops( &rtime, &ptime, &flpops, &mflops );
	if ( retval == PAPI_ENOEVNT ) {
		test_warn( __FILE__, __LINE__, "PAPI_flops", retval);
	}
	else if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_flops", retval );
	}
	else if ( !quiet ) {
		printf( "got Mflops/s, real and processor time\n" );
	}
	//NOTE: There are currently no checks on whether or not returned values are correct 


        /***************************/
	/****** Low Level API ******/
	/***************************/
/*
   int   PAPI_accum(int EventSet, long long * values); // accumulate and reset hardware events from an event set
   int   PAPI_add_event(int EventSet, int Event); // add single PAPI preset or native hardware event to an event set
   int   PAPI_add_events(int EventSet, int *Events, int number); // add array of PAPI preset or native hardware events to an event set
   int   PAPI_assign_eventset_component(int EventSet, int cidx); // assign a component index to an existing but empty eventset
   int   PAPI_attach(int EventSet, unsigned long tid); // attach specified event set to a specific process or thread id
   int   PAPI_cleanup_eventset(int EventSet); // remove all PAPI events from an event set
   int   PAPI_create_eventset(int *EventSet); // create a new empty PAPI event set
   int   PAPI_detach(int EventSet); // detach specified event set from a previously specified process or thread id
   int   PAPI_destroy_eventset(int *EventSet); // deallocates memory associated with an empty PAPI event set
   int   PAPI_enum_event(int *EventCode, int modifier); // return the event code for the next available preset or natvie event
   int   PAPI_event_code_to_name(int EventCode, char *out); // translate an integer PAPI event code into an ASCII PAPI preset or native name
   int   PAPI_event_name_to_code(char *in, int *out); // translate an ASCII PAPI preset or native name into an integer PAPI event code
   int  PAPI_get_dmem_info(PAPI_dmem_info_t *dest); // get dynamic memory usage information
   int   PAPI_get_event_info(int EventCode, PAPI_event_info_t * info); // get the name and descriptions for a given preset or native event code
   const PAPI_exe_info_t *PAPI_get_executable_info(void); // get the executable's address space information
   const PAPI_hw_info_t *PAPI_get_hardware_info(void); // get information about the system hardware
   const PAPI_component_info_t *PAPI_get_component_info(int cidx); // get information about the component features
   int   PAPI_get_multiplex(int EventSet); // get the multiplexing status of specified event set
   int   PAPI_get_opt(int option, PAPI_option_t * ptr); // query the option settings of the PAPI library or a specific event set
   int   PAPI_get_cmp_opt(int option, PAPI_option_t * ptr,int cidx); // query the component specific option settings of a specific event set
   long long PAPI_get_real_cyc(void); // return the total number of cycles since some arbitrary starting point
   long long PAPI_get_real_nsec(void); // return the total number of nanoseconds since some arbitrary starting point
   long long PAPI_get_real_usec(void); // return the total number of microseconds since some arbitrary starting point
   const PAPI_shlib_info_t *PAPI_get_shared_lib_info(void); // get information about the shared libraries used by the process
   int   PAPI_get_thr_specific(int tag, void **ptr); // return a pointer to a thread specific stored data structure
   int   PAPI_get_overflow_event_index(int Eventset, long long overflow_vector, int *array, int *number); // # decomposes an overflow_vector into an event index array
   long long PAPI_get_virt_cyc(void); // return the process cycles since some arbitrary starting point
   long long PAPI_get_virt_nsec(void); // return the process nanoseconds since some arbitrary starting point
   long long PAPI_get_virt_usec(void); // return the process microseconds since some arbitrary starting point
   int   PAPI_is_initialized(void); // return the initialized state of the PAPI library
   int   PAPI_library_init(int version); // initialize the PAPI library
   int   PAPI_list_events(int EventSet, int *Events, int *number); // list the events that are members of an event set
   int   PAPI_list_threads(unsigned long *tids, int *number); // list the thread ids currently known to PAPI
   int   PAPI_lock(int); // lock one of two PAPI internal user mutex variables
   int   PAPI_multiplex_init(void); // initialize multiplex support in the PAPI library
   int   PAPI_num_hwctrs(void); // return the number of hardware counters for the cpu
   int   PAPI_num_cmp_hwctrs(int cidx); // return the number of hardware counters for a specified component
   int   PAPI_num_hwctrs(void); // for backward compatibility
   int   PAPI_num_events(int EventSet); // return the number of events in an event set
   int   PAPI_overflow(int EventSet, int EventCode, int threshold,
                     int flags, PAPI_overflow_handler_t handler); // set up an event set to begin registering overflows
   int   PAPI_perror( char *msg); // convert PAPI error codes to strings
   int   PAPI_profil(void *buf, unsigned bufsiz, caddr_t offset, 
					 unsigned scale, int EventSet, int EventCode, 
					 int threshold, int flags); // generate PC histogram data where hardware counter overflow occurs
   int   PAPI_query_event(int EventCode); // query if a PAPI event exists
   int   PAPI_read(int EventSet, long long * values); // read hardware events from an event set with no reset
   int   PAPI_read_ts(int EventSet, long long * values, long long *cyc);
   int   PAPI_register_thread(void); // inform PAPI of the existence of a new thread
   int   PAPI_remove_event(int EventSet, int EventCode); // remove a hardware event from a PAPI event set
   int   PAPI_remove_events(int EventSet, int *Events, int number); // remove an array of hardware events from a PAPI event set
   int   PAPI_reset(int EventSet); // reset the hardware event counts in an event set
   int   PAPI_set_debug(int level); // set the current debug level for PAPI
   int   PAPI_set_cmp_domain(int domain, int cidx); // set the component specific default execution domain for new event sets
   int   PAPI_set_domain(int domain); // set the default execution domain for new event sets 
   int   PAPI_set_cmp_granularity(int granularity, int cidx); // set the component specific default granularity for new event sets
   int   PAPI_set_granularity(int granularity); //set the default granularity for new event sets
   int   PAPI_set_multiplex(int EventSet); // convert a standard event set to a multiplexed event set
   int   PAPI_set_opt(int option, PAPI_option_t * ptr); // change the option settings of the PAPI library or a specific event set
   int   PAPI_set_thr_specific(int tag, void *ptr); // save a pointer as a thread specific stored data structure
   void  PAPI_shutdown(void); // finish using PAPI and free all related resources
   int   PAPI_sprofil(PAPI_sprofil_t * prof, int profcnt, int EventSet, int EventCode, int threshold, int flags); // generate hardware counter profiles from multiple code regions
   int   PAPI_start(int EventSet); // start counting hardware events in an event set
   int   PAPI_state(int EventSet, int *status); // return the counting state of an event set
   int   PAPI_stop(int EventSet, long long * values); // stop counting hardware events in an event set and return current events
   char *PAPI_strerror(int); // return a pointer to the error message corresponding to a specified error code
   unsigned long PAPI_thread_id(void); // get the thread identifier of the current thread
   int   PAPI_thread_init(unsigned long (*id_fn) (void)); // initialize thread support in the PAPI library
   int   PAPI_unlock(int); // unlock one of two PAPI internal user mutex variables
   int   PAPI_unregister_thread(void); // inform PAPI that a previously registered thread is disappearing
   int   PAPI_write(int EventSet, long long * values); // write counter values into counters
*/

	test_pass( __FILE__ );

	return 0;
}
