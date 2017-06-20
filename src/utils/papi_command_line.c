/* file command_line.c
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
  * <ul>
  *		<li>-u          Display output values as unsigned integers
  *		<li>-x          Display output values as hexadecimal
  *		<li>-h          Display help information about this utility.
  *	</ul>
  *
  *	@section Bugs
  *		There are no known bugs in this utility. 
  *		If you find a bug, it should be reported to the 
  *		PAPI Mailing List at <ptools-perfapi@ptools.org>. 
 */

#include "papi_test.h"

static void
print_help( char **argv )
{
	printf( "Usage: %s [options] [EVENTNAMEs]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "General command options:\n" );
	printf( "\t-u          Display output values as unsigned integers\n" );
	printf( "\t-x          Display output values as hexadecimal\n" );
	printf( "\t-h          Print this help message\n" );
	printf( "\tEVENTNAMEs  Specify one or more preset or native events\n" );
	printf( "\n" );
	printf( "This utility performs work while measuring the specified events.\n" );
	printf( "It can be useful for sanity checks on given events and sets of events.\n" );
}


int
main( int argc, char **argv )
{
	int retval;
	int num_events;
	long long *values;
	char *success;
	PAPI_event_info_t info;
	int EventSet = PAPI_NULL;
	int i, j, event, data_type = PAPI_DATATYPE_INT64;
	int u_format = 0;
	int hex_format = 0;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	/* Automatically pass if no events, for run_tests.sh */
	if ((( TESTS_QUIET ) && ( argc == 2)) || ( argc == 1 )) {
		test_pass( __FILE__, NULL, 0 );
	}

	values =
		( long long * ) malloc( sizeof ( long long ) * ( size_t ) argc );
	success = ( char * ) malloc( ( size_t ) argc );

	if ( success == NULL || values == NULL )
		test_fail_exit( __FILE__, __LINE__, "malloc", PAPI_ESYS );

	for ( num_events = 0, i = 1; i < argc; i++ ) {
		if ( !strcmp( argv[i], "-h" ) ) {
			print_help( argv );
			exit( 1 );
		} else if ( !strcmp( argv[i], "-u" ) ) {
			u_format = 1;
		} else if ( !strcmp( argv[i], "-x" ) ) {
			hex_format = 1;
		} else {
			if ( ( retval = PAPI_add_named_event( EventSet, argv[i] ) ) != PAPI_OK ) {
				printf( "Failed adding: %s\nbecause: %s\n", argv[i], 
					PAPI_strerror(retval));
			} else {
				success[num_events++] = i;
				printf( "Successfully added: %s\n", argv[i] );
			}
		}
	}

	/* Automatically pass if no events, for run_tests.sh */
	if ( num_events == 0 ) {
		test_pass( __FILE__, NULL, 0 );
	}


	printf( "\n" );

	do_flops( 1 );
	do_flush(  );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
	   test_fail_exit( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( NUM_FLOPS );
	do_misses( 1, L1_MISS_BUFFER_SIZE_INTS );

	if ( ( retval = PAPI_stop( EventSet, values ) ) != PAPI_OK )
		test_fail_exit( __FILE__, __LINE__, "PAPI_stop", retval );

	for ( j = 0; j < num_events; j++ ) {
		i = success[j];
		if (! (u_format || hex_format) ) {
			retval = PAPI_event_name_to_code( argv[i], &event );
			if (retval == PAPI_OK) {
				retval = PAPI_get_event_info(event, &info);
				if (retval == PAPI_OK) data_type = info.data_type;
				else data_type = PAPI_DATATYPE_INT64;
			}
			switch (data_type) {
			  case PAPI_DATATYPE_UINT64:
				printf( "%s : \t%llu(u)", argv[i], (unsigned long long)values[j] );
				break;
			  case PAPI_DATATYPE_FP64:
				printf( "%s : \t%0.3f", argv[i], *((double *)(&values[j])) );
				break;
			  case PAPI_DATATYPE_BIT64:
				printf( "%s : \t%#llX", argv[i], values[j] );
				break;
			  case PAPI_DATATYPE_INT64:
			  default:
				printf( "%s : \t%lld", argv[i], values[j] );
				break;
			}
			if (retval == PAPI_OK)  printf( " %s", info.units );
			printf( "\n" );
		}
		if (u_format) printf( "%s : \t%llu(u)\n", argv[i], (unsigned long long)values[j] );
		if (hex_format) printf( "%s : \t%#llX\n", argv[i], values[j] );
	}

	printf( "\n----------------------------------\n" );
	printf
		( "Verification: Checks for valid event name.\n This utility lets you add events from the command line interface to see if they work.\n" );
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
