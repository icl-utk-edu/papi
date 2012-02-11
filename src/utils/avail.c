/**
  * file avail.c
  *	@brief papi_avail utility.
  * @page papi_avail
  *	@section Name
  *	papi_avail - provides availability and detail information for PAPI preset events. 
  * 
  *	@section Synopsis
  *	papi_avail [-adht] [-e event] 
  *
  *	@section Description
  *	papi_avail is a PAPI utility program that reports information about the 
  *	current PAPI installation and supported preset events. 
  *	Using the -e option, it will also display information about specific native events. 
  *
  *	@section Options
  * <ul>
  *		<li>-a	 Display only the available PAPI preset events.
  *		<li>-d	Display PAPI preset event information in a more detailed format.
  *		<li>-h	Display help information about this utility.
  *		<li>-t	Display the PAPI preset event information in a tabular format. This is the default.
  *		<li>-e < event >	Display detailed event information for the named event. 
  *			This event can be either a preset or a native event. 
  *	</ul>
  *
  *	@section Bugs
  *	There are no known bugs in this utility. 
  *	If you find a bug, it should be reported to the PAPI Mailing List at <ptools-perfapi@ptools.org>.
  */

#include "papi_test.h"
extern int TESTS_QUIET;				   /* Declared in test_utils.c */

static char *
is_derived( PAPI_event_info_t * info )
{
	if ( strlen( info->derived ) == 0 )
		return ( "No" );
	else if ( strcmp( info->derived, "NOT_DERIVED" ) == 0 )
		return ( "No" );
	else if ( strcmp( info->derived, "DERIVED_CMPD" ) == 0 )
		return ( "No" );
	else
		return ( "Yes" );
}

static void
print_help( char **argv )
{
	printf( "Usage: %s [options]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "General command options:\n" );
	printf( "\t-a, --avail   Display only available preset events\n" );
	printf
		( "\t-d, --detail  Display detailed information about all preset events\n" );
	printf
		( "\t-e EVENTNAME  Display detail information about specified preset or native event\n" );
	printf( "\t-h, --help    Print this help message\n" );
	printf( "\nEvent filtering options:\n" );
	printf( "\t--br          Display branch related PAPI preset events\n" );
	printf( "\t--cache       Display cache related PAPI preset events\n" );
	printf( "\t--cnd         Display conditional PAPI preset events\n" );
	printf
		( "\t--fp          Display Floating Point related PAPI preset events\n" );
	printf
		( "\t--ins         Display instruction related PAPI preset events\n" );
	printf( "\t--idl         Display Stalled or Idle PAPI preset events\n" );
	printf
		( "\t--l1          Display level 1 cache related PAPI preset events\n" );
	printf
		( "\t--l2          Display level 2 cache related PAPI preset events\n" );
	printf
		( "\t--l3          Display level 3 cache related PAPI preset events\n" );
	printf( "\t--mem         Display memory related PAPI preset events\n" );
	printf( "\t--msc         Display miscellaneous PAPI preset events\n" );
	printf
		( "\t--tlb         Display Translation Lookaside Buffer PAPI preset events\n" );
	printf( "\n" );
	printf
		( "This program provides information about PAPI preset and native events.\n" );
	printf( "PAPI preset event filters can be combined in a logical OR.\n" );
}

static int
parse_unit_masks( PAPI_event_info_t * info )
{
	char *pmask;

	if ( ( pmask = strchr( info->symbol, ':' ) ) == NULL ) {
		return ( 0 );
	}
	memmove( info->symbol, pmask, ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	pmask = strchr( info->long_descr, ':' );
	if ( pmask == NULL )
		info->long_descr[0] = 0;
	else
		memmove( info->long_descr, pmask + sizeof ( char ),
				 ( strlen( pmask ) + 1 ) * sizeof ( char ) );
	return ( 1 );
}

int
main( int argc, char **argv )
{
	int i, j, k;
	int retval;
	unsigned int filter = 0;
	int print_event_info = 0;
	char *name = NULL;
	int print_avail_only = 0;
	int print_tabular = 1;
	PAPI_event_info_t info;
	const PAPI_hw_info_t *hwinfo = NULL;
	int tot_count = 0;
	int avail_count = 0;
	int deriv_count = 0;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	for ( i = 1; i < argc; i++ ) {
		if ( strstr( argv[i], "-e" ) ) {
			print_event_info = 1;
			name = argv[i + 1];
			if ( ( name == NULL ) || ( strlen( name ) == 0 ) ) {
				print_help( argv );
				exit( 1 );
			}
		} else if ( strstr( argv[i], "-a" ) )
			print_avail_only = PAPI_PRESET_ENUM_AVAIL;
		else if ( strstr( argv[i], "-d" ) )
			print_tabular = 0;
		else if ( strstr( argv[i], "-h" ) ) {
			print_help( argv );
			exit( 1 );
		} else if ( strstr( argv[i], "--br" ) )
			filter |= PAPI_PRESET_BIT_BR;
		else if ( strstr( argv[i], "--cache" ) )
			filter |= PAPI_PRESET_BIT_CACH;
		else if ( strstr( argv[i], "--cnd" ) )
			filter |= PAPI_PRESET_BIT_CND;
		else if ( strstr( argv[i], "--fp" ) )
			filter |= PAPI_PRESET_BIT_FP;
		else if ( strstr( argv[i], "--ins" ) )
			filter |= PAPI_PRESET_BIT_INS;
		else if ( strstr( argv[i], "--idl" ) )
			filter |= PAPI_PRESET_BIT_IDL;
		else if ( strstr( argv[i], "--l1" ) )
			filter |= PAPI_PRESET_BIT_L1;
		else if ( strstr( argv[i], "--l2" ) )
			filter |= PAPI_PRESET_BIT_L2;
		else if ( strstr( argv[i], "--l3" ) )
			filter |= PAPI_PRESET_BIT_L3;
		else if ( strstr( argv[i], "--mem" ) )
			filter |= PAPI_PRESET_BIT_BR;
		else if ( strstr( argv[i], "--msc" ) )
			filter |= PAPI_PRESET_BIT_MSC;
		else if ( strstr( argv[i], "--tlb" ) )
			filter |= PAPI_PRESET_BIT_TLB;
	}
	if ( filter == 0 )
		filter = ( unsigned int ) ( -1 );

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( !TESTS_QUIET ) {
		retval = PAPI_set_debug( PAPI_VERB_ECONT );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_debug", retval );
	}

	retval =
		papi_print_header( "Available events and hardware information.\n", 
						   &hwinfo );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	if ( print_event_info ) {
		if ( PAPI_event_name_to_code( name, &i ) == PAPI_OK ) {
			if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
				PAPI_event_info_t n_info;
				if ( i & PAPI_PRESET_MASK ) {
					printf( "%-30s%s\n%-30s0x%-10x\n%-30s%d\n",
							"Event name:", info.symbol, "Event Code:",
							info.event_code, "Number of Native Events:",
							info.count );
					printf( "%-29s|%s|\n%-29s|%s|\n%-29s|%s|\n",
							"Short Description:", info.short_descr,
							"Long Description:", info.long_descr,
							"Developer's Notes:", info.note );
					printf( "%-29s|%s|\n%-29s|%s|\n", "Derived Type:",
							info.derived, "Postfix Processing String:",
							info.postfix );
					for ( j = 0; j < ( int ) info.count; j++ ) {
						printf( " Native Code[%d]: 0x%x |%s|\n", j,
								info.code[j], info.name[j] );
						PAPI_get_event_info( ( int ) info.code[j], &n_info );
						printf( " Number of Register Values: %d\n",
								n_info.count );
						for ( k = 0; k < ( int ) n_info.count; k++ )
							printf( " Register[%2d]: 0x%08x |%s|\n", k,
									n_info.code[k], n_info.name[k] );
						printf( " Native Event Description: |%s|\n\n",
								n_info.long_descr );
					}
				} else {	 /* must be a native event code */
					printf( "%-30s%s\n%-30s0x%-10x\n%-30s%d\n",
							"Event name:", info.symbol, "Event Code:",
							info.event_code, "Number of Register Values:",
							info.count );
					printf( "%-29s|%s|\n", "Description:", info.long_descr );
					for ( k = 0; k < ( int ) info.count; k++ )
						printf( " Register[%2d]: 0x%08x |%s|\n", k,
								info.code[k], info.name[k] );

					/* if unit masks exist but none are specified, process all */
					if ( !strchr( name, ':' ) ) {
						PAPI_component_info_t *c =
							( PAPI_component_info_t * )
							PAPI_get_component_info( PAPI_COMPONENT_INDEX
													 ( i ) );
						if ( c->cntr_umasks ) {
							if ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) ==
								 PAPI_OK ) {
								printf( "\nUnit Masks:\n" );
								do {
									retval = PAPI_get_event_info( i, &info );
									if ( retval == PAPI_OK ) {
										if ( parse_unit_masks( &info ) ) {
											printf( "%-29s|%s|%s|\n",
													" Mask Info:", info.symbol,
													info.long_descr );
											for ( k = 0; k < ( int ) info.count;
												  k++ )
												printf
													( "  Register[%2d]:  0x%08x  |%s|\n",
													  k, info.code[k],
													  info.name[k] );
										}
									}
								} while ( PAPI_enum_event
										  ( &i,
											PAPI_NTV_ENUM_UMASKS ) == PAPI_OK );
							}
						}
					}
				}
			}
		} else
			printf
				( "Sorry, an event by the name '%s' could not be found.\n Is it typed correctly?\n\n",
				  name );
	} else {
		/* For consistency, always ASK FOR the first event */
		i = 0 | PAPI_PRESET_MASK;
		PAPI_enum_event( &i, PAPI_ENUM_FIRST );

		if ( print_tabular ) {
			printf( "    Name        Code    " );
			if ( !print_avail_only )
				printf( "Avail " );
			printf( "Deriv Description (Note)\n" );
		} else {
			printf
				( "%-13s%-11s%-8s%-16s\n |Long Description|\n |Developer's Notes|\n |Derived|\n |PostFix|\n Native Code[n]: <hex> |name|\n",
				  "Symbol", "Event Code", "Count", "|Short Description|" );
		}
		do {
			if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
				if ( print_tabular ) {
					if ( filter & info.event_type ) {
						if ( print_avail_only ) {
							if ( info.count )
								printf( "%-13s0x%x  %-5s%s",
										info.symbol,
										info.event_code,
										is_derived( &info ), info.long_descr );
							if ( info.note[0] )
								printf( " (%s)", info.note );
							printf( "\n" );
						} else {
							printf( "%-13s0x%x  %-6s%-4s %s",
									info.symbol,
									info.event_code,
									( info.count ? "Yes" : "No" ),
									is_derived( &info ), info.long_descr );
							if ( info.note[0] )
								printf( " (%s)", info.note );
							printf( "\n" );
						}
						tot_count++;
						if ( info.count )
							avail_count++;
						if ( !strcmp( is_derived( &info ), "Yes" ) )
							deriv_count++;
					}
				} else {
					if ( ( print_avail_only && info.count ) ||
						 ( print_avail_only == 0 ) ) {
						printf
							( "%s\t0x%x\t%d\t|%s|\n |%s|\n |%s|\n |%s|\n |%s|\n",
							  info.symbol, info.event_code, info.count,
							  info.short_descr, info.long_descr, info.note,
							  info.derived, info.postfix );
						for ( j = 0; j < ( int ) info.count; j++ )
							printf( " Native Code[%d]: 0x%x |%s|\n", j,
									info.code[j], info.name[j] );
					}
					tot_count++;
					if ( info.count )
						avail_count++;
					if ( !strcmp( is_derived( &info ), "Yes" ) )
						deriv_count++;
				}
			}
		} while ( PAPI_enum_event( &i, print_avail_only ) == PAPI_OK );
	}
	printf
		( "-------------------------------------------------------------------------\n" );
	if ( !print_event_info ) {
		if ( print_avail_only ) {
			printf( "Of %d available events, %d ", avail_count, deriv_count );
		} else {
			printf( "Of %d possible events, %d are available, of which %d ",
					tot_count, avail_count, deriv_count );
		}
		if ( deriv_count == 1 ) {
			printf( "is derived.\n\n" );
		} else {
			printf( "are derived.\n\n" );
		}
	}

	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
