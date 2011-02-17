/* This file performs the following test: hardware info and which native events are available */
/** @file native_avail.c
  *	@page papi_native_avail
  * @brief papi_native_avail utility. 
  *	@section  NAME
  *		papi_native_avail - provides detailed information for PAPI native events. 
  *
  *	@section Synopsis
  *
  *	@section Description
  *		papi_native_avail is a PAPI utility program that reports information 
  *		about the native events available on the current platform. 
  *		A native event is an event specific to a specific hardware platform. 
  *		On many platforms, a specific native event may have a number of optional settings. 
  *		In such cases, the native event and the valid settings are presented, 
  *		rather than every possible combination of those settings. 
  *		For each native event, a name, a description, and specific bit patterns are provided.
  *
  *	@section Options
  *		This utility has no command line options.
  *
  *	@section Bugs
  *		There are no known bugs in this utility. 
  *		If you find a bug, it should be reported to the 
  *		PAPI Mailing List at <ptools-perfapi@ptools.org>. 
 */

#include "papi_test.h"
extern int TESTS_QUIET;				   /* Declared in test_utils.c */

#define EVT_LINE 80

typedef struct command_flags
{
	int help;
	int details;
	int named;
	char *name;
	int darr;
	int dear;
	int iarr;
	int iear;
	int opcm;
	int umask;
	int groups;
} command_flags_t;

static void
print_help( char **argv, const PAPI_component_info_t * c )
{
	printf( "This is the PAPI native avail program.\n" );
	printf
		( "It provides availability and detail information for PAPI native events.\n" );
	printf( "Usage: %s [options]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "  --help, -h    print this help message\n" );
	printf
		( "   -d           display detailed information about native events\n" );
	printf
		( "   -e EVENTNAME display detail information about named native event\n" );

	if ( c->data_address_range )
		printf
			( "  --darr        display events supporting Data Address Range Restriction\n" );
	if ( c->cntr_DEAR_events )
		printf
			( "  --dear        display Data Event Address Register events only\n" );
	if ( c->instr_address_range )
		printf
			( "  --iarr        display events supporting Instruction Address Range Restriction\n" );
	if ( c->cntr_IEAR_events )
		printf
			( "  --iear        display Instruction Event Address Register events only\n" );
	if ( c->cntr_OPCM_events )
		printf( "  --opcm        display events supporting OpCode Matching\n" );
	if ( c->cntr_umasks )
		printf( "  --nomasks    suppress display of Unit Mask information\n" );
	if ( c->cntr_groups )
		printf
			( "  --nogroups    suppress display of Event grouping information\n" );
	printf( "\n" );
}

static void
parse_args( int argc, char **argv, command_flags_t * f )
{
	const PAPI_component_info_t *c = NULL;
	int i;

	c = PAPI_get_component_info( 0 );

	/* Look for all currently defined commands */
	memset( f, 0, sizeof ( command_flags_t ) );
	if ( c->cntr_umasks )
		f->umask = 1;
	if ( c->cntr_groups )
		f->groups = 1;
	for ( i = 1; i < argc; i++ ) {
		if ( !strcmp( argv[i], "--darr" ) )
			f->darr = 1;
		else if ( !strcmp( argv[i], "--dear" ) )
			f->dear = 1;
		else if ( !strcmp( argv[i], "--iarr" ) )
			f->iarr = 1;
		else if ( !strcmp( argv[i], "--iear" ) )
			f->iear = 1;
		else if ( !strcmp( argv[i], "--opcm" ) )
			f->opcm = 1;
		else if ( !strcmp( argv[i], "--noumasks" ) )
			f->umask = 0;
		else if ( !strcmp( argv[i], "--nogroups" ) )
			f->groups = 0;
		else if ( !strcmp( argv[i], "-d" ) )
			f->details = 1;
		else if ( !strcmp( argv[i], "-e" ) ) {
			f->named = 1;
			f->name = argv[i + 1];
			if ( ( f->name == NULL ) || ( strlen( f->name ) == 0 ) ||
				 ( f->name[0] == '-' ) )
				f->help = 1;
		} else if ( strstr( argv[i], "-h" ) )
			f->help = 1;
		else
			printf( "%s is not supported\n", argv[i] );
	}

	/* if help requested, print and bail */
	if ( f->help ) {
		print_help( argv, c );
		exit( 1 );
	}

	/* Look for flags unsupported by this component */
	if ( f->darr & !( c->data_address_range ) ) {
		f->darr = 0;
		printf( "-darr not supported\n" );
	}
	if ( f->dear & !( c->cntr_DEAR_events ) ) {
		f->dear = 0;
		printf( "-dear not supported\n" );
	}
	if ( f->iarr & !( c->instr_address_range ) ) {
		f->iarr = 0;
		printf( "-iarr not supported\n" );
	}
	if ( f->iear & !( c->cntr_IEAR_events ) ) {
		f->iear = 0;
		printf( "-iear not supported\n" );
	}
	if ( f->opcm & !( c->cntr_OPCM_events ) ) {
		f->opcm = 0;
		printf( "-opcm not supported\n" );
	}

	/* Look for mutual exclusivity */
	if ( f->darr + f->dear + f->iarr + f->iear + f->opcm > 1 ) {
		printf
			( "-darr, -dear, -iarr, -iear, and -opcm are mutually exclusve\n" );
		exit( 1 );
	}
}

static void
space_pad( char *str, int spaces )
{
	while ( spaces-- > 0 )
		strcat( str, " " );
}

static void
print_event( PAPI_event_info_t * info, int offset )
{
	unsigned int i, j = 0;
	char str[EVT_LINE + EVT_LINE];

	/* indent by offset */
	if ( offset )
		sprintf( str, "  " );
	else
		sprintf( str, "0x" );

	/* copy the code and symbol */
	sprintf( &str[strlen( str )], "%-11x%s", info->event_code, info->symbol );

	if ( strlen( info->long_descr ) > 0 )
		strcat( str, "  | " );

	while ( j <= strlen( info->long_descr ) ) {
		i = EVT_LINE - ( unsigned int ) strlen( str ) - 2;
		if ( i > 0 ) {
			strncat( str, &info->long_descr[j], i );
			j += i;
			i = ( unsigned int ) strlen( str );
			space_pad( str, EVT_LINE - ( int ) i - 1 );
			strcat( str, "|" );
		}
		printf( "%s\n", str );
		str[0] = 0;
		space_pad( str, 11 );
		strcat( str, "| " );
	}
}

static int
parse_unit_masks( PAPI_event_info_t * info )
{
  char *pmask,*ptr;

  /* handle libpfm4-style events which have a pmu::event type event name */
  if ((ptr=strstr(info->symbol, "::"))) {
    ptr+=2;
  }
  else {
    ptr=info->symbol;
  }

	if ( ( pmask = strchr( ptr, ':' ) ) == NULL ) {
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
	int i, j = 0, k;
	int retval;
	PAPI_event_info_t info;
	const PAPI_hw_info_t *hwinfo = NULL;
	command_flags_t flags;
	int enum_modifier;
	int numcmp, cid;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	/* Initialize before parsing the input arguments */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	parse_args( argc, argv, &flags );

	/* Set any needed modifier flags */
	if ( flags.dear )
		enum_modifier = PAPI_NTV_ENUM_DEAR;
	else if ( flags.darr )
		enum_modifier = PAPI_NTV_ENUM_DARR;
	else if ( flags.iear )
		enum_modifier = PAPI_NTV_ENUM_IEAR;
	else if ( flags.iarr )
		enum_modifier = PAPI_NTV_ENUM_IARR;
	else if ( flags.opcm )
		enum_modifier = PAPI_NTV_ENUM_OPCM;
	else
		enum_modifier = PAPI_ENUM_EVENTS;

	if ( !TESTS_QUIET ) {
		retval = PAPI_set_debug( PAPI_VERB_ECONT );
		if ( retval != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_set_debug", retval );
	}

	retval =
		papi_print_header
		( "Available native events and hardware information.\n", 1, &hwinfo );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	if ( flags.named ) {
		if ( PAPI_event_name_to_code( flags.name, &i ) == PAPI_OK ) {
			if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
				printf( "%-30s%s\n%-30s0x%-10x\n%-30s%d\n",
						"Event name:", info.symbol, "Event Code:",
						info.event_code, "Number of Register Values:",
						info.count );
				printf( "%-29s|%s|\n", "Description:", info.long_descr );
				for ( k = 0; k < ( int ) info.count; k++ )
					printf( " Register[%2d]:   0x%08x  |%s|\n", k, info.code[k],
							info.name[k] );

				/* if unit masks exist but none are specified, process all */
				if ( flags.umask && !strchr( flags.name, ':' ) ) {
					if ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) ==
						 PAPI_OK ) {
						printf( "\nUnit Masks:\n" );
						do {
							retval = PAPI_get_event_info( i, &info );
							if ( retval == PAPI_OK ) {
								if ( parse_unit_masks( &info ) ) {
									printf( "%-29s|%s|%s|\n", " Mask Info:",
											info.symbol, info.long_descr );
									for ( k = 0; k < ( int ) info.count; k++ )
										printf
											( "  Register[%2d]:  0x%08x  |%s|\n",
											  k, info.code[k], info.name[k] );
								}
							}
						} while ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) ==
								  PAPI_OK );
					}
				}
			}
		} else {
			printf
				( "Sorry, an event by the name '%s' could not be found.\n Is it typed correctly?\n\n",
				  flags.name );
		        exit( 1 );
		}
	}
        else {

	   printf( "%-12s %s  | %s |\n", "Event Code", "Symbol", "Long Description" );
	   printf
		( "--------------------------------------------------------------------------------\n" );

	   numcmp = PAPI_num_components(  );

	   j = 0;

	      for ( cid = 0; cid < numcmp; cid++ ) {

		   /* For platform independence, always ASK FOR the first event */
		   /* Don't just assume it'll be the first numeric value */
		   i = 0 | PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cid );
		   PAPI_enum_event( &i, PAPI_ENUM_FIRST );

		   do {
			memset( &info, 0, sizeof ( info ) );
			retval = PAPI_get_event_info( i, &info );

			/* This event may not exist */
			if ( retval == PAPI_ENOEVNT )
				continue;

			/* count only events that as supported by host cpu */
			j++;

			print_event( &info, 0 );

			if ( flags.details ) {
				for ( k = 0; k < ( int ) info.count; k++ ) {
					if ( strlen( info.name[k] ) ) {
						printf( "  Register[%d] Name: %-20s  Value: 0x%-28x|\n",
								k, info.name[k], info.code[k] );
					}
				}
			}

/*		modifier = PAPI_NTV_ENUM_GROUPS returns event codes with a
			groups id for each group in which this
			native event lives, in bits 16 - 23 of event code
			terminating with PAPI_ENOEVNT at the end of the list.
*/
			if ( flags.groups ) {
				k = i;
				if ( PAPI_enum_event( &k, PAPI_NTV_ENUM_GROUPS ) == PAPI_OK ) {
					printf( "Groups: " );
					do {
						printf( "%4d",
								( ( k & PAPI_NTV_GROUP_AND_MASK ) >>
								  PAPI_NTV_GROUP_SHIFT ) - 1 );
					} while ( PAPI_enum_event( &k, PAPI_NTV_ENUM_GROUPS ) ==
							  PAPI_OK );
					printf( "\n" );
				}
			}

/*		modifier = PAPI_NTV_ENUM_UMASKS returns an event code for each
			unit mask bit defined for this native event. This can be used
			to get event info for that mask bit. It terminates
			with PAPI_ENOEVNT at the end of the list.
*/
			if ( flags.umask ) {
				k = i;
				if ( PAPI_enum_event( &k, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK ) {
					do {
						retval = PAPI_get_event_info( k, &info );
						if ( retval == PAPI_OK ) {
							if ( parse_unit_masks( &info ) )
								print_event( &info, 2 );
						}
					} while ( PAPI_enum_event( &k, PAPI_NTV_ENUM_UMASKS ) ==
							  PAPI_OK );
				}
				printf
					( "--------------------------------------------------------------------------------\n" );
			}
		} while ( PAPI_enum_event( &i, enum_modifier ) == PAPI_OK );
	      }
	
	
	      printf
		( "--------------------------------------------------------------------------------\n" );
	      printf( "Total events reported: %d\n", j );
	}
	test_pass( __FILE__, NULL, 0 );
	exit( 0 );
}
