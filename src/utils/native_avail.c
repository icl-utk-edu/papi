/* This file utility reports hardware info and native event availability */
/** file native_avail.c
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
  * <ul>
  * <li>--help, -h    print this help message
  * <li>-d            display detailed information about native events
  * <li>-e EVENTNAME  display detailed information about named native event
  * <li>-i EVENTSTR   include only event names that contain EVENTSTR
  * <li>-x EVENTSTR   exclude any event names that contain EVENTSTR
  * <li>--noumasks    suppress display of Unit Mask information
  * </ul>
  *
  * Processor-specific options
  * <ul>
  * <li>--darr        display events supporting Data Address Range Restriction
  * <li>--dear        display Data Event Address Register events only
  * <li>--iarr        display events supporting Instruction Address Range Restriction
  * <li>--iear        display Instruction Event Address Register events only
  * <li>--opcm        display events supporting OpCode Matching
  * <li>--nogroups    suppress display of Event grouping information
  * </ul> 
  *
  *	@section Bugs
  *		There are no known bugs in this utility. 
  *		If you find a bug, it should be reported to the 
  *		PAPI Mailing List at <ptools-perfapi@ptools.org>. 
 */

#include "papi_test.h"

#define EVT_LINE 80

typedef struct command_flags
{
	int help;
	int details;
	int named;
	int include;
	int xclude;
	char *name, *istr, *xstr;
	int darr;
	int dear;
	int iarr;
	int iear;
	int opcm;
	int umask;
	int groups;
} command_flags_t;

static void
print_help( char **argv )
{
	printf( "This is the PAPI native avail program.\n" );
	printf( "It provides availability and detail information for PAPI native events.\n" );
	printf( "Usage: %s [options]\n", argv[0] );
	printf( "\nOptions:\n" );
	printf( "   --help, -h   print this help message\n" );
	printf( "   -d           display detailed information about native events\n" );
	printf( "   -e EVENTNAME display detailed information about named native event\n" );
	printf( "   -i EVENTSTR  include only event names that contain EVENTSTR\n" );
	printf( "   -x EVENTSTR  exclude any event names that contain EVENTSTR\n" );
	printf( "   --noumasks   suppress display of Unit Mask information\n" );
	printf( "\nProcessor-specific options\n");
	printf( "  --darr        display events supporting Data Address Range Restriction\n" );
	printf( "  --dear        display Data Event Address Register events only\n" );
	printf( "  --iarr        display events supporting Instruction Address Range Restriction\n" );
	printf( "  --iear        display Instruction Event Address Register events only\n" );
    printf( "  --opcm        display events supporting OpCode Matching\n" );
	printf( "  --nogroups    suppress display of Event grouping information\n" );
	printf( "\n" );
}

static int
no_str_arg( char *arg )
{
	return ( ( arg == NULL ) || ( strlen( arg ) == 0 ) || ( arg[0] == '-' ) );
}

static void
parse_args( int argc, char **argv, command_flags_t * f )
{

	int i;

	/* Look for all currently defined commands */
	memset( f, 0, sizeof ( command_flags_t ) );
	f->umask = 1;
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
			i++;
			f->name = argv[i];
			if ( i >= argc || no_str_arg( f->name ) ) {
				printf( "Invalid argument for -e\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-i" ) ) {
			f->include = 1;
			i++;
			f->istr = argv[i];
			if ( i >= argc || no_str_arg( f->istr ) ) {
				printf( "Invalid argument for -i\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-x" ) ) {
			f->xclude = 1;
			i++;
			f->xstr = argv[i];
			if ( i >= argc || no_str_arg( f->xstr ) ) {
				printf( "Invalid argument for -x\n");
				exit(1);
			}
		} else if ( !strcmp( argv[i], "-h" ) || !strcmp( argv[i], "--help" ) )
			f->help = 1;
		else {
			printf( "%s is not supported\n", argv[i] );
			exit(1);
		}
	}

	/* if help requested, print and bail */
	if ( f->help ) {
		print_help( argv);
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
	if ( offset ) {
	   printf( "|     %-73s|\n", info->symbol );
	}
	else {
	   printf( "| %-77s|\n", info->symbol );
	}

	while ( j <= strlen( info->long_descr ) ) {
	   i = EVT_LINE - 12 - 2;
	   if ( i > 0 ) {
	      str[0] = 0;
	      strcat(str,"| " );
	      space_pad( str, 11 );
	      strncat( str, &info->long_descr[j], i );
	      j += i;
	      i = ( unsigned int ) strlen( str );
	      space_pad( str, EVT_LINE - ( int ) i - 1 );
	      strcat( str, "|" );
	   }
	   printf( "%s\n", str );
	}
}

static int
parse_unit_masks( PAPI_event_info_t * info )
{
  char *pmask,*ptr;

  /* handle the PAPI component-style events which have a component:::event type */
  if ((ptr=strstr(info->symbol, ":::"))) {
    ptr+=3;
  /* handle libpfm4-style events which have a pmu::event type event name */
  } else if ((ptr=strstr(info->symbol, "::"))) {
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

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );

	/* Initialize before parsing the input arguments */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Parse the command-line arguments */
	parse_args( argc, argv, &flags );

	/* Set enum modifier mask */
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
	   if ( retval != PAPI_OK ) {
	      test_fail( __FILE__, __LINE__, "PAPI_set_debug", retval );
	   }
	}

	retval = papi_print_header( "Available native events and hardware information.\n", &hwinfo );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}


	/* Do this code if the event name option was specified on the commandline */
	if ( flags.named ) {
	   if ( PAPI_event_name_to_code( flags.name, &i ) == PAPI_OK ) {
	      if ( PAPI_get_event_info( i, &info ) == PAPI_OK ) {
		 printf( "%-30s%s\n",
			 "Event name:", info.symbol);
		 printf( "%-29s|%s|\n", "Description:", info.long_descr );

		     /* if unit masks exist but none specified, process all */
		     if ( !strchr( flags.name, ':' ) ) {
			if ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK ) {
			   printf( "\nUnit Masks:\n" );
			   do {
			      retval = PAPI_get_event_info( i, &info );
			      if ( retval == PAPI_OK ) {
				 if ( parse_unit_masks( &info ) ) {
				    printf( "%-29s|%s|%s|\n", " Mask Info:",
					    info.symbol, info.long_descr );
				 }
			      }
			   } while ( PAPI_enum_event( &i, PAPI_NTV_ENUM_UMASKS ) == PAPI_OK );
			}
		     }
	      }
	   } else {
	     printf("Sorry, an event by the name '%s' could not be found.\n",
		    flags.name);
	     printf("Is it typed correctly?\n\n");
	     exit( 1 );
	   }
	}
    else {

	   /* Print *ALL* available events */

	   numcmp = PAPI_num_components(  );

	   j = 0;

	   for ( cid = 0; cid < numcmp; cid++ ) {

	       const PAPI_component_info_t *component;
	       component=PAPI_get_component_info(cid);

	       /* Skip disabled components */
	       if (component->disabled) continue;

	       printf( "===============================================================================\n" );
	       printf( " Native Events in Component: %s\n",component->name);
	       printf( "===============================================================================\n" );
	     
	       /* Always ASK FOR the first event */
	       /* Don't just assume it'll be the first numeric value */
	       i = 0 | PAPI_NATIVE_MASK;

	       retval=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );

	       if (retval==PAPI_OK) 

	       do {
			  memset( &info, 0, sizeof ( info ) );
			  retval = PAPI_get_event_info( i, &info );

			  /* This event may not exist */
			  if ( retval != PAPI_OK )
				 continue;

			  /* Bail if event name doesn't contain include string */
			  if ( flags.include ) {
			  	 if ( !strstr( info.symbol, flags.istr ) ) {
				 	continue;
				 }
			  }

			  /* Bail if event name does contain exclude string */
			  if ( flags.xclude ) {
			  	 if ( strstr( info.symbol, flags.xstr ) )
				 	continue;
			  }
			  
			  /* count only events that are actually processed */
			  j++;

			  print_event( &info, 0 );

			  if (flags.details) {
				if (info.units[0]) printf( "|     Units: %-67s|\n", 
							   info.units );
			  }

/*		modifier = PAPI_NTV_ENUM_GROUPS returns event codes with a
			groups id for each group in which this
			native event lives, in bits 16 - 23 of event code
			terminating with PAPI_ENOEVNT at the end of the list.
*/

			  /* This is an IBM Power issue */
			  if ( flags.groups ) {
				 k = i;
				 if ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_GROUPS, cid ) == PAPI_OK ) {
				printf( "Groups: " );
				do {
				  printf( "%4d", ( ( k & PAPI_NTV_GROUP_AND_MASK ) >>
							 PAPI_NTV_GROUP_SHIFT ) - 1 );
				} while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_GROUPS, cid ) ==PAPI_OK );
				printf( "\n" );
				 }
			  }

			  /* Print umasks */
			  /* components that don't have them can just ignore */

				  if ( flags.umask ) { 
				 k = i;
				 if ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK ) {
					do {
				   retval = PAPI_get_event_info( k, &info );
				   if ( retval == PAPI_OK ) {
					  if ( parse_unit_masks( &info ) )
						 print_event( &info, 2 );
				   }
					} while ( PAPI_enum_cmp_event( &k, PAPI_NTV_ENUM_UMASKS, cid ) == PAPI_OK );
				 }

			  }
			  printf( "--------------------------------------------------------------------------------\n" );

	       } while (PAPI_enum_cmp_event( &i, enum_modifier, cid ) == PAPI_OK );
	   }
	   	   	
	
	   printf("\n");
	   printf( "Total events reported: %d\n", j );
	}

	test_pass( __FILE__, NULL, 0 );
	exit( 0 );
}
