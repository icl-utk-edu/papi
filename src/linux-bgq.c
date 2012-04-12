/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq.c
 * CVS:     $Id: Exp $
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * Blue Gene/Q CPU component: BGPM / Punit
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the BGPM library.
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "linux-bgq.h"
#include "error.h"

/*
 * BG/Q specific 'stuff'
 */
#include <ucontext.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "spi/include/upci/upci.h"


// BG/Q macros
#define get_cycles GetTimeBase

// BG/Q external structures/functions/stuff
#if 1
UPC_Lock_t thdLocks[PAPI_MAX_LOCK];
#else
pthread_mutex_t thdLocks[PAPI_MAX_LOCK];
#endif

/* Defined in papa_data.c */
extern papi_mdi_t _papi_hwi_system_info;

extern papi_vector_t MY_VECTOR;

/* Defined in linux-bgq-memory.c */
extern int _bgq_get_memory_info( PAPI_hw_info_t * pHwInfo, int pCPU_Type );
extern int _bgq_get_dmem_info( PAPI_dmem_info_t * pDmemInfo );


/* Look for an event file 'name' in a couple common locations.
 Return a valid file handle if found */
static FILE *
open_event_table( char *name )
{
#ifdef DEBUG_BGQ
	//printf( "open_event_table\n" );
#endif
	
	FILE *table;
	
	SUBDBG( "Opening %s\n", name );
	printf( "Opening %s\n", name );
	printf("BEFORE ls...\n");
	system("ls");
	printf("\nAFTER ls...\n");
	table = fopen( name, "r" );
	if ( table == NULL ) {
		SUBDBG( "Open %s failed, trying ./%s.\n", name, PAPI_EVENT_FILE );
		printf( "Open %s failed, trying ./%s.\n", name, PAPI_EVENT_FILE );
		sprintf( name, "%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table == NULL ) {
		SUBDBG( "Open ./%s failed, trying ../%s.\n", name, PAPI_EVENT_FILE );
		printf( "Open ./%s failed, trying ../%s.\n", name, PAPI_EVENT_FILE );
		sprintf( name, "../%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table ) {
		SUBDBG( "Open %s succeeded.\n", name );
		printf( "Open %s succeeded.\n", name );
	}
	return ( table );
}


/* parse a single line from either a file or character table
 Strip trailing <cr>; return 0 if empty */
static int
get_event_line( char *line, FILE * table, char **tmp_bgq_events_table )
{
#ifdef DEBUG_BGQ
	//printf( "get_event_line\n" );
#endif
	
	int ret;
	int i;
	
	if ( table ) {
		if ( fgets( line, LINE_MAX, table ) ) {
			ret = 1;
			i = strlen( line );
			if ( line[i - 1] == '\n' )
				line[i - 1] = '\0';
		} else
			ret = 0;
	} else {
		for ( i = 0;
			 **tmp_bgq_events_table && **tmp_bgq_events_table != '\n';
			 i++ ) {
			line[i] = **tmp_bgq_events_table;
			( *tmp_bgq_events_table )++;
		}
		if ( **tmp_bgq_events_table == '\n' ) {
			( *tmp_bgq_events_table )++;
		}
		line[i] = '\0';
		ret = **tmp_bgq_events_table;
	}
	return ( ret );
}



/*  Trims blank space from both ends of a string (in place).
 Returns pointer to new start address */
static char *
trim_string( char *in )
{
#ifdef DEBUG_BGQ
	//printf( "trim_string\n" );
#endif
	
	int len, i = 0;
	char *start = in;
	
	if ( in == NULL )
		return ( in );
	len = strlen( in );
	if ( len == 0 )
		return ( in );
	/* Trim left */
	while ( i < len ) {
		if ( isblank( in[i] ) ) {
			in[i] = '\0';
			start++;
		} else
			break;
		i++;
	}
	/* Trim right */
	i = strlen( start ) - 1;
	while ( i >= 0 ) {
		if ( isblank( start[i] ) )
			start[i] = '\0';
		else
			break;
		i--;
	}
	return ( start );
}


/*  Calls trim_string to remove blank space;
 Removes paired punctuation delimiters from
 beginning and end of string. If the same punctuation 
 appears first and last (quotes, slashes) they are trimmed;
 Also checks for the following pairs: () <> {} [] */
static char *
trim_note( char *in )
{
#ifdef DEBUG_BGQ
	//printf( "trim_note\n" );
#endif
	
	int len;
	char *note, start, end;
	
	note = trim_string( in );
	if ( note != NULL ) {
		len = strlen( note );
		if ( len > 0 ) {
			if ( ispunct( *note ) ) {
				start = *note;
				end = note[len - 1];
				if ( ( start == end )
					|| ( ( start == '(' ) && ( end == ')' ) )
					|| ( ( start == '<' ) && ( end == '>' ) )
					|| ( ( start == '{' ) && ( end == '}' ) )
					|| ( ( start == '[' ) && ( end == ']' ) ) ) {
					note[len - 1] = '\0';
					*note = '\0';
					note++;
				}
			}
		}
	}
	return ( note );
}


static int
find_preset_code( char *tmp, int *code )
{
#ifdef DEBUG_BGQ
	//printf( "find_preset_code\n" );
#endif
	
	int i = 0;
	
	while ( _papi_hwi_presets.info[i].symbol != NULL ) {
		if ( strcasecmp( tmp, _papi_hwi_presets.info[i].symbol ) == 0 ) {
			*code = i | PAPI_PRESET_MASK;
			return ( PAPI_OK );
		}
		i++;
	}
	return ( PAPI_EINVAL );
}


/* Static version of the events file. */
#if defined(STATIC_PAPI_EVENTS_TABLE)
#include "papi_events_table.h"
#else
static char *papi_events_table = NULL;
#endif

/* #define SHOW_LOADS */
#ifdef SHOW_LOADS
#define SUBDBG printf
#endif


static int
load_preset_table( char *pmu_name, int pmu_type,
				  bgq_preset_search_entry_t * here )
{
#ifdef DEBUG_BGQ
	//printf( "load_preset_table\n" );
#endif
	
	char line[LINE_MAX];
	char name[PATH_MAX] = "builtin papi_events_table";
	char *tmp_papi_events_table = NULL;
	char *tmpn;
	FILE *table;
	int line_no = 1, derived = 0, insert = 0, preset = 0;
	int get_presets = 0;			   /* only get PRESETS after CPU is identified */
	int found_presets = 0;			   /* only terminate search after PRESETS are found */
	/* this allows support for synonyms for CPU names */
	
#ifdef SHOW_LOADS
	SUBDBG( "%p\n", here );
#endif
	
	/* try the environment variable first */
	if ( ( tmpn = getenv( "PAPI_BGQ_EVENT_FILE" ) ) &&
		( strlen( tmpn ) != 0 ) ) {
		sprintf( name, "%s", tmpn );
		table = fopen( name, "r" );
	}
	/* if no valid environment variable, look for built-in table */
	else if ( papi_events_table ) {
		tmp_papi_events_table = papi_events_table;
		table = NULL;
	}
	/* if no env var and no built-in, search for default file */
	else {
#ifdef PAPI_DATADIR
		sprintf( name, "%s/%s", PAPI_DATADIR, PAPI_EVENT_FILE );
#else
		sprintf( name, "%s", PAPI_EVENT_FILE );
#endif
		table = open_event_table( name );
	}
	
	
	/* if no valid file or built-in table, bail */
	if ( table == NULL && tmp_papi_events_table == NULL ) {
		PAPIERROR
		( "fopen(%s): %s, please set the PAPI_PERFMON_EVENT_FILE env. variable",
		 name, strerror( errno ) );
		return ( PAPI_ESYS );
	}
	
	/* at this point either a valid file pointer or built-in table pointer */
	while ( get_event_line( line, table, &tmp_papi_events_table ) ) {
		char *t;
		int i;
		t = trim_string( strtok( line, "," ) );
		if ( ( t == NULL ) || ( strlen( t ) == 0 ) )
			continue;
		if ( t[0] == '#' ) {
			/*	  SUBDBG("Comment found on line %d\n",line_no); */
			goto nextline;
		} else if ( strcasecmp( t, "CPU" ) == 0 ) {
#ifdef SHOW_LOADS
			SUBDBG( "CPU token found on line %d\n", line_no );
#endif
			if ( get_presets != 0 && found_presets != 0 ) {
#ifdef SHOW_LOADS
				SUBDBG( "Ending preset scanning at line %d of %s.\n", line_no,
					   name );
#endif
				goto done;
			}
			t = trim_string( strtok( NULL, "," ) );
			if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
				PAPIERROR
				( "Expected name after CPU token at line %d of %s -- ignoring",
				 line_no, name );
				goto nextline;
			}
#ifdef SHOW_LOADS
			SUBDBG( "Examining CPU (%s) vs. (%s)\n", t, pmu_name );
#endif
			if ( strcasecmp( t, pmu_name ) == 0 ) {
				int type;
				
#ifdef SHOW_LOADS
				SUBDBG( "Found CPU %s at line %d of %s.\n", t, line_no, name );
#endif
				t = trim_string( strtok( NULL, "," ) );
				if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
#ifdef SHOW_LOADS
					SUBDBG
					( "No additional qualifier found, matching on string.\n" );
#endif
					get_presets = 1;
				} else if ( ( sscanf( t, "%d", &type ) == 1 ) &&
						   ( type == pmu_type ) ) {
#ifdef SHOW_LOADS
					SUBDBG( "Found CPU %s type %d at line %d of %s.\n",
						   pmu_name, type, line_no, name );
#endif
					get_presets = 1;
				}
				else {
#ifdef SHOW_LOADS
					SUBDBG( "Additional qualifier match failed %d vs %d.\n", pmu_type, type );
#endif
				}
			}
		} else if ( strcasecmp( t, "PRESET" ) == 0 ) {
#ifdef SHOW_LOADS
			SUBDBG( "PRESET token found on line %d\n", line_no );
#endif
			if ( get_presets == 0 )
				goto nextline;
			found_presets = 1;
			t = trim_string( strtok( NULL, "," ) );
			if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
				printf
				( "Expected name after PRESET token at line %d of %s -- ignoring",
				 line_no, name );
				goto nextline;
			}
#ifdef SHOW_LOADS
			SUBDBG( "Examining preset %s\n", t );
#endif
			if ( find_preset_code( t, &preset ) != PAPI_OK ) {
				PAPIERROR
				( "Invalid preset name %s after PRESET token at line %d of %s -- ignoring",
				 t, line_no, name );
				goto nextline;
			}
#ifdef SHOW_LOADS
			SUBDBG( "Found 0x%08x for %s\n", preset, t );
#endif
			t = trim_string( strtok( NULL, "," ) );
			if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
				PAPIERROR
				( "Expected derived type after PRESET token at line %d of %s -- ignoring",
				 line_no, name );
				goto nextline;
			}
#ifdef SHOW_LOADS
			SUBDBG( "Examining derived %s\n", t );
#endif
			if ( _papi_hwi_derived_type( t, &derived ) != PAPI_OK ) {
				PAPIERROR
				( "Invalid derived name %s after PRESET token at line %d of %s -- ignoring",
				 t, line_no, name );
				goto nextline;
			}
#ifdef SHOW_LOADS
			SUBDBG( "Found %d for %s\n", derived, t );
			SUBDBG( "Adding 0x%x,%d to preset search table.\n", preset,
				   derived );
#endif
			here[insert].preset = preset;
			here[insert].derived = derived;
#ifdef SHOW_LOADS
			SUBDBG( "%d Adding 0x%x,%d to preset search table.\n", insert,
				   here[insert].preset, here[insert].derived );
#endif
			
			/* Derived support starts here */
			/* Special handling for postfix */
			if ( derived == DERIVED_POSTFIX ) {
				t = trim_string( strtok( NULL, "," ) );
				if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
					PAPIERROR
					( "Expected Operation string after derived type DERIVED_POSTFIX at line %d of %s -- ignoring",
					 line_no, name );
					goto nextline;
				}
#ifdef SHOW_LOADS
				SUBDBG( "Saving PostFix operations %s\n", t );
#endif
				here[insert].operation = strdup( t );
			}
			/* All derived terms collected here */
			i = 0;
			do {
				t = trim_string( strtok( NULL, "," ) );
				if ( ( t == NULL ) || ( strlen( t ) == 0 ) )
					break;
				if ( strcasecmp( t, "NOTE" ) == 0 )
					break;
				here[insert].findme[i] = strdup( t );
#ifdef SHOW_LOADS
				SUBDBG( "Adding term (%d) %s to preset event 0x%x.\n", i, t,
					   preset );
#endif
			} while ( ++i < MAX_COUNTER_TERMS );
			/* End of derived support */
			
			if ( i == 0 ) {
				PAPIERROR
				( "Expected PFM event after DERIVED token at line %d of %s -- ignoring",
				 line_no, name );
				goto nextline;
			}
			if ( i == MAX_COUNTER_TERMS )
				t = trim_string( strtok( NULL, "," ) );
			
			/* Handle optional NOTEs */
			if ( t && ( strcasecmp( t, "NOTE" ) == 0 ) ) {
#ifdef SHOW_LOADS
				SUBDBG( "%s found on line %d\n", t, line_no );
#endif
				t = trim_note( strtok( NULL, "" ) );	/* read the rest of the line */
				if ( ( t == NULL ) || ( strlen( t ) == 0 ) )
					PAPIERROR( "Expected Note string at line %d of %s\n",
							  line_no, name );
				else {
					here[insert].note = strdup( t );
#ifdef SHOW_LOADS
					SUBDBG( "NOTE: --%s-- found on line %d\n", t, line_no );
#endif
				}
			}
			
			insert++;
		} else {
			PAPIERROR( "Unrecognized token %s at line %d of %s -- ignoring", t,
					  line_no, name );
			goto nextline;
		}
	nextline:
		line_no++;
	}
done:
	if ( table )
		fclose( table );
	return ( insert );
}



static int
bgq_find_full_event( char *name, int *evtcode )
{
#ifdef DEBUG_BGQ
	//printf( "bgq_find_full_event\n" );
#endif
	int i, retval;
	
	/* Valid BGPM event Id's start at 1 */
	for ( i = 1; i < MAX_COUNTERS; i++ ) {
		Bgpm_EventInfo_t pInfo;
		retval = Bgpm_GetEventIdInfo( i, &pInfo );
		CHECK_BGPM_ERROR( retval, "Bgpm_GetEventIdInfo" );
		
		if ( pInfo.unitType == BGPMUnitType_PUNIT ) {
			if ( strcmp( name, pInfo.label ) == 0 ) {
				// (-1) because BGPM events start at 1 and PAPI events stgart at 0
				*evtcode = ( pInfo.evtId | PAPI_NATIVE_MASK ) - 1;
				return PAPI_OK;
			}
		}
	}
	
	return PAPI_ESBSTR;
}


static int
generate_preset_search_map( hwi_search_t ** maploc, hwi_dev_notes_t ** noteloc,
						   bgq_preset_search_entry_t * strmap, int npresets )
{
#ifdef DEBUG_BGQ
	//printf( "generate_preset_search_map\n" );
#endif
	
	int i = 0, j = 0, k = 0, term, ret;
	hwi_search_t *psmap;
	hwi_dev_notes_t *notemap;
	int event;
	
	/* Count up the proposed presets */
	for ( i = 0; i < npresets; i++ )
		SUBDBG( "%d 0x%x\n", i, strmap[i].preset );
	
	/*while (strmap[i].preset){
	 SUBDBG("%d %p\n", i, strmap[i].preset);
	 i++;
	 } */
	SUBDBG( "generate_preset_search_map(%p,%p,%p) %d proposed presets\n",
		   maploc, noteloc, strmap, i );
	i++;
	
	/* Add null entry */
	psmap = ( hwi_search_t * ) malloc( i * sizeof ( hwi_search_t ) );
	notemap = ( hwi_dev_notes_t * ) malloc( i * sizeof ( hwi_dev_notes_t ) );
	if (!psmap || !notemap) {
		free(psmap);
		free(notemap);
		return PAPI_ENOMEM;
	}
	memset( psmap, 0x0, i * sizeof ( hwi_search_t ) );
	memset( notemap, 0x0, i * sizeof ( hwi_dev_notes_t ) );
	
	/*i = 0;
	 while (strmap[i].preset) */
	for ( i = 0; i < npresets; i++ ) {
		/* Handle derived events */
		term = 0;
		do {
			if ( ( ret =
				  bgq_find_full_event( strmap[i].findme[term],
									  &event ) ) == PAPI_OK ) {
					  /*if ((ret = setup_preset_term(&psmap[j].data.native[term], &event)) == PAPI_OK)
					   {
					   term++;
					   }
					   else break;
					   */
					  psmap[j].data.native[term] = event;
					  term++;
					  SUBDBG( "\t%d  0x%x\n", i, event );
				  } else {
					  PAPIERROR( "bgq_find_full_event(%s)",
								strmap[i].findme[term] );
					  term++;
				  }
		} while ( strmap[i].findme[term] != NULL && term < MAX_COUNTER_TERMS );
		
		/* terminate the native term array with PAPI_NULL */
		if ( term < MAX_COUNTER_TERMS )
			psmap[j].data.native[term] = PAPI_NULL;
		
		if ( ret == PAPI_OK ) {
			psmap[j].event_code = strmap[i].preset;
			psmap[j].data.derived = strmap[i].derived;
			if ( strmap[i].derived == DERIVED_POSTFIX ) {
				strncpy( psmap[j].data.operation, strmap[i].operation,
						PAPI_MIN_STR_LEN );
			}
			if ( strmap[i].note ) {
				notemap[k].event_code = strmap[i].preset;
				notemap[k].dev_note = strdup( strmap[i].note );
				k++;
			}
			j++;
		}
		/*i++; */
	}
	if ( i != j ) {
		PAPIERROR( "%d of %d events in %s were not valid", i - j, i,
				  PAPI_EVENT_FILE );
	}
	SUBDBG( "generate_preset_search_map(%p,%p,%p) %d actual presets\n", maploc,
		   noteloc, strmap, j );
	*maploc = psmap;
	*noteloc = notemap;
	return ( PAPI_OK );
}


/* Frees memory for all the strdup'd char strings in a preset string array.
 Assumes the array is initialized to 0 and has at least one 0 entry at the end.
 free()ing a NULL pointer is a NOP. */
static void
free_preset_table( bgq_preset_search_entry_t * here )
{
#ifdef DEBUG_BGQ
	//printf( "free_preset_table\n" );
#endif
	int i = 0, j;
	while ( here[i].preset ) {
		for ( j = 0; j < MAX_COUNTER_TERMS; j++ )
			free( here[i].findme[j] );
		free( here[i].operation );
		free( here[i].note );
		i++;
	}
}


static void
free_notes( hwi_dev_notes_t * here )
{
#ifdef DEBUG_BGQ
	//printf( "free_notes\n" );
#endif
	int i = 0;
	while ( here[i].event_code ) {
		free( here[i].dev_note );
		i++;
	}
}




/*******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT **********
 ******************************************************************************/


/*
 * Lock
 */
void
_papi_hwd_lock( int lock ) 
{
#ifdef DEBUG_BGQ
	printf( _AT_ " _papi_hwd_lock %d\n", lock);
#endif
	assert( lock < PAPI_MAX_LOCK );
#if 1
	UPC_Lock( &thdLocks[lock] );
#else
	pthread_mutex_lock( &thdLocks[lock] );
#endif
	
#ifdef DEBUG_BGQ
	printf( _AT_ " _papi_hwd_lock got lock %d\n", lock );
#endif
	
	return;
}

/*
 * Unlock
 */
void
_papi_hwd_unlock( int lock )
{
#ifdef DEBUG_BGQ
    printf( _AT_ " _papi_hwd_unlock %d\n", lock );
#endif
    assert( lock < PAPI_MAX_LOCK );
#if 1
	UPC_Unlock( &thdLocks[lock] );
#else
	pthread_mutex_unlock( &thdLocks[lock] );
#endif
	
	return;
}



/*
 * Update Shared Library Information
 *
 * NOTE:  pid is not set in the _papi_hwi_system_info structure, and thus, the open
 *        of the map file will fail.    We just short circuit this code and return
 *        PAPI_OK.
 */
int
_bgq_update_shlib_info( papi_mdi_t *mdi )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_update_shlib_info\n" );
#endif
	
	( void ) mdi;
	return ( PAPI_OK );
}

/*
 * Get System Information
 *
 * Initialize system information structure
 */
int
_bgq_get_system_info( papi_mdi_t *mdi )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_get_system_info\n" );
#endif
	
	//( void ) mdi;
	Personality_t personality;
	int retval;

	// NOTE:  Executable regions, require reading the /proc/pid/maps file
	//        and the pid is not filled in the system_info structure.
	//        Basically, _bgq_update_shlib_info() simply returns
	//        with PAPI_OK
	_bgq_update_shlib_info( &_papi_hwi_system_info  );
	
	/* Hardware info */
	retval = Kernel_GetPersonality( &personality, sizeof( Personality_t ) );
	if ( retval ) {
		fprintf( stdout, "Kernel_GetPersonality returned %d (sys error=%d).\n"
				"\t%s\n", retval, errno, strerror( errno ) );
		return PAPI_ESYS;
	}

	/* Returns the number of processors that are associated with the currently
	 * running process */
	_papi_hwi_system_info.hw_info.ncpu = Kernel_ProcessorCount( );
	// TODO: HJ Those values need to be fixed
	_papi_hwi_system_info.hw_info.nnodes = Kernel_ProcessCount( );
	_papi_hwi_system_info.hw_info.totalcpus = _papi_hwi_system_info.hw_info.ncpu;
	
	_papi_hwi_system_info.hw_info.mhz = ( float ) personality.Kernel_Config.FreqMHz;
	SUBDBG( "_bgq_get_system_info:  Detected MHZ is %f\n",
		   _papi_hwi_system_info.hw_info.mhz );
	
	return ( PAPI_OK );
}

/*
 * Setup BG/Q Presets
 *
 * Assign the global native and preset table pointers, find the native
 * table's size in memory and then call the preset setup routine.
 */
static inline int
setup_bgq_presets( char *pmu_name, int cpu_type )
{
#ifdef DEBUG_BGQ
	printf( "setup_bgq_presets\n" );
#endif
	
	int retval, npresets;
	hwi_search_t *preset_search_map = NULL;
	hwi_dev_notes_t *notemap = NULL;
	bgq_preset_search_entry_t *_bgq_preset_search_map;
	
	
	/* allocate and clear array of search string structures */
	_bgq_preset_search_map =
	malloc( sizeof ( bgq_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );
	if ( _bgq_preset_search_map == NULL )
		return ( PAPI_ENOMEM );
	memset( _bgq_preset_search_map, 0x0,
		   sizeof ( bgq_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );
	
	npresets = load_preset_table( pmu_name, cpu_type,
							   _bgq_preset_search_map );
	if (npresets < 0) {
		retval = npresets;
		goto out;
	}
	
	retval = generate_preset_search_map( &preset_search_map, &notemap,
										_bgq_preset_search_map, npresets );
	if ( retval )
		goto out;
	
	retval = _papi_hwi_setup_all_presets( preset_search_map, notemap );
out:
	free_preset_table( _bgq_preset_search_map );
	free( _bgq_preset_search_map );
	free( preset_search_map );
	free_notes( notemap );
	free( notemap );
	
	return retval;
}

/*
 * Initialize Control State
 *
 */
int
_bgq_init_control_state( hwd_control_state_t * ptr )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_init_control_state\n" );
#endif
	
	ptr->EventGroup = Bgpm_CreateEventSet();
	CHECK_BGPM_ERROR( ptr->EventGroup, "Bgpm_CreateEventSet" );
	
	// initialize multiplexing flag to OFF (0)
	ptr->muxOn = 0;
	// initialized BGPM eventGroup flag to NOT applied yet (0)
	ptr->bgpm_eventset_applied = 0;
	
	return PAPI_OK;
}


/*
 * Set Domain
 */
int
_bgq_set_domain( hwd_control_state_t * cntrl, int domain )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_set_domain\n" );
#endif
	int found = 0;
	( void ) cntrl;
	
	if ( PAPI_DOM_USER & domain )
		found = 1;
	
	if ( PAPI_DOM_KERNEL & domain )
		found = 1;
	
	if ( PAPI_DOM_OTHER & domain )
		found = 1;
	
	if ( !found )
		return ( PAPI_EINVAL );
	
	return ( PAPI_OK );
}

/*
 * PAPI Initialization
 * This is called whenever a thread is initialized
 */
int
_bgq_init( hwd_context_t * ctx )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_init\n" );
#endif
	( void ) ctx;
	int retval;

#ifdef DEBUG_BGPM
	Bgpm_PrintOnError(1);
	Bgpm_ExitOnError(0);
#else
	Bgpm_PrintOnError(0);
	// avoid bgpm default of exiting when error occurs - caller will check return code instead.
	Bgpm_ExitOnError(0);	
#endif
	
	retval = Bgpm_Init( BGPM_MODE_SWDISTRIB );
	CHECK_BGPM_ERROR( retval, "Bgpm_Init" );

	//_common_initBgpm();
	
	return PAPI_OK;	
}


/*
 * BPT Map Availabiliy
 *
 * This function examines the event to determine if it can be mapped
 * to counter location ctr.  If the counter location is equal to the
 * event id modulo BGQ_PUNIT_MAX_COUNTERS, then the event
 * can be mapped to the specified counter location.
 * Otherwise, the event cannot be mapped.
 */
int
_bgq_bpt_map_avail( hwd_reg_alloc_t * dst, int ctr )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_bpt_map_avail\n" );
#endif
	
	( void ) dst;
	( void ) ctr;
	// TODO: HJ Not sure if this fct is needed
#if 0
	// printf("_bgq_bpt_map_avail: Counter = %d\n", ctr);
	if ( ( int ) get_bgq_native_event_id( dst->id ) %
		 BGQ_PUNIT_MAX_COUNTERS == ctr )
		return ( 1 );
#endif
	return ( 0 );
}



/*
 * BPT Map Exclusive
 *
 * This function examines the event to determine if it has a single
 * exclusive mapping. Since we are only allowing events from
 * user mode 0 and 1, all events have an exclusive mapping.
 * Always returns true.
 */
int
_bgq_bpt_map_exclusive( hwd_reg_alloc_t * dst )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_bpt_map_exclusive\n" );
#endif
	
	( void ) dst;
	
	return ( 1 );
}

/*
 * BPT Map Shared
 *
 * This function compares the dst and src events to determine
 * if any resources are shared. Typically the src event is
 * exclusive, so this detects a conflict if true.
 * Returns true if conflict, false if no conflict.
 * Since we are only allowing events from user mode 0 and 1,
 * all events have an exclusive mapping, and thus, do not
 * share hardware register resources.
 *
 * Always return false, as there are no 'shared' resources.
 */
int
_bgq_bpt_map_shared( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_bpt_map_shared\n" );
#endif
	
	( void ) dst;
	( void ) src;
	
	return ( 0 );
}





int
_bgq_multiplex( hwd_control_state_t * bgq_state )
{
	int retval;
	uint64_t bgpm_period;
	double Sec, Hz;	

#ifdef DEBUG_BGQ
	printf("_bgq_multiplex BEGIN: Num of Events = %d (vs %d)\n", Bgpm_NumEvents( bgq_state->EventGroup ), bgq_state->count );
#endif
	
	// convert Mhz to Hz ( = cycles / sec )
	Hz = (double) _papi_hwi_system_info.hw_info.mhz * 1000 * 1000;
	// convert PAPI multiplex period (in ns) to BGPM period (in cycles)
	Sec = (double) _bgq_vectors.cmp_info.itimer_ns / ( 1000 * 1000 * 1000 );
	bgpm_period = Hz * Sec;

	// if EventGroup is not empty -- which is required by BGPM before 
	// we can call SetMultiplex() -- then drain the events from the 
	// BGPM EventGroup, turn on multiplex flag, and rebuild BGPM EventGroup.
	if ( 0 < bgq_state->count ) {
		// Delete and re-create BGPM eventset
		_common_deleteRecreate( &bgq_state->EventGroup );
		
		// turn on multiplex for BGPM
		retval = Bgpm_SetMultiplex( bgq_state->EventGroup, bgpm_period, BGPM_NORMAL ); 		
		CHECK_BGPM_ERROR( retval, "Bgpm_SetMultiplex" );
		
		// rebuild BGPM EventGroup
		_common_rebuildEventgroup( bgq_state->count, 
								   bgq_state->EventGroup_local, 
								   &bgq_state->EventGroup );		
	}
	else {
		// need to pass either BGPM_NORMAL or BGPM_NOTNORMAL 
		// BGPM_NORMAL: numbers reported by Bgpm_ReadEvent() are normalized 
		// to the maximum time spent in a multiplexed group
		retval = Bgpm_SetMultiplex( bgq_state->EventGroup, bgpm_period, BGPM_NORMAL ); 		
		CHECK_BGPM_ERROR( retval, "Bgpm_SetMultiplex" );				
	}

#ifdef DEBUG_BGQ
	printf("_bgq_multiplex END: Num of Events = %d (vs %d) --- retval = %d\n", 
		   Bgpm_NumEvents( bgq_state->EventGroup ), bgq_state->count, retval );
#endif
	
	return ( retval );
}





/*
 * Register Allocation
 *
 */
int
_bgq_allocate_registers( EventSetInfo_t * ESI )
{
#ifdef DEBUG_BGQ
	printf("_bgq_allocate_registers\n");
#endif
	int i, natNum;
	int xEventId;

	/*
	 * Start monitoring the events...
	 */
	natNum = ESI->NativeCount;

	for ( i = 0; i < natNum; i++ ) {
		xEventId = ( ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK ) + 1;
		ESI->NativeInfoArray[i].ni_position = i;		
	}

	/* NOTE:  For some unknown reason, a successful return from this routine
	          is indicated with a non-zero value...  We choose 1... */
	return 1;
}


/*
 * PAPI Cleanup Eventset
 *
 * Destroy and re-create the BGPM / Punit EventSet
 */
int
_bgq_cleanup_eventset( hwd_control_state_t * ctrl )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_cleanup_eventset\n" );
#endif
	
	// set multiplexing flag to OFF (0)
	ctrl->muxOn = 0;
	// set BGPM eventGroup flag back to NOT applied yet (0)
	ctrl->bgpm_eventset_applied = 0;

	return ( PAPI_OK );
}


/*
 * Update Control State
 *
 * This function clears the current contents of the control
 * structure and updates it with whatever resources are allocated
 * for all the native events in the native info structure array.
 */
int
_bgq_update_control_state( hwd_control_state_t * ptr,
						   NativeInfo_t * native, int count,
						   hwd_context_t * ctx )
{
#ifdef DEBUG_BGQ
	printf( _AT_ " _bgq_update_control_state: count = %d, EventGroup=%d\n", count, ptr->EventGroup );
#endif
	( void ) ctx;
	int i, index, retval;
	
	// Delete and re-create BGPM eventset
	_common_deleteRecreate( &ptr->EventGroup );
	
#ifdef DEBUG_BGQ
    printf( _AT_ " _bgq_update_control_state: EventGroup=%d, muxOn = %d\n", ptr->EventGroup, ptr->muxOn );
#endif
	
	// add the events to the eventset
	for ( i = 0; i < count; i++ ) {
		index = ( native[i].ni_event & PAPI_NATIVE_AND_MASK ) + 1;

#ifdef DEBUG_BGQ
		printf(_AT_ " _bgq_update_control_state: ADD event: i = %d, index = %d\n", i, index );
#endif
		
		ptr->EventGroup_local[i] = index;

		/* Add events to the BGPM eventGroup */
		retval = Bgpm_AddEvent( ptr->EventGroup, index );
		CHECK_BGPM_ERROR( retval, "Bgpm_AddEvent" );			
	}
	
	// store how many events we added to an EventSet
	ptr->count = count;

	// if muxOn and EventGroup is not empty -- which is required by BGPM before 
	// we can call SetMultiplex() -- then drain the events from the 
	// BGPM EventGroup, turn on multiplex flag, and rebuild BGPM EventGroup.
	if ( 1 == ptr->muxOn ) 
		retval = _bgq_multiplex( ptr );
	
	return ( PAPI_OK );
}



/*
 * PAPI Start
 */
int
_bgq_start( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
#ifdef DEBUG_BGQ
	printf( "BEGIN _bgq_start\n" );
#endif
	( void ) ctx;
	int retval;
		
	retval = Bgpm_Apply( ptr->EventGroup ); 
	CHECK_BGPM_ERROR( retval, "Bgpm_Apply" );
	
	// set flag to 1: BGPM eventGroup HAS BEEN applied
	ptr->bgpm_eventset_applied = 1;
	
	/* Bgpm_Apply() does an implicit reset; 
	 hence no need to use Bgpm_ResetStart */
	retval = Bgpm_Start( ptr->EventGroup );
	CHECK_BGPM_ERROR( retval, "Bgpm_Start" );
	
	return ( PAPI_OK );
}

/*
 * PAPI Stop
 */
int
_bgq_stop( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
#ifdef DEBUG_BGQ
	printf( "BEGIN _bgq_stop\n" );
#endif
	( void ) ctx;
	int retval;
	
	retval = Bgpm_Stop( ptr->EventGroup );
	CHECK_BGPM_ERROR( retval, "Bgpm_Stop" );
	
	return ( PAPI_OK );
}

/*
 * PAPI Read Counters
 *
 * Read the counters into local storage
 */
int
_bgq_read( hwd_context_t * ctx, hwd_control_state_t * ptr,
		   long_long ** dp, int flags )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_read\n" );
#endif
	( void ) ctx;
	( void ) flags;
	int i, numEvts;
	
	numEvts = Bgpm_NumEvents( ptr->EventGroup );
	if ( numEvts == 0 ) {
#ifdef DEBUG_BGPM
		printf ("Error: ret value is %d for BGPM API function Bgpm_NumEvents.\n", numEvts );
		//return ( EXIT_FAILURE );
#endif
	}
	
	for ( i = 0; i < numEvts; i++ ) 
		ptr->counters[i] = _common_getEventValue( i, ptr->EventGroup );

	*dp = ptr->counters;
		
	return ( PAPI_OK );
}

/*
 * PAPI Reset
 *
 * Zero the counter values
 */
int
_bgq_reset( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_reset\n" );
#endif
	( void ) ctx;
	int retval;
	
	/* we can't simply call Bgpm_Reset() since PAPI doesn't have the 
	   restriction that an EventSet has to be stopped before resetting is
	   possible. However, BGPM does have this restriction. 
	   Hence we need to stop, reset and start */
	retval = Bgpm_Stop( ptr->EventGroup );
	CHECK_BGPM_ERROR( retval, "Bgpm_Stop" );
	
	retval = Bgpm_ResetStart( ptr->EventGroup );
	CHECK_BGPM_ERROR( retval, "Bgpm_ResetStart" );

	return ( PAPI_OK );
}


/*
 * PAPI Shutdown
 *
 * This routine is for shutting down threads,
 * including the master thread.
 * Effectively a no-op, same as BG/L/P...
 */
int
_bgq_shutdown( hwd_context_t * ctx )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_shutdown\n" );
#endif
	( void ) ctx;
	int retval;
	
	/* Disable BGPM library */	
	retval = Bgpm_Disable();
	CHECK_BGPM_ERROR( retval, "Bgpm_Disable" );

	return ( PAPI_OK );
	
}

/*
 * PAPI Write
 *
 * Write counter values
 * NOTE:  Could possible support, but signal error as BG/L/P does...
 */
int
_bgq_write( hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_write\n" );
#endif
	( void ) ctx;
	( void ) cntrl;
	( void ) from;
	
	return ( PAPI_ESBSTR );
}

/*
 * Dispatch Timer
 *
 * NOT the same as BG/L/P where we simply return
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */
void
_bgq_dispatch_timer( int signal, hwd_siginfo_t * info, void *uc )
{
#ifdef DEBUG_BGQ
	printf("BEGIN _bgq_dispatch_timer\n");
#endif
	
	return;
}



/*
 * user_signal_handler
 *
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */
void
user_signal_handler( int hEvtSet, uint64_t address, uint64_t ovfVector, const ucontext_t *pContext )
{
#ifdef DEBUG_BGQ
	printf( "user_signal_handler start\n" );
#endif
	
	int retval, i;
	int isHardware = 1;
	int cidx = _bgq_vectors.cmp_info.CmpIdx;
	long_long overflow_bit = 0;
	caddr_t address1;
	_papi_hwi_context_t ctx;
	ctx.ucontext = ( hwd_ucontext_t * ) pContext;
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
	
	//printf(_AT_ " thread = %p\n", thread);	// <<<<<<<<<<<<<<<<<<
	
	EventSetInfo_t *ESI;
	ESI = thread->running_eventset[cidx];
    // Get the indices of all events which have overflowed.
    unsigned ovfIdxs[BGPM_MAX_OVERFLOW_EVENTS];
    unsigned len = BGPM_MAX_OVERFLOW_EVENTS;
	
    retval = Bgpm_GetOverflowEventIndices( hEvtSet, ovfVector, ovfIdxs, &len );

	if ( retval < 0 ) {
#ifdef DEBUG_BGPM
		printf ( "Error: ret value is %d for BGPM API function Bgpm_GetOverflowEventIndices.\n", 
				 retval ); 
#endif
		return;
	}
		
	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in user_signal_handler!" );
		return;
	}
		
	if ( ESI == NULL ) {
		PAPIERROR( "ESI == NULL in user_signal_handler!");
		return;
	}
		
	if ( ESI->overflow.flags == 0 ) {
		PAPIERROR( "ESI->overflow.flags == 0 in user_signal_handler!");
		return;
	}

	for ( i = 0; i < len; i++ ) {
		uint64_t hProf;
        Bgpm_GetEventUser1( hEvtSet, ovfIdxs[i], &hProf );
        if ( hProf ) {
			overflow_bit ^= 1 << ovfIdxs[i];
			break;
        }
		
	}
	
	if ( ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) {
#ifdef DEBUG_BGQ
		printf("OVERFLOW_SOFTWARE\n");
#endif
		address1 = GET_OVERFLOW_ADDRESS( ctx );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address1, NULL, 0, 0, &thread, cidx );
		return;
	}
	else if ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) {
#ifdef DEBUG_BGQ
		printf("OVERFLOW_HARDWARE\n");
#endif
		address1 = GET_OVERFLOW_ADDRESS( ctx );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address1, &isHardware, overflow_bit, 0, &thread, cidx );
	}
	else {
#ifdef DEBUG_BGQ
		printf("OVERFLOW_NONE\n");
#endif
		PAPIERROR( "ESI->overflow.flags is set to something other than PAPI_OVERFLOW_HARDWARE or PAPI_OVERFLOW_FORCE_SW (%x)", thread->running_eventset[cidx]->overflow.flags);
	}
}


/*
 * Set Overflow
 *
 * This is commented out in BG/L/P - need to explore and complete...
 * However, with true 64-bit counters in BG/Q and all counters for PAPI
 * always starting from a true zero (we don't allow write...), the possibility
 * for overflow is remote at best...
 */
int
_bgq_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
#ifdef DEBUG_BGQ
	printf("BEGIN _bgq_set_overflow\n");
#endif
	hwd_control_state_t * this_state = ( hwd_control_state_t * ) ESI->ctl_state;
	int retval;
	int evt_idx;
	uint64_t threshold_for_bgpm;
	
	/*
	 * In case an BGPM eventGroup HAS BEEN applied or attached before
	 * overflow is set, delete the eventGroup and create an new empty one,
	 * and rebuild as it was prior to deletion
	 */
#ifdef DEBUG_BGQ
	printf( "_bgq_set_overflow: bgpm_eventset_applied = %d\n",
		    this_state->bgpm_eventset_applied );
#endif	
	if ( 1 == this_state->bgpm_eventset_applied ) {
		_common_deleteRecreate( &this_state->EventGroup );
		_common_rebuildEventgroup( this_state->count,
								   this_state->EventGroup_local,
								   &this_state->EventGroup );
		
		/* set BGPM eventGroup flag back to NOT applied yet (0) 
		 * because the eventGroup has been recreated from scratch */
		this_state->bgpm_eventset_applied = 0;
	}
	
	/* convert threadhold value assigned by PAPI user to value that is
	 * programmed into the counter. This value is required by Bgpm_SetOverflow() */ 
	threshold_for_bgpm = BGPM_PERIOD2THRES( threshold );
	
	evt_idx = ESI->EventInfoArray[EventIndex].pos[0];
	//evt_id = ( ESI->NativeInfoArray[EventIndex].ni_event & PAPI_NATIVE_AND_MASK ) + 1;
	SUBDBG( "Hardware counter %d (vs %d) used in overflow, threshold %d\n",
		    evt_idx, EventIndex, threshold );
#ifdef DEBUG_BGQ
	printf( "Hardware counter %d (vs %d) used in overflow, threshold %d\n",
		    evt_idx, EventIndex, threshold );
#endif
	/* If this counter isn't set to overflow, it's an error */
	if ( threshold == 0 ) {
		/* Remove the signal handler */
		retval = _papi_hwi_stop_signal( _bgq_vectors.cmp_info.hardware_intr_sig );
		if ( retval != PAPI_OK )
			return ( retval );
	}
	else {
#ifdef DEBUG_BGQ
		printf( "_bgq_set_overflow: Enable the signal handler\n" );
#endif		
		/* Enable the signal handler */
		retval = _papi_hwi_start_signal( _bgq_vectors.cmp_info.hardware_intr_sig, 
										 NEED_CONTEXT, 
										 _bgq_vectors.cmp_info.CmpIdx );
		if ( retval != PAPI_OK )
			return ( retval );
		
		retval = Bgpm_SetOverflow( this_state->EventGroup, 
								   evt_idx,
								   threshold_for_bgpm );

		CHECK_BGPM_ERROR( retval, "Bgpm_SetOverflow" );
		
        retval = Bgpm_SetEventUser1( this_state->EventGroup, 
									 evt_idx,
									 1024 );
		CHECK_BGPM_ERROR( retval, "Bgpm_SetEventUser1" );

		/* user signal handler for overflow case */
		retval = Bgpm_SetOverflowHandler( this_state->EventGroup, user_signal_handler );
		CHECK_BGPM_ERROR( retval, "Bgpm_SetOverflowHandler" );
	}

	return ( PAPI_OK );
}


/*
 * Set Profile
 *
 * Same as for BG/L/P, routine not used and returns error
 */
int
_bgq_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
#ifdef DEBUG_BGQ
	printf("BEGIN _bgq_set_profile\n");
#endif
	
	( void ) ESI;
	( void ) EventIndex;
	( void ) threshold;
	
	return ( PAPI_ESBSTR );
}

/*
 * Stop Profiling
 *
 * Same as for BG/L/P...
 */
int
_bgq_stop_profiling( ThreadInfo_t * master, EventSetInfo_t * ESI )
{
#ifdef DEBUG_BGQ
	printf("BEGIN _bgq_stop_profiling\n");
#endif
	
	( void ) master;
	( void ) ESI;
	
	return ( PAPI_OK );
}

/*
 * PAPI Control
 *
 * Same as for BG/L/P - initialize the domain
 */
int
_bgq_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_ctl\n" );
#endif

	int retval;
	
	switch ( code ) {
		case PAPI_MULTIPLEX:
		{
			hwd_control_state_t * bgq_state = ( ( hwd_control_state_t * ) option->multiplex.ESI->ctl_state );
			bgq_state->muxOn = 1;
			retval = _bgq_multiplex( bgq_state );
			return ( retval );
		}
		default:
			return ( PAPI_OK );
	}
}

/*
 * Get Real Micro-seconds
 */
long long
_bgq_get_real_usec( void )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_get_real_usec\n" );
#endif
	
	/*
	 * NOTE:  _papi_hwi_system_info.hw_info.mhz is really a representation of unit of time per cycle.
	 *        On BG/P, it's value is 8.5e-4.  Therefore, to get cycles per sec, we have to multiply
	 *        by 1.0e12.  To then convert to usec, we have to divide by 1.0e-3.
	 */
	return ( ( long long ) ( ( ( float ) get_cycles(  ) ) /
							 ( ( _papi_hwi_system_info.hw_info.mhz ) ) ) );

}

/*
 * Get Real Cycles
 *
 * Same for BG/L/P, using native function...
 */
long long
_bgq_get_real_cycles( void )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_get_real_cycles\n" );
#endif

	return ( ( long long ) get_cycles(  ) );

}

/*
 * Get Virtual Micro-seconds
 *
 * Same calc as for BG/L/P, returns real usec...
 */
long long
_bgq_get_virt_usec( hwd_context_t * zero )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_get_virt_usec\n" );
#endif
	
	( void ) zero;
	
	return _bgq_get_real_usec(  );
}

/*
 * Get Virtual Cycles
 *
 * Same calc as for BG/L/P, returns real cycles...
 */
long long
_bgq_get_virt_cycles( hwd_context_t * zero )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_get_virt_cycles\n" );
#endif
	
	( void ) zero;
	
	return _bgq_get_real_cycles(  );
}

/*
 * Substrate setup and shutdown
 *
 * Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_bgq_init_substrate( int cidx )
{	
#ifdef DEBUG_BGQ
	printf("_bgq_init_substrate\n");
	//printf("_bgq_init_substrate: 1. BGPM_INITIALIZED = %d \n", BGPM_INITIALIZED);
#endif
	int retval;
	int i;

	_bgq_vectors.cmp_info.CmpIdx = cidx;
	
	/*
	 * Fill in what we can of the papi_system_info
	 */
	SUBDBG( "Before _bgq_get_system_info()...\n" );
	retval = _bgq_get_system_info( &_papi_hwi_system_info );
	SUBDBG( "After _bgq_get_system_info(), retval=%d...\n", retval );
	if ( retval != PAPI_OK )
		return ( retval );

	/*
	 * Setup presets
	 */
	SUBDBG( "Before setup_bgq_presets...\n" );
	retval = setup_bgq_presets( "BGQ", 0 );
	SUBDBG( "After setup_bgq_presets, retval=%d...\n", retval );
	if ( retval )
		return ( retval );
	
	/*
	 * Setup memory info
	 */

	SUBDBG( "Before _bgq_get_memory_info...\n" );
	retval = _bgq_get_memory_info( &_papi_hwi_system_info.hw_info,
								  ( int ) _papi_hwi_system_info.hw_info.
								  model );
	SUBDBG( "After _bgq_get_memory_info, retval=%d...\n", retval );
	if ( retval )
		return ( retval );

#if 1
	/* Setup Locks */
	for ( i = 0; i < PAPI_MAX_LOCK; i++ )
		thdLocks[i] = 0;  // MUTEX_OPEN
#else
	for( i = 0; i < PAPI_MAX_LOCK; i++ ) {
		pthread_mutex_init( &thdLocks[i], NULL );
	}
#endif
	
	return ( PAPI_OK );
}


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/*
 * Event Name to Native Code
 */
int
_bgq_ntv_name_to_code( char *name, unsigned int *event_code )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_ntv_name_to_code\n" );
#endif
	int ret;
	
	/* Return event id matching a given event label string */
	ret = Bgpm_GetEventIdFromLabel ( name );
	
	if ( ret <= 0 ) {
#ifdef DEBUG_BGPM
		printf ("Error: ret value is %d for BGPM API function '%s'.\n",
				ret, "Bgpm_GetEventIdFromLabel" );
#endif
		return PAPI_ENOEVNT;
	}
	else if ( ret > BGQ_PUNIT_MAX_COUNTERS ) // not a PUnit event
		return PAPI_ENOEVNT;
	else
		*event_code = ( ret - 1 ) | PAPI_NATIVE_MASK;
	
	return PAPI_OK;
}


/*
 * Native Code to Event Name
 *
 * Given a native event code, returns the short text label
 */
int
_bgq_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{	
#ifdef DEBUG_BGQ	
	//printf( "_bgq_ntv_code_to_name\n" );
#endif
	int index = ( EventCode & PAPI_NATIVE_AND_MASK ) + 1;
	
	if ( index >= MAX_COUNTERS )
		return PAPI_ENOEVNT;
			
	strncpy( name, Bgpm_GetEventIdLabel( index ), len );
	
	if ( name == NULL ) {
#ifdef DEBUG_BGPM
		printf ("Error: ret value is NULL for BGPM API function Bgpm_GetEventIdLabel.\n" );
#endif
		return PAPI_ENOEVNT;
	}
	
	return ( PAPI_OK );
}

/*
 * Native Code to Event Description
 *
 * Given a native event code, returns the longer native event description
 */
int
_bgq_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{	
#ifdef DEBUG_BGQ
	//printf( "_bgq_ntv_code_to_descr\n" );
#endif
	int retval;
	int index = ( EventCode & PAPI_NATIVE_AND_MASK ) + 1;

	retval = Bgpm_GetLongDesc( index, name, &len );
	CHECK_BGPM_ERROR( retval, "Bgpm_GetLongDesc" );						 

	return ( PAPI_OK );
}

/*
 * Native Code to Bit Configuration
 *
 * Given a native event code, assigns the native event's
 * information to a given pointer.
 * NOTE: The info must be COPIED to location addressed by
 *       the provided pointer, not just referenced!
 * NOTE: For BG/Q, the bit configuration is not needed,
 *       as the native SPI is used to configure events.
 */
int
_bgq_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
#ifdef DEBUG_BGQ
	printf( "_bgq_ntv_code_to_bits\n" );
#endif
	
	( void ) EventCode;
	( void ) bits;
	
	return ( PAPI_OK );
}

/*
 * Native ENUM Events
 *
 */
int
_bgq_ntv_enum_events( unsigned int *EventCode, int modifier )
{
#ifdef DEBUG_BGQ
	//printf( "_bgq_ntv_enum_events\n" );
#endif
	
	switch ( modifier ) {
		case PAPI_ENUM_FIRST:
			*EventCode = PAPI_NATIVE_MASK;
			
			return ( PAPI_OK );
			break;
			
		case PAPI_ENUM_EVENTS:
		{
			int index = ( *EventCode & PAPI_NATIVE_AND_MASK ) + 1;
			
			if ( index < BGQ_PUNIT_MAX_COUNTERS ) {
				*EventCode = *EventCode + 1;
				return ( PAPI_OK );
			} else
				return ( PAPI_ENOEVNT );
			
			break;
		}
		default:
			return ( PAPI_EINVAL );
	}
	
	return ( PAPI_EINVAL );	
}

/*
 * Native Bit Configuration to Information
 *
 * No-op for BG/Q and simply returns 0
 */
int
_bgq_ntv_bits_to_info( hwd_register_t * bits, char *names,
					   unsigned int *values, int name_len, int count )
{
	( void ) bits;
	( void ) names;
	( void ) values;
	( void ) name_len;
	( void ) count;
	
	return ( 0 );
}

/*
 * PAPI Vector Table for BG/Q
 */
papi_vector_t _bgq_vectors = {
	.cmp_info = {
				 /* Default component information (unspecified values are initialized to 0) */
				 .name = "$Id: ",
				 .CmpIdx = 0, 
				 .num_cntrs = BGQ_PUNIT_MAX_COUNTERS,
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .itimer_sig = PAPI_INT_MPX_SIGNAL,
				 .itimer_num = PAPI_INT_ITIMER,
				 .itimer_res_ns = 1,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,
				 .hardware_intr = 1,
				 .kernel_multiplex = 1,
		
				 /* component specific cmp_info initializations */
				 .fast_real_timer = 1,
				 .fast_virtual_timer = 0,
				 .itimer_ns = PAPI_INT_MPX_DEF_US * 1000,
				 }
	,

	/* Sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( hwd_context_t ),
			 .control_state = sizeof ( hwd_control_state_t ),
			 .reg_value = sizeof ( hwd_register_t ),
			 .reg_alloc = sizeof ( hwd_reg_alloc_t ),
			 }
	,
	/* Function pointers in this component */
//   .get_overflow_address =
	.start = _bgq_start,
	.stop = _bgq_stop,
	.read = _bgq_read,
	.reset = _bgq_reset,
	.write = _bgq_write,
	.get_real_cycles = _bgq_get_real_cycles,
	.get_real_usec = _bgq_get_real_usec,
	.get_virt_cycles = _bgq_get_virt_cycles,
	.get_virt_usec = _bgq_get_virt_usec,
	.stop_profiling = _bgq_stop_profiling,
	.init_substrate = _bgq_init_substrate,
	.init = _bgq_init,
	.init_control_state = _bgq_init_control_state,
	.update_shlib_info = _bgq_update_shlib_info,
	.get_system_info = _bgq_get_system_info,
	.get_memory_info = _bgq_get_memory_info,
	.update_control_state = _bgq_update_control_state,
	.ctl = _bgq_ctl,
	.set_overflow = _bgq_set_overflow,
	//.dispatch_timer = _bgq_dispatch_timer,
	.set_profile = _bgq_set_profile,
	.set_domain = _bgq_set_domain,
	.ntv_enum_events = _bgq_ntv_enum_events,
	.ntv_name_to_code = _bgq_ntv_name_to_code,
	.ntv_code_to_name = _bgq_ntv_code_to_name,
	.ntv_code_to_descr = _bgq_ntv_code_to_descr,
	.ntv_code_to_bits = _bgq_ntv_code_to_bits,
	.ntv_bits_to_info = _bgq_ntv_bits_to_info,
	.allocate_registers = _bgq_allocate_registers,
	.bpt_map_avail = _bgq_bpt_map_avail,
	.bpt_map_exclusive = _bgq_bpt_map_exclusive,
	.bpt_map_shared = _bgq_bpt_map_shared,
	.get_dmem_info = _bgq_get_dmem_info,
	.cleanup_eventset = _bgq_cleanup_eventset,
	.shutdown = _bgq_shutdown
//  .shutdown_global      =
//  .user                 =
};
