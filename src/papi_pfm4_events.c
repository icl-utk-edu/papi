/*
* File:    papi_pfm4_events.c
* CVS:     $Id$
* Author:  Dan Terpstra: blantantly extracted from Phil's perfmon.c
*          mucci@cs.utk.edu
*/

/* TODO LIST:
    - Events for all platforms
xxx - Derived events for all platforms
xxx - hwd_ntv_name_to_code
xxx - Make native map carry major events, not umasks
xxx - Enum event uses native_map not pfm()
xxx - bits_to_info uses native_map not pfm()
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_pfm4_events.h"

/* these define cccr and escr register bits, and the p4 event structure */

// find the equivalent in pfm4...
//#include "perfmon/pfmlib_pentium4.h"
//#include "../lib/pfmlib_pentium4_priv.h"

#define P4_REPLAY_REAL_MASK 0x00000003

// find the equivalent in pfm4...
//extern pentium4_escr_reg_t pentium4_escrs[];
//extern pentium4_cccr_reg_t pentium4_cccrs[];
//extern pentium4_event_t pentium4_events[];

extern unsigned char PENTIUM4;
extern papi_vector_t MY_VECTOR;
extern unsigned int PAPI_NATIVE_EVENT_AND_MASK;
extern unsigned int PAPI_NATIVE_EVENT_SHIFT;
extern unsigned int PAPI_NATIVE_UMASK_AND_MASK;
extern unsigned int PAPI_NATIVE_UMASK_MAX;
extern unsigned int PAPI_NATIVE_UMASK_SHIFT;

extern int _papi_pfm4_ntv_code_to_bits( unsigned int EventCode,
									   hwd_register_t * bits );
extern int _papi_pfm4_ntv_bits_to_info( hwd_register_t * bits, char *names,
									   unsigned int *values, int name_len,
									   int count );

/* Globals declared extern elsewhere */

volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

/* NOTE: PAPI stores umask info in a variable sized (16 bit?) bitfield.
    Perfmon2 stores umask info in a large (48 element?) array of values.
    Native event encodings for perfmon2 contain array indices
    encoded as bits in this bitfield. These indices must be converted
    into a umask value before programming the counters. For Perfmon,
    this is done by converting back to an array of values; for 
    perfctr, it must be done by looking up the values.
*/

/* This routine is used to step through all possible combinations of umask
    values. It assumes that mask contains a valid combination of array indices
    for this event. */
static inline int
encode_native_event_raw( unsigned int event, unsigned int mask )
{
	unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
	SUBDBG( "Old native index was 0x%08x with 0x%08x mask\n", tmp, mask );
	tmp = tmp | ( mask << PAPI_NATIVE_UMASK_SHIFT );
	SUBDBG( "New encoding is 0x%08x\n", tmp | PAPI_NATIVE_MASK );
	return ( int ) ( tmp | PAPI_NATIVE_MASK );
}

/* This routine converts array indices contained in the mask_values array
    into bits in the umask field that is OR'd into the native event code.
    These bits are NOT the mask values themselves, but indices into an array
    of mask values contained in the native event table. */
static inline int
encode_native_event( unsigned int event, unsigned int num_mask,
					 unsigned int *mask_values )
{
	unsigned int i;
	unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
	SUBDBG( "Native base event is 0x%08x with %d masks\n", tmp, num_mask );
	for ( i = 0; i < num_mask; i++ ) {
		SUBDBG( "Mask index is 0x%08x\n", mask_values[i] );
		tmp = tmp | ( ( 1 << mask_values[i] ) << PAPI_NATIVE_UMASK_SHIFT );
	}
	SUBDBG( "Full native encoding is 0x%08x\n", tmp | PAPI_NATIVE_MASK );
	return ( int ) ( tmp | PAPI_NATIVE_MASK );
}

/*  Trims blank space from both ends of a string (in place).
    Returns pointer to new start address */
static inline char *
trim_string( char *in )
{
	int len, i = 0;
	char *start = in;

	if ( in == NULL )
		return ( in );
	len = ( int ) strlen( in );
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
	i = ( int ) strlen( start ) - 1;
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
static inline char *
trim_note( char *in )
{
	int len;
	char *note, start, end;

	note = trim_string( in );
	if ( note != NULL ) {
		len = ( int ) strlen( note );
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

static inline int
find_preset_code( char *tmp, int *code )
{
	int i = 0;
	extern hwi_presets_t _papi_hwi_presets;
	while ( _papi_hwi_presets.info[i].symbol != NULL ) {
		if ( strcasecmp( tmp, _papi_hwi_presets.info[i].symbol ) == 0 ) {
			*code = ( int ) ( i | PAPI_PRESET_MASK );
			return ( PAPI_OK );
		}
		i++;
	}
	return ( PAPI_EINVAL );
}

/* Look for an event file 'name' in a couple common locations.
   Return a valid file handle if found */
static FILE *
open_event_table( char *name )
{
	FILE *table;

	SUBDBG( "Opening %s\n", name );
	table = fopen( name, "r" );
	if ( table == NULL ) {
		SUBDBG( "Open %s failed, trying ./%s.\n", name, PAPI_EVENT_FILE );
		sprintf( name, "%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table == NULL ) {
		SUBDBG( "Open ./%s failed, trying ../%s.\n", name, PAPI_EVENT_FILE );
		sprintf( name, "../%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table )
		SUBDBG( "Open %s succeeded.\n", name );
	return ( table );
}

/* parse a single line from either a file or character table
   Strip trailing <cr>; return 0 if empty */
static int
get_event_line( char *line, FILE * table, char **tmp_perfmon_events_table )
{
	int ret;
	int i;

	if ( table ) {
		if ( fgets( line, LINE_MAX, table ) ) {
			ret = 1;
			i = ( int ) strlen( line );
			if ( line[i - 1] == '\n' )
				line[i - 1] = '\0';
		} else
			ret = 0;
	} else {
		for ( i = 0;
			  **tmp_perfmon_events_table && **tmp_perfmon_events_table != '\n';
			  i++ ) {
			line[i] = **tmp_perfmon_events_table;
			( *tmp_perfmon_events_table )++;
		}
		if ( **tmp_perfmon_events_table == '\n' ) {
			( *tmp_perfmon_events_table )++;
		}
		line[i] = '\0';
		ret = **tmp_perfmon_events_table;
	}
	return ( ret );
}

/* Static version of the events file. */
#if defined(STATIC_PAPI_EVENTS_TABLE)
#include "papi_events_table.h"
#else
static char *papi_events_table = NULL;
#endif

/* #define SHOW_LOADS */
#define SHOW_LOADS

static int
load_preset_table( char *pmu_str, int pmu_type,
				   pfm_preset_search_entry_t * here )
{
//	pfm_event_info_t event;
	char pmu_name[PAPI_MIN_STR_LEN];
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

	/* copy the pmu identifier, stripping commas if found */
	tmpn = pmu_name;
	while ( *pmu_str ) {
		if ( *pmu_str != ',' )
			*tmpn++ = *pmu_str;
		pmu_str++;
	}
	*tmpn = '\0';

	/* try the environment variable first */
	if ( ( tmpn = getenv( "PAPI_PERFMON_EVENT_FILE" ) ) &&
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

//#ifdef SHOW_LOADS
				SUBDBG( "Found CPU %s at line %d of %s.\n", t, line_no, name );
//#endif
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
				} else {
#ifdef SHOW_LOADS
					SUBDBG( "Additional qualifier match failed %d vs %d.\n",
							pmu_type, type );
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
				PAPIERROR
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
			SUBDBG( "# events inserted: --%d-- \n", insert );
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
	return ( PAPI_OK );
}

/* Frees memory for all the strdup'd char strings in a preset string array.
    Assumes the array is initialized to 0 and has at least one 0 entry at the end.
    free()ing a NULL pointer is a NOP. */
static void
free_preset_table( pfm_preset_search_entry_t * here )
{
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
free_native( hwi_search_t * here )
{
	int i = 0, j;
	while ( here[i].preset_code ) {
		for ( j = 0; j < MAX_COUNTER_TERMS; j++ ) {
			if ( here[i].data.native[j] == NULL) break;
			free( here[i].data.native[j] );
		}
		i++;
	}
}

static void
free_notes( hwi_dev_notes_t * here )
{
	int i = 0;
	while ( here[i].preset_code ) {
		free( here[i].dev_note );
		i++;
	}
}
#define MAX_ENCODING 8
static int
generate_preset_search_map( hwi_search_t ** maploc, hwi_dev_notes_t ** noteloc,
							pfm_preset_search_entry_t * strmap )
{
	int ret = PAPI_OK;
	int k = 0, term;
	unsigned int i = 0, j = 0;
	hwi_search_t *psmap;
	hwi_dev_notes_t *notemap;
	int event_idx, code_count = MAX_ENCODING;
	uint64_t *code_ptr, code[MAX_ENCODING];

	/* Count up the proposed presets */
	while ( strmap[i].preset )
		i++;
	SUBDBG( "generate_preset_search_map(%p,%p,%p) %d proposed presets\n",
			maploc, noteloc, strmap, i );
	i++;

	/* Add null entry */
	psmap = ( hwi_search_t * ) malloc( i * sizeof ( hwi_search_t ) );
	notemap = ( hwi_dev_notes_t * ) malloc( i * sizeof ( hwi_dev_notes_t ) );
	if ( ( psmap == NULL ) || ( notemap == NULL ) )
		return ( PAPI_ENOMEM );
	memset( psmap, 0x0, i * sizeof ( hwi_search_t ) );
	memset( notemap, 0x0, i * sizeof ( hwi_dev_notes_t ) );

	i = 0;
	code_ptr = &code[0];
	while ( strmap[i].preset ) {
		/* Handle derived events */
		term = 0;
		do {
			SUBDBG( "pfm_get_event_encoding(%s)\n", strmap[i].findme[term] );
			code_count = MAX_ENCODING;
			if ( ( ret =
				pfm_get_event_encoding( strmap[i].findme[term], PFM_PLM3,
										NULL, &event_idx, &code_ptr,
										&code_count ) ) == PFM_SUCCESS ) {
				psmap[j].data.native[term] = strdup( strmap[i].findme[term] );
				if ( psmap[j].data.native[term] != NULL ) {
					term++;
				} else {
					PAPIERROR( "No Memory" );
					break;
				}
			} else {
				fprintf(stderr,"ret: %d\n",ret);
				PAPIERROR( "pfm_get_event_encoding(%s): %s",
					strmap[i].findme[term], pfm_strerror( ret ) );
				term++;
			}
		} while ( strmap[i].findme[term] != NULL && term < MAX_COUNTER_TERMS );

		/* terminate the native term array with NULL */
		if ( term < MAX_COUNTER_TERMS )
			psmap[j].data.native[term] = NULL;

		if ( ret == PAPI_OK ) {
			psmap[j].preset_code = ( unsigned int ) strmap[i].preset;
			psmap[j].data.derived = strmap[i].derived;
			if ( strmap[i].derived == DERIVED_POSTFIX ) {
				strncpy( psmap[j].data.operation, strmap[i].operation,
					PAPI_MIN_STR_LEN );
			}
			if ( strmap[i].note ) {
				notemap[k].preset_code = ( unsigned int ) strmap[i].preset;
				notemap[k].dev_note = strdup( strmap[i].note );
				k++;
			}
			j++;
		}
		i++;
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

/* Break a PAPI native event code into its composite event code and pfm mask bits */
inline int
_pfm_decode_native_event( unsigned int EventCode, unsigned int *event,
						  unsigned int *umask )
{
	unsigned int tevent, major, minor;

	tevent = EventCode & PAPI_NATIVE_AND_MASK;
	major = ( tevent & PAPI_NATIVE_EVENT_AND_MASK ) >> PAPI_NATIVE_EVENT_SHIFT;
	if ( ( int ) major >= MY_VECTOR.cmp_info.num_native_events )
		return ( PAPI_ENOEVNT );

	minor = ( tevent & PAPI_NATIVE_UMASK_AND_MASK ) >> PAPI_NATIVE_UMASK_SHIFT;
	*event = major;
	*umask = minor;
	SUBDBG( "EventCode 0x%08x is event %d, umask 0x%x\n", EventCode, major,
			minor );
	return ( PAPI_OK );
}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
static inline int
prepare_umask( unsigned int foo, unsigned int *values )
{
	unsigned int tmp = foo, i;
	int j = 0;

	SUBDBG( "umask 0x%x\n", tmp );
	while ( ( i = ( unsigned int ) ffs( ( int ) tmp ) ) ) {
		tmp = tmp ^ ( 1 << ( i - 1 ) );
		values[j] = i - 1;
		SUBDBG( "umask %d is %d\n", j, values[j] );
		j++;
	}
	return ( j );
}

/* convert the mask values in a pfm event structure into a PAPI unit mask */
//static inline unsigned int
//convert_pfm_masks( pfmlib_event_t * gete )
//{
//	int ret;
//	unsigned int i, code, tmp = 0;
//
//	for ( i = 0; i < gete->num_masks; i++ ) {
//		if ( ( ret =
//			   pfm_get_event_mask_code( gete->event, gete->unit_masks[i],
//										&code ) ) == PFM_SUCCESS ) {
//			SUBDBG( "Mask value is 0x%08x\n", code );
//			tmp |= code;
//		} else {
//			PAPIERROR( "pfm_get_event_mask_code(0x%x,%d,%p): %s", gete->event,
//					   i, &code, pfm_strerror( ret ) );
//		}
//	}
//	return ( tmp );
//}

/* convert an event code and pfm unit mask into a PAPI unit mask */
//inline unsigned int
//_pfm_convert_umask( unsigned int event, unsigned int umask )
//{
//	pfmlib_event_t gete;
//	memset( &gete, 0, sizeof ( gete ) );
//	gete.event = event;
//	gete.num_masks = ( unsigned int ) prepare_umask( umask, gete.unit_masks );
//	return ( convert_pfm_masks( &gete ) );
//}

int
_papi_pfm4_setup_presets( char *pmu_name, int pmu_type )
{
	int retval;
	hwi_search_t *preset_search_map = NULL;
	hwi_dev_notes_t *notemap = NULL;
	pfm_preset_search_entry_t *_pfm4_preset_search_map;

	/* allocate and clear array of search string structures */
	_pfm4_preset_search_map =
		malloc( sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );
	if ( _pfm4_preset_search_map == NULL )
		return ( PAPI_ENOMEM );
	memset( _pfm4_preset_search_map, 0x0,
			sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );

	retval =
		load_preset_table( pmu_name, pmu_type,
						   _pfm4_preset_search_map );
	if ( retval )
		return ( retval );

	retval =
		generate_preset_search_map( &preset_search_map, &notemap,
									_pfm4_preset_search_map );
	free_preset_table(_pfm4_preset_search_map );
	free( _pfm4_preset_search_map );
	if ( retval )
		return ( retval );

	retval = _papi_hwi_setup_all_presets( preset_search_map, notemap );
	if ( retval ) {
		free_native( preset_search_map );
		free( preset_search_map );
		free_notes( notemap );
		free( notemap );
		return ( retval );
	}
	return ( PAPI_OK );
}

int
_papi_pfm4_init(  )
{
	int retval;
	unsigned int ncnt;

	/* Opened once for all threads. */
	SUBDBG( "pfm_initialize()\n" );
	if ( ( retval = pfm_initialize(  ) ) != PFM_SUCCESS ) {
		PAPIERROR( "pfm_initialize(): %s", pfm_strerror( retval ) );
		return ( PAPI_ESBSTR );
	}

	/* Fill in MY_VECTOR.cmp_info.num_native_events */

	SUBDBG( "pfm_get_nevents(%p)\n", &ncnt );
	if ( ( ncnt = pfm_get_nevents(  ) ) != 0 ) {
		PAPIERROR( "pfm_get_nevents()" );
		return ( PAPI_ESBSTR );
	}
	SUBDBG( "pfm_get_nevents() returns: %d\n", ncnt );
	MY_VECTOR.cmp_info.num_native_events = ( int ) ncnt;
	return ( PAPI_OK );
}

int
_papi_pfm4_ntv_name_to_descr( const char *EventName, char *ntv_descr, int len )
{
	int event_idx;
	pfm_event_info_t info;
	pfm_event_attr_info_t attr_info;
	int i, ret;

	event_idx = pfm_find_event(EventName);
	if (event_idx < 0) {
		PAPIERROR( "pfm_find_event(%s): %s", EventName, pfm_strerror( event_idx ) );
		return PAPI_ENOEVNT;
	}

	memset( &info, 0, sizeof ( info ) );

	if ( (ret = pfm_get_event_info( event_idx, &info )) != PFM_SUCCESS  ) {
		PAPIERROR( "pfm_get_event_info(%d): %s", event_idx, pfm_strerror( ret ) );
		return PAPI_ENOEVNT;
	}
	
	strncpy ( ntv_descr, info.desc, len);
	if ( info.nattrs ) strncat ( ntv_descr, ", Attributes:/n    ", len);
	for ( i=0; i<info.nattrs; i++ ) {
		if ( (ret = pfm_get_event_attr_info( event_idx, i, &attr_info )) != PFM_SUCCESS  ) {
			PAPIERROR( "pfm_get_event_attr_info(%d, %d): %s", event_idx, i, pfm_strerror( ret ) );
			return PAPI_EINVAL;
		}
		strncat ( ntv_descr, attr_info.name, len );
		strncat ( ntv_descr, ": ", len );
		strncat ( ntv_descr, attr_info.desc, len );
		strncat ( ntv_descr, "/n    ", len );
	}
	return PAPI_OK;
}

static int
pfm4_event_name(int event, char *EventName, int len)
{
	pfm_event_info_t info;
	int ret;

	if (event == -1) return PAPI_ENOEVNT;
	ret = pfm_get_event_info(event, &info);
	if (ret != PFM_SUCCESS) {
		PAPIERROR( "pfm_get_event_info(%d,%p): %s", event, &info,
					pfm_strerror( ret ) );
		return PAPI_EINVAL;
	}
	strncpy(EventName, info.name, len);
	return PAPI_OK;
}

int
_papi_pfm4_ntv_enum_named_events( char *EventName, int len, int modifier )
{
	int event;
	int ret;

	if ( modifier == PAPI_ENUM_FIRST ) {
		event = pfm_get_event_first();
		return pfm4_event_name( event, EventName, len);
	}

	event = pfm_find_event(EventName);
	if (event < 0) {
		PAPIERROR( "pfm_find_event(%s): %s", EventName,
					pfm_strerror( ret ) );
		return PAPI_ENOEVNT;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		event = pfm_get_event_next(event);
		return pfm4_event_name( event, EventName, len);
	} else if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		return PAPI_EINVAL;
	} else
		return PAPI_EINVAL;
}


int
_papi_pfm4_ntv_name_to_bits( char *EventName, hwd_register_t * bits )
{
	//unsigned int event, umask;
	//pfmlib_event_t gete;

	///* For PFM & Perfmon, native info is just an index into the PFM event table. */
	//if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
	//	return ( PAPI_ENOEVNT );

	//memset( &gete, 0x0, sizeof ( pfmlib_event_t ) );

	//gete.event = event;
	//gete.num_masks = prepare_umask( umask, gete.unit_masks );

	//memcpy( bits, &gete, sizeof ( pfmlib_event_t ) );
	return ( PAPI_OK );
}

//static char *
//_pmc_name( int i )
//{
//	/* Should get this from /sys */
//	extern int _pfm4_pmu_type;
//
//	switch ( _pfm4_pmu_type ) {
//#if defined(PFMLIB_MIPS_ICE9A_PMU)
//		/* All the counters after the 2 CPU counters, the 4 sample counters are SCB registers. */
//	case PFMLIB_MIPS_ICE9A_PMU:
//	case PFMLIB_MIPS_ICE9B_PMU:
//		switch ( i ) {
//		case 0:
//			return "Core counter 0";
//		case 1:
//			return "Core counter 1";
//		default:
//			return "SCB counter";
//		}
//		break;
//#endif
//	default:
//		return "Event Code";
//	}
//}

int
_papi_pfm4_ntv_name_to_info( char * EventName, char *names,
							unsigned int *values, int name_len, int count )
{
//	int ret;
//	pfmlib_regmask_t selector;
//	int j, n = MY_VECTOR.cmp_info.num_cntrs;
//	int foo, did_something = 0;
	int did_something = 0;
//	unsigned int umask;
//
//	if ( ( ret =
//		   pfm_get_event_counters( ( ( pfm_register_t * ) bits )->event,
//								   &selector ) ) != PFM_SUCCESS ) {
//		PAPIERROR( "pfm_get_event_counters(%d,%p): %s",
//				   ( ( pfm_register_t * ) bits )->event, &selector,
//				   pfm_strerror( ret ) );
//		return ( PAPI_ESBSTR );
//	}
//#if defined(PFMLIB_MIPS_ICE9A_PMU)
//	extern int _pfm4_pmu_type;
//	switch ( _pfm4_pmu_type ) {
//		/* All the counters after the 2 CPU counters, the 4 sample counters are SCB registers. */
//	case PFMLIB_MIPS_ICE9A_PMU:
//	case PFMLIB_MIPS_ICE9B_PMU:
//		if ( n > 7 )
//			n = 7;
//		break;
//	default:
//		break;
//	}
//#endif
//
//	for ( j = 0; n; j++ ) {
//		if ( pfm_regmask_isset( &selector, j ) ) {
//			if ( ( ret =
//				   pfm_get_event_code_counter( ( ( pfm_register_t * ) bits )->
//											   event, j,
//											   &foo ) ) != PFM_SUCCESS ) {
//				PAPIERROR( "pfm_get_event_code_counter(%p,%d,%d,%p): %s",
//						   *( ( pfm_register_t * ) bits ),
//						   ( ( pfm_register_t * ) bits )->event, j, &foo,
//						   pfm_strerror( ret ) );
//				return ( PAPI_EBUG );
//			}
//			/* Overflow check */
//			if ( ( int )
//				 ( did_something * name_len + strlen( _pmc_name( j ) ) + 1 ) >=
//				 count * name_len ) {
//				SUBDBG( "Would overflow register name array." );
//				return ( did_something );
//			}
//			values[did_something] = foo;
//			strncpy( &names[did_something * name_len], _pmc_name( j ),
//					 name_len );
//			did_something++;
//			if ( did_something == count )
//				break;
//		}
//		n--;
//	}
//	/* assumes umask is unchanged, even if event code changes */
//	umask = convert_pfm_masks( bits );
//	if ( umask && ( did_something < count ) ) {
//		values[did_something] = umask;
//		if ( strlen( &names[did_something * name_len] ) )
//			strncpy( &names[did_something * name_len], " Unit Mask", name_len );
//		else
//			strncpy( &names[did_something * name_len], "Unit Mask", name_len );
//		did_something++;
//	}
	return ( did_something );
}

