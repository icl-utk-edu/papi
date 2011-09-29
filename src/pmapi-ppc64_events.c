#include SUBSTRATE
#include "papi.h"
#include "papi_internal.h"

hwd_pminfo_t pminfo;
pm_groups_info_t pmgroups;
native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];
PPC64_native_map_t native_name_map[PAPI_MAX_NATIVE_EVENTS];
hwd_groups_t group_map[MAX_GROUPS] = { 0 };

/* to initialize the native_table */
void
initialize_native_table(  )
{
	int i, j;

	memset( native_table, 0,
			PAPI_MAX_NATIVE_EVENTS * sizeof ( native_event_entry_t ) );
	memset( native_name_map, 0,
			PAPI_MAX_NATIVE_EVENTS * sizeof ( PPC64_native_map_t ) );
	for ( i = 0; i < PAPI_MAX_NATIVE_EVENTS; i++ ) {
		native_name_map[i].index = -1;
		for ( j = 0; j < MAX_COUNTERS; j++ )
			native_table[i].resources.counter_cmd[j] = -1;
	}
}

/* to setup native_table group value */
static void
ppc64_setup_gps( int total )
{
	int i, j, gnum;

	for ( i = 0; i < total; i++ ) {
		for ( j = 0; j < MAX_COUNTERS; j++ ) {
			/*      native_table[i].resources.rgg[j]=-1; */
			if ( native_table[i].resources.selector & ( 1 << j ) ) {
				for ( gnum = 0; gnum < pmgroups.maxgroups; gnum++ ) {
					if ( native_table[i].resources.counter_cmd[j] ==
						 pmgroups.event_groups[gnum].events[j] ) {
						/* could use gnum instead of pmgroups.event_groups[gnum].group_id */
						native_table[i].resources.group[pmgroups.
														event_groups[gnum].
														group_id / 32] |=
							1 << ( pmgroups.event_groups[gnum].group_id % 32 );
					}
				}
			}
		}
	}

	for ( gnum = 0; gnum < pmgroups.maxgroups; gnum++ ) {
		for ( i = 0; i < MAX_COUNTERS; i++ ) {
			/*group_map[gnum].counter_cmd[i] = pmgroups.event_groups[gnum].events[i]; */
		    if (pmgroups.event_groups[gnum].group_id >=MAX_GROUPS) {
		       fprintf(stderr,"ERROR, group number trying to go past MAX GROUPS\n");
		       continue;
		    } 

			group_map[pmgroups.event_groups[gnum].group_id].counter_cmd[i] =
				pmgroups.event_groups[gnum].events[i];
		}
	}
}

/* to setup native_table values, and return number of entries */
int
ppc64_setup_native_table(  )
{
	hwd_pmevents_t *wevp;
	hwd_pminfo_t *info;
	int pmc, ev, i, j, index;

	info = &pminfo;
	index = 0;
	initialize_native_table(  );
	for ( pmc = 0; pmc < info->maxpmcs; pmc++ ) {
		wevp = info->list_events[pmc];
		for ( ev = 0; ev < info->maxevents[pmc]; ev++, wevp++ ) {
			for ( i = 0; i < index; i++ ) {
				if ( strcmp( wevp->short_name, native_table[i].name ) == 0 ) {
					native_table[i].resources.selector |= 1 << pmc;
					native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
					break;
				}
			}
			if ( i == index ) {
				/*native_table[i].index=i; */
				native_table[i].resources.selector |= 1 << pmc;
				native_table[i].resources.counter_cmd[pmc] = wevp->event_id;
				native_table[i].name = wevp->short_name;
				native_table[i].description = wevp->description;
				native_name_map[i].name = native_table[i].name;
				native_name_map[i].index = i;
				index++;
			}
		}
	}
	ppc64_setup_gps( index );

	return index;
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
			i = strlen( line );
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

/*  Trims blank space from both ends of a string (in place).
    Returns pointer to new start address */
static char *
trim_string( char *in )
{
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

extern hwi_presets_t _papi_hwi_presets;

static int
find_preset_code( char *tmp, int *code )
{
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

/*#define SHOW_LOADS */
#ifdef SHOW_LOADS
#define SUBDBG printf
#endif

static int
load_preset_table( char *pmu_name, int pmu_type,
				   pfm_preset_search_entry_t * here )
{
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

/*  here[0].preset = PAPI_TOT_CYC;
  here[0].derived = NOT_DERIVED;
  here[1].preset = PAPI_TOT_INS;
  here[1].derived = NOT_DERIVED;
*/
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
				} else
#ifdef SHOW_LOADS
					SUBDBG( "Additional qualifier match failed %d vs %d.\n",
							pmu_type, type )
#endif
						;
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
pmapi_find_full_event( char *name, int *evtcode )
{
	int i;

	for ( i = 0; i < PAPI_MAX_NATIVE_EVENTS; i++ )
		if ( strcmp( name, native_name_map[i].name ) == 0 ) {
			*evtcode = native_name_map[i].index | PAPI_NATIVE_MASK;
			return PAPI_OK;
		}

	return PAPI_ESBSTR;
}

#if 0
static int
setup_preset_term( int *native, pfmlib_event_t * event )
{
	/* It seems this could be greatly simplified. If impl_cnt is non-zero,
	   the event lives on a counter. Therefore the entire routine could be:
	   if (impl_cnt!= 0) encode_native_event.
	   Am I wrong?
	 */
	pfmlib_regmask_t impl_cnt, evnt_cnt;
	unsigned int n;
	int j, ret;

	/* find out which counters it lives on */
	if ( ( ret =
		   pfm_get_event_counters( event->event,
								   &evnt_cnt ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_event_counters(%d,%p): %s", event->event, &evnt_cnt,
				   pfm_strerror( ret ) );
		return ( PAPI_EBUG );
	}
	if ( ( ret = pfm_get_impl_counters( &impl_cnt ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_impl_counters(%p): %s", &impl_cnt,
				   pfm_strerror( ret ) );
		return ( PAPI_EBUG );
	}

	/* Make sure this event lives on some counter, if so, put in the description. If not, BUG */
	if ( ( ret = pfm_get_num_counters( &n ) ) != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_num_counters(%d): %s", n, pfm_strerror( ret ) );
		return ( PAPI_EBUG );
	}

	for ( j = 0; n; j++ ) {
		if ( pfm_regmask_isset( &impl_cnt, j ) ) {
			n--;
			if ( pfm_regmask_isset( &evnt_cnt, j ) ) {
				*native =
					encode_native_event( event->event, event->num_masks,
										 event->unit_masks );
				return ( PAPI_OK );
			}
		}
	}

	PAPIERROR
		( "PAPI preset 0x%08x PFM event %d did not have any available counters",
		  event->event, j );
	return ( PAPI_ENOEVNT );
}
#endif

static int
generate_preset_search_map( hwi_search_t ** maploc, hwi_dev_notes_t ** noteloc,
							pfm_preset_search_entry_t * strmap, int npresets )
{
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
				   pmapi_find_full_event( strmap[i].findme[term],
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
				PAPIERROR( "pmapi_find_full_event(%s)",
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
free_notes( hwi_dev_notes_t * here )
{
	int i = 0;
	while ( here[i].event_code ) {
		free( here[i].dev_note );
		i++;
	}
}

int
_papi_pmapi_setup_presets( char *pmu_name, int pmu_type )
{
	int retval, npresets;
	hwi_search_t *preset_search_map = NULL;
	hwi_dev_notes_t *notemap = NULL;
	pfm_preset_search_entry_t *_perfmon2_pfm_preset_search_map;

	/* allocate and clear array of search string structures */
	_perfmon2_pfm_preset_search_map =
		malloc( sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );
	if ( _perfmon2_pfm_preset_search_map == NULL )
		return ( PAPI_ENOMEM );
	memset( _perfmon2_pfm_preset_search_map, 0x0,
			sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );

	npresets = load_preset_table( pmu_name, pmu_type,
				      _perfmon2_pfm_preset_search_map );
	if (npresets < 0) {
		retval = npresets;
		goto out;
	}

	retval = generate_preset_search_map( &preset_search_map, &notemap,
					     _perfmon2_pfm_preset_search_map, npresets );
	if ( retval )
		goto out;

	retval = _papi_hwi_setup_all_presets( preset_search_map, notemap );
out:
	free_preset_table( _perfmon2_pfm_preset_search_map );
	free( _perfmon2_pfm_preset_search_map );
	free( preset_search_map );
	free_notes( notemap );
	free( notemap );

	return retval;
}
