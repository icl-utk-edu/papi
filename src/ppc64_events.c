/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    ppc64_events.c
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

#include "papi_internal.h"
#include <string.h>
#include "libperfctr.h"

hwd_groups_t group_map[MAX_GROUPS] = { {0}
, {0}
, {0}
, {0}
, {0}
};
native_event_entry_t native_table[PAPI_MAX_NATIVE_EVENTS];

/* to initialize the native_table */
void
perfctr_initialize_native_table(  )
{
	int i, j;
	memset( native_table, 0,
			PAPI_MAX_NATIVE_EVENTS * sizeof ( native_event_entry_t ) );
	for ( i = 0; i < PAPI_MAX_NATIVE_EVENTS; i++ ) {
		for ( j = 0; j < MAX_COUNTERS; j++ )
			native_table[i].resources.counter_cmd[j] = -1;
	}
}


/* to setup native_table group value */
void
perfctr_ppc64_setup_gps( int total, ntv_event_group_info_t * group_info )
{
	int i, j, gnum;

	for ( i = 0; i < total; i++ ) {
		for ( j = 0; j < MAX_COUNTERS; j++ ) {
			if ( native_table[i].resources.selector & ( 1 << j ) ) {
				for ( gnum = 0; gnum < group_info->maxgroups; gnum++ ) {
					if ( native_table[i].resources.counter_cmd[j] ==
						 group_info->event_groups[gnum]->events[j] ) {
						native_table[i].resources.group[gnum / 32] |=
							1 << ( gnum % 32 );
					}
				}
			}
		}
	}

	for ( gnum = 0; gnum < group_info->maxgroups; gnum++ ) {
		group_map[gnum].mmcr0 = group_info->event_groups[gnum]->mmcr0;
		group_map[gnum].mmcr1L = group_info->event_groups[gnum]->mmcr1L;
		group_map[gnum].mmcr1U = group_info->event_groups[gnum]->mmcr1U;
		group_map[gnum].mmcra = group_info->event_groups[gnum]->mmcra;
		for ( i = 0; i < MAX_COUNTERS; i++ )
			group_map[gnum].counter_cmd[i] =
				group_info->event_groups[gnum]->events[i];
	}
}


/* to setup native_table values, and return number of entries */
int
perfctr_ppc64_setup_native_table(  )
{
	int pmc, ev, i, j, index;
	/* This is for initialisation-testing of consistency between
	   native_name_map and our events file */
	int itemCount = 0;
	index = 0;
	perfctr_initialize_native_table(  );
	ntv_event_info_t *info = perfctr_get_native_evt_info(  );
	if ( info == NULL ) {
		PAPIERROR( EVENT_INFO_FILE_ERROR );
		return PAPI_ECMP;
	}
	ntv_event_t *wevp;
	for ( pmc = 0; pmc < info->maxpmcs; pmc++ ) {
		wevp = info->wev[pmc];
		for ( ev = 0; ev < info->maxevents[pmc]; ev++, wevp++ ) {
			for ( i = 0; i < index; i++ ) {
				if ( strcmp( wevp->symbol, native_table[i].name ) == 0 ) {
					native_table[i].resources.selector |= 1 << pmc;
					native_table[i].resources.counter_cmd[pmc] =
						wevp->event_num;
					break;
				}
			}
			if ( i == index ) {
				//native_table[i].index=i; 
				native_table[i].resources.selector |= 1 << pmc;
				native_table[i].resources.counter_cmd[pmc] = wevp->event_num;
				native_table[i].name =
					( char * ) malloc( strlen( wevp->symbol ) + 1 );
				strcpy( native_table[i].name, wevp->symbol );
				native_table[i].description = wevp->description;
				index++;
				for ( j = 0; j < MAX_NATNAME_MAP_INDEX; j++ ) {
					/* It appears that here, if I'm right, that the events
					   file entry matches the event from native_name_map, */
					/* This here check is to ensure that native_name_map in fact
					   has MAX_NATNAME_MAP_INDEX elements, or rather that it never
					   tries to access one that has not been initialised. */
					if ( native_name_map[j].name == NULL ) {
						SUBDBG( "native_name_map has a NULL at position %i\n",
								j );
						PAPIERROR
							( "Inconsistency between events_map file and events header." );
						return PAPI_EBUG;
					}
					if ( strcmp( native_table[i].name, native_name_map[j].name )
						 == 0 ) {
						native_name_map[j].index = i;
						itemCount++;
						break;
					}
				}
				/* If we never set native_name_map[j], then there is an
				   inconsistency between native_name_map and native_table */
				if ( ( !( j < MAX_NATNAME_MAP_INDEX ) ) ||
					 native_name_map[j].index != i ) {
					SUBDBG
						( "No match found between native_name_map and native_table.  "
						  "Values was %s at position %i in native_table.\n",
						  native_table[i].name, i );
					PAPIERROR
						( "Inconsistency between native_name_map and events file." );
					return PAPI_EBUG;
				}
			}
		}
	}
	/* given the previous evidence that native_name_map is a superset of
	   native_table, ensuring this match in their cardinality shows them to
	   be equivalent. */
	if ( itemCount != MAX_NATNAME_MAP_INDEX ) {
		SUBDBG( "%i events found in native_table, but really should be %i\n",
				itemCount, MAX_NATNAME_MAP_INDEX );
		PAPIERROR
			( "Inconsistent cardinality between native_name_map and events file",
			  itemCount, MAX_NATNAME_MAP_INDEX );
		return PAPI_EBUG;
	}

	ntv_event_group_info_t *gp_info = perfctr_get_native_group_info(  );
	if ( gp_info == NULL ) {
		perfctr_initialize_native_table(  );
		PAPIERROR( EVENT_INFO_FILE_ERROR );
		return PAPI_ECMP;
	}

	perfctr_ppc64_setup_gps( index, gp_info );
	_papi_hwi_system_info.sub_info.num_native_events = index;

	return check_native_name(  );
}

int
check_native_name(  )
{
	enum native_name foo;
	int itemCount = 0;
	int i;

	/* This should ensure that the cardinality of native_name is the same
	   as that of native_name_map which may be true iff native_name 
	   expresses the same data as native_name_map and there is a 1:1 
	   mapping from one onto the other, though there is no guarantee of 
	   order. */
	if ( ( NATNAME_GUARD - PAPI_NATIVE_MASK ) != MAX_NATNAME_MAP_INDEX ) {
		SUBDBG( "%i is the number of elements apparently in native_name, "
				"but really should be %i, according to native_name_map.\n",
				( NATNAME_GUARD - PAPI_NATIVE_MASK ), MAX_NATNAME_MAP_INDEX );
		PAPIERROR
			( "Inconsistent cardinality between native_name and native_name_map "
			  "detected in preliminary check\n" );
		return PAPI_EBUG;
	}

	/* The following is sanity checking only.  It attempts to verify some level
	   of consistency between native_name and native_name_map and native_table.
	   This should imply that native_name is a subset of native_name_map. */
	for ( foo = PAPI_NATIVE_MASK; foo < NATNAME_GUARD; foo++ ) {
		for ( i = 0; i < MAX_NATNAME_MAP_INDEX; i++ ) {
			/* Now, if the event we are on is the native event we seek... */
			if ( ( native_name_map[i].index | PAPI_NATIVE_MASK ) == foo ) {
				itemCount++;
				break;
			}
		}
	}
	if ( itemCount != MAX_NATNAME_MAP_INDEX ) {
		SUBDBG( "Inconsistency between native_name_map and native_name.  "
				"%i events matched, but really should be %i\n", itemCount,
				MAX_NATNAME_MAP_INDEX );
		PAPIERROR
			( "Inconsistent cardinality between native_name and native_name_map\n" );
		return PAPI_EBUG;
	}

	return PAPI_OK;
}

static FILE *
open_file( const char *fname )
{
	char *cpu;
	char *dot = ".";
	char *dot_dot = "..";
#ifdef _POWER5p
	cpu = "power5+";
#elif defined(_POWER5)
	cpu = "power5";
#elif defined(_PPC970)
	cpu = "ppc970";
#else
	cpu = "";
#endif
	char *dir = ( char * ) getenv( "PAPI_EVENTFILE_PATH" );
#ifdef PAPI_DATADIR
	if ( dir == NULL ) {
		dir = PAPI_DATADIR;
	}
#endif
	/* If dir is still NULL, assume current dir holds event_data dir */
	if ( dir == NULL )
		dir = dot;

	char *relative_pathname = ( char * ) malloc( strlen( "/" ) +
												 strlen( "event_data" ) +
												 strlen( "/" ) + strlen( cpu ) +
												 strlen( "/" ) +
												 strlen( fname ) + 1 );
	strcpy( relative_pathname, "/" );
	strcat( relative_pathname, "event_data" );
	strcat( relative_pathname, "/" );
	strcat( relative_pathname, cpu );
	strcat( relative_pathname, "/" );
	strcat( relative_pathname, fname );
	/* Add a little extra space to the malloc for the case where dir = "." since
	 * we may be trying dir = ".." later on. */
	char *pathname =
		( char * ) malloc( strlen( dir ) + strlen( relative_pathname ) + 4 );
	int keep_trying;
	if ( strcmp( dir, dot ) == 0 )
		keep_trying = 2;
	else
		keep_trying = 3;
	FILE *file = NULL;
	while ( file == NULL && keep_trying-- ) {
		strcpy( pathname, dir );
		strcat( pathname, relative_pathname );
		file = fopen( pathname, "r" );
		if ( strcmp( dir, dot ) == 0 ) {
			dir = dot_dot;
		} else {
			dir = dot;
		}
		SUBDBG( "Attempt to open event data file %s %s successful.\n", pathname,
				( file == NULL ) ? "was not" : "was" );
		memset( pathname, '\0', sizeof ( pathname ) );
	}

	free( pathname );
	free( relative_pathname );
	return ( file );
}

static ntv_event_t *
copy_buffer( ntv_event_t events[], int maxevents )
{
	ntv_event_t *cur_wev, *start_wev;
	start_wev = ( ntv_event_t * ) malloc( sizeof ( ntv_event_t ) * maxevents );
	cur_wev = start_wev;

	int cnt;
	for ( cnt = 0; cnt < maxevents; cnt++, cur_wev++ ) {
		cur_wev->event_num = events[cnt].event_num;
		strcpy( cur_wev->symbol, events[cnt].symbol );
		cur_wev->short_description =
			( char * ) malloc( strlen( events[cnt].short_description ) );
		strcpy( cur_wev->short_description, events[cnt].short_description );
		cur_wev->description =
			( char * ) malloc( strlen( events[cnt].description ) );
		strcpy( cur_wev->description, events[cnt].description );
	}
	return start_wev;
}

static ntv_event_info_t *
parse_eventfile( FILE * evfile )
{
	int counter = 0, num_events = 0;
	int i, len, cc;
	int event;
	int line_counter_flag = 0;
	char line_data[1024];
	ntv_event_t events[PAPI_MAX_NATIVE_EVENTS];
	ntv_event_info_t *ntv_evt_info =
		( ntv_event_info_t * ) malloc( sizeof ( ntv_event_info_t ) );
	ntv_evt_info->maxpmcs = 0;
	char data[1024];
	while ( fgets( data, 1022, evfile ) ) {
		if ( feof( evfile ) )
			continue;
		if ( strlen( data ) < 2 )
			continue;

		if ( strncmp( data, "$$$$", 4 ) == 0 ) {
			line_counter_flag = 0;
			ntv_evt_info->maxevents[counter - 1] = num_events;
			ntv_evt_info->wev[counter - 1] = copy_buffer( events, num_events );
			ntv_event_t *temp = ntv_evt_info->wev[counter - 1];
			temp++;
		}
		switch ( line_counter_flag ) {
		case 0:
			if ( sscanf( data, "{ counter %u", &counter ) == 1 ) {
				line_counter_flag = 1;
				num_events = 0;
				ntv_evt_info->maxpmcs++;
			}
			break;
		case 1:
			if ( sscanf( data, "#%u", &event ) != 1 ) {
				fprintf( stderr, "EVS file format error 1 (%s)\n", data );
				return NULL;
			}
			if ( event >= PAPI_MAX_NATIVE_EVENTS ) {
				fprintf( stderr, "EVS file format error 1 (%s)\n", data );
				return NULL;
			}
			events[num_events].event_num = event;
			len = strlen( data );
			int symb_found = 0;
			for ( i = cc = 0; i < len; i++ ) {
				if ( data[i] == ',' )
					cc++;
				if ( cc == 5 && !symb_found ) {
					strcpy( line_data, &data[i + 1] );
					int j = 0;
					while ( line_data[j] != ',' )
						j++;
					strncpy( events[num_events].symbol, line_data, j );
					events[num_events].symbol[j] = 0;
					symb_found = 1;
					i += j;
				} else if ( cc == 6 ) {
					len = strlen( &data[i + 1] );
					events[num_events].short_description =
						( char * ) malloc( len );
					strcpy( events[num_events].short_description,
							&data[i + 1] );
					events[num_events].short_description[len - 1] = 0;
					break;
				}
			}
			line_counter_flag = 2;
			break;
		case 2:
			line_counter_flag = 3;
			break;
		case 3:
			len = strlen( data );
			events[num_events].description = ( char * ) malloc( len );
			strcpy( events[num_events].description, data );
			events[num_events].description[len - 1] = 0;
			line_counter_flag = 1;
			num_events++;
			break;
		}
	}
	if ( counter == 0 ) {
		free( ntv_evt_info );
		ntv_evt_info = NULL;
	}
	if ( counter == MAX_COUNTERS ) {
		ntv_evt_info->maxevents[counter - 1] = num_events;
		ntv_evt_info->wev[counter - 1] = copy_buffer( events, num_events );
	}
	fclose( evfile );
	return ntv_evt_info;
}

static int
any_counter_invalid( int event_id[], int size )
{
	int j;
	for ( j = 0; j < size; j++ ) {
		if ( event_id[j] >= PAPI_MAX_NATIVE_EVENTS )
			return 1;
	}
	return 0;
}

static ntv_event_group_info_t *
parse_groupfile( FILE * grp_file )
{
	char data[1024];
	unsigned int mmcr0, mmcr1H, mmcr1L, mmcra;
	int g, state = 0;
	ntv_event_group_info_t *group_info =
		( ntv_event_group_info_t * )
		malloc( sizeof ( ntv_event_group_info_t ) );
	group_info->maxgroups = 0;
	int event_num[MAX_COUNTERS];
	while ( fgets( data, 1022, grp_file ) ) {
		if ( feof( grp_file ) || ( strlen( data ) < 2 ) )
			continue;

		switch ( state ) {
		case 0:
#if defined(_POWER5) || defined(_POWER5p)
			if ( sscanf
				 ( data, "#%u,%u,%u,%u,%u,%u,%u", &g, &event_num[0],
				   &event_num[1], &event_num[2], &event_num[3], &event_num[4],
				   &event_num[5] ) == 7 ) {
				state = 1;
				if ( any_counter_invalid( event_num, 6 ) ) {
					fprintf( stderr, "ERROR: Maximum events exceeded\n" );
					return NULL;
				}
			}
#else
			if ( sscanf
				 ( data, "#%u,%u,%u,%u,%u,%u,%u,%u,%u", &g, &event_num[0],
				   &event_num[1], &event_num[2], &event_num[3], &event_num[4],
				   &event_num[5], &event_num[6], &event_num[7] ) == 9 ) {
				state = 1;
				if ( any_counter_invalid( event_num, 8 ) ) {
					fprintf( stderr, "ERROR: Maximum events exceeded\n" );
					return NULL;
				}
			}
#endif
			if ( state == 1 ) {
				group_info->event_groups[group_info->maxgroups] =
					( event_group_t * ) malloc( sizeof ( event_group_t ) );
				group_info->event_groups[group_info->maxgroups]->group_id = g;
				int j = 0;
				for ( ; j < MAX_COUNTERS; j++ )
					group_info->event_groups[group_info->maxgroups]->events[j] =
						event_num[j];
			}
			break;
		case 1:
			// unused hex event codes
			state = 2;
			break;
		case 2:
			/* get mmcr values */
			if ( sscanf( data, "%#x,%#x,%#x,%#x", &mmcr0, &mmcr1H, &mmcr1L, &mmcra )
				 != 4 ) {
				fprintf( stderr, "GPS file format error 1 (%s)\n", data );
				return NULL;
			}
			state = 3;
			group_info->event_groups[group_info->maxgroups]->mmcr0 = mmcr0;
			group_info->event_groups[group_info->maxgroups]->mmcr1L = mmcr1L;
			group_info->event_groups[group_info->maxgroups]->mmcr1U = mmcr1H;
			group_info->event_groups[group_info->maxgroups]->mmcra = mmcra;
			group_info->maxgroups++;
			break;
		case 3:
			// unused group name
			state = 0;
		}
	}
	fclose( grp_file );
	return group_info;
}

ntv_event_info_t *
perfctr_get_native_evt_info( void )
{
	ntv_event_info_t *evt_info = NULL;
	FILE *evt_file = open_file( "events" );
	if ( evt_file != NULL ) {
		evt_info = parse_eventfile( evt_file );
	}
	return evt_info;

}

ntv_event_group_info_t *
perfctr_get_native_group_info( void )
{
	ntv_event_group_info_t *groups = NULL;
	FILE *grp_file = NULL;
	if ( ( grp_file = open_file( "groups" ) ) != NULL ) {
		groups = parse_groupfile( grp_file );
	}
	return groups;
}
