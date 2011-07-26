/*
* File:    papi_pfm4_events.c
* Author:  Dan Terpstra: blantantly extracted from Phil's perfmon.c
*          mucci@cs.utk.edu
*/

#include <ctype.h>
#include <string.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "papi_pfm_events.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"

#define PAPI_EVENT_FILE "papi_events.csv"

typedef struct
{
	int preset;		   /* Preset code */
	int derived;		   /* Derived code */
	char *( findme[MAX_COUNTER_TERMS] ); /* Strings to look for, more than 1 means derived */
	char *operation;	   /* PostFix operations between terms */
	char *note;	           /* In case a note is included with a preset */
} pfm_preset_search_entry_t;

extern papi_vector_t MY_VECTOR;
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];


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

  SUBDBG("ENTER\n");

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

  SUBDBG("ENTER\n");
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

#define SHOW_LOADS

static int
load_preset_table( char *pmu_str, int pmu_type,
				   pfm_preset_search_entry_t * here )
{

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

  SUBDBG("ENTER\n");

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

        /* FIXME -- make sure PAPI_TOT_CYC and PAPI_TOT_INS are #1/#2 */

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
			        get_presets=0; found_presets=0;
				/* goto done; */
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
		  //			SUBDBG( "PRESET token found on line %d\n", line_no );
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

			SUBDBG( "Found 0x%08x for %s\n", preset, t );

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

			SUBDBG( "Found %d for %s\n", derived, t );
			SUBDBG( "Adding 0x%x,%d to preset search table.\n", preset,
					derived );

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
/*  done: */
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
  SUBDBG("ENTER\n");
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
  SUBDBG("ENTER\n");
	while ( here[i].event_code ) {
		free( here[i].dev_note );
		i++;
	}
}


static int
generate_preset_search_map( hwi_search_t ** maploc, hwi_dev_notes_t ** noteloc,
							pfm_preset_search_entry_t * strmap )
{

	int k = 0, term;
	unsigned int i = 0, j = 0;
	hwi_search_t *psmap;
	hwi_dev_notes_t *notemap;
	unsigned int event_idx;
  
        SUBDBG("ENTER\n");

	/* Count up the proposed presets */
	while ( strmap[i].preset ) {
	  i++;
	}

	SUBDBG( "generate_preset_search_map(%p,%p,%p) %d proposed presets\n",
			maploc, noteloc, strmap, i );
	i++;

	/* Add null entry */
	psmap = ( hwi_search_t * ) malloc( i * sizeof ( hwi_search_t ) );
	notemap = ( hwi_dev_notes_t * ) malloc( i * sizeof ( hwi_dev_notes_t ) );
	if ( ( psmap == NULL ) || ( notemap == NULL ) ) {
	   return ( PAPI_ENOMEM );
	}
	memset( psmap, 0x0, i * sizeof ( hwi_search_t ) );
	memset( notemap, 0x0, i * sizeof ( hwi_dev_notes_t ) );

	i = 0;
	while ( strmap[i].preset ) {

	   /* Handle derived events */
	   term = 0;
	   do {
	      int ret;

	      SUBDBG("Looking up: %s\n",strmap[i].findme[term]);
	      ret=_papi_pfm_ntv_name_to_code(strmap[i].findme[term],
					     &event_idx);

	      if (ret==PAPI_OK) {
		 SUBDBG("Found %x\n",event_idx);
		 psmap[j].data.native[term]=event_idx;
		 term++;
	      }
              else {
		 SUBDBG("Error finding event %x\n",event_idx);
		 break;
	      } 

	   } while ( strmap[i].findme[term] != NULL &&
					  term < MAX_COUNTER_TERMS );

	   /* terminate the native term array with PAPI_NULL */
	   if ( term < MAX_COUNTER_TERMS ) {
	      psmap[j].data.native[term] = PAPI_NULL;
	   }

	   //if ( ret == PAPI_OK ) {
	      psmap[j].event_code = ( unsigned int ) strmap[i].preset;
	      psmap[j].data.derived = strmap[i].derived;
	      if ( strmap[i].derived == DERIVED_POSTFIX ) {
		 strncpy( psmap[j].data.operation, strmap[i].operation,
							 PAPI_MIN_STR_LEN );
	      }
	      if ( strmap[i].note ) {
		 notemap[k].event_code = ( unsigned int ) strmap[i].preset;
		 notemap[k].dev_note = strdup( strmap[i].note );
		 k++;
	      }
	      j++;
	      //}
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

	return PAPI_OK;
}


/*******************************************************************
 *
 *
 *
 ******************************************************************/

/* FIXME -- make it handle arbitrary number */
#define MAX_NATIVE_EVENTS 1000

static struct native_event_t {
  int component;
  char *pmu;
  int papi_code;
  int perfmon_idx;
  char *allocated_name;
  char *base_name;
  char *canonical_name;
  char *pmu_plus_name;
  int users;
} native_events[MAX_NATIVE_EVENTS];

static int num_native_events=0;


static struct native_event_t *find_existing_event(char *name) {

  int i;

  SUBDBG("Looking for %s\n",name);

  for(i=0;i<num_native_events;i++) {
    // using cannonical breaks enumeration
    //    if (!strcmp(name,native_events[i].canonical_name)) {
    if (!strcmp(name,native_events[i].allocated_name)) {
      SUBDBG("Found %s (%x %x)\n",
	     native_events[i].allocated_name,
	     native_events[i].perfmon_idx,
	     native_events[i].papi_code);

       return &native_events[i];
    }
  }
  SUBDBG("%s not allocated yet\n",name);
  return NULL;
}

static struct native_event_t *find_existing_event_by_number(int eventnum) {

  int i;

  for(i=0;i<num_native_events;i++) {
    if (eventnum==native_events[i].papi_code) {
       return &native_events[i];
    }
  }
  return NULL;
}


int find_event_no_aliases(char *name) {

  int j,i, ret;
  pfm_pmu_info_t pinfo;
  pfm_event_info_t event_info;
  char blah[BUFSIZ];

  SUBDBG("Looking for %s\n",name);

  pfm_for_all_pmus(j) {

    memset(&pinfo,0,sizeof(pfm_pmu_info_t));
    pfm_get_pmu_info(j, &pinfo);
    if (!pinfo.is_present) {
       SUBDBG("PMU %d not present, skipping...\n",j);
       continue;
    }

    SUBDBG("Looking in pmu %d\n",j);   
    i = pinfo.first_event; 
    while(1) {
        memset(&event_info,0,sizeof(pfm_event_info_t));
        ret=pfm_get_event_info(i, PFM_OS_PERF_EVENT, &event_info);
	if (ret<0) break;
	
	sprintf(blah,"%s::%s",pinfo.name,event_info.name);
	//SUBDBG("Trying %x %s\n",i,blah);
	if (!strcmp(name,blah)) {
	  SUBDBG("FOUND %s %s %x\n",name,blah,i);
	  return i;
	}

	//SUBDBG("Trying %x %s\n",i,event_info.name);
	if (!strcmp(name,event_info.name)) {
	  SUBDBG("FOUND %s %s %x\n",name,event_info.name,i);
	  return i;
	}
	i++;
    }
  }
  return -1;

}


int find_next_no_aliases(int code) {

  int current_pmu=0,current_event=0,ret;
  pfm_pmu_info_t pinfo;
  pfm_event_info_t event_info;

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  memset(&event_info,0,sizeof(pfm_event_info_t));

  pfm_get_event_info(code, PFM_OS_PERF_EVENT, &event_info);
  current_pmu=event_info.pmu;
  current_event=code+1;

  SUBDBG("Current is %x guessing next is %x\n",code,current_event);

stupid_loop:

  ret=pfm_get_event_info(current_event, PFM_OS_PERF_EVENT, &event_info);
  if (ret>=0) {
    SUBDBG("Returning %x\n",current_event);
     return current_event;
  }

  /* need to increment pmu */
inc_pmu:
  current_pmu++;
  SUBDBG("Incrementing PMU: %x\n",current_pmu);
  if (current_pmu>PFM_PMU_MAX) return -1;

  pfm_get_pmu_info(current_pmu, &pinfo);
  if (!pinfo.is_present) goto inc_pmu;
 
  current_event=pinfo.first_event;

  goto stupid_loop;

}


static struct native_event_t *allocate_native_event(char *name, 
						    int event_idx) {

  int new_event=num_native_events;

  pfm_err_t ret;
  int count=5;
  unsigned int i;
  uint64_t *codes;
  char *fstr=NULL,*base_start;
  int found_idx;
  pfm_event_info_t info;
  pfm_pmu_info_t pinfo;
  char base[BUFSIZ],pmuplusbase[BUFSIZ];

  /* allocate canonical string */

  codes=calloc(count,sizeof(uint64_t));

  ret=pfm_get_event_encoding(name, 
  			     PFM_PLM0|PFM_PLM3,
  			     &fstr, 
  			     &found_idx, 
  			     &codes, 
  			     &count);

  if (codes) free(codes);
  if (fstr) {
     native_events[new_event].canonical_name=strdup(fstr);
     free(fstr);
  }
  //if (ret!=PFM_SUCCESS) {
  //   return NULL;
  //}

  /* get basename */	      
  memset(&info,0,sizeof(pfm_event_info_t));
  memset(&pinfo,0,sizeof(pfm_pmu_info_t));
  ret = pfm_get_event_info(event_idx, PFM_OS_PERF_EVENT, &info);
  if (ret!=PFM_SUCCESS) {
     return NULL;
  }

  pfm_get_pmu_info(info.pmu, &pinfo);

  strncpy(base,name,BUFSIZ);
  i=0;
  base_start=base;
  while(i<strlen(base)) {
    if (base[i]==':') {
      if (base[i+1]==':') {
          i++;
	  base_start=&base[i+1];
      }
      else {
	base[i]=0;
      }
    }
    i++;
  }

  native_events[new_event].base_name=strdup(base_start);

  { char tmp[BUFSIZ];
    sprintf(tmp,"%s::%s",pinfo.name,info.name);
    native_events[new_event].pmu_plus_name=strdup(tmp);

    sprintf(pmuplusbase,"%s::%s",pinfo.name,base_start);
  }

  native_events[new_event].component=0;
  native_events[new_event].pmu=strdup(pinfo.name);
  native_events[new_event].papi_code=new_event | PAPI_NATIVE_MASK;
    
  native_events[new_event].perfmon_idx=find_event_no_aliases(pmuplusbase);
  SUBDBG("Using %x as index instead of %x for %s\n",
	 native_events[new_event].perfmon_idx,event_idx,pmuplusbase);

  native_events[new_event].allocated_name=strdup(name);

  native_events[new_event].users=0;

  SUBDBG("Creating event %s with papi %x perfidx %x\n",
	 name,
	 native_events[new_event].papi_code,
	 native_events[new_event].perfmon_idx);

  num_native_events++;

  /* FIXME -- simply allocate more */
  if (num_native_events >= MAX_NATIVE_EVENTS) {
     fprintf(stderr,"TOO MANY NATIVE EVENTS\n");
     exit(0);
  }

  return &native_events[new_event];

}


int
_papi_pfm_ntv_name_to_code( char *name, unsigned int *event_code )
{

  int actual_idx;
  struct native_event_t *our_event;

  SUBDBG( "Converting %s\n", name);

  our_event=find_existing_event(name);

  if (our_event==NULL) {

      /* we want this, rather than the canonical event name */
      /* returned by pfm_get_event_encoding() as otherwise */
      /* enumeration doesn't work properly.                */

      SUBDBG("Using pfm to look up event %s\n",name);
      actual_idx=pfm_find_event(name);

      if (actual_idx<0) return PAPI_ENOEVNT;

      SUBDBG("Using %x as the index\n",actual_idx);

      our_event=allocate_native_event(name,actual_idx);
    }

  if (our_event!=NULL) {      
     *event_code=our_event->papi_code;
     SUBDBG("Found code: %x\n",*event_code);
     return PAPI_OK;
  }

  SUBDBG("Event %s not found\n",name);

  return PAPI_ENOEVNT;   

}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
static inline int
prepare_umask( unsigned int foo, unsigned int *values )
{
	unsigned int tmp = foo, i;
	int j = 0;

  SUBDBG("ENTER\n");

	SUBDBG( "umask 0x%x\n", tmp );
	while ( ( i = ( unsigned int ) ffs( ( int ) tmp ) ) ) {
		tmp = tmp ^ ( 1 << ( i - 1 ) );
		values[j] = i - 1;
		SUBDBG( "umask %d is %d\n", j, values[j] );
		j++;
	}
	return ( j );
}

int
_papi_pfm_setup_presets( char *pmu_name, int pmu_type )
{
	int retval;
	hwi_search_t *preset_search_map = NULL;
	hwi_dev_notes_t *notemap = NULL;
	pfm_preset_search_entry_t *_perfmon2_pfm_preset_search_map;

        SUBDBG("ENTER\n");

	/* allocate and clear array of search string structures */
	_perfmon2_pfm_preset_search_map =
		malloc( sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );
	if ( _perfmon2_pfm_preset_search_map == NULL )
		return ( PAPI_ENOMEM );
	memset( _perfmon2_pfm_preset_search_map, 0x0,
		sizeof ( pfm_preset_search_entry_t ) * PAPI_MAX_PRESET_EVENTS );

	retval =
		load_preset_table( pmu_name, pmu_type,
				   _perfmon2_pfm_preset_search_map );
	if ( retval )
		return ( retval );

	retval =
		generate_preset_search_map( &preset_search_map, &notemap,
					    _perfmon2_pfm_preset_search_map );

	free_preset_table( _perfmon2_pfm_preset_search_map );
	free( _perfmon2_pfm_preset_search_map );
	if ( retval )
		return ( retval );

	retval = _papi_hwi_setup_all_presets( preset_search_map, notemap );
	if ( retval ) {
		free( preset_search_map );
		free_notes( notemap );
		free( notemap );
		return ( retval );
	}

	return ( PAPI_OK );
}

int
_papi_pfm_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{

        struct native_event_t *our_event;

        SUBDBG("ENTER %x\n",EventCode);

        our_event=find_existing_event_by_number(EventCode);

	if (our_event==NULL) return PAPI_ENOEVNT;

	/* use actual rather than canonical to not break enum */
	strncpy(ntv_name,our_event->allocated_name,len);

	return PAPI_OK;
}


static int find_max_umask(struct native_event_t *current_event) {

  pfm_event_attr_info_t ainfo;
  char *b;
  int a, ret, max =0;
  pfm_event_info_t info;
  char event_string[BUFSIZ],*ptr;

  SUBDBG("Enter\n");

  SUBDBG("Trying to find max umask in %s\n",current_event->allocated_name);

  strcpy(event_string,current_event->allocated_name);

  if (strstr(event_string,"::")) {
    ptr=strstr(event_string,"::");
    ptr+=2;
    b=strtok(ptr,":");
  }
  else {
     b=strtok(event_string,":");
  }

  if (!b) {
     SUBDBG("No colon!\n");
     return -1;
  }

  memset(&info,0,sizeof(pfm_event_info_t));
  ret = pfm_get_event_info(current_event->perfmon_idx, PFM_OS_PERF_EVENT, &info);
  if (ret!=PFM_SUCCESS) {
     SUBDBG("get_event_info failed\n");
     return -1;
  }

  /* skip first */
  b=strtok(NULL,":");
  if (!b) {
     SUBDBG("Skipping first failed\n");
     return -1;
  }

  while(b) {
    a=0;
    while(1) {

      SUBDBG("get_event_attr %x %d %p\n",current_event->perfmon_idx,a,&ainfo);

      memset(&ainfo,0,sizeof(pfm_event_attr_info_t));

      ret = pfm_get_event_attr_info(current_event->perfmon_idx, a, 
				    PFM_OS_PERF_EVENT, &ainfo);

      if (ret != PFM_SUCCESS) {
	SUBDBG("get_event_attr failed %s\n",pfm_strerror(ret));
	return ret;
      }

      SUBDBG("Trying %s with %s\n",ainfo.name,b);

      if (!strcasecmp(ainfo.name, b)) {
	SUBDBG("Found %s %d\n",b,a);
	if (a>max) max=a;
	goto found_attr;
      }
      a++;
    }

    SUBDBG("attr=%s not found for event %s\n", b, info.name);

    return PAPI_ECNFLCT;

found_attr:

    b=strtok(NULL,":");
  }

  SUBDBG("Found max %d\n", max);

  return max;
}


/* attributes concactinated onto end of descr separated by ", masks" */
/* then comma separated */

int
_papi_pfm_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
  int ret,a,first_mask=1;
  char *eventd, *tmp=NULL;
	//int i, first_desc=1;
  pfm_event_info_t gete;
	//     	size_t total_len = 0;


  pfm_event_attr_info_t ainfo;
  char *b;
  //  pfm_event_info_t info;
  char event_string[BUFSIZ],*ptr;

	struct native_event_t *our_event;

        SUBDBG("ENTER %x\n",EventCode);

	our_event=find_existing_event_by_number(EventCode);

	memset( &gete, 0, sizeof ( gete ) );

	SUBDBG("Getting info on %x\n",our_event->perfmon_idx);
	ret=pfm_get_event_info(our_event->perfmon_idx, PFM_OS_PERF_EVENT, &gete);
	SUBDBG("Return=%d\n",ret);

	/* error check?*/

	eventd=strdup(gete.desc);

	tmp = ( char * ) malloc( strlen( eventd ) + 1 );
	if ( tmp == NULL ) {
	   free( eventd );
	   return PAPI_ENOMEM;
	}
	tmp[0] = '\0';
	strcat( tmp, eventd );
	free( eventd );
	
	/* Handle Umasks */
  strcpy(event_string,our_event->allocated_name);

  if (strstr(event_string,"::")) {
    ptr=strstr(event_string,"::");
    ptr+=2;
    b=strtok(ptr,":");
  }
  else {
    b=strtok(event_string,":");
  }
  
  if (!b) {
     SUBDBG("No colon!\n"); /* no umask */
     goto descr_in_tmp;
  }

  /* skip first */
  b=strtok(NULL,":");
  if (!b) {
     SUBDBG("Skipping first failed\n");
     goto descr_in_tmp;
  }

  while(b) {
    a=0;
    while(1) {

      SUBDBG("get_event_attr %x %p\n",our_event->perfmon_idx,&ainfo);

      memset(&ainfo,0,sizeof(pfm_event_attr_info_t));

      ret = pfm_get_event_attr_info(our_event->perfmon_idx, a, 
				    PFM_OS_PERF_EVENT, &ainfo);

      if (ret != PFM_SUCCESS) {
	SUBDBG("get_event_attr failed %s\n",pfm_strerror(ret));
	return ret;
      }

      SUBDBG("Trying %s with %s\n",ainfo.name,b);

      if (!strcasecmp(ainfo.name, b)) {
	int new_length;

	 SUBDBG("Found %s\n",b);
	 new_length=strlen(ainfo.desc);

	 if (first_mask) {
	    tmp=realloc(tmp,strlen(tmp)+new_length+1+strlen(", masks:"));
	    strcat(tmp,", masks:");
	 }
	 else {
	    tmp=realloc(tmp,strlen(tmp)+new_length+1+strlen(","));
	    strcat(tmp,",");
	 }
	 strcat(tmp,ainfo.desc);

	 goto found_attr;
      }
      a++;
    }

    SUBDBG("attr=%s not found for event %s\n", b, ainfo.name);

    return PAPI_ECNFLCT;

found_attr:

    b=strtok(NULL,":");
  }

descr_in_tmp:
	strncpy( ntv_descr, tmp, ( size_t ) len );
	if ( ( int ) strlen( tmp ) > len - 1 )
		ret = PAPI_EBUF;
	else
		ret = PAPI_OK;
	free( tmp );

	SUBDBG("PFM4 Code: %x %s\n",EventCode,ntv_descr);

	return ret;

}




static int
papi_pfm_get_event_first_active(void)
{
  int pidx, pmu_idx, ret;

  pfm_pmu_info_t pinfo;

  memset(&pinfo,0,sizeof(pfm_pmu_info_t));

  pmu_idx=0;

  while(pmu_idx<PFM_PMU_MAX) {

    ret=pfm_get_pmu_info(pmu_idx, &pinfo);

    if ((ret==PFM_SUCCESS) && pinfo.is_present) {

      pidx=pinfo.first_event;

      return pidx;

    }
    pmu_idx++;

  }
  return PAPI_ENOEVNT;
  
}


/* first PMU, no leading PMU indicator */
/* subsequent, yes */

static int
convert_libpfm4_to_string( int code, char **event_name)
{

  int ret;
  pfm_event_info_t gete;//,first_info;
  pfm_pmu_info_t pinfo;
  char name[BUFSIZ];
  //int first;

  SUBDBG("ENTER %x\n",code);

  //first=papi_pfm_get_event_first_active();

  memset( &gete, 0, sizeof ( pfm_event_info_t ) );
  //memset( &first_info, 0, sizeof ( pfm_event_info_t ) );
  memset(&pinfo,0,sizeof(pfm_pmu_info_t));

  ret=pfm_get_event_info(code, PFM_OS_PERF_EVENT, &gete);
  //  ret=pfm_get_event_info(first, PFM_OS_PERF_EVENT, &first_info);

  pfm_get_pmu_info(gete.pmu, &pinfo);
  /* VMW */
  /* FIXME, make a "is it the default" function */

  if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
       strcmp(pinfo.name,"ix86arch")) {
    //  if (gete.pmu==first_info.pmu) {
     *event_name=strdup(gete.name);
  }
  else {
     sprintf(name,"%s::%s",pinfo.name,gete.name);
     *event_name=strdup(name);
  }

  SUBDBG("Found name: %s\n",*event_name);

  return ret;

}

static int convert_pfmidx_to_native(int code, unsigned int *PapiEventCode) {

  int ret;
  char *name=NULL;

  ret=convert_libpfm4_to_string( code, &name);
  SUBDBG("Converted %x to %s\n",code,name);
  if (ret==PFM_SUCCESS) {
     ret=_papi_pfm_ntv_name_to_code(name,PapiEventCode);
     SUBDBG("RETURNING FIRST: %x %s\n",*PapiEventCode,name);
  }

  if (name) free(name);
  return ret;

}




static int find_next_umask(struct native_event_t *current_event,
                           int current,char *umask_name) {

  char temp_string[BUFSIZ];
  pfm_event_info_t event_info;
  pfm_event_attr_info_t *ainfo=NULL;
  int num_masks=0;
  pfm_err_t ret;
  int i;
  //  int actual_val=0;

  /* get number of attributes */

  memset(&event_info, 0, sizeof(event_info));
  ret=pfm_get_event_info(current_event->perfmon_idx, PFM_OS_PERF_EVENT, &event_info);
	
  SUBDBG("%d possible attributes for event %s\n",
	 event_info.nattrs,
	 event_info.name);

  ainfo = malloc(event_info.nattrs * sizeof(*ainfo));
  if (!ainfo) {
     return PAPI_ENOMEM;
  }

  pfm_for_each_event_attr(i, &event_info) {
     ainfo[i].size = sizeof(*ainfo);

     ret = pfm_get_event_attr_info(event_info.idx, i, PFM_OS_PERF_EVENT, 
				   &ainfo[i]);
     if (ret != PFM_SUCCESS) {
        SUBDBG("Not found\n");
        if (ainfo) free(ainfo);
	return PAPI_ENOEVNT;
     }

     if (ainfo[i].type == PFM_ATTR_UMASK) {
	SUBDBG("nm %d looking for %d\n",num_masks,current);
	if (num_masks==current+1) {	  
	   SUBDBG("Found attribute %d: %s type: %d\n",i,ainfo[i].name,ainfo[i].type);
	
           sprintf(temp_string,"%s",ainfo[i].name);
           strncpy(umask_name,temp_string,BUFSIZ);

	   if (ainfo) free(ainfo);
	   return current+1;
	}
	num_masks++;
     }
  }

  if (ainfo) free(ainfo);
  return -1;

}

int
_papi_pfm_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{
	int code,ret;
	struct native_event_t *current_event;

        SUBDBG("ENTER\n");

	/* return first event if so specified */
	if ( modifier == PAPI_ENUM_FIRST ) {
	   unsigned int blah=0;
           SUBDBG("ENUM_FIRST\n");

	   code=papi_pfm_get_event_first_active();
	   ret=convert_pfmidx_to_native(code, &blah);
	   *PapiEventCode=(unsigned int)blah;
           SUBDBG("FOUND %x (from %x) ret=%d\n",*PapiEventCode,code,ret);

	   return ret;
	}

	current_event=find_existing_event_by_number(*PapiEventCode);
	if (current_event==NULL) {
           SUBDBG("EVENTS %x not found\n",*PapiEventCode);
	   return PAPI_ENOEVNT;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
	   SUBDBG("ENUM_EVENTS %x\n",*PapiEventCode);
	   unsigned int blah=0;

	   code=current_event->perfmon_idx;

	   ret=find_next_no_aliases(code);

	   SUBDBG("find_next_no_aliases() Returned %x\n",ret);
	   if (ret<0) {
	      SUBDBG("<0 so returning\n");
	      return ret;
	   }

	   SUBDBG("VMW BLAH1\n");

	   ret=convert_pfmidx_to_native(ret, &blah);

	   SUBDBG("VMW BLAH2\n");

	     if (ret<0) {
	       SUBDBG("Couldn't convert to native %d %s\n",
		      ret,PAPI_strerror(ret));
	     }
	     *PapiEventCode=(unsigned int)blah;

	     if ((ret!=PAPI_OK) && (blah!=0)) {
	        SUBDBG("Faking PAPI_OK because blah!=0\n");
	        return PAPI_OK;
	     }

             SUBDBG("Returning PAPI_OK\n");
	     return ret;

	}

	if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		return PAPI_ENOEVNT;
	} 

	if ( modifier == PAPI_NTV_ENUM_UMASKS ) {

	   int max_umask,next_umask;
	   char umask_string[BUFSIZ],new_name[BUFSIZ];

	   SUBDBG("Finding maximum mask in event %s\n",
		  		  current_event->allocated_name);

	   max_umask=find_max_umask(current_event);
	   SUBDBG("Found max %d\n",max_umask);
	   next_umask=find_next_umask(current_event,max_umask,
				      umask_string);
	   SUBDBG("Found next %d\n",next_umask);
	   if (next_umask>=0) {
	     unsigned int blah;
	      sprintf(new_name,"%s:%s",current_event->base_name,
		     umask_string);
     
              ret=_papi_pfm_ntv_name_to_code(new_name,&blah);
	      if (ret!=PAPI_OK) return PAPI_ENOEVNT;

	      *PapiEventCode=(unsigned int)blah;
	      SUBDBG("found code %x\n",*PapiEventCode);
	      return PAPI_OK;
	   }

	   SUBDBG("couldn't find umask\n");

	   return PAPI_ENOEVNT;

	} else {
		return PAPI_EINVAL;
	}
}


int
_papi_pfm_ntv_code_to_bits( unsigned int EventCode, hwd_register_t *bits )
{

  *(int *)bits=EventCode;

  return PAPI_OK;
}


/* This function would return info on which counters an event could be in */
/* libpfm4 currently does not support this */

int
_papi_pfm_ntv_bits_to_info( hwd_register_t * bits, char *names,
			    unsigned int *values, int name_len, int count )
{

  (void)bits;
  (void)names;
  (void)values;
  (void)name_len;
  (void)count;

  return PAPI_OK;

}

int _papi_pfm3_vendor_fixups(void) {

	if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
		/* powerpc */
		MY_VECTOR.cmp_info.available_domains |=
			PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
		if ( strcmp( _papi_hwi_system_info.hw_info.model_string, "POWER6" ) ==
			 0 ) {
			MY_VECTOR.cmp_info.default_domain =
				PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
		}
	} else {
		MY_VECTOR.cmp_info.available_domains |= PAPI_DOM_KERNEL;
	}

	if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
		MY_VECTOR.cmp_info.available_domains |= PAPI_DOM_OTHER;
	}

	if ( ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL ) ||
		 ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD ) ) {
		MY_VECTOR.cmp_info.fast_counter_read = 1;
		MY_VECTOR.cmp_info.fast_real_timer = 1;
		MY_VECTOR.cmp_info.cntr_umasks = 1;
	}
	return PAPI_OK;
}


int
_papi_pfm3_init(void) {

  int detected_pmus=0, found_default=0;
   pfm_pmu_info_t default_pmu;
 
   int i, version;
   pfm_err_t retval;
   unsigned int ncnt;
   pfm_pmu_info_t pinfo;

   /* The following checks the version of the PFM library
      against the version PAPI linked to... */
   if ( ( retval = pfm_initialize(  ) ) != PFM_SUCCESS ) {
      PAPIERROR( "pfm_initialize(): %s", pfm_strerror( retval ) );
      return PAPI_ESBSTR;
   }

   /* get the libpfm4 version */
   SUBDBG( "pfm_get_version()\n");
   if ( (version=pfm_get_version( )) < 0 ) {
      PAPIERROR( "pfm_get_version(): %s", pfm_strerror( retval ) );
      return PAPI_ESBSTR;
   }

   /* Set the version */
   sprintf( MY_VECTOR.cmp_info.support_version, "%d.%d",
	    PFM_MAJ_VERSION( version ), PFM_MIN_VERSION( version ) );

   /* Complain if the compiled-against version doesn't match current version */
   if ( PFM_MAJ_VERSION( version ) != PFM_MAJ_VERSION( LIBPFM_VERSION ) ) {
      PAPIERROR( "Version mismatch of libpfm: compiled %x vs. installed %x\n",
				   PFM_MAJ_VERSION( LIBPFM_VERSION ),
				   PFM_MAJ_VERSION( version ) );
      return PAPI_ESBSTR;
   }

   /* Count number of present PMUs */
   detected_pmus=0;
   ncnt=0;
   /* need to init pinfo or pfmlib might complain */
   memset(&pinfo, 0, sizeof(pfm_pmu_info_t));
   /* init default pmu */
   retval=pfm_get_pmu_info(0, &default_pmu);
   
   SUBDBG("Detected pmus:\n");
   for(i=0;i<PFM_PMU_MAX;i++) {
      retval=pfm_get_pmu_info(i, &pinfo);
      if (retval!=PFM_SUCCESS) continue;
      if (pinfo.is_present) {
	SUBDBG("\t%d %s %s %d\n",i,pinfo.name,pinfo.desc,pinfo.type);

         detected_pmus++;
	 ncnt+=pinfo.nevents;
	 if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
              strcmp(pinfo.name,"ix86arch")) {

	    SUBDBG("\t  %s is default\n",pinfo.name);
	    memcpy(&default_pmu,&pinfo,sizeof(pfm_pmu_info_t));
	    found_default++;
	 }
      }
   }
   SUBDBG("%d native events detected on %d pmus\n",ncnt,detected_pmus);

   if (!found_default) {
      PAPIERROR("Could not find default PMU\n");
      return PAPI_ESBSTR;
   }

   if (found_default>1) {
     PAPIERROR("Found too many default PMUs!\n");
     return PAPI_ESBSTR;
   }

   MY_VECTOR.cmp_info.num_native_events = ncnt;

   MY_VECTOR.cmp_info.num_cntrs = default_pmu.num_cntrs+
                                  default_pmu.num_fixed_cntrs;
   SUBDBG( "num_counters: %d\n", MY_VECTOR.cmp_info.num_cntrs );

   MY_VECTOR.cmp_info.num_mpx_cntrs = MAX_MPX_EVENTS;
   
   /* Setup presets */
   retval = _papi_pfm_setup_presets( (char *)default_pmu.name, 
				     default_pmu.pmu );
   if ( retval )
      return retval;
	
   return PAPI_OK;
}

int
_papi_pfm3_setup_counters( struct perf_event_attr *attr,
			   hwd_register_t *ni_bits ) {

  int ret;
  int our_idx;
  char our_name[BUFSIZ];
   
  pfm_perf_encode_arg_t perf_arg;

  memset(&perf_arg,0,sizeof(pfm_perf_encode_arg_t));
  perf_arg.attr=attr;
   
  our_idx=*(int *)(ni_bits);

  _papi_pfm_ntv_code_to_name( our_idx,our_name,BUFSIZ);

  SUBDBG("trying %s %x\n",our_name,our_idx);

  ret = pfm_get_os_event_encoding(our_name, 
				  PFM_PLM0 | PFM_PLM3, 
                                  PFM_OS_PERF_EVENT_EXT, 
				  &perf_arg);
  if (ret!=PFM_SUCCESS) {
     return PAPI_ENOEVNT;
  }
  
  SUBDBG( "pe_event: config 0x%"PRIu64" config2 0x%"PRIu64" type 0x%"PRIu32"\n", 
          perf_arg.attr->config1, 
	  perf_arg.attr->config2,
	  perf_arg.attr->type);
	  

  return PAPI_OK;
}

