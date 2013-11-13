/* 
* File:    papi_preset.c
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
* Author:  Vince Weaver 
*          vweaver1 @ eecs.utk.edu
*          Merge of the libpfm3/libpfm4/pmapi-ppc64_events preset code
*/


#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_preset.h"
#include "extras.h"


/* This routine copies values from a dense 'findem' array of events 
   into the sparse global _papi_hwi_presets array, which is assumed 
   to be empty at initialization. 

   Multiple dense arrays can be copied into the sparse array, allowing 
   event overloading at run-time, or allowing a baseline table to be 
   augmented by a model specific table at init time. 

   This method supports adding new events; overriding existing events, or
   deleting deprecated events.
*/
int
_papi_hwi_setup_all_presets( hwi_search_t * findem, int cidx )
{
    int i, pnum, did_something = 0;
    unsigned int preset_index, j, k;

    /* dense array of events is terminated with a 0 preset.
       don't do anything if NULL pointer. This allows just notes to be loaded.
       It's also good defensive programming. 
     */
    if ( findem != NULL ) {
       for ( pnum = 0; ( pnum < PAPI_MAX_PRESET_EVENTS ) &&
			  ( findem[pnum].event_code != 0 ); pnum++ ) {
	   /* find the index for the event to be initialized */
	   preset_index = ( findem[pnum].event_code & PAPI_PRESET_AND_MASK );
	   /* count and set the number of native terms in this event, 
              these items are contiguous.

	      PAPI_EVENTS_IN_DERIVED_EVENT is arbitrarily defined in the high 
              level to be a reasonable number of terms to use in a derived 
              event linear expression, currently 8.

	      This wastes space for components with less than 8 counters, 
              but keeps the framework independent of the components.

	      The 'native' field below is an arbitrary opaque identifier 
              that points to information on an actual native event. 
              It is not an event code itself (whatever that might mean).
	      By definition, this value can never == PAPI_NULL.
	      - dkt */

	   INTDBG( "Counting number of terms for preset index %d, "
                   "search map index %d.\n", preset_index, pnum );
	   i = 0;
	   j = 0;
	   while ( i < PAPI_EVENTS_IN_DERIVED_EVENT ) {
	      if ( findem[pnum].native[i] != PAPI_NULL ) {
		 j++;
	      }
	      else if ( j ) {
		 break;
	      }
	      i++;
	   }

	   INTDBG( "This preset has %d terms.\n", j );
	   _papi_hwi_presets[preset_index].count = j;
 
           _papi_hwi_presets[preset_index].derived_int = findem[pnum].derived;
	   for(k=0;k<j;k++) {
              _papi_hwi_presets[preset_index].code[k] = 
                     findem[pnum].native[k];
	   }
	   /* preset code list must be PAPI_NULL terminated */
	   if (k<PAPI_EVENTS_IN_DERIVED_EVENT) {
              _papi_hwi_presets[preset_index].code[k] = PAPI_NULL;
	   }

	   _papi_hwi_presets[preset_index].postfix=
	                                   strdup(findem[pnum].operation);

	   did_something++;
       }
    }

    _papi_hwd[cidx]->cmp_info.num_preset_events += did_something;

    return ( did_something ? PAPI_OK : PAPI_ENOEVNT );
}

int
_papi_hwi_cleanup_all_presets( void )
{
        int preset_index,cidx;
	unsigned int j;

	for ( preset_index = 0; preset_index < PAPI_MAX_PRESET_EVENTS;
		  preset_index++ ) {
	    if ( _papi_hwi_presets[preset_index].postfix != NULL ) {
	       free( _papi_hwi_presets[preset_index].postfix );
	       _papi_hwi_presets[preset_index].postfix = NULL;
	    }
	    if ( _papi_hwi_presets[preset_index].note != NULL ) {
	       free( _papi_hwi_presets[preset_index].note );
	       _papi_hwi_presets[preset_index].note = NULL;
	    }
	    for(j=0; j<_papi_hwi_presets[preset_index].count;j++) {
	       free(_papi_hwi_presets[preset_index].name[j]);
	    }
	}
	
	for(cidx=0;cidx<papi_num_components;cidx++) {
	   _papi_hwd[cidx]->cmp_info.num_preset_events = 0;
	}

#if defined(ITANIUM2) || defined(ITANIUM3)
	/* NOTE: This memory may need to be freed for BG/P builds as well */
	if ( preset_search_map != NULL ) {
		papi_free( preset_search_map );
		preset_search_map = NULL;
	}
#endif

	return PAPI_OK;
}



#define PAPI_EVENT_FILE "papi_events.csv"


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
	return note;
}

static inline int
find_preset_code( char *tmp, int *code )
{
    int i = 0;

    while ( _papi_hwi_presets[i].symbol != NULL ) {
	  if ( strcasecmp( tmp, _papi_hwi_presets[i].symbol ) == 0 ) {
	     *code = ( int ) ( i | PAPI_PRESET_MASK );
	     return PAPI_OK;
	  }
	  i++;
    }
    return PAPI_EINVAL;
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
		SUBDBG( "Open %s failed, trying ./%s.\n", 
			name, PAPI_EVENT_FILE );
		sprintf( name, "%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table == NULL ) {
		SUBDBG( "Open ./%s failed, trying ../%s.\n", 
			name, PAPI_EVENT_FILE );
		sprintf( name, "../%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table ) {
		SUBDBG( "Open %s succeeded.\n", name );
	}
	return table;
}

/* parse a single line from either a file or character table
   Strip trailing <cr>; return 0 if empty */
static int
get_event_line( char *line, FILE * table, char **tmp_perfmon_events_table )
{
	int i;

	if ( table ) {
	    if ( fgets( line, LINE_MAX, table ) == NULL)
		return 0;

	    i = ( int ) strlen( line );
	    if (i == 0)
		return 0;
	    if ( line[i-1] == '\n' )
		line[i-1] = '\0';
	    return 1;
	} else {
		for ( i = 0;
			  **tmp_perfmon_events_table && **tmp_perfmon_events_table != '\n';
			  i++, ( *tmp_perfmon_events_table )++ ) 
			line[i] = **tmp_perfmon_events_table;
		if (i == 0)
		    return 0;
		if ( **tmp_perfmon_events_table && **tmp_perfmon_events_table == '\n' ) {
		    ( *tmp_perfmon_events_table )++;
		}
		line[i] = '\0';
		return 1;
	}
}

/* Static version of the events file. */
#if defined(STATIC_PAPI_EVENTS_TABLE)
#include "papi_events_table.h"
#else
static char *papi_events_table = NULL;
#endif

int
_papi_load_preset_table( char *pmu_str, int pmu_type, int cidx)
{

  (void) cidx;  /* We'll use this later */

    char pmu_name[PAPI_MIN_STR_LEN];
    char line[LINE_MAX];
    char name[PATH_MAX] = "builtin papi_events_table";
    char *tmp_papi_events_table = NULL;
    char *tmpn;
    FILE *table;
    int ret;
    unsigned int event_idx;
    int invalid_event;
    int line_no = 1, derived = 0, insert = 0, preset = 0;
    int get_presets = 0;   /* only get PRESETS after CPU is identified      */
    int found_presets = 0; /* only terminate search after PRESETS are found */
	                   /* this allows support for synonyms for CPU names*/

    SUBDBG("ENTER\n");

    /* copy the pmu identifier, stripping commas if found */
    tmpn = pmu_name;
    while ( *pmu_str ) {
       if ( *pmu_str != ',' ) *tmpn++ = *pmu_str;
       pmu_str++;
    }
    *tmpn = '\0';

    /* try the environment variable first */
    if ( ( tmpn = getenv( "PAPI_CSV_EVENT_FILE" ) ) && 
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
       PAPIERROR( "fopen(%s): %s, please set the PAPI_CSV_EVENT_FILE "
                  "env. variable", name, strerror( errno ) );
       return PAPI_ESYS;
    }

    /* at this point either a valid file pointer or built-in table pointer */
    while ( get_event_line( line, table, &tmp_papi_events_table ) ) {
       char *t;
       int i;

       t = trim_string( strtok( line, "," ) );

       /* Skip blank lines */
       if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) continue;

       /* Skip comments */
       if ( t[0] == '#' ) {
	  goto nextline;
       } 

       if ( strcasecmp( t, "CPU" ) == 0 ) {

	  if ( get_presets != 0 && found_presets != 0 ) {
	     SUBDBG( "Ending preset scanning at line %d of %s.\n", 
                     line_no, name );

	     get_presets=0; found_presets=0;
				
	  }
			
	  t = trim_string( strtok( NULL, "," ) );
	  if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
	     PAPIERROR( "Expected name after CPU token at line %d of %s "
			"-- ignoring", line_no, name );
	     goto nextline;
	  }

	  SUBDBG( "Examining CPU (%s) vs. (%s)\n", t, pmu_name );

	  if ( strcasecmp( t, pmu_name ) == 0 ) {
	     int type;

	     SUBDBG( "Found CPU %s at line %d of %s.\n", t, line_no, name );

	     t = trim_string( strtok( NULL, "," ) );
	     if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
		SUBDBG("No additional qualifier found, matching on string.\n");

		get_presets = 1;
	     } else if ( ( sscanf( t,"%d",&type )==1) && (type==pmu_type) ) {
                SUBDBG( "Found CPU %s type %d at line %d of %s.\n",
			pmu_name, type, line_no, name );
		get_presets = 1;
	     } else {
		SUBDBG( "Additional qualifier match failed %d vs %d.\n",
			pmu_type, type );

	     }
	  }
       } else if ( strcasecmp( t, "PRESET" ) == 0 ) {

          if ( get_presets == 0 ) goto nextline;

	  found_presets = 1;
	  t = trim_string( strtok( NULL, "," ) );

	  if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
			       
             PAPIERROR( "Expected name after PRESET token at line %d of %s "
			"-- ignoring", line_no, name );
	     goto nextline;
	  }

	  SUBDBG( "Examining preset %s\n", t );

	  if ( find_preset_code( t, &preset ) != PAPI_OK ) {
	     PAPIERROR ( "Invalid preset name %s after PRESET token "
			 "at line %d of %s -- ignoring",
			 t, line_no, name );
	     goto nextline;
	  }

	  SUBDBG( "Found 0x%08x for %s\n", preset, t );

	  t = trim_string( strtok( NULL, "," ) );
	  if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
	     PAPIERROR( "Expected derived type after PRESET token at "
                        "line %d of %s -- ignoring", line_no, name );
	     goto nextline;
	  }

	  if ( _papi_hwi_derived_type( t, &derived ) != PAPI_OK ) {
	     PAPIERROR( "Invalid derived name %s after PRESET token at "
			"line %d of %s -- ignoring",
			t, line_no, name );
	     goto nextline;
	  }

	  /****************************************/
	  /* Have a preset, let's start assigning */
	  /****************************************/

	  SUBDBG( "Found %d for %s\n", derived, t );
	  SUBDBG( "Adding %#x,%d to preset search table.\n", 
		  preset, derived );
	  
	  insert=preset&PAPI_PRESET_AND_MASK;

	  /* _papi_hwi_presets[insert].event_code = preset; */
	  _papi_hwi_presets[insert].derived_int = derived;

	  /* Derived support starts here */
	  /* Special handling for postfix */
	  if ( derived == DERIVED_POSTFIX ) {
	     t = trim_string( strtok( NULL, "," ) );
	     if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
		PAPIERROR( "Expected Operation string after derived type "
			   "DERIVED_POSTFIX at line %d of %s -- ignoring",
			   line_no, name );
		goto nextline;
	     }

	     SUBDBG( "Saving PostFix operations %s\n", t );

	     _papi_hwi_presets[insert].postfix=strdup(t);
	  }
			
	  /* All derived terms collected here */
	  i = 0;
	  invalid_event=0;
	  do {
	     t = trim_string( strtok( NULL, "," ) );
	     if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) break;
	     if ( strcasecmp( t, "NOTE" ) == 0 ) break;
	     _papi_hwi_presets[insert].name[i]=strdup(t);

	     SUBDBG( "Adding term (%d) %s to preset event %#x.\n", 
		     i, t, preset );

	     SUBDBG("Looking up: %s\n",t);

	     ret=_papi_hwd[cidx]->ntv_name_to_code(t, &event_idx);

	     if (ret==PAPI_OK) {
		_papi_hwi_presets[insert].code[i]=
	              _papi_hwi_native_to_eventcode(cidx,event_idx);
		SUBDBG("Found: %s %#x c%d e%d\n",t,
		       _papi_hwi_presets[insert].code[i],
		       cidx,event_idx);
	     }
	     else {
		PAPIERROR("papi_preset: Error finding event %s",t);
		invalid_event=1;
	     }

	  } while ( ++i < PAPI_EVENTS_IN_DERIVED_EVENT );

	  /* preset code list must be PAPI_NULL terminated */
	  if (i<PAPI_EVENTS_IN_DERIVED_EVENT) {
             _papi_hwi_presets[insert].code[i] = PAPI_NULL;
	  }

	  if (invalid_event) {
	    /* We signify a valid preset if count > 0 */
	     _papi_hwi_presets[insert].count=0;
	  } else {
	     _papi_hwi_presets[insert].count=i;
	  }

	  /* End of derived support */

	  if ( i == 0 ) {
	     PAPIERROR( "Expected PFM event after DERIVED token at "
			"line %d of %s -- ignoring", line_no, name );
	     goto nextline;
	  }
	  if ( i == PAPI_EVENTS_IN_DERIVED_EVENT ) {
	     t = trim_string( strtok( NULL, "," ) );
	  }
			
	  /* Handle optional NOTEs */
	  if ( t && ( strcasecmp( t, "NOTE" ) == 0 ) ) {
	     SUBDBG( "%s found on line %d\n", t, line_no );

	     /* read the rest of the line */
	     t = trim_note( strtok( NULL, "" ) );
	
	     if ( ( t == NULL ) || ( strlen( t ) == 0 ) ) {
		PAPIERROR( "Expected Note string at line %d of %s\n",
			   line_no, name );
	     }
	     else {
	        _papi_hwi_presets[insert].note = strdup( t );
		SUBDBG( "NOTE: --%s-- found on line %d\n", t, line_no );
	     }
	  }
	  _papi_hwd[cidx]->cmp_info.num_preset_events++;

       } else {
	  PAPIERROR( "Unrecognized token %s at line %d of %s -- ignoring", 
		     t, line_no, name );
	  goto nextline;
       }
nextline:
       line_no++;
    }

    if ( table ) {
       fclose( table );
    }

    SUBDBG("Done parsing preset table\n");
	
    return PAPI_OK;
}




/* The following code is proof of principle for reading preset events from an
   xml file. It has been tested and works for pentium3. It relys on the expat
   library and is invoked by adding
   XMLFLAG		= -DXML
   to the Makefile. It is presently hardcoded to look for "./papi_events.xml"
*/
#ifdef XML

#define BUFFSIZE 8192
#define SPARSE_BEGIN 0
#define SPARSE_EVENT_SEARCH 1
#define SPARSE_EVENT 2
#define SPARSE_DESC 3
#define ARCH_SEARCH 4
#define DENSE_EVENT_SEARCH 5
#define DENSE_NATIVE_SEARCH 6
#define DENSE_NATIVE_DESC 7
#define FINISHED 8

char buffer[BUFFSIZE], *xml_arch;
int location = SPARSE_BEGIN, sparse_index = 0, native_index, error = 0;

/* The function below, _xml_start(), is a hook into expat's XML
 * parser.  _xml_start() defines how the parser handles the
 * opening tags in PAPI's XML file.  This function can be understood
 * more easily if you follow along with its logic while looking at
 * papi_events.xml.  The location variable is a global telling us
 * where we are in the XML file.  Have we found our architecture's
 * events yet?  Are we looking at an event definition?...etc.
 */
static void
_xml_start( void *data, const char *el, const char **attr )
{
	int native_encoding;

	if ( location == SPARSE_BEGIN && !strcmp( "papistdevents", el ) ) {
		location = SPARSE_EVENT_SEARCH;
	} else if ( location == SPARSE_EVENT_SEARCH && !strcmp( "papievent", el ) ) {
		_papi_hwi_presets[sparse_index].info.symbol = papi_strdup( attr[1] );
//      strcpy(_papi_hwi_presets.info[sparse_index].symbol, attr[1]);
		location = SPARSE_EVENT;
	} else if ( location == SPARSE_EVENT && !strcmp( "desc", el ) ) {
		location = SPARSE_DESC;
	} else if ( location == ARCH_SEARCH && !strcmp( "availevents", el ) &&
				!strcmp( xml_arch, attr[1] ) ) {
		location = DENSE_EVENT_SEARCH;
	} else if ( location == DENSE_EVENT_SEARCH && !strcmp( "papievent", el ) ) {
		if ( !strcmp( "PAPI_NULL", attr[1] ) ) {
			location = FINISHED;
			return;
		} else if ( PAPI_event_name_to_code( ( char * ) attr[1], &sparse_index )
					!= PAPI_OK ) {
			PAPIERROR( "Improper Preset name given in XML file for %s.",
					   attr[1] );
			error = 1;
		}
		sparse_index &= PAPI_PRESET_AND_MASK;

		/* allocate and initialize data space for this event */
		papi_valid_free( _papi_hwi_presets[sparse_index].data );
		_papi_hwi_presets[sparse_index].data =
			papi_malloc( sizeof ( hwi_preset_data_t ) );
		native_index = 0;
		_papi_hwi_presets[sparse_index].data->native[native_index] = PAPI_NULL;
		_papi_hwi_presets[sparse_index].data->operation[0] = '\0';


		if ( attr[2] ) {	 /* derived event */
			_papi_hwi_presets[sparse_index].data->derived =
				_papi_hwi_derived_type( ( char * ) attr[3] );
			/* where does DERIVED POSTSCRIPT get encoded?? */
			if ( _papi_hwi_presets[sparse_index].data->derived == -1 ) {
				PAPIERROR( "No derived type match for %s in Preset XML file.",
						   attr[3] );
				error = 1;
			}

			if ( attr[5] ) {
				_papi_hwi_presets[sparse_index].count = atoi( attr[5] );
			} else {
				PAPIERROR( "No count given for %s in Preset XML file.",
						   attr[1] );
				error = 1;
			}
		} else {
			_papi_hwi_presets[sparse_index].data->derived = NOT_DERIVED;
			_papi_hwi_presets[sparse_index].count = 1;
		}
		location = DENSE_NATIVE_SEARCH;
	} else if ( location == DENSE_NATIVE_SEARCH && !strcmp( "native", el ) ) {
		location = DENSE_NATIVE_DESC;
	} else if ( location == DENSE_NATIVE_DESC && !strcmp( "event", el ) ) {
		if ( _papi_hwi_native_name_to_code( attr[1], &native_encoding ) !=
			 PAPI_OK ) {
			printf( "Improper Native name given in XML file for %s\n",
					attr[1] );
			PAPIERROR( "Improper Native name given in XML file for %s\n",
					   attr[1] );
			error = 1;
		}
		_papi_hwi_presets[sparse_index].data->native[native_index] =
			native_encoding;
		native_index++;
		_papi_hwi_presets[sparse_index].data->native[native_index] = PAPI_NULL;
	} else if ( location && location != ARCH_SEARCH && location != FINISHED ) {
		PAPIERROR( "Poorly-formed Preset XML document." );
		error = 1;
	}
}

/* The function below, _xml_end(), is a hook into expat's XML
 * parser.  _xml_end() defines how the parser handles the
 * end tags in PAPI's XML file.
 */
static void
_xml_end( void *data, const char *el )
{
	int i;

	if ( location == SPARSE_EVENT_SEARCH && !strcmp( "papistdevents", el ) ) {
		for ( i = sparse_index; i < PAPI_MAX_PRESET_EVENTS; i++ ) {
			_papi_hwi_presets[i].info.symbol = NULL;
			_papi_hwi_presets[i].info.long_descr = NULL;
			_papi_hwi_presets[i].info.short_descr = NULL;
		}
		location = ARCH_SEARCH;
	} else if ( location == DENSE_NATIVE_DESC && !strcmp( "native", el ) ) {
		location = DENSE_EVENT_SEARCH;
	} else if ( location == DENSE_EVENT_SEARCH && !strcmp( "availevents", el ) ) {
		location = FINISHED;
	}
}

/* The function below, _xml_content(), is a hook into expat's XML
 * parser.  _xml_content() defines how the parser handles the
 * text between tags in PAPI's XML file.  The information between
 * tags is usally text for event descriptions.
 */
static void
_xml_content( void *data, const char *el, const int len )
{
	int i;
	if ( location == SPARSE_DESC ) {
		_papi_hwi_presets[sparse_index].info.long_descr =
			papi_malloc( len + 1 );
		for ( i = 0; i < len; i++ )
			_papi_hwi_presets[sparse_index].info.long_descr[i] = el[i];
		_papi_hwi_presets[sparse_index].info.long_descr[len] = '\0';
		/* the XML data currently doesn't contain a short description */
		_papi_hwi_presets[sparse_index].info.short_descr = NULL;
		sparse_index++;
		_papi_hwi_presets[sparse_index].data = NULL;
		location = SPARSE_EVENT_SEARCH;
	}
}

int
_xml_papi_hwi_setup_all_presets( char *arch, hwi_dev_notes_t * notes )
{
	int done = 0;
	FILE *fp = fopen( "./papi_events.xml", "r" );
	XML_Parser p = XML_ParserCreate( NULL );

	if ( !p ) {
		PAPIERROR( "Couldn't allocate memory for XML parser." );
		fclose(fp);
		return ( PAPI_ESYS );
	}
	XML_SetElementHandler( p, _xml_start, _xml_end );
	XML_SetCharacterDataHandler( p, _xml_content );
	if ( fp == NULL ) {
		PAPIERROR( "Error opening Preset XML file." );
		fclose(fp);
		return ( PAPI_ESYS );
	}

	xml_arch = arch;

	do {
		int len;
		void *buffer = XML_GetBuffer( p, BUFFSIZE );

		if ( buffer == NULL ) {
			PAPIERROR( "Couldn't allocate memory for XML buffer." );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		len = fread( buffer, 1, BUFFSIZE, fp );
		if ( ferror( fp ) ) {
			PAPIERROR( "XML read error." );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		done = feof( fp );
		if ( !XML_ParseBuffer( p, len, len == 0 ) ) {
			PAPIERROR( "Parse error at line %d:\n%s\n",
					   XML_GetCurrentLineNumber( p ),
					   XML_ErrorString( XML_GetErrorCode( p ) ) );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		if ( error ) {
			fclose(fp);
			return ( PAPI_ESYS );
		}
	} while ( !done );
	XML_ParserFree( p );
	fclose( fp );
	return ( PAPI_OK );
}
#endif
