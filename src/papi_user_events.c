/** 
 *	@file papi_user_events.c
 *	@author James Ralph
 *			ralph@eecs.utk.edu
 */



/* TODO:
 *		Think about how to support components
 *		Software multiplexing improvments 
 *		Error bounds on multiplexed counts
 *
 *		Keywords for getting info out of hw_info or something like it. 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"  /* for _papi__hwd[].  Should we modify the */
                          /* code to not depend on component 0?      */
#include "papi_memory.h"
#include "papi_user_events.h"

#ifdef STATIC_USER_EVENTS
#include "papi_static_user_events.h"
#endif

#define CONSTANT_DELIM "#define"

#define SHOW_LOADS
extern unsigned int PAPI_NATIVE_EVENT_SHIFT;
extern unsigned int PAPI_NATIVE_UMASK_SHIFT;

typedef struct def_list {
  char name[PAPI_MIN_STR_LEN];
  char value[PAPI_MIN_STR_LEN];
  struct def_list *next;
} list_t;


user_defined_event_t *	_papi_user_events = NULL;
unsigned int			_papi_user_events_count = 0;
list_t					defines;
int						first_time = 1;


void
PRINT_UE(user_defined_event_t* ue)
{
#ifndef DEBUG
  (void)ue; /* clean up unused parameter warning if debugging messages are not used */
#endif
  INTDBG("User Event debug\n");
  INTDBG("\tsymbol=%s\n", ue->symbol);
  INTDBG("\toperation=%s\n", ue->operation);
  INTDBG("\tcount=%d\n", ue->count);
}

void
_papi_cleanup_user_events()
{
  unsigned int i;
  user_defined_event_t *t;
  list_t *a,*b;

  for ( i = 0; i < _papi_user_events_count; i++ ) {
	t = _papi_user_events + i;

	if ( t->short_desc != NULL )
	  free(t->short_desc);
	if ( t->long_desc != NULL )
	  free(t->long_desc);

  }
  
  if ( _papi_user_events != NULL )
	papi_free( _papi_user_events );

  _papi_user_events = NULL;
  _papi_user_events_count = 0;

  /* cleanup the defines list too */
  for ( a = defines.next; a != NULL; ) {
	b=a->next;
	papi_free(a);
	a = b;
  }
}

void
append_to_global_list(user_defined_event_t *more ) 
{  
  user_defined_event_t* new;

  new = papi_malloc(sizeof(user_defined_event_t) * (1 + _papi_user_events_count) );
  if (new == NULL) {
	PAPIERROR("Unable to allocate %d bytes of memory.\n", sizeof(user_defined_event_t)*(1+_papi_user_events_count));
  } else {
	if (_papi_user_events != NULL) {
	  memcpy(new, _papi_user_events, _papi_user_events_count*sizeof(user_defined_event_t));
	  papi_free(_papi_user_events);
	}

	memcpy( new + _papi_user_events_count, more, sizeof(user_defined_event_t)  );

	_papi_user_events = new;
	_papi_user_events_count++;
  }
}

int 
is_operation(char *op) 
{
  char first = op[0];

  switch(first) {
  case '+':
  case '-':
  case '*':
  case '/':
  case '%':
	return 1;
  default:
	return 0;
  }
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


/* parse a single line from either a file or character table
   Strip trailing <cr>; return 0 if empty */
static int
get_event_line( char **place, FILE * table, char **tmp_perfmon_events_table )
{
	int ret = 0;
	int i;
	char *line = place[0];
	int c = 0;

	if ( table ) {
	  if ( fgets( place[0], USER_EVENT_OPERATION_LEN, table ) ) { 

		i = strlen( place[0] );
		c = place[0][i-1];
		/* throw away the rest of the line. */
		while ( c != '\n' && c != EOF ) 
		  c = fgetc( table );

		ret = i;
		line = place[0]; 

		for ( i = 0; i < (int)strlen( place[0] ); i++ )
		  if ( place[0][i] == '\n' )
			place[0][i] = 0;
	  } else
		ret = 0;
	} else {
		for ( i = 0;
			  **tmp_perfmon_events_table && **tmp_perfmon_events_table != '\n' && 
			  i < USER_EVENT_OPERATION_LEN;
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

int add_define( char *line, list_t* LIST ) {
  char *t;
  char local_line[USER_EVENT_OPERATION_LEN];
  list_t *temp;

  strncpy( local_line, line, USER_EVENT_OPERATION_LEN );

  temp = (list_t*)papi_malloc(sizeof(list_t));

  if ( NULL == temp ) {
	PAPIERROR("outof memory" );
	return PAPI_ENOMEM;
  }

  strtok(local_line, " "); /* throw out the #define */
  
  /* next token should be the name */
  t = strtok(NULL, " ");
  strncpy( temp->name, t, PAPI_MIN_STR_LEN);

  /* next token should be the value */
  t = strtok(NULL," ");
  t[strlen(t)] = '\0';
  strncpy( temp->value, t, PAPI_MIN_STR_LEN);

  temp->next = LIST->next;
  LIST->next = temp;

  return PAPI_OK;
}

int renumber_ops_string(char *dest, char *src, int start) {
  char *saveptr, *subtoken;
  char *pass;
  char temp[PAPI_MIN_STR_LEN];
  char orig[PAPI_MIN_STR_LEN];

  memcpy(orig, src, PAPI_MIN_STR_LEN);
  pass = orig;

  while ( (subtoken = strtok_r( pass, "|", &saveptr)) != NULL ) {
	pass = NULL;
	if ( subtoken[0] == 'N' ) {
	  sprintf(temp, "N%d|", start++);
	  strcat(dest, temp);
	} else {
		strcat(dest, subtoken);
		strcat(dest, "|");
	} 
  }
  return start;
}

int is_define(char *t, list_t *next, char *ops) {
  int found = 0;
  char temp[PAPI_MIN_STR_LEN];
  int len = 0;

  /* check if its a predefined constant */
  while( next != (void*) NULL ) {
	if ( ! strcmp(t, next->name) ) {
	  sprintf(temp,"%s|", next->value);
	  len = strlen(ops) + strlen(temp);
	  if ( len >= USER_EVENT_OPERATION_LEN ) {
		fprintf(stderr,"outa room");
		return -1;
	  }

	  strcat(ops,temp);
	  found = 1;
	  break;
	}
	next = next->next;
  }
  return (found);
}

int 
check_native_events(char *target, user_defined_event_t* ue, int* msi)
{
  char temp[PAPI_MIN_STR_LEN];
  int found = 0;
  int ret;
  int magic_string_int = *msi;
  int len = 0;

  if ( ( ret = _papi_hwi_native_name_to_code( target, &ue->events[ue->count] ) ) == PAPI_OK ) {
	if ( ue->events[ue->count] == 0 ) {
	  INTDBG( "Event %s not found!\n", target);
	} else {
	  found = 1;
	  sprintf(temp, "N%d|", magic_string_int++);
	  len = strlen(ue->operation) + strlen(temp);
	  ue->count++;
#ifdef SHOW_LOADS
	  INTDBG("\tFound a native event %s\n", target);
#endif
	  if ( len >= USER_EVENT_OPERATION_LEN ) {
		fprintf(stderr,"outa room");
		return -1;
	  }
	  strcat(ue->operation, temp);
	  *msi = magic_string_int;
	}
  } 

  return found;
}

int 
check_preset_events (char *target, user_defined_event_t* ue, int* msi)
{
  char op;
  int j	= 0;
  int k;
  int found	= 0;
  int magic_string_int = *msi;

  char temp[PAPI_MIN_STR_LEN];

  /* XXX make sure that we don't overflow the ue->operation buffer */
  int length = PAPI_MIN_STR_LEN;

  memset(temp, 0, PAPI_MIN_STR_LEN);
  for ( j = 0; ( j < PAPI_MAX_PRESET_EVENTS) && (_papi_hwi_presets[j].symbol != NULL ); j++ ) {
	if ( strcasecmp( target, _papi_hwi_presets[j].symbol ) == 0) {
#ifdef SHOW_LOADS
	  INTDBG("\tFound a match for preset event %s\n", _papi_hwi_presets[j].symbol);
#endif
	  /* Check that the preset event we're trying to add is actually available on this system */
	  if ( _papi_hwi_presets[j].count == 0 ) {
		PAPIERROR("NEXTLINE:\t%s is not available on this platform. Skipping event %s\n", 
			target, ue->symbol);
		/* clean up this and ditch this whole line */
		memset(ue, 0, sizeof(user_defined_event_t));
		return -1;
	  } 

	  length = strlen(ue->operation);

	  /* Deal with singleton events */
	  if (!_papi_hwi_presets[j].derived_int) {
		sprintf(temp, "N%d|", magic_string_int++);
		length = strlen(ue->operation) + strlen(temp);
		if ( length >= USER_EVENT_OPERATION_LEN ) {
		  fprintf(stderr,"Out of room, the user defined event %s is too large!\n", ue->symbol );
		  return -1;
		}
		strcat(ue->operation, temp);
		ue->events[ue->count++] = _papi_hwi_presets[j].code[0];
	  } else {
		op = '-';
		switch ( _papi_hwi_presets[j].derived_int ) {
		  case DERIVED_ADD:
		  case DERIVED_ADD_PS:
			op = '+';
		  case DERIVED_SUB:
			for ( k = 0; k < (int) _papi_hwi_presets[j].count; k++) {
			  ue->events[ue->count++] = _papi_hwi_presets[j].code[k];
			  if (k%2)
				sprintf(temp, "N%d|%c|", magic_string_int++, op);
			  else 
				sprintf(temp, "N%d|", magic_string_int++);

			  length = strlen(ue->operation) + strlen(temp);
			  if ( USER_EVENT_OPERATION_LEN <= length ) {
				PAPIERROR("The user defined event %s has to may operands in its definition.\n", ue->symbol );
				return -1;
			  }
			  strcat(ue->operation, temp);
			}
			break;

		  case DERIVED_POSTFIX: 
			for ( k = 0; k < (int)_papi_hwi_presets[j].count; k++ ) {
			  ue->events[ue->count++] = _papi_hwi_presets[j].code[k];
			}
			/* so we need to go through the ops string and renumber the N's
			   as we place it in our ue ops string */
			magic_string_int = renumber_ops_string(temp, 
				_papi_hwi_presets[j].postfix, magic_string_int);
			length = strlen( temp ) + strlen( ue->operation );
			if ( length >= USER_EVENT_OPERATION_LEN ) {
			  PAPIERROR( "User Event %s's expression is too long.", ue->symbol );
			  return -1;
			}
			strcat(ue->operation, temp);
		  default:
			break;
		} /* /switch */
	  } /* /derived */ 
	  found = 1;
	  break;
	} /* /symbol match */

  } /* end while(preset events) */

  *msi = magic_string_int;
  return found;
}

int 
check_user_events(char *target, user_defined_event_t* ue, int* msi, user_defined_event_t* search, int search_size)
{
  char temp[PAPI_MIN_STR_LEN];
  int j		= 0;
  int k		= 0;
  int found = 0;
  int magic_string_int = *msi;
  int len = 0;

  memset(temp, 0, PAPI_MIN_STR_LEN);
  for (j=0; j < search_size; j++) {
	if ( strcasecmp( target, search[j].symbol) == 0 ) {
#ifdef SHOW_LOADS
	  INTDBG("\tFount a match for user event %s at search[%d]\n", search[j].symbol, j );
#endif

	  for ( k = 0; k < (int)search[j].count; k++ ) {
		ue->events[ue->count++] = search[j].events[k];
	  }

	  /* so we need to go through the ops string and renumber the N's
		 as we place it in our ue ops string */
	  magic_string_int = renumber_ops_string(temp, 
		  search[j].operation, magic_string_int);

	  len = strlen( temp ) + strlen( ue->operation );
	  if ( len >= USER_EVENT_OPERATION_LEN ) {
		PAPIERROR( "User Event %s is trying to use an invalid expression, its too long.", ue->symbol );
		return -1;
	  }
	  strcat(ue->operation, temp);
	  found = 1;
	}
  }

  *msi = magic_string_int;
  return found;
}

/*
 *	name, expr (events to lookup later)...	
 *
 *	Do we keep rpn? Probably... 
 *	but DATA_CACHE_MISSES|INSTRUCTION_CACHE_MISSES|+ is servicable...
 *
 *	Do we keep the csv format?
 *	N0|N1|+|n2|* == (n0+n1)*n2 ??
 *	if not, we do have to still create the string 
 *	so that add_events can parse it...
 *	
 *	Ok, how do we denote where the event lives (and if there are more than one, what do we do)
 *	COMP_NAME:EVENT_NAME ?
 *	COMP_NAME:SUB_OPT:EVENT_NAME ?
 *	CPU:0:L3_CACHE_FILLS ?
 */

static int 
load_user_event_table( char *file_name)
{
  char *line;
  char temp[PAPI_MIN_STR_LEN];
  char *t;
  char **ptr = NULL;
  int insert		= 0;
  int size			= 0;
  int tokens		= 0;
  int found			= 0;
  int oops;
  int len = 0;
  int magic_string_int;
  FILE* table		= NULL;
  user_defined_event_t *foo;


  if ( file_name == NULL ) {
#ifndef STATIC_USER_EVENTS
	PAPIERROR( "Cowardly refusing to load events file NULL\n" );
	return ( PAPI_EBUG );
#else
	/* Only parse the static events once! */
	if ( !first_time ) {
	  PAPIERROR("Cowardly refusing to load events file NULL\n" );
	  return ( PAPI_EBUG );
	}

	INTDBG("Loading events from papi_static_user_events.h\n");
	ptr = &user_events_table;
	table = NULL;

#endif
  } else {
	table = fopen(file_name, "r");

	if ( !table ) {
	  PAPIERROR( "The user events file '%s' does not exist; bailing!", file_name );
	  return ( PAPI_EBUG );
	}

  }

  line = (char*) papi_malloc( USER_EVENT_OPERATION_LEN + 1 );
  /* Main parse loop */
  while (get_event_line(&line, table, ptr) > 0 ) {
	magic_string_int	= 0;
	len = 0;

	t = trim_string( strtok(line, ","));
	if ( (t==NULL) || (strlen(t) == 0) )
	  continue;

	foo = ( user_defined_event_t *) papi_malloc(sizeof(user_defined_event_t));
	memset( foo, 0x0, sizeof(user_defined_event_t) );

	/* Deal with comments and constants */
	if (t[0] == '#') {
	  if ( 0 == strncmp("define",t+1,6) ) {
		papi_free(foo);
		if ( PAPI_OK != (oops = add_define( t , &defines ) ) ) {
		  papi_free(line);
		  if (table)
		    fclose(table);
		  return oops;
		}
		continue;
	  }
	  goto nextline;
	} 

	strncpy(foo->symbol, t, PAPI_MIN_STR_LEN);
#ifdef SHOW_LOADS
	INTDBG("Found a user event named %s\n", foo->symbol );
#endif
	/* This segment handles the postfix operation string 
	 * converting it from OPERAND1|OPERAND2|+ 
	 * to the papi internal N1|N2|+ : OPERAND1:OPERAND2 
	 * with some basic sanity checking of the operands */

	do {
	  memset(temp, 0, sizeof(temp));
	  found = 0;
	  t = trim_string(strtok(NULL, "|"));
	  if ( (t==NULL) || (strlen(t) == 0) )
		break;

	  if ( is_operation(t) ) {
#ifdef SHOW_LOADS
		INTDBG("\tFound operation %c\n", t[0]);
#endif
		sprintf(temp, "%c|", t[0]);
		len = strlen(foo->operation) + strlen(temp);
		if ( len >= USER_EVENT_OPERATION_LEN ) {
		  PAPIERROR("User Event %s's expression is too long.", foo->symbol );
		  goto nextline;
		}
		strcat(foo->operation, temp);
		tokens--;
	  } else if ( isdigit(t[0]) ) {
		/* its a number, the read time parser handlers those */
		sprintf(temp,"%s|", t);
		len = strlen( foo->operation ) + strlen( temp );
		if ( len >= USER_EVENT_OPERATION_LEN ) {
		  PAPIERROR("Invalid event specification %s's expression is too long.", foo->symbol);
		  goto nextline;
		}
		strcat(foo->operation, temp);
		tokens++;
#ifdef SHOW_LOADS
		INTDBG("\tFound number %s\n", t);
#endif
	  } else if (is_define(t, defines.next, foo->operation)) {
		tokens++;
#ifdef SHOW_LOADS
		INTDBG("\tFound a predefined thing %s\n", t);
#endif
	  } else {
		/* check if its a native event */
		if ( check_native_events(t, foo, &magic_string_int) ) {
		  found = 1;
		}

		/* so its not a native event, is it a preset? */
		if ( !found ) { 
		  found = check_preset_events(t, foo, &magic_string_int);
		} /* end preset check */

		/* its not native nor preset is it a UE that we've already seen? */
		/* look through _papi_user_events */ 
		if ( !found ) {
		  found = check_user_events(t, foo, &magic_string_int, 
									_papi_user_events, _papi_user_events_count);
		}

		/* and the current array of parsed events from this file */
		if ( !found ) {
		  if (insert > 0)
			found = check_user_events(t, foo, &magic_string_int, foo, insert);
		} 

		if ( !found ) {
		  INTDBG("HELP!!! UNABLE TO FIND SYMBOL %s\n", t);
		  PAPIERROR("NEXTLIN:\tSymbol lookup failure, I have no clue what %s is, perhaps you have a bad ops string?\n", t);
		  goto nextline;
		} else {
		  tokens++;
		}
	  } /* END native, preset, user defined event lookups */ 

	} while( (int)foo->count < _papi_hwd[0]->cmp_info.num_mpx_cntrs );

	if ( _papi_hwd[0]->cmp_info.num_mpx_cntrs  - (int)foo->count < tokens ) {
	  INTDBG("Event %s is attempting to use too many terms in its expression.\n", foo->symbol );
	  goto nextline;
	}

	/* refine what we mean here, if we exaust the number of counters, do we still allow constants */
	while ( tokens > 1 ) {
	  t = trim_string(strtok(NULL, "|"));
	  if ( t == NULL ) {
		INTDBG("INVALID event specification (%s)\n", foo->symbol);
		goto nextline;
	  }
	  if ( is_operation(t) ) {
		sprintf(temp,"%c|", t[0]);
		/* TODO */
		len = strlen(temp) + strlen(foo->operation);
		if ( len >= USER_EVENT_OPERATION_LEN ) {
		  PAPIERROR("User Event %s contains too many operations.", foo->symbol );
		  goto nextline;
		}
		strcat(foo->operation, temp);
		tokens--;
	  } else {
		PAPIERROR("INVALID event specification, %s is attempting to use too many events\n", foo->symbol);
		goto nextline;
	  }
	}

	append_to_global_list( foo );
#ifdef SHOW_LOADS
	PRINT_UE(foo);
#endif

	insert++;
	size++;
nextline:
	tokens				= 0;
	papi_free(foo);
  } /* End main parse loop */
  if (table)
	fclose(table);

  papi_free(line);
  return insert;
}

int 
_papi_user_defined_events_setup(char *name)
{
  int retval;

  if ( first_time ) {
	_papi_user_events_count = 0;
	defines.next = NULL;
  }

  retval = load_user_event_table( name );

  if (retval < 0)
	return( retval );

  first_time = 0;
  return( PAPI_OK );
}

