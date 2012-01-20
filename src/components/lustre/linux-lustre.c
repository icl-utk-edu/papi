/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file    linux-lustre.c
* CVS:     $Id$
* @author  Haihang You (in collaboration with Michael Kluge, TU Dresden)
*          you@eecs.utk.edu
* @author  Heike Jagode
*          jagode@eecs.utk.edu
* @brief A component for the luster filesystem.
*/

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include "linux-lustre.h"

/* Default path to lustre stats */
//const char proc_base_path[] = "/proc/fs/lustre/llite";
const char proc_base_path[] = "./components/lustre/fake_proc";

#define BUFFER_SIZE 4096
static char buffer[BUFFER_SIZE];

static FILE *proc_fd_snmp = NULL;
static FILE *proc_fd_dev = NULL;

static counter_info *subscriptions[LUSTRE_MAX_COUNTERS];
static int num_counters = 0;
static int is_finalized = 0;

/* counters are kept in a list */
static counter_info *root_counter = NULL;

/* mount Lustre fs are kept in a list */
static lustre_fs *root_lustre_fs = NULL;

/* network interfaces are kept in a list as well */
static network_if *root_network_if = NULL;

#define lustre_native_table subscriptions

static long long _papi_hwd_lustre_register_start[LUSTRE_MAX_COUNTERS];
static long long _papi_hwd_lustre_register[LUSTRE_MAX_COUNTERS];

static int num_events=0;

/*******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 ******************************************************************************/


/**
 * add a counter to the list of available counters
 * @param name the short name of the counter
 * @param desc a longer description
 * @param unit the unit for this counter
 */
static counter_info *
addCounter( const char *name, const char *desc, const char *unit )
{
	counter_info *cntr, *last;

	cntr = ( counter_info * ) malloc( sizeof ( counter_info ) );

	if ( cntr == NULL ) {
		fprintf( stderr, "can not allocate memory for new counter\n" );
		exit( 1 );
	}

	cntr->name = strdup( name );
	cntr->description = strdup( desc );
	cntr->unit = strdup( unit );
	cntr->value = 0;
	cntr->next = NULL;

	if ( root_counter == NULL ) {
		root_counter = cntr;
	} else {
		last = root_counter;
		while ( last->next != NULL )
			last = last->next;
		last->next = cntr;
	}

	return cntr;
}

/**
 * adds a Lustre fs to the fs list and creates the counters for it
 * @param name fs name
 * @param procpath_general path to the 'stats' file in /proc/fs/lustre/... for this fs
 * @param procpath_readahead path to the 'readahead' file in /proc/fs/lustre/... for this fs
 */
static int
addLustreFS( const char *name,
			 const char *procpath_general, const char *procpath_readahead )
{
	lustre_fs *fs, *last;
	char counter_name[512];

	fs = ( lustre_fs * ) malloc( sizeof ( lustre_fs ) );
	if ( fs == NULL ) {
	   SUBDBG("can not allocate memory for new Lustre FS description\n" );
	   return PAPI_ENOMEM;
	}

	fs->proc_file = fopen( procpath_general, "r" );
	if ( fs->proc_file == NULL ) {
	   SUBDBG("can not open '%s'\n", procpath_general );
	   return PAPI_ESBSTR;
	}

	fs->proc_file_readahead = fopen( procpath_readahead, "r" );
	if ( fs->proc_file_readahead == NULL ) {
	   SUBDBG("can not open '%s'\n", procpath_readahead );
	   return PAPI_ESBSTR;
	}

	sprintf( counter_name, "%s_llread", name );
	fs->read_cntr =
		addCounter( counter_name, "bytes read on this lustre client", "bytes" );

	sprintf( counter_name, "%s_llwrite", name );
	fs->write_cntr =
		addCounter( counter_name, "bytes written on this lustre client",
					"bytes" );

	sprintf( counter_name, "%s_wrong_readahead", name );
	fs->readahead_cntr =
		addCounter( counter_name, "bytes read but discarded due to readahead",
					"bytes" );

	fs->next = NULL;
	num_counters += 3;

	if ( root_lustre_fs == NULL ) {
		root_lustre_fs = fs;
	} else {
		last = root_lustre_fs;

		while ( last->next != NULL )
			last = last->next;

		last->next = fs;
	}
	return PAPI_OK;
}






/**
 * goes through proc and tries to discover all mounted Lustre fs
 */
static int
init_lustre_counter(  )
{
	const char *proc_base_path = "/proc/fs/lustre/llite";
	char path[PATH_MAX];
	char path_readahead[PATH_MAX];
	char *ptr;
	char fs_name[100];
	int idx = 0;
	int tmp_fd;
	DIR *proc_fd;
	struct dirent *entry;

	proc_fd = opendir( proc_base_path );
	if ( proc_fd == NULL ) {
	   SUBDBG("we are not able to read this directory\n");
	   return PAPI_ESBSTR;
	}

	entry = readdir( proc_fd );
	while ( entry != NULL ) {
		memset( path, 0, PATH_MAX );
		snprintf( path, PATH_MAX - 1, "%s/%s/stats", proc_base_path,
				  entry->d_name );
		//fprintf( stderr, "checking for file %s\n", path);
		if ( ( tmp_fd = open( path, O_RDONLY ) ) != -1 ) {
			close( tmp_fd );
			// erase \r and \n at the end of path
			idx = strlen( path );
			idx--;

			while ( path[idx] == '\r' || path[idx] == '\n' )
				path[idx--] = 0;

			//  /proc/fs/lustre/llite/ has a length of 22 byte
			memset( fs_name, 0, 100 );
			idx = 0;
			ptr = &path[22];

			while ( *ptr != '-' && idx < 100 ) {
				fs_name[idx] = *ptr;
				ptr++;
				idx++;
			}

			SUBDBG("found Lustre FS: %s\n", fs_name);
			strncpy( path_readahead, path, PATH_MAX );
			ptr = strrchr( path_readahead, '/' );

			if ( ptr == NULL ) {
				SUBDBG( "no slash in %s ?\n", path_readahead );
				return PAPI_ESBSTR;
			}

			ptr++;
			strcpy( ptr, "read_ahead_stats" );
			addLustreFS( fs_name, path, path_readahead );

			memset( path, 0, PATH_MAX );
		}
		entry = readdir( proc_fd );
	}
	closedir( proc_fd );

	return PAPI_OK;
}

/**
 * updates all Lustre related counters
 */
static void
read_lustre_counter(  )
{
	char *ptr;
	lustre_fs *fs = root_lustre_fs;
	int result;

	while ( fs != NULL ) {
	        result=fread( buffer, 1, BUFFER_SIZE, fs->proc_file );

		ptr = strstr( buffer, "write_bytes" );
		if ( ptr == NULL ) {
			fs->write_cntr->value = 0;
		} else {
			/* goto eol */
			while ( *ptr != '\n' )
				ptr++;

			*ptr = 0;
			while ( *ptr != ' ' )
				ptr--;

			ptr++;
			fs->write_cntr->value = strtoll( ptr, NULL, 10 );
		}

		ptr = strstr( buffer, "read_bytes" );
		if ( ptr == NULL ) {
			fs->read_cntr->value = 0;
		} else {
			/* goto eol */
			while ( *ptr != '\n' )
				ptr++;

			*ptr = 0;
			while ( *ptr != ' ' )
				ptr--;

			ptr++;
			fs->read_cntr->value = strtoll( ptr, NULL, 10 );
		}

		result=fread( buffer, 1, BUFFER_SIZE, fs->proc_file_readahead );
		ptr = strstr( buffer, "read but discarded" );
		if ( ptr == NULL ) {
			fs->write_cntr->value = 0;
		} else {
			/* goto next number */
			while ( *ptr < '0' || *ptr > '9' )
				ptr++;

			fs->readahead_cntr->value = strtoll( ptr, NULL, 10 );
		}
		fs = fs->next;
	}
}


/**
 * read all values for all counters
 */
static void
host_read_values( long long *data )
{
	int loop;

	read_lustre_counter(  );

	for ( loop = 0; loop < LUSTRE_MAX_COUNTERS; loop++ ) {
		if ( subscriptions[loop] == NULL )
			break;

		data[loop] = subscriptions[loop]->value;
	}
}


/**
 * find the pointer for a counter_info structure based on the counter name
 */
static counter_info *
counterFromName( const char *cntr )
{
	int loop = 0;
	char tmp[512];
	counter_info *local_cntr = root_counter;
	while ( local_cntr != NULL ) {
		if ( strcmp( cntr, local_cntr->name ) == 0 )
			return local_cntr;

		local_cntr = local_cntr->next;
		loop++;
	}

	gethostname( tmp, 512 );
	fprintf( stderr, "can not find host counter: %s on %s\n", cntr, tmp );
	fprintf( stderr, "we only have: " );
	local_cntr = root_counter;

	while ( local_cntr != NULL ) {
		fprintf( stderr, "'%s' ", local_cntr->name );
		local_cntr = local_cntr->next;
		loop++;
	}

	fprintf( stderr, "\n" );
	exit( 1 );
	/* never reached */
	return 0;
}


/**
 * allow external code to subscribe to a counter based on the counter name
 */
static uint64_t
host_subscribe( const char *cntr )
{
	int loop;
	counter_info *counter = counterFromName( cntr );

	for ( loop = 0; loop < LUSTRE_MAX_COUNTERS; loop++ ) {
		if ( subscriptions[loop] == NULL ) {
			subscriptions[loop] = counter;
			counter->idx = loop;

			return loop + 1;
		}
	}

	fprintf( stderr, "please subscribe only once to each counter\n" );
	exit( 1 );
	/* never reached */
	return 0;
}


/**
 * return a newly allocated list of strings containing all counter names
 */
static string_list *
host_listCounter( int num_counters1 )
{
	string_list *list;
	counter_info *cntr = root_counter;

	list = malloc( sizeof ( string_list ) );
	if ( list == NULL ) {
		SUBDBG("unable to allocate memory for new string_list" );
		return NULL;
	}
	list->count = 0;
	list->data = ( char ** ) malloc( num_counters1 * sizeof ( char * ) );

	if ( list->data == NULL ) {
	   SUBDBG("unable to allocate memory for %d pointers in a new string_list\n",
				 num_counters1 );
	   return NULL;
	}

	while ( cntr != NULL ) {
		list->data[list->count++] = strdup( cntr->name );
		cntr = cntr->next;
	}

	return list;
}


/**
 * finalizes the library
 */
static void
host_finalize(  )
{
	lustre_fs *fs, *next_fs;
	counter_info *cntr, *next;
	network_if *nwif, *next_nwif;

	if ( is_finalized )
		return;

	if ( proc_fd_snmp != NULL )
		fclose( proc_fd_snmp );

	if ( proc_fd_dev != NULL )
		fclose( proc_fd_dev );

	proc_fd_snmp = NULL;
	proc_fd_dev = NULL;

	cntr = root_counter;

	while ( cntr != NULL ) {
		next = cntr->next;
		free( cntr->name );
		free( cntr->description );
		free( cntr->unit );
		free( cntr );
		cntr = next;
	}

	root_counter = NULL;
	fs = root_lustre_fs;

	while ( fs != NULL ) {
		next_fs = fs->next;
		free( fs );
		fs = next_fs;
	}

	root_lustre_fs = NULL;
	nwif = root_network_if;

	while ( nwif != NULL ) {
		next_nwif = nwif->next;
		free( nwif->name );
		free( nwif );
		nwif = next_nwif;
	}

	root_network_if = NULL;
	is_finalized = 1;
}


/**
 * delete a list of strings
 */
static void
host_deleteStringList( string_list * to_delete )
{
	int loop;

	if ( to_delete->data != NULL ) {
		for ( loop = 0; loop < to_delete->count; loop++ )
			free( to_delete->data[loop] );

		free( to_delete->data );
	}
	free( to_delete );
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * Substrate setup and shutdown
 */


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_lustre_init_substrate(  )
{
	int retval = PAPI_OK, i;
	int ret;

	for ( i = 0; i < LUSTRE_MAX_COUNTERS; i++ ) {
		_papi_hwd_lustre_register_start[i] = -1;
		_papi_hwd_lustre_register[i] = -1;
	}

	ret=init_lustre_counter(  );
	if (ret!=PAPI_OK) return ret;

	return retval;
}





/*
 * This is called whenever a thread is initialized
 */
int
_lustre_init( hwd_context_t * ctx )
{

	string_list *counter_list = NULL;
	int i;

	counter_list = host_listCounter( num_counters );
	if (counter_list==NULL) return PAPI_ENOMEM;

	for ( i = 0; i < counter_list->count; i++ )
		host_subscribe( counter_list->data[i] );

	( ( LUSTRE_context_t * ) ctx )->state.ncounter = counter_list->count;

	host_deleteStringList( counter_list );

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;

	host_finalize(  );

	return PAPI_OK;
}



/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup) functions
 */
int
_lustre_init_control_state( hwd_control_state_t * ptr )
{
	( void ) ptr;

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native,
							 int count, hwd_context_t * ctx )
{
	( void ) ptr;
	( void ) ctx;
	int i, index;

	for ( i = 0; i < count; i++ ) {
		index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = index;
	}

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;

	host_read_values( _papi_hwd_lustre_register_start );
	memcpy( _papi_hwd_lustre_register, _papi_hwd_lustre_register_start,
			LUSTRE_MAX_COUNTERS * sizeof ( long long ) );

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
			 long long **events, int flags )
{
	( void ) flags;
	int i;

	host_read_values( _papi_hwd_lustre_register );

	for ( i = 0; i < ( ( LUSTRE_context_t * ) ctx )->state.ncounter; i++ ) {
		( ( LUSTRE_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_lustre_register[i] - _papi_hwd_lustre_register_start[i];
	}

	*events = ( ( LUSTRE_control_state_t * ) ctrl )->counts;
	return ( PAPI_OK );
}


/*
 *
 */
int
_lustre_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{

	int i;

	host_read_values( _papi_hwd_lustre_register );

	for ( i = 0; i < ( ( LUSTRE_context_t * ) ctx )->state.ncounter; i++ ) {
		( ( LUSTRE_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_lustre_register[i] - _papi_hwd_lustre_register_start[i];
	}

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	_lustre_start( ctx, ctrl );

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from )
{
	( void ) ctx;
	( void ) ctrl;
	( void ) from;

	return PAPI_OK;
}


/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
_lustre_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;

	return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int
_lustre_set_domain( hwd_control_state_t * cntrl, int domain )
{
	( void ) cntrl;
	int found = 0;
	if ( PAPI_DOM_USER & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		found = 1;
	}
	if ( !found )
		return PAPI_EINVAL;

	return PAPI_OK;
}


/*
 *
 */
int
_lustre_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

  int event=EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  if (event >=0 && event < num_events) {
     strncpy( name, lustre_native_table[event]->name, len );
     return PAPI_OK;
  }
  return PAPI_ENOEVNT;
}


/*
 *
 */
int
_lustre_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{

  int event=EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  if (event >=0 && event < num_events) {
	strncpy( name, lustre_native_table[event]->description, len );
	return PAPI_OK;
  }
  return PAPI_ENOEVNT;
}


/*
 *
 */
int
_lustre_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	if ( modifier == PAPI_ENUM_FIRST ) {
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
		return PAPI_OK;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( lustre_native_table[index + 1] ) {
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		} else {
			return PAPI_ENOEVNT;
		}
	} 
		

	return PAPI_EINVAL;
}


/*
 *
 */
papi_vector_t _lustre_vector = {
   .cmp_info = {
        /* component information (unspecified values initialized to 0) */
       .name = "linux-lustre.c",
       .version = "1.9",
       .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
       .num_cntrs = LUSTRE_MAX_COUNTERS,
       .default_domain = PAPI_DOM_USER,
       .default_granularity = PAPI_GRN_THR,
       .available_granularities = PAPI_GRN_THR,
       .hardware_intr_sig = PAPI_INT_SIGNAL,

       /* component specific cmp_info initializations */
       .fast_real_timer = 0,
       .fast_virtual_timer = 0,
       .attach = 0,
       .attach_must_ptrace = 0,
       .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
  },

     /* sizes of framework-opaque component-private structures */
  .size = {
       .context = sizeof ( LUSTRE_context_t ),
       .control_state = sizeof ( LUSTRE_control_state_t ),
       .reg_value = sizeof ( LUSTRE_register_t ),
       .reg_alloc = sizeof ( LUSTRE_reg_alloc_t ),
  },

     /* function pointers in this component */
  .init = _lustre_init,
  .init_substrate = _lustre_init_substrate,
  .init_control_state = _lustre_init_control_state,
  .start = _lustre_start,
  .stop = _lustre_stop,
  .read = _lustre_read,
  .shutdown = _lustre_shutdown,
  .ctl = _lustre_ctl,
  .update_control_state = _lustre_update_control_state,
  .set_domain = _lustre_set_domain,
  .reset = _lustre_reset,

  .ntv_enum_events = _lustre_ntv_enum_events,
  .ntv_code_to_name = _lustre_ntv_code_to_name,
  .ntv_code_to_descr = _lustre_ntv_code_to_descr,

};
