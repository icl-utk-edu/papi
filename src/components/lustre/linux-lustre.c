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

int lustre_init_presets(  );

static size_t pagesz = 0;
static char *buffer = NULL;
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

/* some counters that are always present */
static counter_info *tcp_sent = NULL;
static counter_info *tcp_recv = NULL;
static counter_info *tcp_retr = NULL;

#define lustre_native_table subscriptions

long long _papi_hwd_lustre_register_start[LUSTRE_MAX_COUNTERS];
long long _papi_hwd_lustre_register[LUSTRE_MAX_COUNTERS];


/*******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 ******************************************************************************/

/**
 * open a file and return the FILE handle
 */
static FILE *
proc_fopen( const char *procname )
{
	FILE *fd = fopen( procname, "r" );

	if ( fd == NULL ) {
		fprintf( stderr, "unable to open proc file %s\n", procname );
		return NULL;
	}

	if ( !buffer ) {
		pagesz = getpagesize(  );
		buffer = malloc( pagesz );
		memset( buffer, 0, pagesz );
	}

	setvbuf( fd, buffer, _IOFBF, pagesz );
	return fd;
}


/**
 * read one PAGE_SIZE byte from a FILE handle and reset the file pointer
 */
static void
readPage( FILE * fd )
{
	int count_read;

	count_read = fread( buffer, pagesz, 1, fd );

	if ( fseek( fd, 0L, SEEK_SET ) != 0 ) {
		fprintf( stderr, "can not seek back in proc\n" );
		exit( 1 );
	}
}


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
 * adds a network interface to the list of available
 * interfaces and the counters to the list of available
 * counters
 */
static void
addNetworkIf( const char *name )
{
	network_if *nwif, *last;
	char counter_name[512];

	nwif = ( network_if * ) malloc( sizeof ( network_if ) );
	if ( nwif == NULL ) {
		fprintf( stderr,
				 "can not allocate memory for new network interface description\n" );
		exit( 1 );
	}

	nwif->name = strdup( name );

	sprintf( counter_name, "%s_recv", name );
	nwif->recv_cntr =
		addCounter( counter_name, "bytes received on this interface", "bytes" );

	sprintf( counter_name, "%s_send", name );
	nwif->send_cntr =
		addCounter( counter_name, "bytes written on this interface", "bytes" );

	nwif->next = NULL;
	num_counters += 2;

	if ( root_network_if == NULL ) {
		root_network_if = nwif;
	} else {
		last = root_network_if;
		while ( last->next != NULL )
			last = last->next;
		last->next = nwif;
	}
}


/**
 * adds a Lustre fs to the fs list and creates the counters for it
 * @param name fs name
 * @param procpath_general path to the 'stats' file in /proc/fs/lustre/... for this fs
 * @param procpath_readahead path to the 'readahead' file in /proc/fs/lustre/... for this fs
 */
static void
addLustreFS( const char *name,
			 const char *procpath_general, const char *procpath_readahead )
{
	lustre_fs *fs, *last;
	char counter_name[512];

	fs = ( lustre_fs * ) malloc( sizeof ( lustre_fs ) );
	if ( fs == NULL ) {
		fprintf( stderr,
				 "can not allocate memory for new Lustre FS description\n" );
		exit( 1 );
	}

	fs->proc_fd = proc_fopen( procpath_general );
	if ( fs->proc_fd == NULL ) {
		fprintf( stderr, "can not open '%s'\n", procpath_general );
		exit( 1 );
	}

	fs->proc_fd_readahead = proc_fopen( procpath_readahead );
	if ( fs->proc_fd_readahead == NULL ) {
		fprintf( stderr, "can not open '%s'\n", procpath_readahead );
		exit( 1 );
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
}


/**
 * looks after available IP interfaces/cards
 */
static void
init_tcp_counter(  )
{
	char *ptr;
	char name[100];
	int idx;

	/* init the static stuff from /proc/net/snmp */
	proc_fd_snmp = proc_fopen( "/proc/net/snmp" );
	if ( proc_fd_snmp == NULL ) {
		return;
		/* fprintf( stderr, "can not open /proc/net/snmp\n");
		   exit(1);
		 */
	}

	tcp_sent =
		addCounter( "tcp_segments_sent", "# of TCP segments sent", "segments" );
	tcp_recv =
		addCounter( "tcp_segments_received", "# of TCP segments received",
					"segments" );
	tcp_retr =
		addCounter( "tcp_segments_retransmitted",
					"# of TCP segments retransmitted", "segments" );

	num_counters += 3;

	/* now the individual interfaces */
	proc_fd_dev = proc_fopen( "/proc/net/dev" );
	if ( proc_fd_dev == NULL ) {
		return;
		/* fprintf( stderr, "can not open /proc/net/dev\n");
		   exit(1);
		 */
	}

	readPage( proc_fd_dev );
	ptr = buffer;
	while ( *ptr != 0 ) {
		while ( *ptr != 0 && *ptr != ':' )
			ptr++;

		if ( *ptr == 0 )
			break;

		/* move backwards until space or '\n' */
		while ( *ptr != ' ' && *ptr != '\n' )
			ptr--;

		ptr++;
		memset( name, 0, sizeof ( name ) );
		idx = 0;

		while ( *ptr != ':' )
			name[idx++] = *ptr++;

		ptr++;
		// printf("new interface: '%s'\n", name);
		addNetworkIf( name );
	}
}


/**
 * goes through proc and tries to discover all mounted Lustre fs
 */
static void
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
		// we are not able to read this directory ...
		return;
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

			/* printf("found Lustre FS: %s\n", fs_name); */
			strncpy( path_readahead, path, PATH_MAX );
			ptr = strrchr( path_readahead, '/' );

			if ( ptr == NULL ) {
				fprintf( stderr, "no slash in %s ?\n", path_readahead );
				fflush( stderr );
				exit( 1 );
			}

			ptr++;
			strcpy( ptr, "read_ahead_stats" );
			addLustreFS( fs_name, path, path_readahead );

			memset( path, 0, PATH_MAX );
		}
		entry = readdir( proc_fd );
	}
	closedir( proc_fd );
}


/**
 * reads all cards and updates the associated counters
 */
static void
read_tcp_counter(  )
{
	int64_t in, out, retr;
	char *ptr;
	int num;
	int idx;
	char name[100];
	network_if *current_if = root_network_if;

	if ( proc_fd_snmp != NULL ) {
		readPage( proc_fd_snmp );

		ptr = strstr( buffer, "Tcp:" );
		ptr += 4;
		ptr = strstr( ptr, "Tcp:" );
		ptr += 4;
		num = 0;

		while ( num < 10 ) {
			if ( *ptr == ' ' )
				num++;

			ptr++;
		}

		in = strtoll( ptr, NULL, 10 );

		while ( *ptr != ' ' )
			ptr++;

		ptr++;
		out = strtoll( ptr, NULL, 10 );

		while ( *ptr != ' ' )
			ptr++;

		ptr++;
		retr = strtoll( ptr, NULL, 10 );

		tcp_sent->value = out;
		tcp_recv->value = in;
		tcp_retr->value = retr;
	}

	if ( proc_fd_dev != NULL ) {
		/* now parse /proc/net/dev */
		readPage( proc_fd_dev );
		ptr = buffer;
		// jump over first two \n
		while ( *ptr != 0 && *ptr != '\n' )
			ptr++;

		if ( *ptr == 0 )
			return;

		ptr++;
		while ( *ptr != 0 && *ptr != '\n' )
			ptr++;

		if ( *ptr == 0 )
			return;

		ptr++;

		while ( *ptr != 0 ) {
			if ( current_if == NULL )
				break;

			// move to next non space char
			while ( *ptr == ' ' )
				ptr++;

			if ( *ptr == 0 )
				return;

			// copy name until ':'
			idx = 0;
			while ( *ptr != ':' )
				name[idx++] = *ptr++;

			if ( *ptr == 0 )
				return;

			name[idx] = 0;

			// compare and make sure network interface are still
			// showing up in the same order. adding or deleting
			// some or changing the order during the run is not
			// support yet (overhead)
			if ( current_if == NULL ) {
				fprintf( stderr, "error: current interface is NULL\n" );
				exit( 1 );
			}

			if ( strcmp( name, current_if->name ) != 0 ) {
				fprintf( stderr,
						 "wrong interface, order changed(?): got %s, wanted %s\n",
						 name, current_if->name );
				exit( 1 );
			}
			// move forward to next number
			while ( *ptr < '0' || *ptr > '9' )
				ptr++;

			if ( *ptr == 0 )
				return;

			in = strtoll( ptr, NULL, 10 );

			// move eight numbers forward
			for ( num = 0; num < 8; num++ ) {
				// move to next space
				while ( *ptr != ' ' )
					ptr++;

				if ( *ptr == 0 )
					return;

				// move forward to next number
				while ( *ptr < '0' || *ptr > '9' )
					ptr++;

				if ( *ptr == 0 )
					return;
			}

			out = strtoll( ptr, NULL, 10 );

			// move to next newline
			while ( *ptr != '\n' )
				ptr++;

			ptr++;

			current_if->recv_cntr->value = in;
			current_if->send_cntr->value = out;
			current_if = current_if->next;
		}
	}
}


/**
 * updates all Lustre related counters
 */
static void
read_lustre_counter(  )
{
	char *ptr;
	lustre_fs *fs = root_lustre_fs;

	while ( fs != NULL ) {
		readPage( fs->proc_fd );

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

		readPage( fs->proc_fd_readahead );
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
	read_tcp_counter(  );
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
			//fprintf( stderr, "subscription %d is %s\n", loop, subscriptions[ loop ]->name);
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
		fprintf( stderr, "unable to allocate memory for new string_list" );
		exit( 1 );
	}
	list->count = 0;
	list->data = ( char ** ) malloc( num_counters1 * sizeof ( char * ) );

	if ( list->data == NULL ) {
		fprintf( stderr,
				 "unable to allocate memory for %d pointers in a new string_list\n",
				 num_counters1 );
		exit( 1 );
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

	if ( buffer != NULL )
		free( buffer );

	buffer = NULL;
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
LUSTRE_init_substrate(  )
{
	int retval = PAPI_OK, i;

	for ( i = 0; i < LUSTRE_MAX_COUNTERS; i++ ) {
		_papi_hwd_lustre_register_start[i] = -1;
		_papi_hwd_lustre_register[i] = -1;
	}

	/* Internal function, doesn't necessarily need to be a function */
	lustre_init_presets(  );

	return ( retval );
}


/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
hwi_search_t lustre_preset_map[] = { {0, {0, {PAPI_NULL, PAPI_NULL}
										  , {0,}
										  }
									  }
};


int
lustre_init_presets(  )
{
	return ( _papi_hwi_setup_all_presets( lustre_preset_map, NULL ) );
}


/*
 * This is called whenever a thread is initialized
 */
int
LUSTRE_init( hwd_context_t * ctx )
{
	string_list *counter_list = NULL;
	int i;

	init_tcp_counter(  );
	init_lustre_counter(  );

	counter_list = host_listCounter( num_counters );

	for ( i = 0; i < counter_list->count; i++ )
		host_subscribe( counter_list->data[i] );

	( ( LUSTRE_context_t * ) ctx )->state.ncounter = counter_list->count;

	host_deleteStringList( counter_list );

	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;
	host_finalize(  );
	return ( PAPI_OK );
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup) functions
 */
int
LUSTRE_init_control_state( hwd_control_state_t * ptr )
{
	( void ) ptr;
	return PAPI_OK;
}


/*
 *
 */
int
LUSTRE_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native,
							 int count, hwd_context_t * ctx )
{
	( void ) ptr;
	( void ) ctx;
	int i, index;

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = index;
	}

	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;

	host_read_values( _papi_hwd_lustre_register_start );
	memcpy( _papi_hwd_lustre_register, _papi_hwd_lustre_register_start,
			LUSTRE_MAX_COUNTERS * sizeof ( long long ) );

	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
			 long long **events, int flags )
{
	( void ) ctx;
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
LUSTRE_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	int i;

	host_read_values( _papi_hwd_lustre_register );

	for ( i = 0; i < ( ( LUSTRE_context_t * ) ctx )->state.ncounter; i++ ) {
		( ( LUSTRE_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_lustre_register[i] - _papi_hwd_lustre_register_start[i];
	}

	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	LUSTRE_start( ctx, ctrl );
	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from )
{
	( void ) ctx;
	( void ) ctrl;
	( void ) from;
	return ( PAPI_OK );
}


/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
LUSTRE_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
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
LUSTRE_set_domain( hwd_control_state_t * cntrl, int domain )
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
		return ( PAPI_EINVAL );
	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK &
								 PAPI_COMPONENT_AND_MASK]->name, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK &
								 PAPI_COMPONENT_AND_MASK]->description, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
LUSTRE_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	memcpy( ( LUSTRE_register_t * ) bits,
			lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK &
								PAPI_COMPONENT_AND_MASK],
			sizeof ( LUSTRE_register_t ) );
	return ( PAPI_OK );
}

int
LUSTRE_ntv_bits_to_info( hwd_register_t * bits, char *names,
						 unsigned int *values, int name_len, int count )
{
	( void ) bits;
	( void ) names;
	( void ) values;
	( void ) name_len;
	( void ) count;
	return ( 1 );
}


/*
 *
 */
int
LUSTRE_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	if ( modifier == PAPI_ENUM_FIRST ) {
		/* assumes first native event is always 0x4000000 */
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
		return ( PAPI_OK );
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( lustre_native_table[index + 1] ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	} else
		return ( PAPI_EINVAL );
}


/*
 *
 */
papi_vector_t _lustre_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name =
				 "$Id$",
				 .version = "$Revision$",
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
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( LUSTRE_context_t ),
			 .control_state = sizeof ( LUSTRE_control_state_t ),
			 .reg_value = sizeof ( LUSTRE_register_t ),
			 .reg_alloc = sizeof ( LUSTRE_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = LUSTRE_init,
	.init_substrate = LUSTRE_init_substrate,
	.init_control_state = LUSTRE_init_control_state,
	.start = LUSTRE_start,
	.stop = LUSTRE_stop,
	.read = LUSTRE_read,
	.shutdown = LUSTRE_shutdown,
	.ctl = LUSTRE_ctl,
	.update_control_state = LUSTRE_update_control_state,
	.set_domain = LUSTRE_set_domain,
	.reset = LUSTRE_reset,
	.ntv_enum_events = LUSTRE_ntv_enum_events,
	.ntv_code_to_name = LUSTRE_ntv_code_to_name,
	.ntv_code_to_descr = LUSTRE_ntv_code_to_descr,
	.ntv_code_to_bits = LUSTRE_ntv_code_to_bits,
	.ntv_bits_to_info = LUSTRE_ntv_bits_to_info
};
