/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file    linux-lustre.c
* @author  Haihang You (in collaboration with Michael Kluge, TU Dresden)
*          you@eecs.utk.edu
* @author  Heike Jagode
*          jagode@eecs.utk.edu
* @author  Vince Weaver
*          vweaver1@eecs.utk.edu
* @brief A component for the luster filesystem.
*/

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdint.h>
#include <ctype.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/** describes a single counter with its properties */
typedef struct counter_info_struct
{
	int idx;
	char *name;
	char *description;
	char *unit;
	unsigned long long value;
} counter_info;

typedef struct
{
	int count;
	char **data;
} string_list;


/** describes the infos collected from a mounted Lustre filesystem */
typedef struct lustre_fs_struct
{
	char *proc_file;
	char *proc_file_readahead;
	counter_info *write_cntr;
	counter_info *read_cntr;
	counter_info *readahead_cntr;
	struct lustre_fs_struct *next;
} lustre_fs;

#define LUSTRE_MAX_COUNTERS 100
#define LUSTRE_MAX_COUNTER_TERMS  LUSTRE_MAX_COUNTERS

typedef counter_info LUSTRE_register_t;
typedef counter_info LUSTRE_native_event_entry_t;
typedef counter_info LUSTRE_reg_alloc_t;


typedef struct LUSTRE_control_state
{
	long long start_count[LUSTRE_MAX_COUNTERS];
        long long current_count[LUSTRE_MAX_COUNTERS];
        long long difference[LUSTRE_MAX_COUNTERS];
        int which_counter[LUSTRE_MAX_COUNTERS];
	int num_events;
} LUSTRE_control_state_t;


typedef struct LUSTRE_context
{
	LUSTRE_control_state_t state;
} LUSTRE_context_t;

/* Default path to lustre stats */
#ifdef FAKE_LUSTRE
const char proc_base_path[] = "./components/lustre/fake_proc/fs/lustre/";
#else
const char proc_base_path[] = "/proc/fs/lustre/";
#endif

static counter_info **lustre_native_table = NULL;
static int num_events = 0;
static int table_size = 32;

/* mount Lustre fs are kept in a list */
static lustre_fs *root_lustre_fs = NULL;

papi_vector_t _lustre_vector;

/******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/
static int resize_native_table() {
	counter_info** new_table;
	int new_size = table_size*2;
	new_table = (counter_info**)papi_malloc(sizeof(counter_info*) * new_size);
	if (NULL==new_table)
		return PAPI_ENOMEM;
	if ( lustre_native_table) {
		memcpy(new_table, lustre_native_table, sizeof(counter_info*) * table_size );
		papi_free(lustre_native_table);
	}
	lustre_native_table = new_table;
	table_size*=2;
	return PAPI_OK;
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
    counter_info *cntr;

	if ( num_events >= table_size )
		if (PAPI_OK != resize_native_table())
			return NULL;

    cntr = malloc( sizeof ( counter_info ) );

    if ( cntr == NULL ) {
       SUBDBG("can not allocate memory for new counter\n" );
       return NULL;
    }

    cntr->idx=num_events;
    cntr->name = strdup( name );
    cntr->description = strdup( desc );
    cntr->unit = strdup( unit );
    cntr->value = 0;

    lustre_native_table[num_events]=cntr;

    num_events++;

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
	     const char *procpath_general, 
	     const char *procpath_readahead )
{
	lustre_fs *fs, *last;
	char counter_name[512];
	FILE *fff;

	SUBDBG("Adding lustre fs\n");

	fs = malloc( sizeof ( lustre_fs ) );
	if ( fs == NULL ) {
	   SUBDBG("can not allocate memory for new Lustre FS description\n" );
	   return PAPI_ENOMEM;
	}

	fs->proc_file=strdup(procpath_general);
	fff = fopen( procpath_general, "r" );
	if ( fff == NULL ) {
	  SUBDBG("can not open '%s'\n", procpath_general );
	  free(fs);
	  return PAPI_ESYS;
	}
	fclose(fff);

	fs->proc_file_readahead = strdup(procpath_readahead);
	fff = fopen( procpath_readahead, "r" );
	if ( fff == NULL ) {
	  SUBDBG("can not open '%s'\n", procpath_readahead );
	  free(fs);
	  return PAPI_ESYS;
	}
	fclose(fff);

	sprintf( counter_name, "%s_llread", name );
	if (NULL == (fs->read_cntr = addCounter( counter_name, 
				    "bytes read on this lustre client", 
				    "bytes" ))) {
			free(fs);
			return PAPI_ENOMEM;
	}

	sprintf( counter_name, "%s_llwrite", name );
	if ( NULL == (fs->write_cntr = addCounter( counter_name, 
				     "bytes written on this lustre client",
				     "bytes" ))) {
			free(fs->read_cntr);
			free(fs);
			return PAPI_ENOMEM;
	}

	sprintf( counter_name, "%s_wrong_readahead", name );
	if ( NULL == (fs->readahead_cntr = addCounter( counter_name, 
					 "bytes read but discarded due to readahead",
					 "bytes" ))) {
			free(fs->read_cntr);
			free(fs->write_cntr);
			free(fs);
			return PAPI_ENOMEM;
	}

	fs->next = NULL;

	/* Insert into the linked list */
	/* Does this need locking? */
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
init_lustre_counters( void  )
{
        char lustre_dir[PATH_MAX];
	char path[PATH_MAX];
	char path_readahead[PATH_MAX],path_stats[PATH_MAX];
	char *ptr;
	char fs_name[100];
	int idx = 0;
	int tmp_fd;
	DIR *proc_dir;
	struct dirent *entry;

	sprintf(lustre_dir,"%s/llite",proc_base_path);

	proc_dir = opendir( lustre_dir );
	if ( proc_dir == NULL ) {
	   SUBDBG("Cannot open %s\n",lustre_dir);
	   return PAPI_ESYS;
	}

	entry = readdir( proc_dir );

	while ( entry != NULL ) {
	   memset( path, 0, PATH_MAX );
	   snprintf( path, PATH_MAX - 1, "%s/%s/stats", lustre_dir,
				  entry->d_name );
	   SUBDBG("checking for file %s\n", path);

	   if ( ( tmp_fd = open( path, O_RDONLY ) ) != -1 ) {
	      close( tmp_fd );

	      /* erase \r and \n at the end of path */
	      /* why is this necessary?             */

	      idx = strlen( path );
	      idx--;

	      while ( path[idx] == '\r' || path[idx] == '\n' )
		    path[idx--] = 0;

	      /* Lustre paths are of type server-UUID */

	      idx = 0;

	      ptr = strstr(path,"llite/") + 6;

	      while ( *ptr && idx < 100 ) {
	         fs_name[idx] = *ptr;
		 ptr++;
		 idx++;
	      }

	      SUBDBG("found Lustre FS: %s\n", fs_name);

	      snprintf( path_stats, PATH_MAX - 1, 
			"%s/%s/stats", 
			lustre_dir,
			entry->d_name );
	      SUBDBG("Found file %s\n", path_stats);

	      snprintf( path_readahead, PATH_MAX - 1, 
			"%s/%s/read_ahead_stats", 
			lustre_dir,
			entry->d_name );
	      SUBDBG("Now checking for file %s\n", path_readahead);


	      strcpy( ptr, "read_ahead_stats" );
	      addLustreFS( fs_name, path_stats, path_readahead );

	   }
	   entry = readdir( proc_dir );
	}
	closedir( proc_dir );

	return PAPI_OK;

}

/**
 * updates all Lustre related counters
 */
static void
read_lustre_counter( )
{
	lustre_fs *fs = root_lustre_fs;
	FILE *fff;
	char buffer[BUFSIZ];

	while ( fs != NULL ) {

	  /* read values from stats file */
	  fff=fopen(fs->proc_file,"r" );
	  if (fff != NULL) {
		  while(1) {
			if (fgets(buffer,BUFSIZ,fff)==NULL) break;
	
			if (strstr( buffer, "write_bytes" )) {
			  sscanf(buffer,"%*s %*d %*s %*s %*d %*d %lld",&fs->write_cntr->value);
			  SUBDBG("Read %lld write_bytes\n",fs->write_cntr->value);
			}
	
			if (strstr( buffer, "read_bytes" )) {
			  sscanf(buffer,"%*s %*d %*s %*s %*d %*d %lld",&fs->read_cntr->value);
			  SUBDBG("Read %lld read_bytes\n",fs->read_cntr->value);
			}
		  }
		  fclose(fff);
	  }

	  fff=fopen(fs->proc_file_readahead,"r");
	  if (fff != NULL) {
		  while(1) {
			if (fgets(buffer,BUFSIZ,fff)==NULL) break;
	
			if (strstr( buffer, "read but discarded")) {
			   sscanf(buffer,"%*s %*s %*s %lld",&fs->readahead_cntr->value);
			   SUBDBG("Read %lld discared\n",fs->readahead_cntr->value);
			   break;
			}
	  	  }
		  fclose(fff);
	  }
	  fs = fs->next;
	}
}


/**
 * frees all allocated resources
 */
static void
host_finalize( void )
{
        int i;
	lustre_fs *fs, *next_fs;
	counter_info *cntr;

	for(i=0;i<num_events;i++) {
	   cntr=lustre_native_table[i];
	   if ( cntr != NULL ) {
	      free( cntr->name );
	      free( cntr->description );
	      free( cntr->unit );
	      free( cntr );	      
	   }
	   lustre_native_table[i]=NULL;
	}

	fs = root_lustre_fs;

	while ( fs != NULL ) {
		next_fs = fs->next;
		free(fs->proc_file);
		free(fs->proc_file_readahead);
		free( fs );
		fs = next_fs;
	}

	root_lustre_fs = NULL;
}


/**
 * see if lustre filesystem is supported by kernel
 */
static int
detect_lustre()
{
        char lustre_directory[BUFSIZ];
	DIR *proc_dir;

	sprintf(lustre_directory,"%s/llite",proc_base_path);

	proc_dir = opendir( proc_base_path );
	if ( proc_dir == NULL ) {
	  SUBDBG("we are not able to read %s\n",lustre_directory);
	   return PAPI_ESYS;
	}

	closedir(proc_dir);

	return PAPI_OK;
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * Component setup and shutdown
 */

int
_lustre_init_component( int cidx )
{

	int ret = PAPI_OK;

	/* See if lustre filesystem exists */
	ret=detect_lustre();
	if (ret!=PAPI_OK) {
	   strncpy(_lustre_vector.cmp_info.disabled_reason,
		   "No lustre filesystems found",PAPI_MAX_STR_LEN);
	   return ret;
	}

	resize_native_table();
	ret=init_lustre_counters();

	_lustre_vector.cmp_info.num_native_events=num_events;
	_lustre_vector.cmp_info.CmpIdx = cidx;

	return ret;
}





/*
 * This is called whenever a thread is initialized
 */
int
_lustre_init_thread( hwd_context_t * ctx )
{
  (void) ctx;

  return PAPI_OK;
}


/*
 *
 */
int
_lustre_shutdown_component( void )
{

	host_finalize(  );
	papi_free( lustre_native_table );

	return PAPI_OK;
}

/*
 *
 */
int
_lustre_shutdown_thread( hwd_context_t * ctx )
{
	( void ) ctx;

	return PAPI_OK;
}



/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup) functions
 */
int
_lustre_init_control_state( hwd_control_state_t *ctl )
{
    LUSTRE_control_state_t *lustre_ctl = (LUSTRE_control_state_t *)ctl;

    memset(lustre_ctl->start_count,0,sizeof(long long)*LUSTRE_MAX_COUNTERS);
    memset(lustre_ctl->current_count,0,sizeof(long long)*LUSTRE_MAX_COUNTERS);

    return PAPI_OK;
}


/*
 *
 */
int
_lustre_update_control_state( hwd_control_state_t *ctl, 
			      NativeInfo_t *native,
			      int count, 
			      hwd_context_t *ctx )
{
    LUSTRE_control_state_t *lustre_ctl = (LUSTRE_control_state_t *)ctl;
    ( void ) ctx;
    int i, index;

    for ( i = 0; i < count; i++ ) {
       index = native[i].ni_event;
       lustre_ctl->which_counter[i]=index;
       native[i].ni_position = i;
    }

    lustre_ctl->num_events=count;

    return PAPI_OK;
}


/*
 *
 */
int
_lustre_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;

    LUSTRE_control_state_t *lustre_ctl = (LUSTRE_control_state_t *)ctl;
    int i;

    read_lustre_counter(  );

    for(i=0;i<lustre_ctl->num_events;i++) {
       lustre_ctl->current_count[i]=
                 lustre_native_table[lustre_ctl->which_counter[i]]->value;
    }

    memcpy( lustre_ctl->start_count,
	    lustre_ctl->current_count,
	    LUSTRE_MAX_COUNTERS * sizeof ( long long ) );

    return PAPI_OK;
}


/*
 *
 */
int
_lustre_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    (void) ctx;
    LUSTRE_control_state_t *lustre_ctl = (LUSTRE_control_state_t *)ctl;
    int i;

    read_lustre_counter(  );

    for(i=0;i<lustre_ctl->num_events;i++) {
       lustre_ctl->current_count[i]=
                 lustre_native_table[lustre_ctl->which_counter[i]]->value;
    }

    return PAPI_OK;

}



/*
 *
 */
int
_lustre_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
			 long long **events, int flags )
{
    (void) ctx;
    ( void ) flags;

    LUSTRE_control_state_t *lustre_ctl = (LUSTRE_control_state_t *)ctl;
    int i;

    read_lustre_counter(  );

    for(i=0;i<lustre_ctl->num_events;i++) {
       lustre_ctl->current_count[i]=
                 lustre_native_table[lustre_ctl->which_counter[i]]->value;
       lustre_ctl->difference[i]=lustre_ctl->current_count[i]-
	                                     lustre_ctl->start_count[i];
    }

    *events = lustre_ctl->difference;

    return PAPI_OK;

}




/*
 *
 */
int
_lustre_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{

  /* re-initializes counter_start values to current */

  _lustre_start(ctx,ctrl);

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

/* This function sets various options in the component
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

  int event=EventCode;

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

  int event=EventCode;

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

	if ( modifier == PAPI_ENUM_FIRST ) {
	   if (num_events==0) return PAPI_ENOEVNT;
	   *EventCode = 0;
	   return PAPI_OK;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode;

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
       .name = "lustre",
	   .short_name = "lustre",
       .version = "1.9",
       .description = "Lustre filesystem statistics",
       .num_mpx_cntrs = LUSTRE_MAX_COUNTERS,
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
  .init_thread =           _lustre_init_thread,
  .init_component =        _lustre_init_component,
  .init_control_state =    _lustre_init_control_state,
  .start =                 _lustre_start,
  .stop =                  _lustre_stop,
  .read =                  _lustre_read,
  .shutdown_thread =       _lustre_shutdown_thread,
  .shutdown_component =    _lustre_shutdown_component,
  .ctl =                   _lustre_ctl,
  .update_control_state =  _lustre_update_control_state,
  .set_domain =            _lustre_set_domain,
  .reset =                 _lustre_reset,

  .ntv_enum_events =   _lustre_ntv_enum_events,
  .ntv_code_to_name =  _lustre_ntv_code_to_name,
  .ntv_code_to_descr = _lustre_ntv_code_to_descr,

};




