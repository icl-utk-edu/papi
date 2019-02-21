/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-infiniband_umad.c
 * @author  Heike Jagode (in collaboration with Michael Kluge, TU Dresden)
 *          jagode@eecs.utk.edu
 * @author  Tony Castaldo; minor changes; the infiniband_umad.h differs
 *          on the declaration of umad_get_ca(), and demands a const char*
 *          instead of a char*. Corrected the prototypes below.
 *
 *
 * @ingroup papi_components 		
 * 
 * InfiniBand component 
 * 
 * Tested version of OFED: 1.4
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for InfiniBand devices through the  
 *  OFED library. Since a new interface was introduced with OFED version 1.4 
 *  (released Dec 2008), the current InfiniBand component does not support 
 *  OFED versions < 1.4.
 */
#include <dlfcn.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "linux-infiniband_umad.h"

void (*_dl_non_dynamic_init)(void) __attribute__((weak));

/********  CHANGE PROTOTYPES TO DECLARE Infiniband LIBRARY SYMBOLS AS WEAK  **********
 *  This is done so that a version of PAPI built with the infiniband component can   *
 *  be installed on a system which does not have the infiniband libraries installed. *
 *                                                                                   *
 *  If this is done without these prototypes, then all papi services on the system   *
 *  without the infiniband libraries installed will fail.  The PAPI libraries        *
 *  contain references to the infiniband libraries which are not installed.  The     *
 *  load of PAPI commands fails because the infiniband library references can not    *
 *  be resolved.                                                                     *
 *                                                                                   *
 *  This also defines pointers to the infiniband library functions that we call.     *
 *  These function pointers will be resolved with dlopen/dlsym calls at component    *
 *  initialization time.  The component then calls the infiniband library functions  *
 *  through these function pointers.                                                 *
 *************************************************************************************/
int                 __attribute__((weak)) umad_init              ( void );
int                 __attribute__((weak)) umad_get_cas_names     ( char [][UMAD_CA_NAME_LEN], int  );
int                 __attribute__((weak)) umad_get_ca            ( const char *, umad_ca_t * );
void                __attribute__((weak)) mad_decode_field       ( unsigned char *, enum MAD_FIELDS, void *);
struct ibmad_port * __attribute__((weak)) mad_rpc_open_port      ( char *, int, int *, int );
int                 __attribute__((weak)) ib_resolve_self_via    ( ib_portid_t *, int *, ibmad_gid_t *, const struct ibmad_port * );
uint8_t *           __attribute__((weak)) performance_reset_via  ( void *, ib_portid_t *, int, unsigned, unsigned, unsigned, const struct ibmad_port * );
uint8_t *           __attribute__((weak)) pma_query_via          ( void *, ib_portid_t *, int, unsigned, unsigned, const struct ibmad_port * );

int                  (*umad_initPtr)             ( void );
int                  (*umad_get_cas_namesPtr)    ( char [][UMAD_CA_NAME_LEN], int );
int                  (*umad_get_caPtr)           ( const char *, umad_ca_t * );
void                 (*mad_decode_fieldPtr)      ( unsigned char *, enum MAD_FIELDS, void * );
struct ibmad_port *  (*mad_rpc_open_portPtr)     ( char *, int, int *, int );
int                  (*ib_resolve_self_viaPtr)   (ib_portid_t *, int *, ibmad_gid_t *, const struct ibmad_port * );
uint8_t *            (*performance_reset_viaPtr) (void *, ib_portid_t *, int, unsigned, unsigned, unsigned, const struct ibmad_port * );
uint8_t *            (*pma_query_viaPtr)         (void *, ib_portid_t *, int, unsigned, unsigned, const struct ibmad_port * );

// file handles used to access Infiniband libraries with dlopen
static void* dl1 = NULL;
static void* dl2 = NULL;

static int linkInfinibandLibraries ();

papi_vector_t _infiniband_umad_vector;



struct ibmad_port *srcport;
static ib_portid_t portid;
static int ib_timeout = 0;
static int ibportnum = 0;

static counter_info *subscriptions[INFINIBAND_MAX_COUNTERS];
static int is_initialized = 0;
static int num_counters = 0;
static int is_finalized = 0;

/* counters are kept in a list */
static counter_info *root_counter = NULL;
/* IB ports found are kept in a list */
static ib_port *root_ib_port = NULL;
static ib_port *active_ib_port = NULL;

#define infiniband_native_table subscriptions
/* macro to initialize entire structs to 0 */
#define InitStruct(var, type) type var; memset(&var, 0, sizeof(type))

long long _papi_hwd_infiniband_register_start[INFINIBAND_MAX_COUNTERS];
long long _papi_hwd_infiniband_register[INFINIBAND_MAX_COUNTERS];


/*******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 ******************************************************************************/

/**
 * use libumad to discover IB ports
 */
static void
init_ib_counter(  )
{
	char names[20][UMAD_CA_NAME_LEN];
	int n, i;
	char *ca_name;
	umad_ca_t ca;
	int r;
	int portnum;

//	if ( umad_init(  ) < 0 ) {
//		fprintf( stderr, "can't init UMAD library\n" );
//		exit( 1 );
//	}

	if ( ( n = (*umad_get_cas_namesPtr)( ( void * ) names, UMAD_CA_NAME_LEN ) ) < 0 ) {
		fprintf( stderr, "can't list IB device names\n" );
		exit( 1 );
	}

	for ( i = 0; i < n; i++ ) {
		ca_name = names[i];

		if ( ( r = (*umad_get_caPtr)( ca_name, &ca ) ) < 0 ) {
			fprintf( stderr, "can't read ca from IB device\n" );
			exit( 1 );
		}

		if ( !ca.node_type )
			continue;

		/* port numbers are '1' based in OFED */
		for ( portnum = 1; portnum <= ca.numports; portnum++ )
			addIBPort( ca.ca_name, ca.ports[portnum] );
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
 * add one IB port to the list of available ports and add the
 * counters related to this port to the global counter list
 */
static void
addIBPort( const char *ca_name, umad_port_t * port )
{
	ib_port *nwif, *last;
	char counter_name[512];

	nwif = ( ib_port * ) malloc( sizeof ( ib_port ) );

	if ( nwif == NULL ) {
		fprintf( stderr, "can not allocate memory for IB port description\n" );
		exit( 1 );
	}

	sprintf( counter_name, "%s_%d", ca_name, port->portnum );
	nwif->name = strdup( counter_name );

	sprintf( counter_name, "%s_%d_recv", ca_name, port->portnum );
	nwif->recv_cntr =
		addCounter( counter_name, "bytes received on this IB port", "bytes" );

	sprintf( counter_name, "%s_%d_send", ca_name, port->portnum );
	nwif->send_cntr =
		addCounter( counter_name, "bytes written to this IB port", "bytes" );

	nwif->port_rate = port->rate;
	nwif->is_initialized = 0;
	nwif->port_number = port->portnum;
	nwif->next = NULL;

	num_counters += 2;

	if ( root_ib_port == NULL ) {
		root_ib_port = nwif;
	} else {
		last = root_ib_port;
		while ( last->next != NULL )
			last = last->next;
		last->next = nwif;
	}
}


/**
 * initialize one IB port so that we are able to read values from it
 */
static int
init_ib_port( ib_port * portdata )
{
	int mgmt_classes[4] = { IB_SMI_CLASS, IB_SMI_DIRECT_CLASS, IB_SA_CLASS,
		IB_PERFORMANCE_CLASS
	};
	char *ca = 0;
	static uint8_t pc[1024];
	int mask = 0xFFFF;

	srcport = (*mad_rpc_open_portPtr)( ca, portdata->port_number, mgmt_classes, 4 );
	if ( !srcport ) {
		fprintf( stderr, "Failed to open '%s' port '%d'\n", ca,
				 portdata->port_number );
		exit( 1 );
	}

	if ( (*ib_resolve_self_viaPtr)( &portid, &ibportnum, 0, srcport ) < 0 ) {
		fprintf( stderr, "can't resolve self port\n" );
		exit( 1 );
	}

	/* PerfMgt ClassPortInfo is a required attribute */
	/* might be redundant, could be left out for fast implementation */
	if ( !(*pma_query_viaPtr) ( pc, &portid, ibportnum, ib_timeout, CLASS_PORT_INFO, srcport ) ) {
		fprintf( stderr, "classportinfo query\n" );
		exit( 1 );
	}

	if ( !(*performance_reset_viaPtr) ( pc, &portid, ibportnum, mask, ib_timeout, IB_GSI_PORT_COUNTERS, srcport ) ) {
		fprintf( stderr, "perf reset\n" );
		exit( 1 );
	}

	/* read the initial values */
	(*mad_decode_fieldPtr)( pc, IB_PC_XMT_BYTES_F, &portdata->last_send_val );
	portdata->sum_send_val = 0;
	(*mad_decode_fieldPtr)( pc, IB_PC_RCV_BYTES_F, &portdata->last_recv_val );
	portdata->sum_recv_val = 0;

	portdata->is_initialized = 1;

	return 0;
}


/**
 * read and reset IB counters (reset on demand)
 */
static int
read_ib_counter(  )
{
	uint32_t send_val;
	uint32_t recv_val;
	uint8_t pc[1024];
	/* 32 bit counter FFFFFFFF */
	uint32_t max_val = 4294967295;
	/* if it is bigger than this -> reset */
	uint32_t reset_limit = max_val * 0.7;
	int mask = 0xFFFF;

	if ( active_ib_port == NULL )
		return 0;

	/* reading cost ~70 mirco secs */
	if ( !(*pma_query_viaPtr) ( pc, &portid, ibportnum, ib_timeout, IB_GSI_PORT_COUNTERS, srcport ) ) {
		fprintf( stderr, "perfquery\n" );
		exit( 1 );
	}

	(*mad_decode_fieldPtr)( pc, IB_PC_XMT_BYTES_F, &send_val );
	(*mad_decode_fieldPtr)( pc, IB_PC_RCV_BYTES_F, &recv_val );

	/* multiply the numbers read by 4 as the IB port counters are not
	   counting bytes. they always count 32dwords. see man page of
	   perfquery for details
	   internally a uint64_t ia used to sum up the values */
	active_ib_port->sum_send_val +=
		( send_val - active_ib_port->last_send_val ) * 4;
	active_ib_port->sum_recv_val +=
		( recv_val - active_ib_port->last_recv_val ) * 4;

	active_ib_port->send_cntr->value = active_ib_port->sum_send_val;
	active_ib_port->recv_cntr->value = active_ib_port->sum_recv_val;

	if ( send_val > reset_limit || recv_val > reset_limit ) {
		/* reset cost ~70 mirco secs */
		if ( !(*performance_reset_viaPtr) ( pc, &portid, ibportnum, mask, ib_timeout, IB_GSI_PORT_COUNTERS, srcport ) ) {
			fprintf( stderr, "perf reset\n" );
			exit( 1 );
		}

		(*mad_decode_fieldPtr)( pc, IB_PC_XMT_BYTES_F, &active_ib_port->last_send_val );
		(*mad_decode_fieldPtr)( pc, IB_PC_RCV_BYTES_F, &active_ib_port->last_recv_val );
	} else {
		active_ib_port->last_send_val = send_val;
		active_ib_port->last_recv_val = recv_val;
	}

	return 0;
}


void
host_read_values( long long *data )
{
	int loop;

	read_ib_counter(  );

	for ( loop = 0; loop < INFINIBAND_MAX_COUNTERS; loop++ ) {
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
	int len;
	char tmp_name[512];
	ib_port *aktp;

	counter_info *counter = counterFromName( cntr );

	for ( loop = 0; loop < INFINIBAND_MAX_COUNTERS; loop++ ) {
		if ( subscriptions[loop] == NULL ) {
			subscriptions[loop] = counter;
			counter->idx = loop;

			/* we have an IB counter if the name ends with _send or _recv and
			   the prefix before that is in the ib_port list */
			if ( ( len = strlen( cntr ) ) > 5 ) {
				if ( strcmp( &cntr[len - 5], "_recv" ) == 0 ||
					 strcmp( &cntr[len - 5], "_send" ) == 0 ) {
					/* look through all IB_counters */
					strncpy( tmp_name, cntr, len - 5 );
					tmp_name[len - 5] = 0;
					aktp = root_ib_port;
					// printf("looking for IB port '%s'\n", tmp_name);
					while ( aktp != NULL ) {
						if ( strcmp( aktp->name, tmp_name ) == 0 ) {
							if ( !aktp->is_initialized ) {
								init_ib_port( aktp );
								active_ib_port = aktp;
							}
							return loop + 1;
						}
						/* name does not match, if this counter is
						   initialized, we can't have two active IB ports */
						if ( aktp->is_initialized ) {
#if 0	/* not necessary with OFED version >= 1.4 */
							fprintf( stderr,
									 "unable to activate IB port monitoring for more than one port\n" );
							exit( 1 );
#endif
						}
						aktp = aktp->next;
					}
				}
			}
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
	counter_info *cntr, *next;

	if ( is_finalized )
		return;

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

	ib_port *nwif, *last;
   last = root_ib_port;
   while ( last != NULL ) {                                    // While we have ports; 
      nwif = last;                                             // Copy the pointer.
      last = last->next;                                       // update the loop pointer now.
      if (nwif->name) free(nwif->name);                        // Free any name malloc.

      free(nwif);                                              // free the chain link itself.
	}

   root_ib_port = NULL;                                     // All done with this.

	is_finalized = 1;
} // end host_finalize()


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
 * This is called whenever a thread is initialized
 */
int
INFINIBAND_init_thread( hwd_context_t * ctx )
{
	string_list *counter_list = NULL;
	int i;
	int loop;

	/* initialize portid struct of type ib_portid_t to 0 */
	InitStruct( portid, ib_portid_t );

	if ( is_initialized )
		return PAPI_OK;

	is_initialized = 1;

	init_ib_counter(  );

	for ( loop = 0; loop < INFINIBAND_MAX_COUNTERS; loop++ )
		subscriptions[loop] = NULL;

	counter_list = host_listCounter( num_counters );

	for ( i = 0; i < counter_list->count; i++ )
		host_subscribe( counter_list->data[i] );

	( ( INFINIBAND_context_t * ) ctx )->state.ncounter = counter_list->count;

	host_deleteStringList( counter_list );

	return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
INFINIBAND_init_component( int cidx )
{
	SUBDBG ("Entry: cidx: %d\n", cidx);
	int i;

	/* link in all the infiniband libraries and resolve the symbols we need to use */
	if (linkInfinibandLibraries() != PAPI_OK) {
		SUBDBG ("Dynamic link of Infiniband libraries failed, component will be disabled.\n");
		SUBDBG ("See disable reason in papi_component_avail output for more details.\n");
		return (PAPI_ENOSUPP);
	}

	/* make sure that the infiniband library finds the kernel module loaded. */
	if ( (*umad_initPtr)(  ) < 0 ) {
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Call to initialize umad library failed.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	for ( i = 0; i < INFINIBAND_MAX_COUNTERS; i++ ) {
		_papi_hwd_infiniband_register_start[i] = -1;
		_papi_hwd_infiniband_register[i] = -1;
	}

	/* Export the component id */
	_infiniband_umad_vector.cmp_info.CmpIdx = cidx;

	return ( PAPI_OK );
}


/*
 * Link the necessary Infiniband libraries to use the Infiniband component.  If any of them can not be found, then
 * the Infiniband component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the Infiniband component can be installed and used on systems which have the Infiniband libraries installed
 * and on systems where these libraries are not installed.
 */
static int
linkInfinibandLibraries ()
{
	/* Attempt to guess if we were statically linked to libc, if so bail */
	if ( _dl_non_dynamic_init != NULL ) {
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "The Infiniband component does not support statically linking of libc.", PAPI_MAX_STR_LEN);
		return PAPI_ENOSUPP;
	}

	/* Need to link in the Infiniband libraries, if not found disable the component */
	dl1 = dlopen("libibumad.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dl1)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband library libibumad.so not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	umad_initPtr = dlsym(dl1, "umad_init");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function umad_init not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	umad_get_cas_namesPtr = dlsym(dl1, "umad_get_cas_names");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function umad_get_cas_names not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	umad_get_caPtr = dlsym(dl1, "umad_get_ca");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function umad_get_ca not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	/* Need to link in the Infiniband libraries, if not found disable the component */
	dl2 = dlopen("libibmad.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dl2)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband library libibmad.so not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	mad_decode_fieldPtr = dlsym(dl2, "mad_decode_field");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function mad_decode_field not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	mad_rpc_open_portPtr = dlsym(dl2, "mad_rpc_open_port");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function mad_rpc_open_port not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	ib_resolve_self_viaPtr = dlsym(dl2, "ib_resolve_self_via");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function ib_resolve_self_via not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	performance_reset_viaPtr = dlsym(dl2, "performance_reset_via");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function performance_reset_via not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	pma_query_viaPtr = dlsym(dl2, "pma_query_via");
	if (dlerror() != NULL)
	{
		strncpy(_infiniband_umad_vector.cmp_info.disabled_reason, "Infiniband function pma_query_via not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	return ( PAPI_OK );
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
INFINIBAND_init_control_state( hwd_control_state_t * ctrl )
{
	( void ) ctrl;
	return PAPI_OK;
}


/*
 *
 */
int
INFINIBAND_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;

	host_read_values( _papi_hwd_infiniband_register_start );

	memcpy( _papi_hwd_infiniband_register, _papi_hwd_infiniband_register_start,
			INFINIBAND_MAX_COUNTERS * sizeof ( long long ) );

	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	int i;
	( void ) ctx;

	host_read_values( _papi_hwd_infiniband_register );

	for ( i = 0; i < ( ( INFINIBAND_context_t * ) ctx )->state.ncounter; i++ ) {
		( ( INFINIBAND_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_infiniband_register[i] -
			_papi_hwd_infiniband_register_start[i];
	}

	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
				 long_long ** events, int flags )
{
	int i;
	( void ) flags;

	host_read_values( _papi_hwd_infiniband_register );

	for ( i = 0; i < ( ( INFINIBAND_context_t * ) ctx )->state.ncounter; i++ ) {
		( ( INFINIBAND_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_infiniband_register[i] -
			_papi_hwd_infiniband_register_start[i];
	}

	*events = ( ( INFINIBAND_control_state_t * ) ctrl )->counts;
	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_shutdown_thread( hwd_context_t * ctx )
{
	( void ) ctx;
	host_finalize(  );
	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_shutdown_component( void )
{
	// close the dynamic libraries needed by this component (opened in the init substrate call)
	dlclose(dl1);
	dlclose(dl2);

	return ( PAPI_OK );
} // end Shutdown component.


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
INFINIBAND_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}


//int INFINIBAND_ntv_code_to_bits ( unsigned int EventCode, hwd_register_t * bits );


/*
 *
 */
int
INFINIBAND_update_control_state( hwd_control_state_t * ptr,
								 NativeInfo_t * native, int count,
								 hwd_context_t * ctx )
{
	( void ) ptr;
	( void ) ctx;
	int i, index;

	for ( i = 0; i < count; i++ ) {
		index = native[i].ni_event;
		native[i].ni_position = index;
	}

	return ( PAPI_OK );
}


/*
 * Infiniband counts are system wide, so this is the only domain we will respond to
 */
int
INFINIBAND_set_domain( hwd_control_state_t * cntrl, int domain )
{
	(void) cntrl;
	if ( PAPI_DOM_ALL != domain )
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	INFINIBAND_start( ctx, ctrl );
	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
INFINIBAND_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	if ( modifier == PAPI_ENUM_FIRST ) {
		*EventCode = 0;
		return PAPI_OK;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode;

		if ( infiniband_native_table[index + 1] ) {
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
int
INFINIBAND_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	strncpy( name, infiniband_native_table[EventCode]->name, len );

	return PAPI_OK;
}


/*
 *
 */
int
INFINIBAND_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	strncpy( name, infiniband_native_table[EventCode]->description, len );

	return PAPI_OK;
}


/*
 *
 */
int
INFINIBAND_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	memcpy( ( INFINIBAND_register_t * ) bits,
			infiniband_native_table[EventCode],
			sizeof ( INFINIBAND_register_t ) );

	return PAPI_OK;
}


/*
 *
 */
papi_vector_t _infiniband_umad_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name ="infiniband_umad",
				 .short_name="infiniband_umad",
				 .version = "4.2.1",
				 .description = "Infiniband statistics (for OFED versions < 1.4)",
				 .num_mpx_cntrs = INFINIBAND_MAX_COUNTERS,
				 .num_cntrs = INFINIBAND_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_ALL,
				 .available_domains = PAPI_DOM_ALL,
				 .default_granularity = PAPI_GRN_SYS,
				 .available_granularities = PAPI_GRN_SYS,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( INFINIBAND_context_t ),
			 .control_state = sizeof ( INFINIBAND_control_state_t ),
			 .reg_value = sizeof ( INFINIBAND_register_t ),
			 .reg_alloc = sizeof ( INFINIBAND_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init_thread = INFINIBAND_init_thread,
	.init_component = INFINIBAND_init_component,
	.init_control_state = INFINIBAND_init_control_state,
	.start = INFINIBAND_start,
	.stop = INFINIBAND_stop,
	.read = INFINIBAND_read,
	.shutdown_component = INFINIBAND_shutdown_component,
	.shutdown_thread = INFINIBAND_shutdown_thread,
	.ctl = INFINIBAND_ctl,

	.update_control_state = INFINIBAND_update_control_state,
	.set_domain = INFINIBAND_set_domain,
	.reset = INFINIBAND_reset,

	.ntv_enum_events = INFINIBAND_ntv_enum_events,
	.ntv_code_to_name = INFINIBAND_ntv_code_to_name,
	.ntv_code_to_descr = INFINIBAND_ntv_code_to_descr,
	.ntv_code_to_bits = INFINIBAND_ntv_code_to_bits,
};
