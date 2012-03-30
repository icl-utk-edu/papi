/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-infiniband.c
 * @author  Heike Jagode (in collaboration with Michael Kluge, TU Dresden)
 *          jagode@eecs.utk.edu
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

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include "linux-infiniband.h"

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

	if ( umad_init(  ) < 0 ) {
		fprintf( stderr, "can't init UMAD library\n" );
		exit( 1 );
	}

	if ( ( n = umad_get_cas_names( ( void * ) names, UMAD_CA_NAME_LEN ) ) < 0 ) {
		fprintf( stderr, "can't list IB device names\n" );
		exit( 1 );
	}

	for ( i = 0; i < n; i++ ) {
		ca_name = names[i];

		if ( ( r = umad_get_ca( ca_name, &ca ) ) < 0 ) {
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

	srcport = mad_rpc_open_port( ca, portdata->port_number, mgmt_classes, 4 );
	if ( !srcport ) {
		fprintf( stderr, "Failed to open '%s' port '%d'\n", ca,
				 portdata->port_number );
		exit( 1 );
	}

	if ( ib_resolve_self_via( &portid, &ibportnum, 0, srcport ) < 0 ) {
		fprintf( stderr, "can't resolve self port\n" );
		exit( 1 );
	}

	/* PerfMgt ClassPortInfo is a required attribute */
	/* might be redundant, could be left out for fast implementation */
	if ( !pma_query_via
		 ( pc, &portid, ibportnum, ib_timeout, CLASS_PORT_INFO, srcport ) ) {
		fprintf( stderr, "classportinfo query\n" );
		exit( 1 );
	}

	if ( !performance_reset_via
		 ( pc, &portid, ibportnum, mask, ib_timeout, IB_GSI_PORT_COUNTERS,
		   srcport ) ) {
		fprintf( stderr, "perf reset\n" );
		exit( 1 );
	}

	/* read the initial values */
	mad_decode_field( pc, IB_PC_XMT_BYTES_F, &portdata->last_send_val );
	portdata->sum_send_val = 0;
	mad_decode_field( pc, IB_PC_RCV_BYTES_F, &portdata->last_recv_val );
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
	if ( !pma_query_via
		 ( pc, &portid, ibportnum, ib_timeout, IB_GSI_PORT_COUNTERS,
		   srcport ) ) {
		fprintf( stderr, "perfquery\n" );
		exit( 1 );
	}

	mad_decode_field( pc, IB_PC_XMT_BYTES_F, &send_val );
	mad_decode_field( pc, IB_PC_RCV_BYTES_F, &recv_val );

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
		if ( !performance_reset_via
			 ( pc, &portid, ibportnum, mask, ib_timeout, IB_GSI_PORT_COUNTERS,
			   srcport ) ) {
			fprintf( stderr, "perf reset\n" );
			exit( 1 );
		}

		mad_decode_field( pc, IB_PC_XMT_BYTES_F,
						  &active_ib_port->last_send_val );
		mad_decode_field( pc, IB_PC_RCV_BYTES_F,
						  &active_ib_port->last_recv_val );
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
 * This is called whenever a thread is initialized
 */
int
INFINIBAND_init( hwd_context_t * ctx )
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
INFINIBAND_init_substrate(  )
{
	int i;

	for ( i = 0; i < INFINIBAND_MAX_COUNTERS; i++ ) {
		_papi_hwd_infiniband_register_start[i] = -1;
		_papi_hwd_infiniband_register[i] = -1;
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
INFINIBAND_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;
	host_finalize(  );
	return ( PAPI_OK );
}



/* This function sets various options in the substrate
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
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = index;
	}

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
INFINIBAND_set_domain( hwd_control_state_t * cntrl, int domain )
{
	int found = 0;
	( void ) cntrl;

	if ( PAPI_DOM_USER & domain )
		found = 1;

	if ( PAPI_DOM_KERNEL & domain )
		found = 1;

	if ( PAPI_DOM_OTHER & domain )
		found = 1;

	if ( !found )
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
		/* assumes first native event is always 0x4000000 */
		*EventCode = PAPI_NATIVE_MASK;
		return PAPI_OK;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

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
	strncpy( name,
			 infiniband_native_table[EventCode & PAPI_NATIVE_AND_MASK &
									 PAPI_COMPONENT_AND_MASK]->name, len );

	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 infiniband_native_table[EventCode & PAPI_NATIVE_AND_MASK &
									 PAPI_COMPONENT_AND_MASK]->description,
			 len );

	return ( PAPI_OK );
}


/*
 *
 */
int
INFINIBAND_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	memcpy( ( INFINIBAND_register_t * ) bits,
			infiniband_native_table[EventCode & PAPI_NATIVE_AND_MASK &
									PAPI_COMPONENT_AND_MASK],
			sizeof ( INFINIBAND_register_t ) );

	return ( PAPI_OK );
}


/*
 *
 */
papi_vector_t _infiniband_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name ="linux-infiniband.c",
				 .version = "4.2.1",
				 .description = "Infiniband statistics",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = INFINIBAND_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER,
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
			 .context = sizeof ( INFINIBAND_context_t ),
			 .control_state = sizeof ( INFINIBAND_control_state_t ),
			 .reg_value = sizeof ( INFINIBAND_register_t ),
			 .reg_alloc = sizeof ( INFINIBAND_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = INFINIBAND_init,
	.init_substrate = INFINIBAND_init_substrate,
	.init_control_state = INFINIBAND_init_control_state,
	.start = INFINIBAND_start,
	.stop = INFINIBAND_stop,
	.read = INFINIBAND_read,
	.shutdown = INFINIBAND_shutdown,
	.ctl = INFINIBAND_ctl,

	.update_control_state = INFINIBAND_update_control_state,
	.set_domain = INFINIBAND_set_domain,
	.reset = INFINIBAND_reset,

	.ntv_enum_events = INFINIBAND_ntv_enum_events,
	.ntv_code_to_name = INFINIBAND_ntv_code_to_name,
	.ntv_code_to_descr = INFINIBAND_ntv_code_to_descr,
	.ntv_code_to_bits = INFINIBAND_ntv_code_to_bits,
};
