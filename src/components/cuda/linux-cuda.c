/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-cuda.c
 * CVS:     $Id$
 * @author  Heike Jagode (in collaboration with Robert Dietrich, TU Dresden)
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * CUDA component 
 * 
 * Tested version of CUPTI (CUDA Tools SDK 4.0)
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for GPU devices through the  
 *  CUPTI library.
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include "linux-cuda.h"


/*******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT **********
 ******************************************************************************/
/*
 * Specify device(s): Counts number of cuda events available in this system
 */
static int
detectDevice( void )
{
	CUresult err;
	int skipDevice = 0;
	int id;
	char deviceName_tmp[PAPI_MIN_STR_LEN] = "init";
	
	totalEventCount = 0;
	
	/* CUDA initialization  */
	err = cuInit( 0 );
	if ( err != CUDA_SUCCESS ) {
		printf( "Initialization of CUDA library failed.\n" );
		exit( EXIT_FAILURE );
	}

	/* How many gpgpu devices do we have? */
	err = cuDeviceGetCount( &deviceCount );
	CHECK_CU_ERROR( err, "cuDeviceGetCount" );
	if ( deviceCount == 0 ) {
		printf( "There is no device supporting CUDA.\n" );
		exit( EXIT_FAILURE );
	}

	/* allocate memory for device data table */
	device = ( DeviceData_t * ) malloc( sizeof ( DeviceData_t ) * deviceCount );
	if ( device == NULL ) {
		perror( "malloc(): Failed to allocate memory to CUDA device table" );
		exit( EXIT_FAILURE );
	}

	/* What are the devices? Get Name and # of domains per device */
	for ( id = 0; id < deviceCount; id++ ) {
		err = cuDeviceGet( &device[id].dev, id );
		CHECK_CU_ERROR( err, "cuDeviceGet" );

		err =
			cuDeviceGetName( device[id].name, PAPI_MIN_STR_LEN,
							 device[id].dev );
		CHECK_CU_ERROR( err, "cuDeviceGetName" );

		/* Skip device if there are multiple of the same type 
		   and if it has been already added to the list */
		if ( 0 == strcmp( deviceName_tmp, device[id].name ) ) {
			skipDevice++;
			continue;
		}

		strcpy( deviceName_tmp, device[id].name );

		/* enumerate the domains on the device */
		if ( 0 != enumEventDomains( device[id].dev, id ) )
			exit( EXIT_FAILURE );
	}

	deviceCount = deviceCount - skipDevice;

	/* return number of events provided via CuPTI */
	return totalEventCount;
}


/*
 * Detect supported domains for specified device
 */
static int
enumEventDomains( CUdevice dev, int deviceId )
{
	CUptiResult err = CUPTI_SUCCESS;
	CUpti_EventDomainID *domainId = NULL;
	uint32_t id = 0;
	size_t size = 0;

	device[deviceId].domainCount = 0;

	/* get number of domains for device dev */
	err = cuptiDeviceGetNumEventDomains( dev, &device[deviceId].domainCount );
	CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );

	if ( device[deviceId].domainCount == 0 ) {
		printf( "No domain is exposed by dev = %d\n", dev );
		return -1;
	}

	/* CuPTI domain struct */
	size = sizeof ( CUpti_EventDomainID ) * device[deviceId].domainCount;
	domainId = ( CUpti_EventDomainID * ) malloc( size );
	if ( domainId == NULL ) {
		perror( "malloc(): Failed to allocate memory to CuPTI domain ID" );
		return -1;
	}
	memset( domainId, 0, size );

	/* PAPI domain struct */
	device[deviceId].domain =
		( DomainData_t * ) malloc( sizeof ( DomainData_t ) *
								   device[deviceId].domainCount );
	if ( device[deviceId].domain == NULL ) {
		perror( "malloc(): Failed to allocate memory to PAPI domain struct" );
		return -1;
	}

	/* Enumerates the event domains for a device dev */
	err = cuptiDeviceEnumEventDomains( dev, &size, domainId );
	CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );

	/* enum domains */
	for ( id = 0; id < device[deviceId].domainCount; id++ ) {
		device[deviceId].domain[id].domainId = domainId[id];

		/* query domain name */
		size = PAPI_MIN_STR_LEN;
		err = cuptiEventDomainGetAttribute( dev,
											device[deviceId].domain[id].
											domainId,
											CUPTI_EVENT_DOMAIN_ATTR_NAME, &size,
											( void * ) device[deviceId].
											domain[id].name );
		CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetAttribute" );

		/* query num of events avaialble in the domain */
		size = sizeof ( device[deviceId].domain[id].eventCount );
		err = cuptiEventDomainGetAttribute( dev,
											device[deviceId].domain[id].
											domainId,
											CUPTI_EVENT_DOMAIN_MAX_EVENTS,
											&size,
											( void * ) &device[deviceId].
											domain[id].eventCount );
		CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetAttribute" );

		/* enumerate the events for the domain[id] on the device dev */
		if ( 0 != enumEvents( dev, deviceId, id ) )
			return -1;
	}

	totalDomainCount += device[deviceId].domainCount;
	free( domainId );
	return 0;
}


/*
 * Detect supported events for specified device domain
 */
static int
enumEvents( CUdevice dev, int deviceId, int domainId )
{
	CUptiResult err = CUPTI_SUCCESS;
	CUpti_EventID *eventId = NULL;
	size_t size = 0;
	int id = 0;

	/* CuPTI event struct */
	size =
		sizeof ( CUpti_EventID ) * device[deviceId].domain[domainId].eventCount;
	eventId = ( CUpti_EventID * ) malloc( size );
	if ( eventId == NULL ) {
		perror( "malloc(): Failed to allocate memory to CuPTI event ID" );
		return -1;
	}
	memset( eventId, 0, size );

	/* PAPI event struct */
	device[deviceId].domain[domainId].event =
		( EventData_t * ) malloc( sizeof ( EventData_t ) *
								  device[deviceId].domain[domainId].
								  eventCount );
	if ( device[deviceId].domain[domainId].event == NULL ) {
		perror( "malloc(): Failed to allocate memory to PAPI event struct" );
		return -1;
	}

	/* enumerate the events for the domain[domainId] on the device[deviceId] */
	err =
		cuptiEventDomainEnumEvents( dev,
									( CUpti_EventDomainID ) device[deviceId].
									domain[domainId].domainId, &size, eventId );
	CHECK_CUPTI_ERROR( err, "cuptiEventDomainEnumEvents" );

	/* query event info */
	for ( id = 0; id < device[deviceId].domain[domainId].eventCount; id++ ) {
		device[deviceId].domain[domainId].event[id].eventId = eventId[id];

		/* query event name */
		size = PAPI_MIN_STR_LEN;
		err = cuptiEventGetAttribute( dev,
									  device[deviceId].domain[domainId].
									  event[id].eventId, CUPTI_EVENT_ATTR_NAME,
									  &size,
									  ( uint8_t * ) device[deviceId].
									  domain[domainId].event[id].name );
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute" );

		/* query event description */
		size = PAPI_2MAX_STR_LEN;
		err = cuptiEventGetAttribute( dev,
									  device[deviceId].domain[domainId].
									  event[id].eventId,
									  CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size,
									  ( uint8_t * ) device[deviceId].
									  domain[domainId].event[id].desc );
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute" );
	}

	totalEventCount += device[deviceId].domain[domainId].eventCount;
	free( eventId );
	return 0;
}


/*
 * Create the native events for specified domain and device
 */
static int
createNativeEvents( void )
{
	int deviceId, eventId, id = 0;
	uint32_t domainId;
	int cuptiDomainId;
	int i;
	int devNameLen;
	cmp_id_t component;

	/* component name and description */
	strcpy( component.name, "CUDA" );
	strcpy( component.descr,
			"CuPTI provides the API for monitoring CUDA hardware events" );

	/* create events for every GPU device and every domain per device  */
	for ( deviceId = 0; deviceId < deviceCount; deviceId++ ) {
		/* for the event names, replace blanks in the device name with underscores */
		devNameLen = strlen( device[deviceId].name );
		for ( i = 0; i < devNameLen; i++ )
			if ( device[deviceId].name[i] == ' ' )
				device[deviceId].name[i] = '_';
		
		for ( domainId = 0; domainId < device[deviceId].domainCount;
			  domainId++ ) {
			cuptiDomainId = device[deviceId].domain[domainId].domainId;

			for ( eventId = 0;
				  eventId < device[deviceId].domain[domainId].eventCount;
				  eventId++ ) {
				/* Save native event data */
				sprintf( cuda_native_table[id].name,
						 "%s.%s.%s.%s",
						 component.name,
						 device[deviceId].name,
						 device[deviceId].domain[domainId].name,
						 device[deviceId].domain[domainId].event[eventId].name );

				strncpy( cuda_native_table[id].description,
						 device[deviceId].domain[domainId].event[eventId].desc,
						 PAPI_2MAX_STR_LEN );

				/* The selector has to be !=0 . Starts with 1 */
				cuda_native_table[id].resources.selector = id + 1;

				/* store event ID */
				cuda_native_table[id].resources.eventId =
					device[deviceId].domain[domainId].event[eventId].eventId;

				/* increment the table index counter */
				id++;
			}
		}
	}

	/* Return the number of events created */
	return id;
}


/*
 * Returns all event values from the CuPTI eventGroup 
 */
static int
getEventValue( long long *counts )
{
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	size_t events_read, bufferSizeBytes, arraySizeBytes, i;
	uint64_t *counterDataBuffer;
	CUpti_EventID *eventIDArray;
	int j;

	bufferSizeBytes = addedEvents.count * sizeof ( uint64_t );
	counterDataBuffer = ( uint64_t * ) malloc( bufferSizeBytes );

	arraySizeBytes = addedEvents.count * sizeof ( CUpti_EventID );
	eventIDArray = ( CUpti_EventID * ) malloc( arraySizeBytes );

	/* read counter data for the specified event from the CuPTI eventGroup */
	cuptiErr = cuptiEventGroupReadAllEvents( eventGroup,
											 CUPTI_EVENT_READ_FLAG_NONE,
											 &bufferSizeBytes,
											 counterDataBuffer, &arraySizeBytes,
											 eventIDArray, &events_read );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );

	if ( events_read != ( size_t ) addedEvents.count )
		return -1;

	/* Since there is no guarantee that returned counter values are in the same 
	 order as the counters in the PAPI addedEvents.list, we need to map the
	 CUpti_EventID to PAPI event ID values.
	 According to CuPTI doc: counter return values of counterDataBuffer 
	 correspond to the return event IDs in eventIDArray */
	for ( i = 0; i < events_read; i++ )
		for ( j = 0; j < addedEvents.count; j++ )
			if ( cuda_native_table[addedEvents.list[j]].resources.eventId ==
				eventIDArray[i] )
				counts[addedEvents.list[j]] = counterDataBuffer[i];
	
	free( counterDataBuffer );
	free( eventIDArray );
	return 0;
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
int
CUDA_init( hwd_context_t * ctx )
{
	( void ) ctx;
	return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
CUDA_init_substrate(  )
{
	int i;
	
	/* Initialize CuPTI library */
	if ( 0 != cuInit( 0 ) ) {
		perror( "cuInit(): Failed to initialize the CUDA library" );
		exit( EXIT_FAILURE );
	}
	
	/* Create dynamic event table */
	NUM_EVENTS = detectDevice(  );
	
	cuda_native_table = ( CUDA_native_event_entry_t * )
		malloc( sizeof ( CUDA_native_event_entry_t ) * NUM_EVENTS );
	if ( cuda_native_table == NULL ) {
		perror( "malloc(): Failed to allocate memory to events table" );
		exit( EXIT_FAILURE );
	}
	
	if ( NUM_EVENTS != createNativeEvents(  ) ) {
		fprintf( stderr, "Number of CUDA events mismatch!\n" );
		exit( EXIT_FAILURE );
	}
	
	/* allocate memory for the list of events that are added to the CuPTI eventGroup */
	addedEvents.list = malloc( sizeof ( int ) * NUM_EVENTS );
	if ( addedEvents.list == NULL ) {
		perror
			( "malloc(): Failed to allocate memory to table of events that are added to CuPTI eventGroup" );
		exit( EXIT_FAILURE );
	}
	
	/* initialize the event list */
	for ( i = 0; i < NUM_EVENTS; i++ )
		addedEvents.list[i] = 0;
	
	return ( PAPI_OK );
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
CUDA_init_control_state( hwd_control_state_t * ctrl )
{
	( void ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	CUresult cuErr = CUDA_SUCCESS;

	/* TODO: works only for one device right now; 
	   need to find out if user can use 2 or more devices at same time */

	/* want create a CUDA context for either the default device or
	   the device specified with cudaSetDevice() in user code */
	if ( CUDA_SUCCESS != cudaGetDevice( &currentDeviceID ) ) {
		printf( "There is no device supporting CUDA.\n" );
		exit( EXIT_FAILURE );
	}
	printf( "DEVICE USED: %s (%d)\n", device[currentDeviceID].name,
			currentDeviceID );
	
	/* get the CUDA context from the calling CPU thread */
	cuErr = cuCtxGetCurrent( &cuCtx );

	/* if no CUDA context is bound to the calling CPU thread yet, create one */
	if ( cuErr != CUDA_SUCCESS || cuCtx == NULL ) {
		cuErr = cuCtxCreate( &cuCtx, 0, device[currentDeviceID].dev );
		CHECK_CU_ERROR( cuErr, "cuCtxCreate" );
	}
	
	cuptiErr = cuptiEventGroupCreate( cuCtx, &eventGroup, 0 );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupCreate" );

	return PAPI_OK;
}


/*
 *
 */
int
CUDA_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;

	cuptiErr = cuptiEventGroupEnable( eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupEnable" );

	/* Resets all events in the CuPTI eventGroup to zero */
	cuptiErr = cuptiEventGroupResetAllEvents( eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupResetAllEvents" );

	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;

	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
		   long_long ** events, int flags )
{
	( void ) ctx;
	( void ) flags;

	
	if ( 0 != getEventValue( ( ( CUDA_control_state_t * ) ctrl )->counts ) )
		exit( EXIT_FAILURE );

	*events = ( ( CUDA_control_state_t * ) ctrl )->counts;

	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;
	uint32_t j;
	int i;

	/* deallocate all the memory */
	for ( i = 0; i < deviceCount; i++ )
		for ( j = 0; j < device[i].domainCount; j++ )
			free( device[i].domain[j].event );
	free( device[i].domain );

	free( device );
	free( cuda_native_table );
	free( addedEvents.list );

	return ( PAPI_OK );
}


/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
CUDA_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}


//int CUDA_ntv_code_to_bits ( unsigned int EventCode, hwd_register_t * bits );


/*
 *
 */
int
CUDA_update_control_state( hwd_control_state_t * ptr,
						   NativeInfo_t * native, int count,
						   hwd_context_t * ctx )
{
	( void ) ptr;
	( void ) ctx;
	int index;
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	char *device_tmp;
	
	if ( count == 0 ) {
		printf("Removing CUDA events is currently not supported (Bug in cuda 4.0rc).\n");
		exit( EXIT_FAILURE );
	}
	
	index =
		native[count -
			   1].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
	native[count - 1].ni_position = index;

	/* store events, that have been added to the CuPTI eveentGroup 
	   in a seperate place (addedEvents).
	   Needed, so that we can read the values for the added events only */
	addedEvents.count = count;
	addedEvents.list[count - 1] = index;	
	
	/* determine the device name from the event name chosen */
	device_tmp = strchr( cuda_native_table[index].name, '.' );

	/* if this device name is different from the actual device the code is running on, then exit */
	if ( 0 != strncmp( device[currentDeviceID].name,
						device_tmp + 1,
						strlen( device[currentDeviceID].name ) ) ) {
		printf
			( "Device %s is used -- BUT event %s is collected. \n ---> ERROR: Specify events for the device that is used!\n\n",
				device[currentDeviceID].name, cuda_native_table[index].name );
		exit( EXIT_FAILURE ); 
	}

	/* Add events to the CuPTI eventGroup */
	cuptiErr =
	cuptiEventGroupAddEvent( eventGroup,
							cuda_native_table[index].resources.eventId );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupAddEvent" );
	
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
CUDA_set_domain( hwd_control_state_t * cntrl, int domain )
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
CUDA_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;

	/* Resets all events in the CuPTI eventGroup to zero */
	cuptiErr = cuptiEventGroupResetAllEvents( eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupResetAllEvents" );

	return ( PAPI_OK );
}


/*
 * Disable and Destoy the CUDA eventGroup 
 */
int 
CUDA_destroy_eventset( int * EventSet )
{
	( void ) EventSet;
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	CUresult cuErr = CUDA_SUCCESS;

	/* Disable the CUDA eventGroup; 
	   it also frees the perfmon hardware on the GPU */
	cuptiErr = cuptiEventGroupDisable( eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDisable" );
	
	/* Call the CuPTI cleaning function before leaving */
	cuptiErr = cuptiEventGroupDestroy( eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDestroy" );

	/* destroy floating CUDA context */
	cuErr = cuCtxDestroy( cuCtx );
	CHECK_CU_ERROR( cuErr, "cuCtxDestroy" );
	
	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
CUDA_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );

		return ( PAPI_OK );
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( index < NUM_EVENTS - 1 ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );

		break;
	}
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_EINVAL );
}


/*
 *
 */
int
CUDA_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, cuda_native_table[index].name, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, cuda_native_table[index].description, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	memcpy( ( CUDA_register_t * ) bits,
			&( cuda_native_table[index].resources ),
			sizeof ( CUDA_register_t ) );

	return ( PAPI_OK );
}


/*
 *
 */
papi_vector_t _cuda_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "$Id$",
				 .version = "$Revision$",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = CUDA_MAX_COUNTERS,
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
			 .context = sizeof ( CUDA_context_t ),
			 .control_state = sizeof ( CUDA_control_state_t ),
			 .reg_value = sizeof ( CUDA_register_t ),
			 .reg_alloc = sizeof ( CUDA_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = CUDA_init,
	.init_substrate = CUDA_init_substrate,
	.init_control_state = CUDA_init_control_state,
	.start = CUDA_start,
	.stop = CUDA_stop,
	.read = CUDA_read,
	.shutdown = CUDA_shutdown,
	.destroy_eventset = CUDA_destroy_eventset,
	.ctl = CUDA_ctl,

	.update_control_state = CUDA_update_control_state,
	.set_domain = CUDA_set_domain,
	.reset = CUDA_reset,

	.ntv_enum_events = CUDA_ntv_enum_events,
	.ntv_code_to_name = CUDA_ntv_code_to_name,
	.ntv_code_to_descr = CUDA_ntv_code_to_descr,
	.ntv_code_to_bits = CUDA_ntv_code_to_bits,
	.ntv_bits_to_info = NULL,
};
