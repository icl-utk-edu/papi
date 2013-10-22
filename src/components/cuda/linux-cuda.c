/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-cuda.c
 * @author  Heike Jagode (in collaboration with Robert Dietrich, TU Dresden)
 *          jagode@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for GPU devices through the  
 *  CUPTI library.
 */
#include <dlfcn.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "linux-cuda.h"


/********  CHANGE PROTOTYPES TO DECLARE CUDA LIBRARY SYMBOLS AS WEAK  **********
 *  This is done so that a version of PAPI built with the cuda component can   *
 *  be installed on a system which does not have the cuda libraries installed. *
 *                                                                             *
 *  If this is done without these prototypes, then all papi services on the    *
 *  system without the cuda libraries installed will fail.  The PAPI libraries *
 *  contain references to the cuda libraries which are not installed.  The     *
 *  load of PAPI commands fails because the cuda library references can not be *
 *  resolved.                                                                  *
 *                                                                             *
 *  This also defines pointers to the cuda library functions that we call.     *
 *  These function pointers will be resolved with dlopen/dlsym calls at        *
 *  component initialization time.  The component then calls the cuda library  *
 *  functions through these function pointers.                                 *
 *******************************************************************************/
void (*_dl_non_dynamic_init)(void) __attribute__((weak));
#undef CUDAAPI
#define CUDAAPI __attribute__((weak))
CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy(CUcontext);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *);
CUresult CUDAAPI cuDeviceGet(CUdevice *, int);
CUresult CUDAAPI cuDeviceGetCount(int *);
CUresult CUDAAPI cuDeviceGetName(char *, int, CUdevice);
CUresult CUDAAPI cuInit(unsigned int);

CUresult (*cuCtxCreatePtr)(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult (*cuCtxDestroyPtr)(CUcontext);
CUresult (*cuCtxGetCurrentPtr)(CUcontext *);
CUresult (*cuDeviceGetPtr)(CUdevice *, int);
CUresult (*cuDeviceGetCountPtr)(int *);
CUresult (*cuDeviceGetNamePtr)(char *, int, CUdevice);
CUresult (*cuInitPtr)(unsigned int);

#undef CUDARTAPI
#define CUDARTAPI __attribute__((weak))
cudaError_t CUDARTAPI cudaFree(void *);
cudaError_t CUDARTAPI cudaGetDevice(int *);
cudaError_t CUDARTAPI cudaRuntimeGetVersion( int *);
cudaError_t CUDARTAPI cudaDriverGetVersion( int *);

cudaError_t (*cudaFreePtr)(void *);
cudaError_t (*cudaGetDevicePtr)(int *);
cudaError_t (*cudaRuntimeGetVersionPtr)(int *);
cudaError_t (*cudaDriverGetVersionPtr)(int *);

#undef CUPTIAPI
#define CUPTIAPI __attribute__((weak))
CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains(CUdevice, size_t *, CUpti_EventDomainID *);
CUptiResult CUPTIAPI cuptiDeviceGetEventDomainAttribute(CUdevice, CUpti_EventDomainID, CUpti_EventDomainAttribute, size_t *, void *);
CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains(CUdevice, uint32_t *);
CUptiResult CUPTIAPI cuptiEventDomainEnumEvents(CUpti_EventDomainID, size_t*, CUpti_EventID *);
CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents(CUpti_EventDomainID, uint32_t *);
CUptiResult CUPTIAPI cuptiEventGetAttribute(CUpti_EventID, CUpti_EventAttribute, size_t *, void *);
CUptiResult CUPTIAPI cuptiEventGroupAddEvent(CUpti_EventGroup, CUpti_EventID);
CUptiResult CUPTIAPI cuptiEventGroupCreate(CUcontext, CUpti_EventGroup *, uint32_t);
CUptiResult CUPTIAPI cuptiEventGroupDestroy(CUpti_EventGroup);
CUptiResult CUPTIAPI cuptiEventGroupDisable(CUpti_EventGroup);
CUptiResult CUPTIAPI cuptiEventGroupEnable(CUpti_EventGroup);
CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents(CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *, size_t *);
CUptiResult CUPTIAPI cuptiEventGroupRemoveAllEvents(CUpti_EventGroup);
CUptiResult CUPTIAPI cuptiEventGroupResetAllEvents(CUpti_EventGroup);

CUptiResult (*cuptiDeviceEnumEventDomainsPtr)(CUdevice, size_t *, CUpti_EventDomainID *);
CUptiResult (*cuptiDeviceGetEventDomainAttributePtr)(CUdevice, CUpti_EventDomainID, CUpti_EventDomainAttribute, size_t *, void *);
CUptiResult (*cuptiDeviceGetNumEventDomainsPtr)(CUdevice, uint32_t *);
CUptiResult (*cuptiEventDomainEnumEventsPtr)(CUpti_EventDomainID, size_t*, CUpti_EventID *);
CUptiResult (*cuptiEventDomainGetNumEventsPtr)(CUpti_EventDomainID, uint32_t *);
CUptiResult (*cuptiEventGetAttributePtr)(CUpti_EventID, CUpti_EventAttribute, size_t *, void *);
CUptiResult (*cuptiEventGroupAddEventPtr)(CUpti_EventGroup, CUpti_EventID);
CUptiResult (*cuptiEventGroupCreatePtr)(CUcontext, CUpti_EventGroup *, uint32_t);
CUptiResult (*cuptiEventGroupDestroyPtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupDisablePtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupEnablePtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupReadAllEventsPtr)(CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *, size_t *);
CUptiResult (*cuptiEventGroupRemoveAllEventsPtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupResetAllEventsPtr)(CUpti_EventGroup);

// file handles used to access cuda libraries with dlopen
static void* dl1 = NULL;
static void* dl2 = NULL;
static void* dl3 = NULL;

static int linkCudaLibraries ();

papi_vector_t _cuda_vector;


/******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 *****************************************************************************/
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
	err = (*cuInitPtr)( 0 );
	if ( err != CUDA_SUCCESS ) {
		SUBDBG ("Info: Error from cuInit(): %d\n", err);
		return ( PAPI_ENOSUPP );
	}

	/* How many gpgpu devices do we have? */
	err = (*cuDeviceGetCountPtr)( &deviceCount );
	CHECK_CU_ERROR( err, "cuDeviceGetCount" );
	if ( deviceCount == 0 )
		return ( PAPI_ENOSUPP );

	/* allocate memory for device data table */
	device = ( DeviceData_t * ) malloc( sizeof ( DeviceData_t ) * deviceCount );
	if ( device == NULL ) {
		perror( "malloc(): Failed to allocate memory to CUDA device table" );
		return ( PAPI_ENOSUPP );
	}

	/* What are the devices? Get Name and # of domains per device */
	for ( id = 0; id < deviceCount; id++ ) {
		err = (*cuDeviceGetPtr)( &device[id].dev, id );
		CHECK_CU_ERROR( err, "cuDeviceGet" );

		err = (*cuDeviceGetNamePtr)( device[id].name, PAPI_MIN_STR_LEN, device[id].dev );
		CHECK_CU_ERROR( err, "cuDeviceGetName" );

		SUBDBG ("Cuda deviceName: %s\n", device[id].name);

		/* Skip device if there are multiple of the same type 
		   and if it has been already added to the list */
		if ( 0 == strcmp( deviceName_tmp, device[id].name ) ) {
			skipDevice++;
			continue;
		}

		strcpy( deviceName_tmp, device[id].name );

		/* enumerate the domains on the device */
		if ( 0 != enumEventDomains( device[id].dev, id ) )
			return ( PAPI_ENOSUPP );
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
	err = (*cuptiDeviceGetNumEventDomainsPtr)( dev, &device[deviceId].domainCount );
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
		free(domainId);
		return -1;
	}

	/* Enumerates the event domains for a device dev */
	err = (*cuptiDeviceEnumEventDomainsPtr)( dev, &size, domainId );
	CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );

	/* enum domains */
	for ( id = 0; id < device[deviceId].domainCount; id++ ) {
		device[deviceId].domain[id].domainId = domainId[id];

		/* query domain name */
		size = PAPI_MIN_STR_LEN;
#ifdef CUDA_4_0
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
#else
		err = (*cuptiDeviceGetEventDomainAttributePtr)( dev,
												  device[deviceId].domain[id].domainId,
												  CUPTI_EVENT_DOMAIN_ATTR_NAME, &size,
												  ( void * ) device[deviceId].domain[id].name );
		CHECK_CUPTI_ERROR( err, "cuptiDeviceGetEventDomainAttribute" );

		/* query num of events avaialble in the domain */
		err = (*cuptiEventDomainGetNumEventsPtr)( device[deviceId].domain[id].domainId,
										    &device[deviceId].domain[id].eventCount );
		CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetNumEvents" );

		/* enumerate the events for the domain[id] on the device deviceId */
		if ( 0 != enumEvents( deviceId, id ) )
			return -1;
#endif
	}

	totalDomainCount += device[deviceId].domainCount;
	free( domainId );
	return 0;
}


/*
 * Detect supported events for specified device domain
 */
#ifdef CUDA_4_0
static int
enumEvents( CUdevice dev, int deviceId, int domainId )
#else
static int
enumEvents( int deviceId, int domainId )
#endif
{
	CUptiResult err = CUPTI_SUCCESS;
	CUpti_EventID *eventId = NULL;
	size_t size = 0;
	uint32_t id = 0;

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
		free(eventId);
		return -1;
	}

	/* enumerate the events for the domain[domainId] on the device[deviceId] */
#ifdef CUDA_4_0
	err =
		(*cuptiEventDomainEnumEventsPtr)( dev,
							   ( CUpti_EventDomainID ) device[deviceId].
							   domain[domainId].domainId, &size, eventId );
#else
	err =
		(*cuptiEventDomainEnumEventsPtr)( ( CUpti_EventDomainID ) device[deviceId].
									domain[domainId].domainId, &size, eventId );
#endif
	CHECK_CUPTI_ERROR( err, "cuptiEventDomainEnumEvents" );

	/* query event info */
	for ( id = 0; id < device[deviceId].domain[domainId].eventCount; id++ ) {
		device[deviceId].domain[domainId].event[id].eventId = eventId[id];

		/* query event name */
		size = PAPI_MIN_STR_LEN;
#ifdef CUDA_4_0
		err = (*cuptiEventGetAttributePtr)( dev,
									 device[deviceId].domain[domainId].
									 event[id].eventId, CUPTI_EVENT_ATTR_NAME,
									 &size,
									 ( uint8_t * ) device[deviceId].
									 domain[domainId].event[id].name );		
#else
		err = (*cuptiEventGetAttributePtr)( device[deviceId].domain[domainId].
									  event[id].eventId, CUPTI_EVENT_ATTR_NAME,
									  &size,
									  ( uint8_t * ) device[deviceId].
									  domain[domainId].event[id].name );
#endif
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute" );

		/* query event description */
		size = PAPI_2MAX_STR_LEN;
#ifdef CUDA_4_0
		err = (*cuptiEventGetAttributePtr)( dev,
									 device[deviceId].domain[domainId].
									 event[id].eventId,
									 CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size,
									 ( uint8_t * ) device[deviceId].
									 domain[domainId].event[id].desc );		
#else
		err = (*cuptiEventGetAttributePtr)( device[deviceId].domain[domainId].
									  event[id].eventId,
									  CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size,
									  ( uint8_t * ) device[deviceId].
									  domain[domainId].event[id].desc );
#endif
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
	int deviceId, id = 0;
	uint32_t domainId, eventId;
	int cuptiDomainId;
	int i;
	int devNameLen;

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
						 "%s:%s:%s",
						 device[deviceId].name,
						 device[deviceId].domain[domainId].name,
						 device[deviceId].domain[domainId].event[eventId].
						 name );

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
getEventValue( long long *counts, CUpti_EventGroup eventGroup, AddedEvents_t addedEvents )
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
	cuptiErr = (*cuptiEventGroupReadAllEventsPtr)( eventGroup,
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
				// since cuptiEventGroupReadAllEvents() resets counter values to 0;
				// we have to accumulate ourselves 
				counts[addedEvents.list[j]] = counts[addedEvents.list[j]] + counterDataBuffer[i];

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
CUDA_init_thread( hwd_context_t * ctx )
{
	( void ) ctx;
	
	return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 *
 * NOTE: only called by main thread (not by every thread) !!!
 *
 * Starting in CUDA 4.0, multiple CPU threads can access the same CUDA context.
 * This is a much easier programming model then pre-4.0 as threads - using the 
 * same context - can share memory, data, etc. 
 * It's possible to create a different context for each thread, but then we are
 * likely running into a limitation that only one context can be profiled at a time.
 * ==> and we don't want this. That's why CUDA context creation is done in 
 * CUDA_init_component() (called only by main thread) rather than CUDA_init() 
 * or CUDA_init_control_state() (both called by each thread).
 */
int
CUDA_init_component( int cidx )
{
	SUBDBG ("Entry: cidx: %d\n", cidx);
	CUresult cuErr = CUDA_SUCCESS;

	/* link in all the cuda libraries and resolve the symbols we need to use */
	if (linkCudaLibraries() != PAPI_OK) {
		SUBDBG ("Dynamic link of CUDA libraries failed, component will be disabled.\n");
		SUBDBG ("See disable reason in papi_component_avail output for more details.\n");
		return (PAPI_ENOSUPP);
	}

	/* Create dynamic event table */
	NUM_EVENTS = detectDevice(  );
	if (NUM_EVENTS < 0) {
		strncpy(_cuda_vector.cmp_info.disabled_reason, "Call to detectDevice failed.",PAPI_MAX_STR_LEN);
		return (PAPI_ENOSUPP);
	}
	/* TODO: works only for one device right now;
	 need to find out if user can use 2 or more devices at same time */

	/* want create a CUDA context for either the default device or
	 the device specified with cudaSetDevice() in user code */
	if ( CUDA_SUCCESS != (*cudaGetDevicePtr)( &currentDeviceID ) ) {
		strncpy(_cuda_vector.cmp_info.disabled_reason, "No NVIDIA GPU's found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	
	if ( getenv( "PAPI_VERBOSE" ) ) {
		printf( "DEVICE USED: %s (%d)\n", device[currentDeviceID].name,
			   currentDeviceID );
	}
	
	/* get the CUDA context from the calling CPU thread */
	cuErr = (*cuCtxGetCurrentPtr)( &cuCtx );

	/* if no CUDA context is bound to the calling CPU thread yet, create one */
	if ( cuErr != CUDA_SUCCESS || cuCtx == NULL ) {
		cuErr = (*cuCtxCreatePtr)( &cuCtx, 0, device[currentDeviceID].dev );
		CHECK_CU_ERROR( cuErr, "cuCtxCreate" );
	}

	/* cuCtxGetCurrent() can return a non-null context that is not valid 
	   because the context has not yet been initialized.
	   Here is a workaround: 
	   cudaFree(NULL) forces the context to be initialized
	   if cudaFree(NULL) returns success then we are able to use the context in subsequent calls
	   if cudaFree(NULL) returns an error (or subsequent cupti* calls) then the context is not usable,
	   and will never be useable */
	if ( CUDA_SUCCESS != (*cudaFreePtr)( NULL ) ) {
		strncpy(_cuda_vector.cmp_info.disabled_reason, "Problem initializing CUDA context.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	/* Create dynamic event table */
	cuda_native_table = ( CUDA_native_event_entry_t * )
		malloc( sizeof ( CUDA_native_event_entry_t ) * NUM_EVENTS );
	if ( cuda_native_table == NULL ) {
		perror( "malloc(): Failed to allocate memory to events table" );
		strncpy(_cuda_vector.cmp_info.disabled_reason, "Failed to allocate memory to events table.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	if ( NUM_EVENTS != createNativeEvents(  ) ) {
		strncpy(_cuda_vector.cmp_info.disabled_reason, "Error creating CUDA event list.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	
	/* Export the component id */
	_cuda_vector.cmp_info.CmpIdx = cidx;

	/* Number of events */
	_cuda_vector.cmp_info.num_native_events = NUM_EVENTS;

	return ( PAPI_OK );
}


/*
 * Link the necessary CUDA libraries to use the cuda component.  If any of them can not be found, then
 * the CUDA component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the CUDA component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */
static int 
linkCudaLibraries ()
{
		/* Attempt to guess if we were statically linked to libc, if so bail */
		if ( _dl_non_dynamic_init != NULL ) {
				strncpy(_cuda_vector.cmp_info.disabled_reason, "The cuda component does not support statically linking to libc.",PAPI_MAX_STR_LEN);
				return PAPI_ENOSUPP;
		}
	/* Need to link in the cuda libraries, if not found disable the component */
	dl1 = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dl1)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA library libcuda.so not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuCtxCreatePtr = dlsym(dl1, "cuCtxCreate_v2");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuCtxCreate not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuCtxDestroyPtr = dlsym(dl1, "cuCtxDestroy_v2");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuCtxDestroy not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuCtxGetCurrentPtr = dlsym(dl1, "cuCtxGetCurrent");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuCtxGetCurrent not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuDeviceGetPtr = dlsym(dl1, "cuDeviceGet");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuDeviceGet not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuDeviceGetCountPtr = dlsym(dl1, "cuDeviceGetCount");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuDeviceGetCount not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuDeviceGetNamePtr = dlsym(dl1, "cuDeviceGetName");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuDeviceGetName not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuInitPtr = dlsym(dl1, "cuInit");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA function cuInit not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dl2)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA runtime library libcudart.so not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cudaFreePtr = dlsym(dl2, "cudaFree");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDART function cudaFree not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cudaGetDevicePtr = dlsym(dl2, "cudaGetDevice");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDART function cudaGetDevice not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cudaRuntimeGetVersionPtr = dlsym(dl2, "cudaRuntimeGetVersion");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDART function cudaRuntimeGetVersion not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cudaDriverGetVersionPtr = dlsym(dl2, "cudaDriverGetVersion");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDART function cudaDriverGetVersion not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	dl3 = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
	if (!dl3)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUDA runtime library libcupti.so not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiDeviceEnumEventDomainsPtr = dlsym(dl3, "cuptiDeviceEnumEventDomains");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiDeviceEnumEventDomains not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiDeviceGetEventDomainAttributePtr = dlsym(dl3, "cuptiDeviceGetEventDomainAttribute");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiDeviceGetEventDomainAttribute not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiDeviceGetNumEventDomainsPtr = dlsym(dl3, "cuptiDeviceGetNumEventDomains");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiDeviceGetNumEventDomains not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventDomainEnumEventsPtr = dlsym(dl3, "cuptiEventDomainEnumEvents");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventDomainEnumEvents not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventDomainGetNumEventsPtr = dlsym(dl3, "cuptiEventDomainGetNumEvents");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventDomainGetNumEvents not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGetAttributePtr = dlsym(dl3, "cuptiEventGetAttribute");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGetAttribute not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupAddEventPtr = dlsym(dl3, "cuptiEventGroupAddEvent");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupAddEvent not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupCreatePtr = dlsym(dl3, "cuptiEventGroupCreate");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupCreate not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupDestroyPtr = dlsym(dl3, "cuptiEventGroupDestroy");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupDestroy not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupDisablePtr = dlsym(dl3, "cuptiEventGroupDisable");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupDisable not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupEnablePtr = dlsym(dl3, "cuptiEventGroupEnable");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupEnable not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupReadAllEventsPtr = dlsym(dl3, "cuptiEventGroupReadAllEvents");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupReadAllEvents not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
    cuptiEventGroupRemoveAllEventsPtr = dlsym(dl3, "cuptiEventGroupRemoveAllEvents");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupRemoveAllEvents not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	cuptiEventGroupResetAllEventsPtr = dlsym(dl3, "cuptiEventGroupResetAllEvents");
	if (dlerror() != NULL)
	{
		strncpy(_cuda_vector.cmp_info.disabled_reason, "CUPTI function cuptiEventGroupResetAllEvents not found.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	return ( PAPI_OK );
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
CUDA_init_control_state( hwd_control_state_t * ctrl )
{
	CUDA_control_state_t * CUDA_ctrl = ( CUDA_control_state_t * ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	int i;

	/* allocate memory for the list of events that are added to the CuPTI eventGroup */
	CUDA_ctrl->addedEvents.list = malloc( sizeof ( int ) * NUM_EVENTS );
	if ( CUDA_ctrl->addedEvents.list == NULL ) {
		perror
		( "malloc(): Failed to allocate memory to table of events that are added to CuPTI eventGroup" );
		return ( PAPI_ENOSUPP );
	}
	
	/* initialize the event list */
	for ( i = 0; i < NUM_EVENTS; i++ )
		CUDA_ctrl->addedEvents.list[i] = 0;

	
	
	cuptiErr = (*cuptiEventGroupCreatePtr)( cuCtx, &CUDA_ctrl->eventGroup, 0 );
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
	int i;
	CUDA_control_state_t * CUDA_ctrl = ( CUDA_control_state_t * ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;
	
	// reset all event values to 0
	for ( i = 0; i < NUM_EVENTS; i++ )
		CUDA_ctrl->counts[i] = 0;

	cuptiErr = (*cuptiEventGroupEnablePtr)( CUDA_ctrl->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupEnable" );

	/* Resets all events in the CuPTI eventGroup to zero */
	cuptiErr = (*cuptiEventGroupResetAllEventsPtr)( CUDA_ctrl->eventGroup );
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
	CUDA_control_state_t * CUDA_ctrl = ( CUDA_control_state_t * ) ctrl;


	if ( 0 != getEventValue( CUDA_ctrl->counts, CUDA_ctrl->eventGroup, CUDA_ctrl->addedEvents ) )
		return ( PAPI_ENOSUPP );

	*events = CUDA_ctrl->counts;

	return ( PAPI_OK );
}

/* 
 *
 */
int
CUDA_shutdown_thread( hwd_context_t *ctx )
{
	CUDA_context_t *CUDA_ctx = (CUDA_context_t*)ctx;
	free( CUDA_ctx->state.addedEvents.list );
	return (PAPI_OK);
}

/*
 *
 */
int
CUDA_shutdown_component( void )
{
	CUresult cuErr = CUDA_SUCCESS;
	
	/* if running a threaded application, we need to make sure that 
	   a thread doesn't free the same memory location(s) more than once */
	if ( CUDA_FREED == 0 ) {
		uint32_t j;
		int i;
		
		CUDA_FREED = 1;

		/* deallocate all the memory */
		for ( i = 0; i < deviceCount; i++ ) {
			for ( j = 0; j < device[i].domainCount; j++ )
				free( device[i].domain[j].event );
			
			free( device[i].domain );
		}

		free( device );
		free( cuda_native_table );
		
		/* destroy floating CUDA context */
		cuErr = (*cuCtxDestroyPtr)( cuCtx );
		if ( cuErr != CUDA_SUCCESS )
			return ( PAPI_ENOSUPP );			// Not supported
	}

	// close the dynamic libraries needed by this component (opened in the init substrate call)
	dlclose(dl1);
	dlclose(dl2);
	dlclose(dl3);

	return ( PAPI_OK );
}


/* This function sets various options in the component
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
	( void ) ctx;
	CUDA_control_state_t * CUDA_ptr = ( CUDA_control_state_t * ) ptr;
	int index, i;
	CUptiResult cuptiErr = CUPTI_SUCCESS;

    /* Disable the CUDA eventGroup;
     it also frees the perfmon hardware on the GPU */
	cuptiErr = (*cuptiEventGroupDisablePtr)( CUDA_ptr->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDisable" );

    cuptiErr = (*cuptiEventGroupRemoveAllEventsPtr)( CUDA_ptr->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupRemoveAllEvents" );
    
    // otherwise, add the events to the eventset
	for ( i = 0; i < count; i++ ) {
       
		index = native[i].ni_event;
		native[i].ni_position = index;

		/* store events, that have been added to the CuPTI eveentGroup 
		   in a seperate place (addedEvents).
		   Needed, so that we can read the values for the added events only */
		CUDA_ptr->addedEvents.count = count;
		CUDA_ptr->addedEvents.list[i] = index;

		/* if this device name is different from the actual device the code is running on, then exit */
		if ( 0 != strncmp( device[currentDeviceID].name,
						   cuda_native_table[index].name,
						   strlen( device[currentDeviceID].name ) ) ) {
			fprintf( stderr, "Device %s is used -- BUT event %s is collected. \n ---> ERROR: Specify events for the device that is used!\n\n",
				  device[currentDeviceID].name, cuda_native_table[index].name );
			
			return ( PAPI_ENOSUPP );	// Not supported 
		}

		/* Add events to the CuPTI eventGroup */
		cuptiErr =
			(*cuptiEventGroupAddEventPtr)( CUDA_ptr->eventGroup,
									 cuda_native_table[index].resources.
									 eventId );
		CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupAddEvent" );
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
	CUDA_control_state_t * CUDA_ctrl = ( CUDA_control_state_t * ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;

	/* Resets all events in the CuPTI eventGroup to zero */
	cuptiErr = (*cuptiEventGroupResetAllEventsPtr)( CUDA_ctrl->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupResetAllEvents" );

	return ( PAPI_OK );
}


/*
 * Disable and Destoy the CUDA eventGroup */
int
CUDA_cleanup_eventset( hwd_control_state_t * ctrl )
{
    ( void ) ctrl;
    
    // TODO: after cleanup_eventset() which destroys the eventset, update_control_state()
    // is called, which operates on the already destroyed eventset. Bad!
#if 0
	CUDA_control_state_t * CUDA_ctrl = ( CUDA_control_state_t * ) ctrl;
	CUptiResult cuptiErr = CUPTI_SUCCESS;

	/* Disable the CUDA eventGroup;
	   it also frees the perfmon hardware on the GPU */
	cuptiErr = (*cuptiEventGroupDisablePtr)( CUDA_ctrl->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDisable" );

	/* Call the CuPTI cleaning function before leaving */
	cuptiErr = (*cuptiEventGroupDestroyPtr)( CUDA_ctrl->eventGroup );
	CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDestroy" );
#endif
	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
CUDA_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = 0;

		return ( PAPI_OK );
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = *EventCode;

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
	int index = EventCode;

	strncpy( name, cuda_native_table[index].name, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode;

	strncpy( name, cuda_native_table[index].description, len );
	return ( PAPI_OK );
}


/*
 *
 */
int
CUDA_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode;

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
				 .name = "cuda",
				 .short_name = "cuda",
				 .version = "5.0",
				 .description = "CuPTI provides the API for monitoring NVIDIA GPU hardware events",
				 .num_mpx_cntrs = CUDA_MAX_COUNTERS,
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
	.init_thread = CUDA_init_thread,
	.init_component = CUDA_init_component,
	.init_control_state = CUDA_init_control_state,
	.start = CUDA_start,
	.stop = CUDA_stop,
	.read = CUDA_read,
	.shutdown_component = CUDA_shutdown_component,
	.shutdown_thread = CUDA_shutdown_thread,
	.cleanup_eventset = CUDA_cleanup_eventset,
	.ctl = CUDA_ctl,
	.update_control_state = CUDA_update_control_state,
	.set_domain = CUDA_set_domain,
	.reset = CUDA_reset,

	.ntv_enum_events = CUDA_ntv_enum_events,
	.ntv_code_to_name = CUDA_ntv_code_to_name,
	.ntv_code_to_descr = CUDA_ntv_code_to_descr,
	.ntv_code_to_bits = CUDA_ntv_code_to_bits,
};
