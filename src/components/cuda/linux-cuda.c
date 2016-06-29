/****************************/
/* THIS IS OPEN SOURE CODE */
/****************************/

/**
 * @file    linux-cuda.c
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (in collaboration with Robert Dietrich, TU Dresden) jagode@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief This implements a PAPI component that enables PAPI-C to
 *  access hardware monitoring counters for NVIDIA CUDA GPU devices
 *  through the CUPTI library.
 */

#include <dlfcn.h>
#include <cupti.h>
#include <cuda_runtime_api.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"

/* this number assumes that there will never be more events than indicated */
#define PAPICUDA_MAX_COUNTERS 512

/* Contains device list, pointer to device desciption, and the list of available events */
typedef struct papicuda_context {
    int deviceCount;
    struct papicuda_device_desc *deviceArray;
    uint32_t availEventSize;
    CUpti_EventID* availEventIDArray;
    int* availEventDeviceNum;
    struct papicuda_name_desc* availEventDesc;
} papicuda_context_t;

/* Store the name and description for an event */
typedef struct papicuda_name_desc  {
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_2MAX_STR_LEN];
} papicuda_name_desc_t;

/* For a device, store device description */
typedef struct papicuda_device_desc {
    CUdevice cuDev;             /* CUDA device */
    unsigned int deviceNum;
    char deviceName[PAPI_MIN_STR_LEN];
    uint32_t maxDomains;        /* number of domains per device */
    CUpti_EventDomainID *domainIDArray; /* Array[maxDomains] of domain IDs */
    uint32_t *domainIDNumEvents; /* Array[maxDomains] of num of events in that domain */
} papicuda_device_desc_t;

/* Control structure tracks array of active contexts, records active events and their values */
typedef struct papicuda_control {
    int countOfActiveCUContexts;
    struct papicuda_active_cucontext_s *arrayOfActiveCUContexts[PAPICUDA_MAX_COUNTERS];
    int activeEventCount;
    int activeEventIndex[PAPICUDA_MAX_COUNTERS];
    int activeEventContextIdx[PAPICUDA_MAX_COUNTERS];
    long long activeEventValues[PAPICUDA_MAX_COUNTERS];
} papicuda_control_t;

/* For each active context, which CUDA events are being measured, context eventgroups containing events */
typedef struct papicuda_active_cucontext_s {
    CUcontext context;
    int deviceNum;
    int numEventGroups;
    CUpti_EventGroup eventGroup[PAPICUDA_MAX_COUNTERS];
} papicuda_active_cucontext_t;

// file handles used to access cuda libraries with dlopen
static void *dl1 = NULL;
static void *dl2 = NULL;
static void *dl3 = NULL;

/* The PAPI side (external) variable as a global */
papi_vector_t _cuda_vector;

/* Global variable for hardware description, event and metric lists */
static papicuda_context_t *global_papicuda_context = NULL;

/* This global variable points to the head of the control state list */
static papicuda_control_t *global_papicuda_control = NULL;

/* Macros for error checking... each arg is only referenced/evaluated once */
#define CHECK_CU_ERROR(err, cufunc)                                     \
    if( (err) != CUDA_SUCCESS ) { PAPIERROR( "CUDA Driver API function failed '%s'", cufunc ); return -1; }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
    if( (err) != CUPTI_SUCCESS ) { PAPIERROR( "CUPTI API function failed '%s'", cuptifunc ); return -1; }

#define CHECK_PRINT_EVAL( err, str, eval )      \
    if( (err) ) { PAPIERROR( "%s", str ); eval; }

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
void ( *_dl_non_dynamic_init )( void ) __attribute__( ( weak ) );
#undef CUDAAPI
#define CUDAAPI __attribute__((weak))
CUresult CUDAAPI cuCtxGetCurrent( CUcontext * );
CUresult CUDAAPI cuDeviceGet( CUdevice *, int );
CUresult CUDAAPI cuDeviceGetCount( int * );
CUresult CUDAAPI cuDeviceGetName( char *, int, CUdevice );
CUresult CUDAAPI cuDeviceGetName( char *, int, CUdevice );
CUresult CUDAAPI cuInit( unsigned int );
CUresult CUDAAPI cuCtxPopCurrent( CUcontext * pctx );
CUresult CUDAAPI cuCtxPushCurrent( CUcontext pctx );

CUresult( *cuCtxCreatePtr )( CUcontext * pctx, unsigned int flags, CUdevice dev );
CUresult( *cuCtxDestroyPtr )( CUcontext );
CUresult( *cuCtxGetCurrentPtr )( CUcontext * );
CUresult( *cuDeviceGetPtr )( CUdevice *, int );
CUresult( *cuDeviceGetCountPtr )( int * );
CUresult( *cuDeviceGetNamePtr )( char *, int, CUdevice );
CUresult( *cuInitPtr )( unsigned int );
CUresult( *cuCtxPopCurrentPtr )( CUcontext * pctx );
CUresult( *cuCtxPushCurrentPtr )( CUcontext pctx );

#undef CUDARTAPI
#define CUDARTAPI __attribute__((weak))
cudaError_t CUDARTAPI cudaGetDevice( int * );
cudaError_t CUDARTAPI cudaSetDevice( int );
cudaError_t CUDARTAPI cudaFree( void * );

cudaError_t ( *cudaGetDevicePtr )( int * );
cudaError_t ( *cudaSetDevicePtr )( int );
cudaError_t (*cudaFreePtr)(void *);

#undef CUPTIAPI
#define CUPTIAPI __attribute__((weak))
CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains( CUdevice, size_t *, CUpti_EventDomainID * );
CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains( CUdevice, uint32_t * );
CUptiResult CUPTIAPI cuptiEventDomainEnumEvents( CUpti_EventDomainID, size_t *, CUpti_EventID * );
CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents( CUpti_EventDomainID, uint32_t * );
CUptiResult CUPTIAPI cuptiEventDomainGetAttribute ( CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t* valueSize, void* value );
CUptiResult CUPTIAPI cuptiEventGroupAddEvent( CUpti_EventGroup, CUpti_EventID );
CUptiResult CUPTIAPI cuptiEventGroupCreate( CUcontext, CUpti_EventGroup *, uint32_t );
CUptiResult CUPTIAPI cuptiEventGroupDestroy( CUpti_EventGroup );
CUptiResult CUPTIAPI cuptiEventGroupDisable( CUpti_EventGroup );
CUptiResult CUPTIAPI cuptiEventGroupEnable( CUpti_EventGroup );
CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents( CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *, size_t * );
CUptiResult CUPTIAPI cuptiEventGroupResetAllEvents( CUpti_EventGroup );
CUptiResult CUPTIAPI cuptiEventGetAttribute( CUpti_EventID, CUpti_EventAttribute, size_t *, void * );

CUptiResult( *cuptiDeviceEnumEventDomainsPtr )( CUdevice, size_t *, CUpti_EventDomainID * );
CUptiResult( *cuptiDeviceGetNumEventDomainsPtr )( CUdevice, uint32_t * );
CUptiResult( *cuptiEventDomainEnumEventsPtr )( CUpti_EventDomainID, size_t *, CUpti_EventID * );
CUptiResult( *cuptiEventDomainGetNumEventsPtr )( CUpti_EventDomainID, uint32_t * );
CUptiResult( *cuptiEventDomainGetAttributePtr ) ( CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t* valueSize, void* value );
CUptiResult( *cuptiEventGroupAddEventPtr )( CUpti_EventGroup, CUpti_EventID );
CUptiResult( *cuptiEventGroupCreatePtr )( CUcontext, CUpti_EventGroup *, uint32_t );
CUptiResult( *cuptiEventGroupDestroyPtr )( CUpti_EventGroup );
CUptiResult( *cuptiEventGroupDisablePtr )( CUpti_EventGroup );
CUptiResult( *cuptiEventGroupEnablePtr )( CUpti_EventGroup );
CUptiResult( *cuptiEventGroupReadAllEventsPtr )( CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *, size_t * );
CUptiResult( *cuptiEventGroupResetAllEventsPtr )( CUpti_EventGroup );
CUptiResult( *cuptiEventGetAttributePtr )( CUpti_EventID, CUpti_EventAttribute, size_t *, void * );

/******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT *********
 *****************************************************************************/

/*
 * Link the necessary CUDA libraries to use the cuda component.  If any of them can not be found, then
 * the CUDA component will just be disabled.  This is done at runtime so that a version of PAPI built
 * with the CUDA component can be installed and used on systems which have the CUDA libraries installed
 * and on systems where these libraries are not installed.
 */
#define CHECK_DL_STATUS( err, str ) if( err ) { strncpy( _cuda_vector.cmp_info.disabled_reason, str, PAPI_MAX_STR_LEN ); return ( PAPI_ENOSUPP ); }

static int papicuda_linkCudaLibraries()
{
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if( _dl_non_dynamic_init != NULL ) {
        strncpy( _cuda_vector.cmp_info.disabled_reason, "The cuda component does not support statically linking to libc.", PAPI_MAX_STR_LEN );
        return PAPI_ENOSUPP;
    }
    /* Need to link in the cuda libraries, if not found disable the component */
    dl1 = dlopen( "libcuda.so", RTLD_NOW | RTLD_GLOBAL );
    CHECK_DL_STATUS( !dl1 , "CUDA library libcuda.so not found." );
    cuCtxGetCurrentPtr = dlsym( dl1, "cuCtxGetCurrent" );
    CHECK_DL_STATUS( dlerror()!=NULL , "CUDA function cuCtxGetCurrent not found." );
    cuDeviceGetPtr = dlsym( dl1, "cuDeviceGet" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuDeviceGet not found." );
    cuDeviceGetCountPtr = dlsym( dl1, "cuDeviceGetCount" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuDeviceGetCount not found." );
    cuDeviceGetNamePtr = dlsym( dl1, "cuDeviceGetName" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuDeviceGetName not found." );
    cuInitPtr = dlsym( dl1, "cuInit" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuInit not found." );
    cuCtxPopCurrentPtr = dlsym( dl1, "cuCtxPopCurrent" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuCtxPopCurrent not found." );
    cuCtxPushCurrentPtr = dlsym( dl1, "cuCtxPushCurrent" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDA function cuCtxPushCurrent not found." );

    dl2 = dlopen( "libcudart.so", RTLD_NOW | RTLD_GLOBAL );
    CHECK_DL_STATUS( !dl2, "CUDA runtime library libcudart.so not found." );
    cudaGetDevicePtr = dlsym( dl2, "cudaGetDevice" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDART function cudaGetDevice not found." );
    cudaSetDevicePtr = dlsym( dl2, "cudaSetDevice" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDART function cudaSetDevice not found." );
    cudaFreePtr = dlsym( dl2, "cudaFree" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUDART function cudaFree not found." );

    dl3 = dlopen( "libcupti.so", RTLD_NOW | RTLD_GLOBAL );
    CHECK_DL_STATUS( !dl3, "CUDA runtime library libcupti.so not found." );
    cuptiDeviceEnumEventDomainsPtr = dlsym( dl3, "cuptiDeviceEnumEventDomains" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiDeviceEnumEventDomains not found." );
    cuptiDeviceGetNumEventDomainsPtr = dlsym( dl3, "cuptiDeviceGetNumEventDomains" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiDeviceGetNumEventDomains not found." );
    cuptiEventDomainEnumEventsPtr = dlsym( dl3, "cuptiEventDomainEnumEvents" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventDomainEnumEvents not found." );
    cuptiEventDomainGetNumEventsPtr = dlsym( dl3, "cuptiEventDomainGetNumEvents" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventDomainGetNumEvents not found." );
    cuptiEventGetAttributePtr = dlsym( dl3, "cuptiEventGetAttribute" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGetAttribute not found." );
    cuptiEventGroupAddEventPtr = dlsym( dl3, "cuptiEventGroupAddEvent" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupAddEvent not found." );
    cuptiEventGroupCreatePtr = dlsym( dl3, "cuptiEventGroupCreate" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupCreate not found." );
    cuptiEventGroupDestroyPtr = dlsym( dl3, "cuptiEventGroupDestroy" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupDestroy not found." );
    cuptiEventGroupDisablePtr = dlsym( dl3, "cuptiEventGroupDisable" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupDisable not found." );
    cuptiEventGroupEnablePtr = dlsym( dl3, "cuptiEventGroupEnable" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupEnable not found." );
    cuptiEventGroupReadAllEventsPtr = dlsym( dl3, "cuptiEventGroupReadAllEvents" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupReadAllEvents not found." );
    cuptiEventGroupResetAllEventsPtr = dlsym( dl3, "cuptiEventGroupResetAllEvents" );
    CHECK_DL_STATUS( dlerror()!=NULL, "CUPTI function cuptiEventGroupResetAllEvents not found." );
    return ( PAPI_OK );
}

/* Called during component initialization to get a list of all available events */
static int papicuda_list_all_events( papicuda_context_t *gctxt )
{
    SUBDBG( "Entering\n" );
    CUptiResult cuptiErr;
    CUresult cuErr;
    unsigned int deviceNum;
    uint32_t domainNum, eventNum;
    papicuda_device_desc_t *mydevice;
    char tmpStr[PAPI_MIN_STR_LEN];
    tmpStr[PAPI_MIN_STR_LEN-1]='\0';
    size_t tmpSizeBytes;
    int ii;

    /* How many gpgpu devices do we have? */
    cuErr = ( *cuDeviceGetCountPtr )( &gctxt->deviceCount );
    if ( cuErr==CUDA_ERROR_NOT_INITIALIZED ) {
        /* If CUDA not initilaized, initialized CUDA and retry the device list */
        /* This is required for some of the PAPI tools, that do not call the init functions */
        if ( (( *cuInitPtr )( 0 )) != CUDA_SUCCESS ) {
            strncpy( _cuda_vector.cmp_info.disabled_reason, "CUDA cannot be found and initialized (cuInit failed).", PAPI_MAX_STR_LEN );
            return PAPI_ENOSUPP;
        }
        cuErr = ( *cuDeviceGetCountPtr )( &gctxt->deviceCount );
    }
    CHECK_CU_ERROR( cuErr, "cuDeviceGetCount" );
    if ( gctxt->deviceCount==0 ) {
        strncpy( _cuda_vector.cmp_info.disabled_reason, "CUDA initialized but no CUDA devices found.", PAPI_MAX_STR_LEN );
        return PAPI_ENOSUPP;
    }
    SUBDBG( "Found %d devices\n", gctxt->deviceCount );

    /* allocate memory for device information */
    gctxt->deviceArray = ( papicuda_device_desc_t * ) papi_calloc( gctxt->deviceCount, sizeof( papicuda_device_desc_t ) );
    CHECK_PRINT_EVAL( !gctxt->deviceArray, "ERROR CUDA: Could not allocate memory for CUDA device structure", return( PAPI_ENOSUPP ) );

    /* For each device, get domains and domain-events counts */
    gctxt->availEventSize = 0;
    for( deviceNum = 0; deviceNum < ( uint )gctxt->deviceCount; deviceNum++ ) {
        mydevice = &gctxt->deviceArray[deviceNum];
        /* Get device id for each device */
        CHECK_CU_ERROR( ( *cuDeviceGetPtr )( &mydevice->cuDev, deviceNum ), "cuDeviceGet" );
        /* Get device name */
        CHECK_CU_ERROR( ( *cuDeviceGetNamePtr )( mydevice->deviceName, PAPI_MIN_STR_LEN-1, mydevice->cuDev ), "cuDeviceGetName" );
        mydevice->deviceName[PAPI_MIN_STR_LEN-1]='\0';
        /* Get max num domains for each device */
        CHECK_CUPTI_ERROR( ( *cuptiDeviceGetNumEventDomainsPtr )( mydevice->cuDev, &mydevice->maxDomains ), "cuptiDeviceGetNumEventDomains" );
        /* Allocate space to hold domain IDs */
        mydevice->domainIDArray = ( CUpti_EventDomainID * ) papi_calloc( mydevice->maxDomains, sizeof( CUpti_EventDomainID ) );
        CHECK_PRINT_EVAL( !mydevice->domainIDArray, "ERROR CUDA: Could not allocate memory for CUDA device domains", return( PAPI_ENOMEM ) );
        /* Put domain ids into allocated space */
        size_t domainarraysize = mydevice->maxDomains * sizeof( CUpti_EventDomainID );
        CHECK_CUPTI_ERROR( ( *cuptiDeviceEnumEventDomainsPtr )( mydevice->cuDev, &domainarraysize, mydevice->domainIDArray ), "cuptiDeviceEnumEventDomains" );
        /* Allocate space to hold domain event counts  */
        mydevice->domainIDNumEvents = ( uint32_t * ) papi_calloc( mydevice->maxDomains, sizeof( uint32_t ) );
        CHECK_PRINT_EVAL( !mydevice->domainIDNumEvents, "ERROR CUDA: Could not allocate memory for domain event counts", return( PAPI_ENOMEM ) );
        /* For each domain, get event counts in domainNumEvents[]  */
        for ( domainNum=0; domainNum < mydevice->maxDomains; domainNum++ ) {
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];
            /* Get num events in domain */
            //SUBDBG( "Device %d:%d calling cuptiEventDomainGetNumEventsPtr with domainID %d \n", deviceNum, mydevice->cuDev, domainID );
            CHECK_CUPTI_ERROR(  ( *cuptiEventDomainGetNumEventsPtr ) ( domainID, &mydevice->domainIDNumEvents[domainNum] ), "cuptiEventDomainGetNumEvents" );
            /* Keep track of overall number of events */
            gctxt->availEventSize += mydevice->domainIDNumEvents[domainNum];
        }
    }

    /* Allocate space for all events and descriptors */
    gctxt->availEventIDArray = ( CUpti_EventID * ) papi_calloc( gctxt->availEventSize, sizeof( CUpti_EventID ) );
    CHECK_PRINT_EVAL( !gctxt->availEventIDArray, "ERROR CUDA: Could not allocate memory for events", return( PAPI_ENOMEM ) );
    gctxt->availEventDeviceNum = ( int * ) papi_calloc( gctxt->availEventSize, sizeof( int ) );
    CHECK_PRINT_EVAL( !gctxt->availEventDeviceNum, "ERROR CUDA: Could not allocate memory", return( PAPI_ENOMEM ) );
    gctxt->availEventDesc = ( papicuda_name_desc_t * ) papi_calloc( gctxt->availEventSize, sizeof( papicuda_name_desc_t ) );
    CHECK_PRINT_EVAL( !gctxt->availEventDesc, "ERROR CUDA: Could not allocate memory for events", return( PAPI_ENOMEM ) );
    /* Record the events and descriptions */
    int idxEventArray = 0;
    for( deviceNum = 0; deviceNum < ( uint )gctxt->deviceCount; deviceNum++ ) {
        mydevice = &gctxt->deviceArray[deviceNum];
        //SUBDBG( "For device %d %d maxdomains %d \n", deviceNum, mydevice->cuDev, mydevice->maxDomains );
        /* Get and store event IDs, names, descriptions into the large arrays allocated */
        for ( domainNum=0; domainNum < mydevice->maxDomains; domainNum++ ) {
            /* Get domain id */
            CUpti_EventDomainID domainID = mydevice->domainIDArray[domainNum];
            uint32_t domainNumEvents = mydevice->domainIDNumEvents[domainNum];
            SUBDBG( "For device %d domain %d %d numEvents %d\n", mydevice->cuDev, domainNum, domainID, domainNumEvents );
            /* Allocate temp space for eventIDs for this domain */
            CUpti_EventID *domainEventIDArray = ( CUpti_EventID * ) papi_calloc( domainNumEvents, sizeof( CUpti_EventID ) );
            CHECK_PRINT_EVAL( !domainEventIDArray, "ERROR CUDA: Could not allocate memory for events", return( PAPI_ENOMEM ) );
            /* Load the domain eventIDs in temp space */
            size_t domainEventArraySize = domainNumEvents * sizeof( CUpti_EventID );
            cuptiErr = ( *cuptiEventDomainEnumEventsPtr )  ( domainID, &domainEventArraySize, domainEventIDArray );
            CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventDomainEnumEvents" );
            /* For each event, get and store name and description */
            for ( eventNum=0; eventNum<domainNumEvents; eventNum++ ) {
                /* Record the event IDs in native event array */
                CUpti_EventID myeventID = domainEventIDArray[eventNum];
                gctxt->availEventIDArray[idxEventArray] = myeventID;
                gctxt->availEventDeviceNum[idxEventArray] = deviceNum;
                /* Get event name */
                tmpSizeBytes = PAPI_MIN_STR_LEN-1 * sizeof( char );
                cuptiErr = ( *cuptiEventGetAttributePtr ) ( myeventID, CUPTI_EVENT_ATTR_NAME, &tmpSizeBytes, tmpStr ) ;
                CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGetAttribute" );
                /* Save a full path for the event, filling spaces with underscores */
                //snprintf( gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN, "%s:%d:%s", mydevice->deviceName, deviceNum, tmpStr );
                snprintf( gctxt->availEventDesc[idxEventArray].name, PAPI_MIN_STR_LEN, "device:%d:%s", deviceNum, tmpStr );
                gctxt->availEventDesc[idxEventArray].name[PAPI_MIN_STR_LEN-1] = '\0';
                char *nameTmpPtr = gctxt->availEventDesc[idxEventArray].name;
                for ( ii = 0; ii < ( int )strlen( nameTmpPtr ); ii++ ) if ( nameTmpPtr[ii] == ' ' ) nameTmpPtr[ii] = '_';
                /* Save description in the native event array */
                tmpSizeBytes = PAPI_2MAX_STR_LEN-1 * sizeof( char );
                cuptiErr = ( *cuptiEventGetAttributePtr ) ( myeventID, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &tmpSizeBytes, gctxt->availEventDesc[idxEventArray].description );
                CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGetAttribute" );
                gctxt->availEventDesc[idxEventArray].description[PAPI_2MAX_STR_LEN-1] = '\0';
                // SUBDBG( "Event ID:%d Name:%s Desc:%s\n", gctxt->availEventIDArray[idxEventArray], gctxt->availEventDesc[idxEventArray].name, gctxt->availEventDesc[idxEventArray].description );
                /* Increment index past events in this domain to start of next domain */
                idxEventArray++;
            }
            papi_free ( domainEventIDArray );
        }
    }
    /* return 0 if everything went OK */
    return 0;
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized.
 */
static int papicuda_init_thread( hwd_context_t * ctx )
{
    ( void ) ctx;
    SUBDBG( "Entering\n" );

    return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
/* NOTE: only called by main thread (not by every thread) !!! Starting
   in CUDA 4.0, multiple CPU threads can access the same CUDA
   context. This is a much easier programming model then pre-4.0 as
   threads - using the same context - can share memory, data,
   etc. It's possible to create a different context for each thread.
   That's why CUDA context creation is done in CUDA_init_component()
   (called only by main thread) rather than CUDA_init() or
   CUDA_init_control_state() (both called by each thread). */
static int papicuda_init_component( int cidx )
{
    SUBDBG( "Entering with cidx: %d\n", cidx );
    int err;

    /* link in all the cuda libraries and resolve the symbols we need to use */
    if( papicuda_linkCudaLibraries() != PAPI_OK ) {
        SUBDBG ("Dynamic link of CUDA libraries failed, component will be disabled.\n");
        SUBDBG ("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    /* Create the structure */
    if ( !global_papicuda_context )
        global_papicuda_context = ( papicuda_context_t* ) papi_calloc( 1, sizeof( papicuda_context_t ) );

    /* Get list of all native CUDA events supported */
    err = papicuda_list_all_events( global_papicuda_context );
    if ( err!=0 ) return( err );

    /* Export some information */
    _cuda_vector.cmp_info.CmpIdx = cidx;
    _cuda_vector.cmp_info.num_native_events = global_papicuda_context->availEventSize;
    _cuda_vector.cmp_info.num_cntrs = _cuda_vector.cmp_info.num_native_events;
    _cuda_vector.cmp_info.num_mpx_cntrs = _cuda_vector.cmp_info.num_native_events;

    //SUBDBG( "Exiting PAPI_OK\n" );
    return ( PAPI_OK );
}


/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */
static int papicuda_init_control_state( hwd_control_state_t *ctrl )
{
    SUBDBG( "Entering\n" );
    ( void ) ctrl;
    papicuda_context_t *gctxt = global_papicuda_context;

    CHECK_PRINT_EVAL( !gctxt, "Error: The PAPI CUDA component needs to be initialized first", return( PAPI_ENOINIT ) );
    /* If no events were found during the initial component initialization, return error  */
    if( global_papicuda_context->availEventSize <= 0 ) {
        strncpy( _cuda_vector.cmp_info.disabled_reason, "ERROR CUDA: No events exist", PAPI_MAX_STR_LEN );
        return ( PAPI_EMISC );
    }
    /* If it does not exist, create the global structure to hold CUDA contexts and active events */
    if ( !global_papicuda_control ) {
        global_papicuda_control = ( papicuda_control_t* ) papi_calloc( 1, sizeof( papicuda_control_t ) );
        global_papicuda_control->countOfActiveCUContexts = 0;
        global_papicuda_control->activeEventCount = 0;
    }
    return PAPI_OK;
}

/** Triggered by eventset operations like add or remove.  For CUDA,
 * needs to be called multiple times from each seperate CUDA context
 * with the events to be measured from that context.  For each
 * context, create eventgroups for the events.
 */
static int papicuda_update_control_state( hwd_control_state_t *ctrl, NativeInfo_t *nativeInfo, int nativeCount, hwd_context_t *ctx )
{
    /* Note: NativeInfo_t is defined in papi_internal.h */
    SUBDBG( "Entering with nativeCount %d\n", nativeCount );
    ( void ) ctx;
    ( void ) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    papicuda_active_cucontext_t *currctrl;
    int currDeviceNum, currContextIdx, cuContextIdx;
    CUcontext currCuCtx;
    int index, ii, jj;

    if ( nativeCount == 0 ) {
        /* Does nativeCount=0 implies that the component is being reset!? */
        /* gctrl->activeEventCount = 0;  */
    } else {
        /* nativecount>0 so we need to process the events */
        // SUBDBG( "There are currently %d contexts\n", gctrl->countOfActiveCUContexts );

        /* Get/query some device and context specific information  */
        CHECK_PRINT_EVAL( ( *cudaGetDevicePtr )( &currDeviceNum )!=cudaSuccess, "cudaGetDevice: CUDA device MUST be set before adding events", return( PAPI_EMISC ) );
        CHECK_PRINT_EVAL( ( *cudaFreePtr )( NULL )!=cudaSuccess, "cudaFree: Failed to free in this CUDA context", return( PAPI_EMISC ) );
        CHECK_PRINT_EVAL( ( *cuCtxGetCurrentPtr )( &currCuCtx )!=CUDA_SUCCESS, "cuCtxGetCurrent: CUDA context MUST be initialized before adding events", return ( PAPI_EMISC ) );

        /* Find current context/control, creating it if does not exist */
        for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ )
            if ( gctrl->arrayOfActiveCUContexts[cuContextIdx]->context == currCuCtx ) break;
        CHECK_PRINT_EVAL( cuContextIdx==PAPICUDA_MAX_COUNTERS,  "Exceeded hardcoded maximum number of contexts (PAPICUDA_MAX_COUNTERS)", return( PAPI_EMISC ) );
        if ( cuContextIdx==gctrl->countOfActiveCUContexts ) {
            gctrl->arrayOfActiveCUContexts[cuContextIdx] = papi_calloc( 1, sizeof( papicuda_active_cucontext_t ) );
            CHECK_PRINT_EVAL( ( gctrl->arrayOfActiveCUContexts[cuContextIdx]==NULL ), "Memory allocation for new active context failed", return( PAPI_ENOMEM ) ) ;
            gctrl->arrayOfActiveCUContexts[cuContextIdx]->context = currCuCtx;
            gctrl->arrayOfActiveCUContexts[cuContextIdx]->deviceNum = currDeviceNum;
            gctrl->countOfActiveCUContexts++;
            SUBDBG( "Added a new context ... now %d\n", gctrl->countOfActiveCUContexts );
        }
        currContextIdx = cuContextIdx;
        currctrl = gctrl->arrayOfActiveCUContexts[currContextIdx];
        /* At this point, currCuCtx is at index cuContextIdx in the arrayOfActiveCUContexts array */

        /* For each event, check if it is already added.  If not, try to added it to the current context.
           Try each existing eventgroup.  If none will have this event, create a new event group.  If new event group will not have it... fail */
        /* For each event */
        for( ii = 0; ii < nativeCount; ii++ ) {
            index = nativeInfo[ii].ni_event; /* Get the PAPI event index from the user */
            /* Check to see if event is already in some context */
            SUBDBG( "Searching %d active events to see if event %d %s is already in some context\n", gctrl->activeEventCount, index, gctxt->availEventDesc[index].name );
            int eventAlreadyAdded=0;
            for( jj = 0; jj < gctrl->activeEventCount; jj++ ) {
                if ( gctrl->activeEventIndex[jj] == index ) {
                    eventAlreadyAdded=1;
                    break;
                }
            }

            /* If event was not found in any context.. try to insert it into current context */
            if ( !eventAlreadyAdded ) {
                SUBDBG( "Need to add event %d %s to the current context\n", index, gctxt->availEventDesc[index].name );
                /* Make sure that the device number for the event matches the device for this context */
                CHECK_PRINT_EVAL( (currDeviceNum!=gctxt->availEventDeviceNum[index]), "Current CUDA device cannot use this event", return( PAPI_EINVAL ) );
                /* if this event index corresponds to something from availEventIDArray */
                if ( index < ( int )gctxt->availEventSize ) {
                    /* lookup cuptieventid for this event index */
                    CUpti_EventID cuptieventid = gctxt->availEventIDArray[index];
                    CUpti_EventGroup cuptieventgroup;
                    int addstatus=!CUPTI_SUCCESS, gg;
                    SUBDBG( "Event %s is going to be added to current context %d having %d eventgroups\n", gctxt->availEventDesc[index].name, currContextIdx, currctrl->numEventGroups );
                    /* For each existing eventgroup, try to insert this event */
                    for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
                        cuptieventgroup = currctrl->eventGroup[gg];
                        addstatus = ( *cuptiEventGroupAddEventPtr )( cuptieventgroup, cuptieventid );
                        if ( addstatus==CUPTI_SUCCESS ) {
                            SUBDBG( "Event %s successfully added to current eventgroup %d:%d\n", gctxt->availEventDesc[index].name, currContextIdx, gg );
                            break;
                        }
                    }
                    /* If the event could not be added to any earlier eventgroup, create a new one and try again.  Fail if this does not succeed */
                    if ( addstatus!=CUPTI_SUCCESS ) {
                        //SUBDBG( "Event %s needs a new eventgroup\n", gctxt->availEventDesc[index].name );
                        CHECK_PRINT_EVAL( ( gg>PAPICUDA_MAX_COUNTERS-1 ), "For current CUDA device, could not add event (no more eventgroups can be added)", return( PAPI_EMISC ) );
                        //SUBDBG( "gg %d context %d %p\n", gg, currctrl->context, currctrl->context  );
                        CHECK_CUPTI_ERROR( ( *cuptiEventGroupCreatePtr )( currctrl->context, &currctrl->eventGroup[gg], 0 ), "cuptiEventGroupCreate" );
                        cuptieventgroup = currctrl->eventGroup[gg];
                        currctrl->numEventGroups++;
                        addstatus = ( *cuptiEventGroupAddEventPtr )( cuptieventgroup, cuptieventid );
                        CHECK_PRINT_EVAL( ( addstatus!=CUPTI_SUCCESS ), "cuptiEventGroupAddEvent: Could not add event (event may not match CUDA context)", return( PAPI_EMISC ) );
                        SUBDBG( "Event %s successfully added to new eventgroup %d:%d\n", gctxt->availEventDesc[index].name, currContextIdx, gg );
                    }
                }

                /* Record index of this active event back into the nativeInfo structure */
                nativeInfo[ii].ni_position = gctrl->activeEventCount;
                /* record added event at the higher level */
                CHECK_PRINT_EVAL( ( gctrl->activeEventCount==PAPICUDA_MAX_COUNTERS-1 ), "Exceeded maximum num of events (PAPI_MAX_COUNTERS)", return( PAPI_EMISC ) );
                gctrl->activeEventIndex[gctrl->activeEventCount] = index;
                gctrl->activeEventContextIdx[gctrl->activeEventCount] = currContextIdx;
                gctrl->activeEventValues[gctrl->activeEventCount] = 0;
                gctrl->activeEventCount++;

            }
        }
    }
    return ( PAPI_OK );
}

/** Triggered by PAPI_start().
 * For CUDA component, switch to each context and start all eventgroups.
*/
static int papicuda_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    SUBDBG( "Entering\n" );
    ( void ) ctx;
    ( void ) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    //papicuda_context_t *gctxt = global_papicuda_context;
    papicuda_active_cucontext_t *currctrl;
    int cuContextIdx, gg, ii;
    CUptiResult cuptiErr;
    CUcontext saveCtx, tmpCtx;

    //SUBDBG( "Reset all active event values\n" );
    for ( ii=0; ii<gctrl->activeEventCount; ii++ )
        gctrl->activeEventValues[ii] = 0;

    // SUBDBG( "Switch to each context and enable CUDA eventgroups associated with that context\n" );
    /* Save current cuda context */
    CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &saveCtx ), "cuCtxPopCurrent" );
    /* Switch to each context and enable CUDA eventgroups */
    for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ ) {
        currctrl = gctrl->arrayOfActiveCUContexts[cuContextIdx];
        //SUBDBG( "Try to switch to context %d associated with device %d\n", cuContextIdx, currctrl->deviceNum );
        CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( currctrl->context ),  "cuCtxPushCurrent" );
        for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
            // SUBDBG( "Enable event group\n" );
            cuptiErr = ( *cuptiEventGroupEnablePtr )( currctrl->eventGroup[gg] );
            CHECK_PRINT_EVAL( ( cuptiErr!=CUPTI_SUCCESS ), "cuptiEventGroupEnable: Could not enable one of the event groups", return( PAPI_EMISC ) );
            // SUBDBG( "Reset events in eventgroup\n" );
            cuptiErr = ( *cuptiEventGroupResetAllEventsPtr )( currctrl->eventGroup[gg] );
            CHECK_PRINT_EVAL( ( cuptiErr!=CUPTI_SUCCESS ), "cuptiEventGroupResetAllEvents: Could not reset the event groups", return( PAPI_EMISC ) );
            SUBDBG( "For papicuda context %d on device %d event group %d was enabled and reset\n", cuContextIdx, currctrl->deviceNum, gg );
        }
        // SUBDBG( "Pop temp context\n" );
        CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &tmpCtx ),  "cuCtxPopCurrent" );
    }
    //SUBDBG( "Restore original context\n" );
    CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( saveCtx ),  "cuCtxPushCurrent" );
    return ( PAPI_OK );
}

/** Triggered by PAPI_stop() */
static int papicuda_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    SUBDBG( "Entering to disable all CUPTI eventgroups\n" );
    ( void ) ctx;
    ( void ) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_active_cucontext_t *currctrl;
    int cuContextIdx, gg;
    CUptiResult cuptiErr;
    CUcontext saveCtx, tmpCtx;

    // SUBDBG( "Save initial CUDA context\n" );
    CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &saveCtx ), "cuCtxPopCurrent" );
    // SUBDBG( "Switch to each context and disable CUDA eventgroups\n" );
    for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ ) {
        currctrl = gctrl->arrayOfActiveCUContexts[cuContextIdx];
        //SUBDBG( "Try to switch to context %d associated with device %d\n", cuContextIdx, currctrl->deviceNum );
        CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( currctrl->context ),  "cuCtxPushCurrent" );
        for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
            // SUBDBG( "Disable events in eventgroup\n" );
            cuptiErr = ( *cuptiEventGroupDisablePtr )( currctrl->eventGroup[gg] );
            CHECK_PRINT_EVAL( ( cuptiErr!=CUPTI_SUCCESS ), "cuptiEventGroupDisable: Could not disable the event groups", return( PAPI_EMISC ) );
            SUBDBG( "For papicuda context %d on device %d event group %d was disabled\n", cuContextIdx, currctrl->deviceNum, gg );
        }
        CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &tmpCtx ),  "cuCtxPopCurrent" );
    }
    //SUBDBG( "Restore original context\n" );
    CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( saveCtx ),  "cuCtxPushCurrent" );
    return ( PAPI_OK );
}


/** Triggered by PAPI_read().  For CUDA component, switch to each
 * context, read all the eventgroups, and put the values in the
 * correct places. */
static int papicuda_read( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long ** events, int flags )
{
    SUBDBG( "Entering\n" );
    ( void ) ctx;
    ( void ) ctrl;
    ( void ) flags;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    papicuda_active_cucontext_t *currctrl;
    int cuContextIdx, gg, ii, jj;
    CUcontext saveCtx, tmpCtx;
    CUptiResult cuptiErr;
    size_t readEventValueBufferSize = sizeof( uint64_t )*PAPICUDA_MAX_COUNTERS;
    uint64_t readEventValueBuffer[PAPICUDA_MAX_COUNTERS];
    size_t readEventIDArraySize = sizeof( CUpti_EventID )*PAPICUDA_MAX_COUNTERS;
    CUpti_EventID readEventIDArray[PAPICUDA_MAX_COUNTERS];
    size_t numEventIDsRead;

    SUBDBG( "Switch to each context and read CUDA eventgroups\n" );
    // SUBDBG( "Save initial CUDA context\n" );
    CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &saveCtx ), "cuCtxPopCurrent" );
    /* Switch to each context and enable CUDA eventgroups */
    for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ ) {
        currctrl = gctrl->arrayOfActiveCUContexts[cuContextIdx];
        // SUBDBG( "Switch to context %d associated with device %d\n", cuContextIdx, currctrl->deviceNum );
        CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( currctrl->context ),  "cuCtxPushCurrent" );
        for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
            // SUBDBG( "Read from context %d eventgroup %d\n", cuContextIdx, gg );
            cuptiErr = ( *cuptiEventGroupReadAllEventsPtr )( currctrl->eventGroup[gg], CUPTI_EVENT_READ_FLAG_NONE, &readEventValueBufferSize, readEventValueBuffer, &readEventIDArraySize, readEventIDArray, &numEventIDsRead );
            CHECK_PRINT_EVAL( ( cuptiErr!=CUPTI_SUCCESS ), "cuptiEventGroupReadAllEvents: Could not read from CUPTI eventgroup", return( PAPI_EMISC ) );
            /* Match read values against active events by scanning activeEvents array and matching associated availEventIDs  */
            for( ii = 0; ii < ( int )numEventIDsRead; ii++ ) {
                for( jj = 0; jj < gctrl->activeEventCount; jj++ ) {
                    int eventIndex = gctrl->activeEventIndex[jj];
                    if ( gctrl->activeEventContextIdx[jj]==cuContextIdx && gctxt->availEventIDArray[eventIndex]==readEventIDArray[ii] ) {
                        gctrl->activeEventValues[jj] += ( long long )readEventValueBuffer[ii];
                        SUBDBG( "Matched read-eventID %d:%d value %ld activeEvent %d value %lld \n", jj, (int)readEventIDArray[ii], readEventValueBuffer[ii], eventIndex, gctrl->activeEventValues[jj] );
                        break;
                    }
                }
            }
        }
        CUresult cuErr = ( *cuCtxPopCurrentPtr ) ( &tmpCtx );
        if ( cuErr != CUDA_SUCCESS ) PAPIERROR ( "Error popping context %d\n", cuErr );
        CHECK_CU_ERROR( cuErr,  "cuCtxPopCurrent" );
    }
    //SUBDBG( "Restore original context\n" );
    CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( saveCtx ),  "cuCtxPushCurrent" );
    *events = gctrl->activeEventValues;
    return ( PAPI_OK );
}

/** Called at thread shutdown. Does nothing in the CUDA component. */
int papicuda_shutdown_thread( hwd_context_t * ctx )
{
    SUBDBG( "Entering\n" );
    ( void ) ctx;

    return ( PAPI_OK );
}

/** Triggered by PAPI_shutdown() and frees memory allocated in the CUDA component. */
static int papicuda_shutdown_component( void )
{
    SUBDBG( "Entering\n" );
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_context_t *gctxt = global_papicuda_context;
    int deviceNum, cuContextIdx;
    /* Free context  */
    if ( gctxt ) {
        for( deviceNum = 0; deviceNum < gctxt->deviceCount; deviceNum++ ) {
            papicuda_device_desc_t *mydevice = &gctxt->deviceArray[deviceNum];
            papi_free( mydevice->domainIDArray );
            papi_free( mydevice->domainIDNumEvents );
        }
        papi_free( gctxt->availEventIDArray );
        papi_free( gctxt->availEventDeviceNum );
        papi_free( gctxt->availEventDesc );
        papi_free( gctxt->deviceArray );
        papi_free( gctxt );
        global_papicuda_context = gctxt = NULL;
    }
    /* Free control  */
    if ( gctrl ) {
        for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ )
            if ( gctrl->arrayOfActiveCUContexts[cuContextIdx]!=NULL )
                papi_free( gctrl->arrayOfActiveCUContexts[cuContextIdx] );
        papi_free( gctrl );
        global_papicuda_control = gctrl = NULL;
    }
    // close the dynamic libraries needed by this component (opened in the init substrate call)
    dlclose( dl1 );
    dlclose( dl2 );
    dlclose( dl3 );
    return ( PAPI_OK );
}


/** This function sets various options in the component - Does nothing in the CUDA component.
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
 */
static int papicuda_ctrl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
    SUBDBG( "Entering\n" );
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
static int papicuda_set_domain( hwd_control_state_t * ctrl, int domain )
{
    SUBDBG( "Entering\n" );
    ( void ) ctrl;
    if ( ( PAPI_DOM_USER & domain ) ||
            ( PAPI_DOM_KERNEL & domain ) ||
            ( PAPI_DOM_OTHER & domain ) )
        return ( PAPI_OK );
    else
        return ( PAPI_EINVAL );
    return ( PAPI_OK );
}


/** Triggered by PAPI_reset() but only if the EventSet is currently
 *  running. If the eventset is not currently running, then the saved
 *  value in the EventSet is set to zero without calling this
 *  routine.  */
static int papicuda_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    SUBDBG( "Entering\n" );
    ( void ) ctx;
    ( void ) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_active_cucontext_t *currctrl;
    int cuContextIdx, gg, ii;
    CUptiResult cuptiErr;
    CUcontext saveCtx, tmpCtx;

    //SUBDBG( "Reset all active event values\n" );
    for ( ii=0; ii<gctrl->activeEventCount; ii++ )
        gctrl->activeEventValues[ii] = 0;
    // SUBDBG( "Save initial CUDA context and restore later\n" );
    CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &saveCtx ), "cuCtxPopCurrent" );
    // SUBDBG( "Switch to each context and reset CUDA eventgroups\n" );
    for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ ) {
        currctrl = gctrl->arrayOfActiveCUContexts[cuContextIdx];
        //SUBDBG( "Try to switch to context %d associated with device %d\n", cuContextIdx, currctrl->deviceNum );
        CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( currctrl->context ),  "cuCtxPushCurrent" );
        for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
            // SUBDBG( "Reset events in eventgroup\n" );
            cuptiErr = ( *cuptiEventGroupResetAllEventsPtr )( currctrl->eventGroup[gg] );
            CHECK_PRINT_EVAL( ( cuptiErr!=CUPTI_SUCCESS ), "cuptiEventGroupResetAllEvents: Could not reset the event groups", return( PAPI_EMISC ) );
            SUBDBG( "For papicuda context %d on device %d event group %d was enabled and reset\n", cuContextIdx, currctrl->deviceNum, gg );
        }
        CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &tmpCtx ),  "cuCtxPopCurrent" );
    }
    // SUBDBG( "Restore original context\n" );
    CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( saveCtx ),  "cuCtxPushCurrent" );
    return ( PAPI_OK );
}


/*
 * Disable and destroy the CUDA eventGroup
*/
static int papicuda_cleanup_eventset( hwd_control_state_t * ctrl )
{
    SUBDBG( "Entering\n" );
    ( void ) ctrl;
    papicuda_control_t *gctrl = global_papicuda_control;
    papicuda_active_cucontext_t *currctrl;
    int cuContextIdx, gg;
    CUptiResult cuptiErr;
    CUcontext saveCtx, tmpCtx;

    SUBDBG( "Switch to each context and disable CUDA eventgroups\n" );
    /* Save current cuda context and restore later */
    CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &saveCtx ), "cuCtxPopCurrent" );
    /* Switch to each context and enable CUDA eventgroups */
    for ( cuContextIdx=0; cuContextIdx<gctrl->countOfActiveCUContexts; cuContextIdx++ ) {
        currctrl = gctrl->arrayOfActiveCUContexts[cuContextIdx];
        /* Switch to this device / cuda context */
        CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( currctrl->context ),  "cuCtxPushCurrent" );
        for ( gg=0; gg<currctrl->numEventGroups; gg++ ) {
            /* Destroy the eventGroups; it also frees the perfmon hardware on the GPU */
            cuptiErr = ( *cuptiEventGroupDestroyPtr )( currctrl->eventGroup[gg] );
            CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupDestroy" );
        }
        currctrl->numEventGroups = 0;
        CHECK_CU_ERROR( ( *cuCtxPopCurrentPtr ) ( &tmpCtx ),  "cuCtxPopCurrent" );
    }
    CHECK_CU_ERROR( ( *cuCtxPushCurrentPtr ) ( saveCtx ),  "cuCtxPushCurrent" );
    /* Record that there are no active contexts or events */
    gctrl->activeEventCount = 0;
    return ( PAPI_OK );
}


/** Enumerate Native Events.
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 */
static int papicuda_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    //SUBDBG( "Entering\n" );
    switch( modifier ) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return ( PAPI_OK );
        break;
    case PAPI_ENUM_EVENTS:
        if( *EventCode < global_papicuda_context->availEventSize - 1 ) {
            *EventCode = *EventCode + 1;
            return ( PAPI_OK );
        } else
            return ( PAPI_ENOEVNT );
        break;
    default:
        return ( PAPI_EINVAL );
    }
    return ( PAPI_OK );
}


/** Takes a native event code and passes back the name
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int papicuda_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    //SUBDBG( "Entering EventCode %d\n", EventCode );
    unsigned int index = EventCode;
    papicuda_context_t *gctxt = global_papicuda_context;
    if ( index < gctxt->availEventSize ) {
        strncpy( name, gctxt->availEventDesc[index].name, len );
    } else {
        return ( PAPI_EINVAL );
    }
    //SUBDBG( "EventCode %d: Exit %s\n", EventCode, name );
    return ( PAPI_OK );
}


/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int papicuda_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    //SUBDBG( "Entering\n" );
    unsigned int index = EventCode;
    papicuda_context_t *gctxt = global_papicuda_context;
    if ( index < gctxt->availEventSize ) {
        strncpy( name, gctxt->availEventDesc[index].description, len );
    } else {
        return ( PAPI_EINVAL );
    }
    return ( PAPI_OK );
}


/** Vector that points to entry points for the component */
papi_vector_t _cuda_vector = {
    .cmp_info = {
        /* default component information (unspecified values are initialized to 0) */
        .name = "cuda",
        .short_name = "cuda",
        .version = "5.1",
        .description = "The CUDA component uses CuPTI for NVIDIA GPU hardware events",
        .num_mpx_cntrs = PAPICUDA_MAX_COUNTERS,
        .num_cntrs = PAPICUDA_MAX_COUNTERS,
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
    /* sizes of framework-opaque component-private structures... these are all unused in this component */
    .size = {
        .context = 1, /* sizeof( papicuda_context_t ), */
        .control_state = 1, /*sizeof( papicuda_control_t ), */
        .reg_value = 1, /*sizeof( papicuda_register_t ), */
        .reg_alloc = 1, /*sizeof( papicuda_reg_alloc_t ), */
    },
    /* function pointers in this component */
    .init_thread = papicuda_init_thread, /* ( hwd_context_t * ctx ) */
    .init_component = papicuda_init_component, /* ( int cidx ) */
    .init_control_state = papicuda_init_control_state, /* ( hwd_control_state_t * ctrl ) */
    .start = papicuda_start, /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .stop = papicuda_stop, /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .read = papicuda_read, /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags ) */
    .shutdown_component = papicuda_shutdown_component, /* ( void ) */
    .shutdown_thread = papicuda_shutdown_thread, /* ( hwd_context_t * ctx ) */
    .cleanup_eventset = papicuda_cleanup_eventset, /* ( hwd_control_state_t * ctrl ) */
    .ctl = papicuda_ctrl, /* ( hwd_context_t * ctx, int code, _papi_int_option_t * option ) */
    .update_control_state = papicuda_update_control_state, /* ( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx ) */
    .set_domain = papicuda_set_domain, /* ( hwd_control_state_t * cntrl, int domain ) */
    .reset = papicuda_reset, /* ( hwd_context_t * ctx, hwd_control_state_t * ctrl ) */
    .ntv_enum_events = papicuda_ntv_enum_events, /* ( unsigned int *EventCode, int modifier ) */
    .ntv_code_to_name = papicuda_ntv_code_to_name, /* ( unsigned int EventCode, char *name, int len ) */
    .ntv_code_to_descr = papicuda_ntv_code_to_descr, /* ( unsigned int EventCode, char *name, int len ) */
    //.ntv_code_to_bits = papicuda_ntv_code_to_bits,   /* ( unsigned int EventCode, hwd_register_t * bits ) */

};

