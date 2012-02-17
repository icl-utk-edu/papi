/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    vmware.c
 * @author  Matt Johnson
 *          mrj@eecs.utk.edu
 * @author  John Nelson
 *          jnelso37@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * VMware component
 *
 * @brief
 *	This is the VMware component for PAPI-V. It will allow the user access to
 *	the underlying hardware information available from a VMware virtual machine.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
/* Our component defined headers */
#include "vmware.h"
/* Headers required by VMware */
#include "vmGuestLib.h"

/* Dynamic code base */
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <dlfcn.h>
#endif

/* For inline assembly */
#define readpmc(counter, val) \
__asm__ __volatile__("rdpmc" \
: "=A" (val) \
: "c" (counter))

/* More dynamic code base */
#ifdef _WIN32
#define SLEEP(x) Sleep(x * 1000)
#else
#define SLEEP(x) sleep(x)
#endif

//static Bool done = FALSE;

/* Functions to dynamically load from the GuestLib library. */
char const * (*GuestLib_GetErrorText)(VMGuestLibError);
VMGuestLibError (*GuestLib_OpenHandle)(VMGuestLibHandle*);
VMGuestLibError (*GuestLib_CloseHandle)(VMGuestLibHandle);
VMGuestLibError (*GuestLib_UpdateInfo)(VMGuestLibHandle handle);
VMGuestLibError (*GuestLib_GetSessionId)(VMGuestLibHandle handle, VMSessionId *id);
VMGuestLibError (*GuestLib_GetCpuReservationMHz)(VMGuestLibHandle handle, uint32 *cpuReservationMHz);
VMGuestLibError (*GuestLib_GetCpuLimitMHz)(VMGuestLibHandle handle, uint32 *cpuLimitMHz);
VMGuestLibError (*GuestLib_GetCpuShares)(VMGuestLibHandle handle, uint32 *cpuShares);
VMGuestLibError (*GuestLib_GetCpuUsedMs)(VMGuestLibHandle handle, uint64 *cpuUsedMs);
VMGuestLibError (*GuestLib_GetHostProcessorSpeed)(VMGuestLibHandle handle, uint32 *mhz);
VMGuestLibError (*GuestLib_GetMemReservationMB)(VMGuestLibHandle handle, uint32 *memReservationMB);
VMGuestLibError (*GuestLib_GetMemLimitMB)(VMGuestLibHandle handle, uint32 *memLimitMB);
VMGuestLibError (*GuestLib_GetMemShares)(VMGuestLibHandle handle, uint32 *memShares);
VMGuestLibError (*GuestLib_GetMemMappedMB)(VMGuestLibHandle handle, uint32 *memMappedMB);
VMGuestLibError (*GuestLib_GetMemActiveMB)(VMGuestLibHandle handle, uint32 *memActiveMB);
VMGuestLibError (*GuestLib_GetMemOverheadMB)(VMGuestLibHandle handle, uint32 *memOverheadMB);
VMGuestLibError (*GuestLib_GetMemBalloonedMB)(VMGuestLibHandle handle, uint32 *memBalloonedMB);
VMGuestLibError (*GuestLib_GetMemSwappedMB)(VMGuestLibHandle handle, uint32 *memSwappedMB);
VMGuestLibError (*GuestLib_GetMemSharedMB)(VMGuestLibHandle handle, uint32 *memSharedMB);
VMGuestLibError (*GuestLib_GetMemSharedSavedMB)(VMGuestLibHandle handle, uint32 *memSharedSavedMB);
VMGuestLibError (*GuestLib_GetMemUsedMB)(VMGuestLibHandle handle, uint32 *memUsedMB);
VMGuestLibError (*GuestLib_GetElapsedMs)(VMGuestLibHandle handle, uint64 *elapsedMs);
VMGuestLibError (*GuestLib_GetResourcePoolPath)(VMGuestLibHandle handle, size_t *bufferSize, char *pathBuffer);
VMGuestLibError (*GuestLib_GetCpuStolenMs)(VMGuestLibHandle handle, uint64 *cpuStolenMs);
VMGuestLibError (*GuestLib_GetMemTargetSizeMB)(VMGuestLibHandle handle, uint64 *memTargetSizeMB);
VMGuestLibError (*GuestLib_GetHostNumCpuCores)(VMGuestLibHandle handle, uint32 *hostNumCpuCores);
VMGuestLibError (*GuestLib_GetHostCpuUsedMs)(VMGuestLibHandle handle, uint64 *hostCpuUsedMs);
VMGuestLibError (*GuestLib_GetHostMemSwappedMB)(VMGuestLibHandle handle, uint64 *hostMemSwappedMB);
VMGuestLibError (*GuestLib_GetHostMemSharedMB)(VMGuestLibHandle handle, uint64 *hostMemSharedMB);
VMGuestLibError (*GuestLib_GetHostMemUsedMB)(VMGuestLibHandle handle, uint64 *hostMemUsedMB);
VMGuestLibError (*GuestLib_GetHostMemPhysMB)(VMGuestLibHandle handle, uint64 *hostMemPhysMB);
VMGuestLibError (*GuestLib_GetHostMemPhysFreeMB)(VMGuestLibHandle handle, uint64 *hostMemPhysFreeMB);
VMGuestLibError (*GuestLib_GetHostMemKernOvhdMB)(VMGuestLibHandle handle, uint64 *hostMemKernOvhdMB);
VMGuestLibError (*GuestLib_GetHostMemMappedMB)(VMGuestLibHandle handle, uint64 *hostMemMappedMB);
VMGuestLibError (*GuestLib_GetHostMemUnmappedMB)(VMGuestLibHandle handle, uint64 *hostMemUnmappedMB);

/*
 * Handle for use with shared library.
 */

#ifdef _WIN32
static HMODULE dlHandle = NULL;
#else
static void *dlHandle = NULL;
#endif

/*
 * GuestLib handle.
 */
static VMGuestLibHandle glHandle;
static VMGuestLibError glError;

/*
 * Macro to load a single GuestLib function from the shared library.
 */

#ifdef _WIN32
#define LOAD_ONE_FUNC(funcname)									\
do {                                                            \
(FARPROC)funcname = GetProcAddress(dlHandle, "VM" #funcname);   \
if (funcname == NULL) {                                         \
error = GetLastError();                                         \
fprintf(stderr, "Failed to load \'%s\': %d\n",                  \
#funcname, error);                                              \
return FALSE;                                                   \
}                                                               \
} while (0)
#else
#define LOAD_ONE_FUNC(funcname)                                 \
do {                                                            \
funcname = dlsym(dlHandle, "VM" #funcname);                     \
if ((dlErrStr = dlerror()) != NULL) {                           \
fprintf(stderr, "Failed to load \'%s\': \'%s\'\n",              \
#funcname, dlErrStr);                                           \
return FALSE;                                                   \
}                                                               \
} while (0)
#endif

/*
 *-----------------------------------------------------------------------------
 *
 * LoadFunctions --
 *
 *      Load the functions from the shared library.
 *
 * Results:
 *      TRUE on success
 *      FALSE on failure
 *
 * Side effects:
 *      None
 *
 * Credit: VMware
 *-----------------------------------------------------------------------------
 */

static
Bool
LoadFunctions(void)
{
	/*
	 * First, try to load the shared library.
	 */
#ifdef _WIN32
	DWORD error;
	dlHandle = LoadLibrary("vmGuestLib.dll");
	if (!dlHandle) {
		error = GetLastError();
		fprintf(stderr, "LoadLibrary failed: %d\n", error);
		return FALSE;
	}
#else
	char const *dlErrStr;
	dlHandle = dlopen("libvmGuestLib.so", RTLD_NOW);
	if (!dlHandle) {
		dlErrStr = dlerror();
		fprintf(stderr, "dlopen failed: \'%s\'\n", dlErrStr);
		return FALSE;
	}
#endif
	
	/* Load all the individual library functions. */
	LOAD_ONE_FUNC(GuestLib_GetErrorText);
	LOAD_ONE_FUNC(GuestLib_OpenHandle);
	LOAD_ONE_FUNC(GuestLib_CloseHandle);
	LOAD_ONE_FUNC(GuestLib_UpdateInfo);
	LOAD_ONE_FUNC(GuestLib_GetSessionId);
	LOAD_ONE_FUNC(GuestLib_GetCpuReservationMHz);
	LOAD_ONE_FUNC(GuestLib_GetCpuLimitMHz);
	LOAD_ONE_FUNC(GuestLib_GetCpuShares);
	LOAD_ONE_FUNC(GuestLib_GetCpuUsedMs);
	LOAD_ONE_FUNC(GuestLib_GetHostProcessorSpeed);
	LOAD_ONE_FUNC(GuestLib_GetMemReservationMB);
	LOAD_ONE_FUNC(GuestLib_GetMemLimitMB);
	LOAD_ONE_FUNC(GuestLib_GetMemShares);
	LOAD_ONE_FUNC(GuestLib_GetMemMappedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemActiveMB);
	LOAD_ONE_FUNC(GuestLib_GetMemOverheadMB);
	LOAD_ONE_FUNC(GuestLib_GetMemBalloonedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSwappedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSharedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSharedSavedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemUsedMB);
	LOAD_ONE_FUNC(GuestLib_GetElapsedMs);
	LOAD_ONE_FUNC(GuestLib_GetResourcePoolPath);
	LOAD_ONE_FUNC(GuestLib_GetCpuStolenMs);
	LOAD_ONE_FUNC(GuestLib_GetMemTargetSizeMB);
	LOAD_ONE_FUNC(GuestLib_GetHostNumCpuCores);
	LOAD_ONE_FUNC(GuestLib_GetHostCpuUsedMs);
	LOAD_ONE_FUNC(GuestLib_GetHostMemSwappedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemSharedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemUsedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemPhysMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemPhysFreeMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemKernOvhdMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemMappedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemUnmappedMB);
	
	return TRUE;
}

/* Begin PAPI definitions */
papi_vector_t _vmware_vector;

/** This table contains the native events */
static VMWARE_native_event_entry_t *VMWARE_native_table;
/** number of events in the table*/
static int NUM_EVENTS = 1;

/************************************************************************/
/* Below is the actual "hardware implementation" of our VMWARE counters */
/************************************************************************/

static long_long VMWARE_autoinc_value = 0;

/** Code that resets the hardware.  */
static void
VMWARE_hardware_reset()
{
	VMWARE_autoinc_value = 0;
}

/** Code that reads event values.
 You might replace this with code that accesses
 hardware or reads values from the operatings system. */
static long_long
VMWARE_hardware_read( int which_one )
{
//	Bool success = TRUE;
	u_long_long host_tsc = 0;
	u_long_long elapsed_time = 0;
	u_long_long elapsed_apparent = 0;
	uint32 cpuLimitMHz = 0;
	uint32 cpuReservationMHz = 0;
	uint32 cpuShares = 0;
	uint64 cpuUsedMs = 0;
	uint32 hostMHz = 0;
	uint32 memReservationMB = 0;
	uint32 memLimitMB = 0;
	uint32 memShares = 0;
	uint32 memMappedMB = 0;
	uint32 memActiveMB = 0;
	uint32 memOverheadMB = 0;
	uint32 memBalloonedMB = 0;
	uint32 memSwappedMB = 0;
	uint32 memSharedMB = 0;
	uint32 memSharedSavedMB = 0;
	uint32 memUsedMB = 0;
	uint64 elapsedMs = 0;
	uint64 cpuStolenMs = 0;
	uint64 memTargetSizeMB = 0;
	uint32 hostNumCpuCores = 0;
	uint64 hostCpuUsedMs = 0;
	uint64 hostMemSwappedMB = 0;
	uint64 hostMemSharedMB = 0;
	uint64 hostMemUsedMB = 0;
	uint64 hostMemPhysMB = 0;
	uint64 hostMemPhysFreeMB = 0;
	uint64 hostMemKernOvhdMB = 0;
	uint64 hostMemMappedMB = 0;
	uint64 hostMemUnmappedMB = 0;
	static VMSessionId sessionId = 0;
	
	/* Try to load the library. */
	//    glError = GuestLib_OpenHandle(&glHandle);
	//    if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	//        fprintf(stderr,"OpenHandle failed: %s\n", GuestLib_GetErrorText(glError));
	//        return -1;
	//    }
	
	glError = GuestLib_UpdateInfo(glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr,"UpdateInfo failed: %s\n", GuestLib_GetErrorText(glError));
		return -1;
	}

	/* Retrieve and check the session ID */
	VMSessionId tmpSession;
	glError = GuestLib_GetSessionId(glHandle, &tmpSession);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr, "Failed to get session ID: %s\n", GuestLib_GetErrorText(glError));
		return -1;
	}
	if (tmpSession == 0) {
		fprintf(stderr, "Error: Got zero sessionId from GuestLib\n");
		return -1;
	}
	if (sessionId == 0) {
		sessionId = tmpSession;
//		fprintf(stderr, "Initial session ID is 0x%"FMT64"x\n", sessionId);
	} else if (tmpSession != sessionId) {
		sessionId = tmpSession;
//		fprintf(stderr, "SESSION CHANGED: New session ID is 0x%"FMT64"x\n", sessionId);
	}
	
	//If reading one of the pseudoperformance counters, we don't need to try to load the library
	switch ( which_one ) {

		case VMWARE_CPU_LIMIT_MHZ:          // #define 3
			glError = GuestLib_GetCpuLimitMHz(glHandle, &cpuLimitMHz);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr,"Failed to get CPU limit: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return cpuLimitMHz;
		case VMWARE_CPU_RESERVATION_MHZ:    // #define 4
			glError = GuestLib_GetCpuReservationMHz(glHandle, &cpuReservationMHz);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr,"Failed to get CPU reservation: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return cpuReservationMHz;
		case VMWARE_CPU_SHARES:             // #define 5
			glError = GuestLib_GetCpuShares(glHandle, &cpuShares);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr,"Failed to get cpu shares: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return cpuShares;
		case VMWARE_CPU_STOLEN_MS:          // #define 6
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				if (glError == VMGUESTLIB_ERROR_UNSUPPORTED_VERSION) {
					cpuStolenMs = 0;
					fprintf(stderr, "Skipping CPU stolen, not supported...\n");
				} else {
					fprintf(stderr, "Failed to get CPU stolen: %s\n", GuestLib_GetErrorText(glError));
					return -1;
				}
			}
			return cpuStolenMs;
		case VMWARE_CPU_USED_MS:            // #define 7
			glError = GuestLib_GetCpuUsedMs(glHandle, &cpuUsedMs);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get used ms: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return cpuUsedMs;
		case VMWARE_ELAPSED_MS:             // #define 8
			glError = GuestLib_GetElapsedMs(glHandle, &elapsedMs);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get elapsed ms: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return elapsedMs;
		case VMWARE_MEM_ACTIVE_MB:          // #define 9
			glError = GuestLib_GetMemActiveMB(glHandle, &memActiveMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get active mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memActiveMB;
		case VMWARE_MEM_BALLOONED_MB:       // #define 10
			glError = GuestLib_GetMemBalloonedMB(glHandle, &memBalloonedMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get ballooned mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memBalloonedMB;
		case VMWARE_MEM_LIMIT_MB:           // #define 11
			glError = GuestLib_GetMemLimitMB(glHandle, &memLimitMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				printf("Failed to get mem limit: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memLimitMB;
		case VMWARE_MEM_MAPPED_MB:          // #define 12
			glError = GuestLib_GetMemMappedMB(glHandle, &memMappedMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get mapped mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memMappedMB;
		case VMWARE_MEM_OVERHEAD_MB:        // #define 13
			glError = GuestLib_GetMemOverheadMB(glHandle, &memOverheadMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get overhead mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memOverheadMB;
		case VMWARE_MEM_RESERVATION_MB:     // #define 14
			glError = GuestLib_GetMemReservationMB(glHandle, &memReservationMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get mem reservation: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memReservationMB;
		case VMWARE_MEM_SHARED_MB:          // #define 15
			glError = GuestLib_GetMemSharedMB(glHandle, &memSharedMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get swapped mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memSharedMB;
			//        case VMWARE_MEM_SHARED_SAVED_MB:  // #define later
			//            return 0;
		case VMWARE_MEM_SHARES:             // #define 16
			glError = GuestLib_GetMemShares(glHandle, &memShares);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				if (glError == VMGUESTLIB_ERROR_NOT_AVAILABLE) {
					memShares = 0;
					fprintf(stderr, "Skipping mem shares, not supported...\n");
				} else {
					fprintf(stderr, "Failed to get mem shares: %s\n", GuestLib_GetErrorText(glError));
					return -1;
				}
			}
			return memShares;
		case VMWARE_MEM_SWAPPED_MB:         // #define 17
			glError = GuestLib_GetMemSwappedMB(glHandle, &memSwappedMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get swapped mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memSwappedMB;
		case VMWARE_MEM_TARGET_SIZE_MB:     // #define 18
			glError = GuestLib_GetMemTargetSizeMB(glHandle, &memTargetSizeMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				if (glError == VMGUESTLIB_ERROR_UNSUPPORTED_VERSION) {
					memTargetSizeMB = 0;
					fprintf(stderr, "Skipping target mem size, not supported...\n");
				} else {
					fprintf(stderr, "Failed to get target mem size: %s\n", GuestLib_GetErrorText(glError));
					return -1;
				}
			}
			return memTargetSizeMB;
		case VMWARE_MEM_USED_MB:            // #define 19
			glError = GuestLib_GetMemUsedMB(glHandle, &memUsedMB);
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get swapped mem: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return memUsedMB;
			
		case VMWARE_HOST_CPU_MHZ:               // #define 20
			glError = GuestLib_GetHostProcessorSpeed(glHandle, &hostMHz); 
			if (glError != VMGUESTLIB_ERROR_SUCCESS) {
				fprintf(stderr, "Failed to get host proc speed: %s\n", GuestLib_GetErrorText(glError));
				return -1;
			}
			return hostMHz;
#ifdef WITH_VMWARE_PPERFCTR
		case VMWARE_HOST_TSC:               // #define 0
			readpmc(0x10000, host_tsc);
			return host_tsc;
		case VMWARE_ELAPSED_TIME:           // #define 1
			readpmc(0x10001, elapsed_time);
			return elapsed_time;
		case VMWARE_ELAPSED_APPARENT:       // #define 2
			readpmc(0x10002, elapsed_apparent);
			return elapsed_apparent;
#endif
			/*
			 case VMWARE_HOST_CPU_NUM_CORES:     // #define 21
			 return 0;
			 case VMWARE_HOST_CPU_USED_MS:       // #define 22
			 return 0;
			 case VMWARE_HOST_MEM_SWAPPED_MB:    // #define 23
			 return 0;
			 case VMWARE_HOST_MEM_SHARED_MB:     // #define 24
			 return 0;
			 case VMWARE_HOST_MEM_USED_MB:       // #define 25
			 return 0;
			 case VMWARE_HOST_MEM_PHYS_MB:       // #define 26
			 return 0;
			 case VMWARE_HOST_MEM_PHYS_FREE_MB:  // #define 27
			 return 0;
			 case VMWARE_HOST_MEM_KERN_OVHD_MB:  // #define 28
			 return 0;
			 case VMWARE_HOST_MEM_MAPPED_MB:     // #define 29
			 return 0;
			 case VMWARE_HOST_MEM_UNMAPPED_MB:   // #define 30
			 return 0;
			 case VMWARE_SESSION_ID:             // #define 31
			 return 0;
			 */
		default:
			perror( "Invalid counter read" );
			return -1;
	}
	
	return 0;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
VMWARE_init( hwd_context_t * ctx )
{
	(void) ctx;
	SUBDBG( "VMWARE_init %p...", ctx );
	
	/* FIXME: do we need to make this thread safe? */
	
	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
VMWARE_init_substrate(  )
{
	SUBDBG( "VMWARE_init_substrate..." );
	
//#ifdef WITH_VMWARE_PPERFCTR
//	fprintf(stderr, "IT WORKED!!!!!\n");
//#endif
	
	/* Initialize and try to load the VMware library */
	/* Try to load the library. */
	if (!LoadFunctions()) {
		fprintf(stderr, "GuestLibTest: Failed to load shared library\n");
		return EXIT_FAILURE;
	}
	
	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */
	
	
#ifdef WITH_VMWARE_PPERFCTR
	NUM_EVENTS = 21; //32;
#else
	NUM_EVENTS = 18;
#endif
	
	
	
	/* Make sure we don't allocate too many counters.                 */
	/* This could be avoided if we dynamically allocate counter space */
	/*   when needed.                                                 */
	if ( NUM_EVENTS > VMWARE_MAX_COUNTERS ) {
		perror( "Too many counters allocated" );
		return EXIT_FAILURE;
	}
	
	/* Allocate memory for the our event table */
	VMWARE_native_table = ( VMWARE_native_event_entry_t * )
	malloc( sizeof ( VMWARE_native_event_entry_t ) * NUM_EVENTS );
	if ( VMWARE_native_table == NULL ) {
		perror( "malloc():Could not get memory for events table" );
		return EXIT_FAILURE;
	}
	
	
	
	/* fill in the event table parameters */
	
	// VMware library defines
	strcpy( VMWARE_native_table[0].name, "VMWARE_CPU_LIMIT" );
	strcpy( VMWARE_native_table[0].description, "Retrieves the upper limit of processor use in MHz available to the virtual machine." );
	VMWARE_native_table[0].writable = 0;
	strcpy( VMWARE_native_table[1].name, "VMWARE_CPU_RESERVATION" );
	strcpy( VMWARE_native_table[1].description, "Retrieves the minimum processing power in MHz reserved for the virtual machine." );
	VMWARE_native_table[1].writable = 0;
	strcpy( VMWARE_native_table[2].name, "VMWARE_CPU_SHARES" );
	strcpy( VMWARE_native_table[2].description, "Retrieves the number of CPU shares allocated to the virtual machine." );
	VMWARE_native_table[2].writable = 0;
	strcpy( VMWARE_native_table[3].name, "VMWARE_CPU_STOLEN" );
	strcpy( VMWARE_native_table[3].description, "Retrieves the number of milliseconds that the virtual machine was in a ready state (able to transition to a run state), but was not scheduled to run." );
	VMWARE_native_table[3].writable = 0;
	strcpy( VMWARE_native_table[4].name, "VMWARE_CPU_USED" );
	strcpy( VMWARE_native_table[4].description, "Retrieves the number of milliseconds during which the virtual machine has used the CPU. This value includes the time used by the guest operating system and the time used by virtualization code for tasks for this virtual machine. You can combine this value with the elapsed time (VMWARE_ELAPSED) to estimate the effective virtual machine CPU speed. This value is a subset of elapsedMs." );
	VMWARE_native_table[4].writable = 0;
	strcpy( VMWARE_native_table[5].name, "VMWARE_ELAPSED" );
	strcpy( VMWARE_native_table[5].description, "Retrieves the number of milliseconds that have passed in the virtual machine since it last started running on the server. The count of elapsed time restarts each time the virtual machine is powered on, resumed, or migrated using VMotion. This value counts milliseconds, regardless of whether the virtual machine is using processing power during that time. You can combine this value with the CPU time used by the virtual machine (VMWARE_CPU_USED) to estimate the effective virtual machine xCPU speed. cpuUsedMS is a subset of this value." );
	VMWARE_native_table[5].writable = 0;
	strcpy( VMWARE_native_table[6].name, "VMWARE_MEM_ACTIVE" );
	strcpy( VMWARE_native_table[6].description, "Retrieves the amount of memory the virtual machine is actively using in MB—its estimated working set size." );
	VMWARE_native_table[6].writable = 0;
	strcpy( VMWARE_native_table[7].name, "VMWARE_MEM_BALLOONED" );
	strcpy( VMWARE_native_table[7].description, "Retrieves the amount of memory that has been reclaimed from this virtual machine by the vSphere memory balloon driver (also referred to as the “vmmemctl” driver) in MB." );
	VMWARE_native_table[7].writable = 0;
	strcpy( VMWARE_native_table[8].name, "VMWARE_MEM_LIMIT" );
	strcpy( VMWARE_native_table[8].description, "Retrieves the upper limit of memory that is available to the virtual machine in MB." );
	VMWARE_native_table[8].writable = 0;
	strcpy( VMWARE_native_table[9].name, "VMWARE_MEM_MAPPED" );
	strcpy( VMWARE_native_table[9].description, "Retrieves the amount of memory that is allocated to the virtual machine in MB. Memory that is ballooned, swapped, or has never been accessed is excluded." );
	VMWARE_native_table[9].writable = 0;
	strcpy( VMWARE_native_table[10].name, "VMWARE_MEM_OVERHEAD" );
	strcpy( VMWARE_native_table[10].description, "Retrieves the amount of “overhead” memory associated with this virtual machine that is currently consumed on the host system in MB. Overhead memory is additional memory that is reserved for data structures required by the virtualization layer." );
	VMWARE_native_table[10].writable = 0;
	strcpy( VMWARE_native_table[11].name, "VMWARE_MEM_RESERVATION" );
	strcpy( VMWARE_native_table[11].description, "Retrieves the minimum amount of memory that is reserved for the virtual machine in MB." );
	VMWARE_native_table[11].writable = 0;
	strcpy( VMWARE_native_table[12].name, "VMWARE_MEM_SHARED" );
	strcpy( VMWARE_native_table[12].description, "Retrieves the amount of physical memory associated with this virtual machine that is copy‐on‐write (COW) shared on the host in MB." );
	VMWARE_native_table[12].writable = 0;
	//  strcpy( VMWARE_native_table[INF].name, "VMWARE_MEM_SHARED_SAVED" );
	//	strcpy( VMWARE_native_table[INF].description, "Retrieves the estimated amount of physical memory on the host saved from copy‐on‐write (COW) shared guest physical memory." );
	//	VMWARE_native_table[INF].writable = 0;
	strcpy( VMWARE_native_table[13].name, "VMWARE_MEM_SHARES" );
	strcpy( VMWARE_native_table[13].description, "Retrieves the number of memory shares allocated to the virtual machine." );
	VMWARE_native_table[13].writable = 0;
	strcpy( VMWARE_native_table[14].name, "VMWARE_MEM_SWAPPED" );
	strcpy( VMWARE_native_table[14].description, "Retrieves the amount of memory that has been reclaimed from this virtual machine by transparently swapping guest memory to disk in MB." );
	VMWARE_native_table[14].writable = 0;
	strcpy( VMWARE_native_table[15].name, "VMWARE_MEM_TARGET_SIZE" );
	strcpy( VMWARE_native_table[15].description, "Retrieves the size of the target memory allocation for this virtual machine in MB." );
	VMWARE_native_table[15].writable = 0;
	strcpy( VMWARE_native_table[16].name, "VMWARE_MEM_USED" );
	strcpy( VMWARE_native_table[16].description, "Retrieves the estimated amount of physical host memory currently consumed for this virtual machineʹs physical memory." );
	VMWARE_native_table[16].writable = 0;
	strcpy( VMWARE_native_table[17].name, "VMWARE_HOST_CPU" );
	strcpy( VMWARE_native_table[17].description, "Retrieves the speed of the ESX system’s physical CPU in MHz." );
	VMWARE_native_table[17].writable = 0;
#ifdef WITH_VMWARE_PPERFCTR
	// For VMWare Pseudo Performance Counters
	strcpy( VMWARE_native_table[18].name, "VMWARE_HOST_TSC" );
	strcpy( VMWARE_native_table[18].description, "Physical host TSC" );
	VMWARE_native_table[18].writable = 0;
	strcpy( VMWARE_native_table[19].name, "VMWARE_ELAPSED_TIME" );
	strcpy( VMWARE_native_table[19].description, "Elapsed real time in ns." );
	VMWARE_native_table[19].writable = 0;
	strcpy( VMWARE_native_table[20].name, "VMWARE_ELAPSED_APPARENT" );
	strcpy( VMWARE_native_table[20].description, "Elapsed apparent time in ns." );
	VMWARE_native_table[20].writable = 0;
#endif
	/* The selector has to be !=0 . Starts with 1 */
	VMWARE_native_table[0].resources.selector = 1;
	VMWARE_native_table[1].resources.selector = 2;
	VMWARE_native_table[2].resources.selector = 3;
	VMWARE_native_table[3].resources.selector = 4;
	VMWARE_native_table[4].resources.selector = 5;
	VMWARE_native_table[5].resources.selector = 6;
	VMWARE_native_table[6].resources.selector = 7;
	VMWARE_native_table[7].resources.selector = 8;
	VMWARE_native_table[8].resources.selector = 9;
	VMWARE_native_table[9].resources.selector = 10;
	VMWARE_native_table[10].resources.selector = 11;
	VMWARE_native_table[11].resources.selector = 12;
	VMWARE_native_table[12].resources.selector = 13;
	VMWARE_native_table[13].resources.selector = 14;
	VMWARE_native_table[14].resources.selector = 15;
	VMWARE_native_table[15].resources.selector = 16;
	VMWARE_native_table[16].resources.selector = 17;
	VMWARE_native_table[17].resources.selector = 18;
#ifdef WITH_VMWARE_PPERFCTR
	VMWARE_native_table[18].resources.selector = 19;
	VMWARE_native_table[19].resources.selector = 20;
	VMWARE_native_table[20].resources.selector = 21;
#endif
	/*
	 VMWARE_native_table[21].resources.selector = 22;
	 VMWARE_native_table[22].resources.selector = 23;
	 VMWARE_native_table[23].resources.selector = 24;
	 VMWARE_native_table[24].resources.selector = 25;
	 VMWARE_native_table[25].resources.selector = 26;
	 VMWARE_native_table[26].resources.selector = 27;
	 VMWARE_native_table[27].resources.selector = 28;
	 VMWARE_native_table[28].resources.selector = 29;
	 VMWARE_native_table[29].resources.selector = 30;
	 VMWARE_native_table[30].resources.selector = 31;
	 VMWARE_native_table[31].resources.selector = 32;
	 */
	
	_vmware_vector.cmp_info.num_native_events = NUM_EVENTS;
	
	return PAPI_OK;
}


/** Setup the counter control structure */
int
VMWARE_init_control_state( hwd_control_state_t * ctrl )
{
	SUBDBG( "VMWARE_init_control_state..." );
	
	/* set the hardware to initial conditions */
	VMWARE_hardware_reset(  );
	
	/* set the counters last-accessed time */
	( ( VMWARE_control_state_t * ) ctrl )->lastupdate = PAPI_get_real_usec(  );
	
	return PAPI_OK;
}


/** Enumerate Native Events 
 @param EventCode is the event of interest
 @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 */
int
VMWARE_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );
	switch ( modifier ) {
		/* return EventCode of first event */
		case PAPI_ENUM_FIRST:
			*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
			return PAPI_OK;
			break;
		/* return EventCode of passed-in Event */
		case PAPI_ENUM_EVENTS:{
			int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
			
			if ( index < NUM_EVENTS - 1 ) {
				*EventCode = *EventCode + 1;
				return PAPI_OK;
			} else {
				return PAPI_ENOEVNT;
			}
			break;
		}
		default:
			return PAPI_EINVAL;
	}
	return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 @param EventCode is the native event code
 @param name is a pointer for the name to be copied to
 @param len is the size of the string
 */
int
VMWARE_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
	strncpy( name, VMWARE_native_table[index].name, len );
	return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 @param EventCode is the native event code
 @param name is a pointer for the description to be copied to
 @param len is the size of the string
 */
int
VMWARE_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
	strncpy( name, VMWARE_native_table[index].description, len );
	return PAPI_OK;
}

/** This takes an event and returns the bits that would be written
 out to the hardware device (this is very much tied to CPU-type support */
int
VMWARE_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	(void) EventCode;
	(void) bits;
	SUBDBG( "Want native bits for event %d", EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK);
	return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int
VMWARE_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx )
{
	int i, index;
	(void) ptr;
	(void) ctx;
	SUBDBG( "VMWARE_update_control_state %p %p...", ptr, ctx );
	for ( i = 0; i < count; i++ ) {
		index =
		native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position =
		VMWARE_native_table[index].resources.selector - 1;
		SUBDBG
		( "\nnative[%i].ni_position = VMWARE_native_table[%i].resources.selector-1 = %i;",
		 i, index, native[i].ni_position );
	}
	return PAPI_OK;
}

/** Triggered by PAPI_start() */
int
VMWARE_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	(void) ctx;
	(void) ctrl;
	SUBDBG( "VMWARE_start %p %p...", ctx, ctrl );
	/* anything that would need to be set at counter start time */
	glError = GuestLib_OpenHandle(&glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr,"OpenHandle failed: %s\n", GuestLib_GetErrorText(glError));
		return EXIT_FAILURE;
	}
	return PAPI_OK;
}


/** Triggered by PAPI_stop() */
int
VMWARE_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	(void) ctx;
	(void) ctrl;
	SUBDBG( "VMWARE_stop %p %p...", ctx, ctrl );
	
	/* anything that would need to be done at counter stop time */
	glError = GuestLib_CloseHandle(glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr, "Failed to CloseHandle: %s\n", GuestLib_GetErrorText(glError));
		//        success = FALSE;
		return EXIT_FAILURE;
	}
	
	return PAPI_OK;
}


/** Triggered by PAPI_read() */
int
VMWARE_read( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events, int flags )
{
	(void) ctx;
	(void) ctrl;
	(void) flags;
	SUBDBG( "VMWARE_read... %p %d", ctx, flags );
	
	// update this to a for loop to account for all counters, per Vince.
	int i = 0;
	for (i=0; i<NUM_EVENTS; ++i) {
		((VMWARE_control_state_t * )ctrl)->counter[i] = VMWARE_hardware_read(i);
		if (((VMWARE_control_state_t * )ctrl)->counter[i] < 0) {
			return EXIT_FAILURE;
		}
	}
	
	*events = ( ( VMWARE_control_state_t * ) ctrl )->counter;   // serve cached data
	
	return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
VMWARE_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long events[] )
{
	(void) ctx;
	(void) ctrl;
	(void) events;
	SUBDBG( "VMWARE_write... %p %p", ctx, ctrl );
	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */
	return PAPI_OK;
}


/** Triggered by PAPI_reset */
int
VMWARE_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	(void) ctx;
	SUBDBG( "VMWARE_reset ctx=%p ctrl=%p...", ctx, ctrl );
	/* Reset the hardware */
	VMWARE_hardware_reset(  );
	/* Set the counters last-accessed time */
	( ( VMWARE_control_state_t * ) ctrl )->lastupdate = PAPI_get_real_usec(  );
	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
VMWARE_shutdown( hwd_context_t * ctx )
{
	(void) ctx;
	SUBDBG( "VMWARE_shutdown... %p", ctx );
	/* Last chance to clean up */
#ifdef _WIN32
	if (!FreeLibrary(dlHandle)) {
		DWORD error = GetLastError();
		fprintf(stderr, "Failed to FreeLibrary: %d\n", error);
		return EXIT_FAILURE;
	}
#else
	if (dlclose(dlHandle)) {
		fprintf(stderr, "dlclose failed\n");
		return EXIT_FAILURE;
	}
#endif
	return PAPI_OK;
}



/** This function sets various options in the substrate
  @param ctx
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param option
 */
int
VMWARE_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	
	(void) ctx;
	(void) code;
	(void) option;
	
	SUBDBG( "VMWARE_ctl..." );
	
	/* FIXME.  This should maybe set up more state, such as which counters are active and */
	/*         counter mappings. */
	
	return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
 In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 By default return PAPI_EINVAL if none of those are specified
 and PAPI_OK with success
 PAPI_DOM_USER is only user context is counted
 PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 PAPI_DOM_ALL   is all of the domains
 */
int
VMWARE_set_domain( hwd_control_state_t * cntrl, int domain )
{
	(void) cntrl;
	int found = 0;
	SUBDBG( "VMWARE_set_domain..." );
	if ( PAPI_DOM_USER & domain ) {
		SUBDBG( " PAPI_DOM_USER " );
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		SUBDBG( " PAPI_DOM_KERNEL " );
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		SUBDBG( " PAPI_DOM_OTHER " );
		found = 1;
	}
	if ( PAPI_DOM_ALL & domain ) {
		SUBDBG( " PAPI_DOM_ALL " );
		found = 1;
	}
	if ( !found ) {
		return ( PAPI_EINVAL );
	}
	return PAPI_OK;
}


/** Vector that points to entry points for our component */
papi_vector_t _vmware_vector = {
	.cmp_info = {
		/* default component information (unspecified values are initialized to 0) */
		.name = "vmware.c",
		.version = "$Revision$",
		.num_mpx_cntrs = PAPI_MPX_DEF_DEG,
		.num_cntrs = VMWARE_MAX_COUNTERS,
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
	},
	/* sizes of framework-opaque component-private structures */
	.size = {
		.context = sizeof ( VMWARE_context_t ),
		.control_state = sizeof ( VMWARE_control_state_t ),
		.reg_value = sizeof ( VMWARE_register_t ),
		.reg_alloc = sizeof ( VMWARE_reg_alloc_t ),
	}
	,
	/* function pointers in this component */
	.init = VMWARE_init,
	.init_substrate = VMWARE_init_substrate,
	.init_control_state = VMWARE_init_control_state,
	.start = VMWARE_start,
	.stop = VMWARE_stop,
	.read = VMWARE_read,
	.write = VMWARE_write,
	.shutdown = VMWARE_shutdown,
	.ctl = VMWARE_ctl,
	.bpt_map_set = NULL,
	.bpt_map_avail = NULL,
	.bpt_map_exclusive = NULL,
	.bpt_map_shared = NULL,
	.bpt_map_preempt = NULL,
	.bpt_map_update = NULL,
	
	.update_control_state = VMWARE_update_control_state,
	.set_domain = VMWARE_set_domain,
	.reset = VMWARE_reset,
	
	.ntv_enum_events = VMWARE_ntv_enum_events,
	.ntv_code_to_name = VMWARE_ntv_code_to_name,
	.ntv_code_to_descr = VMWARE_ntv_code_to_descr,
	.ntv_code_to_bits = VMWARE_ntv_code_to_bits,
	.ntv_bits_to_info = NULL,
};

