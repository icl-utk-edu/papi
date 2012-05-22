/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    vmware.c
 * @author  Matt Johnson
 *          mrj@eecs.utk.edu
 * @author  John Nelson
 *          jnelso37@eecs.utk.edu
 * @author  Vince Weaver
 *          vweaver1@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * VMware component
 *
 * @brief
 *	This is the VMware component for PAPI-V. It will allow user access to
 *	hardware information available from a VMware virtual machine.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <unistd.h>
#include <dlfcn.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#define VMWARE_MAX_COUNTERS 256

#define VMWARE_CPU_LIMIT_MHZ            0
#define VMWARE_CPU_RESERVATION_MHZ      1
#define VMWARE_CPU_SHARES               2
#define VMWARE_CPU_STOLEN_MS            3
#define VMWARE_CPU_USED_MS              4
#define VMWARE_ELAPSED_MS               5

#define VMWARE_MEM_ACTIVE_MB            6
#define VMWARE_MEM_BALLOONED_MB         7
#define VMWARE_MEM_LIMIT_MB             8
#define VMWARE_MEM_MAPPED_MB            9
#define VMWARE_MEM_OVERHEAD_MB          10
#define VMWARE_MEM_RESERVATION_MB       11
#define VMWARE_MEM_SHARED_MB            12
#define VMWARE_MEM_SHARES               13
#define VMWARE_MEM_SWAPPED_MB           14
#define VMWARE_MEM_TARGET_SIZE_MB       15
#define VMWARE_MEM_USED_MB              16

#define VMWARE_HOST_CPU_MHZ             17

/* The following 3 require VMWARE_PSEUDO_PERFORMANCE env_var to be set. */

#define VMWARE_HOST_TSC			18
#define VMWARE_ELAPSED_TIME             19
#define VMWARE_ELAPSED_APPARENT         20


/** Structure that stores private information for each event */
struct _vmware_register {
    unsigned int selector;
    /**< Signifies which counter slot is being used */
    /**< Indexed from 1 as 0 has a special meaning  */
};

/** This structure is used to build the table of events */
struct _vmware_native_event_entry {
	char name[PAPI_MAX_STR_LEN];        /**< Name of the counter         */
	char description[PAPI_HUGE_STR_LEN]; /**< Description of counter     */
        char units[PAPI_MIN_STR_LEN];
        int which_counter;
};

struct _vmware_reg_alloc {
	struct _vmware_register ra_bits;
};


/** Holds control flags, usually out-of band configuration of the hardware */
struct _vmware_control_state {
	long_long counter[VMWARE_MAX_COUNTERS];
};

/** Holds per-thread information */
struct _vmware_context {
	struct _vmware_control_state state;
};





/* For inline assembly */
#define readpmc(counter, val) \
__asm__ __volatile__("rdpmc" \
: "=A" (val) \
: "c" (counter))


#ifdef VMGUESTLIB
/* Headers required by VMware */
#include "vmGuestLib.h"

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


static void *dlHandle = NULL;

/*
 * GuestLib handle.
 */
static VMGuestLibHandle glHandle;
static VMGuestLibError glError;

/*
 * Macro to load a single GuestLib function from the shared library.
 */

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

static int
LoadFunctions(void)
{

#ifdef VMGUESTLIB
	/*
	 * First, try to load the shared library.
	 */

	char const *dlErrStr;
	dlHandle = dlopen("libvmGuestLib.so", RTLD_NOW);
	if (!dlHandle) {
		dlErrStr = dlerror();
		fprintf(stderr, "dlopen failed: \'%s\'\n", dlErrStr);
		return FALSE;
	}

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
#endif
	return PAPI_OK;
}



/* Begin PAPI definitions */
papi_vector_t _vmware_vector;

/** This table contains the native events */
static struct _vmware_native_event_entry *_vmware_native_table;
/** number of events in the table*/
static int num_events = 0;
static int use_pseudo=0;

/************************************************************************/
/* Below is the actual "hardware implementation" of our VMWARE counters */
/************************************************************************/

/** Code that reads event values.
 You might replace this with code that accesses
 hardware or reads values from the operatings system. */
static long long
_vmware_hardware_read( int which_one )
{

	uint64_t host_tsc = 0;
	uint64_t elapsed_time = 0;
	uint64_t elapsed_apparent = 0;

#ifdef VMGUESTLIB
	uint32_t cpuLimitMHz = 0;
	uint32_t cpuReservationMHz = 0;
	uint32_t cpuShares = 0;
	uint64_t cpuUsedMs = 0;
	uint32_t hostMHz = 0;
	uint32_t memReservationMB = 0;
	uint32_t memLimitMB = 0;
	uint32_t memShares = 0;
	uint32_t memMappedMB = 0;
	uint32_t memActiveMB = 0;
	uint32_t memOverheadMB = 0;
	uint32_t memBalloonedMB = 0;
	uint32_t memSwappedMB = 0;
	uint32_t memSharedMB = 0;
	uint32_t memSharedSavedMB = 0;
	uint32_t memUsedMB = 0;
	uint64_t elapsedMs = 0;
	uint64_t cpuStolenMs = 0;
	uint64_t memTargetSizeMB = 0;
	uint32_t hostNumCpuCores = 0;
	uint64_t hostCpuUsedMs = 0;
	uint64_t hostMemSwappedMB = 0;
	uint64_t hostMemSharedMB = 0;
	uint64_t hostMemUsedMB = 0;
	uint64_t hostMemPhysMB = 0;
	uint64_t hostMemPhysFreeMB = 0;
	uint64_t hostMemKernOvhdMB = 0;
	uint64_t hostMemMappedMB = 0;
	uint64_t hostMemUnmappedMB = 0;
	static VMSessionId sessionId = 0;

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
	} else if (tmpSession != sessionId) {
		sessionId = tmpSession;
	}
#endif

	switch ( which_one ) {

#ifdef VMGUESTLIB
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
#endif
		case VMWARE_HOST_TSC:
			readpmc(0x10000, host_tsc);
			return host_tsc;
		case VMWARE_ELAPSED_TIME:
			readpmc(0x10001, elapsed_time);
			return elapsed_time;
		case VMWARE_ELAPSED_APPARENT:
			readpmc(0x10002, elapsed_apparent);
			return elapsed_apparent;
		default:
			perror( "Invalid counter read" );
			return -1;
	}

	return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
_vmware_init( hwd_context_t *ctx )
{
	(void) ctx;

	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_vmware_init_substrate( int cidx )
{
  int result;

	SUBDBG( "_vmware_init_substrate..." );

	/* Initialize and try to load the VMware library */
	/* Try to load the library. */
	result=LoadFunctions();

	if (result!=PAPI_OK) {
	   strncpy(_vmware_vector.cmp_info.disabled_reason,
		  "GuestLibTest: Failed to load shared library",
		   PAPI_MAX_STR_LEN);
	   return PAPI_ESBSTR;
	}

	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */

	/* Allocate memory for the our event table */
	_vmware_native_table = ( struct _vmware_native_event_entry * )
	  calloc( VMWARE_MAX_COUNTERS, sizeof ( struct _vmware_native_event_entry ));
	if ( _vmware_native_table == NULL ) {
	   return PAPI_ENOMEM;
	}

	/* fill in the event table parameters */
#ifdef VMGUESTLIB
	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_CPU_LIMIT" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the upper limit of processor use in MHz "
		"available to the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_LIMIT_MHZ;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_CPU_RESERVATION" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the minimum processing power in MHz "
		"reserved for the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_RESERVATION_MHZ;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_CPU_SHARES" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of CPU shares allocated "
		"to the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"shares");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_SHARES;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_CPU_STOLEN" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds that the "
		"virtual machine was in a ready state (able to "
		"transition to a run state), but was not scheduled to run.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_STOLEN_MS;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_CPU_USED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds during which "
		"the virtual machine has used the CPU. This value "
		"includes the time used by the guest operating system "
		"and the time used by virtualization code for tasks for "
		"this virtual machine. You can combine this value with "
		"the elapsed time (VMWARE_ELAPSED) to estimate the "
		"effective virtual machine CPU speed. This value is a "
		"subset of elapsedMs.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_USED_MS;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_ELAPSED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds that have passed "
		"in the virtual machine since it last started running on "
		"the server. The count of elapsed time restarts each time "
		"the virtual machine is powered on, resumed, or migrated "
		"using VMotion. This value counts milliseconds, regardless "
		"of whether the virtual machine is using processing power "
		"during that time. You can combine this value with the CPU "
		"time used by the virtual machine (VMWARE_CPU_USED) to "
		"estimate the effective virtual machine xCPU speed. "
		"cpuUsedMS is a subset of this value.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_ELAPSED_MS;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_ACTIVE" );
	strncpy( _vmware_native_table[num_events].description,
		 "Retrieves the amount of memory the virtual machine is "
		 "actively using in MB - Its estimated working set size.",
		 PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
                 VMWARE_MEM_ACTIVE_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_BALLOONED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that has been reclaimed "
		"from this virtual machine by the vSphere memory balloon "
		"driver (also referred to as the 'vmemctl' driver) in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_BALLOONED_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_LIMIT" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the upper limit of memory that is available "
		"to the virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_LIMIT_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_MAPPED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that is allocated to "
		"the virtual machine in MB. Memory that is ballooned, "
		"swapped, or has never been accessed is excluded.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_MAPPED_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_OVERHEAD" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of 'overhead' memory associated "
		"with this virtual machine that is currently consumed "
		"on the host system in MB. Overhead memory is additional "
		"memory that is reserved for data structures required by "
		"the virtualization layer.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_OVERHEAD_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_RESERVATION" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the minimum amount of memory that is "
		"reserved for the virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_RESERVATION_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_SHARED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of physical memory associated "
		"with this virtual machine that is copy-on-write (COW) "
		"shared on the host in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SHARED_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_SHARES" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of memory shares allocated to "
		"the virtual machine.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"shares");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SHARES;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_SWAPPED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that has been reclaimed "
		"from this virtual machine by transparently swapping "
		"guest memory to disk in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SWAPPED_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_TARGET_SIZE" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the size of the target memory allocation "
		"for this virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_TARGET_SIZE_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_MEM_USED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the estimated amount of physical host memory "
		"currently consumed for this virtual machine's "
		"physical memory.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_USED_MB;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"VMWARE_HOST_CPU" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the speed of the ESX system's physical "
		"CPU in MHz.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_HOST_CPU_MHZ;
	num_events++;
#endif

	/* For VMWare Pseudo Performance Counters */
	if ( getenv( "PAPI_VMWARE_PSEUDOPERFORMANCE" ) ) {

	        use_pseudo=1;

		strcpy( _vmware_native_table[num_events].name,
			"VMWARE_HOST_TSC" );
		strncpy( _vmware_native_table[num_events].description,
			"Physical host TSC",
			PAPI_HUGE_STR_LEN );
		strcpy( _vmware_native_table[num_events].units,"cycles");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_HOST_TSC;
		num_events++;

		strcpy( _vmware_native_table[num_events].name,
			"VMWARE_ELAPSED_TIME" );
		strncpy( _vmware_native_table[num_events].description,
			"Elapsed real time in ns.",
			PAPI_HUGE_STR_LEN );
	        strcpy( _vmware_native_table[num_events].units,"ns");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_ELAPSED_TIME;
		num_events++;

		strcpy( _vmware_native_table[num_events].name,
			"VMWARE_ELAPSED_APPARENT" );
		strncpy( _vmware_native_table[num_events].description,
			"Elapsed apparent time in ns.",
			PAPI_HUGE_STR_LEN );
	        strcpy( _vmware_native_table[num_events].units,"ns");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_ELAPSED_APPARENT;
		num_events++;
	}

	_vmware_vector.cmp_info.num_native_events = num_events;

	return PAPI_OK;
}

/** Setup the counter control structure */
int
_vmware_init_control_state( hwd_control_state_t *ctl )
{
  (void) ctl;

	return PAPI_OK;
}

/** Enumerate Native Events 
 @param EventCode is the event of interest
 @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 */
int
_vmware_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	switch ( modifier ) {
			/* return EventCode of first event */
		case PAPI_ENUM_FIRST:
		     if (num_events==0) return PAPI_ENOEVNT;
		     *EventCode = PAPI_NATIVE_MASK;
		     return PAPI_OK;
		     break;
			/* return EventCode of passed-in Event */
		case PAPI_ENUM_EVENTS:{
		     int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		     if ( index < num_events - 1 ) {
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

int
_vmware_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

  int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

  info->event_code=EventCode;
  strncpy( info->symbol, _vmware_native_table[index].name, 
           sizeof(info->symbol));

  strncpy( info->long_descr, _vmware_native_table[index].description, 
           sizeof(info->symbol));

  strncpy( info->units, _vmware_native_table[index].units, 
           sizeof(info->units));

  return PAPI_OK;
}


/** Takes a native event code and passes back the name 
 @param EventCode is the native event code
 @param name is a pointer for the name to be copied to
 @param len is the size of the string
 */
int
_vmware_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	if ( index >= 0 && index < num_events ) {
	   strncpy( name, _vmware_native_table[index].name, len );
	}
	return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 @param EventCode is the native event code
 @param name is a pointer for the description to be copied to
 @param len is the size of the string
 */
int
_vmware_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	if ( index >= 0 && index < num_events ) {
	   strncpy( name, _vmware_native_table[index].description, len );
	}
	return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int
_vmware_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native, int count, hwd_context_t * ctx )
{
	int i, index;
	(void) ptr;
	(void) ctx;
	SUBDBG( "_vmware_update_control_state %p %p...", ptr, ctx );
	for ( i = 0; i < count; i++ ) {
		index =
		native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		//		native[i].ni_position =
		  //		_vmware_native_table[index].resources.selector - 1;
		SUBDBG
		( "\nnative[%i].ni_position = _vmware_native_table[%i].resources.selector-1 = %i;",
		 i, index, native[i].ni_position );
	}
	return PAPI_OK;
}

/** Triggered by PAPI_start() */
int
_vmware_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	(void) ctx;
	(void) ctl;

#ifdef VMGUESTLIB
	glError = GuestLib_OpenHandle(&glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr,"OpenHandle failed: %s\n", GuestLib_GetErrorText(glError));
		return EXIT_FAILURE;
	}
#endif

	return PAPI_OK;
}

/** Triggered by PAPI_stop() */
int
_vmware_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	(void) ctx;
	(void) ctrl;
	SUBDBG( "_vmware_stop %p %p...", ctx, ctrl );

#ifdef VMGUESTLIB
	glError = GuestLib_CloseHandle(glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
		fprintf(stderr, "Failed to CloseHandle: %s\n", GuestLib_GetErrorText(glError));
		//        success = FALSE;
		return EXIT_FAILURE;
	}
#endif

	return PAPI_OK;
}

/** Triggered by PAPI_read() */
int
_vmware_read( hwd_context_t *ctx, 
	      hwd_control_state_t *ctl,
	      long_long ** events, int flags )
{
	(void) ctx;
	(void) flags;
	SUBDBG( "_vmware_read... %p %d", ctx, flags );

	// update this to a for loop to account for all counters, per Vince.
	int i = 0;
	for (i=0; i<num_events; ++i) {
		((struct _vmware_control_state *)ctl)->counter[i] = _vmware_hardware_read(i);
		if (((struct _vmware_control_state *)ctl)->counter[i] < 0) {
			return EXIT_FAILURE;
		}
	}

	*events = ( ( struct _vmware_control_state *) ctl )->counter;

	return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
_vmware_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long events[] )
{
	(void) ctx;
	(void) ctrl;
	(void) events;
	SUBDBG( "_vmware_write... %p %p", ctx, ctrl );
	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */
	return PAPI_OK;
}

/** Triggered by PAPI_reset */
int
_vmware_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	(void) ctx;
	(void) ctl;

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
_vmware_shutdown( hwd_context_t *ctx )
{
	(void) ctx;
	SUBDBG( "_vmware_shutdown... %p", ctx );
	/* Last chance to clean up */

#ifdef VMGUESTLIB
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
_vmware_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

	(void) ctx;
	(void) code;
	(void) option;

	SUBDBG( "_vmware_ctl..." );

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
_vmware_set_domain( hwd_control_state_t *ctl, int domain )
{
	(void) ctl;

	int found = 0;
	SUBDBG( "_vmware_set_domain..." );
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
		.name = "vmware",
		.short_name = "vmware",
		.description = "Provide support for VMware vmguest and pseudo counters",
		.version = "5.0",
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
		.context = sizeof ( struct _vmware_context ),
		.control_state = sizeof ( struct _vmware_control_state ),
		.reg_value = sizeof ( struct _vmware_register ),
		.reg_alloc = sizeof ( struct _vmware_reg_alloc ),
	}
	,
	/* function pointers in this component */
	.init = _vmware_init,
	.init_substrate = _vmware_init_substrate,
	.init_control_state = _vmware_init_control_state,
	.start = _vmware_start,
	.stop = _vmware_stop,
	.read = _vmware_read,
	.write = _vmware_write,
	.shutdown = _vmware_shutdown,
	.ctl = _vmware_ctl,

	.update_control_state = _vmware_update_control_state,
	.set_domain = _vmware_set_domain,
	.reset = _vmware_reset,

	.ntv_enum_events = _vmware_ntv_enum_events,
	.ntv_code_to_name = _vmware_ntv_code_to_name,
	.ntv_code_to_descr = _vmware_ntv_code_to_descr,
	.ntv_code_to_info = _vmware_ntv_code_to_info,

};

