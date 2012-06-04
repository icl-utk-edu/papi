/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    linux-nvml.c
 * @author  Kiran Kumar Kasichayanula
 *          kkasicha@utk.edu 
 * @author  James Ralph
 *          ralph@eecs.utk.edu
 * @ingroup papi_components
 *
 * @brief
 *	This is an NVML component, it demos the component interface
 *  and implements two counters nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature
 *  from Nvidia Management Library. Please refer to NVML documentation for details
 * about nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature. Power is reported in mW
 * and temperature in Celcius.
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <nvml.h>
/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#include "linux-nvml.h"

#include "nvml.h"
#include "cuda.h"
#include "cuda_runtime_api.h"


/* Declare our vector in advance */
papi_vector_t _nvml_vector;

/* upto 25 events per card how many cards per system should we allow for?! */
#define NVML_MAX_COUNTERS 100

/** Holds control flags.  Usually there's one of these per event-set.
 *    Usually this is out-of band configuration of the hardware 
 */
typedef struct nvml_control_state
{
		int num_events;
		int which_counter[NVML_MAX_COUNTERS];
		long long counter[NVML_MAX_COUNTERS];   /**< Copy of counts, holds results when stopped */
} nvml_control_state_t;

/** Holds per-thread information */
typedef struct nvml_context
{
		nvml_control_state_t state;
} nvml_context_t;

/** This table contains the native events */
static nvml_native_event_entry_t *nvml_native_table;

/** Number of devices detected at substrate_init time */
static int device_count = 0;

/** number of events in the table*/
static int num_events = 0;

static nvmlDevice_t* devices;
static int*			 features;

unsigned long long
getClockSpeed( nvmlDevice_t dev, nvmlClockType_t which_one )
{
		unsigned int ret = 0;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetClockInfo( dev, which_one, &ret );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}

		return (unsigned long long)ret;
}

		unsigned long long
getEccLocalErrors( nvmlDevice_t dev, nvmlEccBitType_t bits, int which_one)
{
		nvmlEccErrorCounts_t counts;

		nvmlReturn_t bad; 
		bad = nvmlDeviceGetDetailedEccErrors( dev, bits, NVML_VOLATILE_ECC , &counts);

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		switch ( which_one ) {
				case LOCAL_ECC_REGFILE:
						return counts.registerFile;
				case LOCAL_ECC_L1:
						return counts.l1Cache;
				case LOCAL_ECC_L2:
						return counts.l2Cache;
				case LOCAL_ECC_MEM:
						return counts.deviceMemory;
				default:
						;
		}
		return (unsigned long long)-1;
}

		unsigned long long 
getFanSpeed( nvmlDevice_t dev ) 
{
		unsigned int ret = 0;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetFanSpeed( dev, &ret );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		return (unsigned long long)ret; 
}

		unsigned long long
getMaxClockSpeed( nvmlDevice_t dev, nvmlClockType_t which_one)
{
		unsigned int ret = 0;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetClockInfo( dev, which_one, &ret );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		return (unsigned long long) ret;
}

		unsigned long long
getMemoryInfo( nvmlDevice_t dev, int which_one )
{
		nvmlMemory_t meminfo;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetMemoryInfo( dev, &meminfo );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}

		switch (which_one) {
				case MEMINFO_TOTAL_MEMORY:
						return meminfo.total;
				case MEMINFO_UNALLOCED:
						return meminfo.free;
				case MEMINFO_ALLOCED:
						return meminfo.used;
				default:
						;
		}
		return (unsigned long long)-1;
}

		unsigned long long
getPState( nvmlDevice_t dev ) 
{
		unsigned int ret = 0;
		nvmlPstates_t state = NVML_PSTATE_15;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetPerformanceState( dev, &state );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		switch ( state ) {
				case NVML_PSTATE_15:
						ret++;
				case NVML_PSTATE_14:
						ret++;
				case NVML_PSTATE_13:
						ret++;
				case NVML_PSTATE_12:
						ret++;
				case NVML_PSTATE_11:
						ret++;
				case NVML_PSTATE_10:
						ret++;
				case NVML_PSTATE_9:
						ret++;
				case NVML_PSTATE_8:
						ret++;
				case NVML_PSTATE_7:
						ret++;
				case NVML_PSTATE_6:
						ret++;
				case NVML_PSTATE_5:
						ret++;
				case NVML_PSTATE_4:
						ret++;
				case NVML_PSTATE_3:
						ret++;
				case NVML_PSTATE_2:
						ret++;
				case NVML_PSTATE_1:
						ret++;
				case NVML_PSTATE_0:
						break;
				case NVML_PSTATE_UNKNOWN:
				default:
						/* This should never happen? 
						 * The API docs just state Unknown performance state... */
						return (unsigned long long) -1;
		}

		return (unsigned long long)ret;
}

		unsigned long long
getPowerUsage( nvmlDevice_t dev )
{
		unsigned int power;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetPowerUsage( dev, &power );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		return (unsigned long long) power;
}

		unsigned long long
getTemperature( nvmlDevice_t dev )
{
		unsigned int ret = 0;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetTemperature( dev, NVML_TEMPERATURE_GPU, &ret );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		return (unsigned long long)ret;
}

		unsigned long long
getTotalEccErrors( nvmlDevice_t dev, nvmlEccBitType_t bits) 
{
		unsigned long long counts = 0;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetTotalEccErrors( dev, bits, NVML_VOLATILE_ECC , &counts);

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		return counts;
}

/* 	0 => gpu util
	1 => memory util
 */
		unsigned long long
getUtilization( nvmlDevice_t dev, int which_one )
{
		nvmlUtilization_t util;
		nvmlReturn_t bad; 
		bad = nvmlDeviceGetUtilizationRates( dev, &util );

		if ( NVML_SUCCESS != bad ) {
				SUBDBG( "something went wrong %s\n", nvmlErrorString(bad));
		}


		switch (which_one) {
				case GPU_UTILIZATION:
						return (unsigned long long) util.gpu;
				case MEMORY_UTILIZATION:
						return (unsigned long long) util.memory;
				default:
						;
		}

		return (unsigned long long) -1;
}

		static void
nvml_hardware_reset(  )
{
		/* nvmlDeviceSet* and nvmlDeviceClear* calls require root/admin access, so while 
		 * possible to implement a reset on the ECC counters, we pass */
		/* 
		   int i;
		   for ( i=0; i < device_count; i++ )
		   nvmlDeviceClearEccErrorCounts( device[i], NVML_VOLATILE_ECC ); 
		 */
}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
		static int 
nvml_hardware_read( long long *value, int which_one)
		//, nvml_context_t *ctx)
{
		nvml_native_event_entry_t *entry;
		nvmlDevice_t handle;
		int cudaIdx = -1;

		entry = &nvml_native_table[which_one];
		*value = (long long) -1;
		/* replace entry->resources with the current cuda_device->nvml device */
		cudaGetDevice( &cudaIdx );

		if ( cudaIdx < 0 || cudaIdx > device_count )
			return PAPI_EINVAL;

		/* Make sure the device we are running on has the requested event */
		if ( !HAS_FEATURE( features[cudaIdx] , entry->type) ) 
				return PAPI_EINVAL;

		handle = devices[cudaIdx];

		switch (entry->type) {
				case FEATURE_CLOCK_INFO:
						*value =  getClockSpeed( 	handle, 
										(nvmlClockType_t)entry->options.clock );
						break;
				case FEATURE_ECC_LOCAL_ERRORS:
						*value = getEccLocalErrors( 	handle, 
										(nvmlEccBitType_t)entry->options.ecc_opts.bits, 
										(int)entry->options.ecc_opts.which_one);
						break;
				case FEATURE_FAN_SPEED:
						*value = getFanSpeed( handle );
						break;
				case FEATURE_MAX_CLOCK:
						*value = getMaxClockSpeed( 	handle, 
										(nvmlClockType_t)entry->options.clock );
						break;
				case FEATURE_MEMORY_INFO:
						*value = getMemoryInfo( 	handle, 
										(int)entry->options.which_one );
						break;
				case FEATURE_PERF_STATES:
						*value = getPState( handle );
						break;
				case FEATURE_POWER:
						*value = getPowerUsage( handle );
						break;
				case FEATURE_TEMP:
						*value = getTemperature( handle );
						break;
				case FEATURE_ECC_TOTAL_ERRORS:
						*value = getTotalEccErrors( 	handle, 
										(nvmlEccBitType_t)entry->options.ecc_opts.bits );
						break;
				case FEATURE_UTILIZATION:
						*value = getUtilization( 	handle, 
										(int)entry->options.which_one );
						break;
				default:
						return PAPI_EINVAL;
		}

		return PAPI_OK;


}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
		int
_papi_nvml_init( hwd_context_t * ctx )
{
		(void) ctx;

		SUBDBG( "_papi_nvml_init %p...", ctx );

		return PAPI_OK;
}

		static int 
detectDevices( ) 
{
		nvmlReturn_t ret;
		nvmlEnableState_t mode = NVML_FEATURE_DISABLED;
		nvmlDevice_t handle;
		nvmlPciInfo_t info;

		cudaError_t cuerr;

		char busId[16];
		char name[64];
		char inforomECC[16];
		char inforomPower[16];
		char names[device_count][64];
		char nvml_busIds[device_count][16];

		float ecc_version = 0.0, power_version = 0.0;

		int i = 0,
			j = 0;
		int isTesla = 0;
		int isFermi	= 0;
		int isUnique = 1;

		unsigned int temp = 0;


		/* list of nvml pci_busids */
	for (i=0; i < device_count; i++) {
		ret = nvmlDeviceGetHandleByIndex( i, &handle );	
		if ( NVML_SUCCESS != ret ) {
			SUBDBG("nvmlDeviceGetHandleByIndex(%d) failed\n", i);
			return PAPI_ESYS;
		}

		ret = nvmlDeviceGetPciInfo( handle, &info );
		if ( NVML_SUCCESS != ret ) {
			SUBDBG("nvmlDeviceGetPciInfo() failed %s\n", nvmlErrorString(ret) );
			return PAPI_ESYS;
		}

		strncpy(nvml_busIds[i], info.busId, 16);
		
	}

	/* We want to key our list of nvmlDevice_ts by each device's cuda index */
	for (i=0; i < device_count; i++) {
			cuerr = cudaDeviceGetPCIBusId( busId, 16, i );
			if ( CUDA_SUCCESS != cuerr ) {
				SUBDBG("cudaDeviceGetPCIBusId failed.\n");
				return PAPI_ESYS;
			}
			for (j=0; j < device_count; j++ ) {
					if ( !strncmp( busId, nvml_busIds[j], 16) ) {
							ret = nvmlDeviceGetHandleByIndex(j, &devices[i] );
							if ( NVML_SUCCESS != ret )
								SUBDBG("nvmlDeviceGetHandleByIndex(%d, &devices[%d]) failed.\n", j, i);
								return PAPI_ESYS;
							break;
					}
			}	
	}

		memset(names, 0x0, device_count*64);
		/* So for each card, check whats querable */
		for (i=0; i < device_count; i++ ) {
				isTesla=0;
				isFermi=1;
				isUnique = 1;
				features[i] = 0;

				ret = nvmlDeviceGetName( devices[i], name, 64 );
				if ( NVML_SUCCESS != ret) {
					SUBDBG("nvmlDeviceGetName failed \n");
					return PAPI_ESYS;
				}

				for (j=0; j < i; j++ ) 
						if ( 0 == strncmp( name, names[j], 64 ) ) {
								/* if we have a match, and IF everything is sane, 
								 * devices with the same name eg Tesla C2075 share features */
								isUnique = 0;
								features[i] = features[j];

						}

				if ( isUnique ) {
						ret = nvmlDeviceGetInforomVersion( devices[i], NVML_INFOROM_ECC, inforomECC, 16);
						if ( NVML_SUCCESS != ret ) {
								SUBDBG("nvmlGetInforomVersion carps %s\n", nvmlErrorString(ret ) );
								isFermi = 0;
						}
						ret = nvmlDeviceGetInforomVersion( devices[i], NVML_INFOROM_POWER, inforomPower, 16);
						if ( NVML_SUCCESS != ret ) {
								/* This implies the card is older then Fermi */
								SUBDBG("nvmlGetInforomVersion carps %s\n", nvmlErrorString(ret ) );
								SUBDBG("Based upon the return to nvmlGetInforomVersion, we conclude this card is older then Fermi.\n");
								isFermi = 0;
						} 

						ecc_version = strtof(inforomECC, NULL );
						power_version = strtof( inforomPower, NULL);

						ret = nvmlDeviceGetName( devices[i], name, 64 );
						isTesla = ( NULL == strstr(name, "Tesla") ) ? 0:1;

						/* For Tesla and Quadro products from Fermi and Kepler families. */
						if ( isFermi ) {
								features[i] |= FEATURE_CLOCK_INFO;
								num_events += 3;
						}

						/* 	For Tesla and Quadro products from Fermi and Kepler families. 
							requires NVML_INFOROM_ECC 2.0 or higher for location-based counts
							requires NVML_INFOROM_ECC 1.0 or higher for all other ECC counts
							requires ECC mode to be enabled. */
						if ( isFermi ) {
								ret = nvmlDeviceGetEccMode( devices[i], &mode, NULL );
								if ( NVML_FEATURE_ENABLED == mode) {
										if ( ecc_version >= 2.0 ) {
												features[i] |= FEATURE_ECC_LOCAL_ERRORS;
												num_events += 8; /* {single bit, two bit errors} x { reg, l1, l2, memory } */
										} 
										if ( ecc_version >= 1.0 ) {
												features[i] |= FEATURE_ECC_TOTAL_ERRORS;
												num_events += 2; /* single bit errors, double bit errors */
										}
								}	
						}

						/* For all discrete products with dedicated fans */
						features[i] |= FEATURE_FAN_SPEED;
						num_events++;

						/* For Tesla and Quadro products from Fermi and Kepler families. */
						if ( isFermi ) {
								features[i] |= FEATURE_MAX_CLOCK;
								num_events += 3;
						}

						/* For all products */
						features[i] |= FEATURE_MEMORY_INFO;
						num_events += 3; /* total, free, used */

						/* For Tesla and Quadro products from the Fermi and Kepler families. */
						if ( isFermi ) {
								features[i] |= FEATURE_PERF_STATES;
								num_events++;
						}

						/* 	For "GF11x" Tesla and Quadro products from the Fermi family
							requires NVML_INFOROM_POWER 3.0 or higher
							For Tesla and Quadro products from the Kepler family
							does not require NVML_INFOROM_POWER */
						if ( isFermi ) {
								ret = nvmlDeviceGetPowerUsage( devices[i], &temp);
								if ( NVML_SUCCESS == ret ) {
										features[i] |= FEATURE_POWER;
										num_events++;
								}
						}

						/* For all discrete and S-class products. */
						features[i] |= FEATURE_TEMP;
						num_events++;

						/* For Tesla and Quadro products from the Fermi and Kepler families */
						if (isFermi) {
								features[i] |= FEATURE_UTILIZATION;
								num_events += 2;
						}

						strncpy( names[i], name, 64); 

				}
		}
		return PAPI_OK;

}

		static void
createNativeEvents( )
{
		char name[64];
		char sanitized_name[PAPI_MAX_STR_LEN];
		char names[device_count][64];

		int i, nameLen = 0, j;
		int isUnique = 1;

		nvml_native_event_entry_t* entry;
		nvmlReturn_t ret;

		nvml_native_table = (nvml_native_event_entry_t*) papi_malloc( 
						sizeof(nvml_native_event_entry_t) * num_events ); 	
		memset( nvml_native_table, 0x0, sizeof(nvml_native_event_entry_t) * num_events );
		entry = &nvml_native_table[0];

		for (i=0; i < device_count; i++ ) {
				memset( names[i], 0x0, 64 );
				isUnique = 1;
				ret = nvmlDeviceGetName( devices[i], name, 64 );

				for (j=0; j < i; j++ ) 
				{
						if ( 0 == strncmp( name, names[j], 64 ) )
								isUnique = 0;
				}

				if ( isUnique ) {
						nameLen = strlen(name);
						strncpy(sanitized_name, name, PAPI_MAX_STR_LEN );
						for (j=0; j < nameLen; j++)
								if ( ' ' == sanitized_name[j] )
										sanitized_name[j] = '_';



						if ( HAS_FEATURE( features[i], FEATURE_CLOCK_INFO ) ) {
								sprintf( entry->name, "NVML.%s.graphics_clock", sanitized_name );
								strncpy(entry->description,"Graphics clock domain (MHz).", PAPI_MAX_STR_LEN );
								entry->options.clock = NVML_CLOCK_GRAPHICS;
								entry->type = FEATURE_CLOCK_INFO;
								entry++;

								sprintf( entry->name, "NVML.%s.sm_clock", sanitized_name);
								strncpy(entry->description,"SM clock domain (MHz).", PAPI_MAX_STR_LEN);
								entry->options.clock = NVML_CLOCK_SM;
								entry->type = FEATURE_CLOCK_INFO;
								entry++;

								sprintf( entry->name, "NVML.%s.memory_clock", sanitized_name);
								strncpy(entry->description,"Memory clock domain (MHz).", PAPI_MAX_STR_LEN);
								entry->options.clock = NVML_CLOCK_MEM;
								entry->type = FEATURE_CLOCK_INFO;
								entry++;
						}	

						if ( HAS_FEATURE( features[i], FEATURE_ECC_LOCAL_ERRORS ) ) { 
								sprintf(entry->name, "NVML.%s.l1_single_ecc_errors", sanitized_name);
								strncpy(entry->description,"L1 cache single bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_SINGLE_BIT_ECC,
												.which_one = LOCAL_ECC_L1,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.l2_single_ecc_errors", sanitized_name);
								strncpy(entry->description,"L2 cache single bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_SINGLE_BIT_ECC,
												.which_one = LOCAL_ECC_L2,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.memory_single_ecc_errors", sanitized_name);
								strncpy(entry->description,"Device memory single bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_SINGLE_BIT_ECC,
												.which_one = LOCAL_ECC_MEM,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.regfile_single_ecc_errors", sanitized_name);
								strncpy(entry->description,"Register file single bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_SINGLE_BIT_ECC,
												.which_one = LOCAL_ECC_REGFILE,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.1l_double_ecc_errors", sanitized_name);
								strncpy(entry->description,"L1 cache double bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_DOUBLE_BIT_ECC,
												.which_one = LOCAL_ECC_L1,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.l2_double_ecc_errors", sanitized_name);
								strncpy(entry->description,"L2 cache double bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_DOUBLE_BIT_ECC,
												.which_one = LOCAL_ECC_L2,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.memory_double_ecc_errors", sanitized_name);
								strncpy(entry->description,"Device memory double bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_DOUBLE_BIT_ECC,
												.which_one = LOCAL_ECC_MEM,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;

								sprintf(entry->name, "NVML.%s.regfile_double_ecc_errors", sanitized_name);
								strncpy(entry->description,"Register file double bit ECC", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){
										.bits = NVML_DOUBLE_BIT_ECC,
												.which_one = LOCAL_ECC_REGFILE,
								};
								entry->type = FEATURE_ECC_LOCAL_ERRORS;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_FAN_SPEED ) ) {
								sprintf( entry->name, "NVML.%s.fan_speed", sanitized_name);
								strncpy(entry->description,"The fan speed expressed as a percent of the maximum, i.e. full speed is 100%", PAPI_MAX_STR_LEN);
								entry->type = FEATURE_FAN_SPEED;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_MAX_CLOCK ) ) {
								sprintf( entry->name, "NVML.%s.graphics_max_clock", sanitized_name);
								strncpy(entry->description,"Maximal Graphics clock domain (MHz).", PAPI_MAX_STR_LEN);
								entry->options.clock = NVML_CLOCK_GRAPHICS;
								entry->type = FEATURE_MAX_CLOCK;
								entry++;

								sprintf( entry->name, "NVML.%s.sm_max_clock", sanitized_name);
								strncpy(entry->description,"Maximal SM clock domain (MHz).", PAPI_MAX_STR_LEN);
								entry->options.clock = NVML_CLOCK_SM;
								entry->type = FEATURE_MAX_CLOCK;
								entry++;

								sprintf( entry->name, "NVML.%s.memory_max_clock", sanitized_name);
								strncpy(entry->description,"Maximal Memory clock domain (MHz).", PAPI_MAX_STR_LEN);
								entry->options.clock = NVML_CLOCK_MEM;
								entry->type = FEATURE_MAX_CLOCK;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_MEMORY_INFO ) ) {
								sprintf( entry->name, "NVML.%s.total_memory", sanitized_name);
								strncpy(entry->description,"Total installed FB memory (in bytes).", PAPI_MAX_STR_LEN);
								entry->options.which_one = MEMINFO_TOTAL_MEMORY;
								entry->type = FEATURE_MEMORY_INFO;
								entry++;

								sprintf( entry->name, "NVML.%s.unallocated_memory", sanitized_name);
								strncpy(entry->description,"Uncallocated FB memory (in bytes).", PAPI_MAX_STR_LEN);
								entry->options.which_one = MEMINFO_UNALLOCED;
								entry->type = FEATURE_MEMORY_INFO;
								entry++;

								sprintf( entry->name, "NVML.%s.allocated_memory", sanitized_name);
								strncpy(entry->description,	"Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping.", PAPI_MAX_STR_LEN);
								entry->options.which_one = MEMINFO_ALLOCED;
								entry->type = FEATURE_MEMORY_INFO;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_PERF_STATES ) ) {
								sprintf( entry->name, "NVML.%s.pstate", sanitized_name);
								strncpy(entry->description,"The performance state of the device.", PAPI_MAX_STR_LEN);
								entry->type = FEATURE_PERF_STATES;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_POWER ) ) {
								sprintf( entry->name, "NVML.%s.power", sanitized_name);
								strncpy(entry->description,"Power usage reading for the device, in miliwatts. This is the power draw for the entire board, including GPU, memory, etc.\n The reading is accurate to within a range of +/-5 watts.", PAPI_MAX_STR_LEN);
								entry->type = FEATURE_POWER;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_TEMP ) ) {
								sprintf( entry->name, "NVML.%s.temperature", sanitized_name);
								strncpy(entry->description,"Current temperature readings for the device, in degrees C.", PAPI_MAX_STR_LEN);
								entry->type = FEATURE_TEMP;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_ECC_TOTAL_ERRORS ) ) {
								sprintf( entry->name, "NVML.%s.total_ecc_errors", sanitized_name);
								strncpy(entry->description,"Total single bit errors.", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){ 
										.bits = NVML_SINGLE_BIT_ECC, 
								};
								entry->type = FEATURE_ECC_TOTAL_ERRORS;
								entry++;

								sprintf( entry->name, "NVML.%s.total_ecc_errors", sanitized_name);
								strncpy(entry->description,"Total double bit errors.", PAPI_MAX_STR_LEN);
								entry->options.ecc_opts = (struct local_ecc){ 
										.bits = NVML_DOUBLE_BIT_ECC, 
								};
								entry->type = FEATURE_ECC_TOTAL_ERRORS;
								entry++;
						}

						if ( HAS_FEATURE( features[i], FEATURE_UTILIZATION ) ) {
								sprintf( entry->name, "NVML.%s.gpu_utilization", sanitized_name);
								strncpy(entry->description,"Percent of time over the past second during which one or more kernels was executing on the GPU.", PAPI_MAX_STR_LEN);
								entry->options.which_one = GPU_UTILIZATION;
								entry->type = FEATURE_UTILIZATION;
								entry++;

								sprintf( entry->name, "NVML.%s.memory_utilization", sanitized_name);
								strncpy(entry->description,"Percent of time over the past second during which global (device) memory was being read or written.", PAPI_MAX_STR_LEN);
								entry->options.which_one = MEMORY_UTILIZATION;
								entry->type = FEATURE_UTILIZATION;
								entry++;
						}
						strncpy( names[i], name, 64); 
				}
		}
}

/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
		int
_papi_nvml_init_substrate( int cidx )
{
		nvmlReturn_t ret;
		cudaError_t cuerr;

		int cuda_count = 0;
		unsigned int nvml_count = 0;

		ret = nvmlInit();
		if ( NVML_SUCCESS != ret ) {
				strcpy(_nvml_vector.cmp_info.disabled_reason, "The NVIDIA managament library failed to initialize.");
				goto disable;
		}

		cuerr = cuInit( 0 );
		if ( CUDA_SUCCESS != cuerr ) {
				strcpy(_nvml_vector.cmp_info.disabled_reason, "The CUDA library failed to initialize.");
				goto disable;
		}

		/* Figure out the number of CUDA devices in the system */
		ret = nvmlDeviceGetCount( &nvml_count );
		if ( NVML_SUCCESS != ret ) {
				strcpy(_nvml_vector.cmp_info.disabled_reason, "Unable to get a count of devices from the NVIDIA managament library.");
				goto disable;
		}

		cuerr = cudaGetDeviceCount( &cuda_count );
		if ( CUDA_SUCCESS != cuerr ) {
				strcpy(_nvml_vector.cmp_info.disabled_reason, "Unable to get a device count from CUDA.");
				goto disable;
		}

		/* We can probably recover from this, when we're clever */
		if ( nvml_count != cuda_count ) {
				strcpy(_nvml_vector.cmp_info.disabled_reason, "Cuda and the NVIDIA managament library have different device counts.");
				goto disable;
		}

		device_count = cuda_count;

		/* A per device representation of what events are present */
		features = (int*)papi_malloc(sizeof(int) * device_count );

		/* Handles to each device */
		devices = (nvmlDevice_t*)papi_malloc(sizeof(nvmlDevice_t) * device_count);

		/* Figure out what events are supported on each card. */
		if ( (papi_errorcode = detectDevices( ) ) != PAPI_OK ) {
			papi_free(features);
			papi_free(devices);
			sprintf(_nvml_vector.cmp_info.disabled_reason, "An error occured in device feature detection, please check your NVIDIA Management Library and CUDA install." );
			goto disable;
		}

		/* The assumption is that if everything went swimmingly in detectDevices, 
			all nvml calls here should be fine. */
		createNativeEvents( );

		/* Export the total number of events available */
		_nvml_vector.cmp_info.num_native_events = num_events;

		/* Export the component id */
		_nvml_vector.cmp_info.CmpIdx = cidx;

		/* Export the number of 'counters' */
		_nvml_vector.cmp_info.num_cntrs = num_events;

		return PAPI_OK;

disable:
		_nvml_vector.cmp_info.num_cntrs = 0;
		return PAPI_OK;	
}


/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

		int
_papi_nvml_init_control_state( hwd_control_state_t * ctl )
{
		SUBDBG( "nvml_init_control_state... %p\n", ctl );
		nvml_control_state_t *nvml_ctl = ( nvml_control_state_t * ) ctl;
		memset( nvml_ctl, 0, sizeof ( nvml_control_state_t ) );

		return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
		int
_papi_nvml_update_control_state( hwd_control_state_t *ctl, 
				NativeInfo_t *native,
				int count, 
				hwd_context_t *ctx )
{
		int i, index;

		nvml_control_state_t *nvml_ctl = ( nvml_control_state_t * ) ctl;   
		(void) ctx;

		SUBDBG( "_papi_nvml_update_control_state %p %p...", ctl, ctx );

		/* if no events, return */
		if (count==0) return PAPI_OK;

		for( i = 0; i < count; i++ ) {
				index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
				nvml_ctl->which_counter[i]=index;
				/* We have no constraints on event position, so any event */
				/* can be in any slot.                                    */
				native[i].ni_position = i;
		}
		nvml_ctl->num_events=count;
		return PAPI_OK;
}
/** Triggered by PAPI_start() */
		int
_papi_nvml_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

		(void) ctx;
		(void) ctl;

		SUBDBG( "nvml_start %p %p...", ctx, ctl );
		/* anything that would need to be set at counter start time */

		/* reset */
		/* start the counting */

		return PAPI_OK;
}


/** Triggered by PAPI_stop() */
		int
_papi_nvml_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
		int i;
		(void) ctx;
		(void) ctl;
		int ret;
		SUBDBG( "nvml_stop %p %p...", ctx, ctl );

		nvml_control_state_t* nvml_ctl = ( nvml_control_state_t*) ctl;

		for (i=0;i<nvml_ctl->num_events;i++) {
				if ( PAPI_OK != 
								( ret = nvml_hardware_read( &nvml_ctl->counter[i], 
															nvml_ctl->which_counter[i]) ))
						return ret;

		}

		return PAPI_OK;
}


/** Triggered by PAPI_read() */
		int
_papi_nvml_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
				long long **events, int flags )
{

		(void) ctx;
		(void) flags;
		int i;
		int ret;
		nvml_control_state_t* nvml_ctl = ( nvml_control_state_t*) ctl;   

		SUBDBG( "nvml_read... %p %d", ctx, flags );

		for (i=0;i<nvml_ctl->num_events;i++) {
				if ( PAPI_OK != 
								( ret = nvml_hardware_read( &nvml_ctl->counter[i], 
															nvml_ctl->which_counter[i]) ))
						return ret;

		}
		/* return pointer to the values we read */
		*events = nvml_ctl->counter;	
		return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
		int
_papi_nvml_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
				long long *events )
{

		(void) ctx;
		(void) ctl;
		(void) events;

		SUBDBG( "nvml_write... %p %p", ctx, ctl );

		/* You can change ECC mode and compute exclusivity modes on the cards */
		/* But I don't see this as a function of a PAPI component at this time */
		/* All implementation issues aside. */
		return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
		int
_papi_nvml_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
		(void) ctx;
		(void) ctl;

		SUBDBG( "nvml_reset ctx=%p ctrl=%p...", ctx, ctl );

		/* Reset the hardware */
		nvml_hardware_reset(  );

		return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
		int
_papi_nvml_shutdown_substrate()
{

		SUBDBG( "nvml_shutdown_substrate..." );

		papi_free(nvml_native_table);
		papi_free(devices);

		nvmlShutdown();

		device_count = 0;
		num_events = 0;

		return PAPI_OK;
}

/** Called at thread shutdown */
		int
_papi_nvml_shutdown( hwd_context_t *ctx )
{

		(void) ctx;

		SUBDBG( "nvml_shutdown... %p", ctx );

		/* Last chance to clean up thread */

		return PAPI_OK;
}



/** This function sets various options in the substrate
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
		int
_papi_nvml_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{

		(void) ctx;
		(void) code;
		(void) option;

		SUBDBG( "nvml_ctl..." );

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
_papi_nvml_set_domain( hwd_control_state_t * cntrl, int domain )
{
		(void) cntrl;

		int found = 0;
		SUBDBG( "nvml_set_domain..." );

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
		if ( !found )
				return ( PAPI_EINVAL );

		return PAPI_OK;
}


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
		int
_papi_nvml_ntv_enum_events( unsigned int *EventCode, int modifier )
{
		int cidx,index;

		/* Get our component index number, this can change depending */
		/* on how PAPI was configured.                               */

		cidx = PAPI_COMPONENT_INDEX( *EventCode );

		switch ( modifier ) {

				/* return EventCode of first event */
				case PAPI_ENUM_FIRST:
						/* return the first event that we support */

						*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
						return PAPI_OK;

						/* return EventCode of next available event */
				case PAPI_ENUM_EVENTS:
						index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

						/* Make sure we are in range */
						if ( index < num_events - 1 ) {

								/* This assumes a non-sparse mapping of the events */
								*EventCode = *EventCode + 1;
								return PAPI_OK;
						} else {
								return PAPI_ENOEVNT;
						}
						break;

				default:
						return PAPI_EINVAL;
		}

		return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
		int
_papi_nvml_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
		int index;

		index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		/* Make sure we are in range */
		if (index >= num_events) return PAPI_ENOEVNT;

		strncpy( name, nvml_native_table[index].name, len );

		return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
		int
_papi_nvml_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
		int index;
		index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if (index >= num_events) return PAPI_ENOEVNT;

		strncpy( descr, nvml_native_table[index].description, len );

		return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _nvml_vector = {
		.cmp_info = {
				/* default component information */
				/* (unspecified values are initialized to 0) */
				/* we explicitly set them to zero in this nvml component */
				/* to show what settings are available            */

				.name = " linux-nvml.c",
				.version = "1.0",
				.support_version = "n/a",
				.kernel_version = "n/a",
				.CmpIdx = 0,            /* set by init_substrate */
				.num_cntrs = 0, 		/* set by init_substrate */
				.num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				.num_preset_events = 0,
				.num_native_events = 0, /* set by init_substrate */
				.default_domain = PAPI_DOM_USER,
				.available_domains = PAPI_DOM_USER,
				.default_granularity = PAPI_GRN_THR,
				.available_granularities = PAPI_GRN_THR,
				.hardware_intr_sig = PAPI_INT_SIGNAL,


				/* component specific cmp_info initializations */
				.hardware_intr = 0,
				.precise_intr = 0,
				.posix1b_timers = 0,
				.kernel_profile = 0,
				.kernel_multiplex = 0,
				.data_address_range = 0,
				.instr_address_range = 0,
				.fast_counter_read = 0,
				.fast_real_timer = 0,
				.fast_virtual_timer = 0,
				.attach = 0,
				.attach_must_ptrace = 0,
				.profile_ear = 0,
				.cntr_groups = 0,
				.cntr_umasks = 0,
				.cntr_IEAR_events = 0,
				.cntr_DEAR_events = 0,
				.cntr_OPCM_events = 0,
				.cpu = 0,
				.inherit = 0,
		},

		/* sizes of framework-opaque component-private structures */
		.size = {
				.context = sizeof ( nvml_context_t ),
				.control_state = sizeof ( nvml_control_state_t ),
				.reg_value = sizeof ( nvml_register_t ),
				//		.reg_alloc = sizeof ( nvml_reg_alloc_t ),
		},

		/* function pointers */

		/* Used for general PAPI interactions */
		.start =                _papi_nvml_start,
		.stop =                 _papi_nvml_stop,
		.read =                 _papi_nvml_read,
		.reset =                _papi_nvml_reset,	
		.write =                _papi_nvml_write,
		.init_substrate =       _papi_nvml_init_substrate,	
		.init =                 _papi_nvml_init,
		.init_control_state =   _papi_nvml_init_control_state,
		.update_control_state = _papi_nvml_update_control_state,	
		.ctl =                  _papi_nvml_ctl,	
		.shutdown =             _papi_nvml_shutdown,
		.shutdown_substrate =   _papi_nvml_shutdown_substrate,
		.set_domain =           _papi_nvml_set_domain,
		.cleanup_eventset =     NULL,
		/* called in add_native_events() */
		.allocate_registers =   NULL,

		/* Used for overflow/profiling */
		.dispatch_timer =       NULL,
		.get_overflow_address = NULL,
		.stop_profiling =       NULL,
		.set_overflow =         NULL,
		.set_profile =          NULL,

		/* OS related functions 
		   .get_real_cycles =      NULL,
		   .get_real_usec =        NULL,
		   .get_virt_cycles =      NULL,
		   .get_virt_usec =        NULL,
		   .update_shlib_info =    NULL,
		   .get_system_info =      NULL,
		   .get_memory_info =      NULL,
		   .get_dmem_info =        NULL,*/

		/* bipartite map counter allocation? 
		   .bpt_map_avail =        NULL,
		   .bpt_map_set =          NULL,
		   .bpt_map_exclusive =    NULL,
		   .bpt_map_shared =       NULL,
		   .bpt_map_preempt =      NULL,
		   .bpt_map_update =       NULL,*/

		/* ??? */
		.user =                 NULL,

		/* Name Mapping Functions */
		.ntv_enum_events =   _papi_nvml_ntv_enum_events,
		.ntv_name_to_code  = NULL,
		.ntv_code_to_name =  _papi_nvml_ntv_code_to_name,
		.ntv_code_to_descr = _papi_nvml_ntv_code_to_descr,

		/* These are only used by _papi_hwi_get_native_event_info() */
		/* Which currently only uses the info for printing native   */
		/* event info, not for any sort of internal use.            */
		//	.ntv_code_to_bits =  NULL,
		//	.ntv_bits_to_info =  NULL,


		/* Old and should be removed */
		//	.add_prog_event =       NULL,


};

