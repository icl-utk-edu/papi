/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    windows-memory.c
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h" /* papi_calloc() */

#include "x86_cpuid_info.h"

#include SUBSTRATE

#include <Psapi.h>
int
_windows_get_dmem_info( PAPI_dmem_info_t * d )
{

	HANDLE proc = GetCurrentProcess(  );
	PROCESS_MEMORY_COUNTERS cntr;
	SYSTEM_INFO SystemInfo;			   // system information structure  

	GetSystemInfo( &SystemInfo );
	GetProcessMemoryInfo( proc, &cntr, sizeof ( cntr ) );

	d->pagesize = SystemInfo.dwPageSize;
	d->size =
		( cntr.WorkingSetSize - cntr.PagefileUsage ) / SystemInfo.dwPageSize;
	d->resident = cntr.WorkingSetSize / SystemInfo.dwPageSize;
	d->high_water_mark = cntr.PeakWorkingSetSize / SystemInfo.dwPageSize;

	return PAPI_OK;
}

/*
 * Architecture-specific cache detection code 
 */


static int
x86_get_memory_info( PAPI_hw_info_t * hw_info )
{
	int retval = PAPI_OK;

	switch ( hw_info->vendor ) {
	case PAPI_VENDOR_AMD:
	case PAPI_VENDOR_INTEL:
		retval = _x86_cache_info( &hw_info->mem_hierarchy );
		break;
	default:
		PAPIERROR( "Unknown vendor in memory information call for x86." );
		return PAPI_ESBSTR;
	}
	return retval;
}


int
_windows_get_memory_info( PAPI_hw_info_t * hwinfo, int cpu_type )
{
	( void ) cpu_type;		 /*unused */
	int retval = PAPI_OK;

	x86_get_memory_info( hwinfo );

	return retval;
}

int
_windows_update_shlib_info( papi_mdi_t *mdi )
{
	return PAPI_OK;
}
