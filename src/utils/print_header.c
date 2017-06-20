#include <stdio.h>

#include "papi.h"

/*  Support routine to display header information to the screen
	from the hardware info data structure. The same code was duplicated
	in a number of tests and utilities. Seems to make sense to refactor.
	This may not be the best place for it to live, but it works for now.
 */
int
papi_print_header( char *prompt, const PAPI_hw_info_t ** hwinfo )
{
	int cnt, mpx;

	if ( ( *hwinfo = PAPI_get_hardware_info(  ) ) == NULL ) {
   		return PAPI_ESYS;
	}

	printf( "%s", prompt );
	printf
		( "--------------------------------------------------------------------------------\n" );
	printf( "PAPI Version             : %d.%d.%d.%d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ),
			PAPI_VERSION_INCREMENT( PAPI_VERSION ) );
	printf( "Vendor string and code   : %s (%d)\n", ( *hwinfo )->vendor_string,
			( *hwinfo )->vendor );
	printf( "Model string and code    : %s (%d)\n", ( *hwinfo )->model_string,
			( *hwinfo )->model );
	printf( "CPU Revision             : %f\n", ( *hwinfo )->revision );
	if ( ( *hwinfo )->cpuid_family > 0 )
		printf
			( "CPUID Info               : Family: %d  Model: %d  Stepping: %d\n",
			  ( *hwinfo )->cpuid_family, ( *hwinfo )->cpuid_model,
			  ( *hwinfo )->cpuid_stepping );
	printf( "CPU Max Megahertz        : %d\n", ( *hwinfo )->cpu_max_mhz );
	printf( "CPU Min Megahertz        : %d\n", ( *hwinfo )->cpu_min_mhz );
	if ( ( *hwinfo )->threads > 0 )
		printf( "Hdw Threads per core     : %d\n", ( *hwinfo )->threads );
	if ( ( *hwinfo )->cores > 0 )
		printf( "Cores per Socket         : %d\n", ( *hwinfo )->cores );
	if ( ( *hwinfo )->sockets > 0 )
		printf( "Sockets                  : %d\n", ( *hwinfo )->sockets );
	if ( ( *hwinfo )->nnodes > 0 )
		printf( "NUMA Nodes               : %d\n", ( *hwinfo )->nnodes );
	printf( "CPUs per Node            : %d\n", ( *hwinfo )->ncpu );
	printf( "Total CPUs               : %d\n", ( *hwinfo )->totalcpus );
	printf( "Running in a VM          : %s\n", ( *hwinfo )->virtualized?
		"yes":"no");
	if ( (*hwinfo)->virtualized) {
           printf( "VM Vendor:               : %s\n", (*hwinfo)->virtual_vendor_string);
	}
	cnt = PAPI_get_opt( PAPI_MAX_HWCTRS, NULL );
	mpx = PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL );
	if ( cnt >= 0 ) {
		printf( "Number Hardware Counters : %d\n",cnt );
	} else {
		printf( "Number Hardware Counters : PAPI error %d: %s\n", cnt, PAPI_strerror(cnt));
	}
	if ( mpx >= 0 ) {
		printf( "Max Multiplex Counters   : %d\n", mpx );
	} else {
		printf( "Max Multiplex Counters   : PAPI error %d: %s\n", mpx, PAPI_strerror(mpx));
	}
	printf
		( "--------------------------------------------------------------------------------\n" );
	printf( "\n" );
	return PAPI_OK;
}

