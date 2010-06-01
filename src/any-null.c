/* 
* File:    any-null.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@eecs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@eecs.utk.edu
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
*/

#include "any-null.h"
#include "papi_internal.h"
#include "papi_vector.h"
#ifndef _WIN32
#include "cycle.h"
#include <string.h>
#endif
#ifdef _AIX
#include <pmapi.h>
#endif
#ifdef __bgp__
#include <common/bgp_personality_inlines.h>
#include <common/bgp_personality.h>
#include <spi/kernel_interface.h>
#endif

extern papi_vector_t MY_VECTOR;
static int frequency = -1;

#ifndef _WIN32
static char *
search_cpu_info( FILE * f, char *search_str, char *line )
{
	char *s;

	while ( fgets( line, 256, f ) != NULL ) {
		if ( strstr( line, search_str ) != NULL ) {
			/* ignore all characters in line up to : */
			for ( s = line; *s && ( *s != ':' ); ++s );
			if ( *s )
				return s;
		}
	}
	return NULL;
}
#endif

void
set_freq(  )
{
#if defined(_AIX)
	frequency = ( int ) pm_cycles(  ) / 1000000;
#elif defined(__bgp__)
	_BGP_Personality_t bgp;
	frequency = BGP_Personality_clockMHz( &bgp );
#elif defined(_WIN32)
#else
	char maxargs[PAPI_HUGE_STR_LEN], *s;
	float mhz = 0.0;
	FILE *f;

	if ( ( f = fopen( "/proc/cpuinfo", "r" ) ) != NULL ) {
		rewind( f );
		s = search_cpu_info( f, "clock", maxargs );

		if ( !s ) {
			rewind( f );
			s = search_cpu_info( f, "cpu MHz", maxargs );
		}

		if ( s )
			sscanf( s + 1, "%f", &mhz );

		frequency = ( int ) mhz;
		fclose( f );
	}
#endif
}

static long long
_any_get_real_usec( void )
{
#if defined(__bgp__)
	return ( long long ) ( _bgp_GetTimeBase(  ) / frequency );
#elif defined(_WIN32)
	/*** NOTE: This differs from the code in win32.c ***/
	LARGE_INTEGER PerformanceCount, Frequency;
	QueryPerformanceCounter( &PerformanceCount );
	QueryPerformanceFrequency( &Frequency );
	return ( ( PerformanceCount.QuadPart * 1000000 ) / Frequency.QuadPart );
#else
	return ( long long ) getticks(  ) / frequency;
#endif
}

static long long
_any_get_real_cycles( void )
{
#if defined(__bgp__)
	return _bgp_GetTimeBase(  );
#elif defined(_WIN32)
#else
	return ( long long ) getticks(  );
#endif
}

static long long
_any_get_virt_usec( const hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
#if defined(__bgp__)
	return ( long long ) ( _bgp_GetTimeBase(  ) / frequency );
#elif defined(_WIN32)
	/*** NOTE: This differs from the code in win32.c ***/
	long long retval;
	HANDLE p;
	BOOL ret;
	FILETIME Creation, Exit, Kernel, User;
	long long virt;
	p = GetCurrentProcess(  );
	ret = GetProcessTimes( p, &Creation, &Exit, &Kernel, &User );
	if ( ret ) {
		virt =
			( ( ( long long ) ( Kernel.dwHighDateTime +
								User.dwHighDateTime ) ) << 32 )
			+ Kernel.dwLowDateTime + User.dwLowDateTime;
		retval = virt / 1000;
	} else
		return ( PAPI_ESBSTR );
#else
	return ( long long ) getticks(  ) / frequency;
#endif
}

static long long
_any_get_virt_cycles( const hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
#if defined(__bgp__)
	return _bgp_GetTimeBase(  );
#elif defined(_WIN32)
#else
	return ( long long ) getticks(  );
#endif
}

papi_vector_t _any_vector = {
	/*Developer's Note: The size data members are set to non-zero values in case 
	   the framework uses them as the size argument to malloc */
	.size = {
			 .context = sizeof ( int ),
			 .control_state = sizeof ( int ),
			 .reg_value = sizeof ( int ),
			 .reg_alloc = sizeof ( int )
			 }
	,
	.get_real_usec = _any_get_real_usec,
	.get_real_cycles = _any_get_real_cycles,
	.get_virt_cycles = _any_get_virt_cycles,
	.get_virt_usec = _any_get_virt_usec
};
