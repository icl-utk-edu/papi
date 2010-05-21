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
#include "cycle.h"
#ifndef _WIN32
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#endif

extern papi_vector_t MY_VECTOR;

static long long
_any_get_real_usec( void )
{
#ifdef _WIN32
	LARGE_INTEGER PerformanceCount, Frequency;
	QueryPerformanceCounter( &PerformanceCount );
	QueryPerformanceFrequency( &Frequency );
	return ( ( PerformanceCount.QuadPart * 1000000 ) / Frequency.QuadPart );
#else
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return ( tv.tv_sec * 1000000 ) + tv.tv_usec;
#endif
}

static long long
_any_get_real_cycles( void )
{
	float usec = ( float ) _any_get_real_usec(  );
	float cyc = usec * _papi_hwi_system_info.hw_info.mhz;
	return ( long long ) cyc;
}

static long long
_any_get_virt_usec( const hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	long long retval;
#if ((defined _BGL) || (defined __bgp__))
	struct rusage ruse;
	getrusage( RUSAGE_SELF, &ruse );
	retval =
		( long long ) ( ruse.ru_utime.tv_sec * 1000000 +
						ruse.ru_utime.tv_usec );
#elif _WIN32
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
	struct tms buffer;
	times( &buffer );
	retval =
		( long long ) buffer.tms_utime * ( long long ) ( 1000000 /
														 sysconf
														 ( _SC_CLK_TCK ) );
#endif
	return ( retval );
}

static long long
_any_get_virt_cycles( const hwd_context_t * ctx )
{
	float usec = ( float ) _any_get_virt_usec( ctx );
	float cyc = usec * _papi_hwi_system_info.hw_info.mhz;
	return ( long long ) cyc;
}

papi_vector_t _any_vector = {
	.get_real_usec = _any_get_real_usec,
	.get_real_cycles = _any_get_real_cycles,
	.get_virt_cycles = _any_get_virt_cycles,
	.get_virt_usec = _any_get_virt_usec
};
