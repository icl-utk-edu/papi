/*
* File:    windows-timer.c
*
*/

#include "papi.h"
#include "papi_internal.h"

/* Hardware clock functions */

/* All architectures should set HAVE_CYCLES in configure if they have these. Not all do
   so for now, we have to guard at the end of the statement, instead of the top. When
   all archs set this, this region will be guarded with:
   #if defined(HAVE_CYCLE)
   which is equivalent to
   #if !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME_REALTIME)
*/

static inline long long
get_cycles( void )
{
	long long ret = 0;
#ifdef __x86_64__
	do {
		unsigned int a, d;
		asm volatile ( "rdtsc":"=a" ( a ), "=d"( d ) );
		( ret ) = ( ( long long ) a ) | ( ( ( long long ) d ) << 32 );
	}
	while ( 0 );
#elif defined WIN32
        ret = __rdtsc(  );
#else
	__asm__ __volatile__( "rdtsc":"=A"( ret ): );
#endif
	return ret;
}

long long
_windows_get_real_usec( void )
{
	long long retval;

	retval = get_cycles(  ) / ( long long ) _papi_hwi_system_info.hw_info.mhz;

	return retval;
}

long long
_windows_get_real_cycles( void )
{
	long long retval;

	retval = get_cycles(  );

	return retval;
}

long long
_windows_get_virt_usec( void )
{

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
                                                                User.dwHighDate\
				      Time ) ) << 32 )
		  + Kernel.dwLowDateTime + User.dwLowDateTime;
                return ( virt / 1000 );
        } else
	  return ( PAPI_ESBSTR );

	return retval;
}

