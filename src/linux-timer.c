/*
* File:    linux-timer.c
*
*/

#include "papi.h"
#include "papi_internal.h"

#if defined(HAVE_MMTIMER)
#include <sys/mman.h>
#include <linux/mmtimer.h>
#include <sys/ioctl.h>
#ifndef MMTIMER_FULLNAME
#define MMTIMER_FULLNAME "/dev/mmtimer"
#endif

static int mmdev_fd;
static unsigned long mmdev_mask;
static unsigned long mmdev_ratio;
static volatile unsigned long *mmdev_timer_addr;

        /* setup mmtimer */
int mmtimer_setup(void) {

	  unsigned long femtosecs_per_tick = 0;
	  unsigned long freq = 0;
	  int result;
	  int offset;

	  SUBDBG( "MMTIMER Opening %s\n", MMTIMER_FULLNAME );
	  if ( ( mmdev_fd = open( MMTIMER_FULLNAME, O_RDONLY ) ) == -1 ) {
	    PAPIERROR( "Failed to open MM timer %s", MMTIMER_FULLNAME );
	    return ( PAPI_ESYS );
	  }
	  SUBDBG( "MMTIMER checking if we can mmap" );
	  if ( ioctl( mmdev_fd, MMTIMER_MMAPAVAIL, 0 ) != 1 ) {
	    PAPIERROR( "mmap of MM timer unavailable" );
	    return ( PAPI_ESBSTR );
	  }
	  SUBDBG( "MMTIMER setting close on EXEC flag\n" );
	  if ( fcntl( mmdev_fd, F_SETFD, FD_CLOEXEC ) == -1 ) {
	    PAPIERROR( "Failed to fcntl(FD_CLOEXEC) on MM timer FD %d: %s",
		       mmdev_fd, strerror( errno ) );
	    return ( PAPI_ESYS );
	  }
	  SUBDBG( "MMTIMER is on FD %d, getting offset\n", mmdev_fd );
	  if ( ( offset = ioctl( mmdev_fd, MMTIMER_GETOFFSET, 0 ) ) < 0 ) {
	    PAPIERROR( "Failed to get offset of MM timer" );
	    return ( PAPI_ESYS );
	  }
	  SUBDBG( "MMTIMER has offset of %d, getting frequency\n", offset );
	  if ( ioctl( mmdev_fd, MMTIMER_GETFREQ, &freq ) == -1 ) {
	    PAPIERROR( "Failed to get frequency of MM timer" );
	    return ( PAPI_ESYS );
	  }
	  SUBDBG( "MMTIMER has frequency %lu Mhz\n", freq / 1000000 );
	  // don't know for sure, but I think this ratio is inverted
	  //     mmdev_ratio = (freq/1000000) / (unsigned long)_papi_hwi_system_info.hw_info.mhz;
          mmdev_ratio =
		  ( unsigned long ) _papi_hwi_system_info.hw_info.mhz / ( freq /
                                                                                        
									  1000000 );
          SUBDBG( "MMTIMER has a ratio of %ld to the CPU's clock, getting resolution\n",
		    mmdev_ratio );
          if ( ioctl( mmdev_fd, MMTIMER_GETRES, &femtosecs_per_tick ) == -1 ) {
		  PAPIERROR( "Failed to get femtoseconds per tick" );
		  return ( PAPI_ESYS );
          }
          SUBDBG( "MMTIMER res is %lu femtosecs/tick (10^-15s) or %f Mhz, getting valid bits\n",
	  femtosecs_per_tick, 1.0e9 / ( double ) femtosecs_per_tick );
          if ( ( result = ioctl( mmdev_fd, MMTIMER_GETBITS, 0 ) ) == -ENOSYS ) {
	     PAPIERROR( "Failed to get number of bits in MMTIMER" );
	     return ( PAPI_ESYS );
          }
          mmdev_mask = ~( 0xffffffffffffffff << result );
          SUBDBG( "MMTIMER has %d valid bits, mask 0x%16lx, getting mmaped page\n",
		    result, mmdev_mask );
          if ( ( mmdev_timer_addr =
		       ( unsigned long * ) mmap( 0, getpagesize(  ), PROT_READ,
						 MAP_PRIVATE, mmdev_fd,
						 0 ) ) == NULL ) {
	     PAPIERROR( "Failed to mmap MM timer" );
	     return ( PAPI_ESYS );
          }
          SUBDBG( "MMTIMER page is at %p, actual address is %p\n",
			mmdev_timer_addr, mmdev_timer_addr + offset );
          mmdev_timer_addr += offset;
          /* mmdev_fd should be closed and page should be unmapped in a global shutdown routine */
	  return PAPI_OK;

}

#else
int mmtimer_setup(void) { return PAPI_OK; }
#endif





/* Hardware clock functions */

/* All architectures should set HAVE_CYCLES in configure if they have these. Not all do
   so for now, we have to guard at the end of the statement, instead of the top. When
   all archs set this, this region will be guarded with:
   #if defined(HAVE_CYCLE)
   which is equivalent to
   #if !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME_REALTIME)
*/

#if defined(HAVE_MMTIMER)

static inline long long
get_cycles( void )
{
	long long tmp = 0;

        tmp = *mmdev_timer_addr & mmdev_mask;
	SUBDBG("MMTIMER is %llu, scaled %llu\n",tmp,tmp*mmdev_ratio);
        tmp *= mmdev_ratio;

	return tmp;
}
#elif defined(__ia64__)
static inline long long
get_cycles( void )
{
	long long tmp = 0;
#if defined(__INTEL_COMPILER)
	tmp = __getReg( _IA64_REG_AR_ITC );
#else
	__asm__ __volatile__( "mov %0=ar.itc":"=r"( tmp )::"memory" );
#endif
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_MONTECITO_PMU:
		tmp = tmp * 4;
		break;
	}
	return tmp;
}
#elif (defined(__i386__)||defined(__x86_64__))
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
#else
	__asm__ __volatile__( "rdtsc":"=A"( ret ): );
#endif
	return ret;
}

/* #define get_cycles _rtc ?? */
#elif defined(__sparc__)
static inline long long
get_cycles( void )
{
	register unsigned long ret asm( "g1" );

	__asm__ __volatile__( ".word 0x83410000"	/* rd %tick, %g1 */
						  :"=r"( ret ) );
	return ret;
}
#elif defined(__powerpc__)
/*
 * It's not possible to read the cycles from user space on ppc970.
 * There is a 64-bit time-base register (TBU|TBL), but its
 * update rate is implementation-specific and cannot easily be translated
 * into a cycle count.  So don't implement get_cycles for now,
 * but instead, rely on the definition of HAVE_CLOCK_GETTIME_REALTIME in
 * _papi_hwd_get_real_usec() for the needed functionality.
*/
#elif !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME_REALTIME)
#error "No get_cycles support for this architecture. Please modify perfmon.c or compile with a different timer"
#endif

long long
_linux_get_real_usec( void )
{
	long long retval;
#if defined(HAVE_CLOCK_GETTIME_REALTIME)
	{
		struct timespec foo;
		syscall( __NR_clock_gettime, HAVE_CLOCK_GETTIME_REALTIME, &foo );
		retval = ( long long ) foo.tv_sec * ( long long ) 1000000;
		retval += ( long long ) ( foo.tv_nsec / 1000 );
	}
#elif defined(HAVE_GETTIMEOFDAY)
	{
		struct timeval buffer;
		gettimeofday( &buffer, NULL );
		retval = ( long long ) buffer.tv_sec * ( long long ) 1000000;
		retval += ( long long ) ( buffer.tv_usec );
	}
#else
	retval = get_cycles(  ) / ( long long ) _papi_hwi_system_info.hw_info.mhz;
#endif
	return retval;
}

long long
_linux_get_real_cycles( void )
{
	long long retval;
#if defined(HAVE_GETTIMEOFDAY)||defined(__powerpc__)
	retval =
		_linux_get_real_usec(  ) *
		( long long ) _papi_hwi_system_info.hw_info.mhz;
#else
	retval = get_cycles(  );
#endif
	return retval;
}


#if defined(USE_PROC_PTTIMER)
static int
init_proc_thread_timer( context_t * thr_ctx )
{
	char buf[LINE_MAX];
	int fd;
	sprintf( buf, "/proc/%d/task/%d/stat", getpid(  ), mygettid(  ) );
	fd = open( buf, O_RDONLY );
	if ( fd == -1 ) {
		PAPIERROR( "open(%s)", buf );
		return PAPI_ESYS;
	}
	thr_ctx->stat_fd = fd;
	return PAPI_OK;
}
#endif

long long
_linux_get_virt_usec( const hwd_context_t * zero )
{
#ifndef USE_PROC_PTTIMER
	( void ) zero;			 /*unused */
#endif
	long long retval;
#if defined(USE_PROC_PTTIMER)
	{
		char buf[LINE_MAX];
		long long utime, stime;
		int rv, cnt = 0, i = 0;

	  again:
		rv = read( zero->stat_fd, buf, LINE_MAX * sizeof ( char ) );
		if ( rv == -1 ) {
			if ( errno == EBADF ) {
				int ret = init_proc_thread_timer( zero );
				if ( ret != PAPI_OK )
					return ret;
				goto again;
			}
			PAPIERROR( "read()" );
			return PAPI_ESYS;
		}
		lseek( zero->stat_fd, 0, SEEK_SET );

		buf[rv] = '\0';
		SUBDBG( "Thread stat file is:%s\n", buf );
		while ( ( cnt != 13 ) && ( i < rv ) ) {
			if ( buf[i] == ' ' ) {
				cnt++;
			}
			i++;
		}
		if ( cnt != 13 ) {
			PAPIERROR( "utime and stime not in thread stat file?" );
			return PAPI_ESBSTR;
		}

		if ( sscanf( buf + i, "%llu %llu", &utime, &stime ) != 2 ) {
			PAPIERROR
				( "Unable to scan two items from thread stat file at 13th space?" );
			return PAPI_ESBSTR;
		}
		retval =
			( utime +
			  stime ) * ( long long ) 1000000 / MY_VECTOR.cmp_info.clock_ticks;
	}
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
	{
		struct timespec foo;
		syscall( __NR_clock_gettime, HAVE_CLOCK_GETTIME_THREAD, &foo );
		retval = ( long long ) foo.tv_sec * ( long long ) 1000000;
		retval += ( long long ) foo.tv_nsec / 1000;
	}
#elif defined(HAVE_PER_THREAD_TIMES)
	{
		struct tms buffer;

		times( &buffer );

		SUBDBG( "user %d system %d\n", ( int ) buffer.tms_utime,
				( int ) buffer.tms_stime );
		retval =
		  ( long long ) ( ( buffer.tms_utime + buffer.tms_stime ) * 1000000 / sysconf( _SC_CLK_TCK ));

		/* NOT CLOCKS_PER_SEC as in the headers! */
	}
#elif defined(HAVE_PER_THREAD_GETRUSAGE)
	{
		struct rusage buffer;
		getrusage( RUSAGE_SELF, &buffer );
		SUBDBG( "user %d system %d\n", ( int ) buffer.tms_utime,
				( int ) buffer.tms_stime );
		retval =
			( long long ) ( buffer.ru_utime.tv_sec +
							buffer.ru_stime.tv_sec ) * ( long long ) 1000000;
		retval +=
			( long long ) ( buffer.ru_utime.tv_usec + buffer.ru_stime.tv_usec );
	}

#else
#error "No working per thread virtual timer"
#endif
	return retval;
}

long long
_linux_get_virt_cycles( const hwd_context_t * zero )
{
	return _linux_get_virt_usec( zero ) *
		( long long ) _papi_hwi_system_info.hw_info.mhz;
}


