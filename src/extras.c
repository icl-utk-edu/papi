/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    extras.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Haihang You
*          you@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
*/

/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. */

/* It also contains a new section at the end with Windows routines
 to emulate standard stuff found in Unix/Linux, but not Windows! */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"
#include "threads.h"

#if (!defined(HAVE_FFSLL) || defined(__bgp__))
int ffsll( long long lli );
#endif

/****************/
/* BEGIN LOCALS */
/****************/

static unsigned int _rnum = DEADBEEF;

/**************/
/* END LOCALS */
/**************/

inline_static unsigned short
random_ushort( void )
{
	return ( unsigned short ) ( _rnum = 1664525 * _rnum + 1013904223 );
}


/* compute the amount by which to increment the bucket.
   value is the current value of the bucket
   this routine is used by all three profiling cases
   it is inlined for speed
*/
inline_static int
profil_increment( long long value,
				  int flags, long long excess, long long threshold )
{
	int increment = 1;

	if ( flags == PAPI_PROFIL_POSIX ) {
		return ( 1 );
	}

	if ( flags & PAPI_PROFIL_RANDOM ) {
		if ( random_ushort(  ) <= ( USHRT_MAX / 4 ) )
			return ( 0 );
	}

	if ( flags & PAPI_PROFIL_COMPRESS ) {
		/* We're likely to ignore the sample if buf[address] gets big. */
		if ( random_ushort(  ) < value ) {
			return ( 0 );
		}
	}

	if ( flags & PAPI_PROFIL_WEIGHTED ) {	/* Increment is between 1 and 255 */
		if ( excess <= ( long long ) 1 )
			increment = 1;
		else if ( excess > threshold )
			increment = 255;
		else {
			threshold = threshold / ( long long ) 255;
			increment = ( int ) ( excess / threshold );
		}
	}
	return ( increment );
}


static void
posix_profil( caddr_t address, PAPI_sprofil_t * prof,
			  int flags, long long excess, long long threshold )
{
	unsigned short *buf16;
	unsigned int *buf32;
	unsigned long long *buf64;
	unsigned long indx;
	unsigned long long lloffset;

	/* SPECIAL CASE: if starting address is 0 and scale factor is 2
	   then all counts go into first bin.
	 */
	if ( ( prof->pr_off == 0 ) && ( prof->pr_scale == 0x2 ) )
		indx = 0;
	else {
		/* compute the profile buffer offset by:
		   - subtracting the profiling base address from the pc address
		   - multiplying by the scaling factor
		   - dividing by max scale (65536, or 2^^16) 
		   - dividing by implicit 2 (2^^1 for a total of 2^^17), for even addresses
		   NOTE: 131072 is a valid scale value. It produces byte resolution of addresses
		 */
		lloffset =
			( unsigned long long ) ( ( address - prof->pr_off ) *
									 prof->pr_scale );
		indx = ( unsigned long ) ( lloffset >> 17 );
	}

	/* confirm addresses within specified range */
	if ( address >= prof->pr_off ) {
		/* test first for 16-bit buckets; this should be the fast case */
		if ( flags & PAPI_PROFIL_BUCKET_16 ) {
			if ( ( indx * sizeof ( short ) ) < prof->pr_size ) {
				buf16 = prof->pr_base;
				buf16[indx] =
					( unsigned short ) ( ( unsigned short ) buf16[indx] +
										 profil_increment( buf16[indx], flags,
														   excess,
														   threshold ) );
				PRFDBG( "posix_profil_16() bucket %lu = %u\n", indx,
						buf16[indx] );
			}
		}
		/* next, look for the 32-bit case */
		else if ( flags & PAPI_PROFIL_BUCKET_32 ) {
			if ( ( indx * sizeof ( int ) ) < prof->pr_size ) {
				buf32 = prof->pr_base;
				buf32[indx] = ( unsigned int ) buf32[indx] +
					( unsigned int ) profil_increment( buf32[indx], flags,
													   excess, threshold );
				PRFDBG( "posix_profil_32() bucket %lu = %u\n", indx,
						buf32[indx] );
			}
		}
		/* finally, fall through to the 64-bit case */
		else {
			if ( ( indx * sizeof ( long long ) ) < prof->pr_size ) {
				buf64 = prof->pr_base;
				buf64[indx] = ( unsigned long long ) buf64[indx] +
					( unsigned long long ) profil_increment( ( long long )
															 buf64[indx], flags,
															 excess,
															 threshold );
				PRFDBG( "posix_profil_64() bucket %lu = %lld\n", indx,
						buf64[indx] );
			}
		}
	}
}

void
_papi_hwi_dispatch_profile( EventSetInfo_t * ESI, caddr_t pc,
							long long over, int profile_index )
{
	EventSetProfileInfo_t *profile = &ESI->profile;
	PAPI_sprofil_t *sprof;
	caddr_t offset = 0;
	caddr_t best_offset = 0;
	int count;
	int best_index = -1;
	int i;

	PRFDBG( "handled IP 0x%p\n", pc );

	sprof = profile->prof[profile_index];
	count = profile->count[profile_index];

	for ( i = 0; i < count; i++ ) {
		offset = sprof[i].pr_off;
		if ( ( offset < pc ) && ( offset > best_offset ) ) {
			best_index = i;
			best_offset = offset;
		}
	}

	if ( best_index == -1 )
		best_index = 0;

	posix_profil( pc, &sprof[best_index], profile->flags, over,
				  profile->threshold[profile_index] );
}

/* if isHardware is true, then the processor is using hardware overflow,
   else it is using software overflow. Use this parameter instead of 
   _papi_hwi_system_info.supports_hw_overflow is in CRAY some processors
   may use hardware overflow, some may use software overflow.

   overflow_bit: if the substrate can get the overflow bit when overflow
                 occurs, then this should be passed by the substrate;

   If both genOverflowBit and isHardwareSupport are true, that means
     the substrate doesn't know how to get the overflow bit from the
     kernel directly, so we generate the overflow bit in this function 
    since this function can access the ESI->overflow struct;
   (The substrate can only set genOverflowBit parameter to true if the
     hardware doesn't support multiple hardware overflow. If the
     substrate supports multiple hardware overflow and you don't know how 
     to get the overflow bit, then I don't know how to deal with this 
     situation).
*/

int
_papi_hwi_dispatch_overflow_signal( void *papiContext, caddr_t address,
				   int *isHardware, long long overflow_bit,
				   int genOverflowBit, ThreadInfo_t ** t,
				   int cidx )
{
	int retval, event_counter, i, overflow_flag, pos;
	int papi_index, j;
	int profile_index = 0;
	long long overflow_vector;

	long long temp[_papi_hwd[cidx]->cmp_info.num_cntrs], over;
	long long latest = 0;
	ThreadInfo_t *thread;
	EventSetInfo_t *ESI;
	_papi_hwi_context_t *ctx = ( _papi_hwi_context_t * ) papiContext;

	OVFDBG( "enter\n" );

	if ( *t )
		thread = *t;
	else
		*t = thread = _papi_hwi_lookup_thread( 0 );

	if ( thread != NULL ) {
		ESI = thread->running_eventset[cidx];

		if ( ( ESI == NULL ) || ( ( ESI->state & PAPI_OVERFLOWING ) == 0 ) ) {
			OVFDBG( "Either no eventset or eventset not set to overflow.\n" );
#ifdef ANY_THREAD_GETS_SIGNAL
			_papi_hwi_broadcast_signal( thread->tid );
#endif
			return ( PAPI_OK );
		}

		if ( ESI->CmpIdx != cidx )
			return ( PAPI_ENOCMP );

		if ( ESI->master != thread ) {
			PAPIERROR
				( "eventset->thread 0x%lx vs. current thread 0x%lx mismatch",
				  ESI->master, thread );
			return ( PAPI_EBUG );
		}

		if ( isHardware ) {
			if ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) {
				ESI->state |= PAPI_PAUSED;
				*isHardware = 1;
			} else
				*isHardware = 0;
		}
		/* Get the latest counter value */
		event_counter = ESI->overflow.event_counter;

		overflow_flag = 0;
		overflow_vector = 0;

		if ( !( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) ) {
			retval = _papi_hwi_read( thread->context[cidx], ESI, ESI->sw_stop );
			if ( retval < PAPI_OK )
				return ( retval );
			for ( i = 0; i < event_counter; i++ ) {
				papi_index = ESI->overflow.EventIndex[i];
				latest = ESI->sw_stop[papi_index];
				temp[i] = -1;

				if ( latest >= ( long long ) ESI->overflow.deadline[i] ) {
					OVFDBG
						( "dispatch_overflow() latest %lld, deadline %lld, threshold %d\n",
						  latest, ESI->overflow.deadline[i],
						  ESI->overflow.threshold[i] );
					pos = ESI->EventInfoArray[papi_index].pos[0];
					overflow_vector ^= ( long long ) 1 << pos;
					temp[i] = latest - ESI->overflow.deadline[i];
					overflow_flag = 1;
					/* adjust the deadline */
					ESI->overflow.deadline[i] =
						latest + ESI->overflow.threshold[i];
				}
			}
		} else if ( genOverflowBit ) {
			/* we had assumed the overflow event can't be derived event */
			papi_index = ESI->overflow.EventIndex[0];

			/* suppose the pos is the same as the counter number
			 * (this is not true in Itanium, but itanium doesn't 
			 * need us to generate the overflow bit
			 */
			pos = ESI->EventInfoArray[papi_index].pos[0];
			overflow_vector = ( long long ) 1 << pos;
		} else
			overflow_vector = overflow_bit;

		if ( ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) || overflow_flag ) {
			if ( ESI->state & PAPI_PROFILING ) {
				int k = 0;
				while ( overflow_vector ) {
					i = ffsll( overflow_vector ) - 1;
					for ( j = 0; j < event_counter; j++ ) {
						papi_index = ESI->overflow.EventIndex[j];
						/* This loop is here ONLY because Pentium 4 can have tagged *
						 * events that contain more than one counter without being  *
						 * derived. You've gotta scan all terms to make sure you    *
						 * find the one to profile. */
						for ( k = 0, pos = 0; k < MAX_COUNTER_TERMS && pos >= 0;
							  k++ ) {
							pos = ESI->EventInfoArray[papi_index].pos[k];
							if ( i == pos ) {
								profile_index = j;
								goto foundit;
							}
						}
					}
					if ( j == event_counter ) {
						PAPIERROR
							( "BUG! overflow_vector is 0, dropping interrupt" );
						return ( PAPI_EBUG );
					}

				  foundit:
					if ( ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) )
						over = 0;
					else
						over = temp[profile_index];
					_papi_hwi_dispatch_profile( ESI, address, over,
												profile_index );
					overflow_vector ^= ( long long ) 1 << i;
				}
				/* do not use overflow_vector after this place */
			} else {
				ESI->overflow.handler( ESI->EventSetIndex, ( void * ) address,
									   overflow_vector, ctx->ucontext );
			}
		}
		ESI->state &= ~( PAPI_PAUSED );
	}
#ifdef ANY_THREAD_GETS_SIGNAL
	else {
		OVFDBG( "I haven't been noticed by PAPI before\n" );
		_papi_hwi_broadcast_signal( ( *_papi_hwi_thread_id_fn ) (  ) );
	}
#endif
	return ( PAPI_OK );
}

#ifdef _WIN32

volatile int _papi_hwi_using_signal = 0;
static MMRESULT wTimerID;			   // unique ID for referencing this timer
static UINT wTimerRes;				   // resolution for this timer

int
_papi_hwi_start_timer( int ns )
{
	int retval = PAPI_OK;
	int milliseconds = ns / 1000000;
	TIMECAPS tc;
	DWORD threadID;

	// get the timer resolution capability on this system
	if ( timeGetDevCaps( &tc, sizeof ( TIMECAPS ) ) != TIMERR_NOERROR )
		return ( PAPI_ESYS );

	// get the ID of the current thread to read the context later
	// NOTE: Use of this code is restricted to W2000 and later...
	threadID = GetCurrentThreadId(  );

	// set the minimum usable resolution of the timer
	wTimerRes = min( max( tc.wPeriodMin, 1 ), tc.wPeriodMax );
	timeBeginPeriod( wTimerRes );

	// initialize a periodic timer
	//    triggering every (milliseconds) 
	//    and calling (_papi_hwd_timer_callback())
	//    with no data
	wTimerID = timeSetEvent( milliseconds, wTimerRes,
							 ( LPTIMECALLBACK ) _papi_hwd_timer_callback,
							 threadID, TIME_PERIODIC );
	if ( !wTimerID )
		return PAPI_ESYS;

	return ( retval );
}

int
_papi_hwi_start_signal( int signal, int need_context, int cidx )
{
	return ( PAPI_OK );
}

int
_papi_hwi_stop_signal( int signal )
{
	return ( PAPI_OK );
}

int
_papi_hwi_stop_timer( void )
{
	int retval = PAPI_OK;

	if ( timeKillEvent( wTimerID ) != TIMERR_NOERROR )
		retval = PAPI_ESYS;
	timeEndPeriod( wTimerRes );
	return ( retval );
}

#else
#include <sys/time.h>
#include <errno.h>
#include <string.h>

int _papi_hwi_using_signal[PAPI_NSIG];

int
_papi_hwi_start_timer( int timer, int signal, int ns )
{
	struct itimerval value;
	int us = ns / 1000;

	if ( us == 0 )
		us = 1;

#ifdef ANY_THREAD_GETS_SIGNAL
	_papi_hwi_lock( INTERNAL_LOCK );
	if ( ( _papi_hwi_using_signal[signal] - 1 ) ) {
		INTDBG( "itimer already installed\n" );
		_papi_hwi_unlock( INTERNAL_LOCK );
		return ( PAPI_OK );
	}
	_papi_hwi_unlock( INTERNAL_LOCK );
#else
	( void ) signal;		 /*unused */
#endif

	value.it_interval.tv_sec = 0;
	value.it_interval.tv_usec = us;
	value.it_value.tv_sec = 0;
	value.it_value.tv_usec = us;

	INTDBG( "Installing itimer %d, with %d us interval\n", timer, us );
	if ( setitimer( timer, &value, NULL ) < 0 ) {
		PAPIERROR( "setitimer errno %d", errno );
		return ( PAPI_ESYS );
	}

	return ( PAPI_OK );
}

int
_papi_hwi_start_signal( int signal, int need_context, int cidx )
{
	struct sigaction action;

	_papi_hwi_lock( INTERNAL_LOCK );
	_papi_hwi_using_signal[signal]++;
	if ( _papi_hwi_using_signal[signal] - 1 ) {
		INTDBG( "_papi_hwi_using_signal is now %d\n",
				_papi_hwi_using_signal[signal] );
		_papi_hwi_unlock( INTERNAL_LOCK );
		return ( PAPI_OK );
	}

	memset( &action, 0x00, sizeof ( struct sigaction ) );
	action.sa_flags = SA_RESTART;
	action.sa_sigaction =
		( void ( * )( int, siginfo_t *, void * ) ) _papi_hwd[cidx]->
		dispatch_timer;
	if ( need_context )
#if (defined(_BGL) /*|| defined (__bgp__)*/)
		action.sa_flags |= SIGPWR;
#else
		action.sa_flags |= SA_SIGINFO;
#endif

	INTDBG( "installing signal handler\n" );
	if ( sigaction( signal, &action, NULL ) < 0 ) {
		PAPIERROR( "sigaction errno %d", errno );
		_papi_hwi_unlock( INTERNAL_LOCK );
		return ( PAPI_ESYS );
	}

	INTDBG( "_papi_hwi_using_signal[%d] is now %d.\n", signal,
			_papi_hwi_using_signal[signal] );
	_papi_hwi_unlock( INTERNAL_LOCK );

	return ( PAPI_OK );
}

int
_papi_hwi_stop_signal( int signal )
{
	_papi_hwi_lock( INTERNAL_LOCK );
	if ( --_papi_hwi_using_signal[signal] == 0 ) {
		INTDBG( "removing signal handler\n" );
		if ( sigaction( signal, NULL, NULL ) == -1 ) {
			PAPIERROR( "sigaction errno %d", errno );
			_papi_hwi_unlock( INTERNAL_LOCK );
			return ( PAPI_ESYS );
		}
	}

	INTDBG( "_papi_hwi_using_signal[%d] is now %d\n", signal,
			_papi_hwi_using_signal[signal] );
	_papi_hwi_unlock( INTERNAL_LOCK );

	return ( PAPI_OK );
}

int
_papi_hwi_stop_timer( int timer, int signal )
{
#ifdef ANY_THREAD_GETS_SIGNAL
	_papi_hwi_lock( INTERNAL_LOCK );
	if ( _papi_hwi_using_signal[signal] > 1 ) {
		INTDBG( "itimer in use by another thread\n" );
		_papi_hwi_unlock( INTERNAL_LOCK );
		return ( PAPI_OK );
	}
	_papi_hwi_unlock( INTERNAL_LOCK );
#else
	( void ) signal;		 /*unused */
#endif

	INTDBG( "turning off timer\n" );
	if ( setitimer( timer, NULL, NULL ) == -1 ) {
		PAPIERROR( "setitimer errno %d", errno );
		return ( PAPI_ESYS );
	}

	return ( PAPI_OK );
}

#endif /* _WIN32 */



#if (!defined(HAVE_FFSLL) || defined(__bgp__))
/* find the first set bit in long long */

int
ffsll( long long lli )
{
	int i, num, t, tmpint, len;

	num = sizeof ( long long ) / sizeof ( int );
	if ( num == 1 )
		return ( ffs( ( int ) lli ) );
	len = sizeof ( int ) * CHAR_BIT;

	for ( i = 0; i < num; i++ ) {
		tmpint = ( int ) ( ( ( lli >> len ) << len ) ^ lli );

		t = ffs( tmpint );
		if ( t ) {
			return ( t + i * len );
		}
		lli = lli >> len;
	}
	return PAPI_OK;
}
#endif


/**********************************************************************
	Windows Compatability stuff
	Delimited by the _WIN32 define
**********************************************************************/
#ifdef _WIN32

/*
 This routine normally lives in <strings> on Unix.
 Microsoft Visual C++ doesn't have this file.
*/
extern int
ffs( int i )
{
	int c = 1;

	do {
		if ( i & 1 )
			return ( c );
		i = i >> 1;
		c++;
	} while ( i );
	return ( 0 );
}

/*
 More Unix routines that I can't find in Windows
 This one returns a pseudo-random integer
 given an unsigned int seed.
*/
extern int
rand_r( unsigned int *Seed )
{
	srand( *Seed );
	return ( rand(  ) );
}

/*
  Another Unix routine that doesn't exist in Windows.
  Kevin uses it in the memory stuff, specifically in PAPI_get_dmem_info().
*/
extern int
getpagesize( void )
{
	SYSTEM_INFO SystemInfo;			   // system information structure  

	GetSystemInfo( &SystemInfo );
	return ( ( int ) SystemInfo.dwPageSize );
}

#endif /* _WIN32 */
