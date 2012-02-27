/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
* @file		papi_hl.c
* CVS:		$Id$
* @author	Philip Mucci
*			mucci@cs.utk.edu
* @author	Kevin London
*           london@cs.utk.edu
* @author	dan terpstra
*			terpstra@cs.utk.edu
* @brief This file contains the 'high level' interface to PAPI. 
*  BASIC is a high level language. ;-) */

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include <string.h>

/* high level papi functions*/

/*
 * Which high-level interface are we using?
 */
#define HL_START_COUNTERS	1
#define HL_FLIPS		2
#define HL_IPC			3
#define HL_FLOPS		4

/* Definitions for reading */
#define PAPI_HL_READ		1
#define PAPI_HL_ACCUM		2

/** @internal 
 * This is stored per thread
 */
typedef struct _HighLevelInfo
{
	int EventSet;				   /**< EventSet of the thread */
	short int num_evts;
	short int running;
	long long initial_time;			   /**< Start time */
	float total_proc_time;			   /**< Total processor time */
	float total_ins;			   /**< Total instructions */
} HighLevelInfo;

int _hl_rate_calls( float *real_time, float *proc_time, long long *ins,
					float *rate, unsigned int EVENT, HighLevelInfo * state );
void _internal_cleanup_hl_info( HighLevelInfo * state );
int _internal_check_state( HighLevelInfo ** state );
int _internal_start_hl_counters( HighLevelInfo * state );
int _internal_hl_read_cnts( long long *values, int array_len, int flag );

/* CHANGE LOG:
  - ksl 10/17/03
   Pretty much a complete rewrite of the high level interface.  Now
   the interface is thread safe and you don't have to worry as much
   about mixing the various high level calls.

  - dkt 11/19/01:
   After much discussion with users and developers, removed FMA and SLOPE
   fudge factors. SLOPE was not being used, and we decided the place to
   apply FMA was at a higher level where there could be a better understanding
   of platform discrepancies and code implications.
   ALL PAPI CALLS NOW RETURN EXACTLY WHAT THE HARDWARE REPORTS
  - dkt 08/14/01:
   Added reinitialization of values and proc_time to new reinit code.
   Added SLOPE and FMA constants to correct for systemic errors on a
   platform-by-platform basis.
   SLOPE is a factor subtracted from flpins on each call to compensate
   for platform overhead in the call.
   FMA is a shifter that doubles floating point counts on platforms that
   count FMA as one op instead of two.
   NOTE: We are making the FLAWED assumption that ALL flpins are FMA!
   This will result in counts that are TOO HIGH on the affected platforms
   in instances where the code is NOT mostly FMA.
  - dkt 08/01/01:
   NOTE: Calling semantics have changed!
   Now, if flpins < 0 (an invalid value) a PAPI_reset is issued to reset the
   counter values. The internal start time is also reset. This should be a 
   benign change, exept in the rare case where a user passes an uninitialized
   (and possibly negative) value for flpins to the routine *AFTER* it has been
   called the first time. This is unlikely, since the first call clears and
   returns th is value.
  - dkt 08/01/01:
   Internal sequencing changes:
   -- initial PAPI_get_real_usec() call moved above PAPI_start to avoid unwanted flops.
   -- PAPI_accum() replaced with PAPI_start() / PAPI_stop pair for same reason.
*/

/** @internal 
 * This function is called to determine the state of the system.
 * We may as well set the HighLevelInfo so you don't have to look it
 * up again.
 */
int
_internal_check_state( HighLevelInfo ** outgoing )
{
	int retval;
	HighLevelInfo *state = NULL;

	/* Only allow one thread at a time in here */
	if ( init_level == PAPI_NOT_INITED ) {
		retval = PAPI_library_init( PAPI_VER_CURRENT );
		if ( retval != PAPI_VER_CURRENT ) {
			return ( retval );
		} else {
			_papi_hwi_lock( HIGHLEVEL_LOCK );
			init_level = PAPI_HIGH_LEVEL_INITED;
			_papi_hwi_unlock( HIGHLEVEL_LOCK );
		}
	}

	/*
	 * Do we have the thread specific data setup yet?
	 */
	if ( ( retval =
		   PAPI_get_thr_specific( PAPI_HIGH_LEVEL_TLS, ( void * ) &state ) )
		 != PAPI_OK || state == NULL ) {
		state = ( HighLevelInfo * ) papi_malloc( sizeof ( HighLevelInfo ) );
		if ( state == NULL )
			return ( PAPI_ENOMEM );

		memset( state, 0, sizeof ( HighLevelInfo ) );
		state->EventSet = -1;

		if ( ( retval = PAPI_create_eventset( &state->EventSet ) ) != PAPI_OK )
			return ( retval );

		if ( ( retval =
			   PAPI_set_thr_specific( PAPI_HIGH_LEVEL_TLS,
									  state ) ) != PAPI_OK )
			return ( retval );
	}
	*outgoing = state;
	return ( PAPI_OK );
}

/** @internal 
 * Make sure to allocate space for values 
 */
int
_internal_start_hl_counters( HighLevelInfo * state )
{
	return ( PAPI_start( state->EventSet ) );
}

void
_internal_cleanup_hl_info( HighLevelInfo * state )
{
	state->num_evts = 0;
	state->running = 0;
	state->initial_time = -1;
	state->total_proc_time = 0;
	state->total_ins = 0;
	return;
}

/** @class PAPI_flips
  *	@brief Simplified call to get Mflips/s (floating point instruction rate), real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_flips( float *rtime, float *ptime, long long *flpins, float *mflips );
  *
  *     @param *rtime
  *		total realtime since the first PAPI_flips() call
  *	@param *ptime
  *		total process time since the first PAPI_flips() call
  *	@param *flpins
  *		total floating point instructions since the first call
  *  
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than: PAPI_flips() or PAPI_flops().
  *	@retval PAPI_ENOEVNT 
  *		The floating point operations, floating point instructions or total cycles 
  *		event does not exist.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_flips() will initialize the PAPI High Level interface, 
  * set up the counters to monitor PAPI_FP_INS and PAPI_TOT_CYC events 
  * and start the counters. 
  * Subsequent calls will read the counters and return total real time, 
  * total process time, total floating point instructions since the start of the 
  * measurement and the Mflip/s rate since latest call to PAPI_flips(). 
  * A call to PAPI_stop_counters()  will stop the counters from running and then 
  * calls such as PAPI_start_counters()  can safely be used. 
  *
  * @internal
  * The next three calls all use _hl_rate_calls() to return an instruction rate value.
  * PAPI_flips returns information related to floating point instructions using 
  * the PAPI_FP_INS event. This is intended to measure instruction rate through the 
  * floating point pipe with no massaging.
  * PAPI_flops return information related to theoretical floating point operations
  * rather than simple instructions. It uses the PAPI_FP_OPS event which attempts to 
  * 'correctly' account for, e.g., FMA undercounts and FP Store overcounts, etc.
  *
  * @see PAPI_stop_counters() PAPI_ipc() PAPI_set_opt()
 */
int
PAPI_flips( float *rtime, float *ptime, long long *flpins, float *mflips )
{
	if ( rtime == NULL || ptime == NULL || flpins == NULL || mflips == NULL )
		return PAPI_EINVAL;

	HighLevelInfo *state = NULL;
	int retval;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK ) {
		return ( retval );
        }

	if ( ( retval =
		   _hl_rate_calls( rtime, ptime, flpins, mflips,
						   ( unsigned int ) PAPI_FP_INS, state ) ) != PAPI_OK )
		return ( retval );

	return ( PAPI_OK );
}

/** @class PAPI_flops
  *	@brief Simplified call to get Mflops/s (floating point instruction rate), real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_flops( float *rtime, float *ptime, long long *flpops, float *mflops );
  *
  *     @param *rtime
  *		total realtime since the first PAPI_flops() call
  *	@param *ptime
  *		total process time since the first PAPI_flops() call
  *	@param *flpins
  *		total floating point instructions since the first call
  *     @param *rtime
  *		total realtime since the first PAPI_flops() call
  *	@param *ptime
  *		total process time since the first PAPI_flops() call
  *	@param *flpops
  *		total floating point instructions since the first call
  * 
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than: PAPI_flips() or PAPI_flops().
  *	@retval PAPI_ENOEVNT 
  *		The floating point operations, floating point instructions or total cycles 
  *		event does not exist.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_flops() will initialize the PAPI High Level interface, 
  * set up the counters to monitor PAPI_FP_OPS and PAPI_TOT_CYC events 
  * and start the counters. 
  * Subsequent calls will read the counters and return total real time, 
  * total process time, total floating point instructions since the start of the 
  * measurement and the Mflop/s rate since latest call to PAPI_flops(). 
  * A call to PAPI_stop_counters()  will stop the counters from running and then 
  * calls such as PAPI_start_counters()  can safely be used. 
  *
  * @internal
  * The next three calls all use _hl_rate_calls() to return an instruction rate value.
  * PAPI_flops returns information related to floating point instructions using 
  * the PAPI_FP_INS event. This is intended to measure instruction rate through the 
  * floating point pipe with no massaging.
  * PAPI_flops return information related to theoretical floating point operations
  * rather than simple instructions. It uses the PAPI_FP_OPS event which attempts to 
  * 'correctly' account for, e.g., FMA undercounts and FP Store overcounts, etc.
  *
  * @see PAPI_stop_counters() PAPI_ipc() PAPI_set_opt()
 */
int
PAPI_flops( float *rtime, float *ptime, long long *flpops, float *mflops )
{
	if ( rtime == NULL || ptime == NULL || flpops == NULL || mflops == NULL )
		return PAPI_EINVAL;

	HighLevelInfo *state = NULL;
	int retval;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK )
		return ( retval );

	if ( ( retval =
		   _hl_rate_calls( rtime, ptime, flpops, mflops,
						   ( unsigned int ) PAPI_FP_OPS, state ) ) != PAPI_OK )
		return ( retval );

	return ( PAPI_OK );
}

/** @class PAPI_ipc
 *	@brief Get instructions per cycle, real and processor time.
 *	
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_ipc( float *rtime, float *ptime, long long *ins, float *ipc );
 *
 *	@param *rtime
 *		total realtime since the first PAPI_flops() call
 *	@param *ptime
 *		total process time since the first PAPI_flops() call
 *	@param *ins
 *		total instructions since the first call
 *	@param *ipc
 *		instructions per cycle achieved since the previous call
 *
 *	@retval PAPI_EINVAL 
 *		The counters were already started by something other than: PAPI_ipc()
 *	@retval PAPI_ENOEVNT 
 *		The total instructions or total cycles event does not exist.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 * The first call to PAPI_ipc() will initialize the PAPI High Level interface,
 * set up the counters to monitor PAPI_TOT_INS and PAPI_TOT_CYC events 
 * and start the counters. 
 * Subsequent calls will read the counters and return total real time, 
 * total process time, total instructions since the start of the measurement 
 * and the instructions per cycle rate since latest call to PAPI_ipc(). 
 * A call to PAPI_stop_counters()  will stop the counters from running and then 
 * calls such as PAPI_start_counters()  can safely be used. 
 *
 * @see PAPI_flops() PAPI_stop_counters() PAPI_set_opt() PAPI_flips()
 */
int
PAPI_ipc( float *rtime, float *ptime, long long *ins, float *ipc )
{
	if ( rtime == NULL || ptime == NULL || ins == NULL || ipc == NULL )
		return PAPI_EINVAL;

	HighLevelInfo *state = NULL;
	int retval;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK )
		return ( retval );

	return _hl_rate_calls( rtime, ptime, ins, ipc,
						   ( unsigned int ) PAPI_TOT_INS, state );
}

int
_hl_rate_calls( float *real_time, float *proc_time, long long *ins, float *rate,
				unsigned int EVENT, HighLevelInfo * state )
{
	long long values[2] = { 0, 0 };
	int retval = 0;
	int level = 0;


	if ( EVENT == ( unsigned int ) PAPI_FP_INS )
		level = HL_FLIPS;
	else if ( EVENT == ( unsigned int ) PAPI_TOT_INS )
		level = HL_IPC;
	else if ( EVENT == ( unsigned int ) PAPI_FP_OPS )
		level = HL_FLOPS;

	if ( state->running != 0 && state->running != level )
		return ( PAPI_EINVAL );

	if ( state->running == 0 ) {
		if ( PAPI_query_event( ( int ) EVENT ) != PAPI_OK )
			return ( PAPI_ENOEVNT );

		if ( ( retval =
			   PAPI_add_event( state->EventSet, ( int ) EVENT ) ) != PAPI_OK ) {
			_internal_cleanup_hl_info( state );
			PAPI_cleanup_eventset( state->EventSet );
			return ( retval );
		}

		if ( PAPI_query_event( ( int ) PAPI_TOT_CYC ) != PAPI_OK )
			return ( PAPI_ENOEVNT );

		if ( ( retval =
			   PAPI_add_event( state->EventSet,
							   ( int ) PAPI_TOT_CYC ) ) != PAPI_OK ) {
			_internal_cleanup_hl_info( state );
			PAPI_cleanup_eventset( state->EventSet );
			return ( retval );
		}

		state->initial_time = PAPI_get_real_usec(  );
		if ( ( retval = PAPI_start( state->EventSet ) ) != PAPI_OK )
			return ( retval );
		state->running = ( short ) level;
	} else {
		if ( ( retval = PAPI_stop( state->EventSet, values ) ) != PAPI_OK )
			return ( retval );
		/* Use Multiplication because it is much faster */
		*real_time = ( float ) ( ( double )
								 ( PAPI_get_real_usec(  ) -
								   state->initial_time ) * .000001 );
		*proc_time =
			( float ) ( ( double ) values[1] * .000001 /
						( ( _papi_hwi_system_info.hw_info.mhz ==
							0 ) ? 1 : _papi_hwi_system_info.hw_info.mhz ) );
		if ( *proc_time > 0 )
			*rate =
				( float ) ( ( float ) values[0] *
							( EVENT ==
							  ( unsigned int ) PAPI_TOT_INS ? 1 :
							  _papi_hwi_system_info.hw_info.mhz ) /
							( float ) ( values[1] == 0 ? 1 : values[1] ) );
		state->total_proc_time += *proc_time;
		state->total_ins += ( float ) values[0];
		*proc_time = state->total_proc_time;
		*ins = ( long long ) state->total_ins;
		if ( ( retval = PAPI_start( state->EventSet ) ) != PAPI_OK ) {
			state->running = 0;
			return ( retval );
		}
	}
	return PAPI_OK;
}

/** @class PAPI_num_counters
  *	@brief Get the number of hardware counters available on the system.
  *
  *	@par C Interface:
  *	\#include <papi.h> @n
  *	int PAPI_num_counters( void );
  *
  * @post 
  *		Initializes the library to PAPI_HIGH_LEVEL_INITED if necessary.
  *
  *	@retval PAPI_EINVAL 
  *		papi.h is different from the version used to compile the PAPI library.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation.
  *	@retval PAPI_ESYS 
  *		A system or C library call failed inside PAPI, see the errno variable. 
  *
  *	@par Examples:
  * @code
  * int num_hwcntrs;
  * //  The installation does not support PAPI 
  * if ((num_hwcntrs = PAPI_num_counters()) < 0 )
  * 	handle_error(1);
  * //  The installation supports PAPI, but has no counters 
  * if ((num_hwcntrs = PAPI_num_counters()) == 0 )
  * 	fprintf(stderr,"Info:: This machine does not provide hardware counters.\n");
  *	@endcode
  *
  * PAPI_num_counters() returns the optimal length of the values array for the high level functions. 
  * This value corresponds to the number of hardware counters supported by the current substrate.
  *
  * @note This function only works for the CPU component. To determine the number of counters on
  * another component, use the low level PAPI_num_cmp_hwctrs().
  */
int
PAPI_num_counters( void )
{
	int retval;
	HighLevelInfo *tmp = NULL;

	/* Make sure the Library is initialized, etc... */
	if ( ( retval = _internal_check_state( &tmp ) ) != PAPI_OK )
		return ( retval );

	return ( PAPI_get_opt( PAPI_MAX_HWCTRS, NULL ) );
}

/** @class PAPI_start_counters
 *	@brief Start counting hardware events.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_start_counters( int *events, int array_len );
 *
 * @param *events
 *		an array of codes for events such as PAPI_INT_INS or a native event code 
 * @param array_len
 *		the number of items in the *events array 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_EISRUN 
 *		Counters have already been started, you must call PAPI_stop_counters() 
 *		before you call this function again.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware cannot count this event and other events 
 *		in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 * PAPI_start_counters() starts counting the events named in the *events array. 
 * This function cannot be called if the counters have already been started. 
 * The user must call PAPI_stop_counters() to stop the events explicitly if 
 * he/she wants to call this function again. 
 * It is the user's responsibility to choose events that can be counted 
 * simultaneously by reading the vendor's documentation. 
 * The length of the *events array should be no longer than the value returned 
 * by PAPI_num_counters(). 
 *
 *	@code
if( PAPI_start_counters( Events, num_hwcntrs ) != PAPI_OK )
	handle_error(1);
 *	@endcode
 *
 * @see PAPI_stop_counters() PAPI_add_event() PAPI_create_eventset()
 */
int
PAPI_start_counters( int *events, int array_len )
{
	int i, retval;
	HighLevelInfo *state = NULL;

	if ( events == NULL || array_len <= 0 )
		return PAPI_EINVAL;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK )
		return ( retval );

	if ( state->running != 0 )
		return ( PAPI_EINVAL );

	/* load events to the new EventSet */
	for ( i = 0; i < array_len; i++ ) {
		retval = PAPI_add_event( state->EventSet, events[i] );
		if ( retval == PAPI_EISRUN )
			return ( retval );

		if ( retval ) {
			/* remove any prior events that may have been added 
			 * and cleanup the high level information
			 */
			_internal_cleanup_hl_info( state );
			PAPI_cleanup_eventset( state->EventSet );
			return ( retval );
		}
	}
	/* start the EventSet */
	if ( ( retval = _internal_start_hl_counters( state ) ) == PAPI_OK ) {
		state->running = HL_START_COUNTERS;
		state->num_evts = ( short ) array_len;
	}
	return ( retval );
}

/*========================================================================*/
/* int PAPI_read_counters(long long *values, int array_len)      */
/*                                                                        */
/* Read the running counters into the values array. This call             */
/* implicitly initializes the internal counters to zero and allows        */
/* them continue to run upon return.                                      */
/*========================================================================*/

int
_internal_hl_read_cnts( long long *values, int array_len, int flag )
{
	int retval;
	HighLevelInfo *state = NULL;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK )
		return ( retval );

	if ( state->running != HL_START_COUNTERS || array_len < state->num_evts )
		return ( PAPI_EINVAL );

	if ( flag == PAPI_HL_ACCUM )
		return ( PAPI_accum( state->EventSet, values ) );
	else if ( flag == PAPI_HL_READ ) {
		if ( ( retval = PAPI_read( state->EventSet, values ) ) != PAPI_OK )
			return ( retval );
		return ( PAPI_reset( state->EventSet ) );
	}

	/* Invalid flag passed in */
	return ( PAPI_EINVAL );
}

/** @class PAPI_read_counters
 *	@brief Read and reset counters.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_read_counters( long long *values, int array_len );
 *
 * @param *values
 *		an array to hold the counter values of the counting events
 * @param arry_len
 *		the number of items in the *events array
 *
 * @pre 
 *		These calls assume an initialized PAPI library and a properly added event set.
 *
 * @post 
 *		The counters are reset and left running after the call.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable. 
 *
 * PAPI_read_counters() copies the event counters into the array *values. 
 *
 *	@code
do_100events();
if ( PAPI_read_counters( values, num_hwcntrs ) != PAPI_OK )
	handlw_error(1);
// values[0] now equals 100 
do_100events();
if ( PAPI_accum_counters( values, num_hwcntrs ) != PAPI_OK )
	handle_error(1);
// values[0] now equals 200
values[0] = -100;
do_100events();
if ( PAPI_accum_counters(values, num_hwcntrs ) != PAPI_OK )
	handle_error();
// values[0] now equals 0
 *	@endcode
 *
 * @see PAPI_set_opt() PAPI_start_counters()
 */
int
PAPI_read_counters( long long *values, int array_len )
{
	return ( _internal_hl_read_cnts( values, array_len, PAPI_HL_READ ) );
}


/** @class PAPI_accum_counters
 *	@brief Accumulate and reset counters.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_accum_counters( long long *values, int array_len );
 *
 * @param *values
 *		an array to hold the counter values of the counting events
 * @param arry_len
 *		the number of items in the *events array
 *
 * @pre 
 *		These calls assume an initialized PAPI library and a properly added event set.
 *
 * @post 
 *		The counters are reset and left running after the call.
 * 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable. 
 *
 * PAPI_accum_counters() adds the event counters into the array *values. 
 *
 *	@code
do_100events();
if ( PAPI_read_counters( values, num_hwcntrs ) != PAPI_OK )
	handlw_error(1);
// values[0] now equals 100 
do_100events();
if ( PAPI_accum_counters( values, num_hwcntrs ) != PAPI_OK )
	handle_error(1);
// values[0] now equals 200
values[0] = -100;
do_100events();
if ( PAPI_accum_counters(values, num_hwcntrs ) != PAPI_OK )
	handle_error();
// values[0] now equals 0
 *	@endcode
 *
 * @see PAPI_set_opt() PAPI_start_counters()
 */
int
PAPI_accum_counters( long long *values, int array_len )
{
	if ( values == NULL || array_len <= 0 )
		return PAPI_EINVAL;

	return ( _internal_hl_read_cnts( values, array_len, PAPI_HL_ACCUM ) );
}

/** @class PAPI_stop_counters
 *	@brief Stop counting hardware events and reset values to zero.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_stop_counters( long long *values, int array_len );
 *
 * @param *values
 *		an array where to put the counter values
 * @param array_len
 *		the number of items in the *values array 
 *
 * @post 
 *	After this function is called, the values are reset to zero. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTRUN 
 *		The EventSet is not started yet.
 *	@retval PAPI_ENOEVST 
 *		The EventSet has not been added yet. 
 *
 * The PAPI_stop_counters() function stops the counters and copies the counts 
 * into the *values array. 
 * The counters must have been started by a previous call to PAPI_start_counters(). 
 *
 *	\code
int Events[2] = { PAPI_TOT_CYC, PAPI_TOT_INS };
long long values[2];
if ( PAPI_start_counters( Events, 2 ) != PAPI_OK )
	handle_error(1);
your_slow_code();
if ( PAPI_stop_counters( values, 2 ) != PAPI_OK )
	handle_error(1);
 *	\endcode
 * 
 * @see PAPI_read_counters() PAPI_start_counters() PAPI_set_opt()
 */
int
PAPI_stop_counters( long long *values, int array_len )
{
	int retval;
	HighLevelInfo *state = NULL;

	if ( values == NULL || array_len <= 0 )
		return PAPI_EINVAL;

	if ( ( retval = _internal_check_state( &state ) ) != PAPI_OK )
		return ( retval );

	if ( state->running == 0 )
		return ( PAPI_ENOTRUN );

	if ( state->running == HL_FLOPS || state->running == HL_FLIPS ||
		 state->running == HL_IPC ) {
		long long tmp_values[2];
		retval = PAPI_stop( state->EventSet, tmp_values );
	} else if ( state->running != HL_START_COUNTERS ||
				array_len < state->num_evts )
		return ( PAPI_EINVAL );
	else
		retval = PAPI_stop( state->EventSet, values );

	if ( retval == PAPI_OK ) {
		_internal_cleanup_hl_info( state );
		PAPI_cleanup_eventset( state->EventSet );
	}
	APIDBG( "PAPI_stop_counters returns %d\n", retval );
	return retval;
}

void
_papi_hwi_shutdown_highlevel(  )
{
	HighLevelInfo *state = NULL;

	if ( PAPI_get_thr_specific( PAPI_HIGH_LEVEL_TLS, ( void * ) &state ) ==
		 PAPI_OK ) {
		if ( state )
			papi_free( state );
	}
}
