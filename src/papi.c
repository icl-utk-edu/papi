/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
* @file:    papi.c
* CVS:     $Id$
* @author:  Philip Mucci
*          mucci@cs.utk.edu
* @author    dan terpstra
*          terpstra@cs.utk.edu
* @author    Min Zhou
*          min@cs.utk.edu
* @author  Kevin London
*	   london@cs.utk.edu
* @author  Per Ekman
*          pek@pdc.kth.se
* Mods:    <Gary Mohr>
*          <gary.mohr@bull.com>
*
* @brief Most of the low-level API is here.
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h> 

unsigned char PENTIUM4 = 0;
/* Native events consist of a flag field, an event field, and a unit mask field. 		
 * These variables define the characteristics of the event and unit mask fields. */
unsigned int PAPI_NATIVE_EVENT_AND_MASK = 0x000003ff;
unsigned int PAPI_NATIVE_EVENT_SHIFT = 0;
unsigned int PAPI_NATIVE_UMASK_AND_MASK = 0x03fffc00;
unsigned int PAPI_NATIVE_UMASK_MAX = 16;
unsigned int PAPI_NATIVE_UMASK_SHIFT = 10;

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

#ifdef DEBUG
#define papi_return(a) return((_papi_hwi_debug_handler ? _papi_hwi_debug_handler(a) : a))
#else
#define papi_return(a) return(a)
#endif

#ifdef ANY_THREAD_GETS_SIGNAL
extern int ( *_papi_hwi_thread_kill_fn ) ( int, int );
#endif

extern unsigned long int ( *_papi_hwi_thread_id_fn ) ( void );
extern papi_mdi_t _papi_hwi_system_info;

/* papi_data.c */

extern hwi_presets_t _papi_hwi_presets;
extern const hwi_describe_t _papi_hwi_derived[];

extern int init_retval;
extern int init_level;

/* Defined by the substrate */
extern hwi_preset_data_t _papi_hwi_preset_data[];

inline_static int
valid_component( int cidx )
{
	if ( _papi_hwi_invalid_cmp( cidx ) )
		return ( PAPI_ENOCMP );
	return ( cidx );
}

inline_static int
valid_ESI_component( EventSetInfo_t * ESI )
{
	return ( valid_component( ESI->CmpIdx ) );
}

static void
set_runtime_config(  )
{
	enum Vendors
	{ INTEL };
	FILE *file;
	char line[256];
	char *token;
	char *delim = ":";
	int vendor = -1, family = -1;

	if ( ( file = fopen( "/proc/cpuinfo", "r" ) ) != NULL ) {
		while ( fgets( line, sizeof ( line ), file ) != NULL ) {
			if ( strstr( line, "vendor_id" ) ) {
				if ( strstr( line, "GenuineIntel" ) )
					vendor = INTEL;
				else
					return;
			}

			if ( strstr( line, "cpu family" ) ) {
				token = strtok( line, delim );
				token = strtok( NULL, delim );
				family = atoi( token );
			}
		}

		if ( vendor == INTEL ) {
			if ( family == 15 ) {	//Pentium4
				PENTIUM4 = 1;
				PAPI_NATIVE_EVENT_AND_MASK = 0x000000ff;
				PAPI_NATIVE_UMASK_AND_MASK = 0x0fffff00;
				PAPI_NATIVE_UMASK_SHIFT = 8;
			} else if ( family == 31 || family == 32 ) {	//Itanium2
				PAPI_NATIVE_EVENT_AND_MASK = 0x00000fff;
				PAPI_NATIVE_UMASK_AND_MASK = 0x0ffff000;
				PAPI_NATIVE_UMASK_SHIFT = 12;
			}
		}
	}
}

/** @class	PAPI_thread_init
 *  initialize thread support in the PAPI library 
 *
 *	@param *id_fn 
 *		Pointer to a function that returns current thread ID. 
 *
 *	PAPI_thread_init initializes thread support in the PAPI library. 
 *	Applications that make no use of threads do not need to call this routine. 
 *	This function MUST return a UNIQUE thread ID for every new thread/LWP created. 
 *	The OpenMP call omp_get_thread_num() violates this rule, as the underlying 
 *	LWPs may have been killed off by the run-time system or by a call to omp_set_num_threads() . 
 *	In that case, it may still possible to use omp_get_thread_num() in 
 *	conjunction with PAPI_unregister_thread() when the OpenMP thread has finished. 
 *	However it is much better to use the underlying thread subsystem's call, 
 *	which is pthread_self() on Linux platforms. 
 *
 *	@code
if ( PAPI_thread_init(pthread_self) != PAPI_OK )
	exit(1);
 *	@endcode
 *
 *	@see PAPI_register_thread PAPI_unregister_thread PAPI_get_thr_specific PAPI_set_thr_specific PAPI_thread_id PAPI_list_threads
 */
int
PAPI_thread_init( unsigned long int ( *id_fn ) ( void ) )
{
	/* Thread support not implemented on Alpha/OSF because the OSF pfm
	 * counter device driver does not support per-thread counters.
	 * When this is updated, we can remove this if statement
	 */
	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );

	if ( ( init_level & PAPI_THREAD_LEVEL_INITED ) )
		papi_return( PAPI_OK );

	/* xxxx this looks at vector 0 only -- I think the value should be promoted
	   out of cmp_info into hw_info. */
	if ( !SUPPORTS_MULTIPLE_THREADS( _papi_hwd[0]->cmp_info ) )
		papi_return( PAPI_ESBSTR );

	init_level |= PAPI_THREAD_LEVEL_INITED;
	papi_return( _papi_hwi_set_thread_id_fn( id_fn ) );
}

/** @class PAPI_thread_id
 *  get the thread identifier of the current thread 
 *
 *	@retval PAPI_EMISC 
 *		is returned if there are no threads registered.
 *	@retval -1 
 *		is returned if the thread id function returns an error. 
 *
 *	This function returns a valid thread identifier. 
 *	It calls the function registered with PAPI through a call to 
 *	PAPI_thread_init().
 *
 *	@code
unsigned long tid;

if ((tid = PAPI_thread_id()) == (unsigned long int)-1 )
	exit(1);

printf("Initial thread id is: %lu\n", tid );
 *	@endcode
 *	@see PAPI_thread_init
 */
unsigned long
PAPI_thread_id( void )
{
	if ( _papi_hwi_thread_id_fn != NULL )
		return ( ( *_papi_hwi_thread_id_fn ) (  ) );
	else
#ifdef DEBUG
	if ( _papi_hwi_debug_handler )
		return ( unsigned long ) _papi_hwi_debug_handler( PAPI_EMISC );
#endif
	return ( unsigned long ) PAPI_EMISC;
}

/* Thread Functions */

/* 
 * Notify PAPI that a thread has 'appeared'
 * We lookup the thread, if it does not exist we create it
 */
/** @class PAPI_register_thread
 *	Notify PAPI that a thread has 'appeared'
 *
 *	@retval PAPI_ENOMEM 
 *		Space could not be allocated to store the new thread information.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ESBSTR 
 *		Hardware counters for this thread could not be initialized. 
 *
 *	PAPI_register_thread should be called when the user wants to force PAPI to 
 *	initialize a thread that PAPI has not seen before. 
 *	Usually this is not necessary as PAPI implicitly detects the thread when an 
 *	eventset is created or other thread local PAPI functions are called. 
 *	However, it can be useful for debugging and performance enhancements in the 
 *	run-time systems of performance tools. 
 *
 *	@see PAPI_thread_id PAPI_thread_init
 */
int
PAPI_register_thread( void )
{
	ThreadInfo_t *thread;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	papi_return( _papi_hwi_lookup_or_create_thread( &thread, 0 ) );
}

/* 
 * Notify PAPI that a thread has 'disappeared'
 * We lookup the thread, if it does not exist we return an error
 */
/** @class PAPI_unregister_thread
 *  Notify PAPI that a thread has 'disappeared'
 *
 *	@retval PAPI_ENOMEM 
 *		Space could not be allocated to store the new thread information.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ESBSTR 
 *		Hardware counters for this thread could not be initialized. 
 *
 *	PAPI_unregister_thread should be called when the user wants to shutdown 
 *	a particular thread and free the associated thread ID. 
 *	THIS IS IMPORTANT IF YOUR THREAD LIBRARY REUSES THE SAME THREAD ID FOR A NEW KERNEL LWP. 
 *	OpenMP does this. OpenMP parallel regions, if separated by a call to 
 *	omp_set_num_threads() will often kill off the underlying kernel LWPs and 
 *	then start new ones for the next region. 
 *	However, omp_get_thread_id() does not reflect this, as the thread IDs 
 *	for the new LWPs will be the same as the old LWPs. 
 *	PAPI needs to know that the underlying LWP has changed so it can set up 
 *	the counters for that new thread. 
 *	This is accomplished by calling this function. 
 */
int
PAPI_unregister_thread( void )
{
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );

	if ( thread )
		papi_return( _papi_hwi_shutdown_thread( thread ) );

	papi_return( PAPI_EMISC );
}

/** @class PAPI_list_threads
 *	list the registered thread ids 
 *
 *	@param *id
 *		A pointer to a preallocated array. 
 *		This may be NULL to only return a count of threads. 
 *		No more than *number codes will be stored in the array.
 *	@param *num
 *		An input and output parameter, input specifies the number of allocated 
 *		elements in *id (if non-NULL) and output specifies the number of threads. 
 *
 *	@retval PAPI_EINVAL
 *	
 *	PAPI_list_threads() returns to the caller a list of all thread ID's known to PAPI.
 *	This call assumes an initialized PAPI library. 
 *
 *	@see  PAPI_get_thr_specific PAPI_set_thr_specific PAPI_register_thread 
 *			PAPI_unregister_thread PAPI_thread_init PAPI_thread_id
 */
int
PAPI_list_threads( PAPI_thread_id_t * id, int *num )
{
	PAPI_all_thr_spec_t tmp;
	int retval;

	/* If id == NULL, then just count the threads, don't gather a list. */
	/* If id != NULL, then we need the length of the id array in num. */

	if ( ( num == NULL ) || ( id && ( *num <= 0 ) ) )
		papi_return( PAPI_EINVAL );

	memset( &tmp, 0x0, sizeof ( tmp ) );

	/* data == NULL, since we don't want the thread specific pointers. */
	/* id may be NULL, if the user doesn't want the thread ID's. */

	tmp.num = *num;
	tmp.id = id;
	tmp.data = NULL;

	retval = _papi_hwi_gather_all_thrspec_data( 0, &tmp );
	if ( retval == PAPI_OK )
		*num = tmp.num;

	papi_return( retval );
}

/** @class PAPI_get_thr_specific
 * @brief Retrieve a pointer to a thread specific data structure 
 *
 *	In C, PAPI_get_thr_specific PAPI_get_thr_specific will retrieve the pointer from the array with index @em tag. 
 *	There are 2 user available locations and @em tag can be either 
 *	PAPI_USR1_TLS or PAPI_USR2_TLS. 
 *	The array mentioned above is managed by PAPI and allocated to each 
 *	thread which has called PAPI_thread_init. 
 *	There is no Fortran equivalent function. 
 *
 *	@par Prototype:
 *		#include <papi.h> @n
 *		int PAPI_get_thr_specific( int tag, void **ptr );
 *
 *	@param tag
 *		An identifier, the value of which is either PAPI_USR1_TLS or 
 *		PAPI_USR2_TLS. This identifier indicates which of several data 
 *		structures associated with this thread is to be accessed.
 *	@param ptr
 *		A pointer to the memory containing the data structure. 
 *
 *	@retval PAPI_EINVAL 
 *		The @em tag argument is out of range. 
 *
 *	@par Example:
 *	@code
 int ret;
 HighLevelInfo *state = NULL;
 ret = PAPI_thread_init(pthread_self);
 if (ret != PAPI_OK) handle_error(ret);
 
 // Do we have the thread specific data setup yet?

ret = PAPI_get_thr_specific(PAPI_USR1_TLS, (void *) &state);
if (ret != PAPI_OK || state == NULL) {
	state = (HighLevelInfo *) malloc(sizeof(HighLevelInfo));
	if (state == NULL) return (PAPI_ESYS);
	memset(state, 0, sizeof(HighLevelInfo));
	state->EventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&state->EventSet);
	if (ret != PAPI_OK) return (PAPI_ESYS);
	ret = PAPI_set_thr_specific(PAPI_USR1_TLS, state);
	if (ret != PAPI_OK) return (ret);
}
*	@endcode
*	@see PAPI_register_thread PAPI_thread_init PAPI_thread_id PAPI_set_thr_specific
*/
int
PAPI_get_thr_specific( int tag, void **ptr )
{
	ThreadInfo_t *thread;
	int doall = 0, retval = PAPI_OK;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( tag & PAPI_TLS_ALL_THREADS ) {
		tag = tag ^ PAPI_TLS_ALL_THREADS;
		doall = 1;
	}
	if ( ( tag < 0 ) || ( tag > PAPI_TLS_NUM ) )
		papi_return( PAPI_EINVAL );

	if ( doall )
		papi_return( _papi_hwi_gather_all_thrspec_data
					 ( tag, ( PAPI_all_thr_spec_t * ) ptr ) );

	retval = _papi_hwi_lookup_or_create_thread( &thread, 0 );
	if ( retval == PAPI_OK )
		*ptr = thread->thread_storage[tag];
	else
		papi_return( retval );

	return ( PAPI_OK );
}

/** @class PAPI_set_thr_specific
 * @brief Store a pointer to a thread specific data structure 
 *
 *	In C, PAPI_set_thr_specific will save @em ptr into an array indexed by @em tag. 
 *	There are 2 user available locations and @em tag can be either 
 *	PAPI_USR1_TLS or PAPI_USR2_TLS. 
 *	The array mentioned above is managed by PAPI and allocated to each 
 *	thread which has called PAPI_thread_init. 
 *	There is no Fortran equivalent function. 
 *
 *	@par Prototype:
 *		#include <papi.h> @n
 *		int PAPI_set_thr_specific( int tag, void *ptr );
 *
 *	@param tag
 *		An identifier, the value of which is either PAPI_USR1_TLS or 
 *		PAPI_USR2_TLS. This identifier indicates which of several data 
 *		structures associated with this thread is to be accessed.
 *	@param ptr
 *		A pointer to the memory containing the data structure. 
 *
 *	@retval PAPI_EINVAL 
 *		The @em tag argument is out of range. 
 *
 *	@par Example:
 *	@code
int ret;
HighLevelInfo *state = NULL;
ret = PAPI_thread_init(pthread_self);
if (ret != PAPI_OK) handle_error(ret);
 
// Do we have the thread specific data setup yet?

ret = PAPI_get_thr_specific(PAPI_USR1_TLS, (void *) &state);
if (ret != PAPI_OK || state == NULL) {
	state = (HighLevelInfo *) malloc(sizeof(HighLevelInfo));
	if (state == NULL) return (PAPI_ESYS);
	memset(state, 0, sizeof(HighLevelInfo));
	state->EventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&state->EventSet);
	if (ret != PAPI_OK) return (PAPI_ESYS);
	ret = PAPI_set_thr_specific(PAPI_USR1_TLS, state);
	if (ret != PAPI_OK) return (ret);
}
 *	@endcode
 *	@see PAPI_register_thread PAPI_thread_init PAPI_thread_id PAPI_get_thr_specific
 */
int
PAPI_set_thr_specific( int tag, void *ptr )
{
	ThreadInfo_t *thread;
	int retval = PAPI_OK;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( ( tag < 0 ) || ( tag > PAPI_NUM_TLS ) )
		papi_return( PAPI_EINVAL );

	retval = _papi_hwi_lookup_or_create_thread( &thread, 0 );
	if ( retval == PAPI_OK )
		thread->thread_storage[tag] = ptr;
	else
		return ( retval );

	return ( PAPI_OK );
}


/** @class PAPI_library_init
  *	initialize the PAPI library. 
 *
 *	@param version 
 *		upon initialization, PAPI checks the argument against the internal 
 *		value of PAPI_VER_CURRENT when the library was compiled. 
 *		This guards against portability problems when updating the PAPI shared 
 *		libraries on your system. 
 *
 *	@retval PAPI_EINVAL 
 *		papi.h is different from the version used to compile the PAPI library.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ESBSTR 
 *		This substrate does not support the underlying hardware.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable. 
 *
 *	PAPI_library_init() initializes the PAPI library. 
 *	It must be called before any low level PAPI functions can be used. 
 *	If your application is making use of threads PAPI_thread_init must also be 
 *	called prior to making any calls to the library other than PAPI_library_init() . 
 *	
 *	@see PAPI_thread_init
 */
int
PAPI_library_init( int version )
{
	int tmp = 0, tmpel;
	/* This is a poor attempt at a lock. 
	   For 3.1 this should be replaced with a 
	   true UNIX semaphore. We cannot use PAPI
	   locks here because they are not initialized yet */
	static int _in_papi_library_init_cnt = 0;
#ifdef DEBUG
	char *var;
#endif

	if ( version != PAPI_VER_CURRENT )
		papi_return( PAPI_EINVAL );

	++_in_papi_library_init_cnt;
	while ( _in_papi_library_init_cnt > 1 ) {
		PAPIERROR( "Multiple callers of PAPI_library_init" );
		sleep( 1 );
	}

#ifndef _WIN32
	/* This checks to see if we have forked or called init more than once.
	   If we have forked, then we continue to init. If we have not forked, 
	   we check to see the status of initialization. */

	APIDBG( "Initializing library: current PID %d, old PID %d\n", getpid(  ),
			_papi_hwi_system_info.pid );
	if ( _papi_hwi_system_info.pid == getpid(  ) )
#endif
	{
		/* If the magic environment variable PAPI_ALLOW_STOLEN is set,
		   we call shutdown if PAPI has been initialized. This allows
		   tools that use LD_PRELOAD to run on applications that use PAPI.
		   In this circumstance, PAPI_ALLOW_STOLEN will be set to 'stolen'
		   so the tool can check for this case. */

		if ( getenv( "PAPI_ALLOW_STOLEN" ) ) {
			char buf[PAPI_HUGE_STR_LEN];
			if ( init_level != PAPI_NOT_INITED )
				PAPI_shutdown(  );
			sprintf( buf, "%s=%s", "PAPI_ALLOW_STOLEN", "stolen" );
			putenv( buf );
		}

		/* If the library has been successfully initialized *OR*
		   the library attempted initialization but failed. */

		else if ( ( init_level != PAPI_NOT_INITED ) ||
				  ( init_retval != DEADBEEF ) ) {
			_in_papi_library_init_cnt--;
			if ( init_retval < PAPI_OK )
				papi_return( init_retval );
			else
				return ( init_retval );
		}

		APIDBG( "system_info was initialized, but init did not succeed\n" );
	}
#ifdef DEBUG
	var = ( char * ) getenv( "PAPI_DEBUG" );
	_papi_hwi_debug = 0;

	if ( var != NULL ) {
		if ( strlen( var ) != 0 ) {
			if ( strstr( var, "SUBSTRATE" ) )
				_papi_hwi_debug |= DEBUG_SUBSTRATE;
			if ( strstr( var, "API" ) )
				_papi_hwi_debug |= DEBUG_API;
			if ( strstr( var, "INTERNAL" ) )
				_papi_hwi_debug |= DEBUG_INTERNAL;
			if ( strstr( var, "THREADS" ) )
				_papi_hwi_debug |= DEBUG_THREADS;
			if ( strstr( var, "MULTIPLEX" ) )
				_papi_hwi_debug |= DEBUG_MULTIPLEX;
			if ( strstr( var, "OVERFLOW" ) )
				_papi_hwi_debug |= DEBUG_OVERFLOW;
			if ( strstr( var, "PROFILE" ) )
				_papi_hwi_debug |= DEBUG_PROFILE;
			if ( strstr( var, "MEMORY" ) )
				_papi_hwi_debug |= DEBUG_MEMORY;
			if ( strstr( var, "LEAK" ) )
				_papi_hwi_debug |= DEBUG_LEAK;
			if ( strstr( var, "ALL" ) )
				_papi_hwi_debug |= DEBUG_ALL;
		}

		if ( _papi_hwi_debug == 0 )
			_papi_hwi_debug |= DEBUG_API;
	}
#endif

	/* Be verbose for now */

	tmpel = _papi_hwi_error_level;
	_papi_hwi_error_level = PAPI_VERB_ECONT;
	set_runtime_config(  );

	/* Initialize internal globals */
	if ( _papi_hwi_init_global_internal(  ) != PAPI_OK ) {
		_in_papi_library_init_cnt--;
		_papi_hwi_error_level = tmpel;
		papi_return( PAPI_EINVAL );
	}

	/* Initialize substrate globals */

	tmp = _papi_hwi_init_global(  );
	if ( tmp ) {
		init_retval = tmp;
		_papi_hwi_shutdown_global_internal(  );
		_in_papi_library_init_cnt--;
		_papi_hwi_error_level = tmpel;
		papi_return( init_retval );
	}

	/* Initialize thread globals, including the main threads
	   substrate */

	tmp = _papi_hwi_init_global_threads(  );
	if ( tmp ) {
		int i;
		init_retval = tmp;
		_papi_hwi_shutdown_global_internal(  );
		for ( i = 0; i < papi_num_components; i++ ) {
			_papi_hwd[i]->shutdown_substrate(  );
		}
		_in_papi_library_init_cnt--;
		_papi_hwi_error_level = tmpel;
		papi_return( init_retval );
	}

	init_level = PAPI_LOW_LEVEL_INITED;
	_in_papi_library_init_cnt--;
	_papi_hwi_error_level = tmpel;
	return ( init_retval = PAPI_VER_CURRENT );
}

/** @class PAPI_query_event
  *	query if PAPI event exists 
 *
 *	@param EventCode
 *		a defined event such as PAPI_TOT_INS. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The hardware event specified is not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_query_event() asks the PAPI library if the PAPI Preset event can be 
 *	counted on this architecture. 
 *	If the event CAN be counted, the function returns PAPI_OK. 
 *	If the event CANNOT be counted, the function returns an error code. 
 *	This function also can be used to check the syntax of a native event. 
 *
 *	@see PAPI_remove_event PAPI_remove_events PAPI_presets PAPI_native
 */
int
PAPI_query_event( int EventCode )
{
	if ( EventCode & PAPI_PRESET_MASK ) {
		EventCode &= PAPI_PRESET_AND_MASK;
		if ( EventCode >= PAPI_MAX_PRESET_EVENTS )
			papi_return( PAPI_ENOTPRESET );

		if ( _papi_hwi_presets.count[EventCode] )
			papi_return( PAPI_OK );
		else
			return ( PAPI_ENOEVNT );
	}

	if ( EventCode & PAPI_NATIVE_MASK ) {
		papi_return( _papi_hwi_query_native_event
					 ( ( unsigned int ) EventCode ) );
	}

	papi_return( PAPI_ENOTPRESET );
}

/** @class PAPI_get_component_info
  *	get information about a specific software component 
 *
 *	@param cidx
 *		Component index
 *
 *	This function returns a pointer to a structure containing detailed 
 *	information about a specific software component in the PAPI library. 
 *	This includes versioning information, preset and native event 
 *	information, and more. 
 *	For full details, see @ref PAPI_component_info_t. 
 *
 *	@see PAPI_get_executable_info PAPI_get_hardware_info PAPI_get_dmem_info PAPI_get_opt PAPI_library_init
 */
const PAPI_component_info_t *
PAPI_get_component_info( int cidx )
{
	if ( _papi_hwi_invalid_cmp( cidx ) )
		return ( NULL );
	else
		return ( &( _papi_hwd[cidx]->cmp_info ) );
}

/* PAPI_get_event_info:
   tests input EventCode and returns a filled in PAPI_event_info_t 
   structure containing descriptive strings and values for the 
   specified event. Handles both preset and native events by 
   calling either _papi_hwi_get_event_info or 
   _papi_hwi_get_native_event_info.
*/
/** @class PAPI_get_event_info
  *	get the event's name and description info 
 *
 *	@param EventCode
 *		event code (preset or native)
 *	@param info 
 *		structure with the event information @ref PAPI_event_info_t
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The PAPI preset mask was set, but the hardware event specified is 
 *		not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	This function fills the event information into a structure. 
 *	In Fortran, some fields of the structure are returned explicitly. 
 *	This function works with existing PAPI preset and native event codes. 
 *
 *	@see PAPI_event_name_to_code PAPI_set_event_info
 */
int
PAPI_get_event_info( int EventCode, PAPI_event_info_t * info )
{
	int i = EventCode & PAPI_PRESET_AND_MASK;

	if ( info == NULL )
		papi_return( PAPI_EINVAL );

	if ( EventCode & PAPI_PRESET_MASK ) {
		if ( i >= PAPI_MAX_PRESET_EVENTS )
			papi_return( PAPI_ENOTPRESET );
		papi_return( _papi_hwi_get_event_info( EventCode, info ) );
	}

	if ( EventCode & PAPI_NATIVE_MASK ) {
		papi_return( _papi_hwi_get_native_event_info
					 ( ( unsigned int ) EventCode, info ) );
	}

	papi_return( PAPI_ENOTPRESET );
}


/** @class PAPI_event_code_to_name
  *	convert a numeric hardware event code to a name.
 *
 *	@param EventCode 
 *		the numeric code for the event 
 *	@param out
 *		string for the name to be placed in
 *
 *	@retval PAPI_EINVAL 
 8		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The hardware event specified is not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The hardware event is not available on the underlying hardware. 
 *
 *	PAPI_event_code_to_name() is used to translate a 32-bit integer PAPI event 
 *	code into an ASCII PAPI event name. 
 *	Either Preset event codes or Native event codes can be passed to this routine. 
 *	Native event codes and names differ from platform to platform.
 *
 *	@see PAPI_remove_event PAPI_get_event_info PAPI_enum_events PAPI_add_event PAPI_presets PAPI_native
 */
int
PAPI_event_code_to_name( int EventCode, char *out )
{
	if ( out == NULL )
		papi_return( PAPI_EINVAL );

	if ( EventCode & PAPI_PRESET_MASK ) {
		EventCode &= PAPI_PRESET_AND_MASK;
		if ( ( EventCode >= PAPI_MAX_PRESET_EVENTS )
			 || ( _papi_hwi_presets.info[EventCode].symbol == NULL ) )
			papi_return( PAPI_ENOTPRESET );

		strncpy( out, _papi_hwi_presets.info[EventCode].symbol,
				 PAPI_MAX_STR_LEN );
		papi_return( PAPI_OK );
	}

	if ( EventCode & PAPI_NATIVE_MASK ) {
		return ( _papi_hwi_native_code_to_name
				 ( ( unsigned int ) EventCode, out, PAPI_MAX_STR_LEN ) );
	}

	papi_return( PAPI_ENOEVNT );
}

/** @class PAPI_event_name_to_code
  *	convert a name to a numeric hardware event code. 
 *
 *	@param in
 *		Name to convert
 *	@param out
 *		code
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The hardware event specified is not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The hardware event is not available on the underlying hardware. 
 *
 *	PAPI_event_name_to_code() is used to translate an ASCII PAPI event name 
 *	into an integer PAPI event code. 
 *
 *	@see PAPI_remove_event PAPI_get_event_info PAPI_enum_events PAPI_add_event PAPI_presets PAPI_native
 */
int
PAPI_event_name_to_code( char *in, int *out )
{
   APIDBG("Entry: in: %p, name: %s, out: %p\n", in, in, out);
	int i;

	if ( ( in == NULL ) || ( out == NULL ) )
		papi_return( PAPI_EINVAL );

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );

	/* With user definable events, we can no longer assume
	   presets begin with "PAPI"...
	   if (strncmp(in, "PAPI", 4) == 0) {
	 */
	for ( i = 0; i < PAPI_MAX_PRESET_EVENTS; i++ ) {
		if ( ( _papi_hwi_presets.info[i].symbol )
			 && ( strcasecmp( _papi_hwi_presets.info[i].symbol, in ) == 0 ) ) {
			*out = ( int ) ( i | PAPI_PRESET_MASK );
			papi_return( PAPI_OK );
		}
	}
	papi_return( _papi_hwi_native_name_to_code( in, out ) );
}

/* Updates EventCode to next valid value, or returns error; 
  modifier can specify {all / available} for presets, or other values for native tables 
  and may be platform specific (Major groups / all mask bits; P / M / E chip, etc) */
/** @class PAPI_enum_event
  *	enumerate PAPI preset or native events 
 *
 *	@param EventCode
 *		a defined preset or native event such as PAPI_TOT_INS.
 *	@param modifier 
 *		modifies the search logic. For preset events, 
 *		TRUE specifies available events only. 
 *		For native events, each platform behaves differently. 
 *		See platform-specific documentation for details
 *
 *	@retval PAPI_ENOEVNT 
 *		The next requested PAPI preset or native event is not available on 
 *		the underlying hardware. 
 *
 *	Given a preset or native event code, PAPI_enum_event() replaces the event 
 *	code with the next available event in either the preset or native table. 
 *	The modifier argument affects which events are returned. 
 *	For all platforms and event types, a value of PAPI_ENUM_ALL (zero) 
 *	directs the function to return all possible events. 
 *
 *	For preset events, a TRUE (non-zero) value currently directs the function 
 *	to return event codes only for PAPI preset events available on this platform. 
 *	This may change in the future. 
 *	For native events, the effect of the modifier argument is different on each platform. 
 *	See the discussion below for platform-specific definitions. 
 *
 *	PENTIUM 4
 *	The following values are implemented for modifier on Pentium 4: 
 *	PAPI_PENT4_ENUM_GROUPS - 45 groups + custom + user event types PAPI_PENT4_ENUM_COMBOS 
 *	- all combinations of mask bits for given group PAPI_PENT4_ENUM_BITS 
 *	- all individual bits for a given group
 *
 *	ITANIUM
 *	The following values are implemented for modifier on Itanium: 
 *	<ul>
 *		<li> PAPI_ITA_ENUM_IARR - Enumerate IAR (instruction address ranging) events 
 *		<li> PAPI_ITA_ENUM_DARR - Enumerate DAR (data address ranging) events 
 *		<li> PAPI_ITA_ENUM_OPCM - Enumerate OPC (opcode matching) events 
 *		<li> PAPI_ITA_ENUM_IEAR - Enumerate IEAR (instr event address register) events 
 *		<li> PAPI_ITA_ENUM_DEAR - Enumerate DEAR (data event address register) events
 *	</ul>
 *
 *	POWER 4
 *	The following values are implemented for modifier on POWER 4: 
 *	<ul>
 *		<li> PAPI_PWR4_ENUM_GROUPS - Enumerate groups to which an event belongs 
 *	</ul>
 *
 *	@see PAPI_get_event_info PAPI_event_name_to_code PAPI_preset PAPI_native
 */
int
PAPI_enum_event( int *EventCode, int modifier )
{
	int i = *EventCode;
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	if ( _papi_hwi_invalid_cmp( cidx ) ||
		 ( ( i & PAPI_PRESET_MASK ) && cidx > 0 ) )
		return ( PAPI_ENOCMP );

	if ( i & PAPI_PRESET_MASK ) {
		if ( modifier == PAPI_ENUM_FIRST ) {
			*EventCode = ( int ) PAPI_PRESET_MASK;
			return ( PAPI_OK );
		}
		i &= PAPI_PRESET_AND_MASK;
		while ( ++i < PAPI_MAX_PRESET_EVENTS ) {
			if ( _papi_hwi_presets.info[i].symbol == NULL )
				return ( PAPI_ENOEVNT );	/* NULL pointer terminates list */
			if ( modifier & PAPI_PRESET_ENUM_AVAIL ) {
				if ( _papi_hwi_presets.count[i] == 0 )
					continue;
			}
			*EventCode = ( int ) ( i | PAPI_PRESET_MASK );
			return ( PAPI_OK );
		}
	} else if ( i & PAPI_NATIVE_MASK ) {
		/* Should check against num native events here */
		return ( _papi_hwd[cidx]->
				 ntv_enum_events( ( unsigned int * ) EventCode, modifier ) );
	}
	papi_return( PAPI_EINVAL );
}

/** @class PAPI_create_eventset
  *	create a new empty PAPI event set 
  * 
  * @param EventSet
  *		Address of an integer location to store the new EventSet handle
  *
  *	@exception PAPI_EINVAL 
  *		The argument handle has not been initialized to PAPI_NULL or the argument is a NULL pointer.
  *
  *	@exception PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * PAPI_create_eventset() creates a new EventSet pointed to by EventSet, 
  * which must be initialized to PAPI_NULL before calling this routine. 
  * The user may then add hardware events to the event set by calling 
  * @ref PAPI_add_event() or similar routines. 
  * NOTE: PAPI-C uses a late binding model to bind EventSets to components. 
  * When an EventSet is first created it is not bound to a component. 
  * This will cause some API calls that modify EventSet options to fail. 
  * An EventSet can be bound to a component explicitly by calling 
  * @ref PAPI_assign_eventset_component() or implicitly by calling 
  * @ref PAPI_add_event() or similar routines. 
  *
  * @see PAPI_add_event() 
  * @see PAPI_assign_eventset_component()
  * @see PAPI_destroy_eventset()
  * @see PAPI_cleanup_eventset()
  */
int
PAPI_create_eventset( int *EventSet )
{
   APIDBG("Entry: EventSet: %p\n", EventSet);
	ThreadInfo_t *master;
	int retval;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	retval = _papi_hwi_lookup_or_create_thread( &master, 0 );
	if ( retval )
		papi_return( retval );

	papi_return( _papi_hwi_create_eventset( EventSet, master ) );
}

/** @class PAPI_assign_eventset_component
 *	assign a component index to an existing but empty EventSet 
 *	
 *	@param EventSet 
 *		An integer identifier for an existing EventSet 
 *	@param cidx 
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_ENOEVST 
 *		The EventSet doesn't exist.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 * PAPI_assign_eventset_component() assigns a specific component index, 
 * as specified by cidx, to a new EventSet identified by EventSet, as obtained 
 * from PAPI_create_eventset(). 
 * EventSets are ordinarily automatically bound to components when the first 
 * event is added. 
 * This routine is useful to explicitly bind an EventSet to a component before 
 * setting component related options.
 *
 * @see PAPI_set_opt() PAPI_create_eventset() PAPI_add_events() PAPI_set_multiplex()
 */
int
PAPI_assign_eventset_component( int EventSet, int cidx )
{
	EventSetInfo_t *ESI;
	int retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

/* validate cidx */
	retval = valid_component( cidx );
	if ( retval < 0 )
		papi_return( retval );

/* cowardly refuse to reassign eventsets */ 
	if ( ESI->CmpIdx >= 0 )
	  return PAPI_EINVAL;

	return ( _papi_hwi_assign_eventset( ESI, cidx ) );
}


int
PAPI_add_pevent( int EventSet, int code, void *inout )
{
	EventSetInfo_t *ESI;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* Of course, it must be stopped in order to modify it. */

	if ( !( ESI->state & PAPI_STOPPED ) )
		papi_return( PAPI_EISRUN );

	/* No multiplexing pevents. */

	if ( ESI->state & PAPI_MULTIPLEXING )
		papi_return( PAPI_EINVAL );

	/* Now do the magic. */

	papi_return( _papi_hwi_add_pevent( ESI, code, inout ) );
}

/** @class PAPI_add_event
 *	add PAPI preset or native hardware event to an event set 
 *
 *	@param EventSet
 *		an integer handle for a PAPI Event Set as created by PAPI_create_eventset()
 *	@param EventCode 
 *		a defined event such as PAPI_TOT_INS. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the event set simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware.
 *	@retval PAPI_EBUG 
 *		Internal error, please send mail to the developers. 
 *
 *	PAPI_add_event() adds one event to a PAPI Event Set.
 *	A hardware event can be either a PAPI preset or a native hardware event code. 
 *	For a list of PAPI preset events, see PAPI_presets() or run the avail test 
 *	case in the PAPI distribution. 
 *	PAPI presets can be passed to PAPI_query_event() to see if they exist on 
 *	the underlying architecture. 
 *	For a list of native events available on current platform, run native_avail 
 *	test case in the PAPI distribution. 
 *	For the encoding of native events, see PAPI_event_name_to_code() to learn 
 *	how to generate native code for the supported native event on the underlying architecture. 
 *
 * @see PAPI_cleanup_eventset() PAPI_destroy_eventset() PAPI_event_code_to_name() PAPI_remove_events() PAPI_query_event() PAPI_presets() PAPI_native() PAPI_remove_event()
 */
int
PAPI_add_event( int EventSet, int EventCode )
{
   APIDBG("Entry: EventSet: %d, EventCode: 0x%x\n", EventSet, EventCode);
	EventSetInfo_t *ESI;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* Check argument for validity */

	if ( ( ( EventCode & PAPI_PRESET_MASK ) == 0 ) &&
		 ( EventCode & PAPI_NATIVE_MASK ) == 0 )
		papi_return( PAPI_EINVAL );

	/* Of course, it must be stopped in order to modify it. */

	if ( ESI->state & PAPI_RUNNING )
		papi_return( PAPI_EISRUN );

	/* Now do the magic. */

	papi_return( _papi_hwi_add_event( ESI, EventCode ) );
}

/** @class PAPI_remove_event
 * @brief removes a hardware event from a PAPI event set. 
 *
 * A hardware event can be either a PAPI Preset or a native hardware event code. 
 * For a list of PAPI preset events, see PAPI_presets or run the papi_avail utility in the PAPI distribution. 
 * PAPI Presets can be passed to PAPI_query_event to see if they exist on the underlying architecture. 
 * For a list of native events available on current platform, run papi_native_avail in the PAPI distribution. 
 *
 *	@par C Prototype:
 *		#include <papi.h> @n
 *		int PAPI_remove_event( int  EventSet, int  EventCode );
 *
 *	@par Fortran Prototype:
 *		#include fpapi.h @n
 *		PAPIF_remove_event( C_INT  EventSet,  C_INT  EventCode,  C_INT  check )
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param EventCode
 *		a defined event such as PAPI_TOT_INS or a native event. 
 *
 *	@retval PAPI_OK 
 *		Everything worked.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other 
 *		events in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	@par Example:
 *	@code
int EventSet = PAPI_NULL;
int ret;

// Create an empty EventSet
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Add Total Instructions Executed to our EventSet
ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
if (ret != PAPI_OK) handle_error(ret);

// Start counting
ret = PAPI_start(EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Stop counting, ignore values
ret = PAPI_stop(EventSet, NULL);
if (ret != PAPI_OK) handle_error(ret);

// Remove event
ret = PAPI_remove_event(EventSet, PAPI_TOT_INS);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see PAPI_cleanup_eventset PAPI_destroy_eventset PAPI_event_name_to_code 
 *		PAPI_presets PAPI_add_event PAPI_add_events
 */
int
PAPI_remove_event( int EventSet, int EventCode )
{
	EventSetInfo_t *ESI;
	int i;

	/* check for pre-existing ESI */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* Check argument for validity */

	if ( ( ( EventCode & PAPI_PRESET_MASK ) == 0 ) &&
		 ( EventCode & PAPI_NATIVE_MASK ) == 0 )
		papi_return( PAPI_EINVAL );

	/* Of course, it must be stopped in order to modify it. */

	if ( !( ESI->state & PAPI_STOPPED ) )
		papi_return( PAPI_EISRUN );

	/* if the state is PAPI_OVERFLOWING, you must first call
	   PAPI_overflow with threshold=0 to remove the overflow flag */

	/* Turn off the even that is overflowing */
	if ( ESI->state & PAPI_OVERFLOWING ) {
		for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
			if ( ESI->overflow.EventCode[i] == EventCode ) {
				PAPI_overflow( EventSet, EventCode, 0, 0,
							   ESI->overflow.handler );
				break;
			}
		}
	}

	/* force the user to call PAPI_profil to clear the PAPI_PROFILING flag */
	if ( ESI->state & PAPI_PROFILING ) {
		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode ) {
				PAPI_sprofil( NULL, 0, EventSet, EventCode, 0, 0 );
				break;
			}
		}
	}

	/* Now do the magic. */

	papi_return( _papi_hwi_remove_event( ESI, EventCode ) );
}

/** @class PAPI_destroy_eventset
 *	deallocates memory associated with an empty PAPI event set 
 *  
 *	@param *EventSet 
 *		a pointer to the integer handle for a PAPI event set as created by PAPI_create_eventset(). 
 *		The value pointed to by EventSet is then set to PAPI_NULL on success. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *		Attempting to destroy a non-empty event set or passing in a null 
 *		pointer to be destroyed.
 *
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *
 *	@retval PAPI_EBUG 
 *		Internal error, send mail to ptools-perfapi@ptools.org and complain. 
 */
int
PAPI_destroy_eventset( int *EventSet )
{
	EventSetInfo_t *ESI;

	/* check for pre-existing ESI */

	if ( EventSet == NULL )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( *EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	if ( !( ESI->state & PAPI_STOPPED ) )
		papi_return( PAPI_EISRUN );

	if ( ESI->NumberOfEvents )
		papi_return( PAPI_EINVAL );

	_papi_hwi_remove_EventSet( ESI );
	*EventSet = PAPI_NULL;

	return ( PAPI_OK );
}

/* simply checks for valid EventSet, calls substrate start() call */
/** @class PAPI_start
 *	start counting hardware events in an event set 
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events. ( PAPI_start() only)
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the EventSet simultaneously. ( PAPI_start() only)
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_start starts counting all of the hardware events contained in the previously defined EventSet. 
 *	All counters are implicitly set to zero before counting.
 *
 *	@see  PAPI_create_eventset PAPI_add_event
 */
int
PAPI_start( int EventSet )
{
	APIDBG("Entry: EventSet: %d\n", EventSet);
	int i;
	int is_dirty=0;
	int retval;
	EventSetInfo_t *ESI;
	ThreadInfo_t *thread = NULL;
	CpuInfo_t *cpu = NULL;
	hwd_context_t *context;
	int cidx;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );
	
	/* only one event set per thread/cpu can be running at any time, */
	/* so if another event set is running, the user must stop that   */
        /* event set explicitly */

	/* check cpu attached case first */
	if (ESI->state & PAPI_CPU_ATTACHED) {
	   cpu = ESI->CpuInfo;
	   if ( cpu->running_eventset[cidx] ) {
	      papi_return( PAPI_EISRUN );
	   }
	} else {
      	    thread = ESI->master;
	    if ( thread->running_eventset[cidx] ) {
	       papi_return( PAPI_EISRUN );
	    }
	} 
	
	/* Check that there are added events */
	if ( ESI->NumberOfEvents < 1 )
		papi_return( PAPI_EINVAL );

	/* If multiplexing is enabled for this eventset,
	   call John May's code. */

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = MPX_start( ESI->multiplex.mpx_evset );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* Update the state of this EventSet */
		ESI->state ^= PAPI_STOPPED;
		ESI->state |= PAPI_RUNNING;

		return ( PAPI_OK );
	}

	/* get the context we should use for this event set */
	context = _papi_hwi_get_context( ESI, &is_dirty );
	if (is_dirty) {
		/* we need to reset the context state because it was last used for some */
		/* other event set and does not contain the information for our events. */
		retval = _papi_hwd[ESI->CmpIdx]->update_control_state( ESI->ctl_state,
														  ESI->NativeInfoArray,
														  ESI->NativeCount,
														  context);
		if ( retval != PAPI_OK ) {
			papi_return( retval );
		}
		
		/* now that the context contains this event sets information, */
		/* make sure the position array in the EventInfoArray is correct */
		for ( i=0 ; i<ESI->NativeCount ; i++ ) {
			_papi_hwi_remap_event_position( ESI, i, ESI->NumberOfEvents );
		}
	}

	/* If overflowing is enabled, turn it on */
	if ( ( ESI->state & PAPI_OVERFLOWING ) &&
		 !( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) ) {
		retval =
			_papi_hwi_start_signal( _papi_hwd[cidx]->cmp_info.itimer_sig,
									NEED_CONTEXT, cidx );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* Update the state of this EventSet and thread before to avoid races */
		ESI->state ^= PAPI_STOPPED;
		ESI->state |= PAPI_RUNNING;
		thread->running_eventset[cidx] = ESI;   /* can not be attached to thread or cpu if overflowing */

		retval = _papi_hwd[cidx]->start( context, ESI->ctl_state );
		if ( retval != PAPI_OK ) {
			_papi_hwi_stop_signal( _papi_hwd[cidx]->cmp_info.itimer_sig );
			ESI->state ^= PAPI_RUNNING;
			ESI->state |= PAPI_STOPPED;
			thread->running_eventset[cidx] = NULL;
			papi_return( retval );
		}

		retval = _papi_hwi_start_timer( _papi_hwd[cidx]->cmp_info.itimer_num,
										_papi_hwd[cidx]->cmp_info.itimer_sig,
										_papi_hwd[cidx]->cmp_info.itimer_ns );
		if ( retval != PAPI_OK ) {
			_papi_hwi_stop_signal( _papi_hwd[cidx]->cmp_info.itimer_sig );
			_papi_hwd[cidx]->stop( context, ESI->ctl_state );
			ESI->state ^= PAPI_RUNNING;
			ESI->state |= PAPI_STOPPED;
			thread->running_eventset[cidx] = NULL;
			papi_return( retval );
		}
	} else {
		/* Update the state of this EventSet and thread before to avoid races */
		ESI->state ^= PAPI_STOPPED;
		ESI->state |= PAPI_RUNNING;
		/* if not attached to cpu or another process */
		if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
			if ( !( ESI->state & PAPI_ATTACHED ) ) {
				thread->running_eventset[cidx] = ESI;
			}
		} else {
			cpu->running_eventset[cidx] = ESI;
		}

		retval = _papi_hwd[cidx]->start( context, ESI->ctl_state );
		if ( retval != PAPI_OK ) {
			_papi_hwd[cidx]->stop( context, ESI->ctl_state );
			ESI->state ^= PAPI_RUNNING;
			ESI->state |= PAPI_STOPPED;
			if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
				if ( !( ESI->state & PAPI_ATTACHED ) ) 
					thread->running_eventset[cidx] = NULL;
			} else {
				cpu->running_eventset[cidx] = NULL;
			}
			papi_return( retval );
		}
	}

	return ( retval );
}

/* checks for valid EventSet, calls substrate stop() fxn. */
/** @class PAPI_stop
 *	stop counting hardware events in an event set  
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param values
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_ENOTRUN 
 *		The EventSet is currently not running. ( PAPI_stop() only) 
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_stop halts the counting of a previously defined event set and the 
 *	counter values contained in that EventSet are copied into the values array
 *	These calls assume an initialized PAPI library and a properly added event set. 
 *
 *	@see  PAPI_create_eventset PAPI_add_event
 */
int
PAPI_stop( int EventSet, long long *values )
{
   APIDBG("Entry: EventSet: %d, values: %p\n", EventSet, values);
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( !( ESI->state & PAPI_RUNNING ) )
		papi_return( PAPI_ENOTRUN );

	/* If multiplexing is enabled for this eventset, turn if off */

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = MPX_stop( ESI->multiplex.mpx_evset, values );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* Update the state of this EventSet */

		ESI->state ^= PAPI_RUNNING;
		ESI->state |= PAPI_STOPPED;

		return ( PAPI_OK );
	}

	/* get the context we should use for this event set */
	context = _papi_hwi_get_context( ESI, NULL );
	/* Read the current counter values into the EventSet */
	retval = _papi_hwi_read( context, ESI, ESI->sw_stop );
	if ( retval != PAPI_OK )
		papi_return( retval );

	/* Remove the control bits from the active counter config. */
	retval = _papi_hwd[cidx]->stop( context, ESI->ctl_state );
	if ( retval != PAPI_OK )
		papi_return( retval );
	if ( values )
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );

	/* If kernel profiling is in use, flush and process the kernel buffer */

	if ( ESI->state & PAPI_PROFILING ) {
		if ( _papi_hwd[cidx]->cmp_info.kernel_profile &&
			 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
			retval = _papi_hwd[cidx]->stop_profiling( ESI->master, ESI );
			if ( retval < PAPI_OK )
				papi_return( retval );
		}
	}

	/* If overflowing is enabled, turn it off */

	if ( ESI->state & PAPI_OVERFLOWING ) {
		if ( !( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) ) {
			retval =
				_papi_hwi_stop_timer( _papi_hwd[cidx]->cmp_info.itimer_num,
									  _papi_hwd[cidx]->cmp_info.itimer_sig );
			if ( retval != PAPI_OK )
				papi_return( retval );
			_papi_hwi_stop_signal( _papi_hwd[cidx]->cmp_info.itimer_sig );
		}
	}

	/* Update the state of this EventSet */

	ESI->state ^= PAPI_RUNNING;
	ESI->state |= PAPI_STOPPED;

	/* Update the running event set for this thread */
	if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
		if ( !( ESI->state & PAPI_ATTACHED ))
			ESI->master->running_eventset[cidx] = NULL;
	} else {
		ESI->CpuInfo->running_eventset[cidx] = NULL;
	}
	
#if defined(DEBUG)
	if ( _papi_hwi_debug & DEBUG_API ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ )
			APIDBG( "PAPI_stop ESI->sw_stop[%d]:\t%llu\n", i, ESI->sw_stop[i] );
	}
#endif

	return ( PAPI_OK );
}

/** @class PAPI_reset
 * @brief reset the hardware event counts in an event set 
 *
 *	PAPI_reset() zeroes the values of the counters contained in EventSet. 
 *	This call assumes an initialized PAPI library and a properly added event set 
 *
 *	@par C Prototype:
 *		#include <papi.h> @n
 *		int PAPI_reset( int EventSet );
 *
 *	@par Fortran Prototype:
 *		#include fpapi.h @n
 *		PAPIF_reset( C_INT  EventSet,  C_INT  check )
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset 
 *
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist. 
 *
 *	@par Example:
 *	@code
 int EventSet = PAPI_NULL;
 int Events[] = {PAPI_TOT_INS, PAPI_FP_OPS};
 int ret;
 
// Create an empty EventSet
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Add two events to our EventSet
ret = PAPI_add_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);

// Start counting
ret = PAPI_start(EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Stop counting, ignore values
ret = PAPI_stop(EventSet, NULL);
if (ret != PAPI_OK) handle_error(ret);

// reset the counters in this EventSet
ret = PAPI_reset(EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see PAPI_create_eventset
 */
int
PAPI_reset( int EventSet )
{
	int retval = PAPI_OK;
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
			retval = MPX_reset( ESI->multiplex.mpx_evset );
		} else {
			/* If we're not the only one running, then just
			   read the current values into the ESI->start
			   array. This holds the starting value for counters
			   that are shared. */
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwd[cidx]->reset( context, ESI->ctl_state );
		}
	} else {
#ifdef __bgp__
		//  For BG/P, we always want to reset the 'real' hardware counters.  The counters
		//  can be controlled via multiple interfaces, and we need to ensure that the values
		//  are truly zero...
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( ESI, NULL );
		retval = _papi_hwd[cidx]->reset( context, ESI->ctl_state );
#endif
		memset( ESI->sw_stop, 0x00,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

	APIDBG( "PAPI_reset returns %d\n", retval );
	papi_return( retval );
}

/** @class PAPI_read
 *	read hardware counters from an event set 
 *
 *	@param EventSet
 *		an integer handle for a PAPI Event Set 
 *		as created by PAPI_create_eventset
 *	@param values 
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist. 
 *	
 *	These calls assume an initialized PAPI library and a properly added event set. 
 *	PAPI_read() copies the counters of the indicated event set into the array values. 
 *	The counters continue counting after the read. 
 *
 * @see  PAPI_start PAPI PAPIF PAPI_set_opt PAPI_reset
 */
int
PAPI_read( int EventSet, long long *values )
{
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval = PAPI_OK;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, values, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, values );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	} else {
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

#if defined(DEBUG)
	if ( ISLEVEL( DEBUG_API ) ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ )
			APIDBG( "PAPI_read values[%d]:\t%lld\n", i, values[i] );
	}
#endif

	APIDBG( "PAPI_read returns %d\n", retval );
	return ( PAPI_OK );
}

int
PAPI_read_ts( int EventSet, long long *values, long long *cyc )
{
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval = PAPI_OK;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, values, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, values );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	} else {
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

	*cyc = _papi_hwd[cidx]->get_real_cycles(  );

#if defined(DEBUG)
	if ( ISLEVEL( DEBUG_API ) ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ )
			APIDBG( "PAPI_read values[%d]:\t%lld\n", i, values[i] );
	}
#endif

	APIDBG( "PAPI_read_ts returns %d\n", retval );
	return ( PAPI_OK );
}

/** @class PAPI_accum
 *	accumulate and reset counters in an event set 
 *	
 *	@param EventSet
 *		an integer handle for a PAPI Event Set 
 *		as created by PAPI_create_eventset
 *	@param values 
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist. 
 *
 * These calls assume an initialized PAPI library and a properly added event set. 
 * PAPI_accum() adds the counters of the indicated event set into the array values. 
 * The counters are zeroed and continue counting after the operation.
 * Note the differences between PAPI_read() and PAPI_accum(), specifically 
 * that PAPI_accum() resets the values array to zero. 
 *
 * @see  PAPI_start PAPI PAPIF PAPI_set_opt PAPI_reset
 */
int
PAPI_accum( int EventSet, long long *values )
{
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int i, cidx, retval;
	long long a, b, c;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, ESI->sw_stop, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, ESI->sw_stop );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	}

	for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
		a = ESI->sw_stop[i];
		b = values[i];
		c = a + b;
		values[i] = c;
	}

	papi_return( PAPI_reset( EventSet ) );
}

/** @class PAPI_write
 *	Write counter values into counters 
 *
 *	@param EventSet 
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param *values
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_ESBSTR 
 *		PAPI_write() is not implemented for this architecture. 
 *		PAPI_ESYS The EventSet is currently counting events and 
 *		the substrate could not change the values of the running counters.
 *
 *	PAPI_write() writes the counter values provided in the array values 
 *	into the event set EventSet. 
 *	The virtual counters managed by the PAPI library will be set to the values provided. 
 *	If the event set is running, an attempt will be made to write the values 
 *	to the running counters. 
 *	This operation is not permitted by all substrates and may result in a run-time error. 
 *
 *	@see PAPI_read
 */
int
PAPI_write( int EventSet, long long *values )
{
	int cidx, retval = PAPI_OK;
	EventSetInfo_t *ESI;
	hwd_context_t *context;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( ESI, NULL );
		retval = _papi_hwd[cidx]->write( context, ESI->ctl_state, values );
		if ( retval != PAPI_OK )
			return ( retval );
	}

	memcpy( ESI->hw_start, values,
			( size_t ) _papi_hwd[cidx]->cmp_info.num_cntrs *
			sizeof ( long long ) );

	return ( retval );
}

/** @class PAPI_cleanup_eventset
 *empty and destroy an EventSet 
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *		Attempting to destroy a non-empty event set or passing in a null pointer to be destroyed.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_EBUG 
 *		Internal error, send mail to ptools-perfapi@ptools.org and complain. 
 *
 * PAPI_cleanup_eventset removes all events from a PAPI event set and turns 
 * off profiling and overflow for all events in the eventset. 
 * This can not be called if the EventSet is not stopped.
 *
 * @see PAPI_profil PAPI_create_eventset PAPI_add_event PAPI_stop
 */
int
PAPI_cleanup_eventset( int EventSet )
{
	EventSetInfo_t *ESI;
	int i, cidx, total, retval;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* if the eventset has no index and no events, return OK
	   otherwise return NOCMP */
	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 ) {
		if ( ESI->NumberOfEvents )
			papi_return( cidx );
		papi_return( PAPI_OK );
	}

	/* Of course, it must be stopped in order to modify it. */

	if ( ESI->state & PAPI_RUNNING )
		papi_return( PAPI_EISRUN );

	/* clear overflow flag and turn off hardware overflow handler */
	if ( ESI->state & PAPI_OVERFLOWING ) {
		total = ESI->overflow.event_counter;
		for ( i = 0; i < total; i++ ) {
			retval = PAPI_overflow( EventSet,
									ESI->overflow.EventCode[0], 0, 0, NULL );
			if ( retval != PAPI_OK )
				papi_return( retval );
		}
	}
	/* clear profile flag and turn off hardware profile handler */
	if ( ( ESI->state & PAPI_PROFILING ) &&
		 _papi_hwd[cidx]->cmp_info.hardware_intr &&
		 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
		total = ESI->profile.event_counter;
		for ( i = 0; i < total; i++ ) {
			retval =
				PAPI_sprofil( NULL, 0, EventSet, ESI->profile.EventCode[0], 0,
							  PAPI_PROFIL_POSIX );
			if ( retval != PAPI_OK )
				papi_return( retval );
		}
	}

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = MPX_cleanup( &ESI->multiplex.mpx_evset );
		if ( retval != PAPI_OK )
			papi_return( retval );
	}

	/* Now do the magic */

	papi_return( _papi_hwi_cleanup_eventset( ESI ) );
}

/** @class PAPI_multiplex_init
 *	initialize multiplex support in the PAPI library 
 *
 *	PAPI_multiplex_init enables and initializes multiplex support in the PAPI library. 
 *	Multiplexing allows a user to count more events than total physical counters 
 *	by time sharing the existing counters at some loss in precision. 
 *	Applications that make no use of multiplexing do not need to call this routine. 
 *
 *	@see PAPI_set_multiplex PAPI_get_multiplex
 */
int
PAPI_multiplex_init( void )
{
	int retval;

	retval = mpx_init( _papi_hwd[0]->cmp_info.itimer_ns );
	papi_return( retval );
}

/** @class PAPI_state
 *	return the counting state of an EventSet 
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param status
 *		an integer containing a boolean combination of one or more of the 
 *		following nonzero constants as defined in the PAPI header file papi.h:
 *		PAPI_STOPPED	EventSet is stopped
 *		PAPI_RUNNING	EventSet is running
 *		PAPI_PAUSED	EventSet temporarily disabled by the library
 *		PAPI_NOT_INIT	EventSet defined, but not initialized
 *		PAPI_OVERFLOWING	EventSet has overflowing enabled
 *		PAPI_PROFILING	EventSet has profiling enabled
 *		PAPI_MULTIPLEXING	EventSet has multiplexing enabled
 *		PAPI_ACCUMULATING	reserved for future use
 *		PAPI_HWPROFILING	reserved for future use 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist. 
 *
 *	PAPI_state() returns the counting state of the specified event set. 
 *
 *	@see PAPI_stop PAPI_start
 */
int
PAPI_state( int EventSet, int *status )
{
	EventSetInfo_t *ESI;

	if ( status == NULL )
		papi_return( PAPI_EINVAL );

	/* check for good EventSetIndex value */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/*read status FROM ESI->state */

	*status = ESI->state;

	return ( PAPI_OK );
}

/** @class PAPI_set_debug
 *	set the current debug level for PAPI 
 *
 *	@param level
 *		one of the constants shown in the table below and defined in the papi.h 
 *		header file. 
 *		
 *	@retval PAPI_EINVAL The debuglevel is invalid. 
 *
 *	The current debug level is used by both the internal error and debug message 
 *	handler subroutines. 
 *	The debug handler is only used if the library was compiled with -DDEBUG. 
 *	The debug handler is called when there is an error upon a call to the PAPI API. 
 *	The error handler is always active and it's behavior cannot be modified except 
 *	for whether or not it prints anything. 
 *	
 *	NOTE: This is the ONLY function that may be called BEFORE PAPI_library_init(). 
 *	The PAPI error handler prints out messages in the following form: 
 *				PAPI Error: message. 
 *
 *	The default PAPI debug handler prints out messages in the following form:
 *		PAPI Error: Error Code code,symbol,description 
 *
 *	If the error was caused from a system call and the return code is PAPI_ESYS, 
 *	the message will have a colon space and the error string as reported by 
 *	strerror() appended to the end. 
 *	The possible debug levels for debugging are shown in the table below.
 *	<ul>
 *		<li>PAPI_QUIET	Do not print anything, just return the error code
 *		<li>PAPI_VERB_ECONT	Print error message and continue
 *		<li>PAPI_VERB_ESTOP	Print error message and exit 
 *	</ul>
 *
 *	@see  PAPI_library_init PAPI_get_opt PAPI_set_opt
 */
int
PAPI_set_debug( int level )
{
	PAPI_option_t option;

	memset( &option, 0x0, sizeof ( option ) );
	option.debug.level = level;
	option.debug.handler = _papi_hwi_debug_handler;
	return ( PAPI_set_opt( PAPI_DEBUG, &option ) );
}

/* Attaches to or detaches from the specified thread id */
inline_static int
_papi_set_attach( int option, int EventSet, unsigned long tid )
{
	PAPI_option_t attach;

	memset( &attach, 0x0, sizeof ( attach ) );
	attach.attach.eventset = EventSet;
	attach.attach.tid = tid;
	return ( PAPI_set_opt( option, &attach ) );
}

/** @class PAPI_attach
 *	attach PAPI event set to the specified thread id 
 *	
 *	@param EventSet 
 *		an integer handle for a PAPI Event Set as created by PAPI_create_eventset)
 *	@param tid 
 *		a thread id as obtained from, for example, PAPI_list_threads or PAPI_thread_id . 
 *
 *	@retval PAPI_ESBSTR 
 *		This feature is unsupported on this substrate.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 * PAPI_attach() is a wrapper function that calls PAPI_set_opt to allow PAPI 
 * to monitor performance counts on a thread other than the one currently executing. 
 * This is sometimes referred to as third party monitoring. 
 * PAPI_attach() connects the specified EventSet to the specifed thread
 *
 * @see PAPI_set_opt PAPI_list_threads PAPI_thread_id PAPI_thread_init
 */
int
PAPI_attach( int EventSet, unsigned long tid )
{
	return ( _papi_set_attach( PAPI_ATTACH, EventSet, tid ) );
}

/** @class PAPI_detach
 *	detach PAPI event set from previously specified thread id and restore to executing thread 
 *	
 *	@param EventSet 
 *		an integer handle for a PAPI Event Set as created by PAPI_create_eventset
 *
 *	@retval PAPI_ESBSTR 
 *		This feature is unsupported on this substrate.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 * PAPI_detach() is a wrapper function that calls PAPI_set_opt to allow PAPI 
 * to monitor performance counts on a thread other than the one currently executing. 
 * This is sometimes referred to as third party monitoring. 
 * PAPI_detach() breaks that connection and restores the EventSet to the original executing thread. 
 *
 * @see PAPI_set_opt PAPI_list_threads PAPI_thread_id PAPI_thread_init
 */
int
PAPI_detach( int EventSet )
{
	return ( _papi_set_attach( PAPI_DETACH, EventSet, 0 ) );
}

/** @class PAPI_set_multiplex
 *	convert a standard event set to a multiplexed event set 
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid, or the EventSet 
 *		is already multiplexed.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 *	PAPI_set_multiplex converts a standard PAPI event set created by a call to 
 *	PAPI_create_eventset into an event set capable of handling multiplexed events. 
 *	This must be done after calling PAPI_multiplex_init  , 
 *	but prior to calling PAPI_start(). 
 *	
 *	Events can be added to an event set either before or after converting it 
 *	into a multiplexed set, but the conversion must be done prior to using it 
 *	as a multiplexed set. 
 *
 *	@see  PAPI_multiplex_init PAPI_set_opt PAPI_create_eventset
 */

int
PAPI_set_multiplex( int EventSet )
{
	PAPI_option_t mpx;
	EventSetInfo_t *ESI;
	int cidx;
	int ret;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );

	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* if the eventset has no index return NOCMP */
	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ( ret = mpx_check( EventSet ) ) != PAPI_OK )
		papi_return( ret );

	memset( &mpx, 0x0, sizeof ( mpx ) );
	mpx.multiplex.eventset = EventSet;
	mpx.multiplex.flags = PAPI_MULTIPLEX_DEFAULT;
	mpx.multiplex.ns = _papi_hwd[cidx]->cmp_info.itimer_ns;
	return ( PAPI_set_opt( PAPI_MULTIPLEX, &mpx ) );
}

/** @class PAPI_set_opt
 *	set PAPI library or event set options 
 *
 *	@param	option
 *		is an input parameter describing the course of action. 
 *		Possible values are defined in papi.h and briefly described in the table below. 
 *		The Fortran calls are implementations of specific options.
 *
 *	@param ptr
 *		is a pointer to a structure that acts as both an input and output parameter.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *
 *	PAPI_get_opt() and PAPI_set_opt() query or change the options of the PAPI 
 *	library or a specific event set created by PAPI_create_eventset . 
 *	Some options may require that the eventset be bound to a component before 
 *	they can execute successfully. 
 *	This can be done either by adding an event or by explicitly calling 
 *	PAPI_assign_eventset_component . 
 *	
 *	The C interface for these functions passes a pointer to the PAPI_option_t structure. 
 *	Not all options require or return information in this structure, and not all 
 *	options are implemented for both get and set. 
 *	Some options require a component index to be provided. 
 *	These options are handled explicitly by the PAPI_get_cmp_opt() call for 'get' 
 *	and implicitly through the option structure for 'set'. 
 *	The Fortran interface is a series of calls implementing various subsets of 
 *	the C interface. Not all options in C are available in Fortran.
 *	NOTE: Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX, 
 *	are also available as separate entry points in both C and Fortran.
 *
 *	The reader is urged to see the example code in the PAPI distribution for usage of PAPI_get_opt. 
 *	The file papi.h contains definitions for the structures unioned in the PAPI_option_t structure. 
 *
 *	@see PAPI_set_debug PAPI_set_multiplex PAPI_set_domain PAPI_option_t
 */
int
PAPI_set_opt( int option, PAPI_option_t * ptr )
{
	APIDBG("Entry:  option: %d, ptr: %p\n", option, ptr);

	_papi_int_option_t internal;
	int retval = PAPI_OK;
	hwd_context_t *context;
	int cidx;

	if ( ( option != PAPI_DEBUG ) && ( init_level == PAPI_NOT_INITED ) )
		papi_return( PAPI_ENOINIT );
	if ( ptr == NULL )
		papi_return( PAPI_EINVAL );

	memset( &internal, 0x0, sizeof ( _papi_int_option_t ) );

	switch ( option ) {
	case PAPI_DETACH:
	{
		internal.attach.ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( internal.attach.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.attach.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.attach == 0 )
			papi_return( PAPI_ESBSTR );

		/* if attached to a cpu, return an error */
		if (internal.attach.ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ESBSTR );

		if ( ( internal.attach.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		if ( ( internal.attach.ESI->state & PAPI_ATTACHED ) == 0 )
			papi_return( PAPI_EINVAL );

		internal.attach.tid = internal.attach.ESI->attach.tid;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.attach.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DETACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		internal.attach.ESI->state ^= PAPI_ATTACHED;
		internal.attach.ESI->attach.tid = 0;
		return ( PAPI_OK );
	}
	case PAPI_ATTACH:
	{
		internal.attach.ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( internal.attach.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.attach.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.attach == 0 )
			papi_return( PAPI_ESBSTR );

		if ( ( internal.attach.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		if ( internal.attach.ESI->state & PAPI_ATTACHED )
			papi_return( PAPI_EINVAL );

		/* if attached to a cpu, return an error */
		if (internal.attach.ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ESBSTR );

		internal.attach.tid = ptr->attach.tid;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.attach.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_ATTACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		internal.attach.ESI->state |= PAPI_ATTACHED;
		internal.attach.ESI->attach.tid = ptr->attach.tid;

		_papi_hwi_lookup_or_create_thread( 
				      &(internal.attach.ESI->master), ptr->attach.tid );

		return ( PAPI_OK );
	}
	case PAPI_CPU_ATTACH:
	{
		APIDBG("eventset: %d, cpu_num: %d\n", ptr->cpu.eventset, ptr->cpu.cpu_num);
		internal.cpu.ESI = _papi_hwi_lookup_EventSet( ptr->cpu.eventset );
		if ( internal.cpu.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		internal.cpu.cpu_num = ptr->cpu.cpu_num;
		APIDBG("internal: %p, ESI: %p, cpu_num: %d\n", &internal, internal.cpu.ESI, internal.cpu.cpu_num);

		cidx = valid_ESI_component( internal.cpu.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.cpu == 0 )
			papi_return( PAPI_ESBSTR );

		// can not attach to a cpu if already attached to a process or 
		// counters set to be inherited by child processes
		if ( internal.cpu.ESI->state & (PAPI_ATTACHED | PAPI_INHERIT) )
			papi_return( PAPI_EINVAL );

		if ( ( internal.cpu.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		retval = _papi_hwi_lookup_or_create_cpu(&internal.cpu.ESI->CpuInfo, internal.cpu.cpu_num);
		if( retval != PAPI_OK) {
			papi_return( retval );
		}

		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.cpu.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_CPU_ATTACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* set to show this event set is attached to a cpu not a thread */
		internal.cpu.ESI->state |= PAPI_CPU_ATTACHED;
		return ( PAPI_OK );
	}
	case PAPI_DEF_MPX_NS:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->multiplex.ns < 0 )
			papi_return( PAPI_EINVAL );
		/* We should check the resolution here with the system, either
		   substrate if kernel multiplexing or PAPI if SW multiplexing. */
		internal.multiplex.ns = ( unsigned long ) ptr->multiplex.ns;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.cpu.ESI, NULL );
		/* Low level just checks/adjusts the args for this substrate */
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DEF_MPX_NS, &internal );
		if ( retval == PAPI_OK ) {
			_papi_hwd[cidx]->cmp_info.itimer_ns = ( int ) internal.multiplex.ns;
			ptr->multiplex.ns = ( int ) internal.multiplex.ns;
		}
		papi_return( retval );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->itimer.ns < 0 )
			papi_return( PAPI_EINVAL );
		internal.itimer.ns = ptr->itimer.ns;
		/* Low level just checks/adjusts the args for this substrate */
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_DEF_ITIMER_NS, &internal );
		if ( retval == PAPI_OK ) {
			_papi_hwd[cidx]->cmp_info.itimer_ns = internal.itimer.ns;
			ptr->itimer.ns = internal.itimer.ns;
		}
		papi_return( retval );
	}
	case PAPI_DEF_ITIMER:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->itimer.ns < 0 )
			papi_return( PAPI_EINVAL );
		memcpy( &internal.itimer, &ptr->itimer,
				sizeof ( PAPI_itimer_option_t ) );
		/* Low level just checks/adjusts the args for this substrate */
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_DEF_ITIMER, &internal );
		if ( retval == PAPI_OK ) {
			_papi_hwd[cidx]->cmp_info.itimer_num = ptr->itimer.itimer_num;
			_papi_hwd[cidx]->cmp_info.itimer_sig = ptr->itimer.itimer_sig;
			if ( ptr->itimer.ns > 0 )
				_papi_hwd[cidx]->cmp_info.itimer_ns = ptr->itimer.ns;
			/* flags are currently ignored, eventually the flags will be able
			   to specify whether or not we use POSIX itimers (clock_gettimer) */
		}
		papi_return( retval );
	}
	case PAPI_MULTIPLEX:
	{
		EventSetInfo_t *ESI;
		ESI = _papi_hwi_lookup_EventSet( ptr->multiplex.eventset );
	   
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );
	   
		if ( !( ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );
		if ( ESI->state & PAPI_MULTIPLEXING )
			papi_return( PAPI_EINVAL );

		if ( ptr->multiplex.ns < 0 )
			papi_return( PAPI_EINVAL );
		internal.multiplex.ESI = ESI;
		internal.multiplex.ns = ( unsigned long ) ptr->multiplex.ns;
		internal.multiplex.flags = ptr->multiplex.flags;
		if ( ( _papi_hwd[cidx]->cmp_info.kernel_multiplex ) &&
			 ( ( ptr->multiplex.flags & PAPI_MULTIPLEX_FORCE_SW ) == 0 ) ) {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwd[cidx]->ctl( context, PAPI_MULTIPLEX, &internal );
		}
		/* Kernel or PAPI may have changed this value so send it back out to the user */
		ptr->multiplex.ns = ( int ) internal.multiplex.ns;
		if ( retval == PAPI_OK )
			papi_return( _papi_hwi_convert_eventset_to_multiplex
						 ( &internal.multiplex ) );
		return ( retval );
	}
	case PAPI_DEBUG:
	{
		int level = ptr->debug.level;
		switch ( level ) {
		case PAPI_QUIET:
		case PAPI_VERB_ESTOP:
		case PAPI_VERB_ECONT:
			_papi_hwi_error_level = level;
			break;
		default:
			papi_return( PAPI_EINVAL );
		}
		_papi_hwi_debug_handler = ptr->debug.handler;
		return ( PAPI_OK );
	}
	case PAPI_DEFDOM:
	{
		int dom = ptr->defdomain.domain;
		if ( ( dom < PAPI_DOM_MIN ) || ( dom > PAPI_DOM_MAX ) )
			papi_return( PAPI_EINVAL );

		/* Change the global structure. The _papi_hwd_init_control_state function 
		   in the substrates gets information from the global structure instead of
		   per-thread information. */
		cidx = valid_component( ptr->defdomain.def_cidx );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the substrate supports */

		if ( dom == PAPI_DOM_ALL )
			dom = _papi_hwd[cidx]->cmp_info.available_domains;

		if ( dom & ~_papi_hwd[cidx]->cmp_info.available_domains )
			papi_return( PAPI_EINVAL );

		_papi_hwd[cidx]->cmp_info.default_domain = dom;

		return ( PAPI_OK );
	}
	case PAPI_DOMAIN:
	{
		int dom = ptr->domain.domain;
		if ( ( dom < PAPI_DOM_MIN ) || ( dom > PAPI_DOM_MAX ) )
			papi_return( PAPI_EINVAL_DOM );

		internal.domain.ESI = _papi_hwi_lookup_EventSet( ptr->domain.eventset );
		if ( internal.domain.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.domain.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the substrate supports */

		if ( dom == PAPI_DOM_ALL )
			dom = _papi_hwd[cidx]->cmp_info.available_domains;

		if ( dom & ~_papi_hwd[cidx]->cmp_info.available_domains )
			papi_return( PAPI_EINVAL_DOM );

		if ( !( internal.domain.ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );

		/* Try to change the domain of the eventset in the hardware */
		internal.domain.domain = dom;
		internal.domain.eventset = ptr->domain.eventset;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.domain.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DOMAIN, &internal );
		if ( retval < PAPI_OK )
			papi_return( retval );

		/* Change the domain of the eventset in the library */

		internal.domain.ESI->domain.domain = dom;

		return ( retval );
	}
	case PAPI_DEFGRN:
	{
		int grn = ptr->defgranularity.granularity;
		if ( ( grn < PAPI_GRN_MIN ) || ( grn > PAPI_GRN_MAX ) )
			papi_return( PAPI_EINVAL );

		cidx = valid_component( ptr->defgranularity.def_cidx );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Change the component structure. The _papi_hwd_init_control_state function 
		   in the components gets information from the global structure instead of
		   per-thread information. */

		/* Check what the substrate supports */

		if ( grn & ~_papi_hwd[cidx]->cmp_info.available_granularities )
			papi_return( PAPI_EINVAL );

		/* Make sure there is only 1 set. */
		if ( grn ^ ( 1 << ( ffs( grn ) - 1 ) ) )
			papi_return( PAPI_EINVAL );

		_papi_hwd[cidx]->cmp_info.default_granularity = grn;

		return ( PAPI_OK );
	}
	case PAPI_GRANUL:
	{
		int grn = ptr->granularity.granularity;

		if ( ( grn < PAPI_GRN_MIN ) || ( grn > PAPI_GRN_MAX ) )
			papi_return( PAPI_EINVAL );

		internal.granularity.ESI =
			_papi_hwi_lookup_EventSet( ptr->granularity.eventset );
		if ( internal.granularity.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.granularity.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the substrate supports */

		if ( grn & ~_papi_hwd[cidx]->cmp_info.available_granularities )
			papi_return( PAPI_EINVAL );

		/* Make sure there is only 1 set. */
		if ( grn ^ ( 1 << ( ffs( grn ) - 1 ) ) )
			papi_return( PAPI_EINVAL );

		internal.granularity.granularity = grn;
		internal.granularity.eventset = ptr->granularity.eventset;
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_GRANUL, &internal );
		if ( retval < PAPI_OK )
			return ( retval );

		internal.granularity.ESI->granularity.granularity = grn;
		return ( retval );
	}
	case PAPI_INHERIT:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		EventSetInfo_t *ESI;
		ESI = _papi_hwi_lookup_EventSet( ptr->inherit.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.inherit == 0 )
			papi_return( PAPI_ESBSTR );

		if ( ( ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		/* if attached to a cpu, return an error */
		if (ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ESBSTR );

		internal.inherit.ESI = ESI;
		internal.inherit.inherit = ptr->inherit.inherit;

		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.inherit.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_INHERIT, &internal );
		if ( retval < PAPI_OK )
			return ( retval );

		ESI->inherit.inherit = ptr->inherit.inherit;
		return ( retval );
	}
	case PAPI_DATA_ADDRESS:
	case PAPI_INSTR_ADDRESS:
	{

		EventSetInfo_t *ESI;

		ESI = _papi_hwi_lookup_EventSet( ptr->addr.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		internal.address_range.ESI = ESI;

		if ( !( internal.address_range.ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );

		/*set domain to be PAPI_DOM_USER */
		internal.address_range.domain = PAPI_DOM_USER;

		internal.address_range.start = ptr->addr.start;
		internal.address_range.end = ptr->addr.end;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.address_range.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, option, &internal );
		ptr->addr.start_off = internal.address_range.start_off;
		ptr->addr.end_off = internal.address_range.end_off;
		papi_return( retval );
	}
	default:
		papi_return( PAPI_EINVAL );
	}
}

/* Preserves API compatibility with older versions */
/** @class PAPI_num_hwctrs
 *	return the number of hardware counters on the cpu 
 *
 *	This is included to preserve backwards compatibility.
 */
int
PAPI_num_hwctrs( void )
{
	return ( PAPI_num_cmp_hwctrs( 0 ) );
}

/** @class PAPI_num_cmp_hwctrs
 *	return the number of hardware counters for the specified component 
 *
 *	@param cidx
 *		An integer identifier for a component. By convention, component 0 is always 
 *		the cpu component.
 *
 *	PAPI_num_cmp_hwctrs() returns the number of counters present in the 
 *	specified component. 
 *	By convention, component 0 is always the cpu. 
 *	This count does not include any special purpose registers or 
 *	other performance hardware. 
 *	PAPI_library_init must be called in order for this function to return 
 *	anything greater than 0. 
 */
int
PAPI_num_cmp_hwctrs( int cidx )
{
	return ( PAPI_get_cmp_opt( PAPI_MAX_HWCTRS, NULL, cidx ) );
}

/** @class PAPI_get_multiplex
 *	get the multiplexing status of specified event set 
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid, or the EventSet 
 *		is already multiplexed.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 *	PAPI_get_multiplex tests the state of the PAPI_MULTIPLEXING flag in the 
 *	specified event set, returning TRUE if a PAPI event set is multiplexed, or FALSE if not. 
 *
 *	@see  PAPI_multiplex_init PAPI_set_opt PAPI_create_eventset
 */
int
PAPI_get_multiplex( int EventSet )
{
	PAPI_option_t popt;
	int retval;

	popt.multiplex.eventset = EventSet;
	retval = PAPI_get_opt( PAPI_MULTIPLEX, &popt );
	if ( retval < 0 )
		retval = 0;
	return retval;
}

/** @class PAPI_get_opt
  *	get PAPI library or event set options 
  *
  *	@param	option
  *		is an input parameter describing the course of action. 
  *		Possible values are defined in papi.h and briefly described in the table below. 
  *		The Fortran calls are implementations of specific options.
  *
  *	@param ptr
  *		is a pointer to a structure that acts as both an input and output parameter.
  *
  *	@retval PAPI_EINVAL 
  *		One or more of the arguments is invalid. 
  *
  *	PAPI_get_opt() and PAPI_set_opt() query or change the options of the PAPI 
  *	library or a specific event set created by PAPI_create_eventset . 
  *	Some options may require that the eventset be bound to a component before 
  *	they can execute successfully. 
  *	This can be done either by adding an event or by explicitly calling 
  *	PAPI_assign_eventset_component . 
  *	
  *	The C interface for these functions passes a pointer to the PAPI_option_t structure. 
  *	Not all options require or return information in this structure, and not all 
  *	options are implemented for both get and set. 
  *	Some options require a component index to be provided. 
  *	These options are handled explicitly by the PAPI_get_cmp_opt() call for 'get' 
  *	and implicitly through the option structure for 'set'. 
  *	The Fortran interface is a series of calls implementing various subsets of 
  *	the C interface. Not all options in C are available in Fortran.
  *	NOTE: Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX, 
  *	are also available as separate entry points in both C and Fortran.
  *
  *	The reader is urged to see the example code in the PAPI distribution for usage of PAPI_get_opt. 
  *	The file papi.h contains definitions for the structures unioned in the PAPI_option_t structure. 
  *
  *	@see PAPI_set_debug PAPI_set_multiplex PAPI_set_domain PAPI_option_t
  */
int
PAPI_get_opt( int option, PAPI_option_t * ptr )
{
	EventSetInfo_t *ESI;

	if ( ( option != PAPI_DEBUG ) && ( init_level == PAPI_NOT_INITED ) )
		papi_return( PAPI_ENOINIT );

	switch ( option ) {
	case PAPI_DETACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->attach.tid = ESI->attach.tid;
		return ( ( ESI->state & PAPI_ATTACHED ) == 0 );
	}
	case PAPI_ATTACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->attach.tid = ESI->attach.tid;
		return ( ( ESI->state & PAPI_ATTACHED ) != 0 );
	}
	case PAPI_CPU_ATTACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->cpu.cpu_num = ESI->CpuInfo->cpu_num;
		return ( ( ESI->state & PAPI_CPU_ATTACHED ) != 0 );
	}
	case PAPI_DEF_MPX_NS:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->multiplex.ns = _papi_hwd[0]->cmp_info.itimer_ns;
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->itimer.ns = _papi_hwd[0]->cmp_info.itimer_ns;
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->itimer.itimer_num = _papi_hwd[0]->cmp_info.itimer_num;
		ptr->itimer.itimer_sig = _papi_hwd[0]->cmp_info.itimer_sig;
		ptr->itimer.ns = _papi_hwd[0]->cmp_info.itimer_ns;
		ptr->itimer.flags = 0;
		return ( PAPI_OK );
	}
	case PAPI_MULTIPLEX:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->multiplex.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->multiplex.ns = ESI->multiplex.ns;
		ptr->multiplex.flags = ESI->multiplex.flags;
		return ( ESI->state & PAPI_MULTIPLEXING ) != 0;
	}
	case PAPI_PRELOAD:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		memcpy( &ptr->preload, &_papi_hwi_system_info.preload_info,
				sizeof ( PAPI_preload_info_t ) );
		break;
	case PAPI_DEBUG:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->debug.level = _papi_hwi_error_level;
		ptr->debug.handler = _papi_hwi_debug_handler;
		break;
	case PAPI_CLOCKRATE:
		return ( ( int ) _papi_hwi_system_info.hw_info.mhz );
	case PAPI_MAX_CPUS:
		return ( _papi_hwi_system_info.hw_info.ncpu );
		/* For now, MAX_HWCTRS and MAX CTRS are identical.
		   At some future point, they may map onto different values.
		 */
	case PAPI_INHERIT:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->inherit.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->inherit.inherit = ESI->inherit.inherit;
		return ( PAPI_OK );
	}
	case PAPI_GRANUL:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->granularity.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->granularity.granularity = ESI->granularity.granularity;
		break;
	case PAPI_EXEINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->exe_info = &_papi_hwi_system_info.exe_info;
		break;
	case PAPI_HWINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->hw_info = &_papi_hwi_system_info.hw_info;
		break;

	case PAPI_DOMAIN:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->domain.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->domain.domain = ESI->domain.domain;
		return ( PAPI_OK );
	case PAPI_LIB_VERSION:
		return ( PAPI_VERSION );
/* The following cases all require a component index 
    and are handled by PAPI_get_cmp_opt() with cidx == 0*/
	case PAPI_MAX_HWCTRS:
	case PAPI_MAX_MPX_CTRS:
	case PAPI_DEFDOM:
	case PAPI_DEFGRN:
	case PAPI_SHLIBINFO:
	case PAPI_COMPONENTINFO:
		return ( PAPI_get_cmp_opt( option, ptr, 0 ) );
	default:
		papi_return( PAPI_EINVAL );
	}
	return ( PAPI_OK );
}

/** @class PAPI_get_cmp_opt
 *	get component specific PAPI options 
 *
 *	@param	option
 *		is an input parameter describing the course of action. 
 *		Possible values are defined in papi.h and briefly described in the table below. 
 *		The Fortran calls are implementations of specific options.
 *	@param ptr
 *		is a pointer to a structure that acts as both an input and output parameter.
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *
 *	PAPI_get_opt() and PAPI_set_opt() query or change the options of the PAPI 
 *	library or a specific event set created by PAPI_create_eventset . 
 *	Some options may require that the eventset be bound to a component before 
 *	they can execute successfully. 
 *	This can be done either by adding an event or by explicitly calling 
 *	PAPI_assign_eventset_component . 
 *	
 *	The C interface for these functions passes a pointer to the PAPI_option_t structure. 
 *	Not all options require or return information in this structure, and not all 
 *	options are implemented for both get and set. 
 *	Some options require a component index to be provided. 
 *	These options are handled explicitly by the PAPI_get_cmp_opt() call for 'get' 
 *	and implicitly through the option structure for 'set'. 
 *	The Fortran interface is a series of calls implementing various subsets of 
 *	the C interface. Not all options in C are available in Fortran.
 *	NOTE: Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX, 
 *	are also available as separate entry points in both C and Fortran.
 *
 *	The reader is urged to see the example code in the PAPI distribution for usage of PAPI_get_opt. 
 *	The file papi.h contains definitions for the structures unioned in the PAPI_option_t structure. 
 *
 *	@see PAPI_set_debug PAPI_set_multiplex PAPI_set_domain PAPI_option_t
 */

int
PAPI_get_cmp_opt( int option, PAPI_option_t * ptr, int cidx )
{
	switch ( option ) {
		/* For now, MAX_HWCTRS and MAX CTRS are identical.
		   At some future point, they may map onto different values.
		 */
	case PAPI_MAX_HWCTRS:
		return ( _papi_hwd[cidx]->cmp_info.num_cntrs );
	case PAPI_MAX_MPX_CTRS:
		return ( _papi_hwd[cidx]->cmp_info.num_mpx_cntrs );
	case PAPI_DEFDOM:
		return ( _papi_hwd[cidx]->cmp_info.default_domain );
	case PAPI_DEFGRN:
		return ( _papi_hwd[cidx]->cmp_info.default_granularity );
	case PAPI_SHLIBINFO:
	{
		int retval;
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		retval = _papi_hwd[cidx]->update_shlib_info( &_papi_hwi_system_info );
		ptr->shlib_info = &_papi_hwi_system_info.shlib_info;
		papi_return( retval );
	}
	case PAPI_COMPONENTINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->cmp_info = &( _papi_hwd[cidx]->cmp_info );
		return ( PAPI_OK );
	default:
		papi_return( PAPI_EINVAL );
	}
	return ( PAPI_OK );
}

/** @class PAPI_num_components
  *	Get the number of components available on the system 
  *
  * @post 
  *		initializes the library to PAPI_HIGH_LEVEL_INITED if necessary
  *
  * @return 
  *		Number of components available on the system
  *
  *	@code
// Query the library for a component count. 
printf("%d components installed., PAPI_num_components() );
  * @endcode
  */
int
PAPI_num_components( void )
{
	return ( papi_num_components );
}

/** @class PAPI_num_events
  *	return the number of events in an event set.
  * 
  * @param EventSet 
  *   an integer handle for a PAPI event set created by PAPI_create_eventset.
  *
  * PAPI_num_events() returns the number of preset events contained in an event set. 
  * The event set should be created by @ref PAPI_create_eventset() .
  */
int
PAPI_num_events( int EventSet )
{
	EventSetInfo_t *ESI;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( !ESI )
		papi_return( PAPI_ENOEVST );

#ifdef DEBUG
	/* Not necessary */
	if ( ESI->NumberOfEvents == 0 )
		papi_return( PAPI_EINVAL );
#endif

	return ( ESI->NumberOfEvents );
}

/** @class PAPI_shutdown
  *	finish using PAPI and free all related resources. 
  *
  * PAPI_shutdown() is an exit function used by the PAPI Library 
  * to free resources and shut down when certain error conditions arise. 
  * It is not necessary for the user to call this function, 
  * but doing so allows the user to have the capability to free memory 
  * and resources used by the PAPI Library. */
void
PAPI_shutdown( void )
{
	DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
	EventSetInfo_t *ESI;
	int i, j = 0;
	ThreadInfo_t *master;

	APIDBG( "Enter\n" );
	if ( init_retval == DEADBEEF ) {
		PAPIERROR( PAPI_SHUTDOWN_str );
		return;
	}

	MPX_shutdown(  );

	master = _papi_hwi_lookup_thread( 0 );

	/* Count number of running EventSets AND */
	/* Stop any running EventSets in this thread */

#ifdef DEBUG
  again:
#endif
	for ( i = 0; i < map->totalSlots; i++ ) {
		ESI = map->dataSlotArray[i];
		if ( ESI ) {
			if ( ESI->master == master ) {
				if ( ESI->state & PAPI_RUNNING )
					PAPI_stop( i, NULL );
				PAPI_cleanup_eventset( i );
				_papi_hwi_free_EventSet( ESI );
			} else if ( ESI->state & PAPI_RUNNING )
				j++;
		}
	}

	/* No locking required, we're just waiting for the others
	   to call shutdown or stop their eventsets. */

#ifdef DEBUG
	if ( j != 0 ) {
		PAPIERROR( PAPI_SHUTDOWN_SYNC_str );
		sleep( 1 );
		j = 0;
		goto again;
	}
#endif

	/* Shutdown the entire substrate */

	_papi_hwi_shutdown_highlevel(  );
	_papi_hwi_shutdown_global_internal(  );
	_papi_hwi_shutdown_global_threads(  );
	for ( i = 0; i < papi_num_components; i++ ) {
		_papi_hwd[i]->shutdown_substrate(  );
	}

	/* Now it is safe to call re-init */

	init_retval = DEADBEEF;
	init_level = PAPI_NOT_INITED;
	_papi_cleanup_all_memory(  );
}

/** @class PAPI_strerror
 *	convert PAPI error codes to strings, and return the error string to user. 
 *
 *	@param errorCode 
 *		the error code to interpret
 *
 *	@retval NULL 
 *		The input error code to PAPI_strerror() is invalid. 
 *
 *	PAPI_strerror() returns a pointer to the error message corresponding to the 
 *	error code code . 
 *	If the call fails the function returns the NULL pointer. 
 *	This function is not implemented in Fortran. 
 *
 *	@see  PAPI_perror PAPI_set_opt PAPI_get_opt PAPI_shutdown PAPI_set_debug
 */
char *
PAPI_strerror( int errorCode )
{
	if ( ( errorCode > 0 ) || ( -errorCode > PAPI_NUM_ERRORS ) )
		return ( NULL );

	return ( ( char * ) _papi_hwi_err[-errorCode].name );
}

/** @class PAPI_descr_error
 *	Return the PAPI error description string to user. 
 *
 *	@param errorCode 
 *		the error code to interpret
 *
 *	@retval NULL 
 *		The input error code to PAPI_descr_error() is invalid, 
 *		or the description string is empty. 
 *
 *	PAPI_descr_error() returns a pointer to the error message corresponding to the 
 *	error code code . 
 *	If the call fails the function returns the NULL pointer. 
 *	This function is not implemented in Fortran. 
 *
 *	@see  PAPI_strerror PAPI_perror
 */
char *
PAPI_descr_error( int errorCode )
{
	if ( ( errorCode > 0 ) || ( -errorCode > PAPI_NUM_ERRORS ) )
		return ( NULL );

	return ( ( char * ) _papi_hwi_err[-errorCode].descr );
}

/** @class PAPI_perror
 *	convert PAPI error codes to strings, and print error message to stderr. 
 *
 *	@param code  
 *      the error code to interpret 
 *  @param destination  
 *      "the error message in quotes"
 *  @param length 
 *      either 0 or strlen(destination)  
 * 
 *  @retval PAPI_EINVAL  
 *      One or more of the arguments to PAPI_perror() is invalid. 
 *  @retval NULL  
 *      The input error code to PAPI_strerror() is invalid. 
 *
 *	PAPI_perror() fills the string destination with the error message 
 *	corresponding to the error code code . 
 *	The function copies length worth of the error description string 
 *	corresponding to code into destination. 
 *	The resulting string is always null terminated. 
 *	If length is 0, then the string is printed on stderr. 
 *
 *	@see  PAPI_set_opt PAPI_get_opt PAPI_shutdown PAPI_set_debug
 */
int
PAPI_perror( int code, char *destination, int length )
{
	char *foo;

	foo = PAPI_strerror( code );
	if ( foo == NULL )
		papi_return( PAPI_EINVAL );

	if ( destination && ( length >= 0 ) )
		strncpy( destination, foo, ( unsigned int ) length );
	else
		fprintf( stderr, "%s\n", foo );

	return ( PAPI_OK );
}

/** @class PAPI_overflow
 *	set up an event set to begin registering overflows 
 *
 * @param EventSet
 *		an integer handle to a PAPI event set as created by @ref PAPI_create_eventset()
 * @param EventCode
 *		the preset or native event code to be set for overflow detection. 
 *		This event must have already been added to the EvenSet.
 * @param threshold
 *		the overflow threshold value for this EventCode.
 * @param flags
 *		bit map that controls the overflow mode of operation. 
 *		Set to @ref PAPI_OVERFLOW_FORCE_SW to force software overflowing, 
 *		even if hardware overflow support is available. 
 *		If hardware overflow support is available on a given system, it will be 
 *		the default mode of operation. 
 *		There are situations where it is advantageous to use software overflow instead. 
 *		Although software overflow is inherently less accurate, with more latency 
 *		and processing overhead, it does allow for overflowing on derived events, 
 *		and for the accurate recording of overflowing event counts. 
 *		These two features are typically not available with hardware overflow. 
 *		Only one type of overflow is allowed per event set, so setting one event 
 *		to hardware overflow and another to forced software overflow will result in an error being returned.
 *	@param handler
 *		pointer to the user supplied handler function to call upon overflow 
 *
 * PAPI_overflow() marks a specific EventCode in an EventSet to generate an 
 * overflow signal after every threshold events are counted. 
 * More than one event in an event set can be used to trigger overflows. 
 * In such cases, the user must call this function once for each overflowing event. 
 * To turn off overflow on a specified event, call this function with a 
 * threshold value of 0.
 *
 * Overflows can be implemented in either software or hardware, but the scope 
 * is the entire event set. 
 * PAPI defaults to hardware overflow if it is available. 
 * In the case of software overflow, a periodic timer interrupt causes PAPI 
 * to compare the event counts against the threshold values and call the overflow 
 * handler if one or more events have exceeded their threshold. 
 * In the case of hardware overflow, the counters are typically set to the 
 * negative of the threshold value and count up to 0. 
 * This zero-crossing triggers a hardware interrupt that calls the overflow handler. 
 * Because of this counter interrupt, the counter values for overflowing counters 
 * may be very small or even negative numbers, and cannot be relied upon as accurate. 
 * In such cases the overflow handler can approximate the counts by supplying 
 * the threshold value whenever an overflow occurs. 
 */
int
PAPI_overflow( int EventSet, int EventCode, int threshold, int flags,
			   PAPI_overflow_handler_t handler )
{
	int retval, cidx, index, i;
	EventSetInfo_t *ESI;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ( ESI->state & PAPI_STOPPED ) != PAPI_STOPPED )
		papi_return( PAPI_EISRUN );

	if ( ESI->state & PAPI_ATTACHED )
		papi_return( PAPI_EINVAL );
	
	if ( ESI->state & PAPI_CPU_ATTACHED )
		papi_return( PAPI_EINVAL );
	
	if ( ( index =
		   _papi_hwi_lookup_EventCodeIndex( ESI,
											( unsigned int ) EventCode ) ) < 0 )
		papi_return( PAPI_ENOEVNT );

	if ( threshold < 0 )
		papi_return( PAPI_EINVAL );

	/* We do not support derived events in overflow */
	/* Unless it's DERIVED_CMPD in which no calculations are done */

	if ( !( flags & PAPI_OVERFLOW_FORCE_SW ) && threshold != 0 &&
		 ( ESI->EventInfoArray[index].derived ) &&
		 ( ESI->EventInfoArray[index].derived != DERIVED_CMPD ) )
		papi_return( PAPI_EINVAL );

	/* the first time to call PAPI_overflow function */

	if ( !( ESI->state & PAPI_OVERFLOWING ) ) {
		if ( handler == NULL )
			papi_return( PAPI_EINVAL );
		if ( threshold == 0 )
			papi_return( PAPI_EINVAL );
	}
	if ( threshold > 0 &&
		 ESI->overflow.event_counter >= _papi_hwd[cidx]->cmp_info.num_cntrs )
		papi_return( PAPI_ECNFLCT );

	if ( threshold == 0 ) {
		for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
			if ( ESI->overflow.EventCode[i] == EventCode )
				break;
		}
		/* EventCode not found */
		if ( i == ESI->overflow.event_counter )
			papi_return( PAPI_EINVAL );
		/* compact these arrays */
		while ( i < ESI->overflow.event_counter - 1 ) {
			ESI->overflow.deadline[i] = ESI->overflow.deadline[i + 1];
			ESI->overflow.threshold[i] = ESI->overflow.threshold[i + 1];
			ESI->overflow.EventIndex[i] = ESI->overflow.EventIndex[i + 1];
			ESI->overflow.EventCode[i] = ESI->overflow.EventCode[i + 1];
			i++;
		}
		ESI->overflow.deadline[i] = 0;
		ESI->overflow.threshold[i] = 0;
		ESI->overflow.EventIndex[i] = 0;
		ESI->overflow.EventCode[i] = 0;
		ESI->overflow.event_counter--;
	} else {
		if ( ESI->overflow.event_counter > 0 ) {
			if ( ( flags & PAPI_OVERFLOW_FORCE_SW ) &&
				 ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) )
				papi_return( PAPI_ECNFLCT );
			if ( !( flags & PAPI_OVERFLOW_FORCE_SW ) &&
				 ( ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) )
				papi_return( PAPI_ECNFLCT );
		}
		for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
			if ( ESI->overflow.EventCode[i] == EventCode )
				break;
		}
		/* A new entry */
		if ( i == ESI->overflow.event_counter ) {
			ESI->overflow.EventCode[i] = EventCode;
			ESI->overflow.event_counter++;
		}
		/* New or existing entry */
		ESI->overflow.deadline[i] = threshold;
		ESI->overflow.threshold[i] = threshold;
		ESI->overflow.EventIndex[i] = index;
		ESI->overflow.flags = flags;
	}

	/* If overflowing is already active, we should check to
	   make sure that we don't specify a different handler
	   or different flags here. You can't mix them. */

	ESI->overflow.handler = handler;

	/* Set up the option structure for the low level.
	   If we have hardware interrupts and we are not using
	   forced software emulated interrupts */

	if ( _papi_hwd[cidx]->cmp_info.hardware_intr &&
		 !( ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) ) {
		retval = _papi_hwd[cidx]->set_overflow( ESI, index, threshold );
		if ( retval == PAPI_OK )
			ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
		else {
			papi_return( retval );	/* We should undo stuff here */
		}
	} else {
		/* Make sure hardware overflow is not set */
		ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );
	}

	APIDBG( "Overflow using: %s\n",
			( ESI->overflow.
			  flags & PAPI_OVERFLOW_HARDWARE ? "[Hardware]" : ESI->overflow.
			  flags & PAPI_OVERFLOW_FORCE_SW ? "[Forced Software]" :
			  "[Software]" ) );

	/* Toggle the overflow flags and ESI state */

	if ( ESI->overflow.event_counter >= 1 )
		ESI->state |= PAPI_OVERFLOWING;
	else {
		ESI->state ^= PAPI_OVERFLOWING;
		ESI->overflow.flags = 0;
		ESI->overflow.handler = NULL;
	}

	return ( PAPI_OK );
}

/** @class PAPI_sprofil
 *	generate PC histogram data from multiple code regions where hardware counter overflow occurs 
 *
 *	@param *prof 
 *		pointer to an array of PAPI_sprofil_t structures.
 *	@param profcnt 
 *		number of structures in the prof array for hardware profiling.
 *	@param EventSet 
 *		The PAPI EventSet to profile. This EventSet is marked as profiling-ready, 
 *		but profiling doesn't actually start until a PAPI_start() call is issued.
 *	@param EventCode
 *		Code of the Event in the EventSet to profile. 
 *		This event must already be a member of the EventSet.
 *	@param threshold 
 *		minimum number of events that must occur before the PC is sampled. 
 *		If hardware overflow is supported for your substrate, this threshold will 
 *		trigger an interrupt when reached. 
 *		Otherwise, the counters will be sampled periodically and the PC will be 
 *		recorded for the first sample that exceeds the threshold. 
 *		If the value of threshold is 0, profiling will be disabled for this event.
 *	@param flags 
 *		bit pattern to control profiling behavior. 
 *		Defined values are given in a table in the documentation for PAPI_pofil 
 *
 *	PAPI_sprofil() is a structure driven profiler that profiles one or more 
 *	disjoint regions of code in a single call. 
 *	It accepts a pointer to a preinitialized array of sprofil structures, and 
 *	initiates profiling based on the values contained in the array. 
 *	Each structure in the array defines the profiling parameters that are 
 *	normally passed to PAPI_profil(). 
 *	For more information on profiling, @ref PAPI_profil
 *
 *	@see PAPI_overflow PAPI_get_executable_info PAPI_profil
 */
int
PAPI_sprofil( PAPI_sprofil_t * prof, int profcnt, int EventSet,
			  int EventCode, int threshold, int flags )
{
	EventSetInfo_t *ESI;
	int retval, index, i, buckets;
	int forceSW = 0;
	int cidx;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	if ( ( ESI->state & PAPI_STOPPED ) != PAPI_STOPPED )
		papi_return( PAPI_EISRUN );

	if ( ESI->state & PAPI_ATTACHED )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_CPU_ATTACHED )
		papi_return( PAPI_EINVAL );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ( index =
		   _papi_hwi_lookup_EventCodeIndex( ESI,
											( unsigned int ) EventCode ) ) < 0 )
		papi_return( PAPI_ENOEVNT );

	/* We do not support derived events in overflow */
	/* Unless it's DERIVED_CMPD in which no calculations are done */
	if ( ( ESI->EventInfoArray[index].derived ) &&
		 ( ESI->EventInfoArray[index].derived != DERIVED_CMPD ) &&
		 !( flags & PAPI_PROFIL_FORCE_SW ) )
		papi_return( PAPI_EINVAL );

	if ( prof == NULL )
		profcnt = 0;

	/* check all profile regions for valid scale factors of:
	   2 (131072/65536),
	   1 (65536/65536),
	   or < 1 (65535 -> 2) as defined in unix profil()
	   2/65536 is reserved for single bucket profiling
	   {0,1}/65536 are traditionally used to terminate profiling
	   but are unused here since PAPI uses threshold instead
	 */
	for ( i = 0; i < profcnt; i++ ) {
		if ( !( ( prof[i].pr_scale == 131072 ) ||
				( ( prof[i].pr_scale <= 65536 && prof[i].pr_scale > 1 ) ) ) ) {
			APIDBG( "Improper scale factor: %d\n", prof[i].pr_scale );
			papi_return( PAPI_EINVAL );
		}
	}

	if ( threshold < 0 )
		papi_return( PAPI_EINVAL );

	/* the first time to call PAPI_sprofil */
	if ( !( ESI->state & PAPI_PROFILING ) ) {
		if ( threshold == 0 )
			papi_return( PAPI_EINVAL );
	}
	if ( threshold > 0 &&
		 ESI->profile.event_counter >= _papi_hwd[cidx]->cmp_info.num_cntrs )
		papi_return( PAPI_ECNFLCT );

	if ( threshold == 0 ) {
		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode )
				break;
		}
		/* EventCode not found */
		if ( i == ESI->profile.event_counter )
			papi_return( PAPI_EINVAL );
		/* compact these arrays */
		while ( i < ESI->profile.event_counter - 1 ) {
			ESI->profile.prof[i] = ESI->profile.prof[i + 1];
			ESI->profile.count[i] = ESI->profile.count[i + 1];
			ESI->profile.threshold[i] = ESI->profile.threshold[i + 1];
			ESI->profile.EventIndex[i] = ESI->profile.EventIndex[i + 1];
			ESI->profile.EventCode[i] = ESI->profile.EventCode[i + 1];
			i++;
		}
		ESI->profile.prof[i] = NULL;
		ESI->profile.count[i] = 0;
		ESI->profile.threshold[i] = 0;
		ESI->profile.EventIndex[i] = 0;
		ESI->profile.EventCode[i] = 0;
		ESI->profile.event_counter--;
	} else {
		if ( ESI->profile.event_counter > 0 ) {
			if ( ( flags & PAPI_PROFIL_FORCE_SW ) &&
				 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) )
				papi_return( PAPI_ECNFLCT );
			if ( !( flags & PAPI_PROFIL_FORCE_SW ) &&
				 ( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) )
				papi_return( PAPI_ECNFLCT );
		}

		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode )
				break;
		}

		if ( i == ESI->profile.event_counter ) {
			i = ESI->profile.event_counter;
			ESI->profile.event_counter++;
			ESI->profile.EventCode[i] = EventCode;
		}
		ESI->profile.prof[i] = prof;
		ESI->profile.count[i] = profcnt;
		ESI->profile.threshold[i] = threshold;
		ESI->profile.EventIndex[i] = index;
	}

	APIDBG( "Profile event counter is %d\n", ESI->profile.event_counter );

	/* Clear out old flags */
	if ( threshold == 0 )
		flags |= ESI->profile.flags;

	/* make sure no invalid flags are set */
	if ( flags &
		 ~( PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED |
			PAPI_PROFIL_COMPRESS | PAPI_PROFIL_BUCKETS | PAPI_PROFIL_FORCE_SW |
			PAPI_PROFIL_INST_EAR | PAPI_PROFIL_DATA_EAR ) )
		papi_return( PAPI_EINVAL );

	if ( ( flags & ( PAPI_PROFIL_INST_EAR | PAPI_PROFIL_DATA_EAR ) ) &&
		 ( _papi_hwd[cidx]->cmp_info.profile_ear == 0 ) )
		papi_return( PAPI_ESBSTR );

	/* if we have kernel-based profiling, then we're just asking for signals on interrupt. */
	/* if we don't have kernel-based profiling, then we're asking for emulated PMU interrupt */
	if ( ( flags & PAPI_PROFIL_FORCE_SW ) &&
		 ( _papi_hwd[cidx]->cmp_info.kernel_profile == 0 ) )
		forceSW = PAPI_OVERFLOW_FORCE_SW;

	/* make sure one and only one bucket size is set */
	buckets = flags & PAPI_PROFIL_BUCKETS;
	if ( !buckets )
		flags |= PAPI_PROFIL_BUCKET_16;	/* default to 16 bit if nothing set */
	else {					 /* return error if more than one set */
		if ( !( ( buckets == PAPI_PROFIL_BUCKET_16 ) ||
				( buckets == PAPI_PROFIL_BUCKET_32 ) ||
				( buckets == PAPI_PROFIL_BUCKET_64 ) ) )
			papi_return( PAPI_EINVAL );
	}

	/* Set up the option structure for the low level */
	ESI->profile.flags = flags;

	if ( _papi_hwd[cidx]->cmp_info.kernel_profile &&
		 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
		retval = _papi_hwd[cidx]->set_profile( ESI, index, threshold );
		if ( ( retval == PAPI_OK ) && ( threshold > 0 ) ) {
			/* We need overflowing because we use the overflow dispatch handler */
			ESI->state |= PAPI_OVERFLOWING;
			ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
		}
	} else {
		retval =
			PAPI_overflow( EventSet, EventCode, threshold, forceSW,
						   _papi_hwi_dummy_handler );
	}
	if ( retval < PAPI_OK )
		papi_return( retval );	/* We should undo stuff here */

	/* Toggle the profiling flags and ESI state */

	if ( ESI->profile.event_counter >= 1 )
		ESI->state |= PAPI_PROFILING;
	else {
		ESI->state ^= PAPI_PROFILING;
		ESI->profile.flags = 0;
	}

	return ( PAPI_OK );
}

/** @class PAPI_profil
 *	generate a histogram of hardware counter overflows vs. PC addresses 
 *
 *	@param *buf
 *		pointer to a buffer of bufsiz bytes in which the histogram counts are 
 *		stored in an array of unsigned short, unsigned int, or 
 *		unsigned long long values, or 'buckets'. 
 *		The size of the buckets is determined by values in the flags argument.
 *	@param bufsiz
 *		the size of the histogram buffer in bytes. 
 *		It is computed from the length of the code region to be profiled, 
 *		the size of the buckets, and the scale factor as discussed below.
 *	@param offset
 *		the start address of the region to be profiled.
 *	@param scale
 *		broadly and historically speaking, a contraction factor that indicates 
 *		how much smaller the histogram buffer is than the region to be profiled. 
 *		More precisely, scale is interpreted as an unsigned 16-bit fixed-point 
 *		fraction with the decimal point implied on the left. 
 *		Its value is the reciprocal of the number of addresses in a subdivision, 
 *		per counter of histogram buffer. 
 *		Below is a table of representative values for scale: 
 *	@param EventSet
 *		The PAPI EventSet to profile. This EventSet is marked as profiling-ready, 
 *		but profiling doesn't actually start until a PAPI_start() call is issued.
 *	@param EventCode
 *		Code of the Event in the EventSet to profile. 
 *		This event must already be a member of the EventSet.
 *	@param threshold
 *		minimum number of events that must occur before the PC is sampled. 
 *		If hardware overflow is supported for your substrate, this threshold 
 *		will trigger an interrupt when reached. 
 *		Otherwise, the counters will be sampled periodically and the PC will be 
 *		recorded for the first sample that exceeds the threshold. 
 *		If the value of threshold is 0, profiling will be disabled for this event.
 *	@param flags
 *		bit pattern to control profiling behavior. 
 *		Defined values are shown in the table below
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval   PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other 
 *		events in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_profil() provides hardware event statistics by profiling the occurence
 *	of specified hardware counter events. 
 *	It is designed to mimic the UNIX SVR4 profil call. 
 *	
 *	The statistics are generated by creating a histogram of hardware counter 
 *	event overflows vs. program counter addresses for the current process. 
 *	The histogram is defined for a specific region of program code to be 
 *	profiled, and the identified region is logically broken up into a set of 
 *	equal size subdivisions, each of which corresponds to a count in the histogram. 
 *	
 *	With each hardware event overflow, the current subdivision is identified 
 *	and its corresponding histogram count is incremented. 
 *	These counts establish a relative measure of how many hardware counter 
 *	events are occuring in each code subdivision. 
 *	
 *	The resulting histogram counts for a profiled region can be used to 
 *	identify those program addresses that generate a disproportionately 
 *	high percentage of the event of interest.
 *
 *	Events to be profiled are specified with the EventSet and EventCode parameters. 
 *	More than one event can be simultaneously profiled by calling PAPI_profil 
 *	several times with different EventCode values. 
 *	Profiling can be turned off for a given event by calling PAPI_profil 
 *	with a threshold value of 0. 
 *
 *	 Representative values for the scale variable
 *	HEX	DECIMAL		DEFININTION
 *	0x20000	131072	Maps precisely one instruction address to a unique bucket in buf.
 *	0x10000	65536	Maps precisely two instruction addresses to a unique bucket in buf.
 *	0xFFFF	65535	Maps approximately two instruction addresses to a unique bucket in buf.
 *	0x8000	32768	Maps every four instruction addresses to a bucket in buf.
 *	0x4000	16384	Maps every eight instruction addresses to a bucket in buf.
 *	0x0002	2	Maps all instruction addresses to the same bucket in buf.
 *	0x0001	1	Undefined.
 *	0x0000	0	Undefined. 
 *
 *
 *	Historically, the scale factor was introduced to allow the allocation of 
 *	buffers smaller than the code size to be profiled. 
 *	Data and instruction sizes were assumed to be multiples of 16-bits. 
 *	These assumptions are no longer necessarily true. 
 *	PAPI_profil has preserved the traditional definition of scale where appropriate, 
 *	but deprecated the definitions for 0 and 1 (disable scaling) and extended 
 *	the range of scale to include 65536 and 131072 to allow for exactly two 
 *	addresses and exactly one address per profiling bucket.
 *
 *	The value of bufsiz is computed as follows:
 *	
 *	bufsiz = (end - start)*(bucket_size/2)*(scale/65536) where
 *
 *	Defined bits for the flags variable
 *	PAPI_PROFIL_POSIX	Default type of profiling, similar to profil (3).
 *	PAPI_PROFIL_RANDOM		Drop a random 25% of the samples.
 *	PAPI_PROFIL_WEIGHTED	Weight the samples by their value.
 *	PAPI_PROFIL_COMPRESS	Ignore samples as values in the hash buckets get big.
 *	PAPI_PROFIL_BUCKET_16	Use unsigned short (16 bit) buckets, This is the default bucket.
 *	PAPI_PROFIL_BUCKET_32	Use unsigned int (32 bit) buckets.
 *	PAPI_PROFIL_BUCKET_64	Use unsigned long long (64 bit) buckets.
 *	PAPI_PROFIL_FORCE_SW	Force software overflow in profiling. 
 *
 *	@see PAPI_get_executable_info PAPI_overflow PAPI_sprofil
 */
int
PAPI_profil( void *buf, unsigned bufsiz, caddr_t offset,
			 unsigned scale, int EventSet, int EventCode, int threshold,
			 int flags )
{
	EventSetInfo_t *ESI;
	int i;
	int retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* scale factors are checked for validity in PAPI_sprofil */

	if ( threshold > 0 ) {
		PAPI_sprofil_t *prof;

		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode )
				break;
		}

		if ( i == ESI->profile.event_counter ) {
			prof =
				( PAPI_sprofil_t * ) papi_malloc( sizeof ( PAPI_sprofil_t ) );
			memset( prof, 0x0, sizeof ( PAPI_sprofil_t ) );
			prof->pr_base = buf;
			prof->pr_size = bufsiz;
			prof->pr_off = offset;
			prof->pr_scale = scale;

			retval =
				PAPI_sprofil( prof, 1, EventSet, EventCode, threshold, flags );

			if ( retval != PAPI_OK )
				papi_free( prof );
		} else {
			prof = ESI->profile.prof[i];
			prof->pr_base = buf;
			prof->pr_size = bufsiz;
			prof->pr_off = offset;
			prof->pr_scale = scale;
			retval =
				PAPI_sprofil( prof, 1, EventSet, EventCode, threshold, flags );
		}
		papi_return( retval );
	}

	for ( i = 0; i < ESI->profile.event_counter; i++ ) {
		if ( ESI->profile.EventCode[i] == EventCode )
			break;
	}
	/* EventCode not found */
	if ( i == ESI->profile.event_counter )
		papi_return( PAPI_EINVAL );

	papi_free( ESI->profile.prof[i] );
	ESI->profile.prof[i] = NULL;

	papi_return( PAPI_sprofil( NULL, 0, EventSet, EventCode, 0, flags ) );
}

/* This function sets the low level default granularity
   for all newly manufactured eventsets. The first function
   preserves API compatibility and assumes component 0;
   The second function takes a component argument. */

/** @class PAPI_set_granularity
 *	set the default counting granularity for eventsets bound to the cpu component 
 *
 *	@param granularity
 *		one of the following constants as defined in the papi.h header file
 *		<ul>
 *			<li> PAPI_GRN_THR	Count each individual thread
 *			<li> PAPI_GRN_PROC	Count each individual process
 *			<li> PAPI_GRN_PROCG	Count each individual process group
 *			<li> PAPI_GRN_SYS	Count the current CPU
 *			<li> PAPI_GRN_SYS_CPU	Count all CPU's individually
 *			<li> PAPI_GRN_MIN	The finest available granularity
 *			<li>PAPI_GRN_MAX	The coarsest available granularity 
 *		</ul>
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	PAPI_set_granularity sets the default counting granularity for all new 
 *	event sets created by PAPI_create_eventset . 
 *	This call implicitly sets the granularity for the cpu component 
 *	(component 0) and is included to preserve backward compatibility. 
 *
 *	@see  PAPI_set_domain PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_granularity( int granularity )
{
	return ( PAPI_set_cmp_granularity( granularity, 0 ) );
}

/** @class PAPI_set_cmp_granularity
 *	set the default counting granularity for eventsets bound to the specified component 
 *
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *	@param granularity
 *		one of the following constants as defined in the papi.h header file
 *		<ul>
 *			<li> PAPI_GRN_THR	Count each individual thread
 *			<li> PAPI_GRN_PROC	Count each individual process
 *			<li> PAPI_GRN_PROCG	Count each individual process group
 *			<li> PAPI_GRN_SYS	Count the current CPU
 *			<li> PAPI_GRN_SYS_CPU	Count all CPU's individually
 *			<li> PAPI_GRN_MIN	The finest available granularity
 *			<li>PAPI_GRN_MAX	The coarsest available granularity 
 *		</ul>
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	PAPI_set_cmp_granularity sets the default counting granularity for all new 
 *	event sets, and requires an explicit component argument. 
 *	Event sets that are already in existence are not affected. 
 *
 *	To change the granularity of an existing event set, please see PAPI_set_opt. 
 *	The reader should note that the granularity of an event set affects only 
 *	the mode in which the counter continues to run. 
 *
 *	@see  PAPI_set_domain PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_cmp_granularity( int granularity, int cidx )
{
	PAPI_option_t ptr;

	memset( &ptr, 0, sizeof ( ptr ) );
	ptr.defgranularity.def_cidx = cidx;
	ptr.defgranularity.granularity = granularity;
	papi_return( PAPI_set_opt( PAPI_DEFGRN, &ptr ) );
}

/* This function sets the low level default counting domain
   for all newly manufactured eventsets. The first function
   preserves API compatibility and assumes component 0;
   The second function takes a component argument. */

/** @class PAPI_set_domain
 *	set the default counting domain for new event sets bound to the cpu component 
 *
 *	@param domain
 *		one of the following constants as defined in the papi.h header file
 *		<ul>
 *			<li> PAPI_DOM_USER	User context counted
 *			<li> PAPI_DOM_KERNEL	Kernel/OS context counted
 *			<li> PAPI_DOM_OTHER	Exception/transient mode counted
 *			<li> PAPI_DOM_SUPERVISOR	Supervisor/hypervisor context counted
 *			<li> PAPI_DOM_ALL	All above contexts counted
 *			<li> PAPI_DOM_MIN	The smallest available context
 *			<li> PAPI_DOM_MAX	The largest available context 
 *		</ul>
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	PAPI_set_domain sets the default counting domain for all new event sets 
 *	created by PAPI_create_eventset in all threads. 
 *	This call implicitly sets the domain for the cpu component (component 0) 
 *	and is included to preserve backward compatibility. 
 *
 *	@see PAPI_set_granularity PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_domain( int domain )
{
	return ( PAPI_set_cmp_domain( domain, 0 ) );
}

/** @class PAPI_set_cmp_domain
 *	set the default counting domain for new event sets bound to the specified component
 *
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *	@param domain
 *		one of the following constants as defined in the papi.h header file
 *		<ul>
 *			<li> PAPI_DOM_USER	User context counted
 *			<li> PAPI_DOM_KERNEL	Kernel/OS context counted
 *			<li> PAPI_DOM_OTHER	Exception/transient mode counted
 *			<li> PAPI_DOM_SUPERVISOR	Supervisor/hypervisor context counted
 *			<li> PAPI_DOM_ALL	All above contexts counted
 *			<li> PAPI_DOM_MIN	The smallest available context
 *			<li> PAPI_DOM_MAX	The largest available context 
 *		</ul>
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	PAPI_set_cmp_domain sets the default counting domain for all new event sets 
 *	in all threads, and requires an explicit component argument. 
 *	Event sets that are already in existence are not affected. 
 *	To change the domain of an existing event set, please see PAPI_set_opt.
 *	The reader should note that the domain of an event set affects only the 
 *	mode in which the counter continues to run. 
 *	Counts are still aggregated for the current process, and not for any other 
 *	processes in the system. 
 *
 *	Thus when requesting PAPI_DOM_KERNEL , the user is asking for events that 
 *	occur on behalf of the process, inside the kernel. 
 *
 *	@see PAPI_set_granularity PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_cmp_domain( int domain, int cidx )
{
	PAPI_option_t ptr;

	memset( &ptr, 0, sizeof ( ptr ) );
	ptr.defdomain.def_cidx = cidx;
	ptr.defdomain.domain = domain;
	papi_return( PAPI_set_opt( PAPI_DEFDOM, &ptr ) );
}

/** @class PAPI_add_events
 *	add PAPI presets or native hardware events to an event set 
 *  
 *	@param EventSet
 *		an integer handle for a PAPI Event Set as created by PAPI_create_eventset ()
 *	@param *Events 
 *		an array of defined events
 *	@param number 
 *		an integer indicating the number of events in the array *EventCode 
 *		It should be noted that PAPI_add_events can partially succeed, 
 *		exactly like PAPI_remove_events. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the event set simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware.
 *	@retval PAPI_EBUG 
 *		Internal error, please send mail to the developers. 
 *
 *	PAPI_add_event() adds one event to a PAPI Event Set.
 *	A hardware event can be either a PAPI preset or a native hardware event code. 
 *	For a list of PAPI preset events, see PAPI_presets() or run the avail test 
 *	case in the PAPI distribution. 
 *	PAPI presets can be passed to PAPI_query_event() to see if they exist on 
 *	the underlying architecture. 
 *	For a list of native events available on current platform, run native_avail 
 *	test case in the PAPI distribution. 
 *	For the encoding of native events, see PAPI_event_name_to_code() to learn 
 *	how to generate native code for the supported native event on the underlying architecture. 
 *
 * @see PAPI_cleanup_eventset() PAPI_destroy_eventset() PAPI_event_code_to_name() PAPI_remove_events() PAPI_query_event() PAPI_presets() PAPI_native() PAPI_remove_event()
 */
int
PAPI_add_events( int EventSet, int *Events, int number )
{
	int i, retval;

	if ( ( Events == NULL ) || ( number <= 0 ) )
		papi_return( PAPI_EINVAL );

	for ( i = 0; i < number; i++ ) {
		retval = PAPI_add_event( EventSet, Events[i] );
		if ( retval != PAPI_OK ) {
			if ( i == 0 )
				papi_return( retval );
			else
				return ( i );
		}
	}
	return ( PAPI_OK );
}

/** @class PAPI_remove_events
 * @brief removes an array of hardware event codes from a PAPI event set.
 *
 * A hardware event can be either a PAPI Preset or a native hardware event code. 
 * For a list of PAPI preset events, see PAPI_presets or run the papi_avail utility in the PAPI distribution. 
 * PAPI Presets can be passed to PAPI_query_event to see if they exist on the underlying architecture. 
 * For a list of native events available on current platform, run papi_native_avail in the PAPI distribution. 
 * It should be noted that PAPI_remove_events can partially succeed, exactly like PAPI_add_events. 
 *
 *	@par C Prototype:
 *		#include <papi.h> @n
 *		int PAPI_remove_events( int  EventSet, int * EventCode, int  number );
 *
 *	@par Fortran Prototype:
 *		#include fpapi.h @n
 *		PAPIF_remove_events( C_INT  EventSet,  C_INT(*)  EventCode,  C_INT  number,  C_INT  check )
 * 
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param *Events
 *		an array of defined events
 *	@param number
 *		an integer indicating the number of events in the array *EventCode 
 *
 *	@retval Positive integer 
 *		The number of consecutive elements that succeeded before the error.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other 
 *		events in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	@par Example:
 *	@code
int EventSet = PAPI_NULL;
int Events[] = {PAPI_TOT_INS, PAPI_FP_OPS};
int ret;
 
 // Create an empty EventSet
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Add two events to our EventSet
ret = PAPI_add_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);

// Start counting
ret = PAPI_start(EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Stop counting, ignore values
ret = PAPI_stop(EventSet, NULL);
if (ret != PAPI_OK) handle_error(ret);

// Remove event
ret = PAPI_remove_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *  @bug The last argument should be a pointer so the count can be returned on partial success in addition
 *  to a real error code.
 *
 *	@see PAPI_cleanup_eventset PAPI_destroy_eventset PAPI_event_name_to_code 
 *		PAPI_presets PAPI_add_event PAPI_add_events
 */
int
PAPI_remove_events( int EventSet, int *Events, int number )
{
	int i, retval;

	if ( ( Events == NULL ) || ( number <= 0 ) )
		papi_return( PAPI_EINVAL );

	for ( i = 0; i < number; i++ ) {
		retval = PAPI_remove_event( EventSet, Events[i] );
		if ( retval != PAPI_OK ) {
			if ( i == 0 )
				papi_return( retval );
			else
				return ( i );
		}
	}
	return ( PAPI_OK );
}

/** @class PAPI_list_events
 *	list the events in an event set 
 *
 *	@param EventSet
 *		An integer handle for a PAPI event set as created by PAPI_create_eventset 
 *	@param *Events 
 *		An array of codes for events, such as PAPI_INT_INS. 
 *		No more than *number codes will be stored into the array.
 *	@param *number 
 *		On input the variable determines the size of the Events array. 
 *		On output the variable contains the number of counters in the event set.
 *		Note that if the given array Events is too short to hold all the counters 
 *		in the event set the *number variable will be greater than the actually 
 *		stored number of counter codes. 
 *
 *	@retval PAPI_EINVAL
 *	@retval PAPI_ENOEVST
 *
 *	PAPI_list_events() decomposes an event set into the hardware events it contains.
 *	This call assumes an initialized PAPI library and a successfully added event set. 
 *
 *	@see PAPI_event_code_to_name PAPI_event_name_to_code PAPI_add_event PAPI_create_eventset
 */
int
PAPI_list_events( int EventSet, int *Events, int *number )
{
	EventSetInfo_t *ESI;
	int i, j;

	if ( ( Events == NULL ) || ( *number <= 0 ) )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( !ESI )
		papi_return( PAPI_ENOEVST );

	for ( i = 0, j = 0; j < ESI->NumberOfEvents; i++ ) {
		if ( ( int ) ESI->EventInfoArray[i].event_code != PAPI_NULL ) {
			Events[j] = ( int ) ESI->EventInfoArray[i].event_code;
			j++;
			if ( j == *number )
				break;
		}
	}

	*number = j;

	return ( PAPI_OK );
}

/* xxx This is OS dependent, not component dependent, right? */
/** @class PAPI_get_dmem_info
 *	get information about the dynamic memory usage of the current program 
 *
 *	@param dest
 *		structure to be filled in @ref PAPI_dmem_info_t
 *	
 *	@retval PAPI_ESBSTR 
 *		The funtion is not implemented for the current substrate.
 *	@retval PAPI_EINVAL 
 *		Any value in the structure or array may be undefined as indicated by 
 *		this error value.
 *	@retval PAPI_SYS 
 *		A system error occured. 
 *
 *	NOTE: This function is only implemented for the Linux operating system.
 *	This function takes a pointer to a PAPI_dmem_info_t structure 
 *	and returns with the structure fields filled in. 
 *	A value of PAPI_EINVAL in any field indicates an undefined parameter. 
 *
 *	@see PAPI_get_executable_info PAPI_get_hardware_info PAPI_get_opt PAPI_library_init
 */
int
PAPI_get_dmem_info( PAPI_dmem_info_t * dest )
{
	if ( dest == NULL )
		return PAPI_EINVAL;

	memset( ( void * ) dest, 0x0, sizeof ( PAPI_dmem_info_t ) );
	return ( _papi_hwd[0]->get_dmem_info( dest ) );
}


/** @class PAPI_get_executable_info
 *	get the executable's address space info 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *
 *	This function returns a pointer to a structure containing information 
 *	about the current program.
 *
 *	@see PAPI_get_opt PAPI_get_hardware_info PAPI_exe_info_t
 */
const PAPI_exe_info_t *
PAPI_get_executable_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_EXEINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.exe_info );
	else
		return ( NULL );
}

/** @class PAPI_get_shared_lib_info
 *	get address info about the shared libraries used by the process 
 *
 *	In C, this function returns a pointer to a structure containing information 
 *	about the shared library used by the program. 
 *	There is no Fortran equivalent call. 
 *	
 *	@see PAPI_get_hardware_info PAPI_get_executable_info PAPI_get_dmem_info PAPI_get_opt PAPI_library_init
 */
const PAPI_shlib_info_t *
PAPI_get_shared_lib_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_SHLIBINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.shlib_info );
	else
		return ( NULL );
}

const PAPI_hw_info_t *
PAPI_get_hardware_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_HWINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.hw_info );
	else
		return ( NULL );
}


/* The next 4 timing functions always use component 0 */

/** @class PAPI_get_real_cyc
 *	get real time counter value in clock cycles 
 *
 *	Returns the total real time passed since some arbitrary starting point. 
 *	The time is returned in clock cycles. 
 *	This call is equivalent to wall clock time.
 *
 *	@see PAPI_get_virt_usec PAPI_get_virt_cyc PAPI_library_init
 */
long long
PAPI_get_real_cyc( void )
{
	return ( _papi_hwd[0]->get_real_cycles(  ) );
}

/** @class PAPI_get_real_nsec
 *	get real time counter value in nanoseconds 
 *
 *	This function returns the total real time passed since some arbitrary 
 *	starting point. 
 *	The time is returned in nanoseconds. 
 *	This call is equivalent to wall clock time.
 *
 *	@see PAPI_get_virt_usec PAPI_get_virt_cyc PAPI_library_init
 */

long long
PAPI_get_real_nsec( void )
{
	return ( ( _papi_hwd[0]->get_real_cycles(  ) * 1000LL ) /
			 ( long long ) _papi_hwi_system_info.hw_info.mhz );
}

/** @class PAPI_get_real_usec
 *	get real time counter value in microseconds 
 *
 *	This function returns the total real time passed since some arbitrary 
 *	starting point. 
 *	The time is returned in microseconds. 
 *	This call is equivalent to wall clock time.
 *
 *	@see PAPI_get_virt_usec PAPI_get_virt_cyc PAPI_library_init
 */
long long
PAPI_get_real_usec( void )
{
	return ( _papi_hwd[0]->get_real_usec(  ) );
}

/** @class PAPI_get_virt_cyc
 *	get virtual time counter value in clock cycles 
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or for threaded 
 *		applications, if there has been no thread id function defined by the 
 *		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread specific
 *		master event created for the current thread, and if the allocation of 
 *		such an event set fails, the call will return PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every platform 
 *	PAPI supports. 
 *	However on some platforms, the resolution can be as bad as 1/Hz as defined 
 *	by the operating system. 
 *
 */
long long
PAPI_get_virt_cyc( void )
{
	ThreadInfo_t *master;
	int retval;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( ( retval = _papi_hwi_lookup_or_create_thread( &master, 0 ) ) != PAPI_OK )
		papi_return( retval );

	return ( ( long long ) _papi_hwd[0]->
			 get_virt_cycles( master->context[0] ) );
}

/** @class PAPI_get_virt_nsec
 *	get virtual time counter values in microseconds 
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or for threaded 
 *		applications, if there has been no thread id function defined by the 
 *		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread specific
 *		master event created for the current thread, and if the allocation of 
 *		such an event set fails, the call will return PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every platform 
 *	PAPI supports. 
 *	However on some platforms, the resolution can be as bad as 1/Hz as defined 
 *	by the operating system. 
 *
 */
long long
PAPI_get_virt_nsec( void )
{
	ThreadInfo_t *master;
	int retval;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( ( retval = _papi_hwi_lookup_or_create_thread( &master, 0 ) ) != PAPI_OK )
		papi_return( retval );

	return ( ( _papi_hwd[0]->get_virt_cycles( master->context[0] ) * 1000LL ) /
			 ( long long ) _papi_hwi_system_info.hw_info.mhz );
}

/** @class PAPI_get_virt_usec
 *	get virtual time counter values in microseconds 
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or for threaded 
 *		applications, if there has been no thread id function defined by the 
 *		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread specific
 *		master event created for the current thread, and if the allocation of 
 *		such an event set fails, the call will return PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every platform 
 *	PAPI supports. 
 *	However on some platforms, the resolution can be as bad as 1/Hz as defined 
 *	by the operating system. 
 *
 */
long long
PAPI_get_virt_usec( void )
{
	ThreadInfo_t *master;
	int retval;

	if ( ( retval = _papi_hwi_lookup_or_create_thread( &master, 0 ) ) != PAPI_OK )
		papi_return( retval );

	return ( ( long long ) _papi_hwd[0]->get_virt_usec( master->context[0] ) );
}

int
PAPI_restore( void )
{
	PAPIERROR( "PAPI_restore is currently not implemented" );
	return ( PAPI_ESBSTR );
}

int
PAPI_save( void )
{
	PAPIERROR( "PAPI_save is currently not implemented" );
	return ( PAPI_ESBSTR );
}

/** @class PAPI_lock
 *	Lock one of two mutex variables defined in papi.h 
 *
 *	@param lck
 *		an integer value specifying one of the two user locks: PAPI_USR1_LOCK or PAPI_USR2_LOCK 
 *
 *	PAPI_lock() Grabs access to one of the two PAPI mutex variables. 
 *	This function is provided to the user to have a platform independent call 
 *	to (hopefully) efficiently implemented mutex.
 *
 *	@see PAPI_thread_init
 */
int
PAPI_lock( int lck )
{
	if ( ( lck < 0 ) || ( lck >= PAPI_NUM_LOCK ) )
		papi_return( PAPI_EINVAL );

	papi_return( _papi_hwi_lock( lck ) );
}

/** @class PAPI_unlock
 *	Unlock one of the mutex variables defined in papi.h 
 *
 *	@param lck
 *		an integer value specifying one of the two user locks: PAPI_USR1_LOCK 
 *		or PAPI_USR2_LOCK 
 *
 *	PAPI_unlock() unlocks the mutex acquired by a call to PAPI_lock .
 *
 *	@see PAPI_thread_init
 */
int
PAPI_unlock( int lck )
{
	if ( ( lck < 0 ) || ( lck >= PAPI_NUM_LOCK ) )
		papi_return( PAPI_EINVAL );

	papi_return( _papi_hwi_unlock( lck ) );
}

/** @class PAPI_is_initialized
 *	check for initialization
 *
 *	@retval PAPI_NOT_INITED
 *		Library has not been initialized
 *	@retval PAPI_LOW_LEVEL_INITED
 *		Low level has called library init
 *	@retval PAPI_HIGH_LEVEL_INITED
 *		High level has called library init 
 *	@retval PAPI_THREAD_LEVEL_INITED	
 *		Threads have been inited 
 *
 *	PAPI_is_initialized() returns the status of the PAPI library. 
 *	The PAPI library can be in one of four states, as described under RETURN VALUES. 
 */
int
PAPI_is_initialized( void )
{
	return ( init_level );
}

/* This function maps the overflow_vector to event indexes in the event
   set, so that user can know which PAPI event overflowed.
   int *array---- an array of event indexes in eventset; the first index
                  maps to the highest set bit in overflow_vector
   int *number--- this is an input/output parameter, user should put the
                  size of the array into this parameter, after the function
                  is executed, the number of indexes in *array is written
                  to this parameter
*/

/** @class PAPI_get_overflow_event_index
 *	converts an overflow vector into an array of indexes to overflowing events 
 *
 *	@param EventSet
 *		an integer handle to a PAPI event set as created by PAPI_create_eventset
 *	@param overflow_vector
 *		a vector with bits set for each counter that overflowed. 
 *		This vector is passed by the system to the overflow handler routine.
 *	@param array
 *		an array of indexes for events in EventSet. 
 *		No more than *number indexes will be stored into the array.
 *	@param number 
 *		On input the variable determines the size of the array. 
 *		On output the variable contains the number of indexes in the array. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *		This could occur if the overflow_vector is empty (zero), if the array or 
 *		number pointers are NULL, if the value of number is less than one, 
 *		or if the EventSet is empty.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist. 
 *	
 *	PAPI_get_overflow_event_index decomposes an overflow_vector into an event 
 *	index array in which the first element corresponds to the least significant
 *	set bit in overflow_vector and so on. Based on overflow_vector, 
 *	the user can only tell which physical counters overflowed. 
 *	Using this function, the user can map overflowing counters to specific 
 *	events in the event set. 
 *
 *	An array is used in this function to support the possibility of 
 *	multiple simultaneous overflow events.
 *
 *	@see PAPI_overflow
 */
int
PAPI_get_overflow_event_index( int EventSet, long long overflow_vector,
							   int *array, int *number )
{
	EventSetInfo_t *ESI;
	int set_bit, j, pos;
	int count = 0, k;

	if ( overflow_vector == ( long long ) 0 )
		papi_return( PAPI_EINVAL );

	if ( ( array == NULL ) || ( number == NULL ) )
		papi_return( PAPI_EINVAL );

	if ( *number < 1 )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* in case the eventset is empty */
	if ( ESI->NumberOfEvents == 0 )
		papi_return( PAPI_EINVAL );

	while ( ( set_bit = ffsll( overflow_vector ) ) ) {
		set_bit -= 1;
		overflow_vector ^= ( long long ) 1 << set_bit;
		for ( j = 0; j < ESI->NumberOfEvents; j++ ) {
			for ( k = 0, pos = 0; k < MAX_COUNTER_TERMS && pos >= 0; k++ ) {
				pos = ESI->EventInfoArray[j].pos[k];
				if ( ( set_bit == pos ) &&
					 ( ( ESI->EventInfoArray[j].derived == NOT_DERIVED ) ||
					   ( ESI->EventInfoArray[j].derived == DERIVED_CMPD ) ) ) {
					array[count++] = j;
					if ( count == *number )
						return ( PAPI_OK );

					break;
				}
			}
		}
	}
	*number = count;
	return ( PAPI_OK );
}
