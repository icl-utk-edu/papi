/*
 * File:	windows.c
 * Author:      Dan Terpstra
 *		terpstra@cs.utk.edu
 * Mods:	Kevin London
 *		london@cs.utk.edu
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

/* This should be in a windows.h header file maybe. */
#define FOPEN_ERROR "fopen(%s) returned NULL"

CRITICAL_SECTION lock[PAPI_MAX_LOCK];


// split the filename from a full path
// roughly equivalent to unix basename()
static void
splitpath( const char *path, char *name )
{
	short i = 0, last = 0;

	while ( path[i] ) {
		if ( path[i] == '\\' )
			last = i;
		i++;
	}
	name[0] = 0;
	i = i - last;
	if ( last > 0 ) {
		last++;
		i--;
	}
	strncpy( name, &path[last], i );
	name[i] = 0;
}

int
_papi_hwd_get_system_info( void )
{
	struct wininfo win_hwinfo;
	HMODULE hModule;
	DWORD len;
	long i = 0;

	/* Path and args */
	_papi_hwi_system_info.pid = getpid(  );

	hModule = GetModuleHandle( NULL );	// current process
	len =
		GetModuleFileName( hModule, _papi_hwi_system_info.exe_info.fullname,
						   PAPI_MAX_STR_LEN );
	if ( len )
		splitpath( _papi_hwi_system_info.exe_info.fullname,
				   _papi_hwi_system_info.exe_info.address_info.name );
	else
		return ( PAPI_ESYS );

	SUBDBG( "Executable is %s\n",
			_papi_hwi_system_info.exe_info.address_info.name );
	SUBDBG( "Full Executable is %s\n",
			_papi_hwi_system_info.exe_info.fullname );
	/* Hardware info */
	if ( !init_hwinfo( &win_hwinfo ) )
		return ( PAPI_ESYS );

	_papi_hwi_system_info.hw_info.ncpu = win_hwinfo.ncpus;
	_papi_hwi_system_info.hw_info.nnodes = win_hwinfo.nnodes;
	_papi_hwi_system_info.hw_info.totalcpus = win_hwinfo.total_cpus;

	_papi_hwi_system_info.hw_info.vendor = win_hwinfo.vendor;
	_papi_hwi_system_info.hw_info.revision = ( float ) win_hwinfo.revision;
	strcpy( _papi_hwi_system_info.hw_info.vendor_string,
			win_hwinfo.vendor_string );

	/* initialize the model to something */
	_papi_hwi_system_info.hw_info.model = PERFCTR_X86_GENERIC;

	if ( IS_P3( &win_hwinfo ) || IS_P3_XEON( &win_hwinfo ) ||
		 IS_CELERON( &win_hwinfo ) )
		_papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_PIII;

	if ( IS_MOBILE( &win_hwinfo ) )
		_papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_PENTM;

	if ( IS_P4( &win_hwinfo ) ) {
		if ( win_hwinfo.model >= 2 )
			/* this is a guess for Pentium 4 Model 2 */
			_papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_P4M2;
		else
			_papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_P4;
	}

	if ( IS_AMDDURON( &win_hwinfo ) || IS_AMDATHLON( &win_hwinfo ) )
		_papi_hwi_system_info.hw_info.model = PERFCTR_X86_AMD_K7;

	strcpy( _papi_hwi_system_info.hw_info.model_string,
			win_hwinfo.model_string );

	_papi_hwi_system_info.num_cntrs = win_hwinfo.nrctr;
	_papi_hwi_system_info.num_gp_cntrs = _papi_hwi_system_info.num_cntrs;

	_papi_hwi_system_info.hw_info.mhz = ( float ) win_hwinfo.mhz;
	_papi_hwi_system_info.hw_info.clock_mhz = mhz;

	return ( PAPI_OK );
}

static void
lock_init( void )
{
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ ) {
		InitializeCriticalSection( &lock[i] );
	}
}

static void
lock_release( void )
{
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ ) {
		DeleteCriticalSection( &lock[i] );
	}
}

HANDLE pmc_dev;						   // device handle for kernel driver


/* At init time, the higher level library should always allocate and
   reserve EventSet zero. */

int
_papi_hwd_init( hwd_context_t * ctx )
{
	/* Initialize our thread/process pointer. */
	if ( ( ctx->self = pmc_dev = pmc_open(  ) ) == NULL ) {
		PAPIERROR( "pmc_open() returned NULL" );
		return ( PAPI_ESYS );
	}
	SUBDBG( "_papi_hwd_init pmc_open() = %p\n", ctx->self );

	/* Linux makes sure that each thread has a virtualized TSC here.
	   This makes no sense on Windows, since the counters aren't
	   saved at context switch. */
	return ( PAPI_OK );
}

/* Called once per process. */
int
_papi_hwd_shutdown_substrate( void )
{
	pmc_close( pmc_dev );
	lock_release(  );
	return ( PAPI_OK );
}

int
_papi_hwd_shutdown( hwd_context_t * ctx )
{
	int retval = 0;
//   retval = vperfctr_unlink(ctx->self);
	SUBDBG( "_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->self,
			retval );
	pmc_close( ctx->self );
	SUBDBG( "_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->self );
	memset( ctx, 0x0, sizeof ( hwd_context_t ) );

	if ( retval )
		return ( PAPI_ESYS );
	return ( PAPI_OK );
}

void CALLBACK
_papi_hwd_timer_callback( UINT wTimerID, UINT msg,
						  DWORD dwUser, DWORD dw1, DWORD dw2 )
{
	_papi_hwi_context_t ctx;
	CONTEXT context;				   // processor specific context structure
	HANDLE threadHandle;
	BOOL error;
	ThreadInfo_t *t = NULL;

	ctx.ucontext = &context;

	// dwUser is the threadID passed by timeSetEvent
	// NOTE: This call requires W2000 or later
	threadHandle = OpenThread( THREAD_GET_CONTEXT, FALSE, dwUser );

	// retrieve the contents of the control registers only
	context.ContextFlags = CONTEXT_CONTROL;
	error = GetThreadContext( threadHandle, &context );
	CloseHandle( threadHandle );

	// pass a void pointer to cpu register data here
	_papi_hwi_dispatch_overflow_signal( ( void * ) ( &ctx ), NULL, 0, 0, &t );
}

/* Collected wisdom indicates that each call to pmc_set_control will write 0's
    into the hardware counters, effecting a reset operation.
*/
int
_papi_hwd_start( hwd_context_t * ctx, hwd_control_state_t * spc )
{
	int error;
	struct pmc_control *ctl =
		( struct pmc_control * ) ( spc->control.cpu_control.evntsel );

	/* clear the accumulating counter values */
	memset( ( void * ) spc->state.sum.pmc, 0,
			_papi_hwi_system_info.num_cntrs * sizeof ( long long ) );
	if ( ( error = pmc_set_control( ctx->self, ctl ) ) < 0 ) {
		SUBDBG( "pmc_set_control returns: %d\n", error );
		{
			PAPIERROR( "pmc_set_control() returned < 0" );
			return ( PAPI_ESYS );
		}
	}
#ifdef DEBUG
	print_control( &spc->control.cpu_control );
#endif
	return ( PAPI_OK );
}

int
_papi_hwd_stop( hwd_context_t * ctx, hwd_control_state_t * state )
{
	/* Since Windows counts system-wide (no counter saves at context switch)
	   and since PAPI 3 no longer merges event sets, this function doesn't
	   need to do anything in the Windows version.
	 */
	return ( PAPI_OK );
}

int
_papi_hwd_read( hwd_context_t * ctx, hwd_control_state_t * spc, long long **dp,
				int flags )
{
	pmc_read_state( _papi_hwi_system_info.num_cntrs, &spc->state );
	*dp = ( long long * ) spc->state.sum.pmc;
#ifdef DEBUG
	{
		if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
			unsigned int i;
			for ( i = 0; i < spc->control.cpu_control.nractrs; i++ ) {
				SUBDBG( "raw val hardware index %d is %lld\n", i,
						( long long ) spc->state.sum.pmc[i] );
			}
		}
	}
#endif
	return ( PAPI_OK );
}


inline_static long long
get_cycles( void )
{
	__asm rdtsc						   // Read Time Stamp Counter
		// This assembly instruction places the 64-bit value in edx:eax
		// Which is exactly where it needs to be for a 64-bit return value...
}

/* Low level functions, should not handle errors, just return codes. */

long long
_papi_hwd_get_real_usec( void )
{
	return ( ( long long ) get_cycles(  ) /
			 ( long long ) _papi_hwi_system_info.hw_info.mhz );
}

long long
_papi_hwd_get_real_cycles( void )
{
	return ( ( long long ) get_cycles(  ) );
}

#ifdef DEBUG
void
print_control( const struct pmc_cpu_control *control )
{
	unsigned int i;

	SUBDBG( "Control used:\n" );
	SUBDBG( "tsc_on\t\t\t%u\n", control->tsc_on );
	SUBDBG( "nractrs\t\t\t%u\n", control->nractrs );
	SUBDBG( "nrictrs\t\t\t%u\n", control->nrictrs );
	for ( i = 0; i < ( control->nractrs + control->nrictrs ); ++i ) {
		if ( control->pmc_map[i] >= 18 ) {
			SUBDBG( "pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i] );
		} else {
			SUBDBG( "pmc_map[%u]\t\t%u\n", i, control->pmc_map[i] );
		}
		SUBDBG( "evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i] );
		if ( control->ireset[i] )
			SUBDBG( "ireset[%u]\t%d\n", i, control->ireset[i] );
	}
}
#endif
