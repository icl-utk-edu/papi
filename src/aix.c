/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This file handles the OS dependent part of the POWER5 and POWER6 architectures.
  It supports both AIX 4 and AIX 5. The switch between AIX 4 and 5 is driven by the 
  system defined value _AIX_VERSION_510.
  Other routines also include minor conditionally compiled differences.
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_defines.h"
#include "papi_vector.h"
#include "papi_memory.h"

extern int _aix_allocate_registers( EventSetInfo_t * ESI );
extern int _aix_update_control_state( hwd_control_state_t * this_state,
									  NativeInfo_t * native, int count,
									  hwd_context_t * context );
extern int _aix_ntv_bits_to_info( hwd_register_t * bits, char *names,
								  unsigned int *values, int name_len,
								  int count );
extern int _aix_get_memory_info( PAPI_hw_info_t * mem_info, int type );
extern int _aix_get_dmem_info( PAPI_dmem_info_t * d );

/* Machine dependent info structure */
extern papi_mdi_t _papi_hwi_system_info;
extern pm_groups_info_t pmgroups;

/* define the vector structure at the bottom of this file */
extern papi_vector_t MY_VECTOR;

/* Locking variables */
volatile int lock_var[PAPI_MAX_LOCK] = { 0 };
atomic_p lock[PAPI_MAX_LOCK];

/* 
 some heap information, start_of_text, start_of_data .....
 ref: http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixprggd/genprogc/sys_mem_alloc.htm#HDRA9E4A4C9921SYLV 
*/
#define START_OF_TEXT &_text
#define END_OF_TEXT   &_etext
#define START_OF_DATA &_data
#define END_OF_DATA   &_edata
#define START_OF_BSS  &_edata
#define END_OF_BSS    &_end

static int maxgroups = 0;
struct utsname AixVer;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/* The following is for any POWER hardware */

/*  Trims trailing blank space and line endings from a string (in place).
    Returns pointer to start address */
static char *
trim_string( char *in )
{
	int len, i = 0;
	char *start = in;

	if ( in == NULL )
		return ( in );
	len = strlen( in );
	if ( len == 0 )
		return ( in );
	/* Trim right */
	i = strlen( start ) - 1;
	while ( i >= 0 ) {
		if ( isblank( start[i] ) || ( start[i] == '\r' ) ||
			 ( start[i] == '\n' ) )
			start[i] = '\0';
		else
			break;
		i--;
	}
	return ( start );
}


/* Routines to support an opaque native event table */
int
_aix_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{
	if ( ( EventCode & PAPI_NATIVE_AND_MASK ) >=
		 MY_VECTOR.cmp_info.num_native_events )
		return ( PAPI_ENOEVNT );
	strncpy( ntv_name,
			 native_name_map[EventCode & PAPI_NATIVE_AND_MASK].name, len );
	trim_string( ntv_name );
	if ( strlen( native_name_map[EventCode & PAPI_NATIVE_AND_MASK].name ) >
		 len - 1 )
		return ( PAPI_EBUF );
	return ( PAPI_OK );
}

int
_aix_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
	if ( ( EventCode & PAPI_NATIVE_AND_MASK ) >=
		 MY_VECTOR.cmp_info.num_native_events )
		return ( PAPI_ENOEVNT );
	strncpy( ntv_descr,
			 native_table[native_name_map[EventCode & PAPI_NATIVE_AND_MASK].
						  index].description, len );
	trim_string( ntv_descr );
	if ( strlen
		 ( native_table
		   [native_name_map[EventCode & PAPI_NATIVE_AND_MASK].index].
		   description ) > len - 1 )
		return ( PAPI_EBUF );
	return ( PAPI_OK );
}

int
_aix_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	bits = &native_table[EventCode & PAPI_NATIVE_AND_MASK].resources;	/* it is not right, different type */
	return ( PAPI_OK );
}

/* this function return the next native event code.
    modifier = PAPI_ENUM_FIRST returns first native event code
    modifier = PAPI_ENUM_EVENTS returns next native event code
    modifier = PAPI_NTV_ENUM_GROUPS return groups in which this
				native event lives, in bits 16 - 23 of event code
				terminating with PAPI_ENOEVNT at the end of the list.
   function return value:
     PAPI_OK successful, event code is valid
	 PAPI_EINVAL bad modifier
     PAPI_ENOEVNT end of list or fail, event code is invalid
*/
int
_aix_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	if ( modifier == PAPI_ENUM_FIRST ) {
		*EventCode = PAPI_NATIVE_MASK;
		return ( PAPI_OK );
	}
	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK;

		if ( native_table[index + 1].resources.selector ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	} else if ( modifier == PAPI_NTV_ENUM_GROUPS ) {
#if defined(_POWER5) || defined(_POWER6)
		unsigned int group =
			( *EventCode & PAPI_NTV_GROUP_AND_MASK ) >> PAPI_NTV_GROUP_SHIFT;
		int index = *EventCode & 0x000000FF;
		int i;
		unsigned int tmpg;

		*EventCode = *EventCode & ( ~PAPI_NTV_GROUP_SHIFT );
		for ( i = 0; i < GROUP_INTS; i++ ) {
			tmpg = native_table[index].resources.group[i];
			if ( group != 0 ) {
				while ( ( ffs( tmpg ) + i * 32 ) <= group && tmpg != 0 )
					tmpg = tmpg ^ ( 1 << ( ffs( tmpg ) - 1 ) );
			}
			if ( tmpg != 0 ) {
				group = ffs( tmpg ) + i * 32;
				*EventCode = *EventCode | ( group << PAPI_NTV_GROUP_SHIFT );
				return ( PAPI_OK );
			}
		}
#endif
		return ( PAPI_ENOEVNT );
	} else
		return ( PAPI_EINVAL );
}

static void
set_config( hwd_control_state_t * ptr, int arg1, int arg2 )
{
	ptr->counter_cmd.events[arg1] = arg2;
}

static void
unset_config( hwd_control_state_t * ptr, int arg1 )
{
	ptr->counter_cmd.events[arg1] = 0;
}

int
init_domain(  )
{
	int domain = 0;

	domain = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_OTHER;
#ifdef PM_INITIALIZE
#ifdef _AIXVERSION_510
	if ( pminfo.proc_feature.b.hypervisor ) {
		domain |= PAPI_DOM_SUPERVISOR;
	}
#endif
#endif
	return ( domain );
}

static int
_aix_set_domain( hwd_control_state_t * this_state, int domain )
{
	pm_mode_t *mode = &( this_state->counter_cmd.mode );
	int did = 0;

	mode->b.user = 0;
	mode->b.kernel = 0;
	if ( domain & PAPI_DOM_USER ) {
		did++;
		mode->b.user = 1;
	}
	if ( domain & PAPI_DOM_KERNEL ) {
		did++;
		mode->b.kernel = 1;
	}
#ifdef PM_INITIALIZE
#ifdef _AIXVERSION_510
	if ( ( domain & PAPI_DOM_SUPERVISOR ) && pminfo.proc_feature.b.hypervisor ) {
		did++;
		mode->b.hypervisor = 1;
	}
#endif
#endif
	if ( did )
		return ( PAPI_OK );
	else
		return ( PAPI_EINVAL );
/*
  switch (domain)
    {
    case PAPI_DOM_USER:
      mode->b.user = 1;
      mode->b.kernel = 0;
      break;
    case PAPI_DOM_KERNEL:
      mode->b.user = 0;
      mode->b.kernel = 1;
      break;
    case PAPI_DOM_ALL:
      mode->b.user = 1;
      mode->b.kernel = 1;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
*/
}

int
_aix_set_granularity( hwd_control_state_t * this_state, int domain )
{
	pm_mode_t *mode = &( this_state->counter_cmd.mode );

	switch ( domain ) {
	case PAPI_GRN_THR:
		mode->b.process = 0;
		mode->b.proctree = 0;
		break;
		/* case PAPI_GRN_PROC:
		   mode->b.process = 1;
		   mode->b.proctree = 0;
		   break;
		   case PAPI_GRN_PROCG:
		   mode->b.process = 0;
		   mode->b.proctree = 1;
		   break; */
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_OK );
}

static int
set_default_domain( EventSetInfo_t * zero, int domain )
{
	hwd_control_state_t *current_state = zero->ctl_state;
	return ( _aix_set_domain( current_state, domain ) );
}

static int
set_default_granularity( EventSetInfo_t * zero, int granularity )
{
	hwd_control_state_t *current_state = zero->ctl_state;
	return ( _aix_set_granularity( current_state, granularity ) );
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
int
_aix_mdi_init(  )
{
	int retval;

	if ( ( retval = uname( &AixVer ) ) < 0 )
		return ( PAPI_ESYS );
	if ( AixVer.version[0] == '4' ) {
		_papi_hwi_system_info.exe_info.address_info.text_start =
			( caddr_t ) START_OF_TEXT;
		_papi_hwi_system_info.exe_info.address_info.text_end =
			( caddr_t ) END_OF_TEXT;
		_papi_hwi_system_info.exe_info.address_info.data_start =
			( caddr_t ) START_OF_DATA;
		_papi_hwi_system_info.exe_info.address_info.data_end =
			( caddr_t ) END_OF_DATA;
		_papi_hwi_system_info.exe_info.address_info.bss_start =
			( caddr_t ) START_OF_BSS;
		_papi_hwi_system_info.exe_info.address_info.bss_end =
			( caddr_t ) END_OF_BSS;
	} else {
		_aix_update_shlib_info(  );
	}

/*   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.sub_info.fast_real_timer = 1;
   _papi_hwi_system_info.sub_info->available_domains = init_domain();*/


	return ( PAPI_OK );
}


static int
_aix_get_system_info( void )
{
	int retval;
	/* pm_info_t pminfo; */
	struct procsinfo psi = { 0 };
	pid_t pid;
	char maxargs[PAPI_HUGE_STR_LEN];
	char pname[PAPI_HUGE_STR_LEN];

	pid = getpid(  );
	if ( pid == -1 )
		return ( PAPI_ESYS );
	_papi_hwi_system_info.pid = pid;
	psi.pi_pid = pid;
	retval = getargs( &psi, sizeof ( psi ), maxargs, PAPI_HUGE_STR_LEN );
	if ( retval == -1 )
		return ( PAPI_ESYS );

	if ( realpath( maxargs, pname ) )
		strncpy( _papi_hwi_system_info.exe_info.fullname, pname,
				 PAPI_HUGE_STR_LEN );
	else
		strncpy( _papi_hwi_system_info.exe_info.fullname, maxargs,
				 PAPI_HUGE_STR_LEN );

	strcpy( _papi_hwi_system_info.exe_info.address_info.name,
			basename( maxargs ) );

#ifdef _POWER6
	/* problem with pm_initialize(): it cannot be called multiple times with 
	   PM_CURRENT; use instead the actual proc type - here PM_POWER6 - 
	   and multiple invocations are no longer a problem */ 
	retval = pm_initialize( PM_INIT_FLAGS, &pminfo, &pmgroups, PM_POWER6 );
#else
#ifdef _AIXVERSION_510
#ifdef PM_INITIALIZE
	SUBDBG( "Calling AIX 5 version of pm_initialize...\n" );
/*#if defined(_POWER5)
    retval = pm_initialize(PM_INIT_FLAGS, &pminfo, &pmgroups, PM_POWER5);
#endif*/
	retval = pm_initialize( PM_INIT_FLAGS, &pminfo, &pmgroups, PM_CURRENT );
#else
	SUBDBG( "Calling AIX 5 version of pm_init...\n" );
	retval = pm_init( PM_INIT_FLAGS, &pminfo, &pmgroups );
#endif

#else
	SUBDBG( "Calling AIX 4 version of pm_init...\n" );
	retval = pm_init( PM_INIT_FLAGS, &pminfo );
#endif
#endif
	SUBDBG( "...Back from pm_init\n" );

	if ( retval > 0 )
		return ( retval );

	strcpy( MY_VECTOR.cmp_info.name, "$Id$" );	/* Name of the substrate we're using */

	_aix_mdi_init(  );

	_papi_hwi_system_info.hw_info.nnodes = 1;
	_papi_hwi_system_info.hw_info.ncpu = _system_configuration.ncpus;
	_papi_hwi_system_info.hw_info.totalcpus =
		_papi_hwi_system_info.hw_info.ncpu *
		_papi_hwi_system_info.hw_info.nnodes;
	_papi_hwi_system_info.hw_info.vendor = -1;
	strcpy( _papi_hwi_system_info.hw_info.vendor_string, "IBM" );
	_papi_hwi_system_info.hw_info.model = _system_configuration.implementation;
	strcpy( _papi_hwi_system_info.hw_info.model_string, pminfo.proc_name );
	_papi_hwi_system_info.hw_info.revision =
		( float ) _system_configuration.version;
	_papi_hwi_system_info.hw_info.mhz = ( float ) ( pm_cycles(  ) / 1000000.0 );
/*   _papi_hwi_system_info.num_gp_cntrs = pminfo.maxpmcs;*/
	MY_VECTOR.cmp_info.num_cntrs = pminfo.maxpmcs;
	MY_VECTOR.cmp_info.cntr_groups = 1;
	MY_VECTOR.cmp_info.available_granularities = PAPI_GRN_THR;
/* This field doesn't appear to exist in the PAPI 3.0 structure 
  _papi_hwi_system_info.cpunum = mycpu(); 
*/
	MY_VECTOR.cmp_info.available_domains = init_domain(  );
	return ( PAPI_OK );
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long
_aix_get_real_usec( void )
{
	timebasestruct_t t;
	long long retval;

	read_real_time( &t, TIMEBASE_SZ );
	time_base_to_time( &t, TIMEBASE_SZ );
	retval = ( t.tb_high * 1000000 ) + t.tb_low / 1000;
	return ( retval );
}

long long
_aix_get_real_cycles( void )
{
	return ( _aix_get_real_usec(  ) *
			 ( long long ) _papi_hwi_system_info.hw_info.mhz );
}

long long
_aix_get_virt_usec( const hwd_context_t * context )
{
	long long retval;
	struct tms buffer;

	times( &buffer );
	SUBDBG( "user %d system %d\n", ( int ) buffer.tms_utime,
			( int ) buffer.tms_stime );
	retval =
		( long long ) ( ( buffer.tms_utime + buffer.tms_stime ) *
						( 1000000 / CLK_TCK ) );
	return ( retval );
}

long long
_aix_get_virt_cycles( const hwd_context_t * context )
{
	return ( _aix_get_virt_usec( context ) *
			 ( long long ) _papi_hwi_system_info.hw_info.mhz );
}

static void
_aix_lock_init( void )
{
	int i;
	for ( i = 0; i < PAPI_MAX_LOCK; i++ )
		lock[i] = ( int * ) ( lock_var + i );
}

/*
papi_svector_t  _aix_table[] = {
     {( void ( * )(  ) ) _papi_hwd_get_overflow_address,
      VEC_PAPI_HWD_GET_OVERFLOW_ADDRESS},
     {( void ( * )(  ) ) _papi_hwd_update_shlib_info,
      VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
     {( void ( * )(  ) ) _papi_hwd_init, VEC_PAPI_HWD_INIT},
     {( void ( * )(  ) ) _papi_hwd_dispatch_timer,
      VEC_PAPI_HWD_DISPATCH_TIMER},
     {( void ( * )(  ) ) _papi_hwd_ctl, VEC_PAPI_HWD_CTL},
     {( void ( * )(  ) ) _papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
     {( void ( * )(  ) ) _papi_hwd_get_real_cycles,
      VEC_PAPI_HWD_GET_REAL_CYCLES},
     {( void ( * )(  ) ) _papi_hwd_get_virt_cycles,
      VEC_PAPI_HWD_GET_VIRT_CYCLES},
     {( void ( * )(  ) ) _papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
     {( void ( * )(  ) ) _papi_hwd_start, VEC_PAPI_HWD_START},
     {( void ( * )(  ) ) _papi_hwd_stop, VEC_PAPI_HWD_STOP},
     {( void ( * )(  ) ) _papi_hwd_read, VEC_PAPI_HWD_READ},
     {( void ( * )(  ) ) _papi_hwd_reset, VEC_PAPI_HWD_RESET},
     {( void ( * )(  ) ) _papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
     {( void ( * )(  ) ) _papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
     {( void ( * )(  ) ) _papi_hwd_ntv_enum_events,
      VEC_PAPI_HWD_NTV_ENUM_EVENTS},
     {( void ( * )(  ) ) _papi_hwd_ntv_code_to_name,
      VEC_PAPI_HWD_NTV_CODE_TO_NAME},
     {( void ( * )(  ) ) _papi_hwd_ntv_code_to_descr,
      VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
     {( void ( * )(  ) ) _papi_hwd_ntv_code_to_bits,
      VEC_PAPI_HWD_NTV_CODE_TO_BITS},
     {( void ( * )(  ) ) _papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN},
     {NULL, VEC_PAPI_END}
};
*/

int
_aix_shutdown( hwd_context_t * ctx )
{
	return ( PAPI_OK );
}

int
_aix_init_substrate( int cidx )
{
	int retval = PAPI_OK, procidx;

	/* Fill in what we can of the papi_system_info. */
	retval = MY_VECTOR.get_system_info(  );
	if ( retval )
		return ( retval );

	/* Setup memory info */
	retval = MY_VECTOR.get_memory_info( &_papi_hwi_system_info.hw_info, 0 );
	if ( retval )
		return ( retval );

	SUBDBG( "Found %d %s %s CPU's at %f Mhz.\n",
			_papi_hwi_system_info.hw_info.totalcpus,
			_papi_hwi_system_info.hw_info.vendor_string,
			_papi_hwi_system_info.hw_info.model_string,
			_papi_hwi_system_info.hw_info.mhz );

	MY_VECTOR.cmp_info.CmpIdx = cidx;
	MY_VECTOR.cmp_info.num_native_events = ppc64_setup_native_table(  );

/*#if !defined(_POWER5)
   if (!_papi_hwd_init_preset_search_map(&pminfo)){ 
      return (PAPI_ESBSTR);}

   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);
#else
   _papi_pmapi_setup_presets("POWER5", 0);
#endif*/
	procidx = pm_get_procindex(  );
	switch ( procidx ) {
	case PM_POWER5:
		_papi_pmapi_setup_presets( "POWER5", 0 );
		break;
	case PM_POWER5_II:
		_papi_pmapi_setup_presets( "POWER5+", 0 );
		break;
	case PM_POWER6:
		_papi_pmapi_setup_presets( "POWER6", 0 );
		break;
	case PM_PowerPC970:
		_papi_pmapi_setup_presets( "PPC970", 0 );
		break;
	default:
		fprintf( stderr, "%s is not supported!\n", pminfo.proc_name );
		return ( PAPI_ESBSTR );
	}

	_aix_lock_init(  );

	return ( retval );
}

int
_aix_init( hwd_context_t * context )
{
	int retval;
	/* Initialize our global control state. */

	_aix_init_control_state( &context->cntrl );
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int
get_avail_hwcntr_bits( int cntr_avail_bits )
{
	int tmp = 0, i = 1 << ( POWER_MAX_COUNTERS - 1 );

	while ( i ) {
		tmp = i & cntr_avail_bits;
		if ( tmp )
			return ( tmp );
		i = i >> 1;
	}
	return ( 0 );
}

static void
set_hwcntr_codes( int selector, unsigned char *from, int *to )
{
	int useme, i;

	for ( i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++ ) {
		useme = ( 1 << i ) & selector;
		if ( useme ) {
			to[i] = from[i];
		}
	}
}


#ifdef DEBUG
void
dump_cmd( pm_prog_t * t )
{
	SUBDBG( "mode.b.threshold %d\n", t->mode.b.threshold );
	SUBDBG( "mode.b.spare %d\n", t->mode.b.spare );
	SUBDBG( "mode.b.process %d\n", t->mode.b.process );
	SUBDBG( "mode.b.kernel %d\n", t->mode.b.kernel );
	SUBDBG( "mode.b.user %d\n", t->mode.b.user );
	SUBDBG( "mode.b.count %d\n", t->mode.b.count );
	SUBDBG( "mode.b.proctree %d\n", t->mode.b.proctree );
	SUBDBG( "events[0] %d\n", t->events[0] );
	SUBDBG( "events[1] %d\n", t->events[1] );
	SUBDBG( "events[2] %d\n", t->events[2] );
	SUBDBG( "events[3] %d\n", t->events[3] );
	SUBDBG( "events[4] %d\n", t->events[4] );
	SUBDBG( "events[5] %d\n", t->events[5] );
	SUBDBG( "events[6] %d\n", t->events[6] );
	SUBDBG( "events[7] %d\n", t->events[7] );
	SUBDBG( "reserved %d\n", t->reserved );
}

void
dump_data( long long *vals )
{
	int i;

	for ( i = 0; i < MAX_COUNTERS; i++ ) {
		SUBDBG( "counter[%d] = %lld\n", i, vals[i] );
	}
}
#endif

int
_aix_reset( hwd_context_t * ESI, hwd_control_state_t * zero )
{
	int retval = pm_reset_data_mythread(  );
	if ( retval > 0 ) {
		if ( _papi_hwi_error_level != PAPI_QUIET )
			pm_error( "PAPI Error: pm_reset_data_mythread", retval );
		return ( retval );
	}
	return ( PAPI_OK );
}


int
_aix_read( hwd_context_t * ctx, hwd_control_state_t * spc,
		   long long **vals, int flags )
{
	int retval;

	retval = pm_get_data_mythread( &spc->state );
	if ( retval > 0 ) {
		if ( _papi_hwi_error_level != PAPI_QUIET )
			pm_error( "PAPI Error: pm_get_data_mythread", retval );
		return ( retval );
	}

	*vals = spc->state.accu;

#ifdef DEBUG
	if ( ISLEVEL( DEBUG_SUBSTRATE ) )
		dump_data( *vals );
#endif

	return ( PAPI_OK );
}

inline_static int
round_requested_ns( int ns )
{
	if ( ns <= _aix_vector.cmp_info.itimer_res_ns ) {
		return _aix_vector.cmp_info.itimer_res_ns;
	} else {
		int leftover_ns = ns % _aix_vector.cmp_info.itimer_res_ns;
		return ( ns - leftover_ns + _aix_vector.cmp_info.itimer_res_ns );
	}
}

int
_aix_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	switch ( code ) {
/* I don't understand what it means to set the default domain 
    case PAPI_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
*/
	case PAPI_DOMAIN:
		return ( _aix_set_domain
				 ( option->domain.ESI->ctl_state, option->domain.domain ) );
/* I don't understand what it means to set the default granularity 
    case PAPI_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
*/
	case PAPI_GRANUL:
		return ( _aix_set_granularity
				 ( option->domain.ESI->ctl_state,
				   option->granularity.granularity ) );
#if 0
	case PAPI_INHERIT:
		return ( set_inherit( option->inherit.inherit ) );
#endif
	case PAPI_DEF_ITIMER:
	{
		/* flags are currently ignored, eventually the flags will be able
		   to specify whether or not we use POSIX itimers (clock_gettimer) */
		if ( ( option->itimer.itimer_num == ITIMER_REAL ) &&
			 ( option->itimer.itimer_sig != SIGALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_VIRTUAL ) &&
			 ( option->itimer.itimer_sig != SIGVTALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_PROF ) &&
			 ( option->itimer.itimer_sig != SIGPROF ) )
			return PAPI_EINVAL;
		if ( option->itimer.ns > 0 )
			option->itimer.ns = round_requested_ns( option->itimer.ns );
		/* At this point, we assume the user knows what he or
		   she is doing, they maybe doing something arch specific */
		return PAPI_OK;
	}
	case PAPI_DEF_MPX_NS:
	{
		option->multiplex.ns = round_requested_ns( option->multiplex.ns );
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		option->itimer.ns = round_requested_ns( option->itimer.ns );
		return ( PAPI_OK );
	}
	default:
		return ( PAPI_ENOSUPP );
	}
}

void
_aix_dispatch_timer( int signal, siginfo_t * si, void *i )
{
	_papi_hwi_context_t ctx;
	ThreadInfo_t *t = NULL;
	caddr_t address;

	ctx.si = si;
	ctx.ucontext = ( hwd_ucontext_t * ) i;

	address = ( caddr_t ) GET_OVERFLOW_ADDRESS( ( &ctx ) );
	_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL, 0, 0,
										&t, MY_VECTOR.cmp_info.CmpIdx );
}

int
_aix_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	hwd_control_state_t *this_state = ESI->ctl_state;

	return ( PAPI_OK );
}

void *
_aix_get_overflow_address( void *context )
{
	void *location;
	struct sigcontext *info = ( struct sigcontext * ) context;
	location = ( void * ) info->sc_jmpbuf.jmp_context.iar;

	return ( location );
}


/* Copy the current control_state into the new thread context */
/*int _papi_hwd_start(EventSetInfo_t *ESI, EventSetInfo_t *zero)*/
int
_aix_start( hwd_context_t * ctx, hwd_control_state_t * cntrl )
{
	int i, retval;
	hwd_control_state_t *current_state = &ctx->cntrl;

	/* If we are nested, merge the global counter structure
	   with the current eventset */

	SUBDBG( "Start\n" );

	/* Copy the global counter structure to the current eventset */

	SUBDBG( "Copying states\n" );
	memcpy( current_state, cntrl, sizeof ( hwd_control_state_t ) );

	retval = pm_set_program_mythread( &current_state->counter_cmd );
	if ( retval > 0 ) {
		if ( retval == 13 ) {
			retval = pm_delete_program_mythread(  );
			if ( retval > 0 ) {
				if ( _papi_hwi_error_level != PAPI_QUIET )
					pm_error( "PAPI Error: pm_delete_program_mythread",
							  retval );
				return ( retval );
			}
			retval = pm_set_program_mythread( &current_state->counter_cmd );
			if ( retval > 0 ) {
				if ( _papi_hwi_error_level != PAPI_QUIET )
					pm_error( "PAPI Error: pm_set_program_mythread", retval );
				return ( retval );
			}
		} else {
			if ( _papi_hwi_error_level != PAPI_QUIET )
				pm_error( "PAPI Error: pm_set_program_mythread", retval );
			return ( retval );
		}
	}

	/* Set up the new merged control structure */

#if 0
	dump_cmd( &current_state->counter_cmd );
#endif

	/* Start the counters */

	retval = pm_start_mythread(  );
	if ( retval > 0 ) {
		if ( _papi_hwi_error_level != PAPI_QUIET )
			pm_error( "pm_start_mythread()", retval );
		return ( retval );
	}

	return ( PAPI_OK );
}

int
_aix_stop( hwd_context_t * ctx, hwd_control_state_t * cntrl )
{
	int retval;

	retval = pm_stop_mythread(  );
	if ( retval > 0 ) {
		if ( _papi_hwi_error_level != PAPI_QUIET )
			pm_error( "pm_stop_mythread()", retval );
		return ( retval );
	}

	retval = pm_delete_program_mythread(  );
	if ( retval > 0 ) {
		if ( _papi_hwi_error_level != PAPI_QUIET )
			pm_error( "pm_delete_program_mythread()", retval );
		return ( retval );
	}

	return ( PAPI_OK );
}

int
_aix_update_shlib_info( void )
{
#if ( ( defined( _AIXVERSION_510) || defined(_AIXVERSION_520)))
	struct ma_msg_s
	{
		long flag;
		char *name;
	} ma_msgs[] = {
		{
		MA_MAINEXEC, "MAINEXEC"}, {
		MA_KERNTEXT, "KERNTEXT"}, {
		MA_READ, "READ"}, {
		MA_WRITE, "WRITE"}, {
		MA_EXEC, "EXEC"}, {
		MA_SHARED, "SHARED"}, {
		MA_BREAK, "BREAK"}, {
	MA_STACK, "STACK"},};

	char fname[80], name[PAPI_HUGE_STR_LEN];
	prmap_t newp;
	int count, t_index, retval, i, j, not_first_flag_bit;
	FILE *map_f;
	void *vaddr;
	prmap_t *tmp1 = NULL;
	PAPI_address_map_t *tmp2 = NULL;

	sprintf( fname, "/proc/%d/map", getpid(  ) );
	map_f = fopen( fname, "r" );
	if ( !map_f ) {
		PAPIERROR( "fopen(%s) returned < 0", fname );
		return ( PAPI_OK );
	}

	/* count the entries we need */
	count = 0;
	t_index = 0;
	while ( ( retval = fread( &newp, sizeof ( prmap_t ), 1, map_f ) ) > 0 ) {
		if ( newp.pr_pathoff > 0 && newp.pr_mapname[0] != '\0' ) {
			if ( newp.pr_mflags & MA_STACK )
				continue;

			count++;
			SUBDBG( "count=%d offset=%ld map=%s\n", count,
					newp.pr_pathoff, newp.pr_mapname );

			if ( ( newp.pr_mflags & MA_READ ) && ( newp.pr_mflags & MA_EXEC ) )
				t_index++;
		}
	}
	rewind( map_f );
	tmp1 = ( prmap_t * ) papi_calloc( ( count + 1 ), sizeof ( prmap_t ) );
	if ( tmp1 == NULL )
		return ( PAPI_ENOMEM );

	tmp2 =
		( PAPI_address_map_t * ) papi_calloc( t_index,
											  sizeof ( PAPI_address_map_t ) );
	if ( tmp2 == NULL )
		return ( PAPI_ENOMEM );

	i = 0;
	t_index = -1;
	while ( ( retval = fread( &tmp1[i], sizeof ( prmap_t ), 1, map_f ) ) > 0 ) {
		if ( tmp1[i].pr_pathoff > 0 && tmp1[i].pr_mapname[0] != '\0' )
			if ( !( tmp1[i].pr_mflags & MA_STACK ) )
				i++;
	}
	for ( i = 0; i < count; i++ ) {
		char c;
		int cc = 0;

		retval = fseek( map_f, tmp1[i].pr_pathoff, SEEK_SET );
		if ( retval != 0 )
			return ( PAPI_ESYS );
		while ( fscanf( map_f, "%c", &c ) != EOF ) {
			name[cc] = c;
			/* how many char are hold in /proc/xxxx/map */
			cc++;
			if ( c == '\0' )
				break;
		}


		/* currently /proc/xxxx/map file holds only 33 char per line (incl NULL char);
		 * if executable name > 32 char, compare first 32 char only */
		if ( strncmp( _papi_hwi_system_info.exe_info.address_info.name,
					  basename( name ), cc - 1 ) == 0 ) {
			if ( strlen( _papi_hwi_system_info.exe_info.address_info.name ) !=
				 cc - 1 )
				PAPIERROR
					( "executable name too long (%d char). Match of first %d char only",
					  strlen( _papi_hwi_system_info.exe_info.address_info.
							  name ), cc - 1 );

			if ( tmp1[i].pr_mflags & MA_READ ) {
				if ( tmp1[i].pr_mflags & MA_EXEC ) {
					_papi_hwi_system_info.exe_info.address_info.
						text_start = ( caddr_t ) tmp1[i].pr_vaddr;
					_papi_hwi_system_info.exe_info.address_info.
						text_end =
						( caddr_t ) ( tmp1[i].pr_vaddr + tmp1[i].pr_size );
				} else if ( tmp1[i].pr_mflags & MA_WRITE ) {
					_papi_hwi_system_info.exe_info.address_info.
						data_start = ( caddr_t ) tmp1[i].pr_vaddr;
					_papi_hwi_system_info.exe_info.address_info.
						data_end =
						( caddr_t ) ( tmp1[i].pr_vaddr + tmp1[i].pr_size );
				}
			}

		} else {
			if ( ( _papi_hwi_system_info.exe_info.address_info.text_start == 0 )
				 && ( _papi_hwi_system_info.exe_info.address_info.text_end ==
					  0 ) &&
				 ( _papi_hwi_system_info.exe_info.address_info.data_start == 0 )
				 && ( _papi_hwi_system_info.exe_info.address_info.data_end ==
					  0 ) )
				PAPIERROR( "executable name not recognized" );

			if ( tmp1[i].pr_mflags & MA_READ ) {
				if ( tmp1[i].pr_mflags & MA_EXEC ) {
					t_index++;
					tmp2[t_index].text_start = ( caddr_t ) tmp1[i].pr_vaddr;
					tmp2[t_index].text_end =
						( caddr_t ) ( tmp1[i].pr_vaddr + tmp1[i].pr_size );
					strncpy( tmp2[t_index].name, name, PAPI_MAX_STR_LEN );
				} else if ( tmp1[i].pr_mflags & MA_WRITE ) {
					tmp2[t_index].data_start = ( caddr_t ) tmp1[i].pr_vaddr;
					tmp2[t_index].data_end =
						( caddr_t ) ( tmp1[i].pr_vaddr + tmp1[i].pr_size );
				}
			}

		}
	}
	fclose( map_f );

	if ( _papi_hwi_system_info.shlib_info.map )
		papi_free( _papi_hwi_system_info.shlib_info.map );
	_papi_hwi_system_info.shlib_info.map = tmp2;
	_papi_hwi_system_info.shlib_info.count = t_index + 1;
	papi_free( tmp1 );

	return ( PAPI_OK );
#else
	return PAPI_ESBSTR;
#endif
}

papi_vector_t _aix_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .itimer_sig = PAPI_INT_MPX_SIGNAL,
				 .itimer_num = PAPI_INT_ITIMER,
				 .itimer_res_ns = 1,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 1,
				 .fast_virtual_timer = 1,
				 .attach = 1,
				 .attach_must_ptrace = 1,
				 .cntr_umasks = 1,
				 }
	,

	/* sizes of framework-opaque component-private structures
	   these are remapped in pmapi_ppc64.h, ppc64_events.h */
	.size = {
			 .context = sizeof ( hwd_context_t ),
			 .control_state = sizeof ( hwd_control_state_t ),
			 .reg_value = sizeof ( hwd_register_t ),
			 .reg_alloc = sizeof ( hwd_reg_alloc_t ),
			 }
	,

	/* function pointers in this component */
	.init_control_state = _aix_init_control_state,
	.start = _aix_start,
	.stop = _aix_stop,
	.read = _aix_read,
	.allocate_registers = _aix_allocate_registers,
	.update_control_state = _aix_update_control_state,
	.set_domain = _aix_set_domain,
	.reset = _aix_reset,
	.set_overflow = _aix_set_overflow,
/*    .stop_profiling =		_aix_stop_profiling, */
	.ntv_enum_events = _aix_ntv_enum_events,
/*    .ntv_name_to_code =		_aix_ntv_name_to_code, */
	.ntv_code_to_name = _aix_ntv_code_to_name,
	.ntv_code_to_descr = _aix_ntv_code_to_descr,
	.ntv_code_to_bits = _aix_ntv_code_to_bits,
	.ntv_bits_to_info = _aix_ntv_bits_to_info,

	/* from OS */
	.get_memory_info = _aix_get_memory_info,
	.get_system_info = _aix_get_system_info,
	.init_substrate = _aix_init_substrate,
	.ctl = _aix_ctl,
	.dispatch_timer = _aix_dispatch_timer,
	.init = _aix_init,
	.get_dmem_info = _aix_get_dmem_info,
	.shutdown = _aix_shutdown,
	.get_real_usec = _aix_get_real_usec,
	.get_real_cycles = _aix_get_real_cycles,
	.get_virt_cycles = _aix_get_virt_cycles,
	.get_virt_usec = _aix_get_virt_usec
};
