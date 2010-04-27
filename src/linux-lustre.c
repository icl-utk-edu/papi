/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    linux-lustre.c
* Author:  Haihang You
*          you@eecs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/
#include <inttypes.h>
#include "papi.h"
#include "papi_internal.h"
#include "linux-lustre.h"
#include "papi_memory.h"

extern int get_cpu_info( PAPI_hw_info_t * hwinfo );
void lustre_init_mdi(  );
int lustre_init_presets(  );
extern counter_info *subscriptions[];

#define lustre_native_table subscriptions

long long _papi_hwd_lustre_register_start[LUSTRE_MAX_COUNTERS];
long long _papi_hwd_lustre_register[LUSTRE_MAX_COUNTERS];

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
LUSTRE_init_substrate(  )
{
	int retval = PAPI_OK, i;

//   host_initialize();

//   lustre_native_table = subscriptions;

	for ( i = 0; i < LUSTRE_MAX_COUNTERS; i++ ) {
		_papi_hwd_lustre_register_start[i] = -1;
		_papi_hwd_lustre_register[i] = -1;
	}
	/* Internal function, doesn't necessarily need to be a function */
	lustre_init_mdi(  );

	/* Internal function, doesn't necessarily need to be a function */
	lustre_init_presets(  );

	return ( retval );
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
hwi_search_t lustre_preset_map[] = {
	{0, {0, {PAPI_NULL, PAPI_NULL}
		 , {0,}
		 }
	 }
};


int
lustre_init_presets(  )
{
	return ( _papi_hwi_setup_all_presets( lustre_preset_map, NULL ) );
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */
void
lustre_init_mdi(  )
{
/* 
   get_cpu_info(&_papi_hwi_system_info.hw_info);
   _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
   _papi_hwi_system_info.supports_program = 0;
   _papi_hwi_system_info.supports_write = 0;
   _papi_hwi_system_info.supports_hw_overflow = 0;
   _papi_hwi_system_info.supports_hw_profile = 0;
   _papi_hwi_system_info.supports_multiple_threads = 0;
   _papi_hwi_system_info.supports_64bit_counters = 0;
   _papi_hwi_system_info.supports_attach = 0;
   _papi_hwi_system_info.supports_real_usec = 0;
   _papi_hwi_system_info.supports_real_cyc = 0;
   _papi_hwi_system_info.supports_virt_usec = 0;
   _papi_hwi_system_info.supports_virt_cyc = 0;
   _papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);
*/ }


/*
 * This is called whenever a thread is initialized
 */
int
LUSTRE_init( hwd_context_t * ctx )
{
	string_list *counter_list = NULL;
	int i;

	host_initialize(  );
	counter_list = host_listCounter(  );
	for ( i = 0; i < counter_list->count; i++ )
		host_subscribe( counter_list->data[i] );
	host_deleteStringList( counter_list );
	( ( LUSTRE_context_t * ) ctx )->state.ncounter = counter_list->count;

	//lustre_native_table = subscriptions;
//   for(i=0;i<counter_list->count;i++)
//    printf("%d   %s\n", i, subscriptions[i]->name);

	return ( PAPI_OK );
}

int
LUSTRE_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;
	host_finalize(  );
	return ( PAPI_OK );
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
LUSTRE_init_control_state( hwd_control_state_t * ptr )
{
	( void ) ptr;
	return PAPI_OK;
}

int
LUSTRE_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native,
							 int count, hwd_context_t * ctx )
{
	( void ) ptr;
	( void ) ctx;
	int i, index;

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = index;
	}
	return ( PAPI_OK );
}

int
LUSTRE_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;
	host_read_values( _papi_hwd_lustre_register_start );
	memcpy( _papi_hwd_lustre_register, _papi_hwd_lustre_register_start,
			LUSTRE_MAX_COUNTERS * sizeof ( long long ) );

	return ( PAPI_OK );
}


int
LUSTRE_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
			 long long **events, int flags )
{
	( void ) ctx;
	( void ) flags;
	int i;

	host_read_values( _papi_hwd_lustre_register );
	for ( i = 0; i < ( ( LUSTRE_control_state_t * ) ctrl )->ncounter; i++ ) {
		( ( LUSTRE_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_lustre_register[i] - _papi_hwd_lustre_register_start[i];
		/*printf("%d  %lld\n", i, ctrl->counts[i]); */
	}
	*events = ( ( LUSTRE_control_state_t * ) ctrl )->counts;
	return ( PAPI_OK );
}

int
LUSTRE_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	int i;

	host_read_values( _papi_hwd_lustre_register );
	for ( i = 0; i < ( ( LUSTRE_control_state_t * ) ctrl )->ncounter; i++ ) {
		( ( LUSTRE_control_state_t * ) ctrl )->counts[i] =
			_papi_hwd_lustre_register[i] - _papi_hwd_lustre_register_start[i];
/*      printf("%d  %lld\n", i, ctrl->counts[i]);*/
	}

	return ( PAPI_OK );
}

int
LUSTRE_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	LUSTRE_start( ctx, ctrl );

	return ( PAPI_OK );
}

int
LUSTRE_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from )
{
	( void ) ctx;
	( void ) ctrl;
	( void ) from;
	return ( PAPI_OK );
}

/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
LUSTRE_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int
LUSTRE_set_domain( hwd_control_state_t * cntrl, int domain )
{
	( void ) cntrl;
	int found = 0;
	if ( PAPI_DOM_USER & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		found = 1;
	}
	if ( !found )
		return ( PAPI_EINVAL );
	return ( PAPI_OK );
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
/*long long _papi_hwd_get_real_usec(void)
{
   return(1);
}

long long _papi_hwd_get_real_cycles(void)
{
   return(1);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return(1);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return(1);
}
*/

int
LUSTRE_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK &
								 PAPI_COMPONENT_AND_MASK]->name, len );
	return ( PAPI_OK );
}

int
LUSTRE_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK &
								 PAPI_COMPONENT_AND_MASK]->description, len );
	return ( PAPI_OK );
}

int
LUSTRE_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	memcpy( ( LUSTRE_register_t * ) bits, lustre_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK], sizeof ( LUSTRE_register_t ) );	/* it is not right, different type */
	return ( PAPI_OK );
}

int
LUSTRE_ntv_bits_to_info( hwd_register_t * bits, char *names,
						 unsigned int *values, int name_len, int count )
{
	( void ) bits;
	( void ) names;
	( void ) values;
	( void ) name_len;
	( void ) count;
	return ( 1 );
}


int
LUSTRE_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	if ( modifier == PAPI_ENUM_FIRST ) {
		/* assumes first native event is always 0x4000000 */
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
		return ( PAPI_OK );
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( lustre_native_table[index + 1] ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	} else
		return ( PAPI_EINVAL );
}

/*
 * Shared Library Information and other Information Functions
 */
/*int _papi_hwd_update_shlib_info(void){
  return(PAPI_OK);
}
*/
papi_vector_t _Lustre_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name =
				 "$Id$",
				 .version = "$Revision$",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = LUSTRE_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( LUSTRE_context_t ),
			 .control_state = sizeof ( LUSTRE_control_state_t ),
			 .reg_value = sizeof ( LUSTRE_register_t ),
			 .reg_alloc = sizeof ( LUSTRE_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = LUSTRE_init,
	.init_substrate = LUSTRE_init_substrate,
	.init_control_state = LUSTRE_init_control_state,
	.start = LUSTRE_start,
	.stop = LUSTRE_stop,
	.read = LUSTRE_read,
	.shutdown = LUSTRE_shutdown,
	.ctl = LUSTRE_ctl,
	.update_control_state = LUSTRE_update_control_state,
	.set_domain = LUSTRE_set_domain,
	.reset = LUSTRE_reset,
/*    .set_overflow =		_p3_set_overflow,
    .stop_profiling =		_p3_stop_profiling,*/
	.ntv_enum_events = LUSTRE_ntv_enum_events,
	.ntv_code_to_name = LUSTRE_ntv_code_to_name,
	.ntv_code_to_descr = LUSTRE_ntv_code_to_descr,
	.ntv_code_to_bits = LUSTRE_ntv_code_to_bits,
	.ntv_bits_to_info = LUSTRE_ntv_bits_to_info
};
