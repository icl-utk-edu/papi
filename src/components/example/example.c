/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    example.c
 * @author  Joachim Protze
 *          joachim.protze@zih.tu-dresden.de
 * @author	Vince Weaver
 *          vweaver1@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *	This is an example component, it demos the component interface
 *  and implements three example counters.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/** This driver supports two counters */
#define EXAMPLE_MAX_COUNTERS 3

papi_vector_t _example_vector;

/** Structure that stores private information for each event */
typedef struct example_register
{
	unsigned int selector;
						 /**< Signifies which counter slot is being used */
						 /**< Indexed from 1 as 0 has a special meaning  */
} example_register_t;

/** This structure is used to build the table of events */
typedef struct example_native_event_entry
{
	example_register_t resources;		  /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];	   /**< Name of the counter         */
	char description[PAPI_MAX_STR_LEN];/**< Description of the counter  */
	int writable;					   /**< Whether counter is writable */
	/* any other counter parameters go here */
} example_native_event_entry_t;

/** This structure is used when doing register allocation 
    it possibly is not necessary when there are no 
    register constraints */
typedef struct example_reg_alloc
{
	example_register_t ra_bits;
} example_reg_alloc_t;

/** Holds control flags, usually out-of band configuration of the hardware */
typedef struct example_control_state
{
	long_long counter[EXAMPLE_MAX_COUNTERS];	/**< Copy of counts, used for caching */
	long_long lastupdate;			   /**< Last update time, used for caching */
} example_control_state_t;

/** Holds per-thread information */
typedef struct example_context
{
	example_control_state_t state;
} example_context_t;

/** This table contains the native events */
static example_native_event_entry_t *example_native_table;
/** number of events in the table*/
static int NUM_EVENTS = 1;


/************************************************************************/
/* Below is the actual "hardware implementation" of our example counters */
/************************************************************************/

#define EXAMPLE_ZERO_REG     0
#define EXAMPLE_CONSTANT_REG 1
#define EXAMPLE_AUTOINC_REG  2

static long_long example_autoinc_value = 0;

/** Code that resets the hardware.  */
static void
example_hardware_reset(  )
{

	example_autoinc_value = 0;

}

/** Code that reads event values.
    You might replace this with code that accesses
    hardware or reads values from the operatings system. */
static long_long
example_hardware_read( int which_one )
{

	long_long old_value;

	switch ( which_one ) {
	case EXAMPLE_ZERO_REG:
		return 0;
	case EXAMPLE_CONSTANT_REG:
		return 42;
	case EXAMPLE_AUTOINC_REG:
		old_value = example_autoinc_value;
		example_autoinc_value++;
		return old_value;
	default:
		perror( "Invalid counter read" );
		return -1;
	}

	return 0;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
example_init( hwd_context_t * ctx )
{
	SUBDBG( "example_init %p...", ctx );

	/* FIXME: do we need to make this thread safe? */

	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
example_init_substrate(  )
{

	SUBDBG( "example_init_substrate..." );

	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */
	NUM_EVENTS = 3;

	/* Make sure we don't allocate too many counters.                 */
	/* This could be avoided if we dynamically allocate counter space */
	/*   when needed.                                                 */
	if ( NUM_EVENTS > EXAMPLE_MAX_COUNTERS ) {
		perror( "Too many counters allocated" );
		return EXIT_FAILURE;
	}

	/* Allocate memory for the our event table */
	example_native_table =
		( example_native_event_entry_t * )
		malloc( sizeof ( example_native_event_entry_t ) * NUM_EVENTS );
	if ( example_native_table == NULL ) {
		perror( "malloc():Could not get memory for events table" );
		return EXIT_FAILURE;
	}

	/* fill in the event table parameters */
	strcpy( example_native_table[0].name, "EXAMPLE_ZERO" );
	strcpy( example_native_table[0].description,
			"This is a example counter, that always returns 0" );
	example_native_table[0].writable = 0;

	strcpy( example_native_table[1].name, "EXAMPLE_CONSTANT" );
	strcpy( example_native_table[1].description,
			"This is a example counter, that always returns a constant value of 42" );
	example_native_table[1].writable = 0;

	strcpy( example_native_table[2].name, "EXAMPLE_AUTOINC" );
	strcpy( example_native_table[2].description,
			"This is a example counter, that reports an auto-incrementing value" );
	example_native_table[2].writable = 1;

	/* The selector has to be !=0 . Starts with 1 */
	example_native_table[0].resources.selector = 1;
	example_native_table[1].resources.selector = 2;
	example_native_table[2].resources.selector = 3;

	_example_vector.cmp_info.num_native_events = NUM_EVENTS;
	return PAPI_OK;
}


/** Setup the counter control structure */
int
example_init_control_state( hwd_control_state_t * ctrl )
{
	SUBDBG( "example_init_control_state..." );

	/* set the hardware to initial conditions */
	example_hardware_reset(  );

	/* set the counters last-accessed time */
	( ( example_control_state_t * ) ctrl )->lastupdate = PAPI_get_real_usec(  );

	return PAPI_OK;
}


/** Enumerate Native Events 
   @param EventCode is the event of interest
   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
*/
int
example_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	switch ( modifier ) {

		/* return EventCode of first event */
	case PAPI_ENUM_FIRST:
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
		return PAPI_OK;
		break;

		/* return EventCode of passed-in Event */
	case PAPI_ENUM_EVENTS:{
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( index < NUM_EVENTS - 1 ) {
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		} else {
			return PAPI_ENOEVNT;
		}
		break;
	}
	default:
		return PAPI_EINVAL;
	}

	return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 @param EventCode is the native event code
 @param name is a pointer for the name to be copied to
 @param len is the size of the string
 */
int
example_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, example_native_table[index].name, len );

	return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 @param EventCode is the native event code
 @param name is a pointer for the description to be copied to
 @param len is the size of the string
 */
int
example_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, example_native_table[index].description, len );

	return PAPI_OK;
}

/** This takes an event and returns the bits that would be written
    out to the hardware device (this is very much tied to CPU-type support */
int
example_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	SUBDBG( "Want native bits for event %d", index );

	return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int
example_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native,
							  int count, hwd_context_t * ctx )
{
	int i, index;

	SUBDBG( "example_update_control_state %p %p...", ptr, ctx );

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position =
			example_native_table[index].resources.selector - 1;
		SUBDBG
			( "\nnative[%i].ni_position = example_native_table[%i].resources.selector-1 = %i;",
			  i, index, native[i].ni_position );
	}

	return PAPI_OK;
}

/** Triggered by PAPI_start() */
int
example_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	SUBDBG( "example_start %p %p...", ctx, ctrl );

	/* anything that would need to be set at counter start time */

	return PAPI_OK;
}


/** Triggered by PAPI_stop() */
int
example_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	SUBDBG( "example_stop %p %p...", ctx, ctrl );

	/* anything that would need to be done at counter stop time */

	return PAPI_OK;
}


/** Triggered by PAPI_read() */
int
example_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
			  long_long ** events, int flags )
{
	SUBDBG( "example_read... %p %d", ctx, flags );

	( ( example_control_state_t * ) ctrl )->counter[0] =
		example_hardware_read( EXAMPLE_ZERO_REG );
	( ( example_control_state_t * ) ctrl )->counter[1] =
		example_hardware_read( EXAMPLE_CONSTANT_REG );
	( ( example_control_state_t * ) ctrl )->counter[2] =
		example_hardware_read( EXAMPLE_AUTOINC_REG );

	*events = ( ( example_control_state_t * ) ctrl )->counter;	// serve cached data

	return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
example_write( hwd_context_t * ctx, hwd_control_state_t * ctrl,
			   long_long events[] )
{
	SUBDBG( "example_write... %p %p", ctx, ctrl );

	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */

	return PAPI_OK;
}


/** Triggered by PAPI_reset */
int
example_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	SUBDBG( "example_reset ctx=%p ctrl=%p...", ctx, ctrl );

	/* Reset the hardware */
	example_hardware_reset(  );
	/* Set the counters last-accessed time */
	( ( example_control_state_t * ) ctrl )->lastupdate = PAPI_get_real_usec(  );

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
example_shutdown( hwd_context_t * ctx )
{
	SUBDBG( "example_shutdown... %p", ctx );

	/* Last chance to clean up */

	return PAPI_OK;
}


int
example_cleanup_eventset( hwd_control_state_t * ctrl )
{
	( void ) ctrl;
	return ( PAPI_OK );
}


/** This function sets various options in the substrate
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
int
example_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	SUBDBG( "example_ctl..." );

	/* FIXME.  This should maybe set up more state, such as which counters are active and */
	/*         counter mappings. */

	return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
    In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
    By default return PAPI_EINVAL if none of those are specified
    and PAPI_OK with success
    PAPI_DOM_USER is only user context is counted
    PAPI_DOM_KERNEL is only the Kernel/OS context is counted
    PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
    PAPI_DOM_ALL   is all of the domains
 */
int
example_set_domain( hwd_control_state_t * cntrl, int domain )
{
	int found = 0;
	SUBDBG( "example_set_domain..." );

	if ( PAPI_DOM_USER & domain ) {
		SUBDBG( " PAPI_DOM_USER " );
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		SUBDBG( " PAPI_DOM_KERNEL " );
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		SUBDBG( " PAPI_DOM_OTHER " );
		found = 1;
	}
	if ( PAPI_DOM_ALL & domain ) {
		SUBDBG( " PAPI_DOM_ALL " );
		found = 1;
	}
	if ( !found )
		return ( PAPI_EINVAL );

	return PAPI_OK;
}


/** Vector that points to entry points for our component */
papi_vector_t _example_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "$Id$",
				 .version = "$Revision$",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = EXAMPLE_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( example_context_t ),
			 .control_state = sizeof ( example_control_state_t ),
			 .reg_value = sizeof ( example_register_t ),
			 .reg_alloc = sizeof ( example_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = example_init,
	.init_substrate = example_init_substrate,
	.init_control_state = example_init_control_state,
	.start = example_start,
	.stop = example_stop,
	.read = example_read,
	.write = example_write,
	.shutdown = example_shutdown,
	.cleanup_eventset = example_cleanup_eventset,
	.ctl = example_ctl,
	.bpt_map_set = NULL,
	.bpt_map_avail = NULL,
	.bpt_map_exclusive = NULL,
	.bpt_map_shared = NULL,
	.bpt_map_preempt = NULL,
	.bpt_map_update = NULL,

	.update_control_state = example_update_control_state,
	.set_domain = example_set_domain,
	.reset = example_reset,

	.ntv_enum_events = example_ntv_enum_events,
	.ntv_code_to_name = example_ntv_code_to_name,
	.ntv_code_to_descr = example_ntv_code_to_descr,
	.ntv_code_to_bits = example_ntv_code_to_bits,
	.ntv_bits_to_info = NULL,
};

