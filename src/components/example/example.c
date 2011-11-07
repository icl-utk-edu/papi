/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    example.c
 * @author  Joachim Protze
 *          joachim.protze@zih.tu-dresden.de
 * @author  Vince Weaver
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

/** This driver supports three counters counting at once      */
/*  This is artificially low to allow testing of multiplexing */
#define EXAMPLE_MAX_SIMULTANEOUS_COUNTERS 3

/* Declare our vector in advance */
papi_vector_t _example_vector;

/** Structure that stores private information for each event */
typedef struct example_register
{
   unsigned int selector;
		           /**< Signifies which counter slot is being used */
			   /**< Indexed from 1 as 0 has a special meaning  */
} example_register_t;

/** This structure is used to build the table of events  */
/*   The contents of this structure will vary based on   */
/*   your component, however having name and description */
/*   fields are probably useful.                         */
typedef struct example_native_event_entry
{
	example_register_t resources;	    /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];	    /**< Name of the counter         */
	char description[PAPI_MAX_STR_LEN]; /**< Description of the counter  */
	int writable;			    /**< Whether counter is writable */
	/* any other counter parameters go here */
} example_native_event_entry_t;

/** This structure is used when doing register allocation 
    it possibly is not necessary when there are no 
    register constraints */
typedef struct example_reg_alloc
{
	example_register_t ra_bits;
} example_reg_alloc_t;

/** Holds control flags.  Usually there's one of these per event-set.
 *    Usually this is out-of band configuration of the hardware 
 */
typedef struct example_control_state
{
  int num_events;
  int domain;
  int multiplexed;
  int overflow;
  int inherit;
  long long autoinc_value;
  int counter_bits[EXAMPLE_MAX_SIMULTANEOUS_COUNTERS]; 
  long long counter[EXAMPLE_MAX_SIMULTANEOUS_COUNTERS];   /**< Copy of counts, holds results when stopped */

} example_control_state_t;

/** Holds per-thread information */
typedef struct example_context
{
	example_control_state_t state;
} example_context_t;

/** This table contains the native events */
static example_native_event_entry_t *example_native_table;

/** number of events in the table*/
static int num_events = 0;


/*************************************************************************/
/* Below is the actual "hardware implementation" of our example counters */
/*************************************************************************/

#define EXAMPLE_ZERO_REG             0
#define EXAMPLE_CONSTANT_REG         1
#define EXAMPLE_AUTOINC_REG          2
#define EXAMPLE_GLOBAL_AUTOINC_REG   3

#define EXAMPLE_TOTAL_EVENTS         4

static long long example_global_autoinc_value = 0;

/** Code that resets the hardware.  */
static void
example_hardware_reset(  )
{

	example_global_autoinc_value = 0;

}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
static long long
example_hardware_read( int which_one )
{

	long long old_value;

	switch ( which_one ) {
	case EXAMPLE_ZERO_REG:
		return 0;
	case EXAMPLE_CONSTANT_REG:
		return 42;
	case EXAMPLE_AUTOINC_REG:
		old_value = example_global_autoinc_value;
		example_global_autoinc_value++;
		return old_value;
	case EXAMPLE_GLOBAL_AUTOINC_REG:
		old_value = example_global_autoinc_value;
		example_global_autoinc_value++;
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
_papi_example_init( hwd_context_t * ctx )
{
        (void) ctx;

	SUBDBG( "_papi_example_init %p...", ctx );

	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_papi_example_init_substrate( int cidx )
{

	SUBDBG( "_papi_example_init_substrate..." );

	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */
	num_events = EXAMPLE_TOTAL_EVENTS;

	/* Allocate memory for the our native event table */
	example_native_table =
		( example_native_event_entry_t * )
		papi_calloc( sizeof(example_native_event_entry_t),num_events);
	if ( example_native_table == NULL ) {
		PAPIERROR( "malloc():Could not get memory for events table" );
		return PAPI_ENOMEM;
	}

	/* fill in the event table parameters */
	/* for complicated components this will be done dynamically */
	/* or by using an external library                          */

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

	strcpy( example_native_table[3].name, "EXAMPLE_GLOBAL_AUTOINC" );
	strcpy( example_native_table[3].description,
			"This is a example counter, that reports a global auto-incrementing value" );
	example_native_table[3].writable = 1;

	/* Export the total number of events available */
	_example_vector.cmp_info.num_native_events = num_events;

	/* Export the component id */
	_example_vector.cmp_info.CmpIdx = cidx;

	

	return PAPI_OK;
}


/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

int
_papi_example_init_control_state( hwd_control_state_t * ctl )
{
   SUBDBG( "example_init_control_state... %p\n", ctl );

   example_control_state_t *example_ctl = ( example_control_state_t * ) ctl;
   memset( example_ctl, 0, sizeof ( example_control_state_t ) );

   return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
int
_papi_example_update_control_state( hwd_control_state_t *ctl, 
				    NativeInfo_t *native,
				    int count, 
				    hwd_context_t *ctx )
{
   int i, index;

   example_control_state_t *example_ctl = ( example_control_state_t * ) ctl;   
   (void) ctx;

   SUBDBG( "_papi_example_update_control_state %p %p...", ctl, ctx );

   /* if no events, return */
   if (count==0) return PAPI_OK;

   for( i = 0; i < count; i++ ) {
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
      example_ctl->counter_bits[i]=index;

      /* We have no constraints on event position, so any event */
      /* can be in any slot.                                    */
      native[i].ni_position = i;
   }

   example_ctl->num_events=count;

   return PAPI_OK;
}

/** Triggered by PAPI_start() */
int
_papi_example_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        (void) ctx;
        (void) ctl;

	SUBDBG( "example_start %p %p...", ctx, ctl );

	/* anything that would need to be set at counter start time */

	/* reset */
	/* start the counting */

	return PAPI_OK;
}


/** Triggered by PAPI_stop() */
int
_papi_example_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        (void) ctx;
        (void) ctl;

	SUBDBG( "example_stop %p %p...", ctx, ctl );

	/* anything that would need to be done at counter stop time */

	

	return PAPI_OK;
}


/** Triggered by PAPI_read() */
int
_papi_example_read( hwd_context_t * ctx, hwd_control_state_t * ctl,
			  long long ** events, int flags )
{

   (void) ctx;
   (void) flags;

   example_control_state_t *example_ctl = ( example_control_state_t * ) ctl;   

   SUBDBG( "example_read... %p %d", ctx, flags );

   int i;

   /* Read counters into expected slot */
   for(i=0;i<example_ctl->num_events;i++) {
      example_ctl->counter[i] =
		example_hardware_read( example_ctl->counter_bits[i] );
   }

   /* return pointer to the values we read */
   *events = example_ctl->counter;	

   return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
_papi_example_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
			   long long *events )
{

        (void) ctx;
	(void) ctl;
	(void) events;

	SUBDBG( "example_write... %p %p", ctx, ctl );

	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */

	return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
int
_papi_example_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
        (void) ctx;
	(void) ctl;

	SUBDBG( "example_reset ctx=%p ctrl=%p...", ctx, ctl );

	/* Reset the hardware */
	example_hardware_reset(  );

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
_papi_example_shutdown_substrate()
{

	SUBDBG( "example_shutdown_substrate..." );

	papi_free(example_native_table);

	return PAPI_OK;
}

/** Called at thread shutdown */
int
_papi_example_shutdown( hwd_context_t *ctx )
{

        (void) ctx;

	SUBDBG( "example_shutdown... %p", ctx );

	/* Last chance to clean up thread */

	return PAPI_OK;
}



/** This function sets various options in the substrate
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
int
_papi_example_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{

        (void) ctx;
	(void) code;
	(void) option;

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
_papi_example_set_domain( hwd_control_state_t * cntrl, int domain )
{
        (void) cntrl;

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


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
int
_papi_example_ntv_enum_events( unsigned int *EventCode, int modifier )
{
  int cidx,index;

  /* Get our component index number, this can change depending */
  /* on how PAPI was configured.                               */

  cidx = PAPI_COMPONENT_INDEX( *EventCode );

  switch ( modifier ) {

		/* return EventCode of first event */
	case PAPI_ENUM_FIRST:
	   /* return the first event that we support */

	   *EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
	   return PAPI_OK;

		/* return EventCode of next available event */
	case PAPI_ENUM_EVENTS:
	   index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	   /* Make sure we are in range */
	   if ( index < num_events - 1 ) {

	      /* This assumes a non-sparse mapping of the events */
	      *EventCode = *EventCode + 1;
	      return PAPI_OK;
	   } else {
	      return PAPI_ENOEVNT;
	   }
	   break;
	
	default:
	   return PAPI_EINVAL;
  }

  return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
int
_papi_example_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
  int index;

  index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  /* Make sure we are in range */
  if (index >= num_events) return PAPI_ENOEVNT;

  strncpy( name, example_native_table[index].name, len );

  return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
int
_papi_example_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
  int index;
  index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  if (index >= num_events) return PAPI_ENOEVNT;

  strncpy( descr, example_native_table[index].description, len );

  return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _example_vector = {
	.cmp_info = {
		/* default component information */
		/* (unspecified values are initialized to 0) */
                /* we explicitly set them to zero in this example */
                /* to show what settings are available            */

		.name = "$Id$",
		.version = "$Revision$",
		.support_version = "n/a",
		.kernel_version = "n/a",
		.CmpIdx = 0,            /* set by init_substrate */
		.num_cntrs = EXAMPLE_MAX_SIMULTANEOUS_COUNTERS, 
		.num_mpx_cntrs = PAPI_MPX_DEF_DEG,
		.num_preset_events = 0,
		.num_native_events = 0, /* set by init_substrate */
		.default_domain = PAPI_DOM_USER,
		.available_domains = PAPI_DOM_USER,
		.default_granularity = PAPI_GRN_THR,
		.available_granularities = PAPI_GRN_THR,
		.itimer_sig = 0,       /* set by init_substrate */
		.itimer_num = 0,       /* set by init_substrate */
		.itimer_ns = 0,        /* set by init_substrate */
		.itimer_res_ns = 0,    /* set by init_substrate */
		.hardware_intr_sig = PAPI_INT_SIGNAL,
		.clock_ticks = 0,      /* set by init_substrate */
		.opcode_match_width = 0, /* set by init_substrate */ 
		.os_version = 0,       /* set by init_substrate */ 


		/* component specific cmp_info initializations */
		.hardware_intr = 0,
		.precise_intr = 0,
		.posix1b_timers = 0,
		.kernel_profile = 0,
		.kernel_multiplex = 0,
		.data_address_range = 0,
		.instr_address_range = 0,
		.fast_counter_read = 0,
		.fast_real_timer = 0,
		.fast_virtual_timer = 0,
		.attach = 0,
		.attach_must_ptrace = 0,
		.edge_detect = 0,
		.invert = 0,
		.profile_ear = 0,
		.cntr_groups = 0,
		.cntr_umasks = 0,
		.cntr_IEAR_events = 0,
		.cntr_DEAR_events = 0,
		.cntr_OPCM_events = 0,
		.cpu = 0,
		.inherit = 0,
	},

	/* sizes of framework-opaque component-private structures */
	.size = {
		.context = sizeof ( example_context_t ),
		.control_state = sizeof ( example_control_state_t ),
		.reg_value = sizeof ( example_register_t ),
		.reg_alloc = sizeof ( example_reg_alloc_t ),
	},

	/* function pointers */

	/* Used for general PAPI interactions */
	.start =                _papi_example_start,
	.stop =                 _papi_example_stop,
	.read =                 _papi_example_read,
	.reset =                _papi_example_reset,	
	.write =                _papi_example_write,
	.init_substrate =       _papi_example_init_substrate,	
	.init =                 _papi_example_init,
	.init_control_state =   _papi_example_init_control_state,
	.update_control_state = _papi_example_update_control_state,	
	.ctl =                  _papi_example_ctl,	
	.shutdown =             _papi_example_shutdown,
	.shutdown_substrate =   _papi_example_shutdown_substrate,
	.set_domain =           _papi_example_set_domain,
	.cleanup_eventset =     NULL,
	/* called in add_native_events() */
	.allocate_registers =   NULL,

	/* Used for overflow/profiling */
	.dispatch_timer =       NULL,
	.get_overflow_address = NULL,
	.stop_profiling =       NULL,
	.set_overflow =         NULL,
	.set_profile =          NULL,

	/* OS related functions */
	.get_real_cycles =      NULL,
	.get_real_usec =        NULL,
	.get_virt_cycles =      NULL,
	.get_virt_usec =        NULL,
	.update_shlib_info =    NULL,
	.get_system_info =      NULL,
	.get_memory_info =      NULL,
	.get_dmem_info =        NULL,

	/* bipartite map counter allocation? */
	.bpt_map_avail =        NULL,
	.bpt_map_set =          NULL,
	.bpt_map_exclusive =    NULL,
	.bpt_map_shared =       NULL,
	.bpt_map_preempt =      NULL,
	.bpt_map_update =       NULL,

	/* ??? */
	.user =                 NULL,

	/* Name Mapping Functions */
	.ntv_enum_events =   _papi_example_ntv_enum_events,
	.ntv_name_to_code  = NULL,
	.ntv_code_to_name =  _papi_example_ntv_code_to_name,
	.ntv_code_to_descr = _papi_example_ntv_code_to_descr,

	/* These are only used by _papi_hwi_get_native_event_info() */
	/* Which currently only uses the info for printing native   */
	/* event info, not for any sort of internal use.            */
	.ntv_code_to_bits =  NULL,
	.ntv_bits_to_info =  NULL,


	/* Old and should be removed */
	.add_prog_event =       NULL,


};

