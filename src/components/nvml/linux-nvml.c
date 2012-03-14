/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    example.c
 * @author  Kiran Kumar Kasichayanula
 *          kkasicha@utk.edu 
 * @author  James Ralph
 *          ralph@eecs.utk.edu
 * @ingroup papi_components
 *
 * @brief
 *	This is an NVML component, it demos the component interface
 *  and implements two counters nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature
 *  from Nvidia Management Library. Please refer to NVML documentation for details
 * about nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <pthread.h>
#include <string.h>
#include <nvml.h>
/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"



/* Declare our vector in advance */
papi_vector_t _nvml_vector;

/** Structure that stores private information for each event */
typedef struct nvml_register
{
   unsigned int selector;
		           /**< Signifies which counter slot is being used */
			   /**< Indexed from 1 as 0 has a special meaning  */
} nvml_register_t;

/** This structure is used to build the table of events  */
typedef struct nvml_native_event_entry
{
	nvml_register_t resources;	    /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];	    /**< Name of the counter         */
	char description[PAPI_MAX_STR_LEN]; /**< Description of the counter  */
	int writable;			    /**< Whether counter is writable */
	/* any other counter parameters go here */
} nvml_native_event_entry_t;

/** This structure is used when doing register allocation 
    it possibly is not necessary when there are no 
    register constraints 
typedef struct nvml_reg_alloc
{
	nvml_register_t ra_bits;
} nvml_reg_alloc_t;*/

/** Holds control flags.  Usually there's one of these per event-set.
 *    Usually this is out-of band configuration of the hardware 
 */
typedef struct nvml_control_state
{
  int num_events;
  int domain;
  int multiplexed;
  int overflow;
  int inherit;
  long long autoinc_value;
  int which_counter[2];
//  int counter_bits[2]; 
  long long counter[2];   /**< Copy of counts, holds results when stopped */

} nvml_control_state_t;

/** Holds per-thread information */
typedef struct nvml_context
{
	nvml_control_state_t state;
} nvml_context_t;

/** This table contains the native events */
static nvml_native_event_entry_t *nvml_native_table;

/** number of events in the table*/
static int num_events = 0;



/** Code that resets the hardware. Since NVML library reads from a sensor we donot have to reset anything*/
static void
nvml_hardware_reset(  )
{

}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
static long long
nvml_hardware_read( int which_one)
//, nvml_context_t *ctx)
{
	 int retval;
    nvmlTemperatureSensors_t temper =  NVML_TEMPERATURE_GPU;
    unsigned int p;
    nvmlDevice_t handle_device;
	char *pch_hardware, *pch_hardware1 ;
    unsigned int t_d;
	pch_hardware = strstr(nvml_native_table[which_one].name, "Power_Usage");
	if (pch_hardware != NULL)
		{
			 retval = nvmlDeviceGetHandleByIndex( 0, &handle_device );
     		 retval = nvmlDeviceGetPowerUsage( handle_device, &p );
        	if (retval == NVML_SUCCESS)
            return p;
		}
	pch_hardware1 = strstr(nvml_native_table[which_one].name, "Temperature");
    if (pch_hardware1 != NULL)
        {
             retval = nvmlDeviceGetHandleByIndex( 0, &handle_device );
             retval = nvmlDeviceGetTemperature(handle_device, temper, &t_d);
       		 if (retval == NVML_SUCCESS)
             return t_d;
        }
	return 0;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
_papi_nvml_init( hwd_context_t * ctx )
{
        (void) ctx;

	SUBDBG( "_papi_nvml_init %p...", ctx );

	return PAPI_OK;
}

static int
detectDevice( void )
{
    int num_events = 0;
    int retval;
    nvmlTemperatureSensors_t temper =  NVML_TEMPERATURE_GPU;
    unsigned int  p, t_d;
    nvmlDevice_t handle_device;

     /* Initiate NVML library */
        retval = nvmlInit();
    if (retval != NVML_SUCCESS)
    {
        printf(" Initialization of NVML library failed\n");
        return ( PAPI_ENOSUPP );
    }
        retval = nvmlDeviceGetHandleByIndex( 0, &handle_device );
	 if ( retval != NVML_SUCCESS  )
        {
            printf("There is no handle for particular device");
            return ( PAPI_ENOSUPP );
        }
        retval = nvmlDeviceGetPowerUsage( handle_device, &p );
        if (retval == NVML_SUCCESS)
        {
            num_events++;
        }

        retval = nvmlDeviceGetTemperature(handle_device, temper, &t_d);
        if (retval == NVML_SUCCESS)
        {
            num_events++;
        }
    return num_events;
}
static int
createNativeEvents( void )
{
//	char nameDevice[2][100];
    char nameDevice[100];
     char temp[300];
    char * pch;
    nvmlDevice_t handle_device;
    int len =100, id = 0;
    unsigned int p;
    nvmlTemperatureSensors_t temper =  NVML_TEMPERATURE_GPU;
    unsigned int t_d, retval;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000;
    cmp_id_t component;
    /* Initiate NVML library */
        retval = nvmlInit();
    if (retval != NVML_SUCCESS)
    {
        printf(" Initialization of NVML library failed\n");
        return ( PAPI_ENOSUPP );
    }
    // component name and description 
     strcpy( component.name, "NVML" );
    strcpy( component.descr, "NVML provides the API for monitoring power and temperature of GPU" );

        retval = nvmlDeviceGetHandleByIndex( 0, &handle_device );
        if ( retval != NVML_SUCCESS  )
        {
                printf("There is no handle for particular device");
            return ( PAPI_ENOSUPP );
        }
    	retval = nvmlDeviceGetName(handle_device, nameDevice, len);
        if (retval != NVML_SUCCESS)
        printf("fetching Name of the device failed");
        memset( temp, 0x0, 300);
		 pch = strtok (nameDevice," ");
        while (pch != NULL)
     {
    strcat(temp, pch);
     pch = strtok (NULL, " ");
    }
	strcpy(nameDevice,temp);
     retval = nvmlDeviceGetPowerUsage( handle_device, &p );
	 if (retval == NVML_SUCCESS)
   {
     sprintf(nvml_native_table[id].name, "%s.%s.Device%d.Device_Get_Power_Usage",component.name, nameDevice,0);
    nvml_native_table[id].resources.selector = id + 1;
        id++;
    }

    retval = nvmlDeviceGetTemperature(handle_device, temper, &t_d);
    if (retval == NVML_SUCCESS)
    {
        sprintf(nvml_native_table[id].name, "%s.%s.Device%d.Device_Get_Temperature",component.name, temp,0);
        nvml_native_table[id].resources.selector = id + 1;
    }
    return id;
}
/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_papi_nvml_init_substrate( int cidx )
{
     //Create dynamic event table
    num_events = detectDevice();
	nvml_native_table = ( nvml_native_event_entry_t * )
        malloc( sizeof ( nvml_native_event_entry_t ) * num_events );
    if ( nvml_native_table == NULL ) {
        perror( "malloc(): Failed to allocate memory to events table" );
    }
    createNativeEvents(  );
	SUBDBG( "_papi_nvml_init_substrate..." );

	/* Export the total number of events available */
	_nvml_vector.cmp_info.num_native_events = num_events;

	/* Export the component id */
	_nvml_vector.cmp_info.CmpIdx = cidx;
	return PAPI_OK;
}


/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

int
_papi_nvml_init_control_state( hwd_control_state_t * ctl )
{
   SUBDBG( "nvml_init_control_state... %p\n", ctl );
   nvml_control_state_t *nvml_ctl = ( nvml_control_state_t * ) ctl;
   memset( nvml_ctl, 0, sizeof ( nvml_control_state_t ) );

   return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
int
_papi_nvml_update_control_state( hwd_control_state_t *ctl, 
				    NativeInfo_t *native,
				    int count, 
				    hwd_context_t *ctx )
{
   int i, index;

   nvml_control_state_t *nvml_ctl = ( nvml_control_state_t * ) ctl;   
   (void) ctx;

   SUBDBG( "_papi_nvml_update_control_state %p %p...", ctl, ctx );

   /* if no events, return */
   if (count==0) return PAPI_OK;

   for( i = 0; i < count; i++ ) {
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
      nvml_ctl->which_counter[i]=index;
      /* We have no constraints on event position, so any event */
      /* can be in any slot.                                    */
      native[i].ni_position = i;
   }
	nvml_ctl->num_events=count;
   return PAPI_OK;
}
/** Triggered by PAPI_start() */
int
_papi_nvml_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        (void) ctx;
        (void) ctl;
		
	SUBDBG( "nvml_start %p %p...", ctx, ctl );
	/* anything that would need to be set at counter start time */

	/* reset */
	/* start the counting */

	return PAPI_OK;
}

/** Triggered by PAPI_stop() */
int
_papi_nvml_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
		int i;
        (void) ctx;
        (void) ctl;
	SUBDBG( "nvml_stop %p %p...", ctx, ctl );
/*	for ( i=0; i < num_events; i++ ) {
//    nvml_ctl->counter[i] = nvml_hardware_read(i);
	((nvml_control_state_t * )ctl)->counter[i] = nvml_hardware_read(i);
        if (((nvml_control_state_t * )ctl)->counter[i] < 0) {
            return EXIT_FAILURE;
        }

    }*/
//	nvml_context_t *nvml_ctx = (nvml_context_t *) ctx;
	nvml_control_state_t* nvml_ctl = ( nvml_control_state_t*) ctl;
	for (i=0;i<nvml_ctl->num_events;i++) {
      nvml_ctl->counter[i] =
        nvml_hardware_read( nvml_ctl->which_counter[i]);
		//, nvml_ctx );
   }
	return PAPI_OK;
}


/** Triggered by PAPI_read() */
int
_papi_nvml_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
			  long long **events, int flags )
{

   (void) ctx;
   (void) flags;
	int i;
   nvml_control_state_t* nvml_ctl = ( nvml_control_state_t*) ctl;   
//	nvml_context_t *nvml_ctx = (nvml_context_t *) ctx;
   SUBDBG( "nvml_read... %p %d", ctx, flags );
/*	for ( i=0; i < num_events; i++ ) {
//    nvml_ctl->counter[i] = nvml_hardware_read(i);
	 ((nvml_control_state_t * )ctl)->counter[i] = nvml_hardware_read(i);
        if (((nvml_control_state_t * )ctl)->counter[i] < 0) {
            return EXIT_FAILURE;
        }*/
for(i=0;i<nvml_ctl->num_events;i++) {
      nvml_ctl->counter[i] =
        nvml_hardware_read( nvml_ctl->which_counter[i]);
//, nvml_ctx );
   }




 //   }
   /* return pointer to the values we read */
   *events = nvml_ctl->counter;	
   return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
_papi_nvml_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
			   long long *events )
{

        (void) ctx;
	(void) ctl;
	(void) events;

	SUBDBG( "nvml_write... %p %p", ctx, ctl );

	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */

	return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
int
_papi_nvml_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
        (void) ctx;
	(void) ctl;

	SUBDBG( "nvml_reset ctx=%p ctrl=%p...", ctx, ctl );

	/* Reset the hardware */
	nvml_hardware_reset(  );

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
_papi_nvml_shutdown_substrate()
{

	SUBDBG( "nvml_shutdown_substrate..." );

	papi_free(nvml_native_table);

	return PAPI_OK;
}

/** Called at thread shutdown */
int
_papi_nvml_shutdown( hwd_context_t *ctx )
{

        (void) ctx;

	SUBDBG( "nvml_shutdown... %p", ctx );

	/* Last chance to clean up thread */

	return PAPI_OK;
}



/** This function sets various options in the substrate
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
int
_papi_nvml_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
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
_papi_nvml_set_domain( hwd_control_state_t * cntrl, int domain )
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
_papi_nvml_ntv_enum_events( unsigned int *EventCode, int modifier )
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
_papi_nvml_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
  int index;

  index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  /* Make sure we are in range */
  if (index >= num_events) return PAPI_ENOEVNT;

  strncpy( name, nvml_native_table[index].name, len );

  return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
int
_papi_nvml_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
  int index;
  index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

  if (index >= num_events) return PAPI_ENOEVNT;

  strncpy( descr, nvml_native_table[index].description, len );

  return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _nvml_vector = {
	.cmp_info = {
		/* default component information */
		/* (unspecified values are initialized to 0) */
                /* we explicitly set them to zero in this example */
                /* to show what settings are available            */

		.name = "$Id: example.c,v 1.15 2011/11/07 18:55:11 vweaver1 Exp $",
		.version = "$Revision: 1.15 $",
		.support_version = "n/a",
		.kernel_version = "n/a",
		.CmpIdx = 0,            /* set by init_substrate */
		.num_cntrs = 2, 
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
		.context = sizeof ( nvml_context_t ),
		.control_state = sizeof ( nvml_control_state_t ),
		.reg_value = sizeof ( nvml_register_t ),
//		.reg_alloc = sizeof ( nvml_reg_alloc_t ),
	},

	/* function pointers */

	/* Used for general PAPI interactions */
	.start =                _papi_nvml_start,
	.stop =                 _papi_nvml_stop,
	.read =                 _papi_nvml_read,
	.reset =                _papi_nvml_reset,	
	.write =                _papi_nvml_write,
	.init_substrate =       _papi_nvml_init_substrate,	
	.init =                 _papi_nvml_init,
	.init_control_state =   _papi_nvml_init_control_state,
	.update_control_state = _papi_nvml_update_control_state,	
	.ctl =                  _papi_nvml_ctl,	
	.shutdown =             _papi_nvml_shutdown,
	.shutdown_substrate =   _papi_nvml_shutdown_substrate,
	.set_domain =           _papi_nvml_set_domain,
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
	.ntv_enum_events =   _papi_nvml_ntv_enum_events,
	.ntv_name_to_code  = NULL,
	.ntv_code_to_name =  _papi_nvml_ntv_code_to_name,
	.ntv_code_to_descr = _papi_nvml_ntv_code_to_descr,

	/* These are only used by _papi_hwi_get_native_event_info() */
	/* Which currently only uses the info for printing native   */
	/* event info, not for any sort of internal use.            */
//	.ntv_code_to_bits =  NULL,
//	.ntv_bits_to_info =  NULL,


	/* Old and should be removed */
//	.add_prog_event =       NULL,


};

