/**
 * @file    linux-io.c
 * @author  Kevin A. Huck
 *          khuck@uoregon.edu
 *
 * @ingroup papi_components
 *
 * @brief io component
 *  This file contains the source code for a component that enables
 *  PAPI-C to access I/O statistics through the /proc file system.
 *  This component will dynamically create a native events table for
 *  all the 7 counters in /proc/self/io.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"    /* defines papi_malloc(), etc. */

/** This driver supports three counters counting at once      */
/*  This is artificially low to allow testing of multiplexing */
#define EXAMPLE_MAX_SIMULTANEOUS_COUNTERS 3
#define EXAMPLE_MAX_MULTIPLEX_COUNTERS 4

/* Declare our vector in advance */
/* This allows us to modify the component info */
papi_vector_t _io_vector;

/* Define our structures */
#include "linux-io.h"

/** This table contains the native events */
static IO_native_event_entry_t *io_native_table;

/** number of events in the table*/
static int num_events = 0;

/*************************************************************************/
/* Below is the actual "hardware implementation" of our example counters */
/*************************************************************************/

/** Code that resets the hardware.  */
    static void
io_hardware_reset( IO_control_state_t *ctl )
{
    /* reset counter values */
    memset(ctl->counter, 0LL, sizeof(long long));

}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operating system. */
    void
io_hardware_read( IO_control_state_t* ctl )
{
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[256] = {0};
    pFile = fopen ("/proc/self/io","r");
    if (pFile == NULL) {
        perror ("Error opening file");
        return;
    }
    /* Read each line */
    int counter_index = 0;
    while (fgets(line, 4096, pFile)) {
        char dummy[32] = {0};
        long long tmplong = 0LL;
        int nf = sscanf( line, "%s %lld\n", dummy, &tmplong);
        if (nf == EOF) {
            perror ("Error reading from file");
            return;
        }
        ctl->counter[counter_index++] = tmplong;
        printf ("%s = %ld\n", dummy, tmplong);
    }
    fclose(pFile);

    return;
}

/** Code that writes event values.                        */
    static int
io_hardware_write( IO_control_state_t *ctl )
{
    (void) ctl; // unused
    /* counters are not writable, do nothing */
    return PAPI_OK;
}

static int
detect_io(void) {

    return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
    static int
_io_init_component( int cidx )
{

    SUBDBG( "_io_init_component..." );


    /* First, detect that our hardware is available */
    if (detect_io()!=PAPI_OK) {
        return PAPI_ECMP;
    }

    /* we know in advance how many events we want                       */
    /* for actual hardware this might have to be determined dynamically */
    num_events = IO_MAX_COUNTERS;

    /* Allocate memory for the our native event table */
    io_native_table =
        ( IO_native_event_entry_t * )
        papi_calloc( num_events, sizeof(IO_native_event_entry_t) );
    if ( io_native_table == NULL ) {
        PAPIERROR( "malloc():Could not get memory for events table" );
        return PAPI_ENOMEM;
    }

    /* fill in the event table parameters */
    /* for complicated components this will be done dynamically */
    /* or by using an external library                          */

    strcpy( io_native_table[0].name, "rchar" );
    strcpy( io_native_table[0].description, "Characters read" );
    io_native_table[0].writable = 0;

    strcpy( io_native_table[1].name, "wchar" );
    strcpy( io_native_table[1].description, "Characters written" );
    io_native_table[1].writable = 0;

    strcpy( io_native_table[2].name, "syscr" );
    strcpy( io_native_table[2].description, "Characters read by system calls" );
    io_native_table[2].writable = 0;

    strcpy( io_native_table[3].name, "syscw" );
    strcpy( io_native_table[3].description, "Characters written by system calls" );
    io_native_table[3].writable = 0;

    strcpy( io_native_table[4].name, "read_bytes" );
    strcpy( io_native_table[4].description, "Binary bytes read" );
    io_native_table[4].writable = 0;

    strcpy( io_native_table[5].name, "write_bytes" );
    strcpy( io_native_table[5].description, "Binary bytes written" );
    io_native_table[5].writable = 0;

    strcpy( io_native_table[6].name, "cancelled_write_bytes" );
    strcpy( io_native_table[6].description, "Binary write bytes cancelled" );
    io_native_table[6].writable = 0;

    /* Export the total number of events available */
    _io_vector.cmp_info.num_native_events = num_events;

    /* Export the component id */
    _io_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}

/** This is called whenever a thread is initialized */
    static int
_io_init_thread( hwd_context_t *ctx )
{
    (void) ctx; // unused

    SUBDBG( "_io_init_thread %p...", ctx );

    return PAPI_OK;
}



/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

    static int
_io_init_control_state( hwd_control_state_t * ctl )
{
    SUBDBG( "io_init_control_state... %p\n", ctl );

    IO_control_state_t *io_ctl = ( IO_control_state_t * ) ctl;
    memset( io_ctl, 0, sizeof ( IO_control_state_t ) );

    return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
    static int
_io_update_control_state( hwd_control_state_t *ctl, 
        NativeInfo_t *native,
        int count, 
        hwd_context_t *ctx )
{

    (void) ctx;
    int i, index;

    IO_control_state_t *io_ctl = ( IO_control_state_t * ) ctl;   

    SUBDBG( "_io_update_control_state %p %p...", ctl, ctx );

    /* if no events, return */
    if (count==0) return PAPI_OK;

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event;

        /* Map counter #i to Measure Event "index" */
        io_ctl->which_counter[i]=index;

        /* We have no constraints on event position, so any event */
        /* can be in any slot.                                    */
        native[i].ni_position = i;
    }

    io_ctl->num_events=count;

    return PAPI_OK;
}

/** Triggered by PAPI_start() */
    static int
_io_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    IO_control_state_t *io_ctl = ( IO_control_state_t *) ctl;
    (void) ctx;

    SUBDBG( "io_start %p %p...", ctx, ctl );

    /* anything that would need to be set at counter start time */

    /* reset counters? */
    /* For hardware that cannot reset counters, store initial        */
    /*     counter state to the ctl and subtract it off at read time */
    io_hardware_reset( io_ctl );

    /* start the counting ?*/

    return PAPI_OK;
}


/** Triggered by PAPI_stop() */
    static int
_io_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    (void) ctx;
    (void) ctl;

    SUBDBG( "io_stop %p %p...", ctx, ctl );

    /* anything that would need to be done at counter stop time */



    return PAPI_OK;
}


/** Triggered by PAPI_read()     */
/*     flags field is never set? */
    static int
_io_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
        long long **events, int flags )
{
    (void) ctx; // unused
    (void) flags;

    IO_control_state_t *io_ctl = ( IO_control_state_t *) ctl;   

    SUBDBG( "io_read... %p %d", ctx, flags );

    /* Read counters into expected slot */
    io_hardware_read( io_ctl );

    /* return pointer to the values we read */
    *events = io_ctl->counter;

    return PAPI_OK;
}

/** Triggered by PAPI_wrte(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
    static int
_io_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
        long long *events )
{
    IO_control_state_t *io_ctl = ( IO_control_state_t *) ctl;   
    (void) ctx; // unused
    (void) events; // unused

    /* Counters are not writable, do nothing */
    io_hardware_write( io_ctl );
    return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
    static int
_io_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    IO_control_state_t *io_ctl = ( IO_control_state_t *) ctl;   
    (void) ctx; // unused

    SUBDBG( "io_reset ctx=%p ctrl=%p...", ctx, ctl );

    /* Reset the hardware */
    io_hardware_reset( io_ctl );

    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
    static int
_io_shutdown_component(void)
{

    SUBDBG( "io_shutdown_component..." );

    /* Free anything we allocated */

    papi_free(io_native_table);

    return PAPI_OK;
}

/** Called at thread shutdown */
    static int
_io_shutdown_thread( hwd_context_t *ctx )
{

    (void) ctx;

    SUBDBG( "io_shutdown_thread... %p", ctx );

    /* Last chance to clean up thread */

    return PAPI_OK;
}



/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, 
  PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
 */
    static int
_io_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

    (void) ctx;
    (void) code;
    (void) option;

    SUBDBG( "io_ctl..." );

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
    static int
_io_set_domain( hwd_control_state_t * cntrl, int domain )
{
    (void) cntrl;

    int found = 0;
    SUBDBG( "io_set_domain..." );

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
    static int
_io_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;


    switch ( modifier ) {

        /* return EventCode of first event */
        case PAPI_ENUM_FIRST:
            /* return the first event that we support */

            *EventCode = 0;
            return PAPI_OK;

            /* return EventCode of next available event */
        case PAPI_ENUM_EVENTS:
            index = *EventCode;

            /* Make sure we have at least 1 more event after us */
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
    static int
_io_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index;

    index = EventCode;

    /* Make sure we are in range */
    if (index >= 0 && index < num_events) {
        strncpy( name, io_native_table[index].name, len );  
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
    static int
_io_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
    int index;
    index = EventCode;

    /* make sure event is in range */
    if (index >= 0 && index < num_events) {
        strncpy( descr, io_native_table[index].description, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}

/** Vector that points to entry points for our component */
papi_vector_t _io_vector = {
    .cmp_info = {
        /* default component information */
        /* (unspecified values are initialized to 0) */
        /* we explicitly set them to zero in this example */
        /* to show what settings are available            */

        .name = "io",
        .short_name = "io",
        .description = "A component to read /proc/self/io",
        .version = "1.0",
        .support_version = "n/a",
        .kernel_version = "n/a",
        .num_cntrs =               IO_MAX_COUNTERS, 
        .num_mpx_cntrs =           IO_MAX_COUNTERS,
        .default_domain =          PAPI_DOM_USER,
        .available_domains =       PAPI_DOM_USER,
        .default_granularity =     PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig =       PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        /* once per thread */
        .context = sizeof ( IO_context_t ),
        /* once per eventset */
        .control_state = sizeof ( IO_control_state_t ),
        /* ?? */
        .reg_value = sizeof ( IO_register_t ),
        /* ?? */
        .reg_alloc = sizeof ( IO_reg_alloc_t ),
    },

    /* function pointers */
    /* by default they are set to NULL */

    /* Used for general PAPI interactions */
    .start =                _io_start,
    .stop =                 _io_stop,
    .read =                 _io_read,
    .reset =                _io_reset,    
    .write =                _io_write,
    .init_component =       _io_init_component,    
    .init_thread =          _io_init_thread,
    .init_control_state =   _io_init_control_state,
    .update_control_state = _io_update_control_state,    
    .ctl =                  _io_ctl,    
    .shutdown_thread =      _io_shutdown_thread,
    .shutdown_component =   _io_shutdown_component,
    .set_domain =           _io_set_domain,
    /* .cleanup_eventset =     NULL, */
    /* called in add_native_events() */
    /* .allocate_registers =   NULL, */

    /* Used for overflow/profiling */
    /* .dispatch_timer =       NULL, */
    /* .get_overflow_address = NULL, */
    /* .stop_profiling =       NULL, */
    /* .set_overflow =         NULL, */
    /* .set_profile =          NULL, */

    /* ??? */
    /* .user =                 NULL, */

    /* Name Mapping Functions */
    .ntv_enum_events =   _io_ntv_enum_events,
    .ntv_code_to_name =  _io_ntv_code_to_name,
    .ntv_code_to_descr = _io_ntv_code_to_descr,
    /* if .ntv_name_to_code not available, PAPI emulates  */
    /* it by enumerating all events and looking manually  */
    .ntv_name_to_code  = NULL,


    /* These are only used by _papi_hwi_get_native_event_info() */
    /* Which currently only uses the info for printing native   */
    /* event info, not for any sort of internal use.            */
    /* .ntv_code_to_bits =  NULL, */

};

