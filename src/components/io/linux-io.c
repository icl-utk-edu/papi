/**
 * @file    linux-io.c
 * @author  Kevin A. Huck
 *          khuck@uoregon.edu
 *
 * @ingroup papi_components
 *
 * @brief io component
 *  This component provides access to the I/O statistics in the
 *  system file /proc/self/io. It typically contains 7 counters,
 *  but for robusness we read the file and create whatever events
 *  it contains.
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

/* Declare our vector in advance */
/* This allows us to modify the component info */
papi_vector_t _io_vector;

/* Maximum expected characters per line in /proc/self/io. */
#define FILE_LINE_SIZE 256

/** This structure is used to build the table of events */
typedef struct IO_native_event_entry
{
    char name[PAPI_MAX_STR_LEN];	      // Name of the counter.
    char desc[PAPI_MAX_STR_LEN];       // Description of the counter.
	int fileIdx;                        // Line in file.
} IO_native_event_entry_t;

/** This table contains the native events */
static IO_native_event_entry_t *io_native_table;

/** number of events in the table*/
static int EventCount;
static int EventSetCount;
static int *EventSetIdx;
static long long *EventSetVal;
static long long *EventSetReport;

// Code to read values; returns PAPI_OK or an error.
// We presume the number of counters and order of them
// will not change from our initialization read.
static int 
io_hardware_read(void)
{
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[FILE_LINE_SIZE] = {0};
    pFile = fopen ("/proc/self/io","r");
    if (pFile == NULL) return(PAPI_ENOCNTR); /* No counters */

    /* Read each line */
    int idx;
    for (idx=0; idx<EventCount; idx++) {
        if (fgets(line, FILE_LINE_SIZE-1, pFile)) {
            char dummy[FILE_LINE_SIZE] = {0};
            long long tmplong = 0LL;
            int nf = sscanf( line, "%s %lld\n", dummy, &tmplong);
            if (nf != 2 || strlen(dummy)<2 || dummy[strlen(dummy)-1] != ':') {
                fclose(pFile);
                return PAPI_ENOCNTR;
            }

            EventSetVal[idx] = tmplong;
//          printf ("%s = %lld\n", dummy, tmplong);
        } else {                            /* Did not read ALL counters. */
            fclose(pFile);
            return(PAPI_EMISC);
        }
    }

    fclose(pFile);
    return(PAPI_OK);
} // END FUNCTION.

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
    int fileIdx;
    SUBDBG( "_io_init_component..." );

    // Open the file.
    FILE * pFile;
    char line[FILE_LINE_SIZE] = {0};
    pFile = fopen ("/proc/self/io","r");
    if (pFile == NULL) {
        strncpy(_io_vector.cmp_info.disabled_reason, 
        "Required File /proc/self/io is not present.", PAPI_MAX_STR_LEN-1);
       return PAPI_ENOSUPP;               // EXIT not supported.
    }

    // Just count the lines, basic vetting.
    EventCount = 0;
    while (1) {
        char *res;
        // fgets guarantees z-terminator, reads at most FILE_LINE_SIZE-1 bytes.
        res = fgets(line, FILE_LINE_SIZE, pFile);
        if (res  == NULL) break;
        // If the read filled the whole buffer, line is too long.
        if (strlen(line) == (FILE_LINE_SIZE-1)) {
            strncpy(_io_vector.cmp_info.disabled_reason,
            "/proc/self/io line too long for current buffer size.", PAPI_MAX_STR_LEN-1);
            fclose(pFile);
            return PAPI_ENOSUPP;
        }

        char dummy[FILE_LINE_SIZE] = {0};
        long long tmplong = 0LL;
        int nf = sscanf( line, "%s %lld\n", dummy, &tmplong);
        if (nf != 2 || strlen(dummy)<2 || dummy[strlen(dummy)-1] != ':') {
            strncpy(_io_vector.cmp_info.disabled_reason,
            "/proc/self/io unexpected format, expect 'name: count'.", PAPI_MAX_STR_LEN-1);
            fclose(pFile);
            return PAPI_ENOSUPP;
        }

        EventCount++; 
    } // END READING.

    EventSetCount=0;
    EventSetIdx = papi_calloc(EventCount, sizeof(int));
    if (EventSetIdx == NULL) {
        snprintf(_io_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-1,
        "Insufficient memory to allocate %i ints.", EventCount);
        fclose(pFile);
        return PAPI_ENOMEM;
    }
        
    EventSetVal = papi_calloc(EventCount, sizeof(long long));
    if (EventSetVal == NULL) {
        snprintf(_io_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-1,
        "Insufficient memory to allocate %i long longs.", EventCount);
        fclose(pFile);
        return PAPI_ENOMEM;
    }

    EventSetReport = papi_calloc(EventCount, sizeof(long long));
    if (EventSetReport == NULL) {
        snprintf(_io_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-1,
        "Insufficient memory to allocate %i long longs.", EventCount);
        fclose(pFile);
        return PAPI_ENOMEM;
    }

   rewind(pFile);

    /* Allocate memory for the native event table */
    io_native_table =
        ( IO_native_event_entry_t * )
        papi_calloc(EventCount, sizeof(IO_native_event_entry_t) );
    if ( io_native_table == NULL ) {
        snprintf(_io_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-1,
        "Insufficient memory to allocate %i Events.", EventCount);
        fclose(pFile);
        return PAPI_ENOMEM;
    }

    for (fileIdx = 0; fileIdx < EventCount; fileIdx++) {
        (void) fgets(line, FILE_LINE_SIZE, pFile);
        char name[FILE_LINE_SIZE] = {0};
        long long tmplong = 0LL;
        // No check for error here, we would have caught it above.
        (void) sscanf( line, "%s %lld\n", name, &tmplong);
        name[strlen(name)-1]=0;     // null terminate over ':' we found.
        strncpy(io_native_table[fileIdx].name, name, PAPI_MAX_STR_LEN-1);
        io_native_table[fileIdx].fileIdx=fileIdx;
        io_native_table[fileIdx].desc[0]=0;         // flag for successful copy.
        if (strcmp("rchar", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Characters read.");
        }
        if (strcmp("wchar", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Characters written."); 
        }
        if (strcmp("syscr", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Characters read by system calls."); 
        }
        if (strcmp("syscw", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Characters written by system calls."); 
        }
        if (strcmp("read_bytes", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Binary bytes read."); 
        }
        if (strcmp("write_bytes", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Binary bytes written."); 
        }
        if (strcmp("cancelled_write_bytes", name) == 0) {
            strcpy(io_native_table[fileIdx].desc, "Binary write bytes cancelled."); 
        }
                     
        // If none of the above found, generic description.
        if (io_native_table[fileIdx].desc[0] == 0) {    
            strcpy(io_native_table[fileIdx].desc, "No description available."); 
        }
    } // END READING.

    // Export the total number of events available.
    _io_vector.cmp_info.num_native_events = EventCount;
   
    _io_vector.cmp_info.num_cntrs = EventCount;
    _io_vector.cmp_info.num_mpx_cntrs = EventCount;

    /* Export the component id */
    _io_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
} // END ROUTINE.

/** This is called whenever a thread is initialized */
static int
_io_init_thread( hwd_context_t *ctx )
{
    (void) ctx; // unused
    SUBDBG( "_io_init_thread %p...", ctx );
    return PAPI_OK;
}

// In general a control state holds the hardware info for an EventSet.
// We don't use one; we use a global state.
static int
_io_init_control_state( hwd_control_state_t * ctl )
{
    (void) ctl;
    return PAPI_OK;
}


// Triggered by eventset operations like add or remove.
// We store the order of the events, and the number.
static int
_io_update_control_state( hwd_control_state_t *ctl, 
        NativeInfo_t *native,
        int count, 
        hwd_context_t *ctx )
{
    (void) ctx;
    (void) ctl;
    
    int i, index;

    EventSetCount = count;
    
    /* if no events, return */
    if (count==0) return PAPI_OK;

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event;
        EventSetIdx[i] = index;    

        /* We have no constraints on event position, so any event */
        /* can be in any slot.                                    */
        native[i].ni_position = i;
    }

    return PAPI_OK;
} // END ROUTINE.

/** Triggered by PAPI_start() */
static int
_io_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctl;
    (void) ctx;
    SUBDBG( "io_start %p %p...", ctx, ctl );
    return PAPI_OK;
}


/** Triggered by PAPI_stop() */
static int
_io_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;
    SUBDBG( "io_stop %p %p...", ctx, ctl );
    // Don't do anything, can't stop the counters.

    return PAPI_OK;
}


// Triggered by PAPI_read(). We read all the events, then 
// pick out the ones the user actually requested, in their
// given order.
static int
_io_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
        long long **events, int flags )
{
    // Prevent 'unused' warnings from compiler.
    (void) ctx;
    (void) ctl; 
    (void) flags;
    int i;
    SUBDBG( "io_read... %p %d", ctx, flags );

    /* Read all counters into EventSetVal */
    io_hardware_read();
    for (i=0; i<EventSetCount; i++) {
        EventSetReport[i]=EventSetVal[EventSetIdx[i]];
    }

    /* return pointer to the values we read */
    *events = EventSetReport; 

    return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
static int
_io_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
        long long *events )
{
    (void) ctx;    // unused
    (void) ctl;    // unused
    (void) events; // unused

    return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
/*  We don't do anything for an io reset.                                   */
static int
_io_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx; // unused
    (void) ctl;
    SUBDBG( "io_reset...");
    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_io_shutdown_component(void)
{
    SUBDBG( "io_shutdown_component..." );
    papi_free(io_native_table);
    papi_free(EventSetIdx);
    papi_free(EventSetVal);
    papi_free(EventSetReport);
    return PAPI_OK;
}

/** Called at thread shutdown */
static int
_io_shutdown_thread( hwd_context_t *ctx )
{
    (void) ctx;
    SUBDBG( "io_shutdown_thread... %p", ctx );
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
            *EventCode = 0;
            return PAPI_OK;

        /* return EventCode of next available event */
        case PAPI_ENUM_EVENTS:
            index = *EventCode;

            /* Make sure we have at least 1 more event after us */
            if ( index < (EventCount-1) ) {
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
} // END ROUTINE

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
    if (index >= 0 && index < EventCount) {
        strncpy(name, io_native_table[index].name, len );  
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
} // END ROUTINE.

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
    if (index >= 0 && index < EventCount) {
        strncpy( descr, io_native_table[index].desc, len );
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
        .num_cntrs =               512,
        .num_mpx_cntrs =           512,
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
        .context = 1, /* unused */
        /* once per eventset */
        .control_state = 1, /* unused */
        .reg_value = 1, /* unused */
        .reg_alloc = 1, /* unused */
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

