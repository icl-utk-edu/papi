/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    appio.c
 * CVS:     $Id: appio.c,v 1.1.2.4 2012/02/01 05:01:00 tmohan Exp $
 *
 * @author  Philip Mucci
 *          phil.mucci@samaratechnologygroup.com
 *
 * @author  Tushar Mohan
 *          tusharmohan@gmail.com
 *
 * Credit to: 
 *          Jose Pedro Oliveira
 *          jpo@di.uminho.pt
 * whose code in the linux net component was used as a template for
 * many sections of code in this component.
 *
 * @ingroup papi_components
 *
 * @brief appio component
 *  This file contains the source code for a component that enables
 *  PAPI to access application level file and socket I/O information.
 *  It does this through function replacement in the first person and
 *  by trapping syscalls in the third person.
 */

#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
//#include <dlfcn.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

#include "appio.h"

/*
#pragma weak dlerror
static void *_dlsym_fake(void *handle, const char* symbol) { (void) handle; (void) symbol; return NULL; }
void *dlsym(void *handle, const char* symbol) __attribute__ ((weak, alias ("_dlsym_fake")));
*/

papi_vector_t _appio_vector;

/*********************************************************************
 * Private
 ********************************************************************/

#define APPIO_FOO 1

static APPIO_native_event_entry_t * _appio_native_events;


static __thread long long _appio_register_current[APPIO_MAX_COUNTERS];

typedef enum {
  READ_BYTES = 0,
  READ_CALLS,
  READ_ERR,
  READ_INTERRUPTED,
  READ_WOULD_BLOCK,
  READ_SHORT,
  READ_EOF,
  READ_BLOCK_SIZE,
  READ_USEC,
  WRITE_BYTES,
  WRITE_CALLS,
  WRITE_ERR,
  WRITE_SHORT,
  WRITE_BLOCK_SIZE,
  WRITE_USEC,
  OPEN_CALLS,
  OPEN_ERR,
  OPEN_FDS,
  OPEN_USEC
} _appio_stats_t ;

static const struct appio_counters {
    const char *name;
    const char *description;
} _appio_counter_info[APPIO_MAX_COUNTERS] = {
    { "READ_BYTES",      "Bytes read"},
    { "READ_CALLS",      "Number of read calls"},
    { "READ_ERR",        "Number of read calls that resulted in an error"},
    { "READ_INTERRUPTED","Number of read calls that timed out or were interruped"},
    { "READ_WOULD_BLOCK","Number of read calls that would have blocked on a descriptor marked as non-blocking"},
    { "READ_SHORT",      "Number of read calls that returned less bytes than requested"},
    { "READ_EOF",        "Number of read calls that returned an EOF"},
    { "READ_BLOCK_SIZE", "Average block size of reads"},
    { "READ_USEC",       "Real microseconds spent in reads"},
    { "WRITE_BYTES",     "Bytes written"},
    { "WRITE_CALLS",     "Number of write calls"},
    { "WRITE_ERR",       "Number of write calls that resulted in an error"},
    { "WRITE_SHORT",     "Number of write calls that wrote less bytes than requested"},
    { "WRITE_BLOCK_SIZE","Mean block size of writes"},
    { "WRITE_USEC",      "Real microseconds spent in writes"},
    { "OPEN_CALLS",      "Number of open calls"},
    { "OPEN_ERR",        "Number of open calls that resulted in an error"},
    { "OPEN_FDS",        "Number of descriptors opened by application since launch"},
    { "OPEN_USEC",       "Real microseconds spent in open calls"}
};


/*********************************************************************
 ***  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ****
 ********************************************************************/

int __open(const char *pathname, int flags, mode_t mode);
int open(const char *pathname, int flags, mode_t mode) {
  int retval;
  SUBDBG("appio: intercepted open(%s,%d,%d)\n", pathname, flags, mode);
  retval = __open(pathname,flags,mode);
  _appio_register_current[OPEN_CALLS]++;
  if (retval < 0) _appio_register_current[OPEN_ERR]++;
  else _appio_register_current[OPEN_FDS]++;
  return retval;
}

ssize_t __read(int fd, void *buf, size_t count);
ssize_t read(int fd, void *buf, size_t count) {
  int retval;
  SUBDBG("appio: intercepted read(%d,%p,%lu)\n", fd, buf, (unsigned long)count);
  long long start_ts = PAPI_get_real_usec();
  retval = __read(fd,buf, count);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[READ_CALLS]++; // read calls
  if (retval > 0) {
    _appio_register_current[READ_BLOCK_SIZE]= (n * _appio_register_current[READ_BLOCK_SIZE] + count)/(n+1); // mean size
    _appio_register_current[READ_BYTES] += retval; // read bytes
    if (retval < (int)count) _appio_register_current[READ_SHORT]++; // read short
    _appio_register_current[READ_USEC] += duration;
  }
  if (retval < 0) { 
    _appio_register_current[READ_ERR]++; // read err
    if (EINTR == retval)
      _appio_register_current[READ_INTERRUPTED]++; // signal interrupted the read
    if ((EAGAIN == retval) || (EWOULDBLOCK == retval)) 
      _appio_register_current[READ_WOULD_BLOCK]++; //read would block on descriptor marked as non-blocking
  }
  if (retval == 0) _appio_register_current[READ_EOF]++; // read eof
  return retval;
}

size_t _IO_fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t retval;
  SUBDBG("appio: intercepted fread(%p,%lu,%lu,%p)\n", ptr, (unsigned long) size, (unsigned long) nmemb, (void*) stream);
  long long start_ts = PAPI_get_real_usec();
  retval = _IO_fread(ptr,size,nmemb,stream);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[READ_CALLS]++; // read calls
  if (retval > 0) {
    _appio_register_current[READ_BLOCK_SIZE]= (n * _appio_register_current[READ_BLOCK_SIZE]+ size*nmemb)/(n+1);//mean size
    _appio_register_current[READ_BYTES]+= retval * size; // read bytes
    if (retval < nmemb) _appio_register_current[READ_SHORT]++; // read short
    _appio_register_current[READ_USEC] += duration;
  }

  /* A value of zero returned means one of two things..*/
  if (retval == 0) {
     if (feof(stream)) _appio_register_current[READ_EOF]++; // read eof
     else _appio_register_current[READ_ERR]++; // read err
  }
  return retval;
}

ssize_t __write(int fd, const void *buf, size_t count);
ssize_t write(int fd, const void *buf, size_t count) {
  int retval;
  SUBDBG("appio: intercepted write(%d,%p,%lu)\n", fd, buf, (unsigned long)count);
  long long start_ts = PAPI_get_real_usec();
  retval = __write(fd,buf, count);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[WRITE_CALLS]++; // write calls
  if (retval >= 0) {
    _appio_register_current[WRITE_BLOCK_SIZE]= (n * _appio_register_current[WRITE_BLOCK_SIZE] + count)/(n+1); // mean size
    _appio_register_current[WRITE_BYTES]+= retval; // write bytes
    if (retval < (int)count) _appio_register_current[WRITE_SHORT]++; // short write
    _appio_register_current[WRITE_USEC] += duration;
  }
  if (retval < 0) _appio_register_current[WRITE_ERR]++; // err
  return retval;
}

size_t _IO_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t retval;
  SUBDBG("appio: intercepted fwrite(%p,%lu,%lu,%p)\n", ptr, (unsigned long) size, (unsigned long) nmemb, (void*) stream);
  long long start_ts = PAPI_get_real_usec();
  retval = _IO_fwrite(ptr,size,nmemb,stream);
  long long duration = PAPI_get_real_usec() - start_ts;
  int n = _appio_register_current[WRITE_CALLS]++; // write calls
  if (retval > 0) {
    _appio_register_current[WRITE_BLOCK_SIZE]= (n * _appio_register_current[WRITE_BLOCK_SIZE] + size*nmemb)/(n+1); // mean block size
    _appio_register_current[WRITE_BYTES]+= retval * size; // write bytes
    if (retval < nmemb) _appio_register_current[WRITE_SHORT]++; // short write
    _appio_register_current[WRITE_USEC] += duration;
  }
  if (retval == 0) _appio_register_current[WRITE_ERR]++; // err
  return retval;
}


/*********************************************************************
 ***************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *********
 *********************************************************************/

/*
 * This is called whenever a thread is initialized
 */
int
_appio_init( hwd_context_t *ctx )
{
    ( void ) ctx;
    SUBDBG("_appio_init %p\n", ctx);
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_appio_init_substrate( int cidx  )
{

    SUBDBG("_appio_substrate %d\n", cidx);
    _appio_native_events = (APPIO_native_event_entry_t *) papi_calloc(sizeof(APPIO_native_event_entry_t), APPIO_MAX_COUNTERS);

    if (_appio_native_events == NULL ) {
      PAPIERROR( "malloc():Could not get memory for events table" );
      return PAPI_ENOMEM;
    }
    int i;
    for (i=0; i<APPIO_MAX_COUNTERS; i++) {
      _appio_native_events[i].name = _appio_counter_info[i].name;
      _appio_native_events[i].description = _appio_counter_info[i].description;
      _appio_native_events[i].resources.selector = i + 1;
    }
  
    /* Export the total number of events available */
    _appio_vector.cmp_info.num_native_events = APPIO_MAX_COUNTERS;;

    /* Export the component id */
    _appio_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
_appio_init_control_state( hwd_control_state_t *ctl )
{
    ( void ) ctl;

    return PAPI_OK;
}


int
_appio_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;

    SUBDBG("_appio_start %p %p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;

    /* this memset needs to move to thread_init */
    memset(_appio_register_current, 0, APPIO_MAX_COUNTERS * sizeof(_appio_register_current[0]));

    /* set initial values to 0 */
    memset(appio_ctl->values, 0, APPIO_MAX_COUNTERS*sizeof(appio_ctl->values[0]));
    
    return PAPI_OK;
}


int
_appio_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
    long long ** events, int flags )
{
    (void) flags;
    (void) ctx;

    SUBDBG("_appio_read %p %p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    int i;

    for ( i=0; i<appio_ctl->num_events; i++ ) {
            int index = appio_ctl->counter_bits[i];
            SUBDBG("event=%d, index=%d, val=%lld\n", i, index, _appio_register_current[index]);
            appio_ctl->values[index] = _appio_register_current[index];
    }
    *events = appio_ctl->values;

    return PAPI_OK;
}


int
_appio_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;

    SUBDBG("_appio_stop ctx=%p ctl=%p\n", ctx, ctl);
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    int i;
    for ( i=0; i<appio_ctl->num_events; i++ ) {
            int index = appio_ctl->counter_bits[i];
            SUBDBG("event=%d, index=%d, val=%lld\n", i, index, _appio_register_current[index]);
            appio_ctl->values[i] = _appio_register_current[index];
    }

    return PAPI_OK;
}


/*
 * Thread shutdown
 */
int
_appio_shutdown( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}


/*
 * Clean up what was setup in appio_init_substrate().
 */
int
_appio_shutdown_substrate( void )
{
    return PAPI_OK;
}


/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and
 * PAPI_SET_INHERIT
 */
int
_appio_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


int
_appio_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t *native, int count, hwd_context_t *ctx )
{
    ( void ) ctx;
    ( void ) ctl;

    SUBDBG("_appio_update_control_state ctx=%p ctl=%p num_events=%d\n", ctx, ctl, count);
    int i, index;
    APPIO_control_state_t *appio_ctl = (APPIO_control_state_t *) ctl;
    (void) ctx;

    for ( i = 0; i < count; i++ ) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
        appio_ctl->counter_bits[i] = index;
        native[i].ni_position = index;
    }
    appio_ctl->num_events = count;

    return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER   is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL    is all of the domains
 */
int
_appio_set_domain( hwd_control_state_t *ctl, int domain )
{
    ( void ) ctl;

    int found = 0;

    if ( PAPI_DOM_USER & domain )   found = 1;
    if ( PAPI_DOM_KERNEL & domain ) found = 1;
    if ( PAPI_DOM_OTHER & domain )  found = 1;

    if ( !found )
        return PAPI_EINVAL;

    return PAPI_OK;
}


int
_appio_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;

    return PAPI_OK;
}


/*
 * Native Event functions
 */
int
_appio_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;
    int cidx = PAPI_COMPONENT_INDEX( *EventCode );

    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            *EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK(cidx);
            return PAPI_OK;
            break;

        case PAPI_ENUM_EVENTS:
            index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
            if ( index < APPIO_MAX_COUNTERS - 1 ) {
                *EventCode = *EventCode + 1;
                return PAPI_OK;
            } else {
                return PAPI_ENOEVNT;
            }
            break;

        default:
            return PAPI_EINVAL;
            break;
    }
    return PAPI_EINVAL;
}


/*
 *
 */
int
_appio_ntv_name_to_code( char *name, unsigned int *EventCode )
{
    int i;

    for ( i=0; i<APPIO_MAX_COUNTERS; i++) {
        if (strcmp(name, _appio_counter_info[i].name) == 0) {
            *EventCode = i |
                PAPI_NATIVE_MASK |
                PAPI_COMPONENT_MASK(_appio_vector.cmp_info.CmpIdx);
            return PAPI_OK;
        }
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
int
_appio_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        strncpy( name, _appio_counter_info[index].name, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
int
_appio_ntv_code_to_descr( unsigned int EventCode, char *desc, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        strncpy(desc, _appio_counter_info[index].description, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
int
_appio_ntv_code_to_bits( unsigned int EventCode, hwd_register_t *bits )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

    if ( index >= 0 && index < APPIO_MAX_COUNTERS ) {
        memcpy( ( APPIO_register_t * ) bits,
                &( _appio_native_events[index].resources ),
                sizeof ( APPIO_register_t ) );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
papi_vector_t _appio_vector = {
    .cmp_info = {
        /* default component information (unspecified values are initialized to 0) */
        .name                  = "appio.c",
        .version               = "$Revision: 1.1.2.4 $",
        .CmpIdx                = 0,              /* set by init_substrate */
        .num_mpx_cntrs         = PAPI_MPX_DEF_DEG,
        .num_cntrs             = APPIO_MAX_COUNTERS,
        .default_domain        = PAPI_DOM_USER,
        //.available_domains   = PAPI_DOM_USER,
        .default_granularity   = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig     = PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
        .fast_real_timer       = 0,
        .fast_virtual_timer    = 0,
        .attach                = 0,
        .attach_must_ptrace    = 0,
        .available_domains     = PAPI_DOM_USER | PAPI_DOM_KERNEL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context               = sizeof ( APPIO_context_t ),
        .control_state         = sizeof ( APPIO_control_state_t ),
        .reg_value             = sizeof ( APPIO_register_t ),
        .reg_alloc             = sizeof ( APPIO_reg_alloc_t ),
    },

    /* function pointers in this component */
    .init                      = _appio_init,
    .init_substrate            = _appio_init_substrate,
    .init_control_state        = _appio_init_control_state,
    .start                     = _appio_start,
    .stop                      = _appio_stop,
    .read                      = _appio_read,
    .shutdown                  = _appio_shutdown,
    .shutdown_substrate        = _appio_shutdown_substrate,
    .ctl                       = _appio_ctl,

    .update_control_state      = _appio_update_control_state,
    .set_domain                = _appio_set_domain,
    .reset                     = _appio_reset,

    .ntv_enum_events           = _appio_ntv_enum_events,
    .ntv_name_to_code          = _appio_ntv_name_to_code,
    .ntv_code_to_name          = _appio_ntv_code_to_name,
    .ntv_code_to_descr         = _appio_ntv_code_to_descr,
    .ntv_code_to_bits          = _appio_ntv_code_to_bits
    /* .ntv_bits_to_info          = NULL, */
};

/* vim:set ts=4 sw=4 sts=4 et: */
