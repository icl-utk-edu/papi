/**
 * @file    linux-limsr.c
 * @author  Asim YarKhan
 *
 * @ingroup papi_components
 *
 * @brief libmsr component
 *
 * This PAPI component provides access to libmsr from LLNL
 * (https://github.com/scalability-llnl/libmsr), specifically the RAPL
 * (Running Average Power Level) access in libmsr, which provides
 * energy measurements on some Intel CPUs (Need a list of CPUs here).
 *
 * To work, either msr_safe kernel module from LLNL
 * (https://github.com/scalability-llnl/msr-safe), or the x86 generic
 * MSR driver must be installed (CONFIG_X86_MSR) and the
 * /dev/cpu/?/<msr_safe | msr> files must have read permissions
 * 
 * If /dev/cpu/?/{msr_safe,msr} have appropriate write permissions,
 * you can write to the events PACKAGE_POWER_LIMIT_{1,2} to change the
 * average power (in watts) consumed by the packages/sockets over a
 * certain time window specified by events
 * PKG_TIME_WINDOW_POWER_LIMIT_{1,2} respectively.
 */
/* Based on the rapl component by Vince Weaver */

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include <msr/msr_core.h>
#include <msr/msr_rapl.h>
#include <msr/msr_counters.h>

typedef enum {
    PKG_JOULES=0,
    PKG_ELAPSED,
    PKG_DELTA_JOULES,
    PKG_WATTS,
    PKG_POWER_LIMIT_1,
    PKG_TIME_WINDOW_POWER_LIMIT_1,
    PKG_POWER_LIMIT_2,
    PKG_TIME_WINDOW_POWER_LIMIT_2,
    NUM_OF_EVENTTYPES
} eventtype_enum;

typedef struct _libmsr_register {
    unsigned int selector;
} _libmsr_register_t;

typedef struct _libmsr_native_event_entry {
    char name[PAPI_MAX_STR_LEN];
    char units[PAPI_MIN_STR_LEN];
    char description[PAPI_MAX_STR_LEN];
    int package_num;            /* which package/socket for this event */
    eventtype_enum eventtype;
    int return_type;
    _libmsr_register_t resources;
} _libmsr_native_event_entry_t;

typedef struct _libmsr_reg_alloc {
    _libmsr_register_t ra_bits;
} _libmsr_reg_alloc_t;

/* actually 32? But setting this to be safe? */
#define LIBMSR_MAX_COUNTERS 64
#define LIBMSR_MAX_PACKAGES 64

typedef struct _libmsr_control_state {
    /* The following are one per event being measured */
    int num_events_measured;
    /* int domain; */
    /* int multiplexed; */
    /* int overflow; */
    /* int inherit; */
    int being_measured[LIBMSR_MAX_COUNTERS];
    int which_counter[LIBMSR_MAX_COUNTERS];
    long long count[LIBMSR_MAX_COUNTERS];
    /* The following is boolean: Is package NN active in for event */
    int package_being_measured[LIBMSR_MAX_PACKAGES];
    /* The following tracks one libmsr_rapl_data for each package */
    struct rapl_data libmsr_rapl_data[LIBMSR_MAX_PACKAGES];
} _libmsr_control_state_t;

typedef struct _libmsr_context {
    _libmsr_control_state_t state;
} _libmsr_context_t;

papi_vector_t _libmsr_vector;

static _libmsr_native_event_entry_t *libmsr_native_events = NULL;
static int num_events_global = 0;

/***************************************************************************/

/* For dynamic linking to libmsr */
/* Using weak symbols allows PAPI to be built with the component, but
 * installed in a system without the required library */
#include <dlfcn.h>
static void* dllib1 = NULL;        
void (*_dl_non_dynamic_init)(void) __attribute__((weak));

/* Functions pointers */
static int (*init_msr_ptr)();
static int (*finalize_msr_ptr)();
static void (*read_rapl_data_ptr) ( const int socket, struct rapl_data *r );
static void (*set_rapl_limit_ptr) ( const int socket, struct rapl_limit* limit1, struct rapl_limit* limit2, struct rapl_limit* dram );
static void (*get_rapl_limit_ptr) ( const int socket, struct rapl_limit* limit1, struct rapl_limit* limit2, struct rapl_limit* dram );

/* Local wrappers for function pointers */
static int libmsr_init_msr () { return ((*init_msr_ptr)()); }
static int libmsr_finalize_msr () { return ((*finalize_msr_ptr)()); }
static void libmsr_read_rapl_data ( const int socket, struct rapl_data *r ) { return (*read_rapl_data_ptr) ( socket, r ); }
static void libmsr_set_rapl_limit ( const int socket, struct rapl_limit* limit1, struct rapl_limit* limit2, struct rapl_limit* dram ) { return (*set_rapl_limit_ptr) ( socket, limit1, limit2, dram ); }
static void libmsr_get_rapl_limit ( const int socket, struct rapl_limit* limit1, struct rapl_limit* limit2, struct rapl_limit* dram ) { return (*get_rapl_limit_ptr) ( socket, limit1, limit2, dram ); }

#define CHECK_DL_STATUS( err, str ) if( err ) { strncpy( _libmsr_vector.cmp_info.disabled_reason, str, PAPI_MAX_STR_LEN ); return ( PAPI_ENOSUPP ); }
static int _local_linkDynamicLibraries()
{
    if ( _dl_non_dynamic_init != NULL ) {
        strncpy( _libmsr_vector.cmp_info.disabled_reason, "The libmsr component REQUIRES dynamic linking capabilities.", PAPI_MAX_STR_LEN);
        return PAPI_ENOSUPP;
    }
    dllib1 = dlopen("libmsr.so", RTLD_NOW | RTLD_GLOBAL);
    CHECK_DL_STATUS( !dllib1 , "Component library libmsr.so not found." );
    init_msr_ptr = dlsym( dllib1, "init_msr" );
    CHECK_DL_STATUS( dlerror()!=NULL , "libmsr function init_msr not found." );
    finalize_msr_ptr = dlsym( dllib1, "finalize_msr" );
    CHECK_DL_STATUS( dlerror()!=NULL, "libmsr function finalize_msr_ptr not found." );
    read_rapl_data_ptr = dlsym( dllib1, "read_rapl_data" );
    CHECK_DL_STATUS( dlerror()!=NULL, "libmsr function read_rapl_data not found." );
    set_rapl_limit_ptr = dlsym( dllib1, "set_rapl_limit" );
    CHECK_DL_STATUS( dlerror()!=NULL, "libmsr function set_rapl_limit not found." );
    get_rapl_limit_ptr = dlsym( dllib1, "get_rapl_limit" );
    CHECK_DL_STATUS( dlerror()!=NULL, "libmsr function get_rapl_limit not found." );
    return( PAPI_OK);
}

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

/* Null terminated version of strncpy */
static char * _local_strlcpy( char *dst, const char *src, size_t size )
{
    char *retval = strncpy( dst, src, size );
    if ( size>0 ) dst[size-1] = '\0';
    return( retval );
}


static int _local_get_kernel_nr_cpus( void )
{
    FILE *fff;
    int num_read, nr_cpus = 1;
    fff = fopen( "/sys/devices/system/cpu/kernel_max", "r" );
    if( fff == NULL )
        return nr_cpus;
    num_read = fscanf( fff, "%d", &nr_cpus );
    fclose( fff );
    if( num_read == 1 ) {
        nr_cpus++;
    } else {
        nr_cpus = 1;
    }
    return nr_cpus;
}

/************************* PAPI Functions **********************************/

/*
 * This is called whenever a thread is initialized
 */
int _libmsr_init_thread( hwd_context_t * ctx )
{
    ( void ) ctx;
    return PAPI_OK;
}


/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
int _libmsr_init_component( int cidx )
{
    SUBDBG( "Enter: cidx: %d\n", cidx );    
    int i, j, package;
    FILE *fff;
    char filename[BUFSIZ];
    int num_packages = 0, num_cpus = 0;
    const PAPI_hw_info_t *hw_info;
    int retval;
    
    /* Dynammically load libmsr API and libraries  */
    retval = _local_linkDynamicLibraries();
    if ( retval!=PAPI_OK ) { 
        SUBDBG ("Dynamic link of libmsr.so libraries failed, component will be disabled.\n");
        SUBDBG ("See disable reason in papi_component_avail output for more details.\n");
        return (PAPI_ENOSUPP);
    }

    /* initialize libmsr */
    retval = libmsr_init_msr();
    if ( retval!=0 ) { PAPIERROR( "init_msr (libmsr) returned error.  Possible problems accessing /dev/cpu/<n>/msr_safe or /dev/cpu/<n>/msr"); return PAPI_EPERM; }
    
    /* Fill packages and cpus with sentinel values */
    int nr_cpus = _local_get_kernel_nr_cpus();
    int packages[nr_cpus];
    for( i = 0; i < nr_cpus; ++i ) packages[i] = -1;
    
    /* check if Intel processor */
    hw_info = &( _papi_hwi_system_info.hw_info );
    /* Can't use PAPI_get_hardware_info() if PAPI library not done initializing yet */
    if( hw_info->vendor != PAPI_VENDOR_INTEL ) {
        strncpy( _libmsr_vector.cmp_info.disabled_reason, "Not an Intel processor", PAPI_MAX_STR_LEN );
        return PAPI_ENOSUPP;
    }

    /* Detect how many packages and count num_cpus */
    num_cpus = 0;
    while( 1 ) {
        int num_read;
        sprintf( filename, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", num_cpus );
        fff = fopen( filename, "r" );
        if( fff == NULL ) break;
        num_read = fscanf( fff, "%d", &package );
        fclose( fff );
        if( num_read != 1 ) {
            strcpy( _libmsr_vector.cmp_info.disabled_reason, "Error reading file: " );
            strncat( _libmsr_vector.cmp_info.disabled_reason, filename, PAPI_MAX_STR_LEN - strlen( _libmsr_vector.cmp_info.disabled_reason ) - 1 );
            _libmsr_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN - 1] = '\0';
            return PAPI_ESYS;
        }
        /* Check if a new package */
        if( ( package >= 0 ) && ( package < nr_cpus ) ) {
            if( packages[package] == -1 ) {
                SUBDBG( "Found package %d out of total %d\n", package, num_packages );
                packages[package] = package;
                num_packages++;
            }
        } else {
            SUBDBG( "Package outside of allowed range\n" );
            strncpy( _libmsr_vector.cmp_info.disabled_reason, "Package outside of allowed range", PAPI_MAX_STR_LEN );
            return PAPI_ESYS;
        }
        num_cpus++;
    }

    /* Error if no accessible packages */
    if( num_packages == 0 ) {
        SUBDBG( "Can't access any physical packages\n" );
        strncpy( _libmsr_vector.cmp_info.disabled_reason, "Can't access /sys/devices/system/cpu/cpu<d>/topology/physical_package_id", PAPI_MAX_STR_LEN );
        return PAPI_ESYS;
    }
    SUBDBG( "Found %d packages with %d cpus\n", num_packages, num_cpus );

    int max_num_events = ( NUM_OF_EVENTTYPES * num_packages );
    /* Allocate space for events */
    libmsr_native_events = ( _libmsr_native_event_entry_t * ) calloc( sizeof( _libmsr_native_event_entry_t ), max_num_events );
    if ( !libmsr_native_events ) SUBDBG("Could not allocate memory\n" );
    
    /* Create events for package power info */
    num_events_global = 0;
    i = 0;
    for( j = 0; j < num_packages; j++ ) {

        sprintf( libmsr_native_events[i].name, "PKG_JOULES:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "J", PAPI_MIN_STR_LEN );
        sprintf(libmsr_native_events[i].description,"Number of Joules consumed by all cores and last level cache on package.  Unit is Joules (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_JOULES;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_WATTS:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "W", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Watts consumed by package. Unit is Watts (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_WATTS;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_ELAPSED:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "S", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Time elapsed since last LIBMSR data reading from package. Unit is seconds (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_ELAPSED;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;
        
        sprintf( libmsr_native_events[i].name, "PKG_DELTA_JOULES:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "J", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Number of Joules consumed by package since last LIBMSR data reading.  Unit is Joules (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_DELTA_JOULES;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_POWER_LIMIT_1:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "W", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Average power limit over PKG_TIME_WINDOW_POWER_LIMIT_1 for package. Read/Write. Unit is Watts (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_POWER_LIMIT_1;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_TIME_WINDOW_POWER_LIMIT_1:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "S", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Time window used for averaging PACKAGE_POWER_LIMIT_1 for package.  Read/Write.  Unit is seconds (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_TIME_WINDOW_POWER_LIMIT_1;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_POWER_LIMIT_2:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "W", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Average power limit over PKG_TIME_WINDOW_POWER_LIMIT_2 for package. Read/Write. Unit is Watts (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_POWER_LIMIT_2;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;

        sprintf( libmsr_native_events[i].name, "PKG_TIME_WINDOW_POWER_LIMIT_2:PACKAGE%d", j );
        strncpy( libmsr_native_events[i].units, "S", PAPI_MIN_STR_LEN );
        sprintf( libmsr_native_events[i].description, "Time window used for averaging PACKAGE_POWER_LIMIT_2 for package.  Read/Write.  Unit is seconds (double precision).");
        libmsr_native_events[i].package_num = j;
        libmsr_native_events[i].resources.selector = i + 1;
        libmsr_native_events[i].eventtype = PKG_TIME_WINDOW_POWER_LIMIT_2;
        libmsr_native_events[i].return_type = PAPI_DATATYPE_FP64;
        i++;
        
        // TODO Add DRAM values
        // DRAM_JOULES
        // DRAM_DELTA_JOULES
        // DRAM_WATTS
        // TODO Add PP0, PP1 events
    }
    num_events_global = i;
    
    /* Export the total number of events available */
    _libmsr_vector.cmp_info.num_native_events = num_events_global;
    _libmsr_vector.cmp_info.num_cntrs = _libmsr_vector.cmp_info.num_native_events;
    _libmsr_vector.cmp_info.num_mpx_cntrs = _libmsr_vector.cmp_info.num_native_events;
    
    /* Export the component id */
    _libmsr_vector.cmp_info.CmpIdx = cidx;
    
    return PAPI_OK;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 */
int _libmsr_init_control_state( hwd_control_state_t * ctl )
{
    SUBDBG( "Enter: ctl: %p\n", ctl );    
    _libmsr_control_state_t *control = ( _libmsr_control_state_t * ) ctl;
    int i;
    
    for( i = 0; i < LIBMSR_MAX_COUNTERS; i++ ) 
        control->which_counter[i] = 0;
    for( i = 0; i < LIBMSR_MAX_PACKAGES; i++ ) 
        control->package_being_measured[i] = 0;
    control->num_events_measured = 0;
    
    return PAPI_OK;
}


int _libmsr_update_control_state( hwd_control_state_t * ctl, NativeInfo_t * native, int count, hwd_context_t * ctx )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    int nn, index;
    ( void ) ctx;
    _libmsr_control_state_t *control = ( _libmsr_control_state_t * ) ctl;
    
    control->num_events_measured = 0;
    /* Track which events need to be measured */
    for( nn = 0; nn < count; nn++ ) {
        index = native[nn].ni_event & PAPI_NATIVE_AND_MASK;
        native[nn].ni_position = nn;
        control->which_counter[nn] = index;
        control->count[nn] = 0;
        /* Track (on/off vector) which packages/sockets need to be measured for these events */
        control->package_being_measured[libmsr_native_events[index].package_num] = 1;
        control->num_events_measured++;
    }
    return PAPI_OK;
}


int _libmsr_start( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    ( void ) ctx;
    _libmsr_control_state_t *control = ( _libmsr_control_state_t * ) ctl;
    int pp;                 /* package */
    
    /* Initialize libmsr rapl data for each package */
    for ( pp = 0; pp < LIBMSR_MAX_PACKAGES; pp++ ) {
        if ( control->package_being_measured[pp] ) {
            control->libmsr_rapl_data[pp].flags = ( 0 & RDF_INIT );
            /* Calling with RDF_INIT initializes structures */
            libmsr_read_rapl_data( pp, &control->libmsr_rapl_data[pp] );
            /* Now set the flag to re-enterant for future calls */
            control->libmsr_rapl_data[pp].flags = ( 0 | RDF_REENTRANT );
        }
    }
   return PAPI_OK;
}


int _libmsr_read( hwd_context_t * ctx, hwd_control_state_t * ctl, long long **events, int flags )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    ( void ) flags;
    ( void ) ctx;
    _libmsr_control_state_t *control = ( _libmsr_control_state_t * ) ctl;
    int nn, pp, ee;                 /* native, package, event indices */
    union { long long ll; double dbl; } event_value_union;
    struct rapl_limit limit1, limit2;
    eventtype_enum eventtype;

    /* For each package/socket marked as being measured, read libmsr libmsr data */
    for ( pp = 0; pp < LIBMSR_MAX_PACKAGES; pp++ ) {
        if ( control->package_being_measured[pp] ) {            
            libmsr_read_rapl_data( pp, &control->libmsr_rapl_data[pp] );
        }
    }
    
    /* Go thru events, assign package data to events as needed */
    SUBDBG("Go thru events, assign package data to events as needed\n");
    for( nn = 0; nn < control->num_events_measured; nn++ ) {
        ee = control->which_counter[nn];
        pp = libmsr_native_events[ee].package_num;
        event_value_union.ll = 0LL;            
        eventtype = libmsr_native_events[ee].eventtype;
        SUBDBG("nn %d ee %d pp %d eventtype %d\n", nn, ee, pp, eventtype);
        switch (eventtype) {
        case PKG_JOULES:
            event_value_union.dbl = control->libmsr_rapl_data[pp].pkg_joules;
            break;
        case PKG_ELAPSED:
            event_value_union.dbl = control->libmsr_rapl_data[pp].elapsed;
            break;
        case PKG_DELTA_JOULES:
            event_value_union.dbl = control->libmsr_rapl_data[pp].pkg_delta_joules;
            break;
        case PKG_WATTS:
            event_value_union.dbl = control->libmsr_rapl_data[pp].pkg_watts;
            break;
        case PKG_POWER_LIMIT_1:
            limit1.bits = 0;  limit1.watts = 0; limit1.seconds = 0;
            libmsr_get_rapl_limit( pp, &limit1, NULL, NULL );
            event_value_union.dbl = limit1.watts;
            break;
        case PKG_TIME_WINDOW_POWER_LIMIT_1:
            limit1.bits = 0;  limit1.watts = 0; limit1.seconds = 0;
            libmsr_get_rapl_limit( pp, &limit1, NULL, NULL );
            event_value_union.dbl = limit1.seconds;
            break;
        case PKG_POWER_LIMIT_2:
            limit2.bits = 0;  limit2.watts = 0; limit2.seconds = 0;
            libmsr_get_rapl_limit( pp, NULL, &limit2, NULL );
            event_value_union.dbl = limit2.watts;
            break;
        case PKG_TIME_WINDOW_POWER_LIMIT_2:
            limit2.bits = 0;  limit2.watts = 0; limit2.seconds = 0;
            libmsr_get_rapl_limit( pp, NULL, &limit2, NULL );
            event_value_union.dbl = limit2.seconds;
            break;
        default:
            SUBDBG("This LIBMSR event is unknown\n");
            /* error here */
        }
        control->count[nn] = event_value_union.ll;
    }

    /* Pass back a pointer to our results */
    if ( events!=NULL ) *events = ( ( _libmsr_control_state_t * ) ctl )->count;
    
    return PAPI_OK;
}


static long long _local_get_eventval_from_values( _libmsr_control_state_t *control, long long *invalues, int package_num, eventtype_enum eventtype, long long defaultval )
{
    int nn, pp, ee;                 /* native, package, event indices */
    /* Loop thru all the events, if package and repltype match, return the value  */
    for( nn = 0; nn < control->num_events_measured; nn++ ) {
        ee = control->which_counter[nn];
        pp = libmsr_native_events[ee].package_num;
        if ( pp == package_num && libmsr_native_events[ee].eventtype == eventtype ) 
            return invalues[ee];
    }
    return defaultval;
}


int _libmsr_write( hwd_context_t * ctx, hwd_control_state_t * ctl, long long *values )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    /* write values */
    ( void ) ctx;
    _libmsr_control_state_t *control = ( _libmsr_control_state_t * ) ctl;
    //long long now = PAPI_get_real_usec();
    int nn, pp, ee;                 /* native, package, event indices */
    union { long long ll; double dbl; } event_value_union;
    union { long long ll; double dbl; } timewin_union;
    struct rapl_limit limit1, limit2;
    eventtype_enum eventtype;
   
    /* Go thru events, assign package data to events as needed */
    for( nn = 0; nn < control->num_events_measured; nn++ ) {
        ee = control->which_counter[nn];
        pp = libmsr_native_events[ee].package_num;
        /* grab value and put into the union structure */
        event_value_union.ll = values[nn]; 
        /* If this is a NULL value, it means that the user does not want to write this value */
        if ( event_value_union.ll == PAPI_NULL ) continue;
        eventtype = libmsr_native_events[ee].eventtype;
        SUBDBG("nn %d ee %d pp %d eventtype %d\n", nn, ee, pp, eventtype);
        switch (eventtype) {
        case PKG_JOULES:
        case PKG_ELAPSED:
        case PKG_WATTS:
        case PKG_DELTA_JOULES:
            /* Read only so do nothing */
            break;
        case PKG_POWER_LIMIT_1:
            timewin_union.ll = _local_get_eventval_from_values( control, values, pp, PKG_TIME_WINDOW_POWER_LIMIT_1, -1 );
            if ( timewin_union.ll > 0 ) {
                limit1.watts = event_value_union.dbl;
                limit1.seconds = timewin_union.dbl;
                limit1.bits = 0;
                //printf("set_libmsr_limit package %d limit1 %lf %lf\n", pp, limit1.watts, limit1.seconds);
                libmsr_set_rapl_limit( pp, &limit1, NULL, NULL );
            } else {
                // Note error - power limit1 is not updated
                SUBDBG("PACKAGE_POWER_LIMIT_1 needs PKG_TIME_WINDOW_POWER_LIMIT_1: Power cap not updated. ");
            }
            break;
        case PKG_POWER_LIMIT_2:
            timewin_union.ll = _local_get_eventval_from_values( control, values, pp, PKG_TIME_WINDOW_POWER_LIMIT_2, -1 );
            if ( timewin_union.ll > 0 ) {
                limit2.watts = event_value_union.dbl;
                limit2.seconds = timewin_union.dbl;
                limit2.bits = 0;
                //printf("set_libmsr_limit package %d limit2 %lf %lf \n", pp, limit2.watts, limit2.seconds);
                libmsr_set_rapl_limit( pp, NULL, &limit2, NULL );
            } else {
                // Write error
                SUBDBG("PACKAGE_POWER_LIMIT_1 needs PKG_TIME_WINDOW_POWER_LIMIT_1: Power cap not updated.");
            }
            break;
        case PKG_TIME_WINDOW_POWER_LIMIT_1:
        case PKG_TIME_WINDOW_POWER_LIMIT_2:
            /* These are only meaningful (and looked up) if the power limits are set */
            break;
        default:
            SUBDBG("This LIBMSR information type is unknown\n");
            /* error here */
        }
    }
    return PAPI_OK;
}


int _libmsr_stop( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );    
    ( void ) ctx;
    ( void ) ctl;    
    return PAPI_OK;
}


/* Shutdown a thread */
int _libmsr_shutdown_thread( hwd_context_t * ctx )
{
    SUBDBG( "Enter: ctl: %p\n", ctx );    
    ( void ) ctx;
    return PAPI_OK;
}


/*
 * Clean up what was setup in  libmsr_init_component().
 */
int _libmsr_shutdown_component( void )
{
    SUBDBG( "Enter\n" );

    if ( libmsr_finalize_msr()!=0 ) {
        strncpy( _libmsr_vector.cmp_info.disabled_reason, "Function libmsr.so:finalize_msr failed. ", PAPI_MAX_STR_LEN );
        return PAPI_ESYS;
    }
    if( libmsr_native_events ) {
        free( libmsr_native_events );
        libmsr_native_events = NULL;
    }
    dlclose( dllib1 );
    return PAPI_OK;
}


/* This function sets various options in the component The valid codes
 * being passed in are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
 * PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT */
int _libmsr_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
    SUBDBG( "Enter: ctx: %p\n", ctx );
    ( void ) ctx;
    ( void ) code;
    ( void ) option;
    
    return PAPI_OK;
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
int _libmsr_set_domain( hwd_control_state_t * ctl, int domain )
{
    SUBDBG( "Enter: ctl: %p\n", ctl );
    ( void ) ctl;
    /* In theory we only support system-wide mode */
    /* How to best handle that? */
    if( domain != PAPI_DOM_ALL )
        return PAPI_EINVAL;
    return PAPI_OK;
}


int _libmsr_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    ( void ) ctx;
    ( void ) ctl;
    
    return PAPI_OK;
}


/*
 * Native Event functions
 */
int _libmsr_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    SUBDBG( "Enter: EventCode: %d\n", *EventCode );
    int index;
    if ( num_events_global == 0 )
        return PAPI_ENOEVNT;
    
    switch ( modifier ) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return PAPI_OK;
        break;
    case PAPI_ENUM_EVENTS:
        index = *EventCode & PAPI_NATIVE_AND_MASK;
        if ( index < num_events_global - 1 ) {
            *EventCode = *EventCode + 1;
            return PAPI_OK;
        } else {
            return PAPI_ENOEVNT;
        }
        break;
    // case PAPI_NTV_ENUM_UMASKS:
    default:
        return PAPI_EINVAL;
    }

    return PAPI_EINVAL;
}


/*
 *
 */
int _libmsr_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode & PAPI_NATIVE_AND_MASK;
    
    if( index >= 0 && index < num_events_global ) {
        _local_strlcpy( name, libmsr_native_events[index].name, len );
        return PAPI_OK;
    }
    return PAPI_ENOEVNT;
}


int _libmsr_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode;
    
    if( ( index < 0 ) || ( index >= num_events_global ) )
        return PAPI_ENOEVNT;
    
    _local_strlcpy( name, libmsr_native_events[index].description, len );
    return PAPI_OK;
}


int _libmsr_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t * info )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode;

    if( ( index < 0 ) || ( index >= num_events_global ) )
        return PAPI_ENOEVNT;
    
    _local_strlcpy( info->symbol, libmsr_native_events[index].name, sizeof( info->symbol ) );
    _local_strlcpy( info->long_descr, libmsr_native_events[index].description, sizeof( info->long_descr ) );
    _local_strlcpy( info->units, libmsr_native_events[index].units, sizeof( info->units ) );
    info->data_type = libmsr_native_events[index].return_type;
    return PAPI_OK;
}


papi_vector_t _libmsr_vector = {
    .cmp_info = {               /* (unspecified values are initialized to 0) */
        .name = "libmsr",
        .short_name = "libmsr",
        .description = "PAPI component for libmsr from LANL for power (RAPL) read/write",
        .version = "5.3.0",
        .default_domain = PAPI_DOM_ALL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .available_domains = PAPI_DOM_ALL,
    },
    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof( _libmsr_context_t ),
        .control_state = sizeof( _libmsr_control_state_t ),
        .reg_value = sizeof( _libmsr_register_t ),
        .reg_alloc = sizeof( _libmsr_reg_alloc_t ),
    },
    /* function pointers in this component */
    .start = _libmsr_start,
    .stop = _libmsr_stop,
    .read = _libmsr_read,
    .reset = _libmsr_reset,
    .write = _libmsr_write,
    .init_component = _libmsr_init_component,
    .init_thread = _libmsr_init_thread,
    .init_control_state = _libmsr_init_control_state,
    .update_control_state = _libmsr_update_control_state,
    .ctl = _libmsr_ctl,
    .set_domain = _libmsr_set_domain,
    .ntv_enum_events = _libmsr_ntv_enum_events,
    .ntv_code_to_name = _libmsr_ntv_code_to_name,
    .ntv_code_to_descr = _libmsr_ntv_code_to_descr,
    .ntv_code_to_info = _libmsr_ntv_code_to_info,
    .shutdown_thread = _libmsr_shutdown_thread,
    .shutdown_component = _libmsr_shutdown_component,
};
