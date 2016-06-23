/**
 * @file    linux-powercap.c
 * @author  Philip Vaccaro
 * @ingroup papi_components
 * @brief powercap component
 *
 * To work, the powercap kernel module must be loaded.
 */

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"


typedef struct _powercap_register {
    unsigned int selector;
} _powercap_register_t;

typedef struct _powercap_native_event_entry {
    char name[PAPI_MAX_STR_LEN];
    char units[PAPI_MIN_STR_LEN];
    char description[PAPI_MAX_STR_LEN];
    int zone_id;
    int subzone_id;
    int attr_id;
    int type;
    int return_type;
    _powercap_register_t resources;
} _powercap_native_event_entry_t;

typedef struct _powercap_reg_alloc {
    _powercap_register_t ra_bits;
} _powercap_reg_alloc_t;

/* actually 32?  But setting this to be safe? */
#define powercap_MAX_COUNTERS 64

typedef struct _powercap_control_state {
    int being_measured[powercap_MAX_COUNTERS];
    long long count[powercap_MAX_COUNTERS];
    long long which_counter[powercap_MAX_COUNTERS];
    int need_difference[powercap_MAX_COUNTERS];
    long long lastupdate;
} _powercap_control_state_t;

typedef struct _powercap_context {
    long long start_value[powercap_MAX_COUNTERS];
    _powercap_control_state_t state;
} _powercap_context_t;

papi_vector_t _powercap_vector;

struct fd_array_t {
    int fd;
    int open;
};

static _powercap_native_event_entry_t * powercap_native_events=NULL;
static int num_events=0;
struct fd_array_t ***zone_fd_array=NULL;
static int num_zones=0,num_subzones=0,num_attr=10;

#define ZONE_ENERGY             0
#define ZONE_MAX_ENERGY_RANGE   1
#define ZONE_MAX_POWER_A        2
#define ZONE_POWER_LIMIT_A      3
#define ZONE_TIME_WINDOW_A	4
#define ZONE_MAX_POWER_B  	5
#define ZONE_POWER_LIMIT_B      6
#define ZONE_TIME_WINDOW_B	7
#define ZONE_ENABLED 	     	8
#define ZONE_NAME 	     	9 /* check this label */
#define ZONE_ENERGY_CNT         10


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

static int _local_read_zone_str( int fd, char *out_buf, int sz )
{
    int ret = 0;
    if ( !out_buf || !sz || fd<0 )
        return -EINVAL;
    ret = read( fd, out_buf, sz );
    if ( ret > 0 && ret < sz )
        out_buf[ret] = '\0';
    close( fd );
    out_buf[sz - 1] = '\0';

    close( fd );
    return ret;
}

static int _local_write_zone_str( int fd, char *in_buf, int sz )
{
    struct flock fl;
    fl.l_type   = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start    = 0;
    fl.l_len      = 0;
    int ret = -1;
    if ( fcntl( fd, F_SETLK, &fl ) == -1 ) {
        if ( errno == EACCES || errno == EAGAIN ) {
            printf( "Already locked by another process\n" );
            /* We can't get the lock at the moment */

        } else {
            /* Handle unexpected error */;
        }
    } else { /* Lock was granted... */
        /* Unlock the locked bytes */
        ret = pwrite( fd, in_buf, sz, 0 );

        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        if ( fcntl( fd, F_SETLK, &fl ) == -1 ) {
            /* Handle error */;
        }
    }

    close( fd );
    return ret;
}


/*
 * get the number of subzones in a given "parent" power zone
 */
static int _local_get_subzone_count( int zone_id )
{
    DIR *dir;
    struct dirent *entry;
    char *base_path = "/sys/class/powercap/";
    int count = 0;
    char buffer[64];
    char buffer_path[64];

    if ( zone_id != -1 ) {
        snprintf( buffer_path, sizeof( buffer_path ), "%sintel-rapl:%d/", base_path, zone_id );
    } else {
        _local_strlcpy( buffer_path, base_path, sizeof( buffer_path ) );
    }

    dir = opendir( buffer_path );
    if ( dir != NULL ) {
        entry = readdir( dir );
        while ( entry != NULL ) {

            if ( zone_id == -1 ) {
                snprintf( buffer, sizeof( buffer ), "intel-rapl:%d", count );
            } else {
                snprintf( buffer, sizeof( buffer ),"intel-rapl:%d:%d", zone_id, count );
            }

            if ( !strncmp( entry->d_name, buffer, strlen( buffer ) ) ) {
                count++;
            }
            entry = readdir( dir );
        }
        closedir( dir );
    }
    return count;
}

/* static int _local_get_zone_count() */
/* { */
/*     int count = 0; */
/*     /\* find number of power zones (same as packages) *\/ */
/*     int idx = 0; */
/*     while( _local_get_subzone_count( idx++ ) != 0 ) */
/*         count++; */
/*     return count; */
/* } */


/*
 * opens file descriptors for events in the powercap sysfs
 */
static int _local_open_zone_fd( int zone_id, int subzone_id, int which, int flag )
{
    int fd = -1;
    char base_path[] = "/sys/class/powercap/";
    char buffer_path[128];
    char final_path[128];

    if( subzone_id == 0 )
        snprintf( buffer_path, sizeof( buffer_path ), "%sintel-rapl:%d/", base_path, zone_id );
    else
        snprintf( buffer_path, sizeof( buffer_path ), "%sintel-rapl:%d:%d/",base_path,zone_id,subzone_id-1 );

    switch( which ) {
    case ZONE_ENERGY:
        snprintf( final_path, sizeof( final_path ), "%senergy_uj", buffer_path );
        break;
    case ZONE_MAX_ENERGY_RANGE:
        snprintf( final_path, sizeof( final_path ), "%smax_energy_range_uj", buffer_path );
        break;
    case ZONE_POWER_LIMIT_A:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_0_power_limit_uw", buffer_path );
        break;
    case ZONE_MAX_POWER_A:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_0_max_power_uw", buffer_path );
        break;
    case ZONE_TIME_WINDOW_A:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_0_time_window_us", buffer_path );
        break;
    case ZONE_POWER_LIMIT_B:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_1_power_limit_uw", buffer_path );
        break;
    case ZONE_MAX_POWER_B:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_1_max_power_uw", buffer_path );
        break;
    case ZONE_TIME_WINDOW_B:
        snprintf( final_path, sizeof( final_path ), "%sconstraint_1_time_window_us", buffer_path );
        break;
    case ZONE_NAME:
        snprintf( final_path, sizeof( final_path ), "%sname", buffer_path );
        break;
    case ZONE_ENABLED:
        snprintf( final_path, sizeof( final_path ), "%senabled", buffer_path );
        break;
    default:
        break;
    }

    if( zone_fd_array[zone_id][subzone_id][which].open )
        close( zone_fd_array[zone_id][subzone_id][which].fd );

    fd = open( final_path, flag );

    if( fd >=0 ) {
        zone_fd_array[zone_id][subzone_id][which].fd = fd;
        zone_fd_array[zone_id][subzone_id][which].open = 1;
    }
    return fd;
}

/* static long long _local_read_zone_attr_value( int zone_id, int subzone_id, int which ) */
/* { */
/*     int fd = _local_open_zone_fd( zone_id, subzone_id, which, O_RDONLY ); */
/*     return _local_read_zone_attr( fd ); */
/* } */

static int _local_read_attr_value_str( int zone_id, int subzone_id, int which, char *out_buff, int sz )
{
    int fd = _local_open_zone_fd( zone_id, subzone_id, which, O_RDONLY );
    if( fd < 0 ) return -1;
    return _local_read_zone_str( fd, out_buff, sz );
}

static int _local_write_attr_value( int zone_id, int subzone_id, int which, long long value )
{
    int fd = _local_open_zone_fd( zone_id, subzone_id, which, O_RDWR );
    if( fd < 0 ) return -1;

    char value_str[PAPI_MAX_STR_LEN];
    snprintf( value_str, sizeof( value_str ), "%lld", value );
    _local_write_zone_str( fd, value_str, PAPI_MAX_STR_LEN );

    char val_str[PAPI_MAX_STR_LEN];
    /* int ret =  */
    _local_read_attr_value_str( zone_id, subzone_id, which, val_str, PAPI_MAX_STR_LEN );
    return 1;
}

/*
 * creates descriptions for each of the events based on the event type
 */
static int _local_read_zone_descr( int zone_id, int subzone_id, char *zone_name_out, int buffsz )
{
    int fd = _local_open_zone_fd( zone_id, subzone_id, ZONE_NAME, O_RDONLY );

    if( fd < 0 || !zone_name_out )
        perror( "failed to read zone name" );
    else {
        int i = 0;
        char ch[1];
        while( read( fd, &ch, sizeof( char ) ) ) {
            if( i < buffsz && ch[0] != '\n' ) {
                zone_name_out[i] = ch[0];
            }
            if( ch[0] != '\0' )
                i++;
        }
        zone_name_out[i] = '\0';
    }

    return fd;
}

/*
 * this returns the total number of events available on the system
 */
static int _local_get_num_events()
{
    int num = 0;
    int i, j, k;
    int fd = -1;
    for( i = 0; i < num_zones; i++ ) {
        for( j = 0; j < num_subzones+1; j++ ) {
            for( k = 0; k < num_attr-1; k++ ) {
                char buf[128];
                fd = _local_read_attr_value_str( i, j, k, buf, 128 );
                if( fd >= 0 ) {
                    num++;
                }
            }
        }
    }
    return num;
}


/*
 * creates actual event names to be used in papi
 */
static int _local_create_powercap_event_name( int zone_id,
        int subzone_id,
        int attr_id,
        char *event_name_out,
        int buffsz )
{
    int ret = 1;
    char zone_str[64];

    /* check if we are dealing with a "parent" zone or a subzone */
    if( subzone_id == 0 )
        snprintf( zone_str, sizeof( zone_str ), "ZONE%d", zone_id );
    else
        snprintf( zone_str, sizeof( zone_str ), "ZONE%d_SUBZONE%d", zone_id, subzone_id );

    /* create event based on attribute id */
    switch( attr_id ) {
    case ZONE_ENERGY:
        snprintf( event_name_out, buffsz, "ENERGY_UJ:%s", zone_str );
        break;
    case ZONE_ENERGY_CNT:
        snprintf( event_name_out, buffsz, "ENERGY_CNT:%s", zone_str );
        break;
    case ZONE_MAX_ENERGY_RANGE:
        snprintf( event_name_out, buffsz, "MAX_ENERGY_RANGE_UJ:%s", zone_str );
        break;
    case ZONE_MAX_POWER_A:
        snprintf( event_name_out, buffsz, "MAX_POWER_A_UW:%s", zone_str );
        break;
    case ZONE_POWER_LIMIT_A:
        snprintf( event_name_out, buffsz, "POWER_LIMIT_A_UW:%s", zone_str );
        break;
    case ZONE_TIME_WINDOW_A:
        snprintf( event_name_out, buffsz, "TIME_WINDOW_A_US:%s", zone_str );
        break;
    case ZONE_MAX_POWER_B:
        snprintf( event_name_out, buffsz, "MAX_POWER_B_UW:%s", zone_str );
        break;
    case ZONE_POWER_LIMIT_B:
        snprintf( event_name_out, buffsz, "POWER_LIMIT_B_UW:%s", zone_str );
        break;
    case ZONE_TIME_WINDOW_B:
        snprintf( event_name_out, buffsz, "TIME_WINDOW_B_US:%s", zone_str );
        break;
    case ZONE_ENABLED:
        snprintf( event_name_out, buffsz, "ENABLED:%s", zone_str );
        break;
    default:
        snprintf( event_name_out, buffsz, "%s", "" );
        ret = 0;
        break;
    }
    return ret;
}


static long long read_powercap_value( int index )
{
    char val_str[PAPI_MAX_STR_LEN];
    int ret = _local_read_attr_value_str( powercap_native_events[index].zone_id,
                                          powercap_native_events[index].subzone_id,
                                          powercap_native_events[index].attr_id,
                                          val_str, PAPI_MAX_STR_LEN );

    if( ret >= 0 )
        return atoll( val_str );
    else
        return -1;
}

static int write_powercap_value( int index, long long value )
{
    int ret = _local_write_attr_value( powercap_native_events[index].zone_id,
                                       powercap_native_events[index].subzone_id,
                                       powercap_native_events[index].attr_id,
                                       value );
    return ret;
}


/************************* PAPI Functions **********************************/

/*
 * This is called whenever a thread is initialized
 */
int _powercap_init_thread( hwd_context_t *ctx )
{
    SUBDBG( "Enter\n" );
    ( void ) ctx;
    return PAPI_OK;
}

/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
int _powercap_init_component( int cidx )
{
    SUBDBG( "Enter\n" );
    int i,j,k,fd;
    const PAPI_hw_info_t *hw_info;

    /* check if Intel processor */
    hw_info=&( _papi_hwi_system_info.hw_info );
    /* Ugh can't use PAPI_get_hardware_info() if
       PAPI library not done initializing yet */
    if ( hw_info->vendor!=PAPI_VENDOR_INTEL ) {
        strncpy( _powercap_vector.cmp_info.disabled_reason,
                 "Not an Intel processor",PAPI_MAX_STR_LEN );
        return PAPI_ENOSUPP;
    }

    /* zone 0 must exist, so we get number of subzones here.
       number of subzones is same for each zone.             */
    num_subzones = _local_get_subzone_count( 0 );
    /* can't have 0 zones */
    if( num_subzones == 0 ) {
        strncpy( _powercap_vector.cmp_info.disabled_reason,
                 "powercap modules did not detect any power zones.",PAPI_MAX_STR_LEN );
        return PAPI_ENOIMPL;
    }

    /* this may seem odd but subzones have half the number of attr as "parent" zones do */
    //int num_subzone_attr = ( ( num_attr-1 )/2 ) * num_subzones;

    /* count up the number of power zones by making sure each zone has subzones */
    num_zones = 1;
    while( _local_get_subzone_count( num_zones ) != 0 )
        num_zones++;

    /* Init fd_array */
    zone_fd_array=( struct fd_array_t*** )papi_calloc( sizeof( struct fd_array_t** ),num_zones );

    for( j = 0; j < num_zones; j++ ) {
        zone_fd_array[j]=( struct fd_array_t** )papi_calloc( sizeof( struct fd_array_t* ),num_subzones+1 );
        for( k = 0; k < num_subzones+1; k++ ) {
            zone_fd_array[j][k]=( struct fd_array_t* )papi_calloc( sizeof( struct fd_array_t ),num_attr );
        }
    }
    for( i = 0; i < num_zones; i++ ) {
        for( j = 0; j < num_subzones+1; j++ ) {
            for( k = 0; k < num_attr; k++ ) {
                zone_fd_array[i][j][k].fd = 0;
                zone_fd_array[i][j][k].open = 0;
            }
        }
    }

    if ( zone_fd_array==NULL ) return PAPI_ENOMEM;

    /* Allocate space for events */
    /* Include room for both counts and scaled values */
    num_events = _local_get_num_events();
    powercap_native_events = ( _powercap_native_event_entry_t* )
                             papi_calloc( sizeof( _powercap_native_event_entry_t ),num_events );

    if( powercap_native_events[0].description == NULL )
        exit( 0 );

    int arr_idx = 0;
    int parent  = 0;
    for( i=0; i < num_zones; i++ ) {
        for( k = 0; k < num_attr - 1; k++ ) {
            if( k == 0 ) {
                parent = arr_idx;
            }

            fd=_local_create_powercap_event_name( i,0,k,
                                                  powercap_native_events[arr_idx].name,
                                                  PAPI_MAX_STR_LEN );
            _local_read_zone_descr( i, 0, powercap_native_events[arr_idx].description, PAPI_MAX_STR_LEN );
            powercap_native_events[arr_idx].resources.selector = arr_idx + 1;
            powercap_native_events[arr_idx].zone_id = i;
            powercap_native_events[arr_idx].subzone_id = 0;
            powercap_native_events[arr_idx].attr_id = k;
            powercap_native_events[arr_idx].type = k;
            powercap_native_events[arr_idx].return_type = PAPI_DATATYPE_UINT64;
            arr_idx++;
        }
        for( j = 1; j < num_subzones+1; j++ ) {
            for( k = 0; k < num_attr - 1; k++ ) {
                char buf[128];
                fd = _local_read_attr_value_str( i, j, k, buf, 128 );
                if( fd < 0 ) {
                    continue;
                }

                fd=_local_create_powercap_event_name( i,j,k,
                                                      powercap_native_events[arr_idx].name,
                                                      PAPI_MAX_STR_LEN );
                _local_read_zone_descr( i,j,powercap_native_events[arr_idx].description,PAPI_MAX_STR_LEN );
                /* add parent description to child zones for more description */
                strcat( powercap_native_events[arr_idx].description, "-" );
                strcat( powercap_native_events[arr_idx].description,
                        powercap_native_events[parent].description );

                powercap_native_events[arr_idx].resources.selector = arr_idx + 1;
                powercap_native_events[arr_idx].zone_id = i;
                powercap_native_events[arr_idx].subzone_id = j;
                powercap_native_events[arr_idx].attr_id = k;
                powercap_native_events[arr_idx].type = k;
                powercap_native_events[arr_idx].return_type = PAPI_DATATYPE_UINT64;
                arr_idx++;
            }
        }
    }

    /* Export the total number of events available */
    _powercap_vector.cmp_info.num_native_events = num_events;
    _powercap_vector.cmp_info.num_cntrs = num_events;
    _powercap_vector.cmp_info.num_mpx_cntrs = num_events;

    /* Export the component id */
    _powercap_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int _powercap_init_control_state( hwd_control_state_t *ctl )
{
    _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
    int i;
    SUBDBG( "Enter\n" );

    for( i=0; i<powercap_MAX_COUNTERS; i++ ) {
        control->being_measured[i]=0;
    }

    return PAPI_OK;
}

int _powercap_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    _powercap_context_t* context = ( _powercap_context_t* ) ctx;
    _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
    long long now = PAPI_get_real_usec();
    int i;
    SUBDBG( "Enter\n" );

    for( i = 0; i < powercap_MAX_COUNTERS; i++ ) {
        //if ((control->being_measured[i]) && (control->need_difference[i])) {
        if ( control->being_measured[i] ) {
            context->start_value[i]=read_powercap_value( i );
        }
    }

    control->lastupdate = now;

    return PAPI_OK;
}

int _powercap_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    /* read values */
    _powercap_context_t* context = ( _powercap_context_t* ) ctx;
    _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
    long long now = PAPI_get_real_usec();
    int i;
    long long temp;
    SUBDBG( "Enter\n" );

    for ( i = 0; i < powercap_MAX_COUNTERS; i++ ) {
        if ( control->being_measured[i] ) {
            temp = read_powercap_value( i );
            if ( context->start_value[i] )
                if ( control->need_difference[i] ) {
                    if ( temp < context->start_value[i] ) {
                        SUBDBG( "Wraparound!\nstart:\t%#016x\ttemp:\t%#016x",
                                ( unsigned )context->start_value[i], ( unsigned )temp );
                        temp += ( 0x100000000 - context->start_value[i] );
                        SUBDBG( "\tresult:\t%#016x\n", ( unsigned )temp );
                    } else {
                        temp -= context->start_value[i];
                    }
                }
            control->count[i] = temp;
        }
    }
    control->lastupdate = now;
    return PAPI_OK;
}

/* Shutdown a thread */
int
_powercap_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;
    SUBDBG( "Enter\n" );
    return PAPI_OK;
}


int
_powercap_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
                long long **events, int flags )
{
    ( void ) flags;
    SUBDBG( "Enter\n" );

    _powercap_stop( ctx, ctl );

    /* Pass back a pointer to our results */
    *events = ( ( _powercap_control_state_t* ) ctl )->count;

    return PAPI_OK;
}

int _powercap_write( hwd_context_t * ctx, hwd_control_state_t * ctl, long long *values )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    /* write values */
    ( void ) ctx;
    _powercap_control_state_t *control = ( _powercap_control_state_t * ) ctl;
    int nn, ee;                 /* native, event indices */
    long long event_value_ll;
    int eventtype;
    /* Go thru events, assign package data to events as needed */
    for( nn = 0; nn < num_events; nn++ ) {
        ee = control->which_counter[nn];
        /* grab value store it */
        event_value_ll = values[nn];
        /* If this is a -1 (PAPI_NULL) value, it means that the user does not want to write this value */
        if ( event_value_ll == (long long)PAPI_NULL ) continue;
        eventtype = powercap_native_events[ee].type;

        event_value_ll = values[nn];
        switch( eventtype ) {
        case ZONE_MAX_ENERGY_RANGE:
        case ZONE_MAX_POWER_A:
        case ZONE_MAX_POWER_B:
        case ZONE_NAME:
        case ZONE_ENERGY:
            break;
        /* read only values */
        case ZONE_POWER_LIMIT_A:
            write_powercap_value( nn, event_value_ll );
            break;
        case ZONE_TIME_WINDOW_A:
            write_powercap_value( nn, event_value_ll );
            break;
        case ZONE_POWER_LIMIT_B:
            write_powercap_value( nn, event_value_ll );
            break;
        case ZONE_TIME_WINDOW_B:
            write_powercap_value( nn, event_value_ll );
            break;
        case ZONE_ENABLED:
            write_powercap_value( nn, event_value_ll );
            break;
        default:
            break;
        }
    }

    return PAPI_OK;
}
/*
 * Clean up what was setup in powercap_init_component().
 */
int _powercap_shutdown_component( void )
{
    SUBDBG( "Enter\n" );
    int i,j,k;
    // TODO Implement me _local_set_to_defaults();
    if ( powercap_native_events ) papi_free( powercap_native_events );
    if ( zone_fd_array ) {
        for( i=0; i<num_zones; i++ ) {
            for( j=0; j<num_subzones+1; j++ ) {
                for( k=0; k<num_attr; k++ ) {
                    if ( zone_fd_array[i][j][k].open ) close( zone_fd_array[i][j][k].fd );
                }
                papi_free( zone_fd_array[i][j] );
            }
            papi_free( zone_fd_array[i] );
        }
        papi_free( zone_fd_array );
    }
    return PAPI_OK;
}

/* This function sets various options in the component. The valid
 * codes being passed in are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
 * PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
int
_powercap_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    SUBDBG( "Enter: ctx: %p\n", ctx );
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


int _powercap_update_control_state( hwd_control_state_t *ctl,
                                NativeInfo_t *native, int count,
                                hwd_context_t *ctx )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    int i, index;
    ( void ) ctx;
    _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;

    for( i=0; i<powercap_MAX_COUNTERS; i++ ) 
        control->being_measured[i]=0;

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
        native[i].ni_position = powercap_native_events[index].resources.selector - 1;
        control->being_measured[index] = 1;
        control->count[i] = 1;
        control->which_counter[i] = index;

        /* Only need to subtract if it's a ZONE_ENERGY */
        control->need_difference[index] = ( powercap_native_events[index].type==ZONE_ENERGY );
    }

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
int _powercap_set_domain( hwd_control_state_t *ctl, int domain )
{
    SUBDBG( "Enter: ctl: %p\n", ctl );
    ( void ) ctl;
    /* In theory we only support system-wide mode */
    /* How to best handle that? */
    if ( PAPI_DOM_ALL != domain )
        return PAPI_EINVAL;

    return PAPI_OK;
}


int _powercap_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    SUBDBG( "Enter: ctl: %p, ctx: %p\n", ctl, ctx );
    ( void ) ctx;
    ( void ) ctl;
    return PAPI_OK;
}

/*
 * Native Event functions
 */
int _powercap_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    SUBDBG( "Enter: EventCode: %d\n", *EventCode );
    int index;
    if ( num_events==0 ) 
        return PAPI_ENOEVNT;

    switch ( modifier ) {
    case PAPI_ENUM_FIRST:
        *EventCode = 0;
        return PAPI_OK;
    case PAPI_ENUM_EVENTS:
        index = *EventCode & PAPI_NATIVE_AND_MASK;
        if ( index < num_events - 1 ) {
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

/*
 *
 */
int _powercap_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode & PAPI_NATIVE_AND_MASK;

    if ( index >= 0 && index < num_events ) {
        _local_strlcpy( name, powercap_native_events[index].name, len );
        return PAPI_OK;
    } 
    return PAPI_ENOEVNT;
}

/* 
 *
 */
int _powercap_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode;

    if ( index < 0 && index >= num_events ) 
        return PAPI_ENOEVNT;
    _local_strlcpy( name, powercap_native_events[index].description, len );
    return PAPI_OK;
}

int _powercap_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info )
{
    SUBDBG( "Enter: EventCode: %d\n", EventCode );
    int index = EventCode;

    if ( index < 0 || index >= num_events ) 
        return PAPI_ENOEVNT;

    _local_strlcpy( info->symbol, powercap_native_events[index].name, sizeof( info->symbol ));
    _local_strlcpy( info->long_descr, powercap_native_events[index].description, sizeof( info->long_descr ) );
    _local_strlcpy( info->units, powercap_native_events[index].units, sizeof( info->units ) );

    info->data_type = powercap_native_events[index].return_type;
    return PAPI_OK;
}


papi_vector_t _powercap_vector = {
    .cmp_info = { /* (unspecified values are initialized to 0) */
        .name = "powercap",
        .short_name = "powercap",
        .description = "Linux powercap energy measurements",
        .version = "5.3.0",
        .default_domain = PAPI_DOM_ALL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .available_domains = PAPI_DOM_ALL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof ( _powercap_context_t ),
        .control_state = sizeof ( _powercap_control_state_t ),
        .reg_value = sizeof ( _powercap_register_t ),
        .reg_alloc = sizeof ( _powercap_reg_alloc_t ),
    },
    /* function pointers in this component */
    .init_thread =          _powercap_init_thread,
    .init_component =       _powercap_init_component,
    .init_control_state =   _powercap_init_control_state,
    .update_control_state = _powercap_update_control_state,
    .start =                _powercap_start,
    .stop =                 _powercap_stop,
    .read =                 _powercap_read,
    .write =                _powercap_write,
    .shutdown_thread =      _powercap_shutdown_thread,
    .shutdown_component =   _powercap_shutdown_component,
    .ctl =                  _powercap_ctl,

    .set_domain =           _powercap_set_domain,
    .reset =                _powercap_reset,

    .ntv_enum_events =      _powercap_ntv_enum_events,
    .ntv_code_to_name =     _powercap_ntv_code_to_name,
    .ntv_code_to_descr =    _powercap_ntv_code_to_descr,
    .ntv_code_to_info =     _powercap_ntv_code_to_info,
};
