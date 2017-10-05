/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    HelloWorld.c
 * CVS:     $Id$
 * @author Asim YarKhan (yarkhan@icl.utk.edu) HelloWorld altered to test power capping (October 2017) 
 * @author Heike Jagode (jagode@icl.utk.edu)
 * Mods:  <your name here> <your email address>
 *
 * @brief

 * This file is a very simple HelloWorld C example which serves
 * (together with its Makefile) as a guideline on how to add tests to
 * components.  This file tests the ability to do power control using
 * NVML.

 * The papi configure and papi Makefile will take care of the
 * compilation of the component tests (if all tests are added to a
 * directory named 'tests' in the specific component dir).  See
 * components/README for more details.
 *
 * The string "Hello World!" is mangled and then restored.
 */

#include <cuda.h>
#include <stdio.h>
#include "papi.h"
#include "papi_test.h"

#define PAPI

// Prototypes
__global__ void helloWorld( char* );

// Host function
int main( int argc, char** argv )
{

#ifdef PAPI
#define NUM_EVENTS 1
    int retval, i;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    /* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT
       FOR THE CUDA DEVICE YOU ARE RUNNING ON.
       RUN papi_native_avail to get a list of CUDA events that are
       supported on your machine */
    // e.g. on a P100 nvml:::Tesla_P100-SXM2-16GB:power
    char *EventName[NUM_EVENTS];
    int events[NUM_EVENTS];
    int eventCount = 0;
    const PAPI_component_info_t *cmpinfo;
    char event_name[PAPI_MAX_STR_LEN];

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) fprintf( stderr, "PAPI_library_init failed\n" );

    printf( "PAPI_VERSION : %4d %6d %7d\n",
            PAPI_VERSION_MAJOR( PAPI_VERSION ),
            PAPI_VERSION_MINOR( PAPI_VERSION ),
            PAPI_VERSION_REVISION( PAPI_VERSION ) );

    int numcmp = PAPI_num_components();
    // printf( "Searching for nvml component among %d components\n", numcmp );
    int cid = 0;
    for( cid=0; cid<numcmp; cid++ ) {
        cmpinfo = PAPI_get_component_info( cid );
        // printf( "Component %d (%d): %s: %d events\n", cid, cmpinfo->CmpIdx, cmpinfo->name, cmpinfo->num_native_events );
        if ( cmpinfo == NULL )
            test_fail( __FILE__, __LINE__,"PAPI_get_component_info failed\n",-1 );
        else if ( strstr( cmpinfo->name, "nvml" ) )
            break;
    }
    if ( cid==numcmp )
        test_skip( __FILE__, __LINE__,"Component nvml is not present\n",-1 );

    printf( "nvml component found: Component Index %d: %s: %d events\n", cmpinfo->CmpIdx, cmpinfo->name, cmpinfo->num_native_events );
    if ( cmpinfo->disabled )
        test_skip( __FILE__,__LINE__,"Component nvml is disabled", 0 );

    int code = PAPI_NATIVE_MASK;
    int ii=0;
    int event_modifier = PAPI_ENUM_FIRST;
    for ( ii=0; ii<cmpinfo->num_native_events; ii++ ) {
        retval = PAPI_enum_cmp_event( &code, event_modifier, cid );
        event_modifier = PAPI_ENUM_EVENTS;
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
        retval = PAPI_event_code_to_name( code, event_name );
        // printf( "Look at event %d %d %s \n", ii, code, event_name );
        if ( strstr( event_name, "power_management_limit" ) )
            break;
    }
    if ( ii==cmpinfo->num_native_events )
        test_skip( __FILE__,__LINE__,"Component nvml does not have a power_management_limit event", 0 );
    printf( "nvml power_management_limit event found (%s)\n", event_name );

    EventName[0] = event_name;

    /* convert PAPI native events to PAPI code */
    for( i = 0; i < NUM_EVENTS; i++ ) {
        retval = PAPI_event_name_to_code( ( char * )EventName[i], &events[i] );
        if( retval != PAPI_OK )
            test_fail( __FILE__,__LINE__,"PAPI_event_name_to_code failed", retval );
        eventCount++;
        // printf( "Event: %s: Code: %#x\n", EventName[i], events[i] );
    }

    /* if we did not find any valid events, just report test failed. */
    if ( eventCount == 0 )
        test_skip( __FILE__,__LINE__,"No valid events found", retval );

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK )
        test_fail( __FILE__,__LINE__,"PAPI_create_eventset failed", retval );

    retval = PAPI_add_events( EventSet, events, eventCount );
    if( retval != PAPI_OK )
        test_fail( __FILE__,__LINE__,"PAPI_add_events failed", retval );
#endif

    int j;
    int device_count;
    int cuda_device;

    cudaGetDeviceCount( &device_count );
    printf( "Found %d cuda devices\n", device_count );


///////////////////////   AYK
    for ( cuda_device = 0; cuda_device < device_count; cuda_device++ ) {
        // for ( cuda_device = 0; cuda_device < 1; cuda_device++ ) {
        printf( "cuda_device %d is being used\n", cuda_device );
        cudaSetDevice( cuda_device );

#ifdef PAPI
        retval = PAPI_start( EventSet );
        if( retval != PAPI_OK )
            test_fail( __FILE__,__LINE__,"PAPI_start failed", retval );

        retval = PAPI_read( EventSet, values );
        if( retval != PAPI_OK ) fprintf( stderr, "PAPI_read failed\n" );
        for( i = 0; i < eventCount; i++ )
            printf( "%s = %lld (read initial power management limit)\n", EventName[i], values[i]);
        long long int initial_power_management_limit = values[0];

        if ( cuda_device==0 ) {
            printf("On device_num %d the power_management_limit is going to be reduced by 30\n", cuda_device);
            // values[0] = 235000
            values[0] = initial_power_management_limit - 30;
            retval = PAPI_write( EventSet, values );
            if ( retval!=PAPI_OK ) {
                test_skip( __FILE__,__LINE__,"Attempted write of power_management_limit failed:  Possible reasons: Insufficient permissions; Power management unavailable. Outside min/max limits", retval );
            } else {
                printf( "Set power_management_limit to %llu milliWatts\n", values[0] );
            }
        }
            


#endif

        // desired output
        char str[] = "Hello World!";

        // mangle contents of output
        // the null character is left intact for simplicity
        for(j = 0; j < 12; j++) 
            str[j] -= j;
        printf( "This mangled string need to be fixed=%s\n", str );

        // allocate memory on the device
        char *d_str;
        size_t size = sizeof( str );
        cudaMalloc( ( void** )&d_str, size );

        // copy the string to the device
        cudaMemcpy( d_str, str, size, cudaMemcpyHostToDevice );

        // set the grid and block sizes
        dim3 dimGrid( 2 ); // one block per word
        dim3 dimBlock( sizeof( str )/2 ); // one thread per character

        // invoke the kernel
        helloWorld<<< dimGrid, dimBlock >>>( d_str );

        // retrieve the results from the device
        cudaMemcpy( str, d_str, size, cudaMemcpyDeviceToHost );

        // free up the allocated memory on the device
        cudaFree( d_str );

        printf( "Device %d Unmangled string = %s\n", cuda_device, str );

#ifdef PAPI
        if ( cuda_device==0 ) {
            retval = PAPI_read( EventSet, values );
            if( retval != PAPI_OK ) fprintf( stderr, "PAPI_read failed\n" );
            for( i = 0; i < eventCount; i++ )
                printf( "%s = %lld (read power management limit after reducing it... was it reduced?) \n", EventName[i], values[i] );
            
            if ( values[0] != initial_power_management_limit - 30 ) {
                printf( "Mismatch: power_management_limit on device %d set to %llu but read as %llu\n", cuda_device, initial_power_management_limit-30, values[0] );
                test_fail( __FILE__,__LINE__,"Mismatch: power_management_limit on device set to one value but read as a different value", -1 );
                
            }

            // AYK papi_reset
            long long resetvalues[NUM_EVENTS];
            resetvalues[0] = initial_power_management_limit;
            retval = PAPI_write( EventSet, resetvalues );
            retval = PAPI_stop( EventSet, values );
        }
#endif
        
    }

    test_pass( __FILE__);
    return 0;
}


// Device kernel
__global__ void
helloWorld( char* str )
{
    // determine where in the thread grid we are
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // unmangle output
    str[idx] += idx;
}

