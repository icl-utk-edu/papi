/*
  Example of using LD_PRELOAD with the CUDA component.  
  Asim YarKhan

  This is designed to work with the simpleMultiGPU_no_counters binary
  in the PAPI CUDA component tests directory.  First trace the library
  calls in simpleMultiGPU_no_counters binary using ltrace.  Note in
  the ltrace output that the CUDA C APIs are different from the CUDA
  calls visible to nvcc. Then figure out appropriate place to attach
  the PAPI calls.  The initialization is attached to the first entry
  to cudaSetDevice.  Each cudaSetDevice is also used to setup the PAPI
  events for that device.  It was harder to figure out where to attach
  the PAPI_start.  After running some tests, I attached it to the 18th
  invocation of gettimeofday (kind of arbitrary! Sorry!).  The
  PAPI_stop was attached to the first invocation of cudaFreeHost.

*/

#define _GNU_SOURCE

#include <stdio.h>
#include <dlfcn.h>

#include "papi.h"

#define MAXDEVICES 5
int EventSet = PAPI_NULL;
int devseen[MAXDEVICES] = {0};

static void *dl1;
int (*PAPI_library_init_ptr)(int version); /**< initialize the PAPI library */
int (*PAPI_create_eventset_ptr)(int *EventSet); /**< create a new empty PAPI event set */
int (*PAPI_add_named_event_ptr)(int EventSet, char *EventName); /**< add an event by name to a PAPI event set */
int (*PAPI_start_ptr)(int EventSet); /**< start counting hardware events in an event set */
int (*PAPI_stop_ptr)(int EventSet, long long * values); /**< stop counting hardware events in an event set and return current events */


int cudaSetDevice(int devnum, int n1, int n2, int n3, void *ptr1) 
{
    static int onetime = 0;
    int retval, retval_cudaSetDevice;
    //printf("cudaSetDevice wrapper %d\n", devnum);
    if ( onetime==0 ) {
        onetime=1;
        // Load the papi library dynamically and read the relevant functions
        dl1 = dlopen( "libpapi.so", RTLD_NOW | RTLD_GLOBAL );
        if ( dl1==NULL ) printf("Intercept cudaSetDevice: Cannot load libpapi.so\n");
        PAPI_library_init_ptr = dlsym( dl1, "PAPI_library_init" );
        PAPI_create_eventset_ptr = dlsym( dl1, "PAPI_create_eventset" );
        PAPI_add_named_event_ptr = dlsym( dl1, "PAPI_add_named_event" );
        PAPI_start_ptr = dlsym( dl1, "PAPI_start" );
        PAPI_stop_ptr = dlsym( dl1, "PAPI_stop" );
        // Start using PAPI
        printf("Intercept cudaSetDevice: Initializing PAPI on device %d\n", devnum);
        retval = (PAPI_library_init_ptr)( PAPI_VER_CURRENT );
        if( retval != PAPI_VER_CURRENT ) fprintf( stdout, "PAPI_library_init failed\n" );
        printf( "PAPI version: %d.%d.%d\n", PAPI_VERSION_MAJOR( PAPI_VERSION ), PAPI_VERSION_MINOR( PAPI_VERSION ), PAPI_VERSION_REVISION( PAPI_VERSION ) );
        retval = (PAPI_create_eventset_ptr)( &EventSet );
        if( retval != PAPI_OK ) fprintf( stdout, "PAPI_create_eventset failed\n" );
    }
    int (*original_function)(int devnum, int n1, int n2, int n3, void *ptr1);
    original_function = dlsym(RTLD_NEXT, "cudaSetDevice");
    retval_cudaSetDevice = (*original_function)( devnum, n1, n2, n3, ptr1 );
    if ( devseen[devnum]==0 ) {
        devseen[devnum]=1;
        char tmpEventName[120];
        printf("Intercept cudaSetDevice: Attaching events for device on device %d\n", devnum);
        snprintf( tmpEventName, 110, "cuda:::device:%d:%s", devnum, "inst_executed" );
        retval = (PAPI_add_named_event_ptr)( EventSet, tmpEventName );
        if (retval!=PAPI_OK) printf( "Could not add event %s\n", tmpEventName );
    }
    return retval_cudaSetDevice;
}


int gettimeofday(void *ptr1, void *ptr2)
{
    static int onetime = 0;
    onetime++;
    // printf("gettimeofday onetime %d\n", onetime);
    // Use above print statement to determine that the N-th gettime of day works
    if ( onetime==17 ) {
        printf("Intercept gettimeofday: Attaching PAPI_start to the %d th call to gettimeofday (this may need to be adjusted)\n", onetime);
        int retval = (PAPI_start_ptr)( EventSet );
        printf("Starting PAPI\n");
        if( retval!=PAPI_OK ) fprintf( stdout, "PAPI_start failed\n" );
    }
    int (*original_function)(void *ptr1, void *ptr2);
    original_function = dlsym(RTLD_NEXT, "gettimeofday");
    return (*original_function)(ptr1, ptr2);
}

int cudaFreeHost(void *ptr1, void *ptr2, int n1, int n2, void *ptr3) 
{
    static int onetime = 0;
    long long values[10];
    int retval, devnum;
    onetime++;
    if ( onetime==1 ) {
        printf("Intercept cudaFreeHost: Used to get PAPI results\n" );
        retval = (PAPI_stop_ptr)( EventSet, values );
        if( retval != PAPI_OK )  fprintf( stderr, "PAPI_stop failed\n" );
        for( devnum = 0; devnum < MAXDEVICES && devseen[devnum]==1  ; devnum++ )
            printf( "PAPI counterValue: cuda::device:%d:%s: %12lld \n", devnum, "inst_executed", values[devnum] );
    }
    int (*original_function)(void *ptr1, void *ptr2, int n1, int n2, void *ptr3);
    original_function = dlsym(RTLD_NEXT, "cudaFreeHost");
    return (*original_function)(ptr1, ptr2, n1, n2, ptr3);
}

