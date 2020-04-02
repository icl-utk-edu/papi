#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "papi_sde_interface.h"

#define MY_EPSILON 0.0001

static const char *event_names[1] = {
    "epsilon_count"
};

void *cntr_handle;

// API functions.
void cclib_init(void);
void cclib_do_work(void);
void cclib_do_more_work(void);

void cclib_init(void){
    papi_handle_t sde_handle;

    sde_handle = papi_sde_init("Lib_With_CC");
    papi_sde_create_counter(sde_handle, event_names[0], PAPI_SDE_DELTA, &cntr_handle);

    return;
}

void cclib_do_work(void){
    int i;

    for(i=0; i<100*1000; i++){
        double r = (double)random() / (double)RAND_MAX;
        if( r < MY_EPSILON ){
            papi_sde_inc_counter(cntr_handle, 1);
        }
        // Do some usefull work here
        if( !(i%100) )
            (void)usleep(1);
    }

    return;
}

void cclib_do_more_work(void){
    int i;

    for(i=0; i<500*1000; i++){
        double r = (double)random() / (double)RAND_MAX;
        if( r < MY_EPSILON ){
            papi_sde_inc_counter(cntr_handle, 1);
        }
        // Do some usefull work here
        if( !(i%20) )
            (void)usleep(1);
    }

    return;
}

// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t sde_handle;
    sde_handle = fptr_struct->init("Lib_With_CC");
    fptr_struct->create_counter(sde_handle, event_names[0], PAPI_SDE_DELTA, &cntr_handle);
    fptr_struct->describe_counter(sde_handle, event_names[0], "Number of times the random value was less than 0.0001");
    return sde_handle;
}
