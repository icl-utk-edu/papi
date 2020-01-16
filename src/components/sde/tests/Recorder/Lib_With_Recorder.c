#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi_sde_interface.h"

static const char *event_names[1] = {
    "simple_recording"
};

void *rcrd_handle;

// API functions.
void recorder_init_(void);
void recorder_do_work_(void);

void recorder_init_(void){
    papi_handle_t tmp_handle;

    tmp_handle = papi_sde_init("Lib_With_Recorder");
    papi_sde_create_recorder(tmp_handle, event_names[0], sizeof(long long), papi_sde_compare_long_long, &rcrd_handle);

    return;
}

void recorder_do_work_(void){
    long long r = random()%123456;
    papi_sde_record(rcrd_handle, sizeof(r), &r);
    return;
}

// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init("Lib_With_Recorder");
    fptr_struct->create_recorder(tmp_handle, event_names[0], sizeof(long long), papi_sde_compare_long_long, &rcrd_handle);
    return tmp_handle;
}
