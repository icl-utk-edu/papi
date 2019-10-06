#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi_sde_interface.h"

static const char *event_names[1] = {
    "simple_recording"
};

static papi_handle_t sde_handle;
void *rcrd_handle;

// API functions.
void recorder_init_(void);
void recorder_do_work_(void);

static papi_handle_t _recorder_papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);

void recorder_init_(void){
    papi_sde_fptr_struct_t fptr_struct;

    POPULATE_SDE_FPTR_STRUCT( fptr_struct );
    sde_handle = _recorder_papi_sde_hook_list_events(&fptr_struct);

    return;
}

papi_handle_t _recorder_papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    static papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init("Recorder");
    fptr_struct->create_recorder(tmp_handle, event_names[0], sizeof(long long), papi_sde_compare_long_long, &rcrd_handle);
    return tmp_handle;

}

// Hook for papi_native_avail utility. No user code which links against the library should call
// this function, since it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    return _recorder_papi_sde_hook_list_events(fptr_struct);
}

void recorder_do_work_(void){
    long long r = random()%123456;
    papi_sde_record(rcrd_handle, sizeof(r), &r);
    return;
}
