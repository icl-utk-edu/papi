#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_sde_interface.h"

long long local_var;

void mintest_init(void){
    local_var =0;
    papi_handle_t *handle = papi_sde_init("Min Example Code");
    papi_sde_register_counter(handle, "Example Event", PAPI_SDE_RO|PAPI_SDE_DELTA, PAPI_SDE_long_long, &local_var);
}

void mintest_dowork(void){
    local_var += 7;
}

int main(int argc, char **argv){
    int ret, Eventset = PAPI_NULL;
    long long counter_values[1];

    (void)argc;
    (void)argv;

    mintest_init();

    // --- Setup PAPI
    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        fprintf(stderr,"PAPI_library_init() error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }
    
    if((ret=PAPI_create_eventset(&Eventset)) != PAPI_OK){
        fprintf(stderr,"PAPI_create_eventset() error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    if((ret=PAPI_add_named_event(Eventset, "sde:::Min Example Code::Example Event")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    // --- Start PAPI
    if((ret=PAPI_start(Eventset)) != PAPI_OK){
        fprintf(stderr,"PAPI_start error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    mintest_dowork();

    // --- Stop PAPI
    if((ret=PAPI_stop(Eventset, counter_values)) != PAPI_OK){
        fprintf(stderr,"PAPI_stop error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    if( counter_values[0] == 7 )
        printf("Success: counter value is %lld, as expected.\n",counter_values[0]);
    else
        printf("Error: counter value is %lld, when it should be 7.\n",counter_values[0]);

    return 0;
}
