#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"

void cclib_init(void);
void cclib_do_work(void);
void cclib_do_more_work(void);
int setup_PAPI(int *event_set);

int main(int argc, char **argv){
    int i, ret, event_set = PAPI_NULL;
    long long counter_values[1] = {0};

    (void)argc;
    (void)argv;

    cclib_init();

    if( 0 != setup_PAPI(&event_set) )
        exit(-1);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_start error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    for(i=0; i<10; i++){

        cclib_do_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            fprintf(stderr,"PAPI_accum error:%s \n",PAPI_strerror(ret));
            exit(-1);
        }
        printf("Epsilon count in cclib_do_work(): %lld\n",counter_values[0]);
        counter_values[0] = 0;

    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        fprintf(stderr,"PAPI_stop error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    return 0;
}


int setup_PAPI(int *event_set){
    int ret;

    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        fprintf(stderr,"PAPI_library_init() error:%s \n",PAPI_strerror(ret));
        return -1;
    }
    
    if((ret=PAPI_create_eventset(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_create_eventset() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Lib_With_CC::epsilon_count")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    return 0;
}

