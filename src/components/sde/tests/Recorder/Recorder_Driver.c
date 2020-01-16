#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"

void recorder_init_(void);
void recorder_do_work_(void);
int setup_PAPI(int *event_set);

int main(int argc, char **argv){
    int i, j, ret, event_set = PAPI_NULL;
    long long counter_values[2];

    (void)argc;
    (void)argv;

    recorder_init_();

    if( 0 != setup_PAPI(&event_set) )
        exit(-1);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_start error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    for(i=0; i<10; i++){

        recorder_do_work_();

        // --- read the event counters
        if((ret=PAPI_read(event_set, counter_values)) != PAPI_OK){
            fprintf(stderr,"PAPI_stop error:%s \n",PAPI_strerror(ret));
            exit(-1);
        }
    
        printf("The number of recordings is: %lld (ptr is: %lld)\n",counter_values[0],counter_values[1]);
        for(j=0; j<counter_values[0]; j++){
            long long *ptr = (long long *)counter_values[1];
            printf("%lld ",*(ptr+j));
        }
        printf("\n");
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

    if((ret=PAPI_add_named_event(*event_set, "sde:::Lib_With_Recorder::simple_recording:CNT")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Lib_With_Recorder::simple_recording")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    return 0;
}

