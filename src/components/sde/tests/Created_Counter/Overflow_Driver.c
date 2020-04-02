#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"

#define EV_THRESHOLD 10

void cclib_init(void);
void cclib_do_work(void);
void cclib_do_more_work(void);
int setup_PAPI(int *event_set, int threshold);

int main(int argc, char **argv){
    int i, ret, event_set = PAPI_NULL;
    long long counter_values[1] = {0};

    (void)argc;
    (void)argv;

    cclib_init();

    if( 0 != setup_PAPI(&event_set, EV_THRESHOLD) )
        exit(-1);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_start error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    for(i=0; i<4; i++){

        cclib_do_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            fprintf(stderr,"PAPI_accum error:%s \n",PAPI_strerror(ret));
            exit(-1);
        }
        printf("Epsilon count in cclib_do_work(): %lld\n",counter_values[0]);
        counter_values[0] = 0;

        cclib_do_more_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            fprintf(stderr,"PAPI_accum error:%s \n",PAPI_strerror(ret));
            exit(-1);
        }
        printf("Epsilon count in cclib_do_more_work(): %lld\n",counter_values[0]);
        counter_values[0] = 0;
    
    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        fprintf(stderr,"PAPI_stop error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    return 0;
}


void overflow_handler(int event_set, void *address, long long overflow_vector, void *context){
    char event_name[PAPI_MAX_STR_LEN];
    int ret, *event_codes, event_index, number=1;

    (void)address;
    (void)context;

    if( (ret = PAPI_get_overflow_event_index(event_set, overflow_vector, &event_index, &number)) != PAPI_OK){
        fprintf(stderr,"PAPI_get_overflow_event_index() error:%s \n",PAPI_strerror(ret));
        return;
    }

    number = event_index+1;
    event_codes = (int *)calloc(number, sizeof(int));

    if( (ret = PAPI_list_events( event_set, event_codes, &number)) != PAPI_OK ){
        fprintf(stderr,"PAPI_list_events() error:%s \n",PAPI_strerror(ret));
        return;
    }

    if( (ret=PAPI_event_code_to_name(event_codes[event_index], event_name)) != PAPI_OK){
        fprintf(stderr,"PAPI_event_code_to_name() error:%s \n",PAPI_strerror(ret));
        return;
    }
    free(event_codes);

    printf("Event \"%s\" at index: %d exceeded its threshold again.\n",event_name, event_index);
    fflush(stdout);

    return;
}

int setup_PAPI(int *event_set, int threshold){
    int ret, event_code;

    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        fprintf(stderr,"PAPI_library_init() error:%s \n",PAPI_strerror(ret));
        return -1;
    }
    
    if((ret=PAPI_create_eventset(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_create_eventset() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_event_name_to_code("sde:::Lib_With_CC::epsilon_count", &event_code)) != PAPI_OK){
        fprintf(stderr,"PAPI_event_name_to_code() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_event(*event_set, event_code)) != PAPI_OK){
        fprintf(stderr,"PAPI_add_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret = PAPI_overflow(*event_set, event_code, threshold, PAPI_OVERFLOW_HARDWARE, overflow_handler)) != PAPI_OK){
        fprintf(stderr,"PAPI_overflow() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    return 0;
}

