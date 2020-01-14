#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "papi.h"

int setup_PAPI(int *event_set);
void simple_init(void);
double simple_compute(double x);

int main(int argc, char **argv){
    int i,ret, event_set = PAPI_NULL;
    long long counter_values[5];
    double *dbl_ptr;

    (void)argc;
    (void)argv;

    simple_init();

    if( 0 != setup_PAPI(&event_set) )
        exit(-1);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        fprintf(stderr,"PAPI_start error:%s \n",PAPI_strerror(ret));
        exit(-1);
    }

    for(i=0; i<10; i++){
        double sum;

        sum = simple_compute(0.87*i);
        printf("sum=%lf\n",sum);

        // --- read the event counters
        if((ret=PAPI_read(event_set, counter_values)) != PAPI_OK){
            fprintf(stderr,"PAPI_stop error:%s \n",PAPI_strerror(ret));
            exit(-1);
        }
    
        // PAPI has packed the bits of the double inside the long long.
        dbl_ptr = (double *)&counter_values[3];
        printf("Low Watermark=%lld, High Watermark=%lld, Any Watermark=%lld, Total Iterations=%lld, Comp. Value=%lf\n",
               counter_values[0], counter_values[1], counter_values[2], counter_values[3], *dbl_ptr);
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

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::LOW_WATERMARK_REACHED")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::HIGH_WATERMARK_REACHED")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::ANY_WATERMARK_REACHED")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::TOTAL_ITERATIONS")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::COMPUTED_VALUE")) != PAPI_OK){
        fprintf(stderr,"PAPI_add_named_event() error:%s \n",PAPI_strerror(ret));
        return -1;
    }

    return 0;
}

