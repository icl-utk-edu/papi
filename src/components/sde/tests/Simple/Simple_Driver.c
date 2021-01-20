#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_test.h"

long long int low_mark[10]  = { 2LL,  18LL,  67LL,  94LL, 110LL, 123LL,  180LL,  188LL,  200LL,  222LL};
long long int high_mark[10] = { 3LL,  18LL,  96LL, 143LL, 167LL, 202LL,  286LL,  295LL,  314LL,  361LL};
long long int tot_iter[10]  = {19LL, 126LL, 459LL, 658LL, 757LL, 903LL, 1271LL, 1314LL, 1400LL, 1588LL};
double comp_val[10] = {7.401931, 48.169870, 161.865161, 231.191221, 265.836745, 322.240655, 446.704770, 460.555435, 495.383479, 560.832737};

void setup_PAPI(int *event_set);
void simple_init(void);
double simple_compute(double x);

int main(int argc, char **argv){
    int i,ret, event_set = PAPI_NULL;
    int discrepancies = 0;
    int be_verbose = 0;
    long long counter_values[4];
    double *dbl_ptr;

    if( (argc > 1) && !strcmp(argv[1], "-verbose") )
        be_verbose = 1;

    simple_init();

    setup_PAPI(&event_set);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
    }

    for(i=0; i<10; i++){
        double sum;

        sum = simple_compute(0.87*i);
        if( be_verbose ) printf("sum=%lf\n",sum);

        // --- read the event counters
        if((ret=PAPI_read(event_set, counter_values)) != PAPI_OK){
            test_fail( __FILE__, __LINE__, "PAPI_read", ret );
        }

        // PAPI has packed the bits of the double inside the long long.
        dbl_ptr = (double *)&counter_values[3];
        if( be_verbose ) printf("Low Mark=%lld, High Mark=%lld, Total Iterations=%lld, Comp. Value=%lf\n",
                                counter_values[0], counter_values[1], counter_values[2], *dbl_ptr);

        if( counter_values[0] != low_mark[i] ||
            counter_values[1] != high_mark[i] ||
            counter_values[2] != tot_iter[i] ||
            (*dbl_ptr-comp_val[i]) > 0.00001 ||
            (*dbl_ptr-comp_val[i]) < -0.00001 ){
           discrepancies++;
       }

    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
    }

    if( !discrepancies )
        test_pass(__FILE__);
    else
        test_fail( __FILE__, __LINE__, "SDE counter values are wrong!", 0 );

    // The following "return" is dead code, because both test_pass() and test_fail() call exit(),
    // however, we need it to prevent compiler warnings.
    return 0;
}


void setup_PAPI(int *event_set){
    int ret;

    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
    }

    if((ret=PAPI_create_eventset(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::LOW_WATERMARK_REACHED")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::HIGH_WATERMARK_REACHED")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::TOTAL_ITERATIONS")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Simple::COMPUTED_VALUE")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    return;
}

