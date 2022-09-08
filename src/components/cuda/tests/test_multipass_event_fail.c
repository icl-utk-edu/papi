#include <stdio.h>
#include "papi.h"
#include "papi_test.h"

int numEvents=3;
char const *EventName[] = {
    "cuda:::smsp__warps_launched.sum:device=0",
    "cuda:::dram__bytes_write.sum:device=0",
    "cuda:::gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed:device=0"
};

int test_PAPI_add_named_event() {
    int EventSet=PAPI_NULL, i, retval;
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    for (i=0; i<numEvents; i++) {
        retval = PAPI_add_named_event(EventSet, EventName[i]);
    }
    if (retval == PAPI_EMULPASS)
        return 1;           // Test pass condition
    return 0;
}

int test_PAPI_add_event() {
    int EventSet=PAPI_NULL, event, i, retval;
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    for (i=0; i<numEvents; i++) {
        retval = PAPI_event_name_to_code(EventName[i], &event);
        retval = PAPI_add_event(EventSet, event);
    }
    if (retval == PAPI_EMULPASS)
        return 1;
    return 0;
}

int test_PAPI_add_events() {
    int EventSet=PAPI_NULL, retval, i;
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    int events[numEvents];

    for (i=0; i<numEvents; i++) {
        retval = PAPI_event_name_to_code(EventName[i], &events[i]);
    }
    retval = PAPI_add_events(EventSet, events, numEvents);
    if (retval == 2)        // Returns index at which error occurred.
        return 1;
    return 0;
}

int main() {
    int retval, pass;

    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if (retval != PAPI_VER_CURRENT) test_fail(__FILE__, __LINE__, "PAPI_library_init() failed", 0);

    retval = PAPI_get_component_index("cuda");
    if (retval < 0 ) test_fail(__FILE__, __LINE__, "CUDA component not configured", 0);

    pass = test_PAPI_add_event();
    pass += test_PAPI_add_named_event();
    pass += test_PAPI_add_events();

    if (pass != 3)
        test_fail(__FILE__, __LINE__, "CUDA framework multipass event test failed.", 0);
    else
        test_pass(__FILE__);

    return 0;
}