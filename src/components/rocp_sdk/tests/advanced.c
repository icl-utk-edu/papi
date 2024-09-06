#include <stdio.h>
#include <unistd.h>
#include <papi.h>
#include <papi_test.h>

extern void launch_kernel(int device_id);

int main(int argc, char *argv[])
{
    int papi_errno;
#define NUM_EVENTS (5)
    long long counters[NUM_EVENTS] = { 0 };

    const char *events[NUM_EVENTS] = {
                  "rocp_sdk:::SQ_CYCLES:device=1",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=1",
                  "rocp_sdk:::SQ_WAVES:device=1",
                  "rocp_sdk:::TCC_READ:device=1",
                  "rocp_sdk:::TCC_CYCLE:device=1"
    };

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        papi_errno = PAPI_add_named_event(eventset, events[i]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
    }

    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=4; ++rep){

        printf("---------------------  launch_kernel(1)\n");
        launch_kernel(1);

        sleep(1);

        papi_errno = PAPI_read(eventset, counters);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
        }
    }

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
            fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
    }

    printf("======================================================\n");
    printf("==================== SECOND ROUND ====================\n");
    printf("======================================================\n");
     
    for(int rep=0; rep<=3; ++rep){
        papi_errno = PAPI_start(eventset);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
        }

        printf("---------------------  launch_kernel(1)\n");
        launch_kernel(1);

        sleep(1);

        papi_errno = PAPI_read(eventset, counters);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
        }

        papi_errno = PAPI_stop(eventset, counters);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
        }

        printf("---------------------  PAPI_stop()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
        }
    }

    papi_errno = PAPI_cleanup_eventset(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}
