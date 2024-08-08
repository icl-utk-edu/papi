/* topdown_validation.c */

/* TODO: Explain */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"
#include "testcode.h"

#define NUM_EVENTS 4
#define NUM_LOOPS 100

/*
 * Applies bitshifts to the metric counter value to extract a topdown metric,
 * and scales the result into a percentage
 */
float get_metric(u_int64_t m, int i)
{
    return (float)(((m) >> (i * 8)) & 0xff) / 0xff * 100.0;
}

/* print metric percentages */
void print_metrics(u_int64_t m)
{
    printf("Metrics:\n\tretiring:\t%02f\n\tbadspec:\t%02f\n\tfrontend bound:\t%02f\n\tbackend bound:\t%02f\n",
           get_metric(m, 0), get_metric(m, 1), get_metric(m, 2), get_metric(m, 3));
}

/* ensure a metric is internally consistent */
void assert_metrics_percentages(u_int64_t m)
{
    double sum = get_metric(m, 0) + get_metric(m, 1) + get_metric(m, 2) + get_metric(m, 3);
    if (!approx_equals(sum, 100))
    {
        test_fail(__FILE__, __LINE__, "Metrics percentages do not sum to 100", 1);
    }
}

int main(int argc, char **argv)
{
    // Set up and call the topdown events
    // Then parse as percentages and ensure it makes some sense

    int retval, tmp, result, i;
    int EventSet1 = PAPI_NULL;
    long long values[NUM_EVENTS];
    long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
    double cycles_error;
    int quiet = 0;

    /* Set TESTS_QUIET variable */
    quiet = tests_quiet(argc, argv);

    /* Init the PAPI library */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT)
    {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
    }

    /* Initialize the EventSet */
    retval = PAPI_create_eventset(&EventSet1);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
    }

    /* Add TOPDOWN:SLOTS first - slots must be the first event in a set */
    retval = PAPI_add_named_event(EventSet1, "TOPDOWN:SLOTS");
    if (retval != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:SLOTS", retval);
    }

    /* Add TOPDOWN:RETIRING_SLOTS */
    retval = PAPI_add_named_event(EventSet1, "TOPDOWN:RETIRING_SLOTS");
    if (retval != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:RETIRING_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:RETIRING_SLOTS", retval);
    }

    /* Add TOPDOWN:BACKEND_BOUND_SLOTS */
    retval = PAPI_add_named_event(EventSet1, "TOPDOWN:BACKEND_BOUND_SLOTS");
    if (retval != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:BACKEND_BOUND_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:BACKEND_BOUND_SLOTS", retval);
    }

    /* Add TOPDOWN:BAD_SPEC_SLOTS */
    retval = PAPI_add_named_event(EventSet1, "TOPDOWN:BAD_SPEC_SLOTS");
    if (retval != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:BAD_SPEC_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:BAD_SPEC_SLOTS", retval);
    }

    /* warm up the processor to pull it out of idle state */
    for (i = 0; i < 100; i++)
    {
        result = instructions_million();
    }

    if (result == CODE_UNIMPLEMENTED)
    {
        if (!quiet)
            printf("Instructions testcode not available\n");
        test_skip(__FILE__, __LINE__, "No instructions code", retval);
    }

    /* Gather before stats */
    elapsed_us = PAPI_get_real_usec();
    elapsed_cyc = PAPI_get_real_cyc();
    elapsed_virt_us = PAPI_get_virt_usec();
    elapsed_virt_cyc = PAPI_get_virt_cyc();

    /* Start PAPI */
    retval = PAPI_start(EventSet1);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_start", retval);
    }

    /* our work code */
    for (i = 0; i < NUM_LOOPS; i++)
    {
        instructions_million();
    }

    /* Stop PAPI */
    retval = PAPI_stop(EventSet1, values);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
    }

    /* Lets see what we got */
    retval = PAPI_cleanup_eventset(EventSet1);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
    }

    retval = PAPI_destroy_eventset(&EventSet1);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);
    }

    printf("Slots: %d\n", values[0]);
    printf("\t%#0x\n\t%#0x\n\t%#0x\n\t#0x\n", values[1], values[2], values[3]);

    print_metrics(values[1]);
    assert_metrics_percentages(values[1]);

    print_metrics(values[2]);
    assert_metrics_percentages(values[2]);

    print_metrics(values[3]);
    assert_metrics_percentages(values[3]);
}