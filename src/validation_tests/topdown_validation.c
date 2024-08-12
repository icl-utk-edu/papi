/* topdown_validation.c */

/* TODO: Explain */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"
#include "testcode.h"

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>

#include <stdint.h>
#include <x86intrin.h>
#include <sys/ioctl.h>

#define NUM_EVENTS 4
#define NUM_LOOPS 100

/*
 * perf_event _rdpmc code
 */

#define RDPMC_FIXED	    (1 << 30)	/* return fixed counters */
#define RDPMC_METRIC	(1 << 29)	/* return metric counters */

#define FIXED_COUNTER_SLOTS		        3   /* on raptorlake it is fiex counter 3 */
#define METRIC_COUNTER_TOPDOWN_L1_L2	0

__attribute__((weak))
int perf_event_open(struct perf_event_attr *attr, pid_t pid,
		    int cpu, int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static inline uint64_t read_slots(void)
{
	return _rdpmc(RDPMC_FIXED | FIXED_COUNTER_SLOTS);
}

/*
The value reported by read_metrics() contains four 8 bit fields
that represent a scaled ratio that represent the Level 1 bottleneck.
All four fields add up to 0xff (= 100%)
*/
static inline uint64_t read_metrics(void)
{
	return _rdpmc(RDPMC_METRIC | METRIC_COUNTER_TOPDOWN_L1_L2);
}

static inline int reset_metrics(int fd)
{
    return ioctl(fd, PERF_EVENT_IOC_RESET, 0);
}

/*
 * When the counter is read with rdpmc, each td event is embedded in the value.
 * Applies bitshifts to the metric counter value to extract a topdown metric,
 * and scales the result into a percentage
 */
float rdpmc_get_metric(u_int64_t m, int i)
{
    return (float)(((m) >> (i * 8)) & 0xff) / 0xff * 100.0;
}

/* takes a metric integer and populates the array with individual percentages */
/* in order: retiring, bad spec, fe bound, be bound */
void topdown_get_from_metrics(double *percentages, uint64_t metrics)
{
    percentages[0] = rdpmc_get_metric(metrics, 0);
    percentages[1] = rdpmc_get_metric(metrics, 1);
    percentages[2] = rdpmc_get_metric(metrics, 2);
    percentages[3] = rdpmc_get_metric(metrics, 3);
}

/* takes papi events and populates the array with individual percentages */
/* in order: retiring, bad spec, fe bound, be bound */
void topdown_get_from_events(double *percentages, uint64_t slots, uint64_t retiring, uint64_t badspec, uint64_t be_bound)
{
    percentages[0] = (double)retiring / (double)slots * 100.0;
    percentages[1] = (double)badspec / (double)slots * 100.0;
    percentages[3] = (double)be_bound / (double)slots * 100.0;
    percentages[2] = slots - percentages[0] - percentages[1] - percentages[3];
}

void print_percs(double *percs)
{
    printf("\tretiring:\t%02f\n\tbadspec:\t%02f\n\tfe_bound:\t%02f\n\tbe_bound:\t%02f\n",
        percs[0], percs[1], percs[2], percs[3]);
}

int main(int argc, char **argv)
{
    // Set up and call the topdown events
    // Then parse as percentages and ensure it makes some sense

    int retval, tmp, result, i;
    uint64_t rdpmc_slots, rdpmc_metrics;
    int EventSet1 = PAPI_NULL;
    long long values[NUM_EVENTS];
    double rdpmc_percs[4], papi_rdpmc_percs[4], papi_events_percs[4];
    double percentages[NUM_EVENTS];
    double cycles_error;
    int quiet = 0;

    /* Set TESTS_QUIET variable */
    quiet = tests_quiet(argc, argv);

    /******************************/
    /* Set up perf_event syscalls */
    /******************************/

    /* Open slots counter file descriptor for current task. */
    struct perf_event_attr slots = {
        .type = PERF_TYPE_RAW,
        .size = sizeof(struct perf_event_attr),
        .config = 0x400,
        .exclude_kernel = 1,
    };

    int slots_fd = perf_event_open(&slots, 0, -1, -1, 0);
    if (slots_fd < 0)
        printf("Error opening the perf event slots\n");

    /* Memory mapping the fd permits _rdpmc calls from userspace */
    void *slots_p = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, slots_fd, 0);
    if (!slots_p)
        printf("Error memory mapping the fd permits for _rdpmc\n");

    /*
    * Open metrics event file descriptor for current task.
    * Set slots event as the leader of the group.
    */
    struct perf_event_attr metrics = {
        .type = PERF_TYPE_RAW,
        .size = sizeof(struct perf_event_attr),
        .config = 0x8000,
        .exclude_kernel = 1,
    };

    int metrics_fd = perf_event_open(&metrics, 0, -1, slots_fd, 0);
    if (metrics_fd < 0)
        printf("Failed to open the metrics fd\n");

    /* Memory mapping the fd permits _rdpmc calls from userspace */
    void *metrics_p = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, metrics_fd, 0);
    if (!metrics_p)
        printf("Failed to memory map the metrics\n");
    

    /***************/
    /* Set up PAPI */
    /***************/

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

    /* make sure the metrics are cleared */
    reset_metrics(slots_fd);
    reset_metrics(metrics_fd);

    /* lets start the actual test code */
    retval = PAPI_start(EventSet1);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_start", retval);
    }

    for (i = 0; i < NUM_LOOPS; i++)
    {
        result = instructions_million();
    }

    retval = PAPI_stop(EventSet1, values);
    if (retval != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
    }

    /* Check and see what _rdpmc got */
    rdpmc_slots = read_slots();
    rdpmc_metrics = read_metrics();
    topdown_get_from_metrics(rdpmc_percs, rdpmc_metrics);

    /* Lets see what we got with PAPI */
    topdown_get_from_metrics(papi_rdpmc_percs, values[1]);
    topdown_get_from_events(papi_events_percs, values[0], values[1], values[2], values[3]);

    printf("rdpmc\n");
    print_percs(rdpmc_percs);
    printf("papi rdpmc\n");
    print_percs(papi_rdpmc_percs);
    printf("papi events\n");
    print_percs(papi_events_percs);

    /* Clean up everything */
    close(slots_fd);
    close(metrics_fd);
    
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
}