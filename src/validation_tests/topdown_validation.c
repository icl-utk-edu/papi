/* topdown_validation.c */

/*
 * This file tests Intel's TOPDOWN level 1 and level 2 events
 * by making sure that PAPI's values agree with results obtained
 * by directly calling rdpmc, and ensuring that the percentages
 * obtained correctly add to 100. 
 * 
 * Additionally, this validation test can serve as an example of
 * how to use topdown events with PAPI, as they have some 
 * differences and quirks that make them difficult to work with.
 */

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

#define NUM_EVENTS 5
#define NUM_TESTS 100   /* test many times as fails can occur at a low rate */

#define PERCENTAGES_TOLERANCE 1.5 // +- range of percentage points for success

/**********************/
/* rdpmc related code */
/**********************/

#define SLOTS 0x0400ull
#define METRICS 0x8000ull

#define RDPMC_FIXED (1 << 30)  /* return fixed counters */
#define RDPMC_METRIC (1 << 29) /* return metric counters */

#define FIXED_COUNTER_SLOTS 3
#define METRIC_COUNTER_TOPDOWN_L1_L2 0

__attribute__((weak)) int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                                          int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static inline uint64_t read_slots(void)
{
    return _rdpmc(RDPMC_FIXED | FIXED_COUNTER_SLOTS);
}

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

/*********************/
/* general functions */
/*********************/

// prints the 4 level 1 topdown event percentages
void print_L1_percs(double *percs_l1)
{
    printf("retiring: %.02f\tbadspec: %.02f fe_bound: %.02f\tbe_bound: %.02f\n",
           percs_l1[0], percs_l1[1], percs_l1[2], percs_l1[3]);
}

// prints the 8 level 2 topdown event percentages as derived from L1 and L2 events
void print_L2_percs(double *percs_l1, double *percs_l2)
{
    double light_ops = percs_l1[0] - percs_l2[0]; // retiring - heavy_ops
    double machine_clears = percs_l1[1] - percs_l2[1]; // bad_spec - branch_mispredicts
    double fetch_bandwidth = percs_l1[2] - percs_l2[2]; // frontend_bound - fetch_latency
    double core_bound = percs_l1[3] - percs_l2[3]; // backend_bound - memory_bound

    printf("heavy ops: %.02f\tlight ops: %.02f\tbranch mispred: %.02f\tmachine clears: %.02f\tfetch lat: %.02f\tfetch band: %.02f\tmem bound: %.02f\tcore bound: %.02f\n",
            percs_l2[0], light_ops, percs_l2[1], machine_clears, percs_l2[2], fetch_bandwidth, percs_l2[3], core_bound);
}


// returns one if a and b are within the tolerance of each other
int eq_within_tolerance(double a, double b, double tolerance)
{
    if (a + tolerance >= b && a - tolerance <= b)
        return 1;

    return 0;
}

// returns 1 if the percentages are within PERCENTAGES_TOLERANCE of each other
int are_percs_equivalent(double *percs_gt, double *percs_b, double *abs_error, int n_percs)
{
    int i;
    int eq = 1;
    for (i = 0; i < n_percs; i++)
    {
        if (!eq_within_tolerance(percs_gt[i], percs_b[i], PERCENTAGES_TOLERANCE))
        {
            eq = 0;
        }
        abs_error[i] = (percs_gt[i] - percs_b[i] > 0) ? percs_gt[i] - percs_b[i] : percs_b[i] - percs_gt[i];
    }

    return eq;
}

// returns 1 if the values in the array sum to the target value
int sum_equiv_target(double target, double *percs, int n_percs)
{
    double sum = 0;
    int i;

    for (i=0; i<n_percs; i++) {
        sum += percs[i];
    }
    return eq_within_tolerance(target, sum, PERCENTAGES_TOLERANCE);
}

int main(int argc, char **argv)
{
    int failures = 0;
    int quiet = 0;
    int ret, i, j;

    int EventSetL1 = PAPI_NULL;
    int EventSetL2 = PAPI_NULL;

    double percs_perf_rdpmc[4] = {0, 0, 0, 0};
    double percs_papi_event[4] = {0, 0, 0, 0};
    double percs_papi_rdpmc[4] = {0, 0, 0, 0};
    double papi_event_error[4] = {0, 0, 0, 0};
    double papi_rdpmc_error[4] = {0, 0, 0, 0};

    long long values[NUM_EVENTS];

    uint64_t slots_val, metrics_val;

    /* Set TESTS_QUIET variable */
    quiet = tests_quiet(argc, argv);

    /* Init the PAPI library */
    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT)
    {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", ret);
    }

    /*****************************************************/
    /* Setup perf_event rdpmc to act as the ground truth */
    /*****************************************************/

    /* Open slots counter file descriptor for current task. */
    struct perf_event_attr slots = {
        .type = PERF_TYPE_RAW,
        .size = sizeof(struct perf_event_attr),
        .config = SLOTS,
        .exclude_kernel = 1,
    };

    int slots_fd = perf_event_open(&slots, 0, -1, -1, 0);
    if (slots_fd < 0)
    {
        test_fail(__FILE__, __LINE__, "Error opening the perf event slots", slots_fd);
    }
    void *slots_p = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, slots_fd, 0);
    if (!slots_p)
    {
        test_fail(__FILE__, __LINE__, "Error memory mapping the fd permits for _rdpmc", ret);
    }

    /*
     * Open metrics event file descriptor for current task.
     * Set slots event as the leader of the group.
     */
    struct perf_event_attr metrics = {
        .type = PERF_TYPE_RAW,
        .size = sizeof(struct perf_event_attr),
        .config = METRICS,
        .exclude_kernel = 1,
    };

    int metrics_fd = perf_event_open(&metrics, 0, -1, slots_fd, 0);
    if (metrics_fd < 0)
    {
        test_fail(__FILE__, __LINE__, "Failed to open the metrics fd", metrics_fd);
    }
    void *metrics_p = mmap(0, getpagesize(),
                           PROT_READ, MAP_SHARED, metrics_fd, 0);
    if (!metrics_p)
    {
        test_fail(__FILE__, __LINE__, "Failed to memory map the metrics", ret);
    }

    /**********************************************/
    /* Set up PAPI eventset for L1 topdown events */
    /**********************************************/

    /* Initialize the EventSet */
    ret = PAPI_create_eventset(&EventSetL1);
    if (ret != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", ret);
    }

    /* Add TOPDOWN:SLOTS first - slots must be the first event in a set */
    ret = PAPI_add_named_event(EventSetL1, "TOPDOWN:SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:SLOTS", ret);
    }

    /* Add TOPDOWN:RETIRING_SLOTS */
    ret = PAPI_add_named_event(EventSetL1, "TOPDOWN:RETIRING_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:RETIRING_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:RETIRING_SLOTS", ret);
    }

    /* Add TOPDOWN:BAD_SPEC_SLOTS */
    ret = PAPI_add_named_event(EventSetL1, "TOPDOWN:BAD_SPEC_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:BAD_SPEC_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:BAD_SPEC_SLOTS", ret);
    }

    /* Add TOPDOWN:FRONTEND_BOUND_SLOTS */
    ret = PAPI_add_named_event(EventSetL1, "TOPDOWN:FRONTEND_BOUND_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:FRONTEND_BOUND_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:FRONTEND_BOUND_SLOTS", ret);
    }

    /* Add TOPDOWN:BACKEND_BOUND_SLOTS */
    ret = PAPI_add_named_event(EventSetL1, "TOPDOWN:BACKEND_BOUND_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:BACKEND_BOUND_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:BACKEND_BOUND_SLOTS", ret);
    }

    /**********************************************/
    /* Set up PAPI eventset for L2 topdown events */
    /**********************************************/

    /* Initialize the EventSet */
    ret = PAPI_create_eventset(&EventSetL2);
    if (ret != PAPI_OK)
    {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", ret);
    }

    /* Add TOPDOWN:SLOTS first - slots must be the first event in a set */
    ret = PAPI_add_named_event(EventSetL2, "TOPDOWN:SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:SLOTS", ret);
    }

    /* Add TOPDOWN:HEAVY_OPS_SLOTS */
    ret = PAPI_add_named_event(EventSetL2, "TOPDOWN:HEAVY_OPS_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:HEAVY_OPS_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:HEAVY_OPS_SLOTS", ret);
    }

    /* Add TOPDOWN:BR_MISPREDICT_SLOTS */
    ret = PAPI_add_named_event(EventSetL2, "TOPDOWN:BR_MISPREDICT_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:BR_MISPREDICT_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:BR_MISPREDICT_SLOTS", ret);
    }

    /* Add TOPDOWN:FETCH_LAT_SLOTS */
    ret = PAPI_add_named_event(EventSetL2, "TOPDOWN:FETCH_LAT_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:FETCH_LAT_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:FETCH_LAT_SLOTS", ret);
    }

    /* Add TOPDOWN:MEMORY_BOUND_SLOTS */
    ret = PAPI_add_named_event(EventSetL2, "TOPDOWN:MEMORY_BOUND_SLOTS");
    if (ret != PAPI_OK)
    {
        if (!quiet)
            printf("Trouble adding TOPDOWN:MEMORY_BOUND_SLOTS\n");
        test_skip(__FILE__, __LINE__, "adding TOPDOWN:MEMORY_BOUND_SLOTS", ret);
    }

    /*
     * Now we have two event sets,
     * EventSetL1 contains the Level 1 topdown events
     * EventSetL2 contains the Level 2 topdown events
     *
     * Additionally, slots_fd and metrics_fd are set up for use with rdpmc()
     *
     * We are now ready to run some code
     */

    /* warm up the processor to pull it out of idle state */
    for (i = 0; i < 100; i++)
    {
        ret = instructions_million();
    }
    if (ret == CODE_UNIMPLEMENTED)
    {
        if (!quiet)
            printf("Instructions testcode not available\n");
        test_skip(__FILE__, __LINE__, "No instructions code", ret);
    }

    /**********************/
    /* Run some test code */
    /**********************/

    failures = 0;
    // test Level 1 topdown events
    printf("Testing L1 Topdown Events\n");
    for (i = 0; i < NUM_TESTS; i++)
    {
        // first get the ground truth (perf_event rdpmc)
        reset_metrics(metrics_fd);
        for (j = 0; j < 100; j++)
            ret = instructions_million();
        slots_val = read_slots();
        metrics_val = read_metrics();

        // then try out papi
        PAPI_start(EventSetL1);
        for (j = 0; j < 100; j++)
            ret = instructions_million();
        PAPI_stop(EventSetL1, values);

        // get percentages from perf_event rdpmc
        for (j = 0; j < 4; j++)
        {
            percs_perf_rdpmc[j] = rdpmc_get_metric(metrics_val, j);
        }

        // get percentages from papi_event (assuming rdpmc)
        for (j = 0; j < NUM_EVENTS - 1; j++)
        {
            percs_papi_rdpmc[j] = rdpmc_get_metric(values[j + 1], j);
        }

        // get percentages from papi_event (assuming non-rdpmc)
        for (j = 0; j < NUM_EVENTS - 1; j++)
        {
            percs_papi_event[j] = (double)values[j + 1] / (double)values[0] * 100.0;
        }

        // if neither result adds to 100, we fail
        if (sum_equiv_target(100.0, percs_papi_event, 4) + 
            sum_equiv_target(100.0, percs_papi_rdpmc, 4) < 1)
        {
            failures++;
            if (!quiet) {
                printf("neither interpretation method yeilded valid percentages\n");
            }

            continue;
        }

        // if neither result matches perf_event rdpmc, we fail
        if (are_percs_equivalent(percs_perf_rdpmc, percs_papi_rdpmc, papi_rdpmc_error, 4) +
                are_percs_equivalent(percs_perf_rdpmc, percs_papi_event, papi_event_error, 4) < 1)
        {
            failures++;
            if (!quiet) {
                printf("Failed test %d:\n", i);
                printf("\tground truth -\n\t");
                print_L1_percs(percs_perf_rdpmc);
                printf("\tassuming rdpmc enabled -\n\t");
                print_L1_percs(percs_papi_rdpmc);
                printf("\tassuming rdpmc disabled -\n\t");
                print_L1_percs(percs_papi_event);

                printf("\trdpmc error - ");
                for(j = 0; j < NUM_EVENTS - 1; j++)
                    printf("%.02f\t", papi_rdpmc_error[j]);
                printf("\n\tevent error - ");
                for(j = 0; j < NUM_EVENTS - 1; j++)
                    printf("%.02f\t", papi_event_error[j]);
                putchar('\n');
            }
        }
    }

    if (!quiet)
        printf("Passed %d/%d tests\n", NUM_TESTS - failures, NUM_TESTS);

    if (failures > 0) {
        test_fail(__FILE__, __LINE__, "Topdown level 1 tests failed: ", failures);
    }

    failures = 0;

    // test Level 2 topdown events
    if (!quiet)
        printf("Testing L2 Topdown Events\n");
    
    for (i = 0; i < NUM_TESTS; i++)
    {
        // first get the ground truth (perf_event rdpmc)
        reset_metrics(metrics_fd);

        for (j = 0; j < 100; j++)
            ret = instructions_million();

        slots_val = read_slots();
        metrics_val = read_metrics();

        // then try out papi
        PAPI_start(EventSetL2);
        for (j = 0; j < 100; j++)
            ret = instructions_million();
        PAPI_stop(EventSetL2, values);

        // get percentages from perf_event rdpmc
        for (j = 0; j < 4; j++)
        {
            percs_perf_rdpmc[j] = rdpmc_get_metric(metrics_val, j + 4);
        }

        // get percentages from papi_event (assuming rdpmc was enabled)
        for (j = 0; j < NUM_EVENTS - 1; j++)
        {
            percs_papi_rdpmc[j] = rdpmc_get_metric(values[j + 1], j + 4);
        }

        // get percentages from papi_event (assuming rdpmc was disabled)
        for (j = 0; j < NUM_EVENTS - 1; j++)
        {
            percs_papi_event[j] = (double)values[j + 1] / (double)values[0] * 100.0;
        }

        // as the 8 level 2 events are derived from the 4 available level 2
        // events, they will never add to 100%. Therefore there is no point
        // therefore there is no point in checking if they do 
        // (see linux kernel tools/perf/Documentation/topdown.txt)

        // if neither result matches perf_event rdpmc, we fail
        if (are_percs_equivalent(percs_perf_rdpmc, percs_papi_rdpmc, papi_rdpmc_error, 4) +
                are_percs_equivalent(percs_perf_rdpmc, percs_papi_event, papi_event_error, 4) < 1)
        {
            failures++;
            if (!quiet) {
                printf("\trdpmc error - ");
                for(j = 0; j < NUM_EVENTS - 1; j++)
                    printf("%.02f\t", papi_rdpmc_error[j]);
                printf("\n\tevent error - ");
                for(j = 0; j < NUM_EVENTS - 1; j++)
                    printf("%.02f\t", papi_event_error[j]);
                putchar('\n');
            }

            continue;
        }
    }

    printf("Passed %d/%d tests\n", NUM_TESTS - failures, NUM_TESTS);

    if (failures > 0) {
        test_fail(__FILE__, __LINE__, "Topdown level 2 tests failed.", ret);
    }

    test_pass(__FILE__);

    return 0;
}