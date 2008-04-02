/*
 *  Test PAPI_overflow() with multiple event counters.
 */

#include <sys/time.h>
#include <sys/types.h>
#include <err.h>
#include <errno.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_EVENTS   3
#define DEF_PROGRAM_TIME  8

int program_time = DEF_PROGRAM_TIME;

int Event[] = {
    PAPI_TOT_CYC,
    PAPI_FP_INS,
    PAPI_FAD_INS,
};

int Threshold[] = {
    2000000,
     100000,
     100000,
};

int Array[NUM_EVENTS];

long count = 0;

void
my_handler(int EventSet, void *pc, long long ovec, void *context)
{
    char name[200];
    int num = NUM_EVENTS;
    int ev;

    count++;

    PAPI_get_overflow_event_index(EventSet, ovec, Array, &num);

    if (num == 1 && count > 40)
	return;

    printf("EventSet: %d, ovec: [%lld]", EventSet, ovec);
    for (ev = 0; ev < num; ev++) {
	PAPI_event_code_to_name(Event[Array[ev]], name);
	printf(", %d = %s", Array[ev], name);
    }
    printf("\n");    
}

void
launch_timer(void)
{
    int EventSet = PAPI_NULL;
    int ev;

    if (PAPI_create_eventset(&EventSet) != PAPI_OK)
        errx(1, "PAPI_create_eventset failed");

    for (ev = 0; ev < NUM_EVENTS; ev++) {
	if (PAPI_add_event(EventSet, Event[ev]) != PAPI_OK)
	    errx(1, "PAPI_add_event failed");
    }

    for (ev = 0; ev < NUM_EVENTS; ev++) {
	if (PAPI_overflow(EventSet, Event[ev], Threshold[ev], 0, my_handler)
	    != PAPI_OK) {
	    errx(1, "PAPI_overflow failed");
	}
    }

    if (PAPI_start(EventSet) != PAPI_OK)
        errx(1, "PAPI_start failed");
}

void
do_cycles(void)
{
    struct timeval start, now;
    double x, sum;

    gettimeofday(&start, NULL);

    for (;;) {
        sum = 1.0;
        for (x = 1.0; x < 250000.0; x += 1.0)
            sum += x;
        if (sum < 0.0)
            printf("==>>  SUM IS NEGATIVE !!  <<==\n");

	gettimeofday(&now, NULL);
	if (now.tv_sec > start.tv_sec + program_time)
	    break;
    }
}

/*
 *  Main program args:
 *  thresholds for Event set.
 */
int
main(int argc, char **argv)
{
    char name[200];
    int ev;

    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        errx(1, "PAPI_library_init failed");

    for (ev = 0; ev < NUM_EVENTS; ev++) {
	PAPI_event_code_to_name(Event[ev], name);
	printf("%s: code = 0x%x, threshold = %d\n",
	       name, Event[ev], Threshold[ev]);
    }


    launch_timer();
    do_cycles();

    return (0);
}
