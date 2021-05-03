/*
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <pthread.h>

#include "papi.h"
#include "papi_test.h"

#define DEFAULT_METRIC   "ComputeBasic.GpuTime"

const PAPI_hw_info_t *hw_info = NULL;

static int *flags;
static int num_threads = 1;
static int duration = 2;

static int *eventSet;
static long long **values;

char const *metric_name[PAPI_MAX_STR_LEN] = {
           "ComputeBasic.GpuTime",
           "ComputeBasic.GpuCoreClocks",
           "ComputeBasic.AvgGpuCoreFrequencyMHz",
           "ComputeBasic.EuActive",
           "ComputeBasic.EuStall",
           "ComputeBasic.GtiReadThroughput",
           "ComputeBasic.GtiWriteThroughput",
}; 
int  num_metrics = 7;

static int threadId = 0;

unsigned long wrapperThreadId(void) { return threadId++; }

#define exitWithError(msg, retval)   {        \
    printf("%s, retval %d\n", msg, retval);   \
    PAPI_shutdown();                          \
    exit(retval); }


void *
threadCall(void *param)
{
    int retval;
    long long elapsed_us, elapsed_cyc;

    int  thrId = *(int *)param - 1;
    printf( "Thread[%d] started\n", thrId);

    elapsed_us = PAPI_get_real_usec(  );
    elapsed_cyc = PAPI_get_real_cyc(  );

    eventSet[thrId] = PAPI_NULL;
    retval = PAPI_create_eventset(&eventSet[thrId]);
    if (retval == PAPI_OK) {
        values[thrId] = calloc(num_metrics, sizeof(long long));
        for (int j=0; j<num_metrics; j++) {
            retval = PAPI_add_named_event(eventSet[thrId], metric_name[j]);
            if (retval < 0) {
                break;
            }
    }
    }
    if (retval == PAPI_OK) {
        retval = PAPI_start( eventSet[thrId] );
    }
    if (retval == PAPI_OK) {
        sleep(duration);
        retval = PAPI_stop( eventSet[thrId], values[thrId] );
    }
    if ( retval == PAPI_OK ) {
        elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
        elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;
        printf( "thread[%u]: Real usec: %lld,   Real CPU cycles: %lld\n", 
             thrId, elapsed_us, elapsed_cyc );
    }
    PAPI_unregister_thread();
    if (retval != PAPI_OK) {
        fprintf(stderr, "thread[%u]: failed,  retval %d\n", thrId, retval);
    }
    return (NULL);
}

int
main( int argc, char **argv )
{
    int retval;
    long long elapsed_us, elapsed_cyc;
    pthread_t *colThreads;
    pthread_attr_t attr;


    if (argc < 3) {
        printf("usage: %s -d <duration> -t <num_threads>\n", argv[0]);
        return 0;
    }
    char ch;
    while ((ch=getopt(argc, argv, "d:t:")) != -1) {
        switch(ch) {
            case 'd': 
                duration = atoi(optarg);
                if ((duration <= 0) || (duration > 3600)) {  //  max 3600 seconds
                printf("invalid input on dueation [1, 3600], use default 3 sec.\n"); 
                duration = 3;
                } 
                break;
            case 't': 
                num_threads = atoi(optarg);
                if (num_threads <= 0)  {
                    num_threads = 1;
                }
                break;
             default:
                break;
        }
    }

    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        printf("PAPI_library_init failed, retval %d\n", retval );
        return -1;
    }

    hw_info = PAPI_get_hardware_info(  );
    if ( hw_info == NULL ) {
        printf("PAPI_get_hardware_info  failed with no info data\n");
        return -1;
    }
  
    printf("total num_metrics = %d\n", num_metrics); 
    for (int j=0; j<num_metrics; j++) {
        printf("named_event %s\n", metric_name[j]);
        retval = PAPI_query_named_event(metric_name[j]);
        if ( retval !=PAPI_OK) {
             printf("Can't find metric %s, retval %d\n", metric_name[j], retval);
             return -1;
        }
    }

    elapsed_us = PAPI_get_real_usec(  );
    elapsed_cyc = PAPI_get_real_cyc(  );

    retval = PAPI_thread_init(wrapperThreadId);
    if ( retval != PAPI_OK ) {
        printf("failed on init threads\n");
        return -1;
    }
    colThreads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    flags = calloc(num_threads, sizeof(int));
    eventSet = calloc(num_threads, sizeof(int));

    values = calloc(num_threads, sizeof(sizeof(long long *)));
    pthread_attr_init(&attr);

    int *id = NULL;
    for (int i=1; i<num_threads; i++) {
        id = calloc(1, sizeof(int));
        *id = i+1;
        int thrid = pthread_create(&colThreads[i], &attr, threadCall, (void *)(id));
        if (thrid != 0)  {
            exitWithError("Error creating thread using pthread_create()", -1);
        }
    }
    id = calloc(1, sizeof(int));
    *id = 1;
    int thrid = pthread_create(&colThreads[0], &attr, threadCall, (void *)(id));
    if (thrid != 0)  {
        exitWithError("Error creating thread using pthread_create()", -1);
    }
    for (int i=0;i<num_threads;i++) {
        pthread_join(colThreads[i], NULL);
    }
    printf("thread join done!\n");

    for (int i=0;i<num_threads;i++) {
        printf("thread[%d]: \n", i);
        for (int j=0; j<num_metrics; j++) {
             printf("\t %-50s  .... %llu\n", metric_name[j], values[i][j]);
        }
    }
    elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;
    elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

    printf( "Master real usec   : \t%lld\n", elapsed_us );
    printf( "Master real cycles : \t%lld\n", elapsed_cyc );

    return 0;
}
