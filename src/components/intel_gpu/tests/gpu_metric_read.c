/* Copyright (c) 2020 Intel Corp. All rights reserved
 * Contributed by Peinan Zhang  <peinan.zhang@intel.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 *  This test case tests time based data collection on ntel GPU performance metrics
 *
 *  @ brief Collect  metric data for a certain time interval, with one or more loops.
 *          By default, the metric data will aggregate overtime for each loop.
 *          When reset, each group reports metric data only for the sepecified tiem duration
 *
 *  @option:
 *     [-d <time interval in second>][-l <number of loops>] [-s (reset)]
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "papi.h"

#define COMP_NAME   "intel_gpu"

int numDevice = 0;

#define MAX_NUM_METRICS    40
#define MAX_STRLEN         128 


char const *metric_name[MAX_STRLEN] = {
           "ComputeBasic.GpuTime",
           "ComputeBasic.GpuCoreClocks", 
           "ComputeBasic.AvgGpuCoreFrequencyMHz",
           "ComputeBasic.EuActive",
           "ComputeBasic.EuStall",
           "ComputeBasic.GtiReadThroughput",
           "ComputeBasic.GtiWriteThroughput",
};

int  num_metrics = 7;

/*
 * This test case tests:
 *  PAPI_start(),  PAPI_read(), PAPI_reset(),  PAPI_stop()
 */

int
main(int argc, char ** argv) {

    (void)argc;
    (void)argv;

    int               i = 0;
    int               retVal = 0;
    int               cid = -1;
    int               loops = 1;
    int               total_metrics = 0;
    int               eventSet = PAPI_NULL;
    int               duration = 3;
    int               reset    = 0;
    PAPI_event_info_t info;

    if (argc < 2) {
        printf("usage: %s -d <duration> [ -l <loops> -s] \n", argv[0]);
        return 0;
    }
    char ch;
    while ((ch=getopt(argc, argv, "d:l:s")) != -1) {
        switch(ch) {
            case 'd':
                duration = atoi(optarg);
                if ((duration <= 0) || (duration > 3600)) {  //  max 3600 seconds
                     printf("invalid input on dueation [1, 3600], use default 3 sec.\n");
                     duration = 3;
                }
                break;
            case 'l':
                loops = atoi(optarg);
                if ((loops <= 0) || (loops > 0x100000)) {   // max 1M
                     printf("invalid input on loops [1, 1M], use default 1 loop.\n");
                     loops  = 1;
                }
                break;
            case 's':
                reset = 1;
                break;
            default: 
                break;
        }
    }

    printf("duation %d, loops %d\n", duration, loops);

    PAPI_component_info_t *aComponent =  NULL;

    // init all components including "intel_gpu"
    retVal = PAPI_library_init( PAPI_VER_CURRENT );
    if( retVal != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed\n" );
        return 1;
    }

    int numComponents  = PAPI_num_components();
    for (i=0; i<numComponents && cid<0; i++) {
        // get the component info.
        aComponent = (PAPI_component_info_t*) PAPI_get_component_info(i);   
        if (aComponent == NULL) {
            continue;
        }
        if (strcmp(COMP_NAME, aComponent->name) == 0) {
            cid=i;                // If we found our match, record it.
        } // end search components.
    }
    if (cid < 0) {
        fprintf(stderr, "Failed to find component [%s] in total %i supported components.\n", 
            COMP_NAME, numComponents);
         PAPI_shutdown();
         return 1;
    }

    i = 0 | PAPI_NATIVE_MASK; 
    retVal=PAPI_enum_cmp_event( &i, PAPI_ENUM_FIRST, cid );
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on enum_cmp_event for component[ %s ],  abort.\n", COMP_NAME);
        PAPI_shutdown();
        return 1;
    }

    total_metrics = 0;

    do {
        memset( &info, 0, sizeof ( info ) );
        retVal =  PAPI_get_event_info( i, &info );
        if (retVal == PAPI_OK)  {
            total_metrics++;
            retVal = PAPI_enum_cmp_event( &i, PAPI_ENUM_EVENTS, cid );
        }
    } while (retVal == PAPI_OK);

    if ((!total_metrics)) {
        fprintf(stderr, "Error on enum_cmp_event, abort.\n");
        PAPI_shutdown();
        return 1;
    }
    retVal = PAPI_create_eventset(&eventSet);
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on PAPI_create_eventset, retVal %d\n", retVal);
        PAPI_shutdown();
        return 1;
    }

    for (int i=0; i<num_metrics; i++) {
        retVal = PAPI_add_named_event(eventSet, metric_name[i]);
        if (retVal < 0) {
            fprintf(stderr, "Error on PAPI_add_named_event %s,  retVal %d\n", metric_name[i], retVal);
            break;
        }
    }
    if (retVal != PAPI_OK) {
        PAPI_shutdown();
        return 1;
    }

    long long *metric_values = (long long *)calloc(num_metrics, sizeof(long long));
    if (!metric_values) {
        fprintf(stderr, "Memory allocation failed, abort.\n");
        PAPI_shutdown();
        return 1;
    }

    retVal = PAPI_start(eventSet);
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on PAPI_start, retVal %d\n", retVal);
        free(metric_values);
        PAPI_shutdown();
        return 1;
    }

    //some work here
    sleep(duration);
    loops--;
    for (int i=0; i<loops; i++) {
        PAPI_read(eventSet, metric_values);
        for (int j=0; j<num_metrics; j++) {
            printf("%-50s ......  %llu\n", metric_name[j], metric_values[j]);
        }
        printf("======\n");
        if (reset) {
            PAPI_reset(eventSet);
        }
        //some work here
        sleep(duration);
    }

    retVal = PAPI_stop(eventSet, metric_values);
    if (retVal != PAPI_OK) {
        fprintf(stderr, "Error on PAPI_stop, retVal %d\n", retVal);
        free(metric_values);
        PAPI_shutdown();
        return 1;
    }

    for (int i=0; i<num_metrics; i++) {
        printf("%-50s ......  %llu\n", metric_name[i], metric_values[i]);
    }

    free(metric_values);
    PAPI_shutdown();
    return 0;
}
