/* 
* File:    multiplex1_pthreads.c
* CVS:     $Id$
* Author:  Rick Kufrin
*          
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file really bangs on the multiplex pthread functionality */

#include <pthread.h>
#include "papi_test.h"

int events[PAPI_MPX_DEF_DEG];
int numevents = 0;

double loop(long n)
{
    long i;
    double a = 0.0012;

    for (i=0; i<n; i++) {
      a += 0.01;
    }
    return a;
}

void *thread(void *arg)
{
    int eventset, ret;
    unsigned long long values[PAPI_MPX_DEF_DEG];

    eventset = PAPI_NULL;
    ret = PAPI_create_eventset(&eventset);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", ret);
    }
    
    printf("Event set %d created\n", eventset);

    ret = PAPI_set_multiplex(eventset);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", ret);
    }

    ret = PAPI_add_events(eventset, events, numevents);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_add_events", ret);
    }

    ret = PAPI_start(eventset);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_start", ret);
    }

    do_both(NUM_ITERS);

    ret = PAPI_stop(eventset, values);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_stop", ret);
    }

    ret = PAPI_cleanup_eventset(eventset);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", ret);
    }

    ret = PAPI_destroy_eventset(&eventset);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", ret);
    }

    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    int nthreads=64, ret, i;
    PAPI_event_info_t info;
    pthread_t *threads;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if (!TESTS_QUIET)
     {
       if ( argc > 1 ) 
	 {
	   int tmp = atoi(argv[1]);
	   if (tmp >= 1)
	     nthreads = tmp;
	 }
    }

    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if ( ret != PAPI_VER_CURRENT ) {
      test_fail(__FILE__, __LINE__, "PAPI_library_init", ret);
    }

   if (!TESTS_QUIET)
     if ((ret = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
       test_fail(__FILE__, __LINE__, "PAPI_set_debug", ret);

    ret = PAPI_thread_init(pthread_self);
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_thread_init", ret);
    }

    ret = PAPI_multiplex_init();
    if ( ret != PAPI_OK ) {
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", ret);
    }

    /* Fill up the event set with as many non-derived events as we can */

    i = PAPI_PRESET_MASK;
    do {
        if ( PAPI_get_event_info(i, &info) == PAPI_OK ) {
            if ( info.count == 1 ) {
                events[numevents++] = info.event_code;
                printf("Added %s\n", info.symbol);
            }
            else {
                printf("Skipping derived event %s\n", info.symbol);
            }
        }
    } while ( (PAPI_enum_event(&i, PAPI_PRESET_ENUM_AVAIL ) == PAPI_OK)
              && (numevents < PAPI_MPX_DEF_DEG) );

    printf("Found %d events\n", numevents);

    printf("Creating %d threads:\n", nthreads);

    threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
    if ( threads == NULL ) {
      test_fail(__FILE__, __LINE__, "malloc", 0);
    }

    /* Create the threads */
    for (i = 0; i < nthreads; i++) {
        ret = pthread_create(&threads[i], NULL, thread, NULL);
        if ( ret != 0 ) {
	  test_fail(__FILE__, __LINE__, "pthread_create", ret);
        }
    }

    /* Wait for thread completion */
    for (i = 0; i < nthreads; i++) {
        ret = pthread_join(threads[i],NULL);
        if ( ret != 0 ) {
	  test_fail(__FILE__, __LINE__, "pthread_join", ret);
        }
    }

    printf("Done.");
    
   PAPI_library_init(PAPI_VER_CURRENT);
   test_pass(__FILE__, NULL, 0);

   exit(0);
}
 

