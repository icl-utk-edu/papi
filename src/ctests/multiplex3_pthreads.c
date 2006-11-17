/* 
* File:    multiplex3_pthreads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    John May
*          johnmay@llnl.gov
*/

/* This file tests the multiplex functionality when there are
 * threads in which the application isn't calling PAPI (and only
 * one thread that is calling PAPI.)
 */

#include <pthread.h>
#include "papi_test.h"

/* A thread function that does nothing forever, while the other
 * tests are running.
 */
void *thread_fn(void *dummy)
{
   while (1){
      do_both(NUM_ITERS);
   }
}

/* Runs a bunch of multiplexed events */

void mainloop(int arg)
{
  int allvalid = 1;
  long long *values;
  int EventSet = PAPI_NULL;
  int retval, i, j = 2;
  PAPI_event_info_t pset;
  const PAPI_hw_info_t *hw_info;

   /* Initialize the library */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   retval = PAPI_multiplex_init();
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   retval = PAPI_set_multiplex(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
   if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   if (!TESTS_QUIET) {
      printf("Added %s\n", "PAPI_TOT_INC");
   }

   retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
   if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   if (!TESTS_QUIET) {
      printf("Added %s\n", "PAPI_TOT_CYC");
   }

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      retval = PAPI_get_event_info(i | PAPI_PRESET_MASK, &pset);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_get_event_info", retval);

      if (pset.count) {
      printf("Adding %s\n", pset.symbol);

      retval = PAPI_add_event(EventSet, pset.event_code);
      if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
	test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      
      if (retval == PAPI_OK) {
	printf("Added %s\n", pset.symbol);
      } else {
	printf("Could not add %s\n", pset.symbol);
      }

      if (retval == PAPI_OK) {
	if (++j >= MAX_TO_ADD)
	  break;
      }
   }
   }

   values = (long long *) malloc(MAX_TO_ADD * sizeof(long long));
   if (values == NULL)
      test_fail(__FILE__, __LINE__, "malloc", 0);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(arg);
   do_misses(1, 1024*1024*4);

   retval = PAPI_stop(EventSet, values);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   test_print_event_header("multiplex3_pthreads:\n", EventSet);
   for (i = 0; i < MAX_TO_ADD; i++) {
     printf(ONENUM, values[i]);
     if (values[i] == 0)
       allvalid = 0;
   }
   printf("\n");
   if (!allvalid)
      test_fail(__FILE__, __LINE__, "one or more counter registered no counts",1);

   retval = PAPI_cleanup_eventset(EventSet);    /* JT */
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   retval = PAPI_destroy_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

   free(values);
   PAPI_shutdown();
}

int main(int argc, char **argv)
{
   int i, rc, retval;
   pthread_t id[NUM_THREADS];
   pthread_attr_t attr;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   printf("%s: Using %d threads\n\n", argv[0], NUM_THREADS);
   printf("Does non-threaded multiplexing work with extraneous threads present?\n");

   /* Create a bunch of unused pthreads, to simulate threads created
    * by the system that the user doesn't know about.
    */
   pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
   retval = pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
   if (retval != 0)
      test_skip(__FILE__, __LINE__, "pthread_attr_setscope", retval);
#endif

#ifdef PPC64
   sigset_t sigprof;
   sigemptyset(&sigprof);
   sigaddset(&sigprof, SIGPROF);
   retval = sigprocmask(SIG_BLOCK, &sigprof, NULL);
   if (retval != 0)
           test_fail(__FILE__, __LINE__, "sigprocmask SIG_BLOCK", retval);
#endif

   for (i = 0; i < NUM_THREADS; i++) {
      rc = pthread_create(&id[i], &attr, thread_fn, NULL);
      if (rc)
         test_fail(__FILE__, __LINE__, "pthread_create", rc);
   }
   pthread_attr_destroy(&attr);

#ifdef PPC64
   retval = sigprocmask(SIG_UNBLOCK, &sigprof, NULL);
   if (retval != 0)
           test_fail(__FILE__, __LINE__, "sigprocmask SIG_UNBLOCK", retval);
#endif

   mainloop(NUM_ITERS);

   test_pass(__FILE__, NULL, 0);
   exit(0);
}
