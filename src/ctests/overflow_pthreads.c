/* This file performs the following test: overflow dispatch with pthreads

   - This tests the dispatch of overflow calls from PAPI. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)
     + PAPI_TOT_CYC

   - Set up overflow
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include <pthread.h>
#include "papi_test.h"

static int total = 0;
static const PAPI_hw_info_t *hw_info = NULL;

void handler(int EventSet, void *address, long_long overflow_vector, void * context)
{
   if (!TESTS_QUIET) {
#ifdef _CRAYT3E
      fprintf(stderr, "handler(%d ) Overflow at %x, thread 0x%x!\n",
              EventSet, address, PAPI_thread_id());
#else
      fprintf(stderr, "handler(%d) Overflow at %p, thread 0x%lx!\n",
              EventSet, address, PAPI_thread_id());
#endif
   }
   total++;
}

void *Thread(void *arg)
{
   int retval, num_tests = 1;
   int EventSet1=PAPI_NULL;
   int mask1, papi_event;
   int num_events1;
   long long **values;
   long long elapsed_us, elapsed_cyc;
   char event_name[PAPI_MAX_STR_LEN];

   if (!TESTS_QUIET) 
      printf("Thread 0x%x \n", (int) pthread_self());

   /* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
      PAPI_TOT_INS, depends on the availability of the event on the 
      platform */
   EventSet1 = add_two_events(&num_events1, &papi_event, hw_info, &mask1);

   values = allocate_test_space(num_tests, num_events1);

   elapsed_us = PAPI_get_real_usec();

   elapsed_cyc = PAPI_get_real_cyc();

   if ((retval = PAPI_overflow(EventSet1, papi_event, THRESHOLD, 0, handler))
                 != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   /* start_timer(1); */
   if ((retval = PAPI_start(EventSet1)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(*(int *) arg);

   if ((retval = PAPI_stop(EventSet1, values[0])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   elapsed_us = PAPI_get_real_usec() - elapsed_us;

   elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

   if ((retval = PAPI_overflow(EventSet1, papi_event, 0, 0, NULL)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   remove_test_events(&EventSet1, mask1);

   if ((retval = PAPI_event_code_to_name(papi_event, event_name)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   if (!TESTS_QUIET) {
      printf("Thread 0x%x %s : \t%lld\n", (int) pthread_self(), 
                     event_name, (values[0])[0]);
      printf("Thread 0x%x PAPI_TOT_CYC: \t%lld\n", (int) pthread_self(), (values[0])[1]);
      printf("Thread 0x%x Real usec   : \t%lld\n", (int) pthread_self(), elapsed_us);
      printf("Thread 0x%x Real cycles : \t%lld\n", (int) pthread_self(), elapsed_cyc);
   }
   free_test_space(values, num_tests);
   pthread_exit(NULL);
   return (NULL);
}

int main(int argc, char **argv)
{
   pthread_t id[NUM_THREADS];
   int flops[NUM_THREADS];
   int i, rc, retval;
   pthread_attr_t attr;
   long long elapsed_us, elapsed_cyc;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if ((retval =
        PAPI_thread_init((unsigned long (*)(void)) (pthread_self))) != PAPI_OK) {
      if (retval == PAPI_ESBSTR)
         test_skip(__FILE__, __LINE__, "PAPI_thread_init", retval);
      else
         test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
   }

   elapsed_us = PAPI_get_real_usec();

   elapsed_cyc = PAPI_get_real_cyc();

   pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
   retval = pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
   if (retval != 0)
      test_skip(__FILE__, __LINE__, "pthread_attr_setscope", retval);
#endif

   for (i = 0; i < NUM_THREADS; i++) {
     flops[i] = NUM_FLOPS * (i+1);
      rc = pthread_create(&id[i], &attr, Thread, (void *)&flops[i]);
      if (rc)
         test_fail(__FILE__, __LINE__, "pthread_create", PAPI_ESYS);
   }
   for (i = 0; i < NUM_THREADS; i++)
      pthread_join(id[i], NULL);

   pthread_attr_destroy(&attr);

   elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

   elapsed_us = PAPI_get_real_usec() - elapsed_us;

   if (!TESTS_QUIET) {
      printf("Master real usec   : \t%lld\n", elapsed_us);
      printf("Master real cycles : \t%lld\n", elapsed_cyc);
   }

   test_pass(__FILE__, NULL, 0);
   pthread_exit(NULL);
   exit(1);
}
