/* 
* File:    multiplex1_pthreads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file tests the multiplex pthread functionality */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi_test.h"
#include "test_utils.h"

#define FLOPS 4000000
#define READS 4000
#define NUM 10
#define NUM_THREADS 4
#define SUCCESS 1
#define FAILURE 0
/*#define METRIC PAPI_FP_INS */
#define METRIC PAPI_L1_DCM
extern void do_flops(int);
extern void do_reads(int);

extern int TESTS_QUIET;         /* Declared in test_utils.c */

unsigned int preset_PAPI_events[PAPI_MPX_DEF_DEG] = {
   PAPI_FP_INS, PAPI_TOT_CYC, PAPI_L1_ICM, PAPI_L1_DCM, 0,
};
static unsigned int PAPI_events[PAPI_MPX_DEF_DEG] = { 0, };
static int num_PAPI_events = 0;

void init_papi_pthreads(void)
{
   int retval;
   const unsigned int *inev;
   unsigned int *outev;
   const PAPI_hw_info_t *hw_info = PAPI_get_hardware_info();

   /* Initialize the library */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if(!(strncmp(hw_info->model_string, "AMD K7", 6))) {
      preset_PAPI_events[0] = PAPI_TOT_INS;
   }
   /* String results can vary, so we're looking at the important portion */
   if(!(strncmp(hw_info->model_string, "UltraSPARC", 10)) && !(strncmp(hw_info->vendor_string, "SUN", 3))) {
      preset_PAPI_events[0] = PAPI_TOT_INS;
      preset_PAPI_events[2] = PAPI_L1_ICA;
      preset_PAPI_events[3] = PAPI_L1_ICH;

   }
   if(!(strcmp(hw_info->model_string, "Intel Pentium 4")) || !(strcmp(hw_info->model_string, "POWER4"))) {
      preset_PAPI_events[2] = PAPI_L1_LDM;
   }
   /* Investigate the set of candidate events */
   num_PAPI_events = 0;
   for (inev = preset_PAPI_events, outev = PAPI_events; *inev; inev++) {
      if (PAPI_query_event(*inev) == PAPI_OK) {
         *outev++ = *inev;
         num_PAPI_events++;
      }
   }
   /* Enable multiplexing support */

   if ((retval = PAPI_multiplex_init()) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", retval);

   /* Turn on automatic error reporting */

   if (!TESTS_QUIET)
      if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);

   /* Turn on thread support in PAPI */

   if ((retval =
        PAPI_thread_init((unsigned long (*)(void)) (pthread_self))) != PAPI_OK) {
      if (retval == PAPI_ESBSTR)
         test_skip(__FILE__, __LINE__, "PAPI_thread_init", retval);
      else
         test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
   }

}

int do_pthreads(void *(*fn) (void *))
{
   int i, rc, retval;
   pthread_attr_t attr;
   pthread_t id[NUM_THREADS];

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
      rc = pthread_create(&id[i], &attr, fn, NULL);
      if (rc)
         return (FAILURE);
   }
   for (i = 0; i < NUM_THREADS; i++)
      pthread_join(id[i], NULL);

   pthread_attr_destroy(&attr);

   return (SUCCESS);
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

void *case1_pthreads(void *arg)
{
   int retval, i, EventSet = PAPI_NULL;
   long long values[2];

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i = 0; i < 2; i++) {
      if ((retval = PAPI_add_event(EventSet, PAPI_events[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < NUM; i++) {
      do_flops(FLOPS);
      do_reads(READS);
   }

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("case1 thread %4x:", (unsigned) pthread_self());
      test_print_event_header("", EventSet);
      printf("case1 thread %4x:", (unsigned) pthread_self());
      printf(TAB2, "", values[0], values[1]);
   }
   if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)   /* JT */
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   return ((void *) SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before adding events */

void *case2_pthreads(void *arg)
{
   int retval, i, EventSet = PAPI_NULL;
   long long values[2];

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((retval = PAPI_set_multiplex(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   for (i = 0; i < 2; i++) {
      if ((retval = PAPI_add_event(EventSet, PAPI_events[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < NUM; i++) {
      do_flops(FLOPS);
      do_reads(READS);
   }

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("case2 thread %4x:", (unsigned) pthread_self());
      test_print_event_header("", EventSet);
      printf("case2 thread %4x:", (unsigned) pthread_self());
      printf(TAB2, "", values[0], values[1]);
   }

   if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)   /* JT */
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   return ((void *) SUCCESS);
}

/* Tests that PAPI_set_multiplex() works after adding events */

void *case3_pthreads(void *arg)
{
   int retval, i, EventSet = PAPI_NULL;
   long long values[2];

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i = 0; i < 2; i++) {
      if ((retval = PAPI_add_event(EventSet, PAPI_events[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   if ((retval = PAPI_set_multiplex(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < NUM; i++) {
      do_flops(FLOPS);
      do_reads(READS);
   }

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("case3 thread %4x:", (unsigned) pthread_self());
      test_print_event_header("", EventSet);
      printf("case3 thread %4x:", (unsigned) pthread_self());
      printf(TAB2, "", values[0], values[1]);
   }

   if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)   /* JT */
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   return ((void *) SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before/after adding events */

void *case4_pthreads(void *arg)
{
   int retval, i, EventSet = PAPI_NULL;
   long long values[4];

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i = 0; i < 2; i++) {
      if ((retval = PAPI_add_event(EventSet, PAPI_events[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   if ((retval = PAPI_set_multiplex(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   for (i = 2; i < num_PAPI_events; i++) {
      if ((retval = PAPI_add_event(EventSet, PAPI_events[i])) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
   }

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < NUM; i++) {
      do_flops(FLOPS);
      do_reads(READS);
   }

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET) {
      printf("case4 thread %4x:", (unsigned) pthread_self());
      test_print_event_header("", EventSet);
      printf("case4 thread %4x:", (unsigned) pthread_self());
      for( i = 0; i < num_PAPI_events;i++) printf(" %12lld", values[i]);
      printf("\n");
   }

   if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)   /* JT */
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   return ((void *) SUCCESS);
}

int case1(void)
{
   int retval;

   init_papi_pthreads();

   retval = do_pthreads(case1_pthreads);

   PAPI_shutdown();

   return (retval);
}

int case2(void)
{
   int retval;

   init_papi_pthreads();

   retval = do_pthreads(case2_pthreads);

   PAPI_shutdown();

   return (retval);
}

int case3(void)
{
   int retval;

   init_papi_pthreads();

   retval = do_pthreads(case3_pthreads);

   PAPI_shutdown();

   return (retval);
}

int case4(void)
{
   int retval;

   init_papi_pthreads();

   retval = do_pthreads(case4_pthreads);

   PAPI_shutdown();

   return (retval);
}

int main(int argc, char **argv)
{
   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

/* Skip Alpha till multiplex is fixed. -KSL */
#if defined(__ALPHA) && defined(__osf__)
   test_pass(__FILE__, NULL, 0);
#endif

   if (!TESTS_QUIET) {
      printf("%s: Using %d threads, %d iterations\n\n", argv[0], NUM_THREADS, NUM);
      printf("case1: Does PAPI_multiplex_init() not break regular operation?\n");
   }
  case1();
  if(!TESTS_QUIET )
     printf("case2: Does setmpx/add work?\n");

   case2();
   if (!TESTS_QUIET)
      printf("case3: Does add/setmpx work?\n");
   case3();
   if (!TESTS_QUIET)
      printf("case4: Does add/setmpx/add work?\n");
   case4();

   PAPI_library_init(PAPI_VER_CURRENT);
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
