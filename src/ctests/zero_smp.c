/* $Id$ */

/* This file performs the following test: start, stop and timer
functionality for 2 slave native SMP threads

   - It attempts to use the following two counters. It may use less
depending on hardware counter resource limitations. These are counted
in the default counting domain and default granularity, depending on
the platform. Usually this is the user domain (PAPI_DOM_USER) and
thread context (PAPI_GRN_THR).

     + PAPI_FP_INS
     + PAPI_TOT_CYC

Each of 2 slave pthreads:
   - Get cyc.
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
   - Get cyc.

Master pthread:
   - Get us.
   - Get cyc.
   - Fork threads
   - Wait for threads to exit
   - Get us.
   - Get cyc.
*/


#include "papi_test.h"

#if defined(sun) && defined(sparc)
#include <thread.h>
#elif defined(mips) && defined(sgi) && defined(unix)
#include <mpc.h>
#elif defined(_AIX)
#include <pthread.h>
#endif

#ifdef NO_FLOPS
  #define PAPI_EVENT 		PAPI_TOT_INS
  #define MASK				MASK_TOT_INS | MASK_TOT_CYC
#else
  #define PAPI_EVENT 		PAPI_FP_INS
  #define MASK				MASK_FP_INS | MASK_TOT_CYC
#endif

int TESTS_QUIET=0; /* Tests in Verbose mode? */

void Thread(int t, int n)
{
  int retval, num_tests = 1;
  int EventSet1;
  int mask1 = MASK;
  int num_events1;
  long_long **values;
  long_long elapsed_us, elapsed_cyc;
  char event_name[PAPI_MAX_STR_LEN];

  retval = PAPI_event_code_to_name(PAPI_EVENT, event_name);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  do_flops(n);
  
  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  remove_test_events(&EventSet1, mask1);

  if ( !TESTS_QUIET ) {
    printf("Thread 0x%x %-12s : \t%lld\n",t, event_name,
	 (values[0])[0]);
    printf("Thread 0x%x PAPI_TOT_CYC : \t%lld\n",t,
	 (values[0])[1]);
  }

  free_test_space(values, num_tests);
  if ( !TESTS_QUIET ) {
  printf("Thread 0x%x Real usec    : \t%lld\n",t,
	 elapsed_us);
  printf("Thread 0x%x Real cycles  : \t%lld\n",t,
	 elapsed_cyc);
  }
}

int main(int argc, char **argv) 
{
  int i, retval;
  long long elapsed_us, elapsed_cyc;

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

#if defined(_AIX)
  retval = PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0);
  if (retval != PAPI_OK){
      if (retval == PAPI_ESBSTR)
	    test_skip(__FILE__, __LINE__, "PAPI_thread_init", retval);
      else
	    test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
  }
#pragma ibm parallel_loop
#elif defined(sgi) && defined(mips)
  retval = PAPI_thread_init((unsigned long (*)(void))(mp_my_threadnum), 0);
  if (retval != PAPI_OK){
	    test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
  }
#pragma parallel
#pragma local(i)
#pragma pfor
#elif defined(sun) && defined(sparc)
  retval = PAPI_thread_init((unsigned long (*)(void))(thr_self), 0);
  if (retval != PAPI_OK){
	    test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
  }
#pragma MP taskloop private(i)
#else
#error "Architecture not included in this test file yet."
#endif
  for (i=1;i<3;i++)
    Thread(i,10000000*i);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  if ( !TESTS_QUIET ) {
  printf("Master real usec   : \t%lld\n",
	 elapsed_us);
  printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);
  }
  test_pass(__FILE__,NULL,0);
}
