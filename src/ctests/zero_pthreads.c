/* This file performs the following test: start, stop and timer
functionality for 2 slave pthreads

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

#include <pthread.h>

#include "papi_test.h"

#ifdef NO_FLOPS
  #define PAPI_EVENT 		PAPI_TOT_INS
  #define MASK				MASK_TOT_INS | MASK_TOT_CYC
#else
  #define PAPI_EVENT 		PAPI_FP_INS
  #define MASK				MASK_FP_INS | MASK_TOT_CYC
#endif

int TESTS_QUIET=0; /* Tests in Verbose mode? */

void *Thread(void *arg)
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

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(*(int *)arg);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  remove_test_events(&EventSet1, mask1);

  if ( !TESTS_QUIET ){
  printf("Thread 0x%x %-12s : \t%lld\n",(int)pthread_self(), event_name,
	 (values[0])[0]);
  printf("Thread 0x%x PAPI_TOT_CYC : \t%lld\n",(int)pthread_self(),
	 (values[0])[1]);
  printf("Thread 0x%x Real usec    : \t%lld\n",(int)pthread_self(),
	 elapsed_us);
  printf("Thread 0x%x Real cycles  : \t%lld\n",(int)pthread_self(),
	 elapsed_cyc);
  }

  free_test_space(values, num_tests);

  pthread_exit(NULL);
  return(NULL);
}

int main(int argc, char **argv) 
{
  pthread_t e_th;
  pthread_t f_th;
  int flops1, flops2;
  int retval, rc;
  pthread_attr_t attr;
  long long elapsed_us, elapsed_cyc;

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  retval = PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0);
  if ( retval != PAPI_OK ) {
	if ( retval == PAPI_ESBSTR )
	   test_pass(__FILE__,NULL,0);
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
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
#endif

  flops1 = 1000000;
  rc = pthread_create(&e_th, &attr, Thread, (void *)&flops1);
  if (rc){
    retval = PAPI_ESYS;
    test_fail(__FILE__,__LINE__,"pthread_create",retval);
  }

  flops2 = 2000000;
  rc = pthread_create(&f_th, &attr, Thread, (void *)&flops2);
  if (rc){
    retval = PAPI_ESYS;
    test_fail(__FILE__,__LINE__,"pthread_create",retval);
  }

  pthread_attr_destroy(&attr);
  pthread_join(f_th, NULL);
  pthread_join(e_th, NULL);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  if ( !TESTS_QUIET ) {
    printf("Master real usec   : \t%lld\n",
	 elapsed_us);
    printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);
  }

  pthread_exit(NULL);
  test_pass(__FILE__,NULL,0);
}
