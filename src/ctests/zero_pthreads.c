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
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#undef NDEBUG
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

void *Thread(void *arg)
{
  int retval, num_tests = 1;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;
  
  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_start(EventSet1);
  assert(retval >= PAPI_OK);

  do_flops(*(int *)arg);
  
  retval = PAPI_stop(EventSet1, values[0]);
  assert(retval >= PAPI_OK);

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  remove_test_events(&EventSet1, mask1);

  printf("Thread 0x%x PAPI_FP_INS : \t%lld\n",(int)pthread_self(),
	 (values[0])[0]);
  printf("Thread 0x%x PAPI_TOT_CYC: \t%lld\n",(int)pthread_self(),
	 (values[0])[1]);
  printf("Thread 0x%x Real usec   : \t%lld\n",(int)pthread_self(),
	 elapsed_us);
  printf("Thread 0x%x Real cycles : \t%lld\n",(int)pthread_self(),
	 elapsed_cyc);

  free_test_space(values, num_tests);

  pthread_exit(NULL);
  return(NULL);
}

int main()
{
  pthread_t e_th;
  pthread_t f_th;
  int flops1, flops2;
  int rc;
  pthread_attr_t attr;
  long long elapsed_us, elapsed_cyc;

  assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);

  assert(PAPI_thread_init(pthread_self, 0) == PAPI_OK);

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
  if (rc)
    exit(-1);

  flops2 = 2000000;
  rc = pthread_create(&f_th, &attr, Thread, (void *)&flops2);
  if (rc)
    exit(-1);

  pthread_attr_destroy(&attr);
  pthread_join(f_th, NULL);
  pthread_join(e_th, NULL);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  printf("Master real usec   : \t%lld\n",
	 elapsed_us);
  printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);

  pthread_exit(NULL);
  exit(0);
}
