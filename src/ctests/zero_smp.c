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
#if defined(sun) && defined(sparc)
#include <thread.h>
#elif defined(mips) && defined(sgi) && defined(unix)
#include <mpc.h>
#elif defined(_AIX)
#include <pthread.h>
#endif

void Thread(int t, int n)
{
  int retval, num_tests = 1, tmp;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;
  
  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  retval = PAPI_start(EventSet1);
  assert(retval >= PAPI_OK);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  do_flops(n);
  
  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  retval = PAPI_stop(EventSet1, values[0]);
  assert(retval >= PAPI_OK);

  remove_test_events(&EventSet1, mask1);

  printf("Thread 0x%x PAPI_FP_INS : \t%lld\n",t,
	 (values[0])[0]);
  printf("Thread 0x%x PAPI_TOT_CYC: \t%lld\n",t,
	 (values[0])[1]);

  free_test_space(values, num_tests);
  printf("Thread 0x%x Real usec   : \t%lld\n",t,
	 elapsed_us);
  printf("Thread 0x%x Real cycles : \t%lld\n",t,
	 elapsed_cyc);
}

int main()
{
  int i, rc;
  long long elapsed_us, elapsed_cyc;

  assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

#if defined(_AIX)
  assert(PAPI_thread_init(pthread_self, 0) == PAPI_OK);
#pragma ibm parallel_loop
#elif defined(sgi) && defined(mips)
  assert(PAPI_thread_init(mp_my_threadnum, 0) == PAPI_OK);
#pragma parallel
#pragma pfor local(i)
#elif defined(sun) && defined(sparc)
  assert(PAPI_thread_init(thr_self, 0) == PAPI_OK);
#pragma MP taskloop private(i)
#elif defined(__ALPHA) && defined(__osf__)
#else
#error "Architecture not included in this test file yet."
#endif
  for (i=1;i<3;i++)
    Thread(i,1000000*i);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  printf("Master real usec   : \t%lld\n",
	 elapsed_us);
  printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);

  exit(0);
}
