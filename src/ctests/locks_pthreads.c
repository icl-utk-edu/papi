/* This file checks to make sure the locking mechanisms work correctly on the platform.
 * Platforms where the locking mechanisms are not implemented or are incorrectly implemented
 * will fail.  -KSL
 */

#include <pthread.h>
#include "papi_test.h"

int count;

void *Slave(void *arg)
{
   int i;

   for (i = 0; i < NUM_ITERS; i++) {
     PAPI_lock(PAPI_USR1_LOCK);
     count++;
     PAPI_unlock(PAPI_USR1_LOCK);
   }
   pthread_exit(NULL);
}


int main(int argc, char **argv)
{
   pthread_t slaves[MAX_THREADS];
   int rc, i, nthr;
   int retval;
  const PAPI_hw_info_t *hwinfo = NULL;

#if defined(__ALPHA) && defined(__osf__)
   test_skip(__FILE__, __LINE__, "thread support not available on this platform!", PAPI_ESBSTR);
#endif

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);
   
   retval = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
   if (retval != PAPI_OK) {
      if (retval == PAPI_ESBSTR)
         test_skip(__FILE__, __LINE__, "PAPI_thread_init", retval);
      else
         test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);
   }

   if (hwinfo->ncpu > MAX_THREADS) 
     nthr = MAX_THREADS;
   else
     nthr = hwinfo->ncpu;

   printf("Creating %d threads\n", nthr);

   for (i=0;i<nthr;i++)
     {
       rc = pthread_create(&slaves[i], NULL, Slave, NULL);
       if (rc) {
	 retval = PAPI_ESYS;
	 test_fail(__FILE__, __LINE__, "pthread_create", retval);
       }
     }

   for (i=0;i<nthr;i++)
     {
       pthread_join(slaves[i], NULL);
     }

   printf("Expected: %d Received: %d\n", nthr*NUM_ITERS, count);
   if (nthr*NUM_ITERS != count)
      test_fail(__FILE__, __LINE__, "Thread Locks", 1);

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
