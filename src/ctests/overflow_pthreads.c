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
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <sys/time.h>
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

#define THRESHOLD 100000

int total = 0;

static void sigprof_handler(int sig)
{
  fprintf(stderr,"sigprof_handler(), PID %d TID %d\n",
	  getpid(),(int)pthread_self());
}

static void start_sig(void)
{
  struct sigaction action;

  memset(&action,0x00,sizeof(struct sigaction));
  action.sa_handler = (void(*)(int))sigprof_handler;
  if (sigaction(SIGPROF, &action, NULL) == -1)
    exit(1);
}

void start_timer(int milliseconds)
{
  struct itimerval value;

  start_sig();
  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  if (setitimer(ITIMER_PROF, &value, NULL) == -1)
    exit(1);
}

void handler(int EventSet, int EventCode, int EventIndex, long long *values, int *threshold, void *context)
{
#ifdef _CRAYT3E
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x, thread %lu!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context),PAPI_thread_id());
#else
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %p) Overflow at %p, thread %lu!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context),PAPI_thread_id());
#endif
  total++;
}

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

  retval = PAPI_overflow(EventSet1, PAPI_FP_INS, THRESHOLD, 0, handler);
  if (retval != PAPI_OK)
    exit(1);

  /* start_timer(1); */
  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(*(int *)arg);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK)
    exit(1);

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

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if (PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0) != PAPI_OK)
    exit(1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
#endif

  flops1 = 10000000;
  rc = pthread_create(&e_th, &attr, Thread, (void *)&flops1);
  if (rc)
    exit(1);

  flops2 = 20000000;
  rc = pthread_create(&f_th, &attr, Thread, (void *)&flops2);
  if (rc)
    exit(1);

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

