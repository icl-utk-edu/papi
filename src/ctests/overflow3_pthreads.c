/* This file performs the following test: overflow dispatch with 1 extraneous pthread

*/

#include <pthread.h>
#include "papi_test.h"

#define FLOPS 10000000
#define READS 4000

int total = 0;

void handler(int EventSet, int EventCode, int EventIndex, long long *values, int *threshold, void *context)
{
  if ( !TESTS_QUIET ) {
#ifdef _CRAYT3E
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x, thread 0x%x!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context),PAPI_thread_id());
#else
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %p) Overflow at %p, thread 0x%lux!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context),PAPI_thread_id());
#endif
  }
  total++;
}

void * thread_fn( void * dummy )
{
	while(1)
	  {
	    do_flops(FLOPS);
	    do_reads(READS);
	  }
}

void mainloop(int arg)
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

  if((retval = PAPI_overflow(EventSet1, PAPI_FP_INS, THRESHOLD, 0, handler))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_overflow",retval);
  /* start_timer(1); */
  if((retval = PAPI_start(EventSet1))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_flops(arg);
  
  if((retval = PAPI_stop(EventSet1, values[0]))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  remove_test_events(&EventSet1, mask1);

  if ( !TESTS_QUIET ){
    printf("Thread 0x%x PAPI_FP_INS : \t%lld\n",(int)pthread_self(),
	 (values[0])[0]);
    printf("Thread 0x%x PAPI_TOT_CYC: \t%lld\n",(int)pthread_self(),
	 (values[0])[1]);
    printf("Thread 0x%x Real usec   : \t%lld\n",(int)pthread_self(),
	 elapsed_us);
    printf("Thread 0x%x Real cycles : \t%lld\n",(int)pthread_self(),
	 elapsed_cyc);
  }
  free_test_space(values, num_tests);
}

int main(int argc, char **argv)
{
  pthread_t e_th;
  int rc,retval;
  pthread_attr_t attr;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ( !TESTS_QUIET )
    if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
#endif

  rc = pthread_create(&e_th, &attr, thread_fn, NULL);
  if (rc){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"pthread_create",retval);
  }

  mainloop(FLOPS);

  pthread_attr_destroy(&attr);

  test_pass(__FILE__,NULL,0);
  exit(1);
}

