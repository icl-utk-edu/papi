/* This file performs the following test: profile for pthreads */

#include <pthread.h>
#include "papi_test.h"

extern int TESTS_QUIET; /* Declared in test_utils.c */

#define THR 1000000
#define FLOPS 1000000000

unsigned long length;
unsigned long my_start, my_end;

void *Thread(void *arg)
{
  int retval, num_tests = 1, i;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;
  unsigned short *profbuf;
  
/*  if ( 20000001 == *arg || 10000001 == *arg) TESTS_QUIET=1;*/

  profbuf = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf == NULL)
    exit(1);
  memset(profbuf,0x00,length*sizeof(unsigned short));

  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_profil(profbuf, length, my_start, 65536, 
		       EventSet1, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX);
  if (retval)
	test_fail(__FILE__,__LINE__,"PAPI_profil",retval);

  if((retval = PAPI_start(EventSet1))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_flops(*(int *)arg);
  
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

    printf("Test case: PAPI_profil() for pthreads\n");
    printf("----Profile buffer for Thread 0x%x---\n",(int)pthread_self());
    for (i=0;i<length;i++)
    {
      if (profbuf[i])
	printf("0x%x\t%d\n",(unsigned int)my_start + 2*i,profbuf[i]);
    }
  }
  for ( i=0;i<length;i++) 
	if ( profbuf[i] )  break;

  if ( i >= length )
	test_fail(__FILE__,__LINE__,"No information in buffers",1);
  free_test_space(values, num_tests);

  pthread_exit(NULL);

  return(NULL);
}

int main(int argc, char **argv)
{
  pthread_t e_th;
  pthread_t f_th;
  int flops1, flops2;
  int rc,retval;
  pthread_attr_t attr;
  long long elapsed_us, elapsed_cyc;
  const PAPI_exe_info_t *prginfo = NULL;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);
  if ( !TESTS_QUIET )
    if ((retval=PAPI_set_debug(PAPI_VERB_ECONT))!= PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);
  if ((retval=PAPI_thread_init((unsigned long(*)(void))(pthread_self),0))!=PAPI_OK){
      if (retval == PAPI_ESBSTR)
	test_skip(__FILE__,__LINE__,"PAPI_thread_init",retval);
      else
	test_fail(__FILE__,__LINE__,"PAPI_thread_init",retval);
  }
  if ((prginfo = PAPI_get_executable_info()) == NULL){
	retval=1;
	test_fail(__FILE__,__LINE__,"PAPI_get_executable_info",retval);
  }
  my_start = (unsigned long)prginfo->address_info.text_start;
  my_end =  (unsigned long)prginfo->address_info.text_end;
  length = my_end - my_start;

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

/*
  if ( !TESTS_QUIET ) flops1 = 10000000;
  else flops1 = 10000001;
*/
  flops1 = FLOPS;
  rc = pthread_create(&e_th, &attr, Thread, (void *)&flops1);
  if (rc){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"pthread_create",retval);
  }
/*
  if ( !TESTS_QUIET ) flops2 = 20000000;
  else flops2 = 20000001;
*/
  flops2 = FLOPS*2;

  rc = pthread_create(&f_th, &attr, Thread, (void *)&flops2);
  if (rc){
	retval=PAPI_ESYS;
	test_fail(__FILE__,__LINE__,"pthread_create",retval);
  }

  pthread_attr_destroy(&attr);
  pthread_join(f_th, NULL); 
  pthread_join(e_th, NULL);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  if (!TESTS_QUIET ){
    printf("Master real usec   : \t%lld\n",
	 elapsed_us);
    printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);
  }

  test_pass(__FILE__,NULL,0);
  pthread_exit(NULL);
  exit(1);
}

