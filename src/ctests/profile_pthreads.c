/* This file performs the following test: profile for pthreads */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#undef NDEBUG
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

#define THR 1000000
unsigned long length;
unsigned long start, end;

void *Thread(void *arg)
{
  int retval, num_tests = 1, i;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;
  unsigned short *profbuf;
  
  profbuf = (unsigned short *)malloc(length*sizeof(unsigned short));
  if (profbuf == NULL)
    exit(1);
  memset(profbuf,0x00,length*sizeof(unsigned short));

  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_profil(profbuf, length, start, 65536, 
		       EventSet1, PAPI_FP_INS, THR, PAPI_PROFIL_POSIX);
  if (retval)
    exit(retval);

  retval = PAPI_start(EventSet1);
  if (retval)
    exit(retval);

  do_flops(*(int *)arg);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval)
    exit(retval);

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

  printf("Test case: PAPI_profil() for pthreads\n");
  printf("----Profile buffer for Thread 0x%x---\n",(int)pthread_self());
  for (i=0;i<length;i++)
    {
      if (profbuf[i])
	printf("0x%x\t%d\n",(unsigned int)start + 2*i,profbuf[i]);
    }

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
  const PAPI_exe_info_t *prginfo = NULL;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if (PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0) != PAPI_OK)
    exit(1);

  if ((prginfo = PAPI_get_executable_info()) == NULL)
    exit(1);
  start = (unsigned long)prginfo->text_start;
  end =  (unsigned long)prginfo->text_end;
  length = end - start;

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

