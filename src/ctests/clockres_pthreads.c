#include <pthread.h>
#include "papi_test.h"

#if 0
#include "libperfctr.h"
#endif

extern int TESTS_QUIET; /* Declared in test_utils.c */
extern void clockcore(void); /* Declared in clockcore.c */

void pthread_main(void *arg)
{
#if 0
  struct vperfctr *ptr = vperfctr_open();
  long long *lcyca;
  int i, iters = atoi(getenv("ITERS"));
  lcyca = (long long *)malloc(sizeof(long long)*iters);

  for (i=0;i<iters;i++)
    {
      lcyca[i] = vperfctr_read_tsc(ptr);
    }

  for (i=1;i<iters;i++)
    if (lcyca[i] - lcyca[i-1] < 0)
      abort();
#endif
      
  clockcore();
}

int main(int argc, char **argv)
{
  pthread_t t1, t2, t3, t4;
  int retval;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  
  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  retval = PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0);
  if ( retval != PAPI_OK ) 
     if (retval == PAPI_ESBSTR)
           test_skip(__FILE__, __LINE__, "PAPI_thread_init", retval);
     else
	   test_fail(__FILE__, __LINE__, "PAPI_thread_init", retval);

  if ( !TESTS_QUIET ) {
  printf("Test case: Clock latency and resolution.\n");
  printf("Note: Virtual timers are proportional to # CPU's.\n");
  printf("-------------------------------------------------\n");
  }

  pthread_create(&t1,NULL,pthread_main,NULL); 
  pthread_create(&t2,NULL,pthread_main,NULL); 
  pthread_create(&t3,NULL,pthread_main,NULL); 
  pthread_create(&t4,NULL,pthread_main,NULL); 
  pthread_main(NULL);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);
  pthread_join(t4, NULL);

  test_pass(__FILE__,NULL,0);
}

