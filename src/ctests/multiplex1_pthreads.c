/* 
* File:    multiplex1_pthreads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file tests the multiplex pthread functionality */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"

#define NUM 10
#define NUM_THREADS 4
#define SUCCESS 1
#define FAILURE 0

extern void do_flops(int);
extern void do_reads(int);

int TESTS_QUIET = 0;

void init_papi_pthreads(void)
{
  int retval;

  /* Initialize the library */

  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  /* Enable multiplexing support */

  if( (retval = PAPI_multiplex_init()) != PAPI_OK )
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  /* Turn on automatic error reporting */

  if ( !TESTS_QUIET )
    if((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK )
       test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  /* Turn on thread support in PAPI */

  if ((retval=PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0)) != PAPI_OK){
        test_fail(__FILE__,__LINE__,"PAPI_thread_init",retval);
  }
}

int do_pthreads(void *(*fn)(void *))
{
  int i, rc;
  pthread_attr_t attr;
  pthread_t id[NUM_THREADS];

  pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
#endif

  for (i=0;i<NUM_THREADS;i++)
    {
      rc = pthread_create(&id[i], &attr, fn, NULL);
      if (rc)
	return(FAILURE);
    }
  for (i=0;i<NUM_THREADS;i++)
    pthread_join(id[i], NULL);

  pthread_attr_destroy(&attr);

  return(SUCCESS);
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

void *case1_pthreads(void *arg) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK )
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  if ((retval = PAPI_add_event(&EventSet, PAPI_FP_INS))!=PAPI_OK )
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  if((retval = PAPI_stop(EventSet, values))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET )
     printf("case1 thread %x: %lld %lld\n",(unsigned)pthread_self(),values[0],values[1]);
  if((retval = PAPI_cleanup_eventset(&EventSet)) !=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  return((void *)SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before adding events */

void *case2_pthreads(void *arg) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  if((retval = PAPI_create_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  if((retval = PAPI_set_multiplex(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_FP_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  if((retval = PAPI_stop(EventSet, values))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET )
     printf("case2 thread %x: %lld %lld\n",(unsigned)pthread_self(),values[0],values[1]);
  if((retval = PAPI_cleanup_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  return((void *)SUCCESS);
}

/* Tests that PAPI_set_multiplex() works after adding events */

void *case3_pthreads(void *arg) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  if((retval = PAPI_create_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_FP_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_set_multiplex(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  if((retval = PAPI_stop(EventSet, values))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) 
      printf("case3 thread %x: %lld %lld\n",(unsigned)pthread_self(),values[0],values[1]);
  if((retval = PAPI_cleanup_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  return((void *)SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before/after adding events */

void *case4_pthreads(void *arg) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[4];

  if((retval = PAPI_create_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_FP_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_set_multiplex(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

#if (defined(i386) && defined(linux)) || (defined(_POWER) && defined(_AIX) || defined(mips))
  if((retval = PAPI_add_event(&EventSet, PAPI_L1_DCM))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_L1_ICM))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif  defined(sparc) && defined(sun)
  if((retval = PAPI_add_event(&EventSet, PAPI_LD_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if((retval = PAPI_add_event(&EventSet, PAPI_SR_INS))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(__ALPHA) && defined(__osf__)
  retval = PAPI_add_event(&EventSet, PAPI_BR_CN);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
  retval = PAPI_add_event(&EventSet, PAPI_RES_STL);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#else
#error "Architecture not ported yet"
#endif

  if ((retval=PAPI_start(EventSet)) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  if((retval = PAPI_stop(EventSet, values))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET )
     printf("case4 thread %x: %lld %lld %lld %lld\n",(unsigned)pthread_self(),values[0],values[1],values[2],values[3]);
  if((retval = PAPI_cleanup_eventset(&EventSet))!=PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  return((void *)SUCCESS);
}

int case1(void) 
{
  int retval;

  init_papi_pthreads();

  retval = do_pthreads(case1_pthreads);

  PAPI_shutdown();

  return(retval);
}

int case2(void) 
{
  int retval;

  init_papi_pthreads();

  retval = do_pthreads(case2_pthreads);

  PAPI_shutdown();

  return(retval);
}

int case3(void) 
{
  int retval;

  init_papi_pthreads();

  retval = do_pthreads(case3_pthreads);

  PAPI_shutdown();

  return(retval);
}

int case4(void) 
{
  int retval;

  init_papi_pthreads();

  retval = do_pthreads(case4_pthreads);

  PAPI_shutdown();

  return(retval);
}

int main(int argc, char **argv)
{
  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if(!TESTS_QUIET ) {
   printf("%s: Using %d threads, %d iterations\n\n",argv[0],NUM_THREADS,NUM);
   printf("case1: Does PAPI_multiplex_init() not break regular operation?\n");
  }
  case1();
  if(!TESTS_QUIET )
     printf("case2: Does setmpx/add work?\n");
  case2();
  if(!TESTS_QUIET )
     printf("case3: Does add/setmpx work?\n");
  case3();
  if(!TESTS_QUIET )
     printf("case4: Does add/setmpx/add work?\n");
  case4();

  PAPI_library_init( PAPI_VER_CURRENT );
  test_pass(__FILE__,NULL,0);
}

