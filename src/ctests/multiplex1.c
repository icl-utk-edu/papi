/* 
* File:    multiplex.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file tests the multiplex functionality, originally developed by 
   John May of LLNL. */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"

#define NUM 10
#define SUCCESS 1

extern void do_flops(int);
extern void do_reads(int);

extern int TESTS_QUIET; /* Declared in test_utils.c */
static int PAPI_event;	/* Event to use in all cases; initialized in init_papi() */

void init_papi(void)
{
  int retval;

  /* Initialize the library */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  /* Turn on automatic error reporting */
  retval = PAPI_set_debug(PAPI_VERB_ECONT);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  /* query and set up the right instruction to monitor */
  if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) PAPI_event = PAPI_FP_INS;
  else PAPI_event = PAPI_TOT_INS;
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

int case1() 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_add_event(&EventSet, PAPI_event);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET )
     printf("case1: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before adding events */

int case2() 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  retval = PAPI_add_event(&EventSet, PAPI_event);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

#ifdef _CRAYT3E
  retval = PAPI_add_event(&EventSet, PAPI_TOT_IIS);
#else
  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
#endif
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) 
     printf("case2: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works after adding events */

int case3() 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_add_event(&EventSet, PAPI_event);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

#ifdef _CRAYT3E
  retval = PAPI_add_event(&EventSet, PAPI_TLB_DM);
#else
  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
#endif
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) 
     printf("case3: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}
/* Tests that PAPI_set_multiplex() works before adding events */

/* Tests that PAPI_add_event() works after
   PAPI_add_event()/PAPI_set_multiplex() */

int case4() 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[4];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_add_event(&EventSet, PAPI_event);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

#ifdef _CRAYT3E
  retval = PAPI_add_event(&EventSet, PAPI_TLB_DM);
#else
  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
#endif
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

#if (defined(i386) && defined(linux)) || (defined(_POWER) && defined(_AIX)) || defined(mips) || defined(_CRAYT3E)
  retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

#elif defined(sparc) && defined(sun)
  retval = PAPI_add_event(&EventSet, PAPI_LD_INS);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

  retval = PAPI_add_event(&EventSet, PAPI_SR_INS);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(__ALPHA) && defined(__osf__)
  retval = PAPI_add_event(&EventSet, PAPI_RES_STL);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#else
#error "Architecture not ported yet"
#endif

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) 
#if defined(__ALPHA) && defined(__osf__)
     printf("case4: %lld %lld %lld\n", values[0], values[1],values[3]);
#else
     printf("case4: %lld %lld %lld %lld\n",values[0],values[1],values[2],values[3]);
#endif
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

int main(int argc, char **argv)
{

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ( !TESTS_QUIET ) {
    printf("%s: Using %d iterations\n\n",argv[0],NUM);

    printf("case1: Does PAPI_multiplex_init() not break regular operation?\n");
  }
  case1();

  if ( !TESTS_QUIET ) 
  	printf("case2: Does setmpx/add work?\n");
  case2();

  if ( !TESTS_QUIET ) 
  	printf("case3: Does add/setmpx work?\n");
  case3();
  if ( !TESTS_QUIET ) 
  	printf("case4: Does add/setmpx/add work?\n");
  case4();
  PAPI_library_init(PAPI_VER_CURRENT);
  test_pass(__FILE__,NULL,0);
  exit(1);
}
