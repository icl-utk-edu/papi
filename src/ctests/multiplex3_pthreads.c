/* 
* File:    multiplex1_pthreads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    John May
*          johnmay@llnl.gov
*/  

/* This file tests the multiplex pthread functionality when there are
 * threads in which the application isn't calling PAPI (and only
 * one thread that is calling PAPI.)
 */ 

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"
#include "test_utils.h"

#define NUM 10
#define NUM_THREADS 4
#define SUCCESS 1
#define FAILURE 0

extern void do_flops(int);
extern void do_reads(int);

extern int TESTS_QUIET; /* Declared in test_utils.c */

/* A thread function that does nothing forever, while the other
 * tests are running.
 */
void * thread_fn( void * dummy )
{
	while(1)
	  do_flops(10000);
}

void init_papi(void)
{
  int retval;

  /* Initialize the library */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ((retval=PAPI_thread_init((unsigned long (*)(void))(pthread_self), 0)) != PAPI_OK){
     if (retval == PAPI_ESBSTR)
        test_skip(__FILE__,__LINE__,"PAPI_thread_init",retval);
     else
        test_fail(__FILE__,__LINE__,"PAPI_thread_init",retval);
  }

  /* Turn on automatic error reporting */

  retval = PAPI_set_debug(PAPI_VERB_ECONT);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);
}

/* Runs a bunch of multiplexed events */

int case1(void) 
{
  int retval, i, EventSet = PAPI_NULL, max_to_add = 6, j = 0;
  int allvalid = 1;
  long long *values;
  const PAPI_preset_info_t *pset;

  init_papi();

  pset = PAPI_query_all_events_verbose();
  if (pset == NULL)
    test_fail(__FILE__,__LINE__,"PAPI_query_all_events_verbose",NULL);

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    {
      if ((pset->avail) && (pset->event_code != PAPI_TOT_CYC))
	{
	  if ( !TESTS_QUIET ) 
	    printf("Adding %s\n",pset->event_name);

	  retval = PAPI_add_event(&EventSet, pset->event_code);
	  if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
	    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

	  if ( !TESTS_QUIET ) 
	    {
	      if (retval == PAPI_OK)
		printf("Added %s\n",pset->event_name);
	      else
		printf("Could not add %s\n",pset->event_name);
	    }

	  if (retval == PAPI_OK)
	    {
	      if (++j >= max_to_add)
		break;
	    }
	}
	pset++;
    }

  values = (long long *)malloc(max_to_add*sizeof(long long));
  if (values == NULL)
    test_fail(__FILE__,__LINE__,"malloc",NULL);

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  do_reads(10000);

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  printf("case1:");
  for( i = 0; i < max_to_add; i++ ) {
	  printf(" %lld", values[i]);

	  /* There should be some sort of value for all events */
	  if( values[i] == 0 ) allvalid = 0;
  }
  printf("\n");

  if( !allvalid )
    test_fail(__FILE__,__LINE__,"case1",-1);

  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_destroy_eventset",retval);
  
  return(SUCCESS);
}

int main(int argc, char **argv)
{

  int i, rc;
  pthread_t id[NUM_THREADS];
  pthread_attr_t attr;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  /* Create a bunch of unused pthreads, to simulate threads created
   * by the system that the user doesn't know about.
   */
  pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
#endif
  for (i=0;i<NUM_THREADS;i++) {
    rc = pthread_create(&id[i], &attr, thread_fn, NULL);
    if (rc)
      test_fail(__FILE__,__LINE__,"pthread_create",rc);
  }

  if ( !TESTS_QUIET ) {
    printf("%s: Using %d iterations\n\n",argv[0],NUM);

    printf("case1: Does multiplexing work with extraneous threads present?\n");
  }
  case1();
  test_pass(__FILE__,NULL,0);

  pthread_attr_destroy(&attr);
}
