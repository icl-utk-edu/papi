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
 *
 * This test will fail on most, if not all platforms due to signal handling.
 */ 

#include <pthread.h>
#include "papi_test.h"

/* A thread function that does nothing forever, while the other
 * tests are running.
 */
void * thread_fn( void * dummy )
{
	while(1)
	  do_both(NUM_ITERS);
}

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
}

/* Runs a bunch of multiplexed events */

int allvalid = 1;
long long *values;
int EventSet = PAPI_NULL, max_to_add = 6;

int case1_first_half(void) 
{
  int retval, i, j = 2;
  const PAPI_preset_info_t *pset;

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_multiplex_init",retval);
  
  pset = PAPI_query_all_events_verbose();
  if (pset == NULL)
    test_fail(__FILE__,__LINE__,"PAPI_query_all_events_verbose",0);

  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_set_multiplex",retval);

  retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
  if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
  if ( !TESTS_QUIET ) 
    {
      printf("Added %s\n","PAPI_TOT_INC");
    }

  retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
  if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
  if ( !TESTS_QUIET ) 
    {
      printf("Added %s\n","PAPI_TOT_CYC");
    }

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    {
	  /* skip total cycles and some cache events */
      if ((pset->avail)
			&& (pset->event_code != PAPI_TOT_CYC)
			&& (pset->event_code != PAPI_TOT_INS)
			&& (pset->event_code != PAPI_CA_SHR))
	{
	  if ( !TESTS_QUIET ) 
	    printf("Adding %s\n",pset->event_name);

	  retval = PAPI_add_event(EventSet, pset->event_code);
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
    test_fail(__FILE__,__LINE__,"malloc",0);

  if (PAPI_start(EventSet) != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_start",retval);

  return 0;
}

int case1_last_half(void) 
{
  int i, retval;

  do_both(NUM_ITERS);

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

  if ( !TESTS_QUIET ) {
    test_print_event_header("case1:",EventSet);
    printf("case1:");
    for( i = 0; i < max_to_add; i++ ) {
      printf(ONENUM, values[i]);

    /* There should be some sort of value for all events */
    if( values[i] == 0 ) allvalid = 0;
	}
    printf("\n");
  }

  if( !allvalid )
    test_fail(__FILE__,__LINE__,"case1 (one or more counter registered no counts)",1);

  retval = PAPI_cleanup_eventset(EventSet);     /* JT */
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset",retval);

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"PAPI_destroy_eventset",retval);
  
  return(SUCCESS);
}

int main(int argc, char **argv)
{

  int i, rc, retval;
  pthread_t id[NUM_THREADS];
  pthread_attr_t attr;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  case1_first_half();

  /* Create a bunch of unused pthreads, to simulate threads created
   * by the system that the user doesn't know about.
   */
  pthread_attr_init(&attr);
#ifdef PTHREAD_CREATE_UNDETACHED
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif
#ifdef PTHREAD_SCOPE_SYSTEM
  retval = pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
  if (retval != 0)
    test_skip(__FILE__, __LINE__, "pthread_attr_setscope", retval);    
#endif

  for (i=0;i<NUM_THREADS;i++) {
    rc = pthread_create(&id[i], &attr, thread_fn, NULL);
    if (rc)
      test_fail(__FILE__,__LINE__,"pthread_create",rc);
  }

  if ( !TESTS_QUIET ) {
    printf("%s: Using %d iterations\n\n",argv[0],NUM_ITERS);

    printf("case1: Does multiplexing work with extraneous threads present?\n");
  }
  case1_last_half();
  test_pass(__FILE__,NULL,0);

  pthread_attr_destroy(&attr);
  exit(0);
}
