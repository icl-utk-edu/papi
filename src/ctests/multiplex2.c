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
#include "test_utils.h"

#define NUM 100
#define SUCCESS 1

extern void do_flops(int);
extern void do_reads(int);

int TESTS_QUIET=0;

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

/* Tests that we can really multiplex a lot. */

int case1(void) 
{
  int retval, i, EventSet = PAPI_NULL, max_to_add = 6, j = 0;
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
	  printf("Adding %s\n",pset->event_name);
	  retval = PAPI_add_event(&EventSet, pset->event_code);
	  if (retval != PAPI_OK)
	    test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
	  printf("Added %s\n",pset->event_name);
	  if (++j >= max_to_add)
	    break;
	}
	pset++;
    }

  values = (long long *)malloc(max_to_add*sizeof(long long));
  if (values == NULL)
    test_fail(__FILE__,__LINE__,"malloc",NULL);

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

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if ( !TESTS_QUIET ) {
    printf("%s: Using %d iterations\n\n",argv[0],NUM);

    printf("case1: Does PAPI_multiplex_init() handle lots of events?\n");
  }
  case1();
  test_pass(__FILE__,NULL,0);
}
