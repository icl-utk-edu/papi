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

void handle_error(char *string, int lineno, int retval)
{
  if (retval < 0)
    fprintf(stderr,"%s failed at line %d, PAPI error code %d: %s\n",
	    string,lineno,retval,PAPI_strerror(retval));
  else
    fprintf(stderr,"%s failed at line %d, %d\n",string,lineno,retval);
  exit(1);
}

void init_papi(void)
{
  int retval;

  /* Initialize the library */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    handle_error("PAPI_library_init",__LINE__,retval);

  /* Turn on automatic error reporting */

  retval = PAPI_set_debug(PAPI_VERB_ECONT);
  if (retval != PAPI_OK)
    handle_error("PAPI_set_debug",__LINE__,retval);
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

int case1(void) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    handle_error("PAPI_multiplex_init",__LINE__,retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_create_eventset",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    handle_error("PAPI_start",__LINE__,retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    handle_error("PAPI_stop",__LINE__,retval);

  printf("case1: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_cleanup_eventset",__LINE__,retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before adding events */

int case2(void) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    handle_error("PAPI_multiplex_init",__LINE__,retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_create_eventset",__LINE__,retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_set_multiplex",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    handle_error("PAPI_start",__LINE__,retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    handle_error("PAPI_stop",__LINE__,retval);

  printf("case2: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_cleanup_eventset",__LINE__,retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works after adding events */

int case3(void) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[2];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    handle_error("PAPI_multiplex_init",__LINE__,retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_create_eventset",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_set_multiplex",__LINE__,retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    handle_error("PAPI_start",__LINE__,retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    handle_error("PAPI_stop",__LINE__,retval);

  printf("case3: %lld %lld\n",values[0],values[1]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_cleanup_eventset",__LINE__,retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}
/* Tests that PAPI_set_multiplex() works before adding events */

/* Tests that PAPI_add_event() works after
   PAPI_add_event()/PAPI_set_multiplex() */

int case4(void) 
{
  int retval, i, EventSet = PAPI_NULL;
  long long values[4];

  init_papi();

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    handle_error("PAPI_multiplex_init",__LINE__,retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_create_eventset",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_set_multiplex",__LINE__,retval);

#if (defined(i386) && defined(linux)) || (defined(_POWER) && defined(_AIX) || defined(mips))
  retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

#elif defined(sparc) && defined(sun)
  retval = PAPI_add_event(&EventSet, PAPI_LD_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);

  retval = PAPI_add_event(&EventSet, PAPI_SR_INS);
  if (retval != PAPI_OK)
    handle_error("PAPI_add_event",__LINE__,retval);
#else
#error "Architecture not ported yet"
#endif

  if (PAPI_start(EventSet) != PAPI_OK)
    handle_error("PAPI_start",__LINE__,retval);

  for (i=0;i<NUM;i++)
    {
      do_flops(1000000);
      do_reads(1000);
    }

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    handle_error("PAPI_stop",__LINE__,retval);

  printf("case4: %lld %lld %lld %lld\n",values[0],values[1],values[2],values[3]);
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error("PAPI_cleanup_eventset",__LINE__,retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

int main(int argc, char **argv)
{
  printf("%s: Using %d iterations\n\n",argv[0],NUM);

  printf("case1: Does PAPI_multiplex_init() not break regular operation?\n");
  case1();
  printf("case2: Does setmpx/add work?\n");
  case2();
  printf("case3: Does add/setmpx work?\n");
  case3();
  printf("case4: Does add/setmpx/add work?\n");
  case4();

  exit(0);
}
