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

#include "papi_test.h"

#define SUCCESS 1

extern int TESTS_QUIET; /* Declared in test_utils.c */

/* Event to use in all cases; initialized in init_papi() */

#if defined(sparc) && defined(sun) || defined(__ALPHA) && defined(__osf__)
const static unsigned int preset_PAPI_events[PAPI_MPX_DEF_DEG] = { PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_ICM, 0 };
#else
const static unsigned int preset_PAPI_events[PAPI_MPX_DEF_DEG] = { PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_DCM, 0 };
#endif

static unsigned int PAPI_events[PAPI_MPX_DEF_DEG] = { 0, };
static int PAPI_events_len;

#ifdef TEST_DRIVER
#define CPP_TEST_FAIL(function, retval) { fprintf(stderr,"%s:%d:%s:%d:%s:%s\n",__FILE__,__LINE__,function,retval,PAPI_strerror(retval),"$Id$"); test_fail(__FILE__, __LINE__, function, retval); }
#else
#define CPP_TEST_FAIL(string, retval) test_fail(__FILE__, __LINE__, string, retval)
#endif

void init_papi(unsigned int *out_events, int *len)
{
  int retval;
  int i, real_len = 0;
  const unsigned int *in_events = preset_PAPI_events;
 
  /* Initialize the library */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    CPP_TEST_FAIL("PAPI_library_init",retval);

  /* Turn on automatic error reporting */
  if ( !TESTS_QUIET ) {
     retval = PAPI_set_debug(PAPI_VERB_ECONT);
     if (retval != PAPI_OK)
    	CPP_TEST_FAIL("PAPI_set_debug",retval);
  }

  for (i=0;in_events[i]!=0;i++)
    {
       char out[PAPI_MAX_STR_LEN];
      /* query and set up the right instruction to monitor */
      retval = PAPI_query_event(in_events[i]);
	if (retval == PAPI_OK) 
	{
	  out_events[real_len++] = in_events[i];
	  PAPI_event_code_to_name(in_events[i],out);
 	  if ( !TESTS_QUIET )
	  	printf("%s exists\n",out);
	if (real_len == *len)
	break;
	}
	else
	{
	  PAPI_event_code_to_name(in_events[i],out);
	  if ( !TESTS_QUIET )
	  	printf("%s does not exist\n",out);
	}
    }
  if (real_len < 1) 
    CPP_TEST_FAIL("No counters available",0);
  *len = real_len;
}

/* Tests that PAPI_multiplex_init does not mess with normal operation. */

int case1() 
{
  int retval, i, EventSet = PAPI_NULL;
  long_long values[2];

  PAPI_events_len = 2;
  init_papi(PAPI_events,&PAPI_events_len);

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_create_eventset",retval);

  for (i=0;i<PAPI_events_len;i++)
    {
      char out[PAPI_MAX_STR_LEN];

      retval = PAPI_add_event(&EventSet, PAPI_events[i]);
      if (retval != PAPI_OK)
	CPP_TEST_FAIL("PAPI_add_event",retval);
      PAPI_event_code_to_name(PAPI_events[i],out);
      if ( !TESTS_QUIET )
      	printf("Added %s\n",out);
    }

  if (PAPI_start(EventSet) != PAPI_OK)
    CPP_TEST_FAIL("PAPI_start",retval);

  do_both(NUM_ITERS);

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_stop",retval);

  if ( !TESTS_QUIET ) {
    test_print_event_header("case1:",EventSet);
    printf(TAB2,"case1:",values[0],values[1]);
  }
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works before adding events */

int case2() 
{
  int retval, i, EventSet = PAPI_NULL;
  long_long values[2];

  PAPI_events_len = 2;
  init_papi(PAPI_events,&PAPI_events_len);

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_create_eventset",retval);

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_set_multiplex",retval);

  for (i=0;i<PAPI_events_len;i++)
    {
      char out[PAPI_MAX_STR_LEN];

      retval = PAPI_add_event(&EventSet, PAPI_events[i]);
      if (retval != PAPI_OK)
	CPP_TEST_FAIL("PAPI_add_event",retval);
      PAPI_event_code_to_name(PAPI_events[i],out);
      if ( !TESTS_QUIET )
      	printf("Added %s\n",out);
    }

  if (PAPI_start(EventSet) != PAPI_OK)
    CPP_TEST_FAIL("PAPI_start",retval);

  do_both(NUM_ITERS);
  
  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_stop",retval);

  if ( !TESTS_QUIET )  {
    test_print_event_header("case2:",EventSet);
    printf(TAB2,"case2:",values[0],values[1]);
  }

  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

/* Tests that PAPI_set_multiplex() works after adding events */

int case3() 
{
  int retval, i, EventSet = PAPI_NULL;
  long_long values[2];

  PAPI_events_len = 2;
  init_papi(PAPI_events,&PAPI_events_len);

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_create_eventset",retval);

  for (i=0;i<PAPI_events_len;i++)
    {
      char out[PAPI_MAX_STR_LEN];

      retval = PAPI_add_event(&EventSet, PAPI_events[i]);
      if (retval != PAPI_OK)
	CPP_TEST_FAIL("PAPI_add_event",retval);
      PAPI_event_code_to_name(PAPI_events[i],out);
      if ( !TESTS_QUIET )
      	printf("Added %s\n",out);
    }

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_set_multiplex",retval);

  if (PAPI_start(EventSet) != PAPI_OK)
    CPP_TEST_FAIL("PAPI_start",retval);

  do_both(NUM_ITERS);

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_stop",retval);

  if ( !TESTS_QUIET )   {
    test_print_event_header("case3:",EventSet);
    printf(TAB2,"case3:",values[0],values[1]);
  }

  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}
/* Tests that PAPI_set_multiplex() works before adding events */

/* Tests that PAPI_add_event() works after
   PAPI_add_event()/PAPI_set_multiplex() */

int case4() 
{
  int retval, i, EventSet = PAPI_NULL;
  long_long values[4];
  int nev,event_codes[4];
  char evname[4][PAPI_MAX_STR_LEN];

  PAPI_events_len = 2;
  init_papi(PAPI_events,&PAPI_events_len);

  retval = PAPI_multiplex_init();
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_multiplex_init",retval);
  
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_create_eventset",retval);

  for (i=0;i<PAPI_events_len;i++)
    {
      char out[PAPI_MAX_STR_LEN];

      retval = PAPI_add_event(&EventSet, PAPI_events[i]);
      if (retval != PAPI_OK)
	CPP_TEST_FAIL("PAPI_add_event",retval);
      PAPI_event_code_to_name(PAPI_events[i],out);
      if ( !TESTS_QUIET )
      	printf("Added %s\n",out);
    }

  retval = PAPI_set_multiplex(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_set_multiplex",retval);

#if (defined(i386) && defined(linux)) || (defined(_POWER) && defined(_AIX)) || defined(mips) || defined(_CRAYT3E) || (defined(__ia64__) && defined(linux)) || defined(WIN32)
  retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);

 #if (defined(_POWER4))
  retval = PAPI_add_event(&EventSet, PAPI_L1_DCA);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);
 #else
  retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);
 #endif

#elif defined(sparc) && defined(sun)
  retval = PAPI_add_event(&EventSet, PAPI_LD_INS);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);

  retval = PAPI_add_event(&EventSet, PAPI_SR_INS);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);
#elif defined(__ALPHA) && defined(__osf__)
  retval = PAPI_add_event(&EventSet, PAPI_TLB_DM);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_add_event",retval);
#else
#error "Architecture not ported yet"
#endif

  if (PAPI_start(EventSet) != PAPI_OK)
    CPP_TEST_FAIL("PAPI_start",retval);

  do_both(NUM_ITERS);

  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_stop",retval);

  nev=4;
  retval = PAPI_list_events(EventSet, event_codes, &nev);
  for(i=0;i<nev;i++)
    PAPI_event_code_to_name(event_codes[i],evname[i]);

  if ( !TESTS_QUIET ) {
     test_print_event_header("case4:",EventSet);
     printf(TAB4,"case4:",values[0],values[1],values[2],values[3]);
  }
  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);
  
  PAPI_shutdown();
  return(SUCCESS);
}

int main(int argc, char **argv)
{

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ( !TESTS_QUIET ) {
    printf("%s: Using %d iterations\n\n",argv[0],NUM_ITERS);

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
  exit(0);
}
