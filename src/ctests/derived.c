/* This file performs the following test: start, stop with a derived event */

#include "papi_test.h"

#ifdef OLD_TEST_DRIVER
#define CPP_TEST_FAIL(string, retval) test_fail(__FILE__, __LINE__, string, retval)
#define CPP_TEST_PASS() { test_pass(__FILE__, NULL, 0); }
#define CPP_TEST_SKIP() { test_skip(__FILE__,__LINE__,NULL,0); }
#else
#define CPP_TEST_FAIL(function, retval) { fprintf(stderr,"%s:%d:%s:%d:%s:%s\n",__FILE__,__LINE__,function,retval,PAPI_strerror(retval),"$Id$\n"); test_fail(__FILE__, __LINE__, function, retval); }
#define CPP_TEST_PASS() { fprintf(stderr,"$Id$\n%s:\tPASSED\n",__FILE__); exit(0); }
#define CPP_TEST_SKIP() { fprintf(stderr,"$Id$\n%s:\tSKIPPED\n",__FILE__); exit(0); }
#endif

#define NUM_ITERS 5000

#define QUIETPRINTF if (!TESTS_QUIET) printf
const static unsigned int PAPI_events[PAPI_MPX_DEF_DEG] = { PAPI_L2_DCM, 0 };
const static int PAPI_events_len = 1;
extern int TESTS_QUIET;

int main(int argc, char **argv) 
{
  int retval, tmp;
  int EventSet;
  int i;
  long_long values;
  char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];

#if !defined(i386) || !defined(linux)
  CPP_TEST_SKIP();
#endif

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  QUIETPRINTF("Test case %s: start, stop with a derived counter.\n",__FILE__);
  QUIETPRINTF("------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  QUIETPRINTF("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  QUIETPRINTF("Default granularity is: %d (%s)\n\n",tmp,stringify_granularity(tmp));

  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_create_eventset",retval);

  for (i=0;i<PAPI_events_len;i++)
    {
      PAPI_event_code_to_name(PAPI_events[i],event_name);
      if (!TESTS_QUIET) QUIETPRINTF("Adding %s\n",event_name);
      retval = PAPI_add_event(&EventSet, PAPI_events[i]);
      if (retval != PAPI_OK)
	CPP_TEST_FAIL("PAPI_add_event",retval);
    }

  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  QUIETPRINTF("Running %d iterations of do_reads().\n",NUM_ITERS);
  
  do_reads(NUM_ITERS);
  
  retval = PAPI_stop(EventSet, &values);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  sprintf(add_event_str, "%-12s : \t", event_name);
  QUIETPRINTF(TAB1, add_event_str, values);
  QUIETPRINTF("------------------------------------------------\n");

  retval = PAPI_cleanup_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    CPP_TEST_FAIL("PAPI_cleanup_eventset",retval);

  PAPI_shutdown();
  
  QUIETPRINTF("Verification: Does produce a non-zero value?\n");

  if (values == 0)
    CPP_TEST_FAIL("Zero count returned",0);

  CPP_TEST_PASS();
}
