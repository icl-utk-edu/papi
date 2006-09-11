/* This file performs the following test: event_code_to_name */

#include "papi_test.h"

int main(int argc, char **argv)
{
  int retval;
  int code = PAPI_TOT_CYC;
  char event_name[PAPI_MAX_STR_LEN];

  tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
  
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

  /* Highly doubtful we have this many natives */

  code = PAPI_NATIVE_MASK | (PAPI_NATIVE_MASK-1);

  retval = PAPI_event_code_to_name(code, event_name);
  if ((retval == PAPI_ENOEVNT) || (retval == PAPI_OK))
   test_pass(__FILE__, 0, 0);
   
   test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", PAPI_EBUG);

   exit(1);
}
