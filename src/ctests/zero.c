/* This file performs the following test: start, stop and timer functionality

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include "papi_test.h"

#ifdef NO_FLOPS
  #define PAPI_EVENT 		PAPI_TOT_INS
  #define MASK				MASK_TOT_INS | MASK_TOT_CYC
#else
  #define PAPI_EVENT 		PAPI_FP_INS
  #define MASK				MASK_FP_INS | MASK_TOT_CYC
#endif

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv) 
{
  int retval, num_tests = 1, tmp;
  int EventSet1;
  int mask1 = MASK;
  int num_events1;
  long_long **values;
  long_long elapsed_us, elapsed_cyc;
  char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  retval = PAPI_event_code_to_name(PAPI_EVENT, event_name);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  sprintf(add_event_str, "PAPI_add_event[%s]", event_name);


  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);
 
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  remove_test_events(&EventSet1, mask1);

  if ( !TESTS_QUIET ) {
	printf("Test case 0: start, stop.\n");
	printf("-----------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type    : \t           1\n");

	sprintf(add_event_str, "%-12s : \t", event_name);
	printf(TAB1, add_event_str,
	 (values[0])[0]);
	printf(TAB1, "PAPI_TOT_CYC : \t",
	 (values[0])[1]);
	printf(TAB1, "Real usec    : \t",
	 elapsed_us);
	printf(TAB1, "Real cycles  : \t",
	 elapsed_cyc);

	printf("-------------------------------------------------------------------------\n");

	printf("Verification: none\n");
  }
  test_pass(__FILE__, values, num_tests);
  exit(1);
}
