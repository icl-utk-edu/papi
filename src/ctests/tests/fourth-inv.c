/* This file performs the following test: nested eventsets that share 
   all counter values, but added in reverse order.

   - It attempts to use two eventsets simultaneously. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     Eventset 1 and 2 both have:
     + PAPI_FP_INS or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
     + PAPI_TOT_CYC

   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Start eventset 1
   - Do flops
   - Start eventset 2
   - Do flops
   - Stop and read eventset 2
   - Do flops
   - Stop and read eventset 1
   - Start eventset 2
   - Do flops
   - Stop eventset 2
*/

#include "papi_test.h"

#ifdef NO_FLOPS
  #define PAPI_EVENT 		PAPI_TOT_INS
#else
  #define PAPI_EVENT 		PAPI_FP_INS
#endif

extern int TESTS_QUIET; /* Declared in test_utils.c */


int main(int argc, char **argv) 
{
  int retval, num_tests = 4, tmp;
  long_long **values;
  int EventSet1, EventSet2;
  int num_events = 2;
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


  retval = PAPI_create_eventset(&EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  retval = PAPI_create_eventset(&EventSet2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  retval = PAPI_add_event(&EventSet1, PAPI_TOT_CYC);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_add_event[PAPI_TOT_CYC]", retval);

  retval = PAPI_add_event(&EventSet2, PAPI_TOT_CYC);
  if (retval != PAPI_OK) {
	/* if there's a conflict, we can't complete this test */
	if(retval == PAPI_ECNFLCT ) test_fail(__FILE__, __LINE__, 0, PAPI_OK);
	else test_pass(__FILE__, NULL, 0);
  }

  retval = PAPI_add_event(&EventSet1, PAPI_EVENT);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, add_event_str, retval);


  retval = PAPI_add_event(&EventSet2, PAPI_EVENT);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_add_event[PAPI_TOT_CYC]", retval);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);
 
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[1]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet1, values[2]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[3]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
  
  if ( !TESTS_QUIET ) {
	printf("Test case 4: Overlapping start and stop of 2 (inverted) eventsets.\n");
	printf("------------------------------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type    : %12s%12s%12s%12s\n", "1", "2", "3", "4");
	sprintf(add_event_str, "%-12s : ", event_name);
	printf(TAB4, add_event_str,
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1]);
	printf(TAB4, "PAPI_TOT_CYC : ",
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0]);
	printf("-------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf("Column 1 approximately equals column 2\n");
	printf("Column 3 approximately equals three times column 2\n");
	printf("Column 4 approximately equals column 2\n");
  }
  {
	long_long min, max;
	min = (long_long)(values[1][1]*.9);
	max = (long_long)(values[1][1]*1.1);
	if ( values[0][1] > max || values[0][1]<min || values[2][1]>(3*max)||
	     values[2][1]<(3*min)||values[3][1]<min||values[3][1]>max ){
			test_fail(__FILE__, __LINE__, event_name, 1);
        }
	min = (long_long)(values[1][0]*.8);
	max = (long_long)(values[1][0]*1.2);
	if ( values[0][0] > max || values[0][0]<min || values[2][0]>(3*max)||
	     values[2][0]<(3*min)||values[3][0]<min||values[3][0]>max ){
 			test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
        }
  }
  test_pass(__FILE__, values, num_tests);
}

