/* This file performs the following test: nested eventsets that share 
   all counter values

   - It attempts to use two eventsets simultaneously. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     Eventset 1 and 2 both have:
     + PAPI_FP_INS  or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
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

#define TEST_NAME "fourth"

#ifdef NO_FLOPS
  #define PAPI_EVENT 		PAPI_TOT_INS
  #define MASK				0x3
#else
  #define PAPI_EVENT 		PAPI_FP_INS
  #define MASK				0x5
#endif

int TESTS_QUIET=0; /* Tests in Verbose mode? */

int main(int argc, char **argv) 
{
  int retval, num_tests = 4, tmp;
  long_long **values;
  int EventSet1, EventSet2;
  int mask1 = MASK, mask2 = MASK;
  int num_events1, num_events2;
  char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];


  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_set_debug", retval);
  }

  retval = PAPI_event_code_to_name(PAPI_EVENT, event_name);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_event_code_to_name", retval);
  sprintf(add_event_str, "PAPI_add_event[%s]", event_name);

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(TEST_NAME, "PAPI_library_init", retval);

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_flops(NUM_FLOPS);
 
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[1]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet1, values[2]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[3]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);
  
  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);

  if ( !TESTS_QUIET ){
	printf("Test case 4: Overlapping start and stop of 2 eventsets.\n");
	printf("-------------------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type   : \t1\t\t2\t\t3\t\t4\n");
	sprintf(add_event_str, "%s : ", event_name);
	printf(TAB4, add_event_str,
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0]);
	printf(TAB4, "PAPI_TOT_CYC: ",
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1]);
	printf("-------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf("Column 1 approximately equals column 2\n");
	printf("Column 3 approximately equals three times column 2\n");
	printf("Column 4 approximately equals column 2\n");
  }

  {
        long_long min, max;
        min = (long_long)(values[1][0]*.9);
        max = (long_long)(values[1][0]*1.1);
        if ( values[0][0] > max || values[0][0] < min || values[2][0]>(3*max)
          || values[2][0] < (min*3) || values[3][0] < min || values[3][0]>max)
			{
				test_fail(TEST_NAME, event_name, 1);
			}
        min = (long_long)(values[1][1]*.9);
        max = (long_long)(values[1][1]*1.1);
        if ( values[0][1] > max || values[0][1] < min || values[2][1]>(3*max)
          || values[2][1] < (min*3) || values[3][1] < min || values[3][1]>max)
 			{
  				test_fail(TEST_NAME, "PAPI_TOT_CYC", 1);
			}
  }
  test_pass(TEST_NAME, values, num_tests);
}

