/* This file performs the following test: nested eventsets that do not share any counter values

   - It attempts to use two eventsets simultaneously. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is the user domain 
     (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).

     Eventset 1 has:
     + PAPI_TOT_CYC

     EventSet 2 has:
     + PAPI_FP_INS
	 or
	 + PAPI_TOT_INS

   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Start eventset 1
   - Do flops
   - Start eventset 2
   - Read eventset 1
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
  #define MASK				MASK_TOT_INS
#else
  #define PAPI_EVENT 		PAPI_FP_INS
  #define MASK				MASK_FP_INS
#endif

int TESTS_QUIET=0; /* Tests in Verbose mode? */

int main(int argc, char **argv) 
{
  int retval, num_tests = 5, tmp;
  long_long **values;
  int EventSet1, EventSet2;
  int mask1 = MASK_TOT_CYC, mask2 = MASK;
  int num_events1, num_events2;
  char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

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
  EventSet2 = add_test_events(&num_events2,&mask2);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

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

  retval = PAPI_read(EventSet1, values[1]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_read", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[2]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet1, values[3]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet2, values[4]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);

  if ( !TESTS_QUIET ) {
	printf("Test case 3: Overlapping start and stop of 2 eventsets with different counters.\n");
	printf("-------------------------------------------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("--------------------------------------------------------------------------\n");

	printf("Test type   :         1           2           3           4           5\n");
 	sprintf(add_event_str, "%s : ", event_name);
	printf(TAB5, add_event_str,
	 (long_long)0,(long_long)0,(values[2])[0],(long_long)0,(values[4])[0]);
	printf(TAB5, "PAPI_TOT_CYC: ",
	 (values[0])[0],(values[1])[0],(long_long)0,(values[3])[0],(long_long)0);
	printf("--------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf("Row 1 approximately equals %d %d N %d N\n",0,0,0);
	printf("Row 2 approximately equals X X %d X %d\n",0,0);
	printf("Column 1 approximately equals column 2\n");
	printf("Column 4 approximately equals three times column 1\n");
	printf("Column 5 approximately equals column 3\n");
  }

  {
 	long_long min, max;

	min = (long_long)(values[2][0]*.9);
 	max = (long_long)(values[2][0]*1.1);
	if ( values[2][0] == 0 || values[4][0] == 0
		|| values[4][0] > max || values[4][0] < min )
			test_fail(__FILE__, __LINE__, event_name, 1);

	min = (long_long)(values[0][0]*.9);
	max = (long_long)(values[0][0]*1.1);
	if ( values[0][0] == 0 || values[1][0] == 0 || values[3][0] == 0
		|| values[1][0] > max || values[1][0] < min
		|| values[3][0]>(max*3) || values[3][0] < (min*3))
  			test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
  }
  test_pass(__FILE__, values, num_tests);
}
