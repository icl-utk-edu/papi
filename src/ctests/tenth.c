#define ITERS 100

/* This file performs the following test: start, stop and timer functionality for 
   PAPI_L1_TCM derived event

   - They are counted in the default counting domain and default
     granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/


#if defined(sun) && defined(sparc)
  #define CACHE_LEVEL "PAPI_L2_TCM"
  #define EVT1		  PAPI_L2_TCM
  #define EVT2		  PAPI_L2_TCA
  #define EVT3		  PAPI_L2_TCH
  #define EVT1_STR	  "PAPI_L2_TCM"
  #define EVT2_STR	  "PAPI_L2_TCA"
  #define EVT3_STR	  "PAPI_L2_TCH"
  #define MASK1		  MASK_L2_TCM
  #define MASK2		  MASK_L2_TCA
  #define MASK3		  MASK_L2_TCH
#else
 #if (_POWER4)
  #define CACHE_LEVEL "PAPI_L1_DCA"
  #define EVT1		  PAPI_L1_DCA
  #define EVT2		  PAPI_L1_DCW
  #define EVT3		  PAPI_L1_DCR
  #define EVT1_STR	  "PAPI_L1_DCA"
  #define EVT2_STR	  "PAPI_L1_DCW"
  #define EVT3_STR	  "PAPI_L1_DCR"
  #define MASK1		  MASK_L1_DCA
  #define MASK2		  MASK_L1_DCW
  #define MASK3		  MASK_L1_DCR
 #else
  #define CACHE_LEVEL "PAPI_L1_TCM"
  #define EVT1		  PAPI_L1_TCM
  #define EVT2		  PAPI_L1_ICM
  #define EVT3		  PAPI_L1_DCM
  #define EVT1_STR	  "PAPI_L1_TCM"
  #define EVT2_STR	  "PAPI_L1_ICM"
  #define EVT3_STR	  "PAPI_L1_DCM"
  #define MASK1		  MASK_L1_TCM
  #define MASK2		  MASK_L1_ICM
  #define MASK3		  MASK_L1_DCM
 #endif
#endif

#include "papi_test.h"

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv) 
{
  int retval, num_tests = 3, tmp;
  int EventSet1;
  int EventSet2;
  int EventSet3;
  int mask1 = MASK1;
  int mask2 = MASK2;
  int mask3 = MASK3;
  int num_events1;
  int num_events2;
  int num_events3;
  long_long **values;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  /* Make sure that required resources are available */
  retval = PAPI_query_event(EVT1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, EVT1_STR, retval);

  retval = PAPI_query_event(EVT2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, EVT2_STR, retval);

  retval = PAPI_query_event(EVT3);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, EVT3_STR, retval);


  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);
  EventSet3 = add_test_events(&num_events3,&mask3);

  values = allocate_test_space(num_tests, 1);

  /* Warm me up */

  do_l1misses(ITERS);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet2, values[1]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_start(EventSet3);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet3, values[2]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);
  remove_test_events(&EventSet3, mask3);

  if ( !TESTS_QUIET ) {
	printf("Test case 10: start, stop for derived event %s.\n", CACHE_LEVEL);
	printf("--------------------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type   : %12s%12s%12s\n", "1", "2", "3");
#if defined(sun) && defined(sparc)
	printf(TAB3, "PAPI_L2_TCM : ", (values[0])[0], (long_long)0, (long_long)0);
	printf(TAB3, "PAPI_L2_TCA : ", (long_long)0, (values[1])[0], (long_long)0);
	printf(TAB3, "PAPI_L2_TCH : ", (long_long)0, (long_long)0, (values[2])[0]);
	printf("-------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf(TAB1, "First number row 1 approximately equals (2,2) - (3,3) or",(values[1])[0]-(values[2])[0]);
#else
	printf(TAB3, "PAPI_L1_TCM : ", (values[0])[0], (long_long)0, (long_long)0);
	printf(TAB3, "PAPI_L1_ICM : ", (long_long)0, (values[1])[0], (long_long)0);
	printf(TAB3, "PAPI_L1_DCM : ", (long_long)0, (long_long)0, (values[2])[0]);
	printf("-------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf(TAB1, "First number row 1 approximately equals (2,2) + (3,3) or",(values[1])[0]+(values[2])[0]);
#endif
  }

  {
 	long_long min, max;

#if defined(sun) && defined(sparc)
  	max = (long_long)((values[1])[0]-(values[2])[0]);
#else
  	max = (long_long)((values[1])[0]+(values[2])[0]);
#endif

	min = (long_long)(max * 0.9);
	max = (long_long)(max * 1.1);
  	if ( values[0][0] > max || values[0][0] < min)
		test_fail(__FILE__, __LINE__, CACHE_LEVEL, 1);
  }
  test_pass(__FILE__, values, num_tests);
  exit(1);
}
