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

#include "papi_test.h"

#define TEST_NAME "tenth"

int TESTS_QUIET=0; /* Tests in Verbose mode? */

int main(int argc, char **argv) 
{
  int retval, num_tests = 3, tmp;
  int EventSet1;
  int EventSet2;
  int EventSet3;
#if defined(sun) && defined(sparc)
  #define CACHE_LEVEL "PAPI_L2_TCM"
  int mask1 = MASK_L2_TCM;
  int mask2 = MASK_L2_TCA;
  int mask3 = MASK_L2_TCH;
#else
  #define CACHE_LEVEL "PAPI_L1_TCM"
  int mask1 = MASK_L1_TCM;
  int mask2 = MASK_L1_ICM;
  int mask3 = MASK_L1_DCM;
#endif
  int num_events1;
  int num_events2;
  int num_events3;
  long_long **values;

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_set_debug", retval);
  }

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(TEST_NAME, "PAPI_library_init", retval);


#ifndef _CRAYT3E
  EventSet1 = add_test_events(&num_events1,&mask1);
#endif
  EventSet2 = add_test_events(&num_events2,&mask2);
  EventSet3 = add_test_events(&num_events3,&mask3);

  values = allocate_test_space(num_tests, 1);

  /* Warm me up */

  do_l1misses(ITERS);

#ifndef _CRAYT3E 
  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);
#else
  (values[0])[0] = 0LL;
#endif

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet2, values[1]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);

  retval = PAPI_start(EventSet3);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_start", retval);

  do_l1misses(ITERS);
  
  retval = PAPI_stop(EventSet3, values[2]);
  if (retval != PAPI_OK) test_fail(TEST_NAME, "PAPI_stop", retval);

#ifndef _CRAYT3E
  remove_test_events(&EventSet1, mask1);
#endif
  remove_test_events(&EventSet2, mask2);
  remove_test_events(&EventSet3, mask3);

  if ( !TESTS_QUIET ) {
	printf("Test case 9: start, stop for derived event %s.\n", CACHE_LEVEL);
	printf("------------------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type   :         1           2           3\n");
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
  	max = (values[1])[0]-(values[2])[0]);
#else
  	max = (long_long)((values[1])[0]+(values[2])[0]);
#endif

	min = (long_long)(max * 0.9);
	max = (long_long)(max * 1.1);
  	if ( values[0][0] > max || values[0][0] < min)
		test_fail(TEST_NAME, CACHE_LEVEL, 1);
  }
  test_pass(TEST_NAME, values, num_tests);
}
