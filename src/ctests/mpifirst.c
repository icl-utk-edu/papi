/* This file performs the following test: start, read, stop and again functionality

   - It attempts to use the following three counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
     + PAPI_TOT_CYC
   - Start counters
   - Do flops
   - Read counters
   - Reset counters
   - Do flops
   - Read counters
   - Do flops
   - Read counters
   - Do flops
   - Stop and read counters
   - Read counters
*/

#include "papi_test.h"

#ifdef _WIN32
  #define FORMAT	"min: %I64d max: %I64d  1st: %I64d  2nd: %I64d  3rd:  %I64d 4th: %I64d 5th: %I64d\n"
#else
  #define FORMAT	"min: %lld max: %lld  1st: %lld  2nd: %lld  3rd:  %lld 4th: %lld 5th: %lld\n"
#endif

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
  int retval, num_tests = 5, num_events, tmp;
  long_long **values;
  int EventSet;
  int mask = MASK;
  char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];


  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  MPI_Init(argc,argv);

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  retval = PAPI_event_code_to_name(PAPI_EVENT, event_name);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  sprintf(add_event_str, "PAPI_add_event[%s]", event_name);

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[0]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_read", retval);

  retval = PAPI_reset(EventSet);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_reset", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[1]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_read", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[2]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_read", retval);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet, values[3]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_read(EventSet, values[4]);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_read", retval);

  remove_test_events(&EventSet, mask);

  if ( !TESTS_QUIET ) {
	printf("Test case 1: Non-overlapping start, stop, read.\n");
	printf("-----------------------------------------------\n");
	tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
	printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
	tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
	printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-------------------------------------------------------------------------\n");

	printf("Test type   : \t1\t\t2\t\t3\t\t4\t\t5\n");
	sprintf(add_event_str, "%s : ", event_name);
	printf(TAB5, add_event_str,
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0],(values[4])[0]);
	printf(TAB5, "PAPI_TOT_CYC: ",
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1],(values[4])[1]);
	printf("-------------------------------------------------------------------------\n");

	printf("Verification:\n");
	printf("Column 1 approximately equals column 2\n");
	printf("Column 3 approximately equals 2 * column 2\n");
	printf("Column 4 approximately equals 3 * column 2\n");
	printf("Column 4 exactly equals column 5\n");
  }

  {
    long_long min, max;
	min = (long_long)(values[1][0]*.9);
	max = (long_long)(values[1][0]*1.1);

	if ( values[0][0] > max || values[0][0] < min || values[2][0]>(2*max)
	|| values[2][0]<(2*min) || values[3][0]>(3*max)||values[3][0]<(3*min)
	|| values[3][0]!=values[4][0])
	{
		printf(FORMAT, min, max, values[0][0], values[1][0], values[2][0],values[3][0], values[4][0] );
		test_fail(__FILE__, __LINE__, event_name, 1);
	}

	min = (long_long)(values[1][1]*.9);
	max = (long_long)(values[1][1]*1.1);
	if ( values[0][1] > max || values[0][1] < min || values[2][1]>(2*max)
	|| values[2][1]<(2*min) || values[3][1]>(3*max)||values[3][1]<(3*min)
	|| values[3][1]!=values[4][1])
	{
  		test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
	}
  }
  test_pass(__FILE__, values, num_tests);

  MPI_Finalize();
  exit(1);
}
