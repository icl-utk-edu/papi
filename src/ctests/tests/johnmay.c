#include "papi_test.h"
extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int CycEventSet, FPEventSet;
   long_long int values, values2, values3, values4, values5;
   int PAPI_event, retval;
   char event_name[PAPI_MAX_STR_LEN];

   tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

   if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  /* query and set up the right instruction to monitor */
   if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK)  PAPI_event = PAPI_FP_INS;
   else PAPI_event = PAPI_TOT_INS;

   if ( !TESTS_QUIET )
     if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

   if ((retval=PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_query_event",retval);

   if ((retval=PAPI_create_eventset(&CycEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

   if ((retval=PAPI_add_event(&CycEventSet, PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

   if ((retval=PAPI_query_event(PAPI_event)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_query_event",retval);

   if ((retval=PAPI_create_eventset(&FPEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

   if (PAPI_add_event(&FPEventSet, PAPI_event) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);

   if ((retval=PAPI_start(FPEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);

   do_flops(1000000);

   if ((retval=PAPI_stop(FPEventSet, &values)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

   if ((retval=PAPI_start(FPEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);
   do_flops(1000000);
   if ((retval=PAPI_stop(FPEventSet, &values2)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);

   if ((retval=PAPI_start(FPEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);
   if ((retval=PAPI_start(CycEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);
   do_flops(1000000);
   if ((retval=PAPI_stop(FPEventSet, &values3)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
  
   if ((retval=PAPI_start(FPEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval);
   do_flops(1000000);
   if ((retval=PAPI_stop(FPEventSet, &values4)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
   if ((retval=PAPI_stop(CycEventSet, &values5)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
  
  if ( !TESTS_QUIET ) {
	  if ((retval=PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
		test_fail(__FILE__,__LINE__,"PAPI_event_code_to_name",retval);

	  printf("Test case John May: start & stop of overlapped unshared counters.\n");
	  printf("-----------------------------------------------------------------\n");
	  printf("Test run    : \t1\t2\t3\t4\n");
	  printf("%s : \t%lld\t%lld\t%lld\t%lld\n", event_name, values, values2, values3, values4);
	  printf("PAPI_TOT_CYC: \t\t\t\t%lld\n", values5);
	  printf("-----------------------------------------------------------------\n");
  }
  test_pass(__FILE__,NULL,0);
  exit(1);
}
