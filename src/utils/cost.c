#include "papi_test.h"

int main(int argc, char **argv)
{
   int i, retval, EventSet = PAPI_NULL;
   long_long totcyc, values[2];


   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if (!TESTS_QUIET) {
      printf("Cost of execution for PAPI start/stop and PAPI read.\n");
      printf("This test takes a while. Please be patient...\n");
   }

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   if ((retval = PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_query_event", retval);
   if ((retval = PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_query_event", retval);
   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      if ((retval = PAPI_add_event(EventSet, PAPI_TOT_IIS)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);


   /* Make sure no errors */

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   if ((retval = PAPI_stop(EventSet, NULL)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   /* Start the start/stop eval */

   if (!TESTS_QUIET)
      printf("Performing start/stop test...\n");
   totcyc = PAPI_get_real_cyc();
   for (i = 0; i < ITERS; i++) {
      PAPI_start(EventSet);
      PAPI_stop(EventSet, values);
   }

   totcyc = PAPI_get_real_cyc() - totcyc;

   if (!TESTS_QUIET) {
      printf("\n");

      printf("\nTotal cost for PAPI_start/stop(2 counters) over %d iterations\n",ITERS);
      printf(LLDFMT, totcyc);
      printf("total cyc,\n%f cyc/call pair\n",((float) totcyc) / (float)(ITERS+1));

      /* Start the read eval */
      printf("\n\nPerforming read test...\n");
   }
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   totcyc = PAPI_get_real_cyc();

   for (i = 0; i < ITERS; i++) {
      PAPI_read(EventSet, values);
   }

   totcyc = PAPI_get_real_cyc() - totcyc;

   if (!TESTS_QUIET) {
      printf("\nTotal cost for PAPI_read(2 counters) over %d iterations\n",ITERS);
      printf(LLDFMT, totcyc);
      printf("total cyc,\n%f cyc/call\n",((float) totcyc) / (float)(ITERS+1));
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
