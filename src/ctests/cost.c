#include "papi_test.h"

int main(int argc, char **argv)
{
   int i, retval, EventSet = PAPI_NULL;
   long_long totcyc, values[2];
   long_long *array;
   long_long min, max, tmp;
   double  average, std;


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
   array = (long_long *)malloc(NUM_ITERS*sizeof(long_long));
   if (array == NULL ) 
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   for (i = 0; i < NUM_ITERS; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_start(EventSet);
      PAPI_stop(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i]=totcyc;
   }
   min = max = array[0]; 
   printf("array[0] = %lld \n", array[0]);
   average = 0;
   for(i=0; i < NUM_ITERS; i++ ) {
      average += array[i]; 
      if (min > array[i]) min = array[i];
      if (max < array[i]) max = array[i];
   }
   average = (long) (average/NUM_ITERS); 
   std=0;
   for(i=0; i < NUM_ITERS; i++ ) {
      tmp = array[i]-average; 
      std += tmp * tmp;
   }
   std = sqrt(std/(NUM_ITERS-1));

   if (!TESTS_QUIET) {
      printf("\n");

      printf("\nCost for PAPI_start/stop(2 counters) over %d iterations\n",NUM_ITERS);
      printf("min cyc/pair  max cycle/pair  average cycle/pair  std \n ");
      printf("%lld          %lld            %lf         %lf\n", min, max, average, std);

      /* Start the read eval */
      printf("\n\nPerforming read test...\n");
   }
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < NUM_ITERS; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_read(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i]=totcyc;
   }

   min = max = array[0]; 
   printf("array[0] = %lld \n", array[0]);
   average = 0;
   for(i=0; i < NUM_ITERS; i++ ) {
      average += array[i]; 
      if (min > array[i]) min = array[i];
      if (max < array[i]) max = array[i];
   }
   average = (long) (average/NUM_ITERS); 
   std=0;
   for(i=0; i < NUM_ITERS; i++ ) {
      tmp = array[i]-average; 
      std += tmp * tmp;
   }
   std = sqrt(std/(NUM_ITERS-1));

   if (!TESTS_QUIET) {
      printf("\nTotal cost for PAPI_read(2 counters) over %d iterations\n",NUM_ITERS);
      printf("min cyc/pair  max cycle/pair  average cycle/pair  std \n ");
      printf("%lld          %lld            %lf         %lf\n", min, max, average, std);
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
