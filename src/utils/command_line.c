/* 
 * This simply tries to add the events listed on the command line one at a time
 * then starts and stops the counters and prints the results
*/

#include "papi_test.h"

int main(int argc, char **argv)
{
   int retval;
   int num_events;
   long_long *values;
   char *success;
   int EventSet = PAPI_NULL, ExEventSet=PAPI_NULL;
/*   int i, j, event; */
   int i, event;
   char errstr[PAPI_HUGE_STR_LEN];
   char * evt[] = {"PAPI_FP_INS", "ACPI_TEMP"};

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */


   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((retval = PAPI_create_sbstr_eventset(&ExEventSet,1)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_sbstr_eventset", retval);

   if ( TESTS_QUIET ) 
     i = 2;
   else
     i = 1;

   num_events = 2;

   /* Automatically pass if no events, for run_tests.sh */
   if ( num_events == 0 ) 
     test_pass(__FILE__, NULL, 0);

   values = (long_long *) malloc(sizeof(long_long)*num_events);
   success = (char *) malloc(argc);

   if ( success == NULL || values == NULL ) 
      test_fail(__FILE__, __LINE__, "malloc", PAPI_ESYS);

   if ( (retval = PAPI_event_name_to_code(evt[0], &event)) != PAPI_OK )
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

   if ( (retval = PAPI_add_event(EventSet, event)) != PAPI_OK ) {
       PAPI_perror(retval, errstr, 1024 );
       printf("Failed adding: %s\nbecause: %s\n", evt[0], errstr );
     }
   else {
      printf("Successfully added: %s\n", evt[0]);
   }
   if ( (retval = PAPI_event_name_to_code(evt[1], &event)) != PAPI_OK )
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

   if ( (retval = PAPI_add_event(ExEventSet, event)) != PAPI_OK ) {
       PAPI_perror(retval, errstr, 1024 );
       printf("Failed adding: %s\nbecause: %s\n", evt[1], errstr );
     }
   else {
      printf("Successfully added: %s\n", evt[1]);
   }
   printf("\n");

   do_flops(1);
   do_flush();

   if ((retval = PAPI_start(ExEventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);
   do_misses(1,L1_MISS_BUFFER_SIZE_INTS);

   if ((retval = PAPI_stop(ExEventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if ((retval = PAPI_stop(EventSet, &values[1])) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

    printf("Fpins: %lld\tTemp: %lld\n", values[1], values[0]);

   printf("\n----------------------------------\n");
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
