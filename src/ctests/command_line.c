/* 
 * This simply tries to add the events listed on the command line one at a time
 * then starts and stops the counters and prints the results
*/

#include "papi_test.h"

extern int TESTS_QUIET;         /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int retval;
   int num_events;

   #define NUM_EVENTS 4
   long_long *values;
   unsigned int *Events;
   int EventSet = PAPI_NULL;
   char errstr[PAPI_MAX_STR_LEN];
   int i,preset;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */


   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ( TESTS_QUIET ) 
     i = 2;
   else
     i = 1;

   num_events = argc-i;

   /* Automatically pass if no events, for run_tests.sh */
   if ( num_events == 0 ) 
     test_pass(__FILE__, NULL, 0);

   Events = (unsigned int *) malloc(sizeof(unsigned int)*num_events);
   values = (long_long *) malloc(sizeof(long_long)*num_events);

   if ( Events == NULL || values == NULL ) 
      test_fail(__FILE__, __LINE__, "malloc", PAPI_ESYS);

   for ( ;i<argc; i++ ){
     if ( (retval = PAPI_event_name_to_code(argv[i], &preset)) != PAPI_OK )
       test_fail(__FILE__, __LINE__, "PAPI_event_name_to_code", retval);

     if ( (retval = PAPI_add_event(EventSet, preset)) != PAPI_OK ) {
         if ( !TESTS_QUIET ) {
            printf("Failed adding: %s\n", argv[i] );
         }
         if ( !TESTS_QUIET ) {
            PAPI_perror(retval, errstr, 1024 );
            printf("Failed because: %s\n", errstr);
         }
     }
     else {
         if ( !TESTS_QUIET ) 
           printf("Sucessfully added: %s\n", argv[i]);
     }
   }
   if ( !TESTS_QUIET )
     printf("\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if (!TESTS_QUIET){
      for(i=1; i< argc; i++ ){
        printf("%s : \t%lld\n", argv[i], values[i-1]);
      }
   }

   if (!TESTS_QUIET) {
      printf("\n----------------------------------\n");
      printf("Verification: There is no verification, this is a utility that allows the ability\n");
      printf("to add events in different orders by the command line interface to determine if.\n");
      printf("there are problems adding events in different orders.\n");
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
