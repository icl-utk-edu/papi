/* This file tries to add,start,stop all events in a substrate it
 * is meant not to test the accuracy of the mapping but to make sure
 * that all events in the substrate will at least start (Helps to
 * catch typos.
 *
 * Author: Kevin London
 *         london@cs.utk.edu
 */
#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int retval, i;
   int EventSet, count = 0;
   long_long values;
   PAPI_event_info_t info;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      if (PAPI_get_event_info(PRESET_MASK|i, &info) != PAPI_OK)
         continue;
      if (!(info.count))
         continue;
      retval = PAPI_add_event(EventSet, info.event_code);       /* JT */
      if (retval != PAPI_OK) {
         if (!TESTS_QUIET)
            printf("Error adding %s\n", info.symbol);
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
      } else { 
         if (!TESTS_QUIET) 
            printf("Added %s successful\n", info.symbol);
         count++;
      }
      retval = PAPI_start(EventSet);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_start", retval);
      retval = PAPI_stop(EventSet, &values);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

      retval = PAPI_remove_event(EventSet, info.event_code);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);

   }
   if (!TESTS_QUIET)
      printf("Successfully added,started and stopped %d events.\n", count);
   if ( count > 0 )
      test_pass(__FILE__, NULL, 0);
   else
      test_fail(__FILE__, __LINE__, "No events added", 1);
   exit(1);
}
