/* 
* File:    multiplex.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file tests the multiplex functionality, originally developed by 
   John May of LLNL. */

#include "papi_test.h"

void init_papi(void)
{
   int retval;

   /* Initialize the library */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   /* Turn on automatic error reporting */

   retval = PAPI_set_debug(PAPI_VERB_ECONT);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
}

/* Tests that we can really multiplex a lot. */

int case1(void)
{
   int retval, i, EventSet = PAPI_NULL, j = 0, allvalid = 1;
   long long *values;
   PAPI_event_info_t pset;
   
   init_papi();

   retval = PAPI_multiplex_init();
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   retval = PAPI_set_multiplex(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      if ( (i|PAPI_PRESET_MASK) == PAPI_L1_ICM ) continue;
      if ( (i|PAPI_PRESET_MASK) == PAPI_L2_ICM ) continue;
      if ( (i|PAPI_PRESET_MASK) == PAPI_TLB_IM ) continue;

      retval = PAPI_get_event_info(i | PAPI_PRESET_MASK, &pset);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_get_event_info", retval);

      if (pset.count) {
            printf("Adding %s\n", pset.symbol);

         retval = PAPI_add_event(EventSet, pset.event_code);
         if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

    if (retval == PAPI_OK) {
               printf("Added %s\n", pset.symbol);
	   } else {
               printf("Could not add %s\n", pset.symbol);
	   }
 
         if (retval == PAPI_OK) {
            if (++j >= MAX_TO_ADD)
               break;
         }
      }
   }

   values = (long long *) malloc(MAX_TO_ADD * sizeof(long long));
   if (values == NULL)
      test_fail(__FILE__, __LINE__, "malloc", 0);

   if (PAPI_start(EventSet) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(NUM_ITERS);
   do_misses(10, 1024*1024*4);

   retval = PAPI_stop(EventSet, values);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   test_print_event_header("multiplex2:\n", EventSet);
   for (i = 0; i < MAX_TO_ADD; i++) {
     printf(ONENUM, values[i]);
     if (values[i] == 0)
       allvalid = 0;
   }
   printf("\n");
   if (!allvalid){
      if (!TESTS_QUIET ){
        printf("Warning:  one or more counter[s] registered no counts\n");
      /*test_fail(__FILE__, __LINE__, "one or more counter registered no counts",1);*/
      }
   }

   retval = PAPI_cleanup_eventset(EventSet);    /* JT */
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);

   retval = PAPI_destroy_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);

   return (SUCCESS);
}

int main(int argc, char **argv)
{

   if (argc > 1) {
      if (!strcmp(argv[1], "TESTS_QUIET"))
         TESTS_QUIET = 1;
   }

   if (!TESTS_QUIET) {
      printf("%s: Using %d iterations\n\n", argv[0], NUM_ITERS);

      printf("case1: Does PAPI_multiplex_init() handle lots of events?\n");
   }
   case1();
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
