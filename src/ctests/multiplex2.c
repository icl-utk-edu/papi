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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "papi_test.h"

#define SUCCESS 1

extern void do_flops(int);
extern void do_reads(int);

extern int TESTS_QUIET;

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
   int retval, i, EventSet = PAPI_NULL, max_to_add = 6, j = 0;
   long long *values;
   PAPI_event_info_t pset;

   init_papi();

   retval = PAPI_multiplex_init();
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   retval = PAPI_set_multiplex(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      retval = PAPI_get_event_info(i | PRESET_MASK, &pset);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_get_event_info", retval);

      if ((pset.count) && (pset.event_code != PAPI_TOT_CYC)) {
         if (!TESTS_QUIET)
            printf("Adding %s\n", pset.symbol);

         retval = PAPI_add_event(EventSet, pset.event_code);
         if ((retval != PAPI_OK) && (retval != PAPI_ECNFLCT))
            test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

         if (!TESTS_QUIET) {
            if (retval == PAPI_OK)
               printf("Added %s\n", pset.symbol);
            else
               printf("Could not add %s\n", pset.symbol);
         }

         if (retval == PAPI_OK) {
            if (++j >= max_to_add)
               break;
         }
      }
   }

   values = (long long *) malloc(max_to_add * sizeof(long long));
   if (values == NULL)
      test_fail(__FILE__, __LINE__, "malloc", 0);

   if (PAPI_start(EventSet) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_both(NUM_ITERS);

   retval = PAPI_stop(EventSet, values);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

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
