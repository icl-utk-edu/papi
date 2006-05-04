/*
 * File:    all_native_events.c
 * Author:  Haihang You
 *	        you@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

/* This file performs the following test: hardware info and which native events are available */

#include "papi_test.h"
extern int TESTS_QUIET;         /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int i, j, k, loop, EventSet=PAPI_NULL, count=0;
   long_long values;
   int retval;
   PAPI_event_info_t info;
   const PAPI_hw_info_t *hwinfo = NULL;
#ifdef PENTIUM4
   int l;
#endif

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   /*for(i=0;i<argc;i++) */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if (!TESTS_QUIET) {
      printf
          ("Test case NATIVE_AVAIL: Available native events and hardware information.\n");
      printf
          ("-------------------------------------------------------------------------\n");
      printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
             hwinfo->vendor);
      printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
      printf("CPU Revision             : %f\n", hwinfo->revision);
      printf("CPU Megahertz            : %f\n", hwinfo->mhz);
      printf("CPU's in this Node       : %d\n", hwinfo->ncpu);
      printf("Nodes in this System     : %d\n", hwinfo->nnodes);
      printf("Total CPU's              : %d\n", hwinfo->totalcpus);
      printf("Number Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
      printf("Max Multiplex Counters   : %d\n", PAPI_MPX_DEF_DEG);
      printf
          ("-------------------------------------------------------------------------\n");

      printf("The following correspond to fields in the PAPI_event_info_t structure.\n");
      
      printf("Symbol\tEvent Code\tCount\n |Short Description|\n |Long Description|\n |Derived|\n |PostFix|\n");
      printf("The count field indicates whether it is a) available (count >= 1) and b) derived (count > 1)\n");

   }
   i = 0 | PAPI_NATIVE_MASK;
   j = 0;
#ifdef __crayx1
   PAPI_enum_event(&i, 0);
#endif
   do {
         j++;
         retval = PAPI_get_event_info(i, &info);

	 printf("%s\t0x%x  %d\n",
		info.symbol,
		info.event_code,
		info.count);

#ifdef PENTIUM4
      k = i;
      if (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK) {
         l = strlen(info.long_descr);
         do {
            j++;
            retval = PAPI_get_event_info(k, &info);
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
			/*
            retval = PAPI_start(EventSet);
            if (retval != PAPI_OK)
               test_fail(__FILE__, __LINE__, "PAPI_start", retval);
			   
			for(loop=0;loop<10;loop++);
			   
            retval = PAPI_stop(EventSet, &values);
            if (retval != PAPI_OK)
               test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
            */
            retval = PAPI_remove_event(EventSet, info.event_code);
            if (retval != PAPI_OK)
               test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);
         } while (PAPI_enum_event(&k, PAPI_PENT4_ENUM_BITS) == PAPI_OK);
	  }
      if (!TESTS_QUIET && retval == PAPI_OK)
         printf("\n");
   } while (PAPI_enum_event(&i, PAPI_PENT4_ENUM_GROUPS) == PAPI_OK);
#else
      retval = PAPI_add_event(EventSet, info.event_code);       /* JT */
      if (retval != PAPI_OK) {
         if (!TESTS_QUIET)
            printf("Error adding %s\n", info.symbol);
         /*test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);*/ continue;
      } else { 
         if (!TESTS_QUIET) 
            printf("Added %s successful\n", info.symbol);
         count++;
      }
	  /*
      retval = PAPI_start(EventSet);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_start", retval);
      retval = PAPI_stop(EventSet, &values);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
      */
      retval = PAPI_remove_event(EventSet, info.event_code);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);
#if defined(_POWER4)
/* this function would return the next native event code.
    modifer = PAPI_ENUM_ALL
		 it simply returns next native event code
    modifer = PAPI_PWR4_ENUM_GROUPS
		 it would return information of groups this native event lives
                 0x400000ed is the native code of PM_FXLS_FULL_CYC,
		 before it returns 0x400000ee which is the next native event's
		 code, it would return *EventCode=0x400400ed, the digits 16-23
		 indicate group number
   function return value:
     PAPI_OK successful, next event is valid
     PAPI_ENOEVNT  fail, next event is invalid
*/
   } while (PAPI_enum_event(&i, PAPI_PWR4_ENUM_GROUPS) == PAPI_OK);
#else
   } while (PAPI_enum_event(&i, PAPI_ENUM_ALL) == PAPI_OK);
#endif
#endif

   if (!TESTS_QUIET)
      printf("Successfully added,started and stopped %d events.\n", count);
   if ( count > 0 )
      test_pass(__FILE__, NULL, 0);
   else
      test_fail(__FILE__, __LINE__, "No events added", 1);
   exit(1);
}
